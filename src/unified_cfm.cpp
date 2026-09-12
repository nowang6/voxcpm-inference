/**
 * @file unified_cfm.cpp
 * @brief VoxCPM Unified Conditional Flow Matching solver implementation
 */

#include "voxcpm/unified_cfm.h"

#include <algorithm>
#include <cmath>

namespace voxcpm {

UnifiedCFM::UnifiedCFM(LocDiTModel& estimator, const CFMConfig& config)
    : estimator_(estimator),
      config_(config) {
}

std::vector<float> UnifiedCFM::compute_t_span(int n_timesteps, float sway_sampling_coef) {
    VOXCPM_ASSERT(n_timesteps > 0);

    std::vector<float> t_span(static_cast<size_t>(n_timesteps) + 1);
    for (int i = 0; i <= n_timesteps; ++i) {
        const float base = 1.0f - static_cast<float>(i) / static_cast<float>(n_timesteps);
        t_span[static_cast<size_t>(i)] =
            base + sway_sampling_coef * (std::cos(static_cast<float>(M_PI_2) * base) - 1.0f + base);
    }
    return t_span;
}

ggml_tensor* UnifiedCFM::optimized_scale(VoxCPMContext& ctx,
                                         ggml_tensor* positive,
                                         ggml_tensor* negative,
                                         float eps) const {
    ggml_context* raw = ctx.raw_context();

    ggml_tensor* dot_product = ggml_sum(raw, ggml_mul(raw, positive, negative));
    ggml_tensor* squared_norm = ggml_sum(raw, ggml_mul(raw, negative, negative));
    ggml_tensor* eps_tensor = ggml_arange(raw, eps, eps + 1.0f, 1.0f);
    ggml_tensor* denom = ggml_add(raw, squared_norm, eps_tensor);
    return ggml_div(raw, dot_product, denom);
}

ggml_tensor* UnifiedCFM::compute_velocity_with_cfg(VoxCPMContext& ctx,
                                                   ggml_tensor* x,
                                                   ggml_tensor* mu,
                                                   ggml_tensor* cond_proj,
                                                   int prefix_len,
                                                   ggml_tensor* delta_time_zero,
                                                   ggml_tensor* combined_time_embedding,
                                                   float t,
                                                   float cfg_value,
                                                   bool use_cfg_zero_star) {
    ggml_context* raw = ctx.raw_context();
    ggml_tensor* x_proj = estimator_.project_input(ctx, x);
    if (combined_time_embedding == nullptr) {
        ggml_tensor* t_scalar = ggml_arange(raw, t, t + 1.0f, 1.0f);
        combined_time_embedding = estimator_.compute_time_embedding(ctx, t_scalar);
        if (delta_time_zero != nullptr) {
            combined_time_embedding = ggml_add(raw, combined_time_embedding, delta_time_zero);
        }
    }

    ggml_tensor* dphi_dt_cond = nullptr;
    ggml_tensor* dphi_dt_uncond = nullptr;
    estimator_.forward_cfg_pair_projected(
        ctx,
        x_proj,
        mu,
        combined_time_embedding,
        cond_proj,
        prefix_len,
        &dphi_dt_cond,
        &dphi_dt_uncond);

    if (use_cfg_zero_star) {
        ggml_tensor* st_star = optimized_scale(ctx, dphi_dt_cond, dphi_dt_uncond);
        ggml_tensor* uncond_scaled = ggml_mul(raw, dphi_dt_uncond, ggml_repeat(raw, st_star, dphi_dt_uncond));
        ggml_tensor* diff = ggml_sub(raw, dphi_dt_cond, uncond_scaled);
        ggml_tensor* cfg_scaled = ggml_scale(raw, diff, cfg_value);
        return ggml_add(raw, uncond_scaled, cfg_scaled);
    }

    ggml_tensor* diff = ggml_sub(raw, dphi_dt_cond, dphi_dt_uncond);
    ggml_tensor* cfg_scaled = ggml_scale(raw, diff, cfg_value);
    return ggml_add(raw, dphi_dt_uncond, cfg_scaled);
}

ggml_tensor* UnifiedCFM::solve_euler(VoxCPMContext& ctx,
                                     ggml_tensor* x,
                                     const std::vector<float>& t_span,
                                     ggml_tensor* mu,
                                     ggml_tensor* cond_proj,
                                     int prefix_len,
                                     ggml_tensor* delta_time_zero,
                                     ggml_tensor* precomputed_time_table,
                                     float cfg_value,
                                     bool use_cfg_zero_star) {
    VOXCPM_ASSERT(x != nullptr);
    VOXCPM_ASSERT(mu != nullptr);
    VOXCPM_ASSERT(cond_proj != nullptr || prefix_len == 0);
    VOXCPM_ASSERT(t_span.size() >= 2);

    ggml_context* raw = ctx.raw_context();
    float t = t_span[0];
    float dt = t_span[0] - t_span[1];

    const int n_steps = static_cast<int>(t_span.size()) - 1;
    const int zero_init_steps = n_steps > 1
        ? std::max(1, static_cast<int>(t_span.size() * 0.04f))
        : 0;

    for (int step = 1; step <= n_steps; ++step) {
        ggml_tensor* dphi_dt = nullptr;

        if (use_cfg_zero_star && step <= zero_init_steps) {
            dphi_dt = ggml_scale(raw, x, 0.0f);
        } else {
            ggml_tensor* combined_time_embedding = nullptr;
            if (precomputed_time_table != nullptr) {
                const size_t offset = static_cast<size_t>(step - 1) * static_cast<size_t>(estimator_.config().hidden_size) *
                    sizeof(float);
                combined_time_embedding = ggml_view_1d(raw,
                                                       precomputed_time_table,
                                                       estimator_.config().hidden_size,
                                                       offset);
            }
            dphi_dt = compute_velocity_with_cfg(
                ctx,
                x,
                mu,
                cond_proj,
                prefix_len,
                delta_time_zero,
                combined_time_embedding,
                t,
                cfg_value,
                use_cfg_zero_star);
        }

        x = ggml_sub(raw, x, ggml_scale(raw, dphi_dt, dt));
        t -= dt;

        if (step < n_steps) {
            dt = t - t_span[static_cast<size_t>(step + 1)];
        }
    }

    return x;
}

ggml_tensor* UnifiedCFM::forward(VoxCPMContext& ctx,
                                 ggml_tensor* z,
                                 ggml_tensor* mu,
                                 int patch_size,
                                 ggml_tensor* cond,
                                 int n_timesteps,
                                 float cfg_value,
                                 float temperature,
                                 float sway_sampling_coef,
                                 bool use_cfg_zero_star,
                                 ggml_tensor* precomputed_time_table) {
    VOXCPM_UNUSED(patch_size);

    VOXCPM_ASSERT(z != nullptr);
    VOXCPM_ASSERT(mu != nullptr);
    VOXCPM_ASSERT(cond != nullptr);
    VOXCPM_ASSERT(ggml_n_dims(z) == 2);
    VOXCPM_ASSERT(ggml_n_dims(mu) == 1);
    VOXCPM_ASSERT(ggml_n_dims(cond) == 2);

    ggml_tensor* x = (temperature == 1.0f) ? z : ggml_scale(ctx.raw_context(), z, temperature);
    const std::vector<float> t_span = compute_t_span(n_timesteps, sway_sampling_coef);
    const int prefix_len = static_cast<int>(cond->ne[1]);

    ggml_tensor* cond_proj = prefix_len > 0 ? estimator_.project_condition(ctx, cond) : nullptr;
    ggml_tensor* delta_time_zero = nullptr;
    if (precomputed_time_table == nullptr) {
        ggml_tensor* zero_scalar = ggml_arange(ctx.raw_context(), 0.0f, 1.0f, 1.0f);
        delta_time_zero = estimator_.compute_delta_time_embedding(ctx, zero_scalar);
    }

    return solve_euler(
        ctx, x, t_span, mu, cond_proj, prefix_len, delta_time_zero, precomputed_time_table, cfg_value, use_cfg_zero_star);
}

}  // namespace voxcpm
