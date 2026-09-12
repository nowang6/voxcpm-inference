/**
 * @file locdit.h
 * @brief VoxCPM Local Diffusion Transformer built on top of MiniCPM
 */

#ifndef VOXCPM_LOCDIT_H
#define VOXCPM_LOCDIT_H

#include "voxcpm/context.h"
#include "voxcpm/minicpm.h"

#include <memory>
#include <string>

namespace voxcpm {

class VoxCPMBackend;
class UnifiedCFM;
class VoxCPMWeightStore;

struct LocDiTWeights {
    ggml_tensor* in_proj_weight = nullptr;
    ggml_tensor* in_proj_bias = nullptr;
    ggml_tensor* cond_proj_weight = nullptr;
    ggml_tensor* cond_proj_bias = nullptr;
    ggml_tensor* out_proj_weight = nullptr;
    ggml_tensor* out_proj_bias = nullptr;

    ggml_tensor* time_mlp_linear1_weight = nullptr;
    ggml_tensor* time_mlp_linear1_bias = nullptr;
    ggml_tensor* time_mlp_linear2_weight = nullptr;
    ggml_tensor* time_mlp_linear2_bias = nullptr;

    ggml_tensor* delta_time_mlp_linear1_weight = nullptr;
    ggml_tensor* delta_time_mlp_linear1_bias = nullptr;
    ggml_tensor* delta_time_mlp_linear2_weight = nullptr;
    ggml_tensor* delta_time_mlp_linear2_bias = nullptr;
};

class LocDiTModel {
public:
    LocDiTModel() = default;
    ~LocDiTModel();

    LocDiTModel(const LocDiTModel&) = delete;
    LocDiTModel& operator=(const LocDiTModel&) = delete;

    bool load_from_gguf(const std::string& gguf_path,
                        VoxCPMContext& weight_ctx,
                        VoxCPMContext& graph_ctx,
                        VoxCPMBackend& backend);
    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store,
                         VoxCPMBackend& backend);

    // x / cond / output: [feat_dim, seq_len, batch]
    // mu: [hidden_size * n_mu_tokens, batch]
    // t / dt: [batch]
    ggml_tensor* forward(VoxCPMContext& ctx,
                         ggml_tensor* x,
                         ggml_tensor* mu,
                         ggml_tensor* t,
                         ggml_tensor* cond,
                         ggml_tensor* dt);

    void forward_cfg_pair(VoxCPMContext& ctx,
                          ggml_tensor* x,
                          ggml_tensor* mu,
                          ggml_tensor* t_scalar,
                          ggml_tensor* cond,
                          ggml_tensor* dt_scalar,
                          ggml_tensor** conditioned,
                          ggml_tensor** unconditioned);

    // Precompute the zero-delta CFG time table used by UnifiedCFM cached
    // graphs. The result is laid out as [hidden_size, t_values.size()].
    std::vector<float> precompute_cfg_time_table(const std::vector<float>& t_values) const;

    const MiniCPMConfig& config() const { return decoder_.config(); }
    const LocDiTWeights& weights() const { return weights_; }
    int feat_dim() const { return feat_dim_; }
    const MiniCPMModel& decoder_model() const { return decoder_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    friend class UnifiedCFM;

    ggml_tensor* sinusoidal_embedding(VoxCPMContext& ctx, ggml_tensor* scalar, int dim, float scale) const;
    ggml_tensor* timestep_mlp(VoxCPMContext& ctx,
                              ggml_tensor* input,
                              ggml_tensor* linear1_w,
                              ggml_tensor* linear1_b,
                              ggml_tensor* linear2_w,
                              ggml_tensor* linear2_b) const;

    ggml_tensor* compute_time_embedding(VoxCPMContext& ctx, ggml_tensor* t_scalar) const;
    ggml_tensor* compute_delta_time_embedding(VoxCPMContext& ctx, ggml_tensor* dt_scalar) const;
    ggml_tensor* project_input(VoxCPMContext& ctx, ggml_tensor* x) const;
    ggml_tensor* project_condition(VoxCPMContext& ctx, ggml_tensor* cond) const;
    ggml_tensor* build_time_token(VoxCPMContext& ctx,
                                  ggml_tensor* t_scalar,
                                  ggml_tensor* dt_scalar) const;
    ggml_tensor* reshape_mu_tokens(VoxCPMContext& ctx, ggml_tensor* mu) const;
    ggml_tensor* build_cfg_pair_positions(VoxCPMContext& ctx, int branch_len) const;
    ggml_tensor* build_cfg_pair_attention_mask(VoxCPMContext& ctx, int branch_len) const;
    bool ensure_cfg_pair_constants(int branch_len);
    ggml_tensor* forward_projected(VoxCPMContext& ctx,
                                   ggml_tensor* x_proj,
                                   ggml_tensor* prefix_tokens,
                                   ggml_tensor* cond_proj,
                                   int generated_prefix_tokens,
                                   int prefix_len,
                                   int seq_len);
    void forward_cfg_pair_projected(VoxCPMContext& ctx,
                                    ggml_tensor* x_proj,
                                    ggml_tensor* mu,
                                    ggml_tensor* time_token,
                                    ggml_tensor* cond_proj,
                                    int prefix_len,
                                    ggml_tensor** conditioned,
                                    ggml_tensor** unconditioned);

    ggml_tensor* forward_single(VoxCPMContext& ctx,
                                ggml_tensor* x,
                                ggml_tensor* mu,
                                ggml_tensor* t_scalar,
                                ggml_tensor* cond,
                                ggml_tensor* dt_scalar);

    bool init_scratch_cache(VoxCPMBackend& backend);

    LocDiTWeights weights_;
    MiniCPMModel decoder_;

    int feat_dim_ = 0;

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    ggml_context* cfg_pair_ctx_ = nullptr;
    ggml_backend_buffer_t cfg_pair_buffer_ = nullptr;
    ggml_tensor* cfg_pair_positions_ = nullptr;
    ggml_tensor* cfg_pair_attention_mask_ = nullptr;
    int cfg_pair_branch_len_ = 0;
    VoxCPMBackend* backend_ = nullptr;
    std::unique_ptr<MiniCPMKVCache> scratch_kv_cache_;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

}  // namespace voxcpm

#endif  // VOXCPM_LOCDIT_H
