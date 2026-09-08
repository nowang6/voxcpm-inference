/**
 * @file audio-vae.cpp
 * @brief VoxCPM AudioVAE implementation
 */

#include "voxcpm/audio-vae.h"

#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace voxcpm {

struct AudioVAEDepthwiseConvOpData {
    int stride = 1;
    int dilation = 1;
    int padding = 0;
};

struct AudioVAEConv1DSpec {
    int64_t kernel = 0;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
};

// ggml_conv_transpose_1d 权重约定为 [K, Cout, Cin](ne[2] == 输入通道)。
struct AudioVAETransposeConv1DSpec {
    int64_t kernel = 0;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
};

namespace {

static float load_f32_scalar(const uint8_t* ptr, ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float*>(ptr);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t*>(ptr));
        default:
            VOXCPM_ASSERT(false && "unsupported tensor type");
            return 0.0f;
    }
}

static AudioVAEConv1DSpec resolve_conv1d_spec(const ggml_tensor* weight, int expected_kernel) {
    VOXCPM_ASSERT(weight != nullptr);

    AudioVAEConv1DSpec spec;
    if (ggml_n_dims(weight) == 3) {
        spec.kernel = weight->ne[0];
        spec.in_channels = weight->ne[1];
        spec.out_channels = weight->ne[2];
        return spec;
    }

    VOXCPM_ASSERT(ggml_n_dims(weight) == 2);
    VOXCPM_ASSERT(expected_kernel > 0);

    // 量化产物(q8_0 / q4_0)的普通卷积核由量化端折叠为 2 维 [K*Cin, Cout]
    // (纯 reshape:行内 j = cin*K + k,k 最快),
    // 与 im2col 激活布局天然对齐;Q4_0 与 Q8_0 分块布局同为 32 元素/块。
    if (weight->type == GGML_TYPE_Q8_0 || weight->type == GGML_TYPE_Q4_0) {
        VOXCPM_ASSERT(weight->ne[0] % 32 == 0);
        VOXCPM_ASSERT(weight->ne[0] % expected_kernel == 0);
        spec.kernel = expected_kernel;
        spec.in_channels = weight->ne[0] / expected_kernel;
        spec.out_channels = weight->ne[1];
        return spec;
    }

    if (weight->ne[0] == expected_kernel) {
        spec.kernel = weight->ne[0];
        spec.in_channels = weight->ne[1];
        spec.out_channels = 1;
        return spec;
    }

    VOXCPM_ASSERT(weight->ne[0] % expected_kernel == 0);

    spec.kernel = expected_kernel;
    spec.in_channels = weight->ne[0] / expected_kernel;
    spec.out_channels = weight->ne[1];
    return spec;
}

static ggml_tensor* reshape_conv1d_weight_2d(ggml_context* ctx,
                                             ggml_tensor* weight,
                                             const AudioVAEConv1DSpec& spec) {
    if (ggml_n_dims(weight) == 2 && !(weight->ne[0] == spec.kernel && spec.out_channels == 1)) {
        return weight;
    }
    ggml_tensor* reshaped = ggml_reshape_2d(ctx, weight, spec.kernel * spec.in_channels, spec.out_channels);
    ggml_set_name(reshaped, weight->name);
    return reshaped;
}

static AudioVAETransposeConv1DSpec resolve_transpose_conv1d_spec(const ggml_tensor* weight, int expected_kernel) {
    VOXCPM_ASSERT(weight != nullptr);

    AudioVAETransposeConv1DSpec spec;
    if (ggml_n_dims(weight) == 3) {
        spec.kernel = weight->ne[0];
        spec.out_channels = weight->ne[1];
        spec.in_channels = weight->ne[2];
        return spec;
    }

    // 折叠形式 [K*Cout, Cin]:量化产物中转置卷积跟随基础类型(Q8_0 或 Q4_0,
    // 两者块大小均为 32)。行内 j = cout*K + k,k 最快。
    VOXCPM_ASSERT(ggml_n_dims(weight) == 2);
    VOXCPM_ASSERT(expected_kernel > 0);
    VOXCPM_ASSERT(weight->ne[0] % expected_kernel == 0);

    spec.kernel = expected_kernel;
    spec.out_channels = weight->ne[0] / expected_kernel;
    spec.in_channels = weight->ne[1];
    return spec;
}

// 把折叠 2 维转置卷积核还原为 ggml_conv_transpose_1d 所需的 3 维 [K, Cout, Cin]。
// 折叠是纯 reshape,展开无需任何置换;量化权重
// (Q8_0/Q4_0)先上转 F32(vendored ggml 的 cast 仅支持量化类型 → F32),
// 3 维 F16/F32 权重原样返回。
static ggml_tensor* unfold_transpose_conv1d_weight(ggml_context* ctx,
                                                   ggml_tensor* weight,
                                                   int expected_kernel) {
    if (ggml_n_dims(weight) == 3) {
        return weight;
    }
    const AudioVAETransposeConv1DSpec spec = resolve_transpose_conv1d_spec(weight, expected_kernel);
    ggml_tensor* weight_f32 = to_f32(ctx, weight);
    ggml_tensor* unfolded =
        ggml_reshape_3d(ctx, weight_f32, spec.kernel, spec.out_channels, spec.in_channels);
    ggml_set_name(unfolded, weight->name);
    return unfolded;
}

static void depthwise_conv_custom(ggml_tensor* dst,
                                  const ggml_tensor* x,
                                  const ggml_tensor* weight,
                                  const ggml_tensor* bias,
                                  int ith,
                                  int nth,
                                  void* userdata) {
    const auto* op = static_cast<const AudioVAEDepthwiseConvOpData*>(userdata);
    const int64_t t_len = x->ne[0];
    const int64_t channels = x->ne[1];
    const int64_t batch = x->ne[2];
    const int64_t kernel = weight->ne[0];

    VOXCPM_ASSERT(batch >= 1);
    VOXCPM_ASSERT(weight->ne[1] == 1);
    VOXCPM_ASSERT(weight->ne[2] == channels);
    VOXCPM_ASSERT(dst->ne[0] == x->ne[0]);
    VOXCPM_ASSERT(dst->ne[1] == x->ne[1]);
    VOXCPM_ASSERT(dst->ne[2] == x->ne[2]);
    VOXCPM_ASSERT(x->type == GGML_TYPE_F32);
    VOXCPM_ASSERT(dst->type == GGML_TYPE_F32);
    VOXCPM_ASSERT(weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_F16);
    VOXCPM_ASSERT(bias == nullptr || bias->type == GGML_TYPE_F32 || bias->type == GGML_TYPE_F16);

    const uint8_t* x_data = static_cast<const uint8_t*>(x->data);
    const uint8_t* w_data = static_cast<const uint8_t*>(weight->data);
    const uint8_t* b_data = bias ? static_cast<const uint8_t*>(bias->data) : nullptr;
    uint8_t* dst_data = static_cast<uint8_t*>(dst->data);

    const int64_t work_items = channels * batch;
    const int64_t items_per_thread = (work_items + nth - 1) / nth;
    const int64_t item_begin = ith * items_per_thread;
    const int64_t item_end = std::min<int64_t>(item_begin + items_per_thread, work_items);

    for (int64_t item = item_begin; item < item_end; ++item) {
        const int64_t b = item / channels;
        const int64_t c = item % channels;
        const float bias_val = b_data ? load_f32_scalar(b_data + c * bias->nb[0], bias->type) : 0.0f;

        const uint8_t* x_channel = x_data + b * x->nb[2] + c * x->nb[1];
        const uint8_t* w_channel = w_data + c * weight->nb[2];
        uint8_t* dst_channel = dst_data + b * dst->nb[2] + c * dst->nb[1];

        for (int64_t t = 0; t < t_len; ++t) {
            float sum = bias_val;
            for (int64_t k = 0; k < kernel; ++k) {
                const int64_t src_t = t * op->stride + k * op->dilation - op->padding * 2;
                if (src_t < 0 || src_t >= t_len) {
                    continue;
                }
                const float x_val = load_f32_scalar(x_channel + src_t * x->nb[0], x->type);
                const float w_val = load_f32_scalar(w_channel + k * weight->nb[0], weight->type);
                sum += x_val * w_val;
            }
            *reinterpret_cast<float*>(dst_channel + t * dst->nb[0]) = sum;
        }
    }
}


// 原 L1 路径，做 im2col 展开 + permute + cont（纯拷贝开销）
static ggml_tensor* conv1d_mul_mat_impl(ggml_context* ctx,
                                        ggml_tensor* weight,
                                        ggml_tensor* input,
                                        int expected_kernel,
                                        int stride,
                                        int dilation) {
    const AudioVAEConv1DSpec spec = resolve_conv1d_spec(weight, expected_kernel);
    ggml_tensor* weight_2d = reshape_conv1d_weight_2d(ctx, weight, spec);

    ggml_tensor* kernel_shape = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, spec.kernel, spec.in_channels, spec.out_channels);
    ggml_tensor* im2col =
        ggml_im2col(ctx, kernel_shape, input, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor* activations = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1] * im2col->ne[2]);
    ggml_tensor* result = ggml_mul_mat(ctx, weight_2d, activations);
    result = ggml_reshape_3d(ctx, result, spec.out_channels, im2col->ne[1], im2col->ne[2]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 1, 0, 2, 3));
}

static std::string encoder_res_prefix(int block_idx, int res_idx) {
    return "audio_vae.encoder.block." + std::to_string(block_idx) + ".block." + std::to_string(res_idx) + ".block.";
}

static std::string decoder_res_prefix(int model_idx, int res_idx) {
    return "audio_vae.decoder.model." + std::to_string(model_idx) + ".block." + std::to_string(res_idx) + ".block.";
}

static int decoder_final_snake_model_idx(const AudioVAEConfig& config) {
    return config.num_decoder_blocks() + 2;
}

static int decoder_final_conv_model_idx(const AudioVAEConfig& config) {
    return config.num_decoder_blocks() + 3;
}

// 分桶后张量可能位于 CPU 或 HTP 桶，统一经 store 跨桶查询
static bool get_required_tensor(const VoxCPMWeightStore& store, const std::string& name,
                                ggml_tensor** dst) {
    *dst = store.get_tensor(name.c_str());
    return *dst != nullptr;
}

static ggml_tensor* get_optional_tensor(const VoxCPMWeightStore& store, const std::string& name) {
    return store.get_tensor(name.c_str());
}

static ggml_tensor* reshape_bias_3d(ggml_context* ctx, ggml_tensor* bias) {
    const int64_t channels = bias->ne[0];
    return ggml_reshape_3d(ctx, bias, 1, channels, 1);
}

// F16 bias 上转:二元 add 不支持 F32(src0)+F16(src1)。
static ggml_tensor* conv_add_bias(ggml_context* ctx, ggml_tensor* result, ggml_tensor* bias) {
    if (bias == nullptr) {
        return result;
    }
    return ggml_add(ctx, result, reshape_bias_3d(ctx, to_f32(ctx, bias)));
}

}  // namespace

ggml_tensor* snake_activation(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, float eps) {
    // F16 alpha 上转:后续 mul/div/add 都是二元算子,不支持 F32(src0)+F16(src1)。
    alpha = to_f32(ctx, alpha);
    const int64_t channels = alpha->ne[1] > 1 ? alpha->ne[1] : alpha->ne[0];
    ggml_tensor* alpha_view = ggml_reshape_3d(ctx, alpha, 1, channels, 1);
    ggml_tensor* alpha_broadcast = ggml_repeat(ctx, alpha_view, x);
    ggml_tensor* alpha_eps = ggml_add1(ctx, alpha_broadcast, ggml_arange(ctx, eps, eps + 1.0f, 1.0f));
    ggml_tensor* ax = ggml_mul(ctx, x, alpha_broadcast);
    ggml_tensor* sin_sq = ggml_sqr(ctx, ggml_sin(ctx, ax));
    ggml_tensor* one = ggml_arange(ctx, 1.0f, 2.0f, 1.0f);
    return ggml_add(ctx, x, ggml_mul(ctx, sin_sq, ggml_div(ctx, ggml_repeat(ctx, one, alpha_eps), alpha_eps)));
}

AudioVAE::AudioVAE(const AudioVAEConfig& config)
    : config_(config) {
}

AudioVAE::~AudioVAE() {
    if (weight_buffer_) {
        ggml_backend_buffer_free(weight_buffer_);
        weight_buffer_ = nullptr;
    }
    if (weight_ctx_) {
        ggml_free(weight_ctx_);
        weight_ctx_ = nullptr;
    }
    if (derived_.buffer) {
        ggml_backend_buffer_free(derived_.buffer);
        derived_.buffer = nullptr;
    }
    if (derived_.ctx) {
        ggml_free(derived_.ctx);
        derived_.ctx = nullptr;
    }
    if (derived_cpu_.buffer) {
        ggml_backend_buffer_free(derived_cpu_.buffer);
        derived_cpu_.buffer = nullptr;
    }
    if (derived_cpu_.ctx) {
        ggml_free(derived_cpu_.ctx);
        derived_cpu_.ctx = nullptr;
    }
    if (convt_input_buffer_) {
        ggml_backend_buffer_free(convt_input_buffer_);
        convt_input_buffer_ = nullptr;
    }
    if (convt_input_ctx_) {
        ggml_free(convt_input_ctx_);
        convt_input_ctx_ = nullptr;
    }
    convt_inputs_.clear();
}

bool AudioVAE::load_encoder_weights(const VoxCPMWeightStore& store) {
    bool ok = true;
    ok &= get_required_tensor(store, "audio_vae.encoder.block.0.weight", &weights_.encoder_block_0_weight);
    ok &= get_required_tensor(store, "audio_vae.encoder.block.0.bias", &weights_.encoder_block_0_bias);
    ok &= get_required_tensor(store, "audio_vae.encoder.fc_mu.weight", &weights_.encoder_fc_mu_weight);
    ok &= get_required_tensor(store, "audio_vae.encoder.fc_mu.bias", &weights_.encoder_fc_mu_bias);

    weights_.encoder_blocks.resize(static_cast<size_t>(config_.num_encoder_blocks()));
    for (int i = 0; i < config_.num_encoder_blocks(); ++i) {
        EncoderBlockWeights& block = weights_.encoder_blocks[static_cast<size_t>(i)];
        const int block_idx = i + 1;
        const std::string block_prefix = "audio_vae.encoder.block." + std::to_string(block_idx) + ".block.";

        auto load_res = [&](ResidualUnitWeights& res, int res_idx) {
            const std::string prefix = encoder_res_prefix(block_idx, res_idx);
            ok &= get_required_tensor(store, prefix + "0.alpha", &res.snake1_alpha);
            ok &= get_required_tensor(store, prefix + "1.weight", &res.conv1_weight);
            ok &= get_required_tensor(store, prefix + "1.bias", &res.conv1_bias);
            ok &= get_required_tensor(store, prefix + "2.alpha", &res.snake2_alpha);
            ok &= get_required_tensor(store, prefix + "3.weight", &res.conv2_weight);
            ok &= get_required_tensor(store, prefix + "3.bias", &res.conv2_bias);
        };

        load_res(block.res0, 0);
        load_res(block.res1, 1);
        load_res(block.res2, 2);
        ok &= get_required_tensor(store, block_prefix + "3.alpha", &block.snake_alpha);
        ok &= get_required_tensor(store, block_prefix + "4.weight", &block.conv_weight);
        ok &= get_required_tensor(store, block_prefix + "4.bias", &block.conv_bias);
    }

    return ok;
}

bool AudioVAE::load_decoder_weights(const VoxCPMWeightStore& store) {
    bool ok = true;
    ok &= get_required_tensor(store, "audio_vae.decoder.model.0.weight", &weights_.decoder_model_0_weight);
    ok &= get_required_tensor(store, "audio_vae.decoder.model.0.bias", &weights_.decoder_model_0_bias);
    ok &= get_required_tensor(store, "audio_vae.decoder.model.1.weight", &weights_.decoder_model_1_weight);
    ok &= get_required_tensor(store, "audio_vae.decoder.model.1.bias", &weights_.decoder_model_1_bias);

    const int final_snake_idx = decoder_final_snake_model_idx(config_);
    const int final_conv_idx = decoder_final_conv_model_idx(config_);

    ok &= get_required_tensor(store,
                              "audio_vae.decoder.model." + std::to_string(final_snake_idx) + ".alpha",
                              &weights_.decoder_final_snake_alpha);
    ok &= get_required_tensor(store,
                              "audio_vae.decoder.model." + std::to_string(final_conv_idx) + ".weight",
                              &weights_.decoder_final_conv_weight);
    ok &= get_required_tensor(store,
                              "audio_vae.decoder.model." + std::to_string(final_conv_idx) + ".bias",
                              &weights_.decoder_final_conv_bias);

    weights_.decoder_blocks.resize(static_cast<size_t>(config_.num_decoder_blocks()));
    for (int i = 0; i < config_.num_decoder_blocks(); ++i) {
        DecoderBlockWeights& block = weights_.decoder_blocks[static_cast<size_t>(i)];
        const int model_idx = i + 2;
        const std::string block_prefix = "audio_vae.decoder.model." + std::to_string(model_idx) + ".block.";
        const std::string sr_cond_prefix = "audio_vae.decoder.sr_cond_model." + std::to_string(model_idx) + ".";

        block.sr_cond.scale_embed = get_optional_tensor(store, sr_cond_prefix + "scale_embed.weight");
        block.sr_cond.bias_embed = get_optional_tensor(store, sr_cond_prefix + "bias_embed.weight");
        block.sr_cond.cond_embed = get_optional_tensor(store, sr_cond_prefix + "cond_embed.weight");
        block.sr_cond.out_snake_alpha = get_optional_tensor(store, sr_cond_prefix + "out_layer.0.alpha");
        block.sr_cond.out_weight = get_optional_tensor(store, sr_cond_prefix + "out_layer.1.weight");
        block.sr_cond.out_bias = get_optional_tensor(store, sr_cond_prefix + "out_layer.1.bias");

        ok &= get_required_tensor(store, block_prefix + "0.alpha", &block.snake_alpha);
        ok &= get_required_tensor(store, block_prefix + "1.weight", &block.conv_weight);
        ok &= get_required_tensor(store, block_prefix + "1.bias", &block.conv_bias);

        auto load_res = [&](ResidualUnitWeights& res, int res_idx) {
            const std::string prefix = decoder_res_prefix(model_idx, res_idx + 2);
            ok &= get_required_tensor(store, prefix + "0.alpha", &res.snake1_alpha);
            ok &= get_required_tensor(store, prefix + "1.weight", &res.conv1_weight);
            ok &= get_required_tensor(store, prefix + "1.bias", &res.conv1_bias);
            ok &= get_required_tensor(store, prefix + "2.alpha", &res.snake2_alpha);
            ok &= get_required_tensor(store, prefix + "3.weight", &res.conv2_weight);
            ok &= get_required_tensor(store, prefix + "3.bias", &res.conv2_bias);
        };

        load_res(block.res0, 0);
        load_res(block.res1, 1);
        load_res(block.res2, 2);
    }

    return ok;
}

bool AudioVAE::load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store) {
    if (!store || !store->owns_storage()) {
        return false;
    }

    shared_store_ = store;

    uint32_t u32 = 0;
    const bool has_encoder_dim = store->get_u32("voxcpm_audio_vae_config_encoder_dim", u32);
    if (has_encoder_dim) config_.encoder_dim = static_cast<int>(u32);
    const bool has_decoder_dim = store->get_u32("voxcpm_audio_vae_config_decoder_dim", u32);
    if (has_decoder_dim) config_.decoder_dim = static_cast<int>(u32);
    const bool has_latent_dim = store->get_u32("voxcpm_audio_vae_config_latent_dim", u32);
    if (has_latent_dim) config_.latent_dim = static_cast<int>(u32);
    const bool has_sample_rate = store->get_u32("voxcpm_audio_vae_config_sample_rate", u32);
    if (has_sample_rate) config_.sample_rate = static_cast<int>(u32);
    const bool has_out_sample_rate = store->get_u32("voxcpm_audio_vae_config_out_sample_rate", u32);
    if (has_out_sample_rate) config_.out_sample_rate = static_cast<int>(u32);
    const bool has_depthwise = store->get_bool("voxcpm_audio_vae_config_depthwise", config_.depthwise);
    const bool has_use_noise_block = store->get_bool("voxcpm_audio_vae_config_use_noise_block", config_.use_noise_block);
    const bool has_encoder_rates = store->get_i32_array("voxcpm_audio_vae_config_encoder_rates", config_.encoder_rates);
    const bool has_decoder_rates = store->get_i32_array("voxcpm_audio_vae_config_decoder_rates", config_.decoder_rates);
    store->get_i32_array("voxcpm_audio_vae_config_sr_bin_boundaries", config_.sr_bin_boundaries);
    store->get_string("voxcpm_audio_vae_config_cond_type", config_.cond_type);
    if (store->get_u32("voxcpm_audio_vae_config_cond_dim", u32)) config_.cond_dim = static_cast<int>(u32);
    store->get_bool("voxcpm_audio_vae_config_cond_out_layer", config_.cond_out_layer);

    if (!has_encoder_dim || !has_decoder_dim || !has_latent_dim || !has_sample_rate ||
        !has_encoder_rates || !has_decoder_rates) {
        return false;
    }
    if (config_.encoder_rates.empty() || config_.decoder_rates.empty()) {
        return false;
    }
    if (!has_depthwise) {
        config_.depthwise = true;
    }
    if (!has_use_noise_block) {
        config_.use_noise_block = false;
    }

    return load_encoder_weights(*store) && load_decoder_weights(*store);
}

std::vector<float> AudioVAE::preprocess(std::vector<float> audio_data, int sample_rate) const {
    const int actual_sample_rate = sample_rate < 0 ? config_.sample_rate : sample_rate;
    VOXCPM_ASSERT(actual_sample_rate == config_.sample_rate);

    const int hop = config_.hop_length();
    const size_t length = audio_data.size();
    const size_t aligned = ((length + static_cast<size_t>(hop) - 1) / static_cast<size_t>(hop)) * static_cast<size_t>(hop);
    audio_data.resize(aligned, 0.0f);
    return audio_data;
}

ggml_tensor* AudioVAE::causal_conv1d(ggml_context* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* weight,
                                     ggml_tensor* bias,
                                     int kernel_size,
                                     int stride,
                                     int dilation,
                                     int padding) const {
    ggml_tensor* padded = x;
    if (padding > 0) {
        padded = ggml_pad_ext(ctx, x, padding * 2, 0, 0, 0, 0, 0, 0, 0);
    }
    ggml_tensor* result = conv1d_mul_mat_impl(ctx, weight, padded, kernel_size, stride, dilation);
    if (bias) {
        result = conv_add_bias(ctx, result, bias);
    }
    return result;
}

ggml_tensor* AudioVAE::causal_conv1d_dw(ggml_context* ctx,
                                        ggml_tensor* x,
                                        ggml_tensor* weight,
                                        ggml_tensor* bias,
                                        int stride,
                                        int dilation,
                                        int padding) const {
    auto op = std::make_unique<AudioVAEDepthwiseConvOpData>();
    op->stride = stride;
    op->dilation = dilation;
    op->padding = padding;

    AudioVAEDepthwiseConvOpData* op_ptr = op.get();
    depthwise_ops_.push_back(std::move(op));
    return ggml_map_custom3(ctx, x, weight, bias, depthwise_conv_custom, GGML_N_TASKS_MAX, op_ptr);
}

ggml_tensor* AudioVAE::causal_transpose_conv1d(ggml_context* ctx,
                                               ggml_tensor* x,
                                               ggml_tensor* weight,
                                               ggml_tensor* bias,
                                               int expected_kernel,
                                               int stride,
                                               int padding,
                                               int output_padding) const {
    weight = unfold_transpose_conv1d_weight(ctx, weight, expected_kernel);
    ggml_tensor* result = ggml_conv_transpose_1d(ctx, weight, x, stride, 0, 1);
    if (result->ne[3] == 1) {
        result = ggml_reshape_3d(ctx, result, result->ne[0], result->ne[1], result->ne[2]);
    }

    const int crop = padding * 2 - output_padding;
    if (crop > 0) {
        result = ggml_view_3d(ctx, result, result->ne[0] - crop, result->ne[1], result->ne[2], result->nb[1], result->nb[2], 0);
    }
    if (bias) {
        result = conv_add_bias(ctx, result, bias);
    }
    return result;
}

ggml_tensor* AudioVAE::residual_unit_forward(ggml_context* ctx,
                                             ggml_tensor* x,
                                             const ResidualUnitWeights& weights,
                                             int dilation) const {
    ggml_tensor* h = snake_activation(ctx, x, weights.snake1_alpha);
    h = causal_conv1d_dw(ctx, h, weights.conv1_weight, weights.conv1_bias, 1, dilation, ((7 - 1) * dilation) / 2);
    h = snake_activation(ctx, h, weights.snake2_alpha);
    h = causal_conv1d(ctx, h, weights.conv2_weight, weights.conv2_bias, 1, 1, 1, 0);

    if (x->ne[0] != h->ne[0]) {
        const int64_t target = std::min<int64_t>(x->ne[0], h->ne[0]);
        x = ggml_view_3d(ctx, x, target, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
        h = ggml_view_3d(ctx, h, target, h->ne[1], h->ne[2], h->nb[1], h->nb[2], 0);
    }
    return ggml_add(ctx, x, h);
}

ggml_tensor* AudioVAE::encoder_block_forward(ggml_context* ctx,
                                             ggml_tensor* x,
                                             const EncoderBlockWeights& weights,
                                             int stride) const {
    x = residual_unit_forward(ctx, x, weights.res0, 1);
    x = residual_unit_forward(ctx, x, weights.res1, 3);
    x = residual_unit_forward(ctx, x, weights.res2, 9);
    x = snake_activation(ctx, x, weights.snake_alpha);
    return causal_conv1d(
        ctx,
        x,
        weights.conv_weight,
        weights.conv_bias,
        stride * 2,
        stride,
        1,
        static_cast<int>(std::ceil(stride / 2.0f)));
}

ggml_tensor* AudioVAE::decoder_block_forward(ggml_context* ctx,
                                             ggml_tensor* x,
                                             const DecoderBlockWeights& weights,
                                             ggml_tensor* sr_bucket,
                                             int stride) const {
    x = sample_rate_condition_forward(ctx, x, weights.sr_cond, sr_bucket);
    x = snake_activation(ctx, x, weights.snake_alpha);
    x = causal_transpose_conv1d(ctx,
                                x,
                                weights.conv_weight,
                                weights.conv_bias,
                                stride * 2,
                                stride,
                                static_cast<int>(std::ceil(stride / 2.0f)),
                                stride % 2);
    x = residual_unit_forward(ctx, x, weights.res0, 1);
    x = residual_unit_forward(ctx, x, weights.res1, 3);
    x = residual_unit_forward(ctx, x, weights.res2, 9);
    return x;
}

ggml_tensor* AudioVAE::sample_rate_condition_forward(
    ggml_context* ctx,
    ggml_tensor* x,
    const DecoderBlockWeights::SampleRateConditionWeights& weights,
    ggml_tensor* sr_bucket) const {
    if (!weights.active()) {
        return x;
    }

    VOXCPM_ASSERT(sr_bucket != nullptr);
    VOXCPM_ASSERT(sr_bucket->type == GGML_TYPE_I32);

    ggml_tensor* conditioned = x;
    if (config_.cond_type == "scale_bias" || config_.cond_type == "scale_bias_init") {
        VOXCPM_ASSERT(weights.scale_embed != nullptr);
        VOXCPM_ASSERT(weights.bias_embed != nullptr);
        ggml_tensor* scale = ggml_get_rows(ctx, weights.scale_embed, sr_bucket);
        ggml_tensor* bias = ggml_get_rows(ctx, weights.bias_embed, sr_bucket);
        ggml_tensor* scale_3d = ggml_reshape_3d(ctx, scale, 1, scale->ne[0], 1);
        ggml_tensor* bias_3d = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
        conditioned = ggml_add(ctx,
                               ggml_mul(ctx, conditioned, ggml_repeat(ctx, scale_3d, conditioned)),
                               ggml_repeat(ctx, bias_3d, conditioned));
    } else if (config_.cond_type == "add") {
        VOXCPM_ASSERT(weights.cond_embed != nullptr);
        ggml_tensor* cond = ggml_get_rows(ctx, weights.cond_embed, sr_bucket);
        ggml_tensor* cond_3d = ggml_reshape_3d(ctx, cond, 1, cond->ne[0], 1);
        conditioned = ggml_add(ctx, conditioned, ggml_repeat(ctx, cond_3d, conditioned));
    } else if (config_.cond_type == "concat") {
        VOXCPM_ASSERT(weights.cond_embed != nullptr);
        ggml_tensor* cond = ggml_get_rows(ctx, weights.cond_embed, sr_bucket);
        ggml_tensor* cond_3d = ggml_reshape_3d(ctx, cond, 1, cond->ne[0], 1);
        ggml_tensor* cond_repeat = ggml_repeat(ctx, cond_3d, ggml_new_tensor_3d(ctx, conditioned->type, conditioned->ne[0], cond->ne[0], conditioned->ne[2]));
        conditioned = ggml_cont(ctx, ggml_concat(ctx, conditioned, cond_repeat, 1));
    } else {
        VOXCPM_ASSERT(false && "unsupported AudioVAE sample-rate conditioning type");
    }

    if (weights.out_weight != nullptr) {
        VOXCPM_ASSERT(weights.out_snake_alpha != nullptr);
        conditioned = snake_activation(ctx, conditioned, weights.out_snake_alpha);
        conditioned = causal_conv1d(ctx, conditioned, weights.out_weight, weights.out_bias, 1, 1, 1, 0);
    }

    return conditioned;
}

ggml_tensor* AudioVAE::encode_tensor(VoxCPMContext& ctx,
                                     ggml_tensor* audio) const {
    ggml_context* raw = ctx.raw_context();
    ggml_tensor* x = causal_conv1d(raw, audio, weights_.encoder_block_0_weight, weights_.encoder_block_0_bias, 7, 1, 1, 3);

    for (int i = 0; i < config_.num_encoder_blocks(); ++i) {
        x = encoder_block_forward(raw, x, weights_.encoder_blocks[static_cast<size_t>(i)], config_.encoder_rates[static_cast<size_t>(i)]);
    }

    return causal_conv1d(raw, x, weights_.encoder_fc_mu_weight, weights_.encoder_fc_mu_bias, 3, 1, 1, 1);
}

ggml_tensor* AudioVAE::encode(VoxCPMContext& ctx,
                              std::vector<float>& audio_data,
                              int sample_rate) {
    depthwise_ops_.clear();
    last_preprocessed_audio_ = preprocess(audio_data, sample_rate);
    last_input_tensor_ = ctx.new_tensor_3d(GGML_TYPE_F32, static_cast<int64_t>(last_preprocessed_audio_.size()), 1, 1);
    VOXCPM_ASSERT(last_input_tensor_ != nullptr);
    ggml_set_input(last_input_tensor_);

    ggml_tensor* latent = encode_tensor(ctx, last_input_tensor_);
    ggml_set_output(latent);
    return latent;
}

ggml_tensor* AudioVAE::decode(VoxCPMContext& ctx,
                              ggml_tensor* z) {
    depthwise_ops_.clear();
    last_decode_sr_cond_tensor_ = nullptr;
    last_decode_sr_bucket_ = 0;
    VOXCPM_ASSERT(z != nullptr);
    VOXCPM_ASSERT(derived_.ready && "call ensure_derived_weights() before decode()");
    ggml_context* raw = ctx.raw_context();

    ggml_tensor* x = z;
    if (ggml_n_dims(x) == 2) {
        x = ggml_reshape_3d(raw, x, x->ne[0], x->ne[1], 1);
    }
    VOXCPM_ASSERT(x->ne[1] == config_.latent_dim);
    VOXCPM_ASSERT(x->ne[2] == 1);

    ggml_tensor* sr_bucket = nullptr;
    const bool has_sr_conditioning = std::any_of(weights_.decoder_blocks.begin(),
                                                 weights_.decoder_blocks.end(),
                                                 [](const DecoderBlockWeights& block) {
                                                     return block.sr_cond.active();
                                                 });
    if (has_sr_conditioning) {
        last_decode_sr_bucket_ = static_cast<int32_t>(config_.sample_rate_bucket(config_.output_sample_rate()));
        last_decode_sr_cond_tensor_ = ggml_new_tensor_1d(raw, GGML_TYPE_I32, 1);
        ggml_set_input(last_decode_sr_cond_tensor_);
        sr_bucket = last_decode_sr_cond_tensor_;
    }

    // Stage 2c：decoder 全链切到 L2 布局图（pw K=1 短路 + convT lowering +
    // snake 5 节点；sin 与 final conv k7 留 CPU）
    x = decode_tensor_l2(raw, x, sr_bucket);
    ggml_set_output(x);
    return x;
}

void AudioVAE::prepare_decode_inputs(VoxCPMBackend& backend) const {
    if (last_decode_sr_cond_tensor_ != nullptr) {
        backend.tensor_set(last_decode_sr_cond_tensor_, &last_decode_sr_bucket_, 0, sizeof(last_decode_sr_bucket_));
    }
}

// =============================================================================
// Stage 2c：decoder L2 布局图（激活 [C, T, 1]，通道在 ne[0]）
//
// 与 L1 原图的数学等价性：
//   - pw k1：im2col(k1,s1) 的激活恰为 x^T，mul_mat(w[Cin,Cout], x) 直接给出
//     L2 输出，省去 im2col 拷贝与尾部 permute+cont；
//   - convT：原生 op 的 crop(convT) 语义由双半权重 GEMM + 视图错位 + concat
//     等价实现（Stage 1 冒烟 T8 已对拍），尾部 s 样本内建丢弃；
//   - snake：alpha 的 F32 化与 inv=1/(alpha+eps) 在派生权重中预计算，
//     13 节点缩为 5 节点（sin 留 CPU）；
//   - dw k7：自定义核按 L2 索引（时间步进 nb[1]，通道步进 nb[0]）。
// =============================================================================

namespace {

// Q4_0 每行字节数（32 元素/块 = fp16 d + 16B nibble = 18B）
size_t q4_0_row_bytes(int64_t ne0) {
    VOXCPM_ASSERT(ne0 % 32 == 0);
    return static_cast<size_t>(ne0 / 32) * 18;
}

// convT 折叠权重反量化 + 拆半重排（输出 F16）。
// 真实文件布局：ggml [K*Cout, Cin]，量化块沿 ne[0]——ne[0]=j（j = k + cout*K，
// k 最快）是块维，ne[1]=ci 是行；元素 (j, ci) = w3(k=j%K, cout=j/K, ci)。
// lowering 需要（W_A/W_B 张量 [Cin, s*Cout]，F16）：
//   W_A(ci, i = k*Cout + cout)     = file 值 (j = cout*K + k, ci)   (k∈[0,s))
//   W_B(ci, i = (k-s)*Cout + cout) = file 值 (j = cout*K + k, ci)   (k∈[s,2s))
// file 块的 scale d 被同块 32 个 j 共享，而各 j 拆往不同目标行 → 无法字节级
// 无损拆分；统一走"反量化 → F32 重排 → F16"。
void split_convt_weight_rows(const ggml_tensor* w, int s, std::vector<uint8_t>& wa,
                             std::vector<uint8_t>& wb) {
    VOXCPM_ASSERT(w->type == GGML_TYPE_Q4_0 && ggml_n_dims(w) == 2);
    const int K = 2 * s;
    const int64_t j_n = w->ne[0];              // = K*Cout（块维）
    const int64_t cin = w->ne[1];              // 行数
    VOXCPM_ASSERT(j_n % K == 0);
    const int cout_n = static_cast<int>(j_n / K);
    const size_t rb = q4_0_row_bytes(w->ne[0]);
    const uint8_t* src = static_cast<const uint8_t*>(w->data);
    const int nblk = static_cast<int>(j_n / 32);

    // 反量化 file → f32[ci][j]
    std::vector<float> deq(static_cast<size_t>(cin) * j_n);
    for (int64_t ci = 0; ci < cin; ++ci) {
        const uint8_t* row = src + static_cast<size_t>(ci) * rb;
        for (int b = 0; b < nblk; ++b) {
            const uint8_t* blk = row + static_cast<size_t>(b) * 18;
            ggml_fp16_t d16;
            memcpy(&d16, blk, 2);
            const float df = ggml_fp16_to_fp32(d16);
            float* dst = deq.data() + static_cast<size_t>(ci) * j_n + static_cast<size_t>(b) * 32;
            for (int i = 0; i < 16; ++i) {
                dst[i] = (static_cast<int>(blk[2 + i] & 0x0F) - 8) * df;
                dst[16 + i] = (static_cast<int>(blk[2 + i] >> 4) - 8) * df;
            }
        }
    }

    // F32 重排 → Q8_0 输出（行 = i = k*Cout + cout，行内 ci）。
    // dtype 选择的真机实测依据（SM6650/v73，htp_smoke T8/T9 矩阵）：
    //   Q8_0 是 dspqueue 与 mempool 两变体均数值正确的唯一 GEMM 权重类型；
    //   F16 在真机触发 DSP 端 kernel 拒绝（AEE_EUNSUPPORTED）或数值错误，
    //   Q4_0 在 dspqueue 变体输出全零。Q8_0 per-32 块量化精度损失可忽略，
    //   CPU 回退路径 ggml 原生支持（仅影响 VOXCPM_BACKEND=cpu 保底模式性能）。
    const size_t row_elems = cin;
    std::vector<float> waf(row_elems * static_cast<size_t>(s) * cout_n);
    std::vector<float> wbf(row_elems * static_cast<size_t>(s) * cout_n);
    for (int c = 0; c < cout_n; ++c) {
        for (int k = 0; k < K; ++k) {
            const int64_t j = static_cast<int64_t>(c) * K + k;
            std::vector<float>* dst_f32 = (k < s) ? &waf : &wbf;
            const int i = (k < s) ? (k * cout_n + c) : ((k - s) * cout_n + c);
            for (int64_t ci = 0; ci < cin; ++ci) {
                (*dst_f32)[static_cast<size_t>(i) * cin + ci] =
                    deq[static_cast<size_t>(ci) * j_n + j];
            }
        }
    }

    auto to_q8_0_bytes = [](const std::vector<float>& f, int64_t row_elems) {
        VOXCPM_ASSERT(row_elems % 32 == 0);
        const int64_t nblk = row_elems / 32;
        const int64_t nrows =
            static_cast<int64_t>(f.size()) / row_elems;
        std::vector<uint8_t> bytes(static_cast<size_t>(f.size() / row_elems) *
                                   static_cast<size_t>(nblk) * 34);
        for (int64_t r = 0; r < nrows; ++r) {
            for (int64_t b = 0; b < nblk; ++b) {
                const float* blk =
                    f.data() + static_cast<size_t>(r) * row_elems +
                    static_cast<size_t>(b) * 32;
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    amax = std::max(amax, std::abs(blk[i]));
                }
                const float d = amax / 127.0f;
                const ggml_fp16_t d16 = ggml_fp32_to_fp16(d);
                uint8_t* out = bytes.data() +
                               (static_cast<size_t>(r) * nblk +
                                static_cast<size_t>(b)) * 34;
                memcpy(out, &d16, 2);
                const float inv_d = d > 0.0f ? 1.0f / d : 0.0f;
                for (int i = 0; i < 32; ++i) {
                    out[2 + i] = static_cast<uint8_t>(
                        static_cast<int>(std::lrintf(blk[i] * inv_d)));
                }
            }
        }
        return bytes;
    };
    wa = to_q8_0_bytes(waf, cin);
    wb = to_q8_0_bytes(wbf, cin);
}

// 任意 F16/F32 张量 → host F32
void to_f32_host(const ggml_tensor* t, std::vector<float>& out) {
    const int64_t n = ggml_nelements(t);
    out.resize(static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, sizeof(float) * static_cast<size_t>(n));
        return;
    }
    VOXCPM_ASSERT(t->type == GGML_TYPE_F16);
    const auto* src = static_cast<const ggml_fp16_t*>(t->data);
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(i)] = ggml_fp16_to_fp32(src[i]);
    }
}

}  // namespace

bool AudioVAE::ensure_derived_weights(VoxCPMBackend& backend) {
    if (derived_.ready) {
        return true;
    }

    // 目标 buffer：W_A/W_B（GEMM 大头，仅被 MUL_MAT 消费）在异构模式建在
    // HTP（USAGE_WEIGHTS）；bias/alpha/inv 体积小、与 CPU 侧 snake/dw 同侧
    // 消费，恒留 CPU——避免 CPU/HTP 交叉过密。两种模式共用同一张 L2 图。
    // VOXCPM_HTP_DERIVED=0 可把 W_A/W_B 也留在 CPU（A/B 调试开关）。
    const bool use_htp = backend.is_htp_active() && backend.htp_buffer_type() != nullptr &&
                         [](const char* e) { return e == nullptr || e[0] != '0'; }(
                             std::getenv("VOXCPM_HTP_DERIVED"));
    ggml_backend_buffer_type_t buft =
        use_htp ? backend.htp_buffer_type() : backend.buffer_type();

    const size_t n_derived_hint = 128;
    derived_.ctx = ggml_init({
        /*mem_size=*/ggml_tensor_overhead() * n_derived_hint + 4096,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    });
    derived_cpu_.ctx = ggml_init({
        /*mem_size=*/ggml_tensor_overhead() * n_derived_hint + 4096,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    });
    if (!derived_.ctx || !derived_cpu_.ctx) {
        return false;
    }

    // 待灌数据表：派生名 → (张量指针, host 数据, 是否 HTP 桶)
    struct PendingSet {
        ggml_tensor* tensor;
        std::vector<uint8_t> bytes;
        bool htp = false;
    };
    std::vector<PendingSet> pending;

    auto new_derived_2d = [&](const std::string& key, ggml_type type, int64_t ne0,
                              int64_t ne1, bool htp_bucket) -> ggml_tensor* {
        ggml_context* ctx = htp_bucket ? derived_.ctx : derived_cpu_.ctx;
        ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        (htp_bucket ? derived_ : derived_cpu_).tensors.emplace(key, t);
        return t;
    };

    // ---- convT：行重排拆半 W_A / W_B ----
    for (size_t bi = 0; bi < weights_.decoder_blocks.size(); ++bi) {
        const DecoderBlockWeights& block = weights_.decoder_blocks[bi];
        const int s = config_.decoder_rates[bi];
        const ggml_tensor* w = block.conv_weight;
        VOXCPM_ASSERT(w != nullptr);
        std::vector<uint8_t> wa, wb;
        split_convt_weight_rows(w, s, wa, wb);
        PendingSet& pa = pending.emplace_back();
        // mul_mat 权重约定 [行内元素数=Cin, 行数=s*Cout]；Q8_0 见
        // split_convt_weight_rows 内 dtype 说明
        pa.tensor = new_derived_2d(std::string(w->name) + "/wa", GGML_TYPE_Q8_0,
                                   w->ne[1], w->ne[0] / 2, use_htp);
        pa.bytes = std::move(wa);
        PendingSet& pb = pending.emplace_back();
        pb.tensor = new_derived_2d(std::string(w->name) + "/wb", GGML_TYPE_Q8_0,
                                   w->ne[1], w->ne[0] / 2, use_htp);
        pb.bytes = std::move(wb);
    }

    // ---- bias / alpha / inv 的 F32 化 ----
    auto derive_vector = [&](const std::string& key, const ggml_tensor* src, float eps,
                             bool with_inv) {
        if (src == nullptr) {
            return;
        }
        std::vector<float> v;
        to_f32_host(src, v);
        PendingSet& p = pending.emplace_back();
        p.tensor = new_derived_2d(key, GGML_TYPE_F32, ggml_nelements(src), 1, false);
        p.bytes.resize(v.size() * sizeof(float));
        memcpy(p.bytes.data(), v.data(), p.bytes.size());
        if (with_inv) {
            std::vector<float> inv(v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                inv[i] = 1.0f / (v[i] + eps);
            }
            PendingSet& pi = pending.emplace_back();
            pi.tensor = new_derived_2d(key + "/inv", GGML_TYPE_F32, static_cast<int64_t>(inv.size()), 1, false);
            pi.bytes.resize(inv.size() * sizeof(float));
            memcpy(pi.bytes.data(), inv.data(), pi.bytes.size());
        }
    };

    derive_vector("decoder.model.0.bias/f32", weights_.decoder_model_0_bias, 0.0f, false);
    derive_vector("decoder.model.1.bias/f32", weights_.decoder_model_1_bias, 0.0f, false);
    for (const DecoderBlockWeights& block : weights_.decoder_blocks) {
        derive_vector(std::string(block.conv_bias->name) + "/f32", block.conv_bias, 0.0f, false);
        derive_vector(std::string(block.snake_alpha->name) + "/f32", block.snake_alpha,
                      1e-9f, true);
        for (const ResidualUnitWeights* res : {&block.res0, &block.res1, &block.res2}) {
            derive_vector(std::string(res->snake1_alpha->name) + "/f32", res->snake1_alpha,
                          1e-9f, true);
            derive_vector(std::string(res->conv1_bias->name) + "/f32", res->conv1_bias, 0.0f, false);
            derive_vector(std::string(res->snake2_alpha->name) + "/f32", res->snake2_alpha,
                          1e-9f, true);
            derive_vector(std::string(res->conv2_bias->name) + "/f32", res->conv2_bias, 0.0f, false);
        }
        if (block.sr_cond.out_bias != nullptr) {
            derive_vector(std::string(block.sr_cond.out_bias->name) + "/f32",
                          block.sr_cond.out_bias, 0.0f, false);
        }
        if (block.sr_cond.out_snake_alpha != nullptr) {
            derive_vector(std::string(block.sr_cond.out_snake_alpha->name) + "/f32",
                          block.sr_cond.out_snake_alpha, 1e-9f, true);
        }
    }
    derive_vector("decoder.final.bias/f32", weights_.decoder_final_conv_bias, 0.0f, false);
    derive_vector("decoder.final.alpha/f32", weights_.decoder_final_snake_alpha, 1e-9f, true);

    // 分配：HTP 桶（必须 USAGE_WEIGHTS）+ CPU 桶
    if (use_htp) {
        derived_.buffer = ggml_backend_alloc_ctx_tensors_from_buft(derived_.ctx, buft);
    }
    ggml_backend_buffer_type_t cpu_buft = backend.buffer_type();
    derived_cpu_.buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(derived_cpu_.ctx, cpu_buft);
    if (use_htp) {
        ggml_backend_buffer_set_usage(derived_.buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    if ((use_htp && !derived_.buffer) || !derived_cpu_.buffer) {
        return false;
    }

    for (PendingSet& p : pending) {
        backend.tensor_set(p.tensor, p.bytes.data(), 0, p.bytes.size());
    }
    derived_.ready = true;
    return true;
}

ggml_tensor* AudioVAE::lookup_derived(const std::string& key) const {
    {
        const auto it = derived_.tensors.find(key);
        if (it != derived_.tensors.end()) {
            return it->second;
        }
    }
    const auto it = derived_cpu_.tensors.find(key);
    if (it == derived_cpu_.tensors.end()) {
        std::fprintf(stderr, "[derived] missing key: %s\n", key.c_str());
        VOXCPM_ASSERT(false && "missing derived weight");
    }
    return it->second;
}

const std::vector<ggml_tensor*>& AudioVAE::ensure_convt_inputs(VoxCPMBackend& backend,
                                                               int64_t t_len) {
    if (convt_input_t_ == t_len && !convt_inputs_.empty()) {
        return convt_inputs_;
    }
    // T 变化则重建
    if (convt_input_buffer_) {
        ggml_backend_buffer_free(convt_input_buffer_);
        convt_input_buffer_ = nullptr;
    }
    if (convt_input_ctx_) {
        ggml_free(convt_input_ctx_);
        convt_input_ctx_ = nullptr;
    }
    convt_inputs_.clear();

    const bool use_htp = backend.is_htp_active() && backend.htp_buffer_type() != nullptr;
    const ggml_backend_buffer_type_t buft =
        use_htp ? backend.htp_buffer_type() : backend.buffer_type();
    const size_t n = weights_.decoder_blocks.size();
    convt_input_ctx_ = ggml_init({
        /*mem_size=*/ggml_tensor_overhead() * (n + 1) + 4096,
        /*mem_buffer=*/nullptr,
        /*no_alloc=*/true,
    });
    if (!convt_input_ctx_) {
        return convt_inputs_;
    }
    for (size_t bi = 0; bi < n; ++bi) {
        const ggml_tensor* w = weights_.decoder_blocks[bi].conv_weight;
        const int64_t cin = w->ne[1];
        ggml_tensor* t = ggml_new_tensor_2d(convt_input_ctx_, GGML_TYPE_F32, cin, t_len);
        convt_inputs_.push_back(t);
    }
    convt_input_buffer_ =
        ggml_backend_alloc_ctx_tensors_from_buft(convt_input_ctx_, buft);
    if (!convt_input_buffer_) {
        convt_inputs_.clear();
        return convt_inputs_;
    }
    convt_input_t_ = t_len;
    return convt_inputs_;
}

// =============================================================================
// Stage 2c：L2 前向实现
// =============================================================================

namespace {

// L2 深度卷积自定义核：x [C, T, 1] / w [K, 1, C] / b [C]
// 语义与 L1 版 depthwise_conv_custom 一致：y[c,t] = b[c] + Σ_k x[c, t*s + k*d - 2p] * w[k,c]
void depthwise_conv_l2_custom(ggml_tensor* dst,
                              const ggml_tensor* x,
                              const ggml_tensor* weight,
                              const ggml_tensor* bias,
                              int ith,
                              int nth,
                              void* userdata) {
    const auto* op = static_cast<const AudioVAEDepthwiseConvOpData*>(userdata);
    const int64_t t_len = x->ne[1];      // L2：时间在 ne[1]
    const int64_t channels = x->ne[0];   // 通道在 ne[0]
    const int64_t batch = x->ne[2];
    const int64_t kernel = weight->ne[0];

    VOXCPM_ASSERT(weight->ne[1] == 1);
    VOXCPM_ASSERT(weight->ne[2] == channels);
    VOXCPM_ASSERT(dst->ne[0] == channels);
    VOXCPM_ASSERT(dst->ne[1] == t_len);
    VOXCPM_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const uint8_t* x_data = static_cast<const uint8_t*>(x->data);
    const uint8_t* w_data = static_cast<const uint8_t*>(weight->data);
    const uint8_t* b_data = bias ? static_cast<const uint8_t*>(bias->data) : nullptr;
    uint8_t* dst_data = static_cast<uint8_t*>(dst->data);

    const int64_t work_items = channels * batch;
    const int64_t items_per_thread = (work_items + nth - 1) / nth;
    const int64_t item_begin = ith * items_per_thread;
    const int64_t item_end = std::min<int64_t>(item_begin + items_per_thread, work_items);

    for (int64_t item = item_begin; item < item_end; ++item) {
        const int64_t b = item / channels;
        const int64_t c = item % channels;
        const float bias_val = b_data ? load_f32_scalar(b_data + c * bias->nb[0], bias->type) : 0.0f;

        // L2：通道间步进 nb[0]，时间间步进 nb[1]
        const uint8_t* x_channel = x_data + b * x->nb[2] + c * x->nb[0];
        const uint8_t* w_channel = w_data + c * weight->nb[2];
        uint8_t* dst_channel = dst_data + b * dst->nb[2] + c * dst->nb[0];

        for (int64_t t = 0; t < t_len; ++t) {
            float sum = bias_val;
            for (int64_t k = 0; k < kernel; ++k) {
                const int64_t src_t = t * op->stride + k * op->dilation - op->padding * 2;
                if (src_t < 0 || src_t >= t_len) {
                    continue;
                }
                const float x_val = load_f32_scalar(x_channel + src_t * x->nb[1], x->type);
                const float w_val = load_f32_scalar(w_channel + k * weight->nb[0], weight->type);
                sum += x_val * w_val;
            }
            *reinterpret_cast<float*>(dst_channel + t * dst->nb[1]) = sum;
        }
    }
}

// L2 snake（5 节点）：y = x + sin(x·a)²·inv，alpha/inv 以 [C,1,1] 视图广播。
// sin 留 CPU，其余节点可上 HTP。
ggml_tensor* snake_l2(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha_f32,
                      ggml_tensor* inv_f32) {
    const int64_t channels = ggml_nelements(alpha_f32);
    ggml_tensor* a3 = ggml_reshape_3d(ctx, alpha_f32, channels, 1, 1);
    ggml_tensor* i3 = ggml_reshape_3d(ctx, inv_f32, channels, 1, 1);
    ggml_tensor* ax = ggml_mul(ctx, x, a3);
    ggml_tensor* s = ggml_sin(ctx, ax);
    return ggml_add(ctx, x, ggml_mul(ctx, ggml_sqr(ctx, s), i3));
}

// L2 pw k1 + bias：x [Cin, T, 1] → [Cout, T, 1]。
// k1 的 im2col 恰为 x^T，故 mul_mat(w[Cin,Cout], XT) 直接给出 L2 输出。
ggml_tensor* pointwise_l2(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_q4_0,
                          ggml_tensor* bias_f32) {
    const int64_t cin = w_q4_0->ne[0];
    const int64_t t_len = x->ne[1];
    VOXCPM_ASSERT(x->ne[0] == cin);
    ggml_tensor* xt = ggml_reshape_2d(ctx, x, cin, t_len);
    ggml_tensor* y = ggml_mul_mat(ctx, w_q4_0, xt);   // [Cout, T]
    ggml_tensor* y3 = ggml_reshape_3d(ctx, y, w_q4_0->ne[1], t_len, 1);
    if (bias_f32 == nullptr) {
        return y3;
    }
    const int64_t cout_n = ggml_nelements(bias_f32);
    ggml_tensor* b3 = ggml_reshape_3d(ctx, bias_f32, cout_n, 1, 1);
    return ggml_add(ctx, y3, b3);
}

}  // namespace

ggml_tensor* AudioVAE::residual_unit_forward_l2(ggml_context* ctx,
                                                ggml_tensor* x,
                                                const ResidualUnitWeights& weights,
                                                int dilation) const {
    auto op = std::make_unique<AudioVAEDepthwiseConvOpData>();
    op->stride = 1;
    op->dilation = dilation;
    op->padding = ((7 - 1) * dilation) / 2;
    AudioVAEDepthwiseConvOpData* op_ptr = op.get();
    depthwise_ops_.push_back(std::move(op));

    ggml_tensor* h = snake_l2(ctx, x,
                              lookup_derived(std::string(weights.snake1_alpha->name) + "/f32"),
                              lookup_derived(std::string(weights.snake1_alpha->name) + "/f32/inv"));
    h = ggml_map_custom3(ctx, h, weights.conv1_weight, weights.conv1_bias,
                         depthwise_conv_l2_custom, GGML_N_TASKS_MAX, op_ptr);
    h = snake_l2(ctx, h,
                 lookup_derived(std::string(weights.snake2_alpha->name) + "/f32"),
                 lookup_derived(std::string(weights.snake2_alpha->name) + "/f32/inv"));
    h = pointwise_l2(ctx, h, weights.conv2_weight,
                     lookup_derived(std::string(weights.conv2_bias->name) + "/f32"));

    // 残差对齐（L1 版对齐时间轴 ne[0]，L2 下时间在 ne[1]）
    if (x->ne[1] != h->ne[1]) {
        const int64_t target = std::min<int64_t>(x->ne[1], h->ne[1]);
        x = ggml_view_3d(ctx, x, x->ne[0], target, x->ne[2], x->nb[1], x->nb[2], 0);
        h = ggml_view_3d(ctx, h, h->ne[0], target, h->ne[2], h->nb[1], h->nb[2], 0);
    }
    return ggml_add(ctx, x, h);
}

ggml_tensor* AudioVAE::sample_rate_condition_forward_l2(
    ggml_context* ctx,
    ggml_tensor* x,
    const DecoderBlockWeights::SampleRateConditionWeights& weights,
    ggml_tensor* sr_bucket) const {
    if (!weights.active()) {
        return x;
    }
    // 当前模型未携带 sr_cond 权重（实测 0 张量），此路径仅在带权重的模型上生效。
    // L2 语义：embed get_rows → [C, 1, 1] 广播（省 REPEAT）。
    VOXCPM_ASSERT(sr_bucket != nullptr && sr_bucket->type == GGML_TYPE_I32);
    ggml_tensor* conditioned = x;
    if (config_.cond_type == "scale_bias" || config_.cond_type == "scale_bias_init") {
        ggml_tensor* scale = ggml_get_rows(ctx, weights.scale_embed, sr_bucket);
        ggml_tensor* bias = ggml_get_rows(ctx, weights.bias_embed, sr_bucket);
        conditioned = ggml_add(ctx,
                               ggml_mul(ctx, conditioned,
                                        ggml_reshape_3d(ctx, scale, scale->ne[0], 1, 1)),
                               ggml_reshape_3d(ctx, bias, bias->ne[0], 1, 1));
    } else if (config_.cond_type == "add") {
        ggml_tensor* cond = ggml_get_rows(ctx, weights.cond_embed, sr_bucket);
        conditioned = ggml_add(ctx, conditioned,
                               ggml_reshape_3d(ctx, cond, cond->ne[0], 1, 1));
    } else {
        VOXCPM_ASSERT(false && "unsupported cond_type in L2 decoder");
    }

    if (weights.out_weight != nullptr) {
        conditioned = snake_l2(ctx, conditioned,
                               lookup_derived(std::string(weights.out_snake_alpha->name) + "/f32"),
                               lookup_derived(std::string(weights.out_snake_alpha->name) + "/f32/inv"));
        conditioned = pointwise_l2(ctx, conditioned, weights.out_weight,
                                   lookup_derived(std::string(weights.out_bias->name) + "/f32"));
    }
    return conditioned;
}

ggml_tensor* AudioVAE::decoder_block_forward_l2(ggml_context* ctx,
                                                ggml_tensor* x,
                                                const DecoderBlockWeights& weights,
                                                ggml_tensor* sr_bucket,
                                                int stride) const {
    x = sample_rate_condition_forward_l2(ctx, x, weights.sr_cond, sr_bucket);
    x = snake_l2(ctx, x,
                 lookup_derived(std::string(weights.snake_alpha->name) + "/f32"),
                 lookup_derived(std::string(weights.snake_alpha->name) + "/f32/inv"));

    // convT：行重排派生 W_A/W_B + lowering（尾部 crop 内建）
    const std::string wn(weights.conv_weight->name);
    const std::string bn(weights.conv_bias->name);
    x = causal_transpose_conv1d_l2(ctx, x,
                                   lookup_derived(wn + "/wa"),
                                   lookup_derived(wn + "/wb"),
                                   lookup_derived(bn + "/f32"),
                                   stride);

    x = residual_unit_forward_l2(ctx, x, weights.res0, 1);
    x = residual_unit_forward_l2(ctx, x, weights.res1, 3);
    x = residual_unit_forward_l2(ctx, x, weights.res2, 9);
    return x;
}

ggml_tensor* AudioVAE::decode_tensor_l2(ggml_context* ctx, ggml_tensor* z,
                                        ggml_tensor* sr_bucket) const {
    // 入口 [T, C, 1]（L1 latent）→ L2 [C, T, 1]
    ggml_tensor* x = ggml_cont(ctx, ggml_permute(ctx, z, 1, 0, 2, 3));

    // model.0：dw k7（causal pad 3）
    {
        auto op = std::make_unique<AudioVAEDepthwiseConvOpData>();
        op->stride = 1;
        op->dilation = 1;
        op->padding = 3;
        AudioVAEDepthwiseConvOpData* op_ptr = op.get();
        depthwise_ops_.push_back(std::move(op));
        x = ggml_map_custom3(ctx, x, weights_.decoder_model_0_weight,
                             weights_.decoder_model_0_bias, depthwise_conv_l2_custom,
                             GGML_N_TASKS_MAX, op_ptr);
    }

    // model.1：pw 64→1536（K=1 短路，替代 im2col 路径）
    // L2 布局（[C,T,1]，通道最快维）下激活本身就是 XT [Cin, T]——一次 reshape_2d 视图（零拷贝）直接喂 GEMM，省掉 im2col 展开和尾部转置
    x = pointwise_l2(ctx, x, weights_.decoder_model_1_weight,
                     lookup_derived("decoder.model.1.bias/f32"));

    const bool has_sr_conditioning = std::any_of(
        weights_.decoder_blocks.begin(), weights_.decoder_blocks.end(),
        [](const DecoderBlockWeights& block) { return block.sr_cond.active(); });

    for (size_t i = 0; i < weights_.decoder_blocks.size(); ++i) {
        x = decoder_block_forward_l2(ctx, x, weights_.decoder_blocks[i],
                                     has_sr_conditioning ? sr_bucket : nullptr,
                                     config_.decoder_rates[i]);
    }

    // final snake → 回 L1 → final conv k7（普通 conv，全 CPU 原路径）→ tanh
    x = snake_l2(ctx, x, lookup_derived("decoder.final.alpha/f32"),
                 lookup_derived("decoder.final.alpha/f32/inv"));
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));   // [T, C, 1]
    x = causal_conv1d(ctx, x, weights_.decoder_final_conv_weight,
                      weights_.decoder_final_conv_bias, 7, 1, 1, 3);
    return ggml_tanh(ctx, x);
}

// L2 convT lowering：x [Cin, T, 1] → [Cout, T*s, 1]
//   F_A = mul_mat(W_A[Cin, s·Cout], XT) → [s·Cout, T]
//   FA3 = reshape_3d(F_A, Cout, s, T)     // (c, r, t)，r = 半核 k 索引
//   D   = add(view(FA3, t∈[1,T)), view(FB3, t∈[0,T-1)))   // [Cout, s, T-1]
//   D2  = reshape_2d(D, Cout, (T-1)·s)
//   Y   = concat(view(FA3, [Cout, s], t=0), D2, dim=1)    // [Cout, T·s]
// 尾部 s 样本（原 convT crop 语义）内建丢弃。
ggml_tensor* AudioVAE::causal_transpose_conv1d_l2(ggml_context* ctx, ggml_tensor* x,
                                                  ggml_tensor* wa, ggml_tensor* wb,
                                                  ggml_tensor* bias_f32, int stride) const {
    // W_A/W_B 为 mul_mat 权重 [Cin, s*Cout]，行 i = k*Cout + cout（k 主序）
    const int64_t cin = wa->ne[0];
    const int64_t cout_n = wa->ne[1] / stride;
    const int64_t t_len = x->ne[1];
    VOXCPM_ASSERT(x->ne[0] == cin);

    ggml_tensor* xt = ggml_reshape_2d(ctx, x, cin, t_len);
    ggml_tensor* fa = ggml_mul_mat(ctx, wa, xt);
    ggml_tensor* fb = ggml_mul_mat(ctx, wb, xt);
    ggml_tensor* fa3 = ggml_reshape_3d(ctx, fa, cout_n, stride, t_len);
    ggml_tensor* fb3 = ggml_reshape_3d(ctx, fb, cout_n, stride, t_len);

    const size_t nb1 = sizeof(float) * cout_n;
    const size_t nb2 = nb1 * stride;
    ggml_tensor* va = ggml_view_3d(ctx, fa3, cout_n, stride, t_len - 1, nb1, nb2,
                                   sizeof(float) * cout_n * stride);
    ggml_tensor* vb = ggml_view_3d(ctx, fb3, cout_n, stride, t_len - 1, nb1, nb2, 0);
    ggml_tensor* d = ggml_add(ctx, va, vb);
    ggml_tensor* d2 = ggml_reshape_2d(ctx, d, cout_n, (t_len - 1) * stride);
    ggml_tensor* head = ggml_view_2d(ctx, fa3, cout_n, stride, nb1, 0);
    ggml_tensor* y = ggml_concat(ctx, head, d2, 1);   // [Cout, T*s]

    ggml_tensor* y3 = ggml_reshape_3d(ctx, y, cout_n, t_len * stride, 1);
    if (bias_f32 == nullptr) {
        return y3;
    }
    ggml_tensor* b3 = ggml_reshape_3d(ctx, bias_f32, ggml_nelements(bias_f32), 1, 1);
    return ggml_add(ctx, y3, b3);
}



// =============================================================================
// Stage 2d：AudioVAEL2Plan——拆段解码计划
//
// L2 decoder 按后端拆成 2n+1 张单后端图（CPU/HTP 交替），段间激活经
// tensor_get/tensor_set 显式搬运，绕开 sched 跨设备拷贝失效问题；
// 图一次构建多步复用，uid 稳定使 HTP opbatch 只编译一次。
// =============================================================================

struct AudioVAEL2Plan::Segment {
    // VoxCPMContext 无默认构造：段创建时以占位参数构造后由 build 重新赋值（move）
    VoxCPMContext ctx = VoxCPMContext(ContextType::Graph, 1, 1);
    ggml_cgraph* graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    // 真机（SM6650/v73 + mempool 变体）实测：direct graph_compute 提交该段
    // opbatch 时 DSP 返回 AEE_EUNSUPPORTED（htp_smoke T10 复现），而
    // sched 路径（T10_MODE=sched1/sched2）数值正确——执行一律走 sched
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t backend = nullptr;  // 本段唯一执行后端
    bool is_htp = false;
    ggml_tensor* input = nullptr;      // 段输入（段 ctx 内，分配于本段 buffer）
    ggml_tensor* output = nullptr;     // 段输出
    ggml_tensor* input2 = nullptr;     // 第二输入（HTP convT 段：F_B 镜像）
    ggml_tensor* output2 = nullptr;    // 第二输出（HTP convT 段：F_B）
    ggml_backend_buffer_t input_buffer = nullptr;   // 段输入宿主 buffer（CPU）
    ggml_backend_buffer_t input2_buffer = nullptr;
    std::vector<std::unique_ptr<AudioVAEDepthwiseConvOpData>> dw_data;  // 深度卷积参数生命周期
};

AudioVAEL2Plan::AudioVAEL2Plan() = default;

AudioVAEL2Plan::~AudioVAEL2Plan() {
    for (auto& seg : segs_) {
        if (seg) {
            if (seg->gallocr) {
                ggml_gallocr_free(seg->gallocr);
            }
            if (seg->sched) {
                ggml_backend_sched_free(seg->sched);
            }
            if (seg->input_buffer) {
                ggml_backend_buffer_free(seg->input_buffer);
            }
            if (seg->input2_buffer) {
                ggml_backend_buffer_free(seg->input2_buffer);
            }
        }
    }
}

bool AudioVAEL2Plan::build(AudioVAE& vae, VoxCPMBackend& backend, int64_t t_len) {
    if (!segs_.empty()) {
        return true;
    }
    if (!vae.ensure_derived_weights(backend)) {
        return false;
    }
    const bool htp = backend.is_htp_active();
    ggml_backend_t be_htp = htp ? backend.htp_backend() : nullptr;
    ggml_backend_t be_cpu = backend.raw_backend();
    (void)be_htp;

    const AudioVAEConfig& cfg = vae.config();
    const int n_blocks = cfg.num_decoder_blocks();
    const int n_segs = 2 * n_blocks + 1;
    segs_.resize(n_segs);  // 占位（unique_ptr）

    // 各段公共：建段（ctx/后端/输入张量），返回段引用供组图
    auto new_seg = [&](int si, int64_t in_ne0, int64_t in_ne1) -> Segment& {
        auto seg = std::make_unique<Segment>();
        seg->is_htp = (si % 2 == 1) && htp;
        seg->backend = seg->is_htp ? be_htp : be_cpu;
        seg->ctx = VoxCPMContext(ContextType::Graph, 8192, 65536);
        seg->input = seg->ctx.new_tensor_2d(GGML_TYPE_F32, in_ne0, in_ne1);
        // sched 需要 input/output 标记为无 buffer leaf 指定归属 backend
        ggml_set_input(seg->input);
        segs_[si] = std::move(seg);
        return *segs_[si];
    };

    // 线性推进：in_shape 由上一段输出给出；首段输入 = latent [t_len, C]
    int64_t in0 = t_len, in1 = cfg.latent_dim;

    // ---- seg0（CPU）：L1→L2 → dw0 → pw1 → block2 头链（srcond 透传 + snake）→ XT2 ----
    {
        Segment& seg = new_seg(0, in0, in1);
        ggml_context* c = seg.ctx.raw_context();
        ggml_tensor* z = seg.input;
        ggml_tensor* x = ggml_cont(c, ggml_permute(c, z, 1, 0, 2, 3));   // → [C, T] L2

        // model.0 dw k7（pad 3）
        seg.dw_data.push_back(std::make_unique<AudioVAEDepthwiseConvOpData>(
            AudioVAEDepthwiseConvOpData{1, 1, 3}));
        x = ggml_map_custom3(c, x, vae.weights().decoder_model_0_weight,
                             vae.weights().decoder_model_0_bias, depthwise_conv_l2_custom,
                             GGML_N_TASKS_MAX, seg.dw_data.back().get());

        // model.1 pw k1 + bias
        {
            ggml_tensor* xt = ggml_reshape_2d(c, x, x->ne[0], x->ne[1]);
            ggml_tensor* yw = ggml_mul_mat(c, vae.weights().decoder_model_1_weight, xt);
            x = ggml_reshape_3d(c, yw, vae.weights().decoder_model_1_weight->ne[1], x->ne[1], 1);
            ggml_tensor* b3 = ggml_reshape_3d(c, vae.lookup_derived("decoder.model.1.bias/f32"),
                                              ggml_nelements(vae.lookup_derived("decoder.model.1.bias/f32")), 1, 1);
            x = ggml_add(c, x, b3);
        }

        // block2 头链（srcond 当前恒透传 + snake）
        const DecoderBlockWeights& blk = vae.weights().decoder_blocks[0];
        x = snake_l2(c, x,
                             vae.lookup_derived(std::string(blk.snake_alpha->name) + "/f32"),
                             vae.lookup_derived(std::string(blk.snake_alpha->name) + "/f32/inv"));

        seg.output = x;
        seg.graph = seg.ctx.new_graph(512);
        ggml_build_forward_expand(seg.graph, x);
    }

    // ---- 奇数段（HTP）：convT bi；偶数段（CPU）：res×3 + 下一块头链 ----
    for (int bi = 0; bi < n_blocks; ++bi) {
        const DecoderBlockWeights& blk = vae.weights().decoder_blocks[static_cast<size_t>(bi)];
        const int s = cfg.decoder_rates[static_cast<size_t>(bi)];
        const std::string wn(blk.conv_weight->name);
        const std::string bn(blk.conv_bias->name);

        // convT 段（输入形状 = 前一 CPU 段输出）
        {
            const ggml_tensor* prev_out = segs_[static_cast<size_t>(2 * bi)]->output;
            Segment& seg = new_seg(2 * bi + 1, prev_out->ne[0], prev_out->ne[1]);
            ggml_context* c = seg.ctx.raw_context();
            ggml_tensor* xt = seg.input;   // [Cin, T]
            // HTP 只做双 mul_mat（GEMM 是需要加速的部分）；view/concat/bias
            // 在下一段 CPU 侧组装（CONCAT 的直接 graph_compute 在该后端异常）
            seg.output = ggml_mul_mat(c, vae.lookup_derived(wn + "/wa"), xt);   // F_A
            seg.output2 = ggml_mul_mat(c, vae.lookup_derived(wn + "/wb"), xt);  // F_B
            seg.graph = seg.ctx.new_graph(512);
            ggml_build_forward_expand(seg.graph, seg.output);
            ggml_build_forward_expand(seg.graph, seg.output2);
        }

        // 末块的收尾段（res×3 + final）特殊处理；中间块为 res×3 + 下一块头链
        const bool last = (bi == n_blocks - 1);
        if (!last) {
            const ggml_tensor* fa_out = segs_[static_cast<size_t>(2 * bi + 1)]->output;
            const ggml_tensor* fb_out = segs_[static_cast<size_t>(2 * bi + 1)]->output2;
            const int64_t s_cout = fa_out->ne[0];
            const int64_t t_len = fa_out->ne[1];
            const int64_t cout_n = vae.lookup_derived(wn + "/wa")->ne[1] / s;
            Segment& seg = new_seg(2 * bi + 2, s_cout, t_len);   // 输入 F_A 镜像
            seg.input2 = ggml_new_tensor_2d(seg.ctx.raw_context(), GGML_TYPE_F32,
                                            s_cout, t_len);      // F_B 镜像
            ggml_context* c = seg.ctx.raw_context();
            // convT 语义组装（CPU）：错位视图相加 + concat + bias
            ggml_tensor* fa3 = ggml_reshape_3d(c, seg.input, cout_n, s, t_len);
            ggml_tensor* fb3 = ggml_reshape_3d(c, seg.input2, cout_n, s, t_len);
            const size_t nb1 = sizeof(float) * cout_n;
            const size_t nb2 = nb1 * s;
            ggml_tensor* d = ggml_add(c,
                ggml_view_3d(c, fa3, cout_n, s, t_len - 1, nb1, nb2, sizeof(float) * cout_n * s),
                ggml_view_3d(c, fb3, cout_n, s, t_len - 1, nb1, nb2, 0));
            ggml_tensor* d2 = ggml_reshape_2d(c, d, cout_n, (t_len - 1) * s);
            ggml_tensor* head = ggml_view_2d(c, fa3, cout_n, s, nb1, 0);
            ggml_tensor* y = ggml_concat(c, head, d2, 1);
            ggml_tensor* y3 = ggml_reshape_3d(c, y, cout_n, t_len * s, 1);
            ggml_tensor* b3 = ggml_reshape_3d(c, vae.lookup_derived(bn + "/f32"),
                                              ggml_nelements(vae.lookup_derived(bn + "/f32")), 1, 1);
            ggml_tensor* x = ggml_add(c, y3, b3);

            const DecoderBlockWeights& nb = vae.weights().decoder_blocks[static_cast<size_t>(bi + 1)];
            auto res_chain = [&](const DecoderBlockWeights& b) {
                const ResidualUnitWeights* ress[3] = {&b.res0, &b.res1, &b.res2};
                const int dilations[3] = {1, 3, 9};
                for (int ri = 0; ri < 3; ++ri) {
                    x = vae.residual_unit_forward_l2(c, x, *ress[ri], dilations[ri]);
                }
            };
            res_chain(blk);
            x = snake_l2(c, x,
                         vae.lookup_derived(std::string(nb.snake_alpha->name) + "/f32"),
                         vae.lookup_derived(std::string(nb.snake_alpha->name) + "/f32/inv"));
            seg.output = x;
            seg.graph = seg.ctx.new_graph(512);
            ggml_build_forward_expand(seg.graph, seg.output);
        } else {
            const ggml_tensor* fa_out = segs_[static_cast<size_t>(2 * bi + 1)]->output;
            const ggml_tensor* fb_out = segs_[static_cast<size_t>(2 * bi + 1)]->output2;
            const int64_t s_cout = fa_out->ne[0];
            const int64_t t_len = fa_out->ne[1];
            const int64_t cout_n = vae.lookup_derived(wn + "/wa")->ne[1] / s;
            Segment& seg = new_seg(2 * bi + 2, s_cout, t_len);
            seg.input2 = ggml_new_tensor_2d(seg.ctx.raw_context(), GGML_TYPE_F32,
                                            s_cout, t_len);
            ggml_set_input(seg.input2);
            ggml_context* c = seg.ctx.raw_context();
            ggml_tensor* fa3 = ggml_reshape_3d(c, seg.input, cout_n, s, t_len);
            ggml_tensor* fb3 = ggml_reshape_3d(c, seg.input2, cout_n, s, t_len);
            const size_t nb1 = sizeof(float) * cout_n;
            const size_t nb2 = nb1 * s;
            ggml_tensor* d = ggml_add(c,
                ggml_view_3d(c, fa3, cout_n, s, t_len - 1, nb1, nb2, sizeof(float) * cout_n * s),
                ggml_view_3d(c, fb3, cout_n, s, t_len - 1, nb1, nb2, 0));
            ggml_tensor* d2 = ggml_reshape_2d(c, d, cout_n, (t_len - 1) * s);
            ggml_tensor* head = ggml_view_2d(c, fa3, cout_n, s, nb1, 0);
            ggml_tensor* y = ggml_concat(c, head, d2, 1);
            ggml_tensor* y3 = ggml_reshape_3d(c, y, cout_n, t_len * s, 1);
            ggml_tensor* b3 = ggml_reshape_3d(c, vae.lookup_derived(bn + "/f32"),
                                              ggml_nelements(vae.lookup_derived(bn + "/f32")), 1, 1);
            ggml_tensor* x = ggml_add(c, y3, b3);
            const int dilations[3] = {1, 3, 9};
            const ResidualUnitWeights* ress[3] = {&blk.res0, &blk.res1, &blk.res2};
            for (int ri = 0; ri < 3; ++ri) {
                x = vae.residual_unit_forward_l2(c, x, *ress[ri], dilations[ri]);
            }
            x = snake_l2(c, x, vae.lookup_derived("decoder.final.alpha/f32"),
                         vae.lookup_derived("decoder.final.alpha/f32/inv"));
            x = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3));   // 回 L1 [T, C]
            x = vae.causal_conv1d(c, x, vae.weights().decoder_final_conv_weight,
                                  vae.weights().decoder_final_conv_bias, 7, 1, 1, 3);
            x = ggml_tanh(c, x);
            seg.output = x;
            seg.graph = seg.ctx.new_graph(512);
            ggml_build_forward_expand(seg.graph, seg.output);
        }
    }

    // 各段 sched reserve + alloc（执行走 sched_graph_compute，见 Segment 注释）
    // 段输入预分配 CPU buffer：无 buffer 的 input（含其 view/reshape 派生 leaf）
    // 在 sched split 时 backend_id=-1 直接 abort（ggml-alloc buffer_id>=0）。
    // 预分配后归属 CPU 段；HTP 段由 sched 的 split-input 拷贝送入（真机
    // htp-smoke T9 split-copy 已验证的正确通路）。段间搬运本就是 host 中转。
    for (auto& seg : segs_) {
        ggml_backend_buffer_type_t cpu_buft = backend.buffer_type();
        auto alloc_input = [&](ggml_tensor* t, ggml_backend_buffer_t& buf) {
            if (t == nullptr || t->buffer != nullptr) {
                return true;
            }
            buf = ggml_backend_buft_alloc_buffer(cpu_buft, ggml_nbytes(t));
            if (buf == nullptr) {
                return false;
            }
            t->buffer = buf;
            t->data = ggml_backend_buffer_get_base(buf);
            return t->data != nullptr;
        };
        if (!alloc_input(seg->input, seg->input_buffer) ||
            !alloc_input(seg->input2, seg->input2_buffer)) {
            return false;
        }
    }
    for (auto& seg : segs_) {
        // 顺序约束：sched 要求最后一个 backend 为 CPU。HTP 段以 {HTP, CPU}
        // 双后端建 sched（图内 op 权重均在 HTP buffer → 全图归 HTP split）；
        // CPU 段单后端。
        ggml_backend_t bes[2] = {seg->backend, be_cpu};
        const int n_bes = (seg->backend != be_cpu) ? 2 : 1;
        seg->sched = ggml_backend_sched_new(bes, nullptr, n_bes, 512,
                                            /*parallel=*/false, /*op_offload=*/false);
        if (!seg->sched) {
            return false;
        }
        if (!ggml_backend_sched_reserve(seg->sched, seg->graph) ||
            !ggml_backend_sched_alloc_graph(seg->sched, seg->graph)) {
            return false;
        }
    }
    // 段间搬运 scratch：按最大边界一次分配，run 循环中复用（避免每步 malloc）
    size_t max_boundary = 0;
    for (auto& seg : segs_) {
        if (seg->output) {
            max_boundary = std::max(max_boundary, ggml_nbytes(seg->output));
        }
        if (seg->output2) {
            max_boundary = std::max(max_boundary, ggml_nbytes(seg->output2));
        }
    }
    copy_scratch_.resize(max_boundary);
    return true;
}

ggml_status AudioVAEL2Plan::run(VoxCPMBackend& backend) {
    static int64_t prof_htp_us = 0, prof_cpu_us = 0, prof_copy_us = 0;
    static int64_t prof_n = 0;
    static int64_t prof_seg_us[16] = {0};   // 每段累计（VOXCPM_PROFILE_SEGS=1 时打印）
    static const bool prof_segs = std::getenv("VOXCPM_PROFILE_SEGS") != nullptr;
    const int64_t t_run0 = ggml_time_us();
    for (size_t i = 0; i < segs_.size(); ++i) {
        auto& seg = segs_[i];
        const int64_t t0 = ggml_time_us();
        ggml_status st = ggml_backend_sched_graph_compute(seg->sched, seg->graph);
        const int64_t dt = ggml_time_us() - t0;
        if (seg->is_htp) {
            prof_htp_us += dt;
        } else {
            prof_cpu_us += dt;
        }
        if (i < 16) {
            prof_seg_us[i] += dt;
        }
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "[l2plan] seg=%zu failed st=%d\n", i,
                         static_cast<int>(st));
            std::fflush(stderr);
            return st;
        }
        if (i + 1 < segs_.size()) {
            const int64_t tc0 = ggml_time_us();
            const ggml_tensor* out = segs_[i]->output;
            ggml_tensor* next_in = segs_[i + 1]->input;
            const size_t nb = ggml_nbytes(out);
            if (nb > copy_scratch_.size()) {   // build 已按最大边界分配，防御性兜底
                copy_scratch_.resize(nb);
            }
            ggml_backend_tensor_get(out, copy_scratch_.data(), 0, nb);
            ggml_backend_tensor_set(next_in, copy_scratch_.data(), 0, nb);
            if (segs_[i]->output2 != nullptr) {
                const ggml_tensor* out2 = segs_[i]->output2;
                ggml_tensor* next_in2 = segs_[i + 1]->input2;
                const size_t nb2 = ggml_nbytes(out2);
                if (nb2 > copy_scratch_.size()) {
                    copy_scratch_.resize(nb2);
                }
                ggml_backend_tensor_get(out2, copy_scratch_.data(), 0, nb2);
                ggml_backend_tensor_set(next_in2, copy_scratch_.data(), 0, nb2);
            }
            prof_copy_us += ggml_time_us() - tc0;
        }
    }
    prof_n++;
    const int64_t t_total = ggml_time_us() - t_run0;
    static int64_t prof_total_us = 0;
    prof_total_us += t_total;
    if (prof_n % 32 == 0) {
        std::fprintf(stderr,
                     "[l2plan] profile avg over %lld calls: total=%.2fms "
                     "(htp=%.2f cpu=%.2f copy=%.2f, other=%.2f)\n",
                     (long long)prof_n, prof_total_us / 1000.0 / prof_n,
                     prof_htp_us / 1000.0 / prof_n,
                     prof_cpu_us / 1000.0 / prof_n, prof_copy_us / 1000.0 / prof_n,
                     (prof_total_us / prof_n - prof_htp_us / prof_n -
                      prof_cpu_us / prof_n - prof_copy_us / prof_n) / 1000.0);
        if (prof_segs) {
            // seg0/2/4/6/8 CPU（头链/res 组装）、seg1/3/5/7 HTP（convT 双 GEMM）
            std::fprintf(stderr, "[l2plan] segs:");
            for (size_t i = 0; i < segs_.size() && i < 16; ++i) {
                std::fprintf(stderr, " %c%zu=%.2f", segs_[i]->is_htp ? 'H' : 'C',
                             i, prof_seg_us[i] / 1000.0 / prof_n);
            }
            std::fprintf(stderr, " (ms)\n");
            std::memset(prof_seg_us, 0, sizeof(prof_seg_us));
        }
        std::fflush(stderr);
        prof_htp_us = prof_cpu_us = prof_copy_us = 0;
        prof_total_us = 0;
        prof_n = 0;
    }
    return GGML_STATUS_SUCCESS;
}

ggml_tensor* AudioVAEL2Plan::latent_input() {
    return segs_.empty() ? nullptr : segs_.front()->input;
}

ggml_tensor* AudioVAEL2Plan::audio_output() {
    return segs_.empty() ? nullptr : segs_.back()->output;
}

}  // namespace voxcpm
