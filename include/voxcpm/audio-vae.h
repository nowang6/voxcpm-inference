#ifndef VOXCPM_AUDIO_VAE_H
#define VOXCPM_AUDIO_VAE_H

#include "voxcpm/common.h"
#include "voxcpm/config.h"
#include "voxcpm/context.h"

#include <memory>
#include <vector>

namespace voxcpm {

class VoxCPMBackend;
class VoxCPMWeightStore;
struct AudioVAEDepthwiseConvOpData;

ggml_tensor* snake_activation(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha, float eps = 1e-9f);

struct ResidualUnitWeights {
    ggml_tensor* snake1_alpha = nullptr;
    ggml_tensor* conv1_weight = nullptr;
    ggml_tensor* conv1_bias = nullptr;
    ggml_tensor* snake2_alpha = nullptr;
    ggml_tensor* conv2_weight = nullptr;
    ggml_tensor* conv2_bias = nullptr;
};

struct EncoderBlockWeights {
    ResidualUnitWeights res0;
    ResidualUnitWeights res1;
    ResidualUnitWeights res2;
    ggml_tensor* snake_alpha = nullptr;
    ggml_tensor* conv_weight = nullptr;
    ggml_tensor* conv_bias = nullptr;
};

struct DecoderBlockWeights {
    struct SampleRateConditionWeights {
        ggml_tensor* scale_embed = nullptr;
        ggml_tensor* bias_embed = nullptr;
        ggml_tensor* cond_embed = nullptr;
        ggml_tensor* out_snake_alpha = nullptr;
        ggml_tensor* out_weight = nullptr;
        ggml_tensor* out_bias = nullptr;

        bool active() const {
            return scale_embed != nullptr || bias_embed != nullptr || cond_embed != nullptr;
        }
    };

    SampleRateConditionWeights sr_cond;
    ggml_tensor* snake_alpha = nullptr;
    ggml_tensor* conv_weight = nullptr;
    ggml_tensor* conv_bias = nullptr;
    ResidualUnitWeights res0;
    ResidualUnitWeights res1;
    ResidualUnitWeights res2;
};

struct AudioVAEWeights {
    ggml_tensor* encoder_block_0_weight = nullptr;
    ggml_tensor* encoder_block_0_bias = nullptr;
    std::vector<EncoderBlockWeights> encoder_blocks;
    ggml_tensor* encoder_fc_mu_weight = nullptr;
    ggml_tensor* encoder_fc_mu_bias = nullptr;

    ggml_tensor* decoder_model_0_weight = nullptr;
    ggml_tensor* decoder_model_0_bias = nullptr;
    ggml_tensor* decoder_model_1_weight = nullptr;
    ggml_tensor* decoder_model_1_bias = nullptr;
    std::vector<DecoderBlockWeights> decoder_blocks;
    ggml_tensor* decoder_final_snake_alpha = nullptr;
    ggml_tensor* decoder_final_conv_weight = nullptr;
    ggml_tensor* decoder_final_conv_bias = nullptr;
};

class AudioVAE {
public:
    explicit AudioVAE(const AudioVAEConfig& config = AudioVAEConfig());
    ~AudioVAE();

    AudioVAE(const AudioVAE&) = delete;
    AudioVAE& operator=(const AudioVAE&) = delete;

    bool load_from_store(const std::shared_ptr<VoxCPMWeightStore>& store);

    std::vector<float> preprocess(std::vector<float> audio_data, int sample_rate = -1) const;

    ggml_tensor* encode(VoxCPMContext& ctx,
                        std::vector<float>& audio_data,
                        int sample_rate = -1);
    ggml_tensor* decode(VoxCPMContext& ctx, ggml_tensor* z);
    void prepare_decode_inputs(VoxCPMBackend& backend) const;

    const AudioVAEConfig& config() const { return config_; }
    const AudioVAEWeights& weights() const { return weights_; }
    ggml_tensor* last_input_tensor() const { return last_input_tensor_; }
    ggml_tensor* last_decode_sr_cond_tensor() const { return last_decode_sr_cond_tensor_; }
    int32_t last_decode_sr_bucket() const { return last_decode_sr_bucket_; }
    const std::vector<float>& last_preprocessed_audio() const { return last_preprocessed_audio_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    ggml_tensor* causal_conv1d(ggml_context* ctx,
                               ggml_tensor* x,
                               ggml_tensor* weight,
                               ggml_tensor* bias,
                               int kernel_size,
                               int stride,
                               int dilation,
                               int padding) const;

    ggml_tensor* causal_conv1d_dw(ggml_context* ctx,
                                  ggml_tensor* x,
                                  ggml_tensor* weight,
                                  ggml_tensor* bias,
                                  int stride,
                                  int dilation,
                                  int padding) const;

    ggml_tensor* causal_transpose_conv1d(ggml_context* ctx,
                                         ggml_tensor* x,
                                         ggml_tensor* weight,
                                         ggml_tensor* bias,
                                         int expected_kernel,
                                         int stride,
                                         int padding,
                                         int output_padding) const;

    ggml_tensor* residual_unit_forward(ggml_context* ctx,
                                       ggml_tensor* x,
                                       const ResidualUnitWeights& weights,
                                       int dilation) const;

    ggml_tensor* encoder_block_forward(ggml_context* ctx,
                                       ggml_tensor* x,
                                       const EncoderBlockWeights& weights,
                                       int stride) const;

    ggml_tensor* decoder_block_forward(ggml_context* ctx,
                                       ggml_tensor* x,
                                       const DecoderBlockWeights& weights,
                                       ggml_tensor* sr_bucket,
                                       int stride) const;
    ggml_tensor* sample_rate_condition_forward(ggml_context* ctx,
                                               ggml_tensor* x,
                                               const DecoderBlockWeights::SampleRateConditionWeights& weights,
                                               ggml_tensor* sr_bucket) const;

    ggml_tensor* encode_tensor(VoxCPMContext& ctx, ggml_tensor* audio) const;

    bool load_encoder_weights(ggml_context* ggml_ctx_ptr) ;
    bool load_decoder_weights(ggml_context* ggml_ctx_ptr) ;

    AudioVAEConfig config_;
    AudioVAEWeights weights_;

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    ggml_tensor* last_input_tensor_ = nullptr;
    ggml_tensor* last_decode_sr_cond_tensor_ = nullptr;
    int32_t last_decode_sr_bucket_ = 0;
    std::vector<float> last_preprocessed_audio_;
    mutable std::vector<std::unique_ptr<AudioVAEDepthwiseConvOpData>> depthwise_ops_;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

}  // namespace voxcpm

#endif  // VOXCPM_AUDIO_VAE_H
