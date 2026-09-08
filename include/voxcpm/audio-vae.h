#ifndef VOXCPM_AUDIO_VAE_H
#define VOXCPM_AUDIO_VAE_H

#include "voxcpm/common.h"
#include "voxcpm/config.h"
#include "voxcpm/context.h"

#include <map>
#include <memory>
#include <string>
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

    /**
     * @brief 构建 decoder 派生权重（幂等，首帧 decode 前调用一次）
     *
     * 派生内容：convT 权重行重排拆半 W_A/W_B（Q4_0 字节级，行序 k 主序）、
     * bias/alpha 的 F32 化、inv = 1/(alpha+eps)（折叠 snake 除法）。
     * 存放 buffer 按后端模式：异构时建在 HTP buft（USAGE_WEIGHTS），纯 CPU 时
     * 建在 CPU buft——两种模式共用同一张 L2 decoder 图。
     */
    bool ensure_derived_weights(VoxCPMBackend& backend);
    ggml_tensor* lookup_derived(const std::string& key) const;

    // ---- Stage 2d：拆段解码计划（声明见下方 AudioVAEL2Plan）----
    struct SegmentCtx;
    friend class AudioVAEL2Plan;
    void build_l2_plan_segments(class AudioVAEL2Plan& plan, int64_t t_len) const;

    /**
     * @brief 确保各 convT 的输入搬运张量存在（幂等；T 变化时重建）
     *
     * HTP 模式下 convT 在 HTP 执行，其 CPU 侧上游的激活经 ggml_backend_sched
     * 的跨设备拷贝不可靠（dspqueue 缓存语义，Stage 2c 实测）。绕过方式：
     * convT 改读持久 HTP 张量 XT_htp[bi]；每步第一遍 compute 完成后，由调用方
     * 将 CPU 段算出的 XT（last_convt_inputs_[bi]，CPU buffer）用 tensor_set
     * 拷入，再跑第二遍 compute 即得正确结果。
     *
     * @return 需要 CPU→HTP 搬运的 convT 输入张量列表（HTP 模式非空）
     */
    const std::vector<ggml_tensor*>& ensure_convt_inputs(VoxCPMBackend& backend, int64_t t_len);
    const std::vector<ggml_tensor*>& last_convt_inputs() const { return convt_inputs_; }

    const AudioVAEConfig& config() const { return config_; }
    const AudioVAEWeights& weights() const { return weights_; }
    ggml_tensor* last_input_tensor() const { return last_input_tensor_; }
    ggml_tensor* last_decode_sr_cond_tensor() const { return last_decode_sr_cond_tensor_; }
    int32_t last_decode_sr_bucket() const { return last_decode_sr_bucket_; }
    const std::vector<float>& last_preprocessed_audio() const { return last_preprocessed_audio_; }
    const void* shared_store_token() const { return shared_store_.get(); }
    bool uses_shared_weights() const { return shared_store_ != nullptr; }

private:
    // ---- Stage 2c：decoder L2 布局图（[C, T, 1]，通道在 ne[0]）专用前向 ----
    ggml_tensor* decode_tensor_l2(ggml_context* ctx, ggml_tensor* z, ggml_tensor* sr_bucket) const;
    ggml_tensor* causal_transpose_conv1d_l2(ggml_context* ctx, ggml_tensor* x,
                                            ggml_tensor* wa, ggml_tensor* wb,
                                            ggml_tensor* bias_f32, int stride) const;
    ggml_tensor* decoder_block_forward_l2(ggml_context* ctx,
                                          ggml_tensor* x,
                                          const DecoderBlockWeights& weights,
                                          ggml_tensor* sr_bucket,
                                          int stride) const;
    ggml_tensor* sample_rate_condition_forward_l2(ggml_context* ctx,
                                                  ggml_tensor* x,
                                                  const DecoderBlockWeights::SampleRateConditionWeights& weights,
                                                  ggml_tensor* sr_bucket) const;
    ggml_tensor* residual_unit_forward_l2(ggml_context* ctx,
                                          ggml_tensor* x,
                                          const ResidualUnitWeights& weights,
                                          int dilation) const;

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

    bool load_encoder_weights(const VoxCPMWeightStore& store) ;
    bool load_decoder_weights(const VoxCPMWeightStore& store) ;

    AudioVAEConfig config_;
    AudioVAEWeights weights_;

    ggml_context* weight_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;

    // Stage 2c 派生权重（decoder L2 图专用）。W_A/W_B（GEMM 大头）单独一个
    // ctx/buffer，可落 HTP；bias/alpha/inv 体积小、与 CPU 侧 snake/dw 同侧消费，
    // 恒留 CPU，避免 CPU/HTP 交叉过密。
    struct Derived {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        std::map<std::string, ggml_tensor*> tensors;
        bool ready = false;
    };
    Derived derived_;
    Derived derived_cpu_;

    // convT 输入搬运张量（HTP 模式专用，[Cin, T] per block）
    std::vector<ggml_tensor*> convt_inputs_;
    ggml_context* convt_input_ctx_ = nullptr;
    ggml_backend_buffer_t convt_input_buffer_ = nullptr;
    int64_t convt_input_t_ = -1;

    ggml_tensor* last_input_tensor_ = nullptr;
    ggml_tensor* last_decode_sr_cond_tensor_ = nullptr;
    int32_t last_decode_sr_bucket_ = 0;
    std::vector<float> last_preprocessed_audio_;
    mutable std::vector<std::unique_ptr<AudioVAEDepthwiseConvOpData>> depthwise_ops_;
    std::shared_ptr<VoxCPMWeightStore> shared_store_;
};

/**
 * @brief Stage 2d：HTP/CPU 拆段解码计划
 *
 * 把 L2 decoder 拆成 9 张交替的单后端图（5 CPU + 4 HTP），段间激活经
 * tensor_get/tensor_set 显式搬运（绕开 ggml_backend_sched 跨设备拷贝在
 * dspqueue 后端上的失效问题）。图一次构建、多步复用（uid 稳定 → HTP
 * opbatch 编译缓存命中）。
 *
 *   seg0(CPU): latent → dw0 → pw1 → block2 头链   → XT2
 *   seg1(HTP): XT2 → convT2(+bias)                → Y2
 *   seg2(CPU): Y2 → block2 res×3 → block3 头链     → XT3
 *   seg3(HTP): XT3 → convT3                       → Y3
 *   seg4(CPU): Y3 → block3 res×3 → block4 头链     → XT4
 *   seg5(HTP): XT4 → convT4                       → Y4
 *   seg6(CPU): Y4 → block4 res×3 → block5 头链     → XT5
 *   seg7(HTP): XT5 → convT5                       → Y5
 *   seg8(CPU): Y5 → block5 res×3 → final snake →
 *              final conv k7 → tanh               → audio
 */
class AudioVAEL2Plan {
public:
    AudioVAEL2Plan();
    ~AudioVAEL2Plan();
    AudioVAEL2Plan(const AudioVAEL2Plan&) = delete;
    AudioVAEL2Plan& operator=(const AudioVAEL2Plan&) = delete;

    /**
     * @brief 一次性组建全部段（幂等；t_len 变化时重建）
     * @param t_len latent 帧数（每步 decode 的 z 帧数，需全程一致）
     */
    bool build(AudioVAE& vae, VoxCPMBackend& backend, int64_t t_len);

    /** @brief 段输入张量（latent，[t_len, C] L1；每步先 tensor_set 灌入） */
    ggml_tensor* latent_input();
    /** @brief 段输出张量（audio，[T_out, 1, 1]；run 后 tensor_get 取波形） */
    ggml_tensor* audio_output();

    /** @brief 执行一次完整解码（含段间搬运） */
    ggml_status run(VoxCPMBackend& backend);

private:
    struct Segment;
    std::vector<std::unique_ptr<Segment>> segs_;
    void* impl_ = nullptr;   // 段构建上下文（AudioVAE 内部使用）
};

}  // namespace voxcpm

#endif  // VOXCPM_AUDIO_VAE_H
