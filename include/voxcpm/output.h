#ifndef VOXCPM_OUTPUT_H
#define VOXCPM_OUTPUT_H

#include "voxcpm/backend.h"
#include "voxcpm/common.h"

#include <array>
#include <vector>

namespace voxcpm {

struct OutputPoolShape {
    int feat_dim = 0;
    int patch_size = 0;
    int max_latent_patches = 0;
};

struct DecodeOutputView {
    ggml_tensor* patch = nullptr;
    ggml_tensor* stop_logits = nullptr;
    ggml_tensor* latent_seq = nullptr;
};

struct HostDecodeOutput {
    std::vector<float> patch;
    std::array<float, 2> stop_logits = {0.0f, 0.0f};
};

class VoxCPMOutputPool {
public:
    VoxCPMOutputPool() = default;
    ~VoxCPMOutputPool();

    VoxCPMOutputPool(const VoxCPMOutputPool&) = delete;
    VoxCPMOutputPool& operator=(const VoxCPMOutputPool&) = delete;

    VoxCPMOutputPool(VoxCPMOutputPool&& other) noexcept;
    VoxCPMOutputPool& operator=(VoxCPMOutputPool&& other) noexcept;

    bool initialize(VoxCPMBackend& backend, const OutputPoolShape& shape, ggml_type type = GGML_TYPE_F32);
    void reset();

    bool is_initialized() const { return ctx_ != nullptr && buffer_ != nullptr; }
    const OutputPoolShape& shape() const { return shape_; }
    bool has_patch_output() const { return has_patch_output_; }
    bool has_stop_logits() const { return has_stop_logits_; }

    ggml_context* context() const { return ctx_; }
    ggml_backend_buffer_t buffer() const { return buffer_; }

    ggml_tensor* patch_output() const { return patch_output_; }
    ggml_tensor* stop_logits() const { return stop_logits_; }
    ggml_tensor* latent_seq() const { return latent_seq_; }
    ggml_tensor* latent_patch_view(int patch_index) const;

    void publish_patch_output(VoxCPMBackend& backend, ggml_tensor* patch_src);
    void publish_stop_logits(VoxCPMBackend& backend, ggml_tensor* stop_src);
    void publish_latent_seq(VoxCPMBackend& backend, ggml_tensor* latent_src);
    bool publish_latent_seq_prefix(VoxCPMBackend& backend, const VoxCPMOutputPool& src_pool, int frame_count);
    bool publish_patch_to_latent_seq(VoxCPMBackend& backend, ggml_tensor* patch_src, int patch_index);
    bool publish_patch_range_to_latent_seq(VoxCPMBackend& backend,
                                           ggml_tensor* patch_range_src,
                                           int first_patch_index,
                                           int patch_count);
    void publish_decode_outputs(VoxCPMBackend& backend, ggml_tensor* patch_src, ggml_tensor* stop_src);
    bool publish_patch_output_from_host(VoxCPMBackend& backend,
                                        const float* patch_data,
                                        size_t patch_count);
    bool publish_stop_logits_from_host(VoxCPMBackend& backend,
                                       const float* stop_data,
                                       size_t stop_count);
    bool publish_decode_outputs_from_host(VoxCPMBackend& backend,
                                          const float* patch_data,
                                          size_t patch_count,
                                          const float* stop_data,
                                          size_t stop_count);
    bool write_patch_range_to_latent_seq_from_host(VoxCPMBackend& backend,
                                                   const float* patch_data,
                                                   int first_patch_index,
                                                   int patch_count);
    bool write_patch_to_latent_seq_from_host(VoxCPMBackend& backend,
                                             const float* patch_data,
                                             size_t patch_count,
                                             int patch_index);
    DecodeOutputView view_decode_outputs() const;
    std::vector<float> export_patch_to_host(VoxCPMBackend& backend) const;
    std::array<float, 2> export_stop_logits_to_host(VoxCPMBackend& backend) const;
    std::vector<float> export_latent_seq_to_host(VoxCPMBackend& backend, int patch_count) const;
    std::vector<float> export_latent_seq_range_to_host(VoxCPMBackend& backend,
                                                       int first_patch_index,
                                                       int patch_count) const;
    ggml_tensor* make_audio_vae_latent_view(ggml_context* ctx,
                                            int frame_offset,
                                            int frame_count) const;
    std::vector<float> export_audio_vae_latent_to_host(VoxCPMBackend& backend,
                                                       int frame_offset,
                                                       int frame_count) const;
    HostDecodeOutput export_decode_output_to_host(VoxCPMBackend& backend) const;

private:
    OutputPoolShape shape_;
    VoxCPMBackend* backend_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor* patch_output_ = nullptr;
    ggml_tensor* stop_logits_ = nullptr;
    ggml_tensor* latent_seq_ = nullptr;
    std::vector<ggml_tensor*> latent_patch_views_;
    bool has_patch_output_ = false;
    bool has_stop_logits_ = false;
};

}  // namespace voxcpm

#endif  // VOXCPM_OUTPUT_H
