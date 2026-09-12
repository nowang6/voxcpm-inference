#ifndef VOXCPM_STATE_H
#define VOXCPM_STATE_H

#include "voxcpm/backend.h"
#include "voxcpm/common.h"

namespace voxcpm {

struct PersistentStateShape {
    int hidden_size = 0;
    int feat_dim = 0;
    int patch_size = 0;
};

class VoxCPMPersistentState {
public:
    VoxCPMPersistentState() = default;
    ~VoxCPMPersistentState();

    VoxCPMPersistentState(const VoxCPMPersistentState&) = delete;
    VoxCPMPersistentState& operator=(const VoxCPMPersistentState&) = delete;

    VoxCPMPersistentState(VoxCPMPersistentState&& other) noexcept;
    VoxCPMPersistentState& operator=(VoxCPMPersistentState&& other) noexcept;

    bool initialize(VoxCPMBackend& backend, const PersistentStateShape& shape, ggml_type type = GGML_TYPE_F32);
    void reset();

    bool is_initialized() const { return ctx_ != nullptr && buffer_ != nullptr; }
    const PersistentStateShape& shape() const { return shape_; }

    ggml_context* context() const { return ctx_; }
    ggml_backend_buffer_t buffer() const { return buffer_; }

    ggml_tensor* lm_hidden() const { return lm_hidden_; }
    ggml_tensor* residual_hidden() const { return residual_hidden_; }
    ggml_tensor* prefix_patch() const { return prefix_patch_; }

    bool set_lm_hidden_from_host(VoxCPMBackend& backend, const float* data, size_t count);
    bool set_residual_hidden_from_host(VoxCPMBackend& backend, const float* data, size_t count);
    bool set_prefix_patch_from_host(VoxCPMBackend& backend, const float* data, size_t count);

    bool get_lm_hidden_to_host(VoxCPMBackend& backend, float* data, size_t count) const;
    bool get_residual_hidden_to_host(VoxCPMBackend& backend, float* data, size_t count) const;
    bool get_prefix_patch_to_host(VoxCPMBackend& backend, float* data, size_t count) const;

private:
    PersistentStateShape shape_;
    VoxCPMBackend* backend_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor* lm_hidden_ = nullptr;
    ggml_tensor* residual_hidden_ = nullptr;
    ggml_tensor* prefix_patch_ = nullptr;
};

}  // namespace voxcpm

#endif  // VOXCPM_STATE_H
