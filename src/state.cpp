#include "voxcpm/state.h"

namespace voxcpm {

namespace {

ggml_context* make_state_context() {
    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    return ggml_init(params);
}

bool tensor_matches_count(const ggml_tensor* tensor, size_t count) {
    return tensor != nullptr && ggml_nelements(tensor) == static_cast<int64_t>(count);
}

}  // namespace

VoxCPMPersistentState::~VoxCPMPersistentState() {
    reset();
}

VoxCPMPersistentState::VoxCPMPersistentState(VoxCPMPersistentState&& other) noexcept
    : shape_(other.shape_),
      backend_(other.backend_),
      ctx_(other.ctx_),
      buffer_(other.buffer_),
      lm_hidden_(other.lm_hidden_),
      residual_hidden_(other.residual_hidden_),
      prefix_patch_(other.prefix_patch_) {
    other.ctx_ = nullptr;
    other.backend_ = nullptr;
    other.buffer_ = nullptr;
    other.lm_hidden_ = nullptr;
    other.residual_hidden_ = nullptr;
    other.prefix_patch_ = nullptr;
    other.shape_ = {};
}

VoxCPMPersistentState& VoxCPMPersistentState::operator=(VoxCPMPersistentState&& other) noexcept {
    if (this != &other) {
        reset();
        shape_ = other.shape_;
        backend_ = other.backend_;
        ctx_ = other.ctx_;
        buffer_ = other.buffer_;
        lm_hidden_ = other.lm_hidden_;
        residual_hidden_ = other.residual_hidden_;
        prefix_patch_ = other.prefix_patch_;

        other.ctx_ = nullptr;
        other.backend_ = nullptr;
        other.buffer_ = nullptr;
        other.lm_hidden_ = nullptr;
        other.residual_hidden_ = nullptr;
        other.prefix_patch_ = nullptr;
        other.shape_ = {};
    }
    return *this;
}

bool VoxCPMPersistentState::initialize(VoxCPMBackend& backend, const PersistentStateShape& shape, ggml_type type) {
    reset();

    if (shape.hidden_size <= 0 || shape.feat_dim <= 0 || shape.patch_size <= 0) {
        return false;
    }

    ctx_ = make_state_context();
    if (!ctx_) {
        return false;
    }

    shape_ = shape;
    backend_ = &backend;
    lm_hidden_ = ggml_new_tensor_1d(ctx_, type, shape.hidden_size);
    residual_hidden_ = ggml_new_tensor_1d(ctx_, type, shape.hidden_size);
    prefix_patch_ = ggml_new_tensor_2d(ctx_, type, shape.feat_dim, shape.patch_size);

    if (!lm_hidden_ || !residual_hidden_ || !prefix_patch_) {
        reset();
        return false;
    }

    ggml_set_name(lm_hidden_, "state.lm_hidden");
    ggml_set_name(residual_hidden_, "state.residual_hidden");
    ggml_set_name(prefix_patch_, "state.prefix_patch");

    buffer_ = backend.alloc_buffer(ctx_, BufferUsage::State);
    if (!buffer_) {
        reset();
        return false;
    }

    ggml_backend_buffer_clear(buffer_, 0);

    return true;
}

void VoxCPMPersistentState::reset() {
    if (buffer_) {
        if (backend_) {
            backend_->free_buffer(buffer_);
        } else {
            ggml_backend_buffer_free(buffer_);
        }
        buffer_ = nullptr;
    }
    if (ctx_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
    lm_hidden_ = nullptr;
    residual_hidden_ = nullptr;
    prefix_patch_ = nullptr;
    backend_ = nullptr;
    shape_ = {};
}

bool VoxCPMPersistentState::set_lm_hidden_from_host(VoxCPMBackend& backend, const float* data, size_t count) {
    if (!tensor_matches_count(lm_hidden_, count) || data == nullptr) {
        return false;
    }
    backend.tensor_set(lm_hidden_, data, 0, count * sizeof(float));
    return true;
}

bool VoxCPMPersistentState::set_residual_hidden_from_host(VoxCPMBackend& backend, const float* data, size_t count) {
    if (!tensor_matches_count(residual_hidden_, count) || data == nullptr) {
        return false;
    }
    backend.tensor_set(residual_hidden_, data, 0, count * sizeof(float));
    return true;
}

bool VoxCPMPersistentState::set_prefix_patch_from_host(VoxCPMBackend& backend, const float* data, size_t count) {
    if (!tensor_matches_count(prefix_patch_, count) || data == nullptr) {
        return false;
    }
    backend.tensor_set(prefix_patch_, data, 0, count * sizeof(float));
    return true;
}

bool VoxCPMPersistentState::get_lm_hidden_to_host(VoxCPMBackend& backend, float* data, size_t count) const {
    if (!tensor_matches_count(lm_hidden_, count) || data == nullptr) {
        return false;
    }
    backend.tensor_get(lm_hidden_, data, 0, count * sizeof(float));
    return true;
}

bool VoxCPMPersistentState::get_residual_hidden_to_host(VoxCPMBackend& backend, float* data, size_t count) const {
    if (!tensor_matches_count(residual_hidden_, count) || data == nullptr) {
        return false;
    }
    backend.tensor_get(residual_hidden_, data, 0, count * sizeof(float));
    return true;
}

bool VoxCPMPersistentState::get_prefix_patch_to_host(VoxCPMBackend& backend, float* data, size_t count) const {
    if (!tensor_matches_count(prefix_patch_, count) || data == nullptr) {
        return false;
    }
    backend.tensor_get(prefix_patch_, data, 0, count * sizeof(float));
    return true;
}

}  // namespace voxcpm
