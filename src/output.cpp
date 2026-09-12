#include "voxcpm/output.h"
#include "voxcpm/context.h"

namespace voxcpm {

namespace {

ggml_context* make_output_context(const OutputPoolShape& shape) {
    const size_t tensor_count = static_cast<size_t>(3 + std::max(0, shape.max_latent_patches));
    const size_t metadata_bytes =
        ggml_tensor_overhead() * tensor_count +
        256 * 1024;  // headroom for view tensors, alignment, and future small metadata growth
    ggml_init_params params = {};
    params.mem_size = std::max<size_t>(4 * 1024 * 1024, metadata_bytes);
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    return ggml_init(params);
}

bool tensor_matches_count(const ggml_tensor* tensor, size_t count) {
    return tensor != nullptr && ggml_nelements(tensor) == static_cast<int64_t>(count);
}

size_t patch_element_count(const OutputPoolShape& shape) {
    return static_cast<size_t>(shape.feat_dim * shape.patch_size);
}

ggml_tensor* make_latent_seq_frame_view(ggml_context* ctx,
                                        ggml_tensor* latent_seq,
                                        const OutputPoolShape& shape,
                                        int frame_offset,
                                        int frame_count) {
    if (ctx == nullptr || latent_seq == nullptr || frame_count <= 0 || frame_offset < 0) {
        return nullptr;
    }
    if (frame_offset + frame_count > shape.max_latent_patches) {
        return nullptr;
    }

    const int total_patches = frame_count * shape.patch_size;
    const int patch_offset = frame_offset * shape.patch_size;
    return ggml_view_2d(ctx,
                        latent_seq,
                        shape.feat_dim,
                        total_patches,
                        latent_seq->nb[1],
                        static_cast<size_t>(patch_offset) * latent_seq->nb[1]);
}

}  // namespace

VoxCPMOutputPool::~VoxCPMOutputPool() {
    reset();
}

VoxCPMOutputPool::VoxCPMOutputPool(VoxCPMOutputPool&& other) noexcept
    : shape_(other.shape_),
      backend_(other.backend_),
      ctx_(other.ctx_),
      buffer_(other.buffer_),
      patch_output_(other.patch_output_),
      stop_logits_(other.stop_logits_),
      latent_seq_(other.latent_seq_),
      latent_patch_views_(std::move(other.latent_patch_views_)),
      has_patch_output_(other.has_patch_output_),
      has_stop_logits_(other.has_stop_logits_) {
    other.ctx_ = nullptr;
    other.backend_ = nullptr;
    other.buffer_ = nullptr;
    other.patch_output_ = nullptr;
    other.stop_logits_ = nullptr;
    other.latent_seq_ = nullptr;
    other.shape_ = {};
    other.has_patch_output_ = false;
    other.has_stop_logits_ = false;
}

VoxCPMOutputPool& VoxCPMOutputPool::operator=(VoxCPMOutputPool&& other) noexcept {
    if (this != &other) {
        reset();
        shape_ = other.shape_;
        backend_ = other.backend_;
        ctx_ = other.ctx_;
        buffer_ = other.buffer_;
        patch_output_ = other.patch_output_;
        stop_logits_ = other.stop_logits_;
        latent_seq_ = other.latent_seq_;
        latent_patch_views_ = std::move(other.latent_patch_views_);
        has_patch_output_ = other.has_patch_output_;
        has_stop_logits_ = other.has_stop_logits_;

        other.ctx_ = nullptr;
        other.backend_ = nullptr;
        other.buffer_ = nullptr;
        other.patch_output_ = nullptr;
        other.stop_logits_ = nullptr;
        other.latent_seq_ = nullptr;
        other.latent_patch_views_.clear();
        other.shape_ = {};
        other.has_patch_output_ = false;
        other.has_stop_logits_ = false;
    }
    return *this;
}

bool VoxCPMOutputPool::initialize(VoxCPMBackend& backend, const OutputPoolShape& shape, ggml_type type) {
    reset();

    if (shape.feat_dim <= 0 || shape.patch_size <= 0 || shape.max_latent_patches <= 0) {
        return false;
    }

    ctx_ = make_output_context(shape);
    if (!ctx_) {
        return false;
    }

    shape_ = shape;
    backend_ = &backend;
    has_patch_output_ = false;
    has_stop_logits_ = false;
    patch_output_ = ggml_new_tensor_2d(ctx_, type, shape.feat_dim, shape.patch_size);
    stop_logits_ = ggml_new_tensor_1d(ctx_, type, 2);
    latent_seq_ = ggml_new_tensor_2d(ctx_, type, shape.feat_dim, shape.max_latent_patches * shape.patch_size);

    if (!patch_output_ || !stop_logits_ || !latent_seq_) {
        reset();
        return false;
    }

    ggml_set_name(patch_output_, "output.patch");
    ggml_set_name(stop_logits_, "output.stop_logits");
    ggml_set_name(latent_seq_, "output.latent_seq");

    latent_patch_views_.clear();
    latent_patch_views_.reserve(static_cast<size_t>(shape.max_latent_patches));
    for (int patch = 0; patch < shape.max_latent_patches; ++patch) {
        ggml_tensor* patch_view = ggml_view_2d(ctx_,
                                               latent_seq_,
                                               shape.feat_dim,
                                               shape.patch_size,
                                               latent_seq_->nb[1],
                                               static_cast<size_t>(patch) * shape.patch_size * latent_seq_->nb[1]);
        if (!patch_view) {
            reset();
            return false;
        }
        ggml_set_name(patch_view, "output.latent_seq.patch_view");
        latent_patch_views_.push_back(patch_view);
    }

    buffer_ = backend.alloc_buffer(ctx_, BufferUsage::Output);
    if (!buffer_) {
        reset();
        return false;
    }

    ggml_backend_buffer_clear(buffer_, 0);

    return true;
}

void VoxCPMOutputPool::reset() {
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
    patch_output_ = nullptr;
    stop_logits_ = nullptr;
    latent_seq_ = nullptr;
    latent_patch_views_.clear();
    backend_ = nullptr;
    shape_ = {};
    has_patch_output_ = false;
    has_stop_logits_ = false;
}

ggml_tensor* VoxCPMOutputPool::latent_patch_view(int patch_index) const {
    if (patch_index < 0 || patch_index >= static_cast<int>(latent_patch_views_.size())) {
        return nullptr;
    }
    return latent_patch_views_[static_cast<size_t>(patch_index)];
}

void VoxCPMOutputPool::publish_patch_output(VoxCPMBackend& backend, ggml_tensor* patch_src) {
    if (patch_src && patch_output_) {
        backend.tensor_copy(patch_src, patch_output_);
        has_patch_output_ = true;
    }
}

void VoxCPMOutputPool::publish_stop_logits(VoxCPMBackend& backend, ggml_tensor* stop_src) {
    if (stop_src && stop_logits_) {
        backend.tensor_copy(stop_src, stop_logits_);
        has_stop_logits_ = true;
    }
}

void VoxCPMOutputPool::publish_latent_seq(VoxCPMBackend& backend, ggml_tensor* latent_src) {
    if (latent_src && latent_seq_) {
        backend.tensor_copy(latent_src, latent_seq_);
    }
}

bool VoxCPMOutputPool::publish_latent_seq_prefix(VoxCPMBackend& backend,
                                                 const VoxCPMOutputPool& src_pool,
                                                 int frame_count) {
    if (latent_seq_ == nullptr || src_pool.latent_seq_ == nullptr || frame_count <= 0 || frame_count > shape_.max_latent_patches) {
        return false;
    }
    if (shape_.feat_dim != src_pool.shape_.feat_dim ||
        shape_.patch_size != src_pool.shape_.patch_size ||
        frame_count > src_pool.shape_.max_latent_patches) {
        return false;
    }
    for (int frame = 0; frame < frame_count; ++frame) {
        ggml_tensor* src_view = src_pool.latent_patch_views_[static_cast<size_t>(frame)];
        ggml_tensor* dst_view = latent_patch_views_[static_cast<size_t>(frame)];
        if (src_view == nullptr || dst_view == nullptr || ggml_nbytes(src_view) != ggml_nbytes(dst_view)) {
            return false;
        }
        backend.tensor_copy(src_view, dst_view);
    }
    return true;
}

bool VoxCPMOutputPool::publish_patch_to_latent_seq(VoxCPMBackend& backend, ggml_tensor* patch_src, int patch_index) {
    if (patch_src == nullptr || patch_index < 0 || patch_index >= shape_.max_latent_patches) {
        return false;
    }
    if (latent_patch_views_.empty()) {
        return false;
    }
    ggml_tensor* patch_view = latent_patch_views_[static_cast<size_t>(patch_index)];
    if (patch_view == nullptr || ggml_nbytes(patch_src) != ggml_nbytes(patch_view)) {
        return false;
    }

    backend.tensor_copy(patch_src, patch_view);
    return true;
}

bool VoxCPMOutputPool::publish_patch_range_to_latent_seq(VoxCPMBackend& backend,
                                                         ggml_tensor* patch_range_src,
                                                         int first_patch_index,
                                                         int patch_count) {
    if (patch_range_src == nullptr || latent_seq_ == nullptr) {
        return false;
    }
    if (first_patch_index < 0 || patch_count <= 0) {
        return false;
    }
    if (first_patch_index + patch_count > shape_.max_latent_patches) {
        return false;
    }

    VoxCPMContext view_ctx(ContextType::Graph, 4, 4);
    ggml_tensor* dst_view =
        make_latent_seq_frame_view(view_ctx.raw_context(), latent_seq_, shape_, first_patch_index, patch_count);
    if (dst_view == nullptr) {
        return false;
    }
    if (ggml_nbytes(dst_view) != ggml_nbytes(patch_range_src)) {
        return false;
    }

    backend.tensor_copy(patch_range_src, dst_view);
    return true;
}

void VoxCPMOutputPool::publish_decode_outputs(VoxCPMBackend& backend, ggml_tensor* patch_src, ggml_tensor* stop_src) {
    publish_patch_output(backend, patch_src);
    publish_stop_logits(backend, stop_src);
}

bool VoxCPMOutputPool::publish_patch_output_from_host(VoxCPMBackend& backend,
                                                      const float* patch_data,
                                                      size_t patch_count) {
    if (!tensor_matches_count(patch_output_, patch_count) || patch_data == nullptr) {
        return false;
    }

    backend.tensor_set(patch_output_, patch_data, 0, patch_count * sizeof(float));
    has_patch_output_ = true;
    return true;
}

bool VoxCPMOutputPool::publish_stop_logits_from_host(VoxCPMBackend& backend,
                                                     const float* stop_data,
                                                     size_t stop_count) {
    if (!tensor_matches_count(stop_logits_, stop_count) || stop_data == nullptr) {
        return false;
    }

    backend.tensor_set(stop_logits_, stop_data, 0, stop_count * sizeof(float));
    has_stop_logits_ = true;
    return true;
}

bool VoxCPMOutputPool::publish_decode_outputs_from_host(VoxCPMBackend& backend,
                                                        const float* patch_data,
                                                        size_t patch_count,
                                                        const float* stop_data,
                                                        size_t stop_count) {
    return publish_patch_output_from_host(backend, patch_data, patch_count) &&
           publish_stop_logits_from_host(backend, stop_data, stop_count);
}

bool VoxCPMOutputPool::write_patch_range_to_latent_seq_from_host(VoxCPMBackend& backend,
                                                                 const float* patch_data,
                                                                 int first_patch_index,
                                                                 int patch_count) {
    if (latent_seq_ == nullptr || patch_data == nullptr) {
        return false;
    }
    if (first_patch_index < 0 || patch_count <= 0) {
        return false;
    }
    if (first_patch_index + patch_count > shape_.max_latent_patches) {
        return false;
    }

    const size_t single_patch_bytes = patch_element_count(shape_) * sizeof(float);
    const size_t total_bytes = static_cast<size_t>(patch_count) * single_patch_bytes;
    const size_t offset = static_cast<size_t>(first_patch_index) * single_patch_bytes;
    backend.tensor_set(latent_seq_, patch_data, offset, total_bytes);
    return true;
}

bool VoxCPMOutputPool::write_patch_to_latent_seq_from_host(VoxCPMBackend& backend,
                                                           const float* patch_data,
                                                           size_t patch_count,
                                                           int patch_index) {
    if (patch_count != patch_element_count(shape_)) {
        return false;
    }
    return write_patch_range_to_latent_seq_from_host(backend, patch_data, patch_index, 1);
}

DecodeOutputView VoxCPMOutputPool::view_decode_outputs() const {
    return DecodeOutputView{patch_output_, stop_logits_, latent_seq_};
}

std::vector<float> VoxCPMOutputPool::export_patch_to_host(VoxCPMBackend& backend) const {
    std::vector<float> patch;
    if (patch_output_) {
        patch.resize(static_cast<size_t>(shape_.feat_dim * shape_.patch_size));
        if (has_patch_output_) {
            backend.tensor_get(patch_output_, patch.data(), 0, patch.size() * sizeof(float));
        }
    }
    return patch;
}

std::array<float, 2> VoxCPMOutputPool::export_stop_logits_to_host(VoxCPMBackend& backend) const {
    std::array<float, 2> stop = {0.0f, 0.0f};
    if (stop_logits_ && has_stop_logits_) {
        backend.tensor_get(stop_logits_, stop.data(), 0, stop.size() * sizeof(float));
    }
    return stop;
}

std::vector<float> VoxCPMOutputPool::export_latent_seq_to_host(VoxCPMBackend& backend, int patch_count) const {
    return export_latent_seq_range_to_host(backend, 0, patch_count);
}

std::vector<float> VoxCPMOutputPool::export_latent_seq_range_to_host(VoxCPMBackend& backend,
                                                                     int first_patch_index,
                                                                     int patch_count) const {
    if (latent_seq_ == nullptr || patch_count <= 0 || first_patch_index < 0) {
        return {};
    }
    if (first_patch_index + patch_count > shape_.max_latent_patches) {
        return {};
    }

    std::vector<float> latent(static_cast<size_t>(patch_count) * patch_element_count(shape_), 0.0f);
    const size_t offset = static_cast<size_t>(first_patch_index) * patch_element_count(shape_) * sizeof(float);
    backend.tensor_get(latent_seq_, latent.data(), offset, latent.size() * sizeof(float));
    return latent;
}

ggml_tensor* VoxCPMOutputPool::make_audio_vae_latent_view(ggml_context* ctx,
                                                          int frame_offset,
                                                          int frame_count) const {
    if (ctx == nullptr || latent_seq_ == nullptr || frame_count <= 0 || frame_offset < 0) {
        return nullptr;
    }
    if (frame_offset + frame_count > shape_.max_latent_patches) {
        return nullptr;
    }

    ggml_tensor* patch_major_view = make_latent_seq_frame_view(ctx, latent_seq_, shape_, frame_offset, frame_count);
    if (patch_major_view == nullptr) {
        return nullptr;
    }

    return ggml_cont(ctx, ggml_transpose(ctx, patch_major_view));
}

std::vector<float> VoxCPMOutputPool::export_audio_vae_latent_to_host(VoxCPMBackend& backend,
                                                                     int frame_offset,
                                                                     int frame_count) const {
    if (latent_seq_ == nullptr || frame_count <= 0 || frame_offset < 0) {
        return {};
    }
    if (frame_offset + frame_count > shape_.max_latent_patches) {
        return {};
    }

    const int total_patches = frame_count * shape_.patch_size;
    VoxCPMContext graph_ctx(ContextType::Graph, 1024, 8192);
    ggml_tensor* latent = make_audio_vae_latent_view(graph_ctx.raw_context(), frame_offset, frame_count);
    if (latent == nullptr) {
        return {};
    }
    ggml_set_output(latent);

    ggml_cgraph* graph = graph_ctx.new_graph();
    graph_ctx.build_forward(graph, latent);
    backend.reserve_compute_memory(graph, "output_pool.export_audio_vae_latent");
    backend.alloc_graph(graph, "output_pool.export_audio_vae_latent");
    if (backend.compute(graph) != GGML_STATUS_SUCCESS) {
        return {};
    }

    std::vector<float> latent_host(static_cast<size_t>(total_patches) * shape_.feat_dim, 0.0f);
    backend.tensor_get(latent, latent_host.data(), 0, latent_host.size() * sizeof(float));
    return latent_host;
}

HostDecodeOutput VoxCPMOutputPool::export_decode_output_to_host(VoxCPMBackend& backend) const {
    HostDecodeOutput out;
    out.patch = export_patch_to_host(backend);
    out.stop_logits = export_stop_logits_to_host(backend);
    return out;
}

}  // namespace voxcpm
