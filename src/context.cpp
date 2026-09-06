/**
 * @file context.cpp
 * @brief VoxCPM Context Implementation
 */

#include "voxcpm/context.h"
#include <cstring>

namespace voxcpm {

// =============================================================================
// Construction / Destruction
// =============================================================================

VoxCPMContext::VoxCPMContext(ContextType type, int n_tensors, int max_nodes, size_t extra_mem_size)
    : type_(type), ctx_(nullptr), mem_size_(0), max_nodes_(max_nodes) {

    // Calculate memory size (metadata only, no tensor data)
    mem_size_ = calc_context_size(n_tensors, max_nodes) + extra_mem_size;

    // Allocate graph buffer if needed
    if (type == ContextType::Graph) {
        graph_buffer_.resize(mem_size_);
    }

    // Initialize context with no_alloc=true
    struct ggml_init_params params = {
        .mem_size = mem_size_,
        .mem_buffer = (type == ContextType::Graph) ? graph_buffer_.data() : nullptr,
        .no_alloc = true,  // Critical: only store metadata
    };

    ctx_ = ggml_init(params);
    if (!ctx_) {
        throw Error(ErrorCode::OutOfMemory, "Failed to create context");
    }
}

VoxCPMContext::~VoxCPMContext() {
    if (ctx_) {
        ggml_free(ctx_);
    }
}

VoxCPMContext::VoxCPMContext(VoxCPMContext&& other) noexcept
    : type_(other.type_),
      ctx_(other.ctx_),
      mem_size_(other.mem_size_),
      max_nodes_(other.max_nodes_),
      graph_buffer_(std::move(other.graph_buffer_)) {
    other.ctx_ = nullptr;
    other.mem_size_ = 0;
    other.max_nodes_ = 0;
}

VoxCPMContext& VoxCPMContext::operator=(VoxCPMContext&& other) noexcept {
    if (this != &other) {
        if (ctx_) {
            ggml_free(ctx_);
        }

        type_ = other.type_;
        ctx_ = other.ctx_;
        mem_size_ = other.mem_size_;
        max_nodes_ = other.max_nodes_;
        graph_buffer_ = std::move(other.graph_buffer_);

        other.ctx_ = nullptr;
        other.mem_size_ = 0;
        other.max_nodes_ = 0;
    }
    return *this;
}

// =============================================================================
// Tensor Creation
// =============================================================================

ggml_tensor* VoxCPMContext::new_tensor_2d(ggml_type type, int64_t ne0, int64_t ne1) {
    return ggml_new_tensor_2d(ctx_, type, ne0, ne1);
}

ggml_tensor* VoxCPMContext::new_tensor_3d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    return ggml_new_tensor_3d(ctx_, type, ne0, ne1, ne2);
}

// =============================================================================
// Compute Graph
// =============================================================================

ggml_cgraph* VoxCPMContext::new_graph() {
    // Use stored max_nodes_ or fall back to GGML_DEFAULT_GRAPH_SIZE
    size_t size = max_nodes_ > 0 ? max_nodes_ : GGML_DEFAULT_GRAPH_SIZE;
    return ggml_new_graph_custom(ctx_, size, false);
}

ggml_cgraph* VoxCPMContext::new_graph(size_t size) {
    return ggml_new_graph_custom(ctx_, size, false);
}

void VoxCPMContext::build_forward(ggml_cgraph* graph, ggml_tensor* output) {
    ggml_build_forward_expand(graph, output);
}

}  // namespace voxcpm
