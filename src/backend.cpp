/**
 * @file backend.cpp
 * @brief VoxCPM Backend Implementation
 *
 * 纯 CPU 版本:自 VoxCPM.cpp 迁移时剥离了 CUDA/Vulkan/Metal 后端初始化与
 * 多后端 scheduler,仅保留 CPU 后端 + gallocr 计算竞技场 + 缓冲区管理。
 */

#include "voxcpm/backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace voxcpm {

namespace {

struct BackendInitResult {
    ggml_backend_t backend = nullptr;
    BackendType type = BackendType::CPU;
    bool is_gpu = false;
    std::string name;
    std::string description;
};

std::string to_lower_copy(const char* value) {
    std::string result = value ? value : "";
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return result;
}

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return false;
    }

    const std::string value = to_lower_copy(raw);
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

std::string format_mib(size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB";
    return oss.str();
}

BackendInitResult init_cpu_backend(int n_threads) {
    BackendInitResult result;
    result.backend = ggml_backend_cpu_init();
    if (!result.backend) {
        throw Error(ErrorCode::BackendError, "Failed to initialize CPU backend");
    }

    ggml_backend_cpu_set_n_threads(result.backend, n_threads);
    result.type = BackendType::CPU;
    result.is_gpu = false;
    result.name = ggml_backend_name(result.backend);
    result.description = "CPU backend";
    return result;
}

BackendInitResult init_requested_backend(BackendType type, int n_threads) {
    switch (type) {
        case BackendType::CPU:
            return init_cpu_backend(n_threads);

        default:
            throw Error(ErrorCode::BackendError, "Requested backend is not implemented in VoxCPM yet");
    }
}

void free_tracked_buffers(std::vector<ggml_backend_buffer_t>& buffers) {
    for (auto& buf : buffers) {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
    }
    buffers.clear();
}

}  // namespace

// =============================================================================
// Construction / Destruction
// =============================================================================

VoxCPMBackend::VoxCPMBackend(BackendType type, int n_threads)
    : type_(type), n_threads_(n_threads), backend_(nullptr), gallocr_(nullptr) {
    BackendInitResult result = init_requested_backend(type, n_threads);
    backend_ = result.backend;
    type_ = result.type;
    is_gpu_ = result.is_gpu;
    allocator_logging_enabled_ = env_flag_enabled("VOXCPM_LOG_ALLOCATOR");
    backend_name_ = std::move(result.name);
    backend_description_ = std::move(result.description);
}

VoxCPMBackend::~VoxCPMBackend() {
    free_tracked_buffers(buffers_);

    // Free allocator
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
    }

    // Free backend
    if (backend_) {
        ggml_backend_free(backend_);
    }
}

VoxCPMBackend::VoxCPMBackend(VoxCPMBackend&& other) noexcept
    : type_(other.type_),
      n_threads_(other.n_threads_),
      backend_(other.backend_),
      gallocr_(other.gallocr_),
      is_gpu_(other.is_gpu_),
      allocator_logging_enabled_(other.allocator_logging_enabled_),
      backend_name_(std::move(other.backend_name_)),
      backend_description_(std::move(other.backend_description_)),
      buffers_(std::move(other.buffers_)) {
    other.backend_ = nullptr;
    other.gallocr_ = nullptr;
    other.is_gpu_ = false;
    other.buffers_.clear();
}

VoxCPMBackend& VoxCPMBackend::operator=(VoxCPMBackend&& other) noexcept {
    if (this != &other) {
        // Free current resources
        free_tracked_buffers(buffers_);
        if (gallocr_) ggml_gallocr_free(gallocr_);
        if (backend_) ggml_backend_free(backend_);

        // Move from other
        type_ = other.type_;
        n_threads_ = other.n_threads_;
        backend_ = other.backend_;
        gallocr_ = other.gallocr_;
        is_gpu_ = other.is_gpu_;
        allocator_logging_enabled_ = other.allocator_logging_enabled_;
        backend_name_ = std::move(other.backend_name_);
        backend_description_ = std::move(other.backend_description_);
        buffers_ = std::move(other.buffers_);

        other.backend_ = nullptr;
        other.gallocr_ = nullptr;
        other.is_gpu_ = false;
        other.buffers_.clear();
    }
    return *this;
}

// =============================================================================
// Buffer Management
// =============================================================================

ggml_backend_buffer_t VoxCPMBackend::alloc_buffer(ggml_context* ctx, BufferUsage usage) {
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend_);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);

    if (!buffer) {
        throw Error(ErrorCode::OutOfMemory, "Failed to allocate buffer");
    }

    // Set usage for weights
    if (usage == BufferUsage::Weights) {
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // Track buffer
    buffers_.push_back(buffer);

    return buffer;
}

void VoxCPMBackend::free_buffer(ggml_backend_buffer_t buffer) {
    if (buffer) {
        // Remove from tracking
        auto it = std::find(buffers_.begin(), buffers_.end(), buffer);
        if (it != buffers_.end()) {
            buffers_.erase(it);
        }
        ggml_backend_buffer_free(buffer);
    }
}

// =============================================================================
// Graph Allocator
// =============================================================================

void VoxCPMBackend::init_allocator() {
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
    }
    gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!gallocr_) {
        throw Error(ErrorCode::OutOfMemory, "Failed to create graph allocator");
    }
}

void VoxCPMBackend::reset_request_state() {
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
        gallocr_ = nullptr;
    }
}

void VoxCPMBackend::reserve_compute_memory(ggml_cgraph* graph, const char* stage) {
    if (!gallocr_) {
        init_allocator();
    }
    const size_t before = compute_buffer_size();
    ggml_gallocr_reserve(gallocr_, graph);
    if (allocator_logging_enabled_) {
        const size_t after = compute_buffer_size();
        std::cerr << "[allocator] action=reserve"
                  << " stage=" << (stage ? stage : "(unnamed)")
                  << " before=" << format_mib(before)
                  << " after=" << format_mib(after)
                  << " delta=" << format_mib(after >= before ? after - before : 0)
                  << "\n";
    }
}

void VoxCPMBackend::alloc_graph(ggml_cgraph* graph, const char* stage) {
    if (!gallocr_) {
        init_allocator();
    }
    const size_t before = compute_buffer_size();
    ggml_gallocr_alloc_graph(gallocr_, graph);
    if (allocator_logging_enabled_) {
        const size_t after = compute_buffer_size();
        std::cerr << "[allocator] action=alloc"
                  << " stage=" << (stage ? stage : "(unnamed)")
                  << " before=" << format_mib(before)
                  << " after=" << format_mib(after)
                  << " delta=" << format_mib(after >= before ? after - before : 0)
                  << "\n";
    }
}

// =============================================================================
// Graph Execution
// =============================================================================

ggml_status VoxCPMBackend::compute(ggml_cgraph* graph) {
    return ggml_backend_graph_compute(backend_, graph);
}

// =============================================================================
// Data Transfer
// =============================================================================

void VoxCPMBackend::tensor_set(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    if (!tensor) {
        throw Error(ErrorCode::BackendError, "tensor_set received a null tensor");
    }
    if (!data && size > 0) {
        throw Error(ErrorCode::BackendError, "tensor_set received a null data pointer");
    }
    const size_t tensor_bytes = ggml_nbytes(tensor);
    if (offset > tensor_bytes || size > tensor_bytes - offset) {
        std::ostringstream oss;
        oss << "tensor_set overflow for tensor '" << tensor->name
            << "': tensor_bytes=" << tensor_bytes
            << ", offset=" << offset
            << ", size=" << size;
        throw Error(ErrorCode::BackendError, oss.str());
    }
    if (tensor->buffer == nullptr || tensor->data == nullptr) {
        std::ostringstream oss;
        oss << "tensor_set target is not allocated for tensor '" << tensor->name
            << "': buffer=" << tensor->buffer
            << ", data=" << tensor->data
            << ", tensor_bytes=" << tensor_bytes;
        throw Error(ErrorCode::BackendError, oss.str());
    }

    const auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(tensor, data, offset, size);
    const auto end = std::chrono::steady_clock::now();
    transfer_stats_.host_to_device_bytes += size;
    transfer_stats_.host_to_device_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
}

void VoxCPMBackend::tensor_get(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    const auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(tensor, data, offset, size);
    const auto end = std::chrono::steady_clock::now();
    transfer_stats_.device_to_host_bytes += size;
    transfer_stats_.device_to_host_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
}

void VoxCPMBackend::tensor_copy(ggml_tensor* src, ggml_tensor* dst) {
    const auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_copy(src, dst);
    const auto end = std::chrono::steady_clock::now();
    transfer_stats_.device_to_device_bytes += std::min(ggml_nbytes(src), ggml_nbytes(dst));
    transfer_stats_.device_to_device_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// Utilities
// =============================================================================

bool VoxCPMBackend::is_host_buffer(ggml_backend_buffer_t buffer) const {
    return ggml_backend_buffer_is_host(buffer);
}

ggml_backend_buffer_type_t VoxCPMBackend::buffer_type() const {
    return ggml_backend_get_default_buffer_type(backend_);
}

size_t VoxCPMBackend::compute_buffer_size() const {
    size_t total = 0;
    if (gallocr_) {
        total += ggml_gallocr_get_buffer_size(gallocr_, 0);
    }
    return total;
}

}  // namespace voxcpm
