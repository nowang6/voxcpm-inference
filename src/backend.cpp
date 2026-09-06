/**
 * @file backend.cpp
 * @brief VoxCPM Backend Implementation
 */

#include "voxcpm/backend.h"
#include "ggml-cpu.h"
#include <algorithm>
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
    result.name = ggml_backend_name(result.backend);
    result.description = "CPU backend";
    return result;
}

BackendInitResult init_requested_backend(BackendType type, int n_threads) {
    if (type != BackendType::CPU) {
        throw Error(ErrorCode::BackendError, "Requested backend is not implemented in VoxCPM yet");
    }
    return init_cpu_backend(n_threads);
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
    allocator_logging_enabled_ = env_flag_enabled("VOXCPM_LOG_ALLOCATOR");
    backend_name_ = std::move(result.name);
    backend_description_ = std::move(result.description);
}

VoxCPMBackend::~VoxCPMBackend() {
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
      allocator_logging_enabled_(other.allocator_logging_enabled_),
      backend_name_(std::move(other.backend_name_)),
      backend_description_(std::move(other.backend_description_)) {
    other.backend_ = nullptr;
    other.gallocr_ = nullptr;
}

VoxCPMBackend& VoxCPMBackend::operator=(VoxCPMBackend&& other) noexcept {
    if (this != &other) {
        // Free current resources
        if (gallocr_) ggml_gallocr_free(gallocr_);
        if (backend_) ggml_backend_free(backend_);

        // Move from other
        type_ = other.type_;
        n_threads_ = other.n_threads_;
        backend_ = other.backend_;
        gallocr_ = other.gallocr_;
        allocator_logging_enabled_ = other.allocator_logging_enabled_;
        backend_name_ = std::move(other.backend_name_);
        backend_description_ = std::move(other.backend_description_);

        other.backend_ = nullptr;
        other.gallocr_ = nullptr;
    }
    return *this;
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

    ggml_backend_tensor_set(tensor, data, offset, size);
}

void VoxCPMBackend::tensor_get(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    ggml_backend_tensor_get(tensor, data, offset, size);
}

void VoxCPMBackend::tensor_copy(ggml_tensor* src, ggml_tensor* dst) {
    ggml_backend_tensor_copy(src, dst);
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

// =============================================================================
// Helper Functions
// =============================================================================

}  // namespace voxcpm
