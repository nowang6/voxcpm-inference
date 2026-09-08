/**
 * @file backend.cpp
 * @brief VoxCPM Backend Implementation
 */

#include "voxcpm/backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"  // ggml_cgraph 完整定义（n_nodes/n_leafs）
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

// 解析 backend 选择：VOXCPM_BACKEND=cpu 强制 CPU（含 HTP 权重 A/B 对拍），
// 未设置或 =htp 时按请求类型。返回 true 表示强制 CPU。
bool backend_forced_cpu() {
    const char* raw = std::getenv("VOXCPM_BACKEND");
    if (!raw) {
        return false;
    }
    return to_lower_copy(raw) == "cpu";
}

}  // namespace

// =============================================================================
// Construction / Destruction
// =============================================================================

VoxCPMBackend::VoxCPMBackend(BackendType type, int n_threads)
    : type_(type), n_threads_(n_threads), backend_(nullptr), gallocr_(nullptr) {
    BackendInitResult result = init_requested_backend(BackendType::CPU, n_threads);
    backend_ = result.backend;
    type_ = BackendType::CPU;
    backend_name_ = std::move(result.name);
    backend_description_ = std::move(result.description);

    // CPU+HTP 异构：HTP0 设备初始化失败或 VOXCPM_BACKEND=cpu 时回退纯 CPU
    if (type == BackendType::HTP && !backend_forced_cpu()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name("HTP0");
        if (dev == nullptr) {
            dev = ggml_backend_dev_by_name("HTP");
        }
        if (dev != nullptr) {
            htp_backend_ = ggml_backend_dev_init(dev, nullptr);
            if (htp_backend_ != nullptr) {
                htp_buft_ = ggml_backend_dev_buffer_type(dev);
                type_ = BackendType::HTP;
                backend_name_ = std::string(ggml_backend_name(htp_backend_)) + "+" + backend_name_;
                backend_description_ =
                    std::string(ggml_backend_dev_description(dev)) + " + " + backend_description_;
            }
        }
        if (htp_backend_ == nullptr) {
            std::cerr << "[backend] HTP device unavailable, falling back to CPU-only\n";
        }
    }

    allocator_logging_enabled_ = env_flag_enabled("VOXCPM_LOG_ALLOCATOR");
}

VoxCPMBackend::~VoxCPMBackend() {
    // Free scheduler
    if (sched_) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }

    // Free allocator
    if (gallocr_) {
        ggml_gallocr_free(gallocr_);
    }

    // Free backends
    if (htp_backend_) {
        ggml_backend_free(htp_backend_);
        htp_backend_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
    }
}

VoxCPMBackend::VoxCPMBackend(VoxCPMBackend&& other) noexcept
    : type_(other.type_),
      n_threads_(other.n_threads_),
      backend_(other.backend_),
      gallocr_(other.gallocr_),
      htp_backend_(other.htp_backend_),
      htp_buft_(other.htp_buft_),
      sched_(other.sched_),
      allocator_logging_enabled_(other.allocator_logging_enabled_),
      backend_name_(std::move(other.backend_name_)),
      backend_description_(std::move(other.backend_description_)) {
    other.backend_ = nullptr;
    other.gallocr_ = nullptr;
    other.htp_backend_ = nullptr;
    other.htp_buft_ = nullptr;
    other.sched_ = nullptr;
}

VoxCPMBackend& VoxCPMBackend::operator=(VoxCPMBackend&& other) noexcept {
    if (this != &other) {
        // Free current resources
        if (sched_) ggml_backend_sched_free(sched_);
        if (gallocr_) ggml_gallocr_free(gallocr_);
        if (htp_backend_) ggml_backend_free(htp_backend_);
        if (backend_) ggml_backend_free(backend_);

        // Move from other
        type_ = other.type_;
        n_threads_ = other.n_threads_;
        backend_ = other.backend_;
        gallocr_ = other.gallocr_;
        htp_backend_ = other.htp_backend_;
        htp_buft_ = other.htp_buft_;
        sched_ = other.sched_;
        allocator_logging_enabled_ = other.allocator_logging_enabled_;
        backend_name_ = std::move(other.backend_name_);
        backend_description_ = std::move(other.backend_description_);

        other.backend_ = nullptr;
        other.gallocr_ = nullptr;
        other.htp_backend_ = nullptr;
        other.htp_buft_ = nullptr;
        other.sched_ = nullptr;
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

void VoxCPMBackend::sched_reset_for(ggml_cgraph* graph) {
    // sched 与图绑定：图重建则 sched 重建。Stage 2d 图复用后本函数整个推理
    // 过程只走一次（uid 稳定 → HTP opbatch 编译缓存命中）。
    if (sched_) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }
    // 顺序约束：HTP 在前、CPU 在后（sched 要求最后一个 backend 为 CPU）
    // graph_size 决定 sched 内部 hash_set 容量，须留足余量（split_graph 会
    // 记录节点/叶子之外的视图与中间张量），不足时 ggml_hash_find 会直接 abort
    ggml_backend_t backends[2] = {htp_backend_, backend_};
    const int graph_size = 4 * static_cast<int>(graph->n_nodes + graph->n_leafs) + 512;
    // op_offload=false：CPU 权重的 op 严格留在 CPU。开启 offload 会把 CPU
    // 权重的 GEMM 也调到 HTP，而 HTP 无法正确读取 CPU malloc 内存（无 dmabuf），
    // 产生全错结果（Stage 2b/2c 实测）。
    const bool op_offload = env_flag_enabled("VOXCPM_HTP_OP_OFFLOAD");
    sched_ = ggml_backend_sched_new(backends, nullptr, 2, graph_size,
                                    /*parallel=*/false, /*op_offload=*/op_offload);
    if (!sched_) {
        throw Error(ErrorCode::OutOfMemory, "Failed to create heterogeneous scheduler");
    }
    if (!ggml_backend_sched_reserve(sched_, graph)) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
        throw Error(ErrorCode::OutOfMemory, "Failed to reserve heterogeneous scheduler buffers");
    }
    // 显式分配图张量：v0.22 sched 的图张量（含输入 leaf）默认推迟到首次
    // compute 才分配，而 mock 在 set 输入时需要 buffer 已就位
    if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
        throw Error(ErrorCode::OutOfMemory, "Failed to allocate heterogeneous scheduler graph");
    }
}

void VoxCPMBackend::reserve_compute_memory(ggml_cgraph* graph, const char* stage) {
    if (htp_backend_ != nullptr) {
        sched_reset_for(graph);
        if (allocator_logging_enabled_) {
            std::cerr << "[allocator] action=sched-reserve"
                      << " stage=" << (stage ? stage : "(unnamed)") << "\n";
        }
        return;
    }
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
    if (htp_backend_ != nullptr) {
        // sched 路径：图张量在首次 sched_graph_compute 时分配（v0.22 语义），
        // 无需独立 alloc 步骤
        (void)stage;
        return;
    }
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
    if (htp_backend_ != nullptr) {
        const ggml_status status = ggml_backend_sched_graph_compute(sched_, graph);
        if (env_flag_enabled("VOXCPM_LOG_SCHED")) {
            std::cerr << "[sched] n_splits=" << ggml_backend_sched_get_n_splits(sched_)
                      << " n_nodes=" << graph->n_nodes << "\n";
        }
        return status;
    }
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
