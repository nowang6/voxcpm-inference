/**
 * @file common.h
 * @brief VoxCPM Common Definitions
 *
 * Common macros, types, and utilities for VoxCPM GGML implementation.
 */

#ifndef VOXCPM_COMMON_H
#define VOXCPM_COMMON_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

// GGML headers
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

namespace voxcpm {

// =============================================================================
// Version
// =============================================================================

constexpr int VOXCPM_VERSION_MAJOR = 0;
constexpr int VOXCPM_VERSION_MINOR = 1;
constexpr int VOXCPM_VERSION_PATCH = 0;

// =============================================================================
// Macros
// =============================================================================

#define VOXCPM_UNUSED(x) (void)(x)

#define VOXCPM_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "VOXCPM_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

// =============================================================================
// fp16 权重组图辅助
// =============================================================================

// 权重按 GGUF 中的原生类型常驻(F16/Q8_0/F32,见 weight-store)。ggml 的二元算子
// (add/mul 等)没有 F32(src0) + F16(src1) → F32 路径,把 F16 权重喂给这类算子前
// 需要上转;mul_mat/conv_transpose_1d/get_rows 等算子原生支持 F16 权重,无需上转。
// 已是 F32 的张量原样返回,不产生额外图节点。
inline ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* tensor) {
    if (tensor == nullptr || tensor->type == GGML_TYPE_F32) {
        return tensor;
    }
    return ggml_cast(ctx, tensor, GGML_TYPE_F32);
}

// =============================================================================
// Error Handling
// =============================================================================

enum class ErrorCode {
    Success = 0,
    InvalidArgument,
    OutOfMemory,
    FileNotFound,
    InvalidFormat,
    BackendError,
    NotImplemented,
};

class Error : public std::runtime_error {
public:
    explicit Error(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code() const { return code_; }

private:
    ErrorCode code_;
};

// =============================================================================
// Memory Utilities
// =============================================================================

/**
 * @brief Calculate context memory size for given tensor count
 *
 * Following GGML best practice: context only stores metadata, not tensor data.
 * Size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead() + margin
 */
inline size_t calc_context_size(int n_tensors, int max_nodes = 0) {
    size_t size = 0;
    size += n_tensors * ggml_tensor_overhead();
    if (max_nodes > 0) {
        size += ggml_graph_overhead_custom(max_nodes, false);
    } else {
        // Default to GGML_DEFAULT_GRAPH_SIZE for graph contexts
        size += ggml_graph_overhead();  // Uses GGML_DEFAULT_GRAPH_SIZE
    }
    size += 1024;  // Safety margin
    return size;
}

}  // namespace voxcpm

#endif  // VOXCPM_COMMON_H
