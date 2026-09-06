/**
 * @file context.h
 * @brief VoxCPM Context Management
 *
 * Encapsulates GGML context creation and management following best practices:
 * - Uses no_alloc=true for all contexts
 * - Context stores only tensor metadata, not data
 * - Provides tensor creation and lookup utilities
 */

#ifndef VOXCPM_CONTEXT_H
#define VOXCPM_CONTEXT_H

#include "common.h"
#include <memory>
#include <vector>

namespace voxcpm {

/**
 * @brief Context type enumeration
 */
enum class ContextType {
    Weights,    // Model weights (persistent)
    Graph,      // Compute graph (temporary)
};

/**
 * @brief GGML Context Wrapper
 *
 * This class encapsulates GGML context operations following best practices:
 * - Uses no_alloc=true: context stores only metadata
 * - Memory size calculation includes only overhead, not tensor data
 * - Tensor data is allocated separately via Backend
 *
 * Thread Safety: Each instance manages its own memory pool independently.
 */
class VoxCPMContext {
public:
    // =========================================================================
    // Construction / Destruction
    // =========================================================================

    /**
     * @brief Construct a context
     * @param type Context type
     * @param n_tensors Expected number of tensors
     * @param max_nodes Maximum graph nodes (for Graph type)
     * @param extra_mem_size Additional metadata headroom for known-large graphs
     */
    VoxCPMContext(ContextType type, int n_tensors, int max_nodes = 0, size_t extra_mem_size = 0);

    ~VoxCPMContext();

    // Non-copyable
    VoxCPMContext(const VoxCPMContext&) = delete;
    VoxCPMContext& operator=(const VoxCPMContext&) = delete;

    // Movable
    VoxCPMContext(VoxCPMContext&& other) noexcept;
    VoxCPMContext& operator=(VoxCPMContext&& other) noexcept;

    // =========================================================================
    // Tensor Creation
    // =========================================================================

    /**
     * @brief Create a 2D tensor
     */
    ggml_tensor* new_tensor_2d(ggml_type type, int64_t ne0, int64_t ne1);

    /**
     * @brief Create a 3D tensor
     */
    ggml_tensor* new_tensor_3d(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);

    // =========================================================================
    // Compute Graph
    // =========================================================================

    /**
     * @brief Create a new compute graph
     *
     * Uses the max_nodes_ value passed at construction time, or GGML_DEFAULT_GRAPH_SIZE
     */
    ggml_cgraph* new_graph();

    /**
     * @brief Create a new compute graph with custom size
     * @param size Maximum number of nodes in the graph
     */
    ggml_cgraph* new_graph(size_t size);

    /**
     * @brief Build forward computation graph
     * @param graph Graph to build
     * @param output Output tensor
     */
    void build_forward(ggml_cgraph* graph, ggml_tensor* output);

    // =========================================================================
    // Utilities
    // =========================================================================

    /**
     * @brief Get raw context handle
     */
    ggml_context* raw_context() const { return ctx_; }

    /**
     * @brief Get context type
     */
    ContextType type() const { return type_; }

    /**
     * @brief Get memory size
     */
    size_t mem_size() const { return mem_size_; }

    /**
     * @brief Get maximum graph nodes
     */
    size_t max_nodes() const { return max_nodes_; }

    /**
     * @brief Check if context is valid
     */
    bool is_valid() const { return ctx_ != nullptr; }

private:
    ContextType type_;
    ggml_context* ctx_;
    size_t mem_size_;
    size_t max_nodes_;  // Maximum graph nodes for graph contexts

    // Graph context memory pool (avoids frequent malloc/free)
    std::vector<uint8_t> graph_buffer_;
};

}  // namespace voxcpm

#endif  // VOXCPM_CONTEXT_H
