#ifndef VOXCPM_WEIGHT_STORE_H
#define VOXCPM_WEIGHT_STORE_H

#include "voxcpm/common.h"

#include <memory>
#include <string>
#include <vector>

namespace voxcpm {

class VoxCPMBackend;

/**
 * @brief 权重分组谓词：返回 true 的张量放入 HTP buffer，否则留在 CPU buffer。
 *
 * 判据可用张量名（t->name）、类型（t->type）与形状（t->ne）。
 * 传 nullptr 时全部权重留在 CPU（原有行为）。
 */
using WeightGroupFn = bool (*)(const ggml_tensor* t);

class VoxCPMWeightStore {
public:
    VoxCPMWeightStore() = default;
    ~VoxCPMWeightStore();

    VoxCPMWeightStore(const VoxCPMWeightStore&) = delete;
    VoxCPMWeightStore& operator=(const VoxCPMWeightStore&) = delete;

    /**
     * @brief 加载 GGUF 权重并按 group_fn 分桶
     * @param group_fn 分组谓词（nullptr = 全部 CPU，兼容原行为）
     *
     * HTP 桶的 buffer 以 USAGE_WEIGHTS 分配在 backend.htp_buffer_type() 上；
     * 仅当 backend 处于异构模式（is_htp_active）且 group_fn 命中时才有 HTP 桶。
     */
    bool load_from_file(const std::string& gguf_path, VoxCPMBackend& backend,
                        WeightGroupFn group_fn = nullptr);

    /** @brief 按名取张量（跨 CPU/HTP 两桶查询） */
    ggml_tensor* get_tensor(const char* name) const;

    bool get_u32(const char* key, uint32_t& value) const;
    bool get_bool(const char* key, bool& value) const;
    bool get_string(const char* key, std::string& value) const;
    bool get_i32_array(const char* key, std::vector<int>& values) const;

    gguf_context* gguf() const { return gguf_ctx_; }
    ggml_context* ggml_ctx() const { return ggml_ctx_; }
    ggml_context* htp_ggml_ctx() const { return htp_ggml_ctx_; }
    ggml_backend_buffer_t buffer() const { return buffer_; }
    ggml_backend_buffer_t htp_buffer() const { return htp_buffer_; }
    const std::string& path() const { return path_; }

    size_t buffer_size() const;
    size_t htp_buffer_size() const;
    int tensor_count() const;
    bool owns_storage() const {
        return gguf_ctx_ != nullptr && ggml_ctx_ != nullptr && buffer_ != nullptr;
    }

private:
    gguf_context* gguf_ctx_ = nullptr;
    ggml_context* ggml_ctx_ = nullptr;       // CPU 桶（GGUF 元数据 ctx）
    ggml_context* htp_ggml_ctx_ = nullptr;   // HTP 桶
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_buffer_t htp_buffer_ = nullptr;
    std::string path_;
};

using SharedWeightStore = std::shared_ptr<VoxCPMWeightStore>;

}  // namespace voxcpm

#endif  // VOXCPM_WEIGHT_STORE_H
