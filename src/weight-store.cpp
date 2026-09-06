#include "voxcpm/weight-store.h"

#include "voxcpm/backend.h"

#include <cstdio>
#include <vector>

namespace voxcpm {

namespace {

// 张量按文件中的原生类型(F16/Q8_0/Q4_0/F32)常驻,这里只做按字节读入,不做类型转换。
bool load_tensor_data(FILE* file,
                      gguf_context* gguf_ctx,
                      int tensor_idx,
                      ggml_tensor* tensor,
                      ggml_backend_buffer_t buffer) {
    if (!file || !gguf_ctx || !tensor || !buffer) {
        return false;
    }

    const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
    const size_t nbytes = ggml_nbytes(tensor);

#ifdef _WIN32
    if (_fseeki64(file, static_cast<__int64>(offset), SEEK_SET) != 0) {
#else
    if (fseeko(file, static_cast<off_t>(offset), SEEK_SET) != 0) {
#endif
        return false;
    }

    const bool host_buffer = ggml_backend_buffer_is_host(buffer);

    if (gguf_get_tensor_size(gguf_ctx, tensor_idx) != nbytes) {
        // 张量类型在重建时被改过才会走到这里,当前按原生类型加载,不应发生。
        return false;
    }

    if (host_buffer) {
        return fread(tensor->data, 1, nbytes, file) == nbytes;
    }

    std::vector<uint8_t> temp(nbytes);
    if (fread(temp.data(), 1, nbytes, file) != nbytes) {
        return false;
    }

    ggml_backend_tensor_set(tensor, temp.data(), 0, nbytes);
    return true;
}

}  // namespace

VoxCPMWeightStore::~VoxCPMWeightStore() {
    if (buffer_) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (gguf_ctx_) {
        gguf_free(gguf_ctx_);
        gguf_ctx_ = nullptr;
    }
    if (ggml_ctx_) {
        ggml_free(ggml_ctx_);
        ggml_ctx_ = nullptr;
    }
}

bool VoxCPMWeightStore::load_from_file(const std::string& gguf_path, VoxCPMBackend& backend) {
    if (owns_storage()) {
        return path_ == gguf_path;
    }

    // 元数据 + 原始张量定义(no_alloc,不分配数据)。
    ggml_context* meta_ctx = nullptr;
    gguf_init_params params = {
        .no_alloc = true,
        .ctx = &meta_ctx,
    };

    gguf_context* gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx || !meta_ctx) {
        if (gguf_ctx) {
            gguf_free(gguf_ctx);
        }
        if (meta_ctx) {
            ggml_free(meta_ctx);
        }
        return false;
    }

    // 权重按 GGUF 中的原生类型(F16/Q8_0/F32)直接常驻后端 buffer,不再统一上转。
    // F16 权重的消费方式:ggml_mul_mat / conv_transpose_1d / get_rows 等算子原生支持
    // F16 权重;二元算子(add/mul 等)缺少 F32(src0)+F16(src1) 路径,由各调用点在组图时
    // 用 to_f32()(common.h)按需上转。
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(meta_ctx, backend.raw_backend());
    if (!buffer) {
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE* file = fopen(gguf_path.c_str(), "rb");
    if (!file) {
        ggml_backend_buffer_free(buffer);
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    bool ok = true;
    for (int i = 0; i < n_tensors && ok; ++i) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        ggml_tensor* tensor = ggml_get_tensor(meta_ctx, name);
        if (!tensor) {
            ok = false;
            break;
        }
        ok = load_tensor_data(file, gguf_ctx, i, tensor, buffer);
    }

    fclose(file);

    if (!ok) {
        ggml_backend_buffer_free(buffer);
        gguf_free(gguf_ctx);
        ggml_free(meta_ctx);
        return false;
    }

    gguf_ctx_ = gguf_ctx;
    ggml_ctx_ = meta_ctx;
    buffer_ = buffer;
    path_ = gguf_path;
    return true;
}

ggml_tensor* VoxCPMWeightStore::get_tensor(const char* name) const {
    if (!ggml_ctx_ || !name) {
        return nullptr;
    }
    return ggml_get_tensor(ggml_ctx_, name);
}

bool VoxCPMWeightStore::get_u32(const char* key, uint32_t& value) const {
    if (!gguf_ctx_ || !key) {
        return false;
    }
    const int idx = gguf_find_key(gguf_ctx_, key);
    if (idx < 0) {
        return false;
    }
    value = gguf_get_val_u32(gguf_ctx_, idx);
    return true;
}

bool VoxCPMWeightStore::get_bool(const char* key, bool& value) const {
    if (!gguf_ctx_ || !key) {
        return false;
    }
    const int idx = gguf_find_key(gguf_ctx_, key);
    if (idx < 0) {
        return false;
    }
    const gguf_type type = gguf_get_kv_type(gguf_ctx_, idx);
    if (type == GGUF_TYPE_BOOL) {
        value = gguf_get_val_bool(gguf_ctx_, idx);
        return true;
    }
    if (type == GGUF_TYPE_UINT32) {
        value = gguf_get_val_u32(gguf_ctx_, idx) != 0;
        return true;
    }
    return false;
}

bool VoxCPMWeightStore::get_string(const char* key, std::string& value) const {
    if (!gguf_ctx_ || !key) {
        return false;
    }
    const int idx = gguf_find_key(gguf_ctx_, key);
    if (idx < 0 || gguf_get_kv_type(gguf_ctx_, idx) != GGUF_TYPE_STRING) {
        return false;
    }
    const char* data = gguf_get_val_str(gguf_ctx_, idx);
    if (!data) {
        return false;
    }
    value = data;
    return true;
}

bool VoxCPMWeightStore::get_i32_array(const char* key, std::vector<int>& values) const {
    if (!gguf_ctx_ || !key) {
        return false;
    }
    const int idx = gguf_find_key(gguf_ctx_, key);
    if (idx < 0) {
        return false;
    }
    const int32_t* data = static_cast<const int32_t*>(gguf_get_arr_data(gguf_ctx_, idx));
    const size_t n = gguf_get_arr_n(gguf_ctx_, idx);
    if (!data && n != 0) {
        return false;
    }
    values.assign(data, data + n);
    return true;
}

size_t VoxCPMWeightStore::buffer_size() const {
    return buffer_ ? ggml_backend_buffer_get_size(buffer_) : 0;
}

int VoxCPMWeightStore::tensor_count() const {
    return gguf_ctx_ ? gguf_get_n_tensors(gguf_ctx_) : 0;
}

}  // namespace voxcpm
