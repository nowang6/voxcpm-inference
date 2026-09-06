// 临时调试工具:用 VoxCPMWeightStore 加载 GGUF,打印指定张量的类型/形状、
// F32 张量的 FNV-1a(按 IEEE 位型)哈希与前几个值。权重常驻原生量化类型、
// 不再上转 F32,非 F32 张量直接跳过。
#include "voxcpm/backend.h"
#include "voxcpm/weight-store.h"

#include <cstdio>
#include <cstring>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [tensor_name ...]\n", argv[0]);
        return 1;
    }
    voxcpm::VoxCPMBackend backend(voxcpm::BackendType::CPU, 4);
    auto store = std::make_shared<voxcpm::VoxCPMWeightStore>();
    if (!store->load_from_file(argv[1], backend)) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    for (int i = 2; i < argc; ++i) {
        ggml_tensor* t = store->get_tensor(argv[i]);
        if (!t) {
            printf("%s: NOT FOUND\n", argv[i]);
            continue;
        }
        printf("%s: type=%s ne=[%lld,%lld,%lld,%lld]\n", argv[i], ggml_type_name(t->type),
               (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
        if (t->type == GGML_TYPE_F32) {
            const float* d = static_cast<const float*>(t->data);
            const int64_t n = ggml_nelements(t);
            uint32_t h = 2166136261u;
            for (int64_t k = 0; k < n; ++k) {
                uint32_t bits;
                memcpy(&bits, &d[k], sizeof(bits));
                h = (h ^ bits) * 16777619u;
            }
            printf("  fnv1a(bits)=%08x first8:", h);
            for (int k = 0; k < 8 && k < n; ++k) {
                printf(" %.7g", d[k]);
            }
            printf("\n");
        } else {
            printf("  (non-f32, skip)\n");
        }
    }
    return 0;
}
