#ifndef HTP_MATMUL_OPS_H
#define HTP_MATMUL_OPS_H

#include <stdint.h>
#include <stddef.h>
#include "htp-ops.h"
#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- Tile Constraints (32-wide tiling shared by the HVX kernels) ---
#define HTP_MM_TILE_N_COLS    32
#define HTP_MM_TILE_N_ROWS    32
#define HTP_MM_TILE_SIZE      (32 * 32 * sizeof(__fp16)) // 2048 bytes
#define HTP_MM_TILE_N_ELMS    1024
#define HTP_MM_MIN_NROWS      4 // src1 row-count threshold: tiled HVX kernels use 16 prefetch buffers above it, 2 below

// --- Weight Repacked Tile Sizes ---
#define HTP_MM_WEIGHT_TILE_SIZE_Q4_0   576
#define HTP_MM_WEIGHT_TILE_SIZE_Q4_1   640
#define HTP_MM_WEIGHT_TILE_SIZE_Q8_0   1088
#define HTP_MM_WEIGHT_TILE_SIZE_IQ4_NL 576
#define HTP_MM_WEIGHT_TILE_SIZE_MXFP4  544

// --- Weight Repacked Aligned Tile Sizes ---
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0   640
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_1   640
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0   1152
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_IQ4_NL 640
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_MXFP4  640

// --- Activation Tiled Block Sizes (including padding) ---
#define HTP_MM_ACT_TILE_SIZE_Q8_0      1152
#define HTP_MM_ACT_TILE_SIZE_Q8_1      1280

#define HTP_MM_MAX_PREFETCH 16

enum htp_mm_kernel_type {
    HTP_MM_KERNEL_UNSUPPORTED = 0,

    // HVX floating-point paths
    HTP_MM_KERNEL_HVX_F16_F16_VTCM,
    HTP_MM_KERNEL_HVX_F16_F16_DDR,
    HTP_MM_KERNEL_HVX_F16_F32_DDR,

    HTP_MM_KERNEL_HVX_F32_F32_VTCM,
    HTP_MM_KERNEL_HVX_F32_F32_DDR,
    HTP_MM_KERNEL_HVX_F32_F16_DDR,

    // HVX quantized paths
    HTP_MM_KERNEL_HVX_QUANT_ROW,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_BLOCK,    // parallel block-wise quantization
    HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT, // row-wise fallback flat quantization
};

// Op-specific struct for precomputed matmul params
struct htp_mm_kernel_params {
    int32_t  kernel_type;        // enum htp_mm_kernel_type
    int32_t  pipeline;           // 1 = pipelined execution, 0 = standard
    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  n_threads;          // Number of threads to spawn
    int32_t  n_act_threads;      // Number of threads for activation preparation
    int32_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    int32_t  tile_size;          // Weight tile size
    int32_t  aligned_tile_size;  // Aligned weight tile size (padded to 128)
    int32_t  src1_row_size;      // Row size for quantized activation
    int32_t  vtcm_size;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_size;     // src0 scratchpad size in VTCM
    int32_t  vtcm_src1_size;     // src1 scratchpad size in VTCM
    int32_t  vtcm_src2_size;     // src2 scratchpad size in VTCM (fused only)
    int32_t  vtcm_src3_size;     // src3 scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_size;      // dst scratchpad size in VTCM
    int32_t  n_weights;          // Number of weights for fused NX

    // Precomputed division values
    struct fastdiv_values div_ne12_ne1;
    struct fastdiv_values div_ne1;
    struct fastdiv_values div_r2;
    struct fastdiv_values div_r3;
    struct fastdiv_values div_ne11;
    struct fastdiv_values div_n_act_threads;
    struct fastdiv_values div_ne00_padded;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#endif

struct mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// --- Tile Size Helpers ---
static inline uint32_t htp_mm_get_weight_tile_size(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
            return HTP_MM_WEIGHT_TILE_SIZE_Q4_0;
        case HTP_TYPE_Q4_1:
            return HTP_MM_WEIGHT_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q8_0:
            return HTP_MM_WEIGHT_TILE_SIZE_Q8_0;
        case HTP_TYPE_MXFP4:
            return HTP_MM_WEIGHT_TILE_SIZE_MXFP4;
        default:
            return 0;
    }
}

static inline uint32_t htp_mm_get_weight_aligned_tile_size(int weight_type) {
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_0;
        case HTP_TYPE_Q4_1:
            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q4_1;
        case HTP_TYPE_Q8_0:
            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q8_0;
        case HTP_TYPE_MXFP4:
            return HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_MXFP4;
        default:
            return 0;
    }
}

// --- Activation/Row Size Helpers ---
static inline size_t htp_mm_q8_0_tiled_row_size(uint32_t ne) {
    const uint32_t ne_padded = ((ne + 127) / 128) * 128;
    const uint32_t nb_32 = ne_padded / 32;
    return nb_32 * HTP_MM_ACT_TILE_SIZE_Q8_0;
}

static inline size_t htp_mm_q8_1_tiled_row_size(uint32_t ne) {
    const uint32_t ne_padded = ((ne + 127) / 128) * 128;
    const uint32_t nb_32 = ne_padded / 32;
    return nb_32 * HTP_MM_ACT_TILE_SIZE_Q8_1;
}

static inline size_t htp_mm_q8_0_flat_row_size(uint32_t ne) {
    const uint32_t quants_size = hex_align_up(ne, 128);
    const uint32_t num_scales = (ne + 31) / 32;
    const uint32_t scales_size = hex_align_up(num_scales * 2, 128);
    return quants_size + scales_size;
}

static inline size_t htp_mm_q8_1_flat_row_size(uint32_t ne) {
    const uint32_t quants_size = hex_align_up(ne, 128);
    const uint32_t num_scales = (ne + 31) / 32;
    const uint32_t scales_size = hex_align_up(num_scales * 4, 128);
    return quants_size + scales_size;
}

static inline size_t htp_mm_get_tiled_row_stride(int weight_type, uint32_t k) {
    uint32_t nb = (k + QK_Q4_0_TILED - 1) / QK_Q4_0_TILED;
    switch (weight_type) {
        case HTP_TYPE_Q4_0:
        case HTP_TYPE_IQ4_NL:
        case HTP_TYPE_Q4_1:
        case HTP_TYPE_Q8_0:
        case HTP_TYPE_MXFP4:
            return (size_t) nb * htp_mm_get_weight_tile_size(weight_type);
        case HTP_TYPE_F16:
            return (size_t) k * sizeof(__fp16);
        case HTP_TYPE_F32:
            return (size_t) k * sizeof(float);
        default:
            return 0;
    }
}

static inline size_t htp_mm_round_up(size_t n, size_t m) {
    return ((n + m - 1) / m) * m;
}

struct htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_src1;          // vtcm_src1 (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_src2;          // vtcm_src2 (Wq / fused only)
    size_t off_src3;          // vtcm_src3 (Wv / fused only)
    size_t off_dst;           // vtcm_dst (output scratch)

    // Cached sizes
    size_t src0_bytes;
    size_t src1_bytes;
    size_t src2_bytes;
    size_t src3_bytes;
    size_t dst_bytes;

    size_t total_bytes;
};

static inline void htp_mm_hvx_vtcm_layout_build(
    struct htp_mm_hvx_vtcm_layout * L,
    int kernel_type,
    int wtype,
    uint32_t ne10,       // k
    uint32_t src1_nrows, // m_total
    uint32_t n_threads,
    size_t dst_row_size,
    size_t src0_row_size,
    size_t src1_row_size,
    size_t src2_row_size,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    size_t src0_sz = 0;
    size_t src1_sz = 0;
    size_t src2_sz = src2_row_size > 0 ? htp_mm_round_up(src2_row_size, 128) : 0;
    size_t src3_sz = 0;
    size_t dst_sz  = 0;

    const bool is_repack = (wtype == HTP_TYPE_Q4_0 || wtype == HTP_TYPE_Q4_1 ||
                            wtype == HTP_TYPE_Q8_0 || wtype == HTP_TYPE_IQ4_NL ||
                            wtype == HTP_TYPE_MXFP4);

    if (is_fused_nx) {
        const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
        const size_t quant_scratch_size = hex_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float)) * n_threads;

        size_t weight_sz_per_thread = 0;

        if (is_repack) {
            uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
            uint32_t n_k_tiles = hex_round_up(ne10, 32) / 32;
            uint32_t tile_row_size = n_k_tiles * aligned_tile_size;

            weight_sz_per_thread = hex_round_up(n_prefetch * tile_row_size, 128);
        } else {
            weight_sz_per_thread = hex_round_up(n_prefetch * src0_row_size_padded, 128);
        }

        size_t flat_act_row_size  = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10)  : htp_mm_q8_0_flat_row_size(ne10);
        size_t tiled_act_row_size = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

        size_t act_sz = (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT)
            ? hex_round_up(flat_act_row_size  * src1_nrows, 128)
            : hex_round_up(tiled_act_row_size * src1_nrows, 128);

        src0_sz = weight_sz_per_thread * n_threads; // shared single-weight prefetch buffer
        src1_sz = act_sz;                           // quantized activation buffer
        src2_sz = 0;
        src3_sz = 0;
        dst_sz  = quant_scratch_size;
    } else if (is_matmul_id) {
        const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
        const size_t src1_row_size_tiled = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10)
                                                                    : htp_mm_q8_0_tiled_row_size(ne10);

        size_t src0_sz_per_thread = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
        src1_sz                   = htp_mm_round_up(src1_row_size_tiled * src1_nrows, 256);

        if (is_repack) {
            const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
            const uint32_t n_k_tiles         = ne10 / 32;
            const uint32_t tile_row_size     = n_k_tiles * aligned_tile_size;
            size_t repacked_vtcm_size        = htp_mm_round_up(n_prefetch * tile_row_size, 256);
            src0_sz_per_thread               = repacked_vtcm_size;
        }

        src0_sz = src0_sz_per_thread * n_threads;
        dst_sz  = htp_mm_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float)) * n_threads;
    } else {
        const size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
        const size_t dst_nrows = (src1_nrows > 1) ? 0 : 1;

        switch (kernel_type) {
            case HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_src1_row_size = htp_mm_round_up(ne10 * 2, 128);
                src1_sz = htp_mm_round_up(f16_src1_row_size * src1_nrows, 256);
                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                dst_sz  = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                break;
            }
            case HTP_MM_KERNEL_HVX_F16_F32_DDR:
            case HTP_MM_KERNEL_HVX_F16_F16_DDR:
            case HTP_MM_KERNEL_HVX_F32_F32_DDR:
            case HTP_MM_KERNEL_HVX_F32_F16_DDR: {
                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size, 256) * n_threads;
                src1_sz = htp_mm_round_up(n_prefetch * src1_row_size, 256) * n_threads;
                dst_sz  = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                break;
            }
            case HTP_MM_KERNEL_HVX_F32_F32_VTCM: {
                size_t f32_src1_row_size = htp_mm_round_up(ne10 * 4, 128);
                src1_sz = htp_mm_round_up(f32_src1_row_size * src1_nrows, 256);
                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                dst_sz  = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                break;
            }
            case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
            case HTP_MM_KERNEL_HVX_QUANT_ROW: {
                size_t q_src1_row_size = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
                src1_sz = htp_mm_round_up(q_src1_row_size * src1_nrows, 256);

                src0_sz = src0_sz * n_threads;

                if (is_repack) {
                    uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
                    uint32_t n_k_tiles = ne10 / 32;
                    uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
                    size_t repacked_vtcm_size = htp_mm_round_up(n_prefetch * tile_row_size, 256);
                    src0_sz = repacked_vtcm_size * n_threads;
                }

                size_t quant_scratch_size_per_thread = htp_mm_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
                size_t dst_slice_per_thread = (dst_nrows > 0 && src1_nrows == 1) ? htp_mm_round_up((dst_row_size + n_threads - 1) / n_threads, 128) : 0;
                size_t dst_size_per_thread = (dst_slice_per_thread > quant_scratch_size_per_thread) ? dst_slice_per_thread : quant_scratch_size_per_thread;
                dst_sz = dst_size_per_thread * n_threads;
                break;
            }
            case HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT: {
                size_t q_src1_row_size = (wtype == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

                src0_sz = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256);
                src1_sz = htp_mm_round_up(q_src1_row_size * src1_nrows, 256);

                src0_sz = src0_sz * n_threads;

                if (is_repack) {
                    uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(wtype);
                    uint32_t n_k_tiles = ne10 / 32;
                    uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
                    size_t repacked_vtcm_size = htp_mm_round_up(n_prefetch * tile_row_size, 256);
                    src0_sz = repacked_vtcm_size * n_threads;
                }

                size_t quant_scratch_size_per_thread = htp_mm_round_up(ne10 * sizeof(float), QK_Q8_0_TILED * sizeof(float));
                size_t dst_slice_per_thread = dst_nrows > 0 ? htp_mm_round_up((dst_row_size + n_threads - 1) / n_threads, 128) : 0;
                size_t dst_size_per_thread = (dst_slice_per_thread > quant_scratch_size_per_thread) ? dst_slice_per_thread : quant_scratch_size_per_thread;
                dst_sz = dst_size_per_thread * n_threads;
                break;
            }
            default:
                break;
        }
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_sz);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_sz);
    VTCM_LAYOUT_ALLOC(off, off_src3, src3_sz);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_sz);

    L->src0_bytes = src0_sz;
    L->src1_bytes = src1_sz;
    L->src2_bytes = src2_sz;
    L->src3_bytes = src3_sz;
    L->dst_bytes  = dst_sz;
    L->total_bytes = off;
}

#ifdef __cplusplus
}
#endif

#endif // HTP_MATMUL_OPS_H
