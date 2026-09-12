#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <HAP_compute_res.h>

#include <math.h>
#include <string.h>
#include <stdatomic.h>

#include "hex-dma.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
#include "hvx-arith.h"
#include "hvx-reduce.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "matmul-ops.h"
#include "htp-vtcm.h"

static void hvx_tensor_add_f32_grid(
    const struct htp_tensor * restrict dst,
    const struct htp_tensor * restrict src2,
    uint32_t start_row,
    uint32_t end_row,
    uint32_t start_col,
    uint32_t end_col,
    const struct fastdiv_values * div_ne11_12,
    const struct fastdiv_values * div_ne11
);

struct htp_mm_context {
    const char * type;
    struct htp_ops_context * octx;
    const struct htp_tensor * act;

    void (*vec_dot_1x1)(const uint32_t n, float * restrict s0,
         const void * restrict vx0,
         const void * restrict vy0);

    void (*vec_dot_2x1)(const uint32_t n, float * restrict s0,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0);

    void (*vec_dot_2x2)(const uint32_t n, float * restrict s0, float * restrict s1,
         const void * restrict vx0, const void * restrict vx1,
         const void * restrict vy0, const void * restrict vy1);

    void (*vec_dot_32x1)(const uint32_t n, float * restrict s,
         const void * restrict vx,
         const void * restrict vy, uint32_t valid_rows,
         const float * restrict sz);

    // Precomputed values
    uint32_t src0_nrows_per_thread;
    uint32_t src0_row_size_padded;
    uint32_t src1_nrows;

    struct fastdiv_values mm_div_ne12_ne1;
    struct fastdiv_values mm_div_ne1;
    struct fastdiv_values mm_div_r2;
    struct fastdiv_values mm_div_r3;
    struct fastdiv_values mm_div_ne11;

    // Per thread quant tasks
    // Precomputed block-parallel quantization values
    worker_callback_t quant_task_func;
    uint32_t          quant_ib_first[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_ib_last[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_r[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          quant_c[WORK_QUEUE_MAX_N_THREADS];
    uint32_t          n_quant_tasks;
    uint32_t          n_quant_rows_per_thread;
    atomic_uint       quant_barrier;

    // Fields for scattered mapping in MUL_MAT_ID
    const uint32_t * matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows;
    uint32_t mapping_stride;

    // Dynamic VTCM pointers allocated sequentially
    uint8_t * vtcm_src0;
    uint8_t * vtcm_src1;
    uint8_t * vtcm_src2;
    uint8_t * vtcm_src3;
    uint8_t * vtcm_dst;

    // Cached strides
    uint32_t vtcm_src0_stride;
    uint32_t vtcm_src1_stride;
    uint32_t vtcm_src2_stride;
    uint32_t vtcm_src3_stride;

    // Cached thread offsets/sizes
    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_src2_size_per_thread;
    uint32_t vtcm_src3_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;
};

// vdelta control to expand first 32 e8m0 values into 32 uint32 elements
static const uint8_t __attribute__((aligned(128))) expand_x32_e8m0[128] = {
    0x00, 0x00, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00, 0x00,
    0x00, 0x11, 0x10, 0x10, 0x10, 0x02, 0x00, 0x04, 0x00, 0x01, 0x02, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04,
    0x00, 0x00, 0x22, 0x20, 0x20, 0x20, 0x21, 0x22, 0x20, 0x24, 0x04, 0x00, 0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x02,
    0x00, 0x04, 0x00, 0x11, 0x12, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x01, 0x04, 0x00, 0x00, 0x02, 0x00, 0x08, 0x08,
    0x01, 0x02, 0x00, 0x04, 0x44, 0x40, 0x40, 0x40, 0x41, 0x40, 0x40, 0x40, 0x42, 0x40, 0x44, 0x40, 0x41, 0x42, 0x48,
    0x48, 0x08, 0x08, 0x00, 0x00, 0x01, 0x04, 0x00, 0x00, 0x12, 0x10, 0x10, 0x10, 0x01, 0x02, 0x00, 0x04, 0x04, 0x00,
    0x00, 0x00, 0x09, 0x08, 0x00, 0x00, 0x22, 0x20, 0x24, 0x20, 0x21, 0x22, 0x20, 0x20,
};

// IQ4_NL dequantization LUT: maps 4-bit index (0-15) to int8 kvalue
// kvalues: -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
static const uint8_t __attribute__((aligned(VLEN))) kvalues_iq4nl_lut[] = {
    0x81, 0, 0x98, 0, 0xAD, 0, 0xBF, 0, 0xCF, 0, 0xDD, 0, 0xEA, 0, 0xF6, 0, 0x01, 0, 0x0D, 0, 0x19, 0, 0x26, 0,
    0x35, 0, 0x45, 0, 0x59, 0, 0x71, 0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0, 0,    0,
};

static const uint8_t __attribute__((aligned(VLEN))) kvalues_mxfp4_lut[] = {
    0,    0, 1,    0, 2,    0, 3, 0, 4, 0, 6, 0, 8, 0, 12, 0, 0, 0, 0xff, 0, 0xfe, 0, 0xfd, 0, 0xfc, 0,
    0xfa, 0, 0xf8, 0, 0xf4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0, 0,    0,
    0,    0, 0,    0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,    0, 0,    0, 0,    0,
};

#define htp_matmul_tensors_preamble                                 \
    const struct htp_tensor * restrict src0 = octx->src[0];         \
    const struct htp_tensor * restrict src1 = octx->src[1];         \
    const struct htp_tensor * restrict src2 = octx->src[2];         \
    const struct htp_tensor * restrict  dst = octx->dst;            \
                                                                    \
    const uint32_t ne00 = src0->ne[0];                              \
    const uint32_t ne01 = src0->ne[1];                              \
    const uint32_t ne02 = src0->ne[2];                              \
    const uint32_t ne03 = src0->ne[3];                              \
                                                                    \
    const uint32_t ne10 = src1->ne[0];                              \
    const uint32_t ne11 = src1->ne[1];                              \
    const uint32_t ne12 = src1->ne[2];                              \
    const uint32_t ne13 = src1->ne[3];                              \
                                                                    \
    const uint32_t ne20 = src2 ? src2->ne[0] : 0;                   \
    const uint32_t ne21 = src2 ? src2->ne[1] : 0;                   \
    const uint32_t ne22 = src2 ? src2->ne[2] : 0;                   \
    const uint32_t ne23 = src2 ? src2->ne[3] : 0;                   \
                                                                    \
    const uint32_t ne0 = dst->ne[0];                                \
    const uint32_t ne1 = dst->ne[1];                                \
    const uint32_t ne2 = dst->ne[2];                                \
    const uint32_t ne3 = dst->ne[3];                                \
                                                                    \
    const uint32_t nb00 = src0->nb[0];                              \
    const uint32_t nb01 = src0->nb[1];                              \
    const uint32_t nb02 = src0->nb[2];                              \
    const uint32_t nb03 = src0->nb[3];                              \
                                                                    \
    const uint32_t nb10 = src1->nb[0];                              \
    const uint32_t nb11 = src1->nb[1];                              \
    const uint32_t nb12 = src1->nb[2];                              \
    const uint32_t nb13 = src1->nb[3];                              \
                                                                    \
    const uint32_t nb0 = dst->nb[0];                                \
    const uint32_t nb1 = dst->nb[1];                                \
    const uint32_t nb2 = dst->nb[2];                                \
    const uint32_t nb3 = dst->nb[3];

#define htp_matmul_preamble                                         \
    struct htp_mm_context * mmctx  = data;                          \
    struct htp_ops_context * octx  = mmctx->octx;                   \
    dma_queue *dma_queue           = octx->ctx->dma[ith];           \
    uint32_t src0_nrows_per_thread = mmctx->src0_nrows_per_thread;  \
    htp_matmul_tensors_preamble;

static inline void hvx_mm_run_quant_task(struct htp_mm_context * mmctx, unsigned int ith) {
    if (mmctx->quant_task_func) {
        if (ith < mmctx->n_quant_tasks) {
            mmctx->quant_task_func(mmctx->n_quant_tasks, ith, mmctx);
            atomic_fetch_sub(&mmctx->quant_barrier, 1);
        }
        while (atomic_load(&mmctx->quant_barrier) > 0) {
            // spin
        }
    }
}

// *** matmul with support for 4d tensors and full broadcasting

static void hvx_mm_4d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const uint32_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const uint32_t nr1 = ne1 * ne2 * ne3;

    // distribute the thread work across the inner or outer loop based on which one is larger
    uint32_t dr0, dr1, ith0, ith1;
    if (nr0 > nr1) {
        dr0  = fastdiv(nr0 + nth - 1, &octx->ctx->n_threads_div);
        dr1  = nr1;
        ith0 = ith;
        ith1 = 0;
    } else {
        dr0  = nr0;
        dr1  = fastdiv(nr1 + nth - 1, &octx->ctx->n_threads_div);
        ith0 = 0;
        ith1 = ith;
    }

    const uint32_t ir0_start = dr0 * ith0;
    const uint32_t ir0_end   = MIN(ir0_start + dr0, nr0);

    const uint32_t ir1_start = dr1 * ith1;
    const uint32_t ir1_end   = MIN(ir1_start + dr1, nr1);

    // no work for this thread
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0_start);

    const uint32_t blck_0 = 64;
    const uint32_t blck_1 = 64;

    for (uint32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (uint32_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (uint32_t ir1 = iir1; ir1 < MIN(iir1 + blck_1, ir1_end); ir1++) {
                const uint32_t i13 = fastdiv(ir1, &mmctx->mm_div_ne12_ne1);
                const uint32_t i12 = fastdiv(ir1 - i13 * ne12 * ne1, &mmctx->mm_div_ne1);
                const uint32_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const uint32_t i03 = fastdiv(i13, &mmctx->mm_div_r3);
                const uint32_t i02 = fastdiv(i12, &mmctx->mm_div_r2);

                const uint32_t i1 = i11;
                const uint32_t i2 = i12;
                const uint32_t i3 = i13;

                const uint8_t * restrict src0_base = (const uint8_t *) src0->data + (0 + i02 * nb02 + i03 * nb03);
                const uint8_t * restrict src1_col  = (const uint8_t *) src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13);
                float * dst_col = (float *) ((uint8_t * restrict) dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                const uint32_t ir0_block_end = MIN(iir0 + blck_0, ir0_end);
                for (uint32_t ir0 = iir0; ir0 < ir0_block_end; ir0++) {
                    const uint8_t * restrict src0_row = src0_base + ir0 * nb01;
                    mmctx->vec_dot_1x1(ne00, &dst_col[ir0], src0_row, src1_col);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0_start);
    if (src2) {
        hvx_tensor_add_f32_grid(dst, src2, ir1_start, ir1_end, ir0_start, ir0_end, &mmctx->mm_div_ne12_ne1, &mmctx->mm_div_ne1);
    }
}

#include "hvx-transfer-kernels.h"
#include "hvx-mm-kernels-tiled.h"
#include "hvx-mm-kernels-flat.h"

// Specialized repacked matmul macros
#define MATMUL_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                              \
static void hvx_mm_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                        \
    htp_matmul_preamble;                                                                                                          \
                                                                                                                                  \
    const uint32_t src0_nrows = ne01 * ne02 * ne03;                                                                               \
    const uint32_t src1_nrows = ne11 * ne12 * ne13;                                                                               \
                                                                                                                                  \
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;                                                                 \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                                     \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const size_t dst_row_size  = nb1;                                                                                             \
    const size_t src1_row_size = nb11;                                                                                            \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
    const size_t src2_stride = src2 ? ((src2->ne[1] == 1) ? 0 : src2->nb[1]) : 0;                                                 \
                                                                                                                                  \
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;                                 \
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                 \
    uint8_t * restrict src1_data = mmctx->vtcm_src1;                                                                              \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    uint32_t ct_start = src0_start_row / 32;                                                                                      \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                 \
                                                                                                                                  \
    uint32_t push_ct = ct_start;                                                                                                  \
    if (src0_start_row < src0_end_row) {                                                                                          \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                            \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    hvx_mm_run_quant_task(mmctx, ith);                                                                                            \
                                                                                                                                  \
    if (src0_start_row >= src0_end_row) {                                                                                         \
        return;                                                                                                                   \
    }                                                                                                                             \
                                                                                                                                  \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                             \
        const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                    \
                                                                                                                                  \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                               \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                 \
                                                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
        uint32_t ir1 = 0;                                                                                                         \
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                                  \
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                           \
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                           \
            float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                         \
            float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                         \
                                                                                                                                  \
            float * dst_ptr0 = &dst_row0[ct * 32];                                                                                \
            float * dst_ptr1 = &dst_row1[ct * 32];                                                                                \
                                                                                                                                  \
            const float * src2_ptr0 = NULL;                                                                                       \
            const float * src2_ptr1 = NULL;                                                                                       \
            if (src2) {                                                                                                           \
                const float * restrict src2_row0 = (const float *) ((const uint8_t *) src2->data + ((ir1+0) * src2_stride));      \
                const float * restrict src2_row1 = (const float *) ((const uint8_t *) src2->data + ((ir1+1) * src2_stride));      \
                src2_ptr0 = &src2_row0[ct * 32];                                                                                  \
                src2_ptr1 = &src2_row1[ct * 32];                                                                                  \
            }                                                                                                                     \
            DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, src2_ptr0, src2_ptr1);                    \
        }                                                                                                                         \
                                                                                                                                  \
        for (; ir1 < src1_nrows; ++ir1) {                                                                                         \
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                                \
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));                                     \
            float * dst_ptr = &dst_row[ct * 32];                                                                                  \
                                                                                                                                  \
            const float * src2_ptr = NULL;                                                                                        \
            if (src2) {                                                                                                           \
                const float * restrict src2_row = (const float *) ((const uint8_t *) src2->data + (ir1 * src2_stride));           \
                src2_ptr = &src2_row[ct * 32];                                                                                    \
            }                                                                                                                     \
            DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, src2_ptr);                                                       \
        }                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                     \
                                                                                                                                  \
        if (push_ct < ct_end) {                                                                                                   \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),                      \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            push_ct++;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
}

#define MATVEC_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X1)                                                                       \
static void hvx_mv_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                        \
    htp_matmul_preamble;                                                                                                          \
                                                                                                                                  \
    const uint32_t src0_nrows = ne01;                                                                                             \
                                                                                                                                  \
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;                                                                 \
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);                                     \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
                                                                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const size_t dst_row_size  = nb1;                                                                                             \
    const size_t src1_row_size = nb11;                                                                                            \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst + mmctx->vtcm_dst_size_per_thread * ith;                                            \
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                                          \
    uint8_t * src1_data = mmctx->vtcm_src1;                                                                                       \
                                                                                                                                  \
    float * tmp = (float *) vtcm_dst_ptr;                                                                                         \
                                                                                                                                  \
    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;                                                             \
                                                                                                                                  \
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;                                                              \
    float * restrict dst_col          = (float *) dst->data;                                                                      \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
                                                                                                                                  \
    uint32_t n_k_tiles_w = ne00 / 32;                                                                                             \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                           \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    uint32_t ct_start = src0_start_row / 32;                                                                                      \
    uint32_t ct_end   = (src0_end_row + 31) / 32;                                                                                 \
                                                                                                                                  \
    uint32_t push_ct = ct_start;                                                                                                  \
    if (src0_start_row < src0_end_row) {                                                                                          \
        if (src2) {                                                                                                               \
            float * vtcm_src2_ptr = (float *) mmctx->vtcm_src2 + src0_start_row;                                                  \
            const float * src2_ptr = (const float *) src2->data + src0_start_row;                                                 \
            int slice_size = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                                   \
            if (slice_size > 0) {                                                                                                 \
                dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr, src2_ptr),                                                  \
                               slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);            \
                dma_queue_pop_nowait(dma_queue);                                                                                  \
            }                                                                                                                     \
        }                                                                                                                         \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned,                            \
                           src0_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);          \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    hvx_mm_run_quant_task(mmctx, ith);                                                                                            \
                                                                                                                                  \
    if (src0_start_row >= src0_end_row) {                                                                                         \
        return;                                                                                                                   \
    }                                                                                                                             \
                                                                                                                                  \
    for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                             \
        const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                    \
                                                                                                                                  \
        float * dst_ptr = &tmp[ct * 32 - src0_start_row];                                                                         \
        int valid_rows = (int)ne0 - (int)(ct * 32);                                                                               \
        valid_rows = MIN(32, MAX(0, valid_rows));                                                                                 \
                                                                                                                                  \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                    \
        DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                               \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                     \
                                                                                                                                  \
        if (push_ct < ct_end) {                                                                                                   \
            dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),                      \
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                                 \
            push_ct++;                                                                                                            \
        }                                                                                                                         \
    }                                                                                                                             \
                                                                                                                                  \
    int copy_cnt = (int)MIN(src0_end_row, ne0) - (int)src0_start_row;                                                             \
    if (copy_cnt > 0) {                                                                                                           \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                                \
        if (src2) {                                                                                                               \
            hvx_add_f32_uaa((uint8_t *) &dst_col[src0_start_row],                                                                 \
                            (const uint8_t *) tmp,                                                                                \
                            (const uint8_t *) ((const float *) mmctx->vtcm_src2 + src0_start_row),                                \
                            copy_cnt);                                                                                            \
        } else {                                                                                                                  \
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);                                     \
        }                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct_end);                                                                 \
    }                                                                                                                             \
}

#define MATMUL_NX_2D_REPACKED_IMPL(SUFFIX, TILE_SIZE, DOT_2X2, DOT_2X1)                                                           \
static void hvx_mm_nx_2d_repacked_##SUFFIX(unsigned int nth, unsigned int ith, void * data) {                                     \
    struct htp_mm_context * mmctx = data;                                                                                         \
    struct htp_ops_context * octx = mmctx->octx;                                                                                  \
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;                      \
    const uint32_t n_weights = kparams->n_weights;                                                                                \
                                                                                                                                  \
    const struct htp_tensor * restrict act = octx->src[n_weights]; /* x */                                                        \
    const uint32_t ne10 = act->ne[0];                                                                                             \
    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];                                                             \
    const size_t src1_stride = mmctx->vtcm_src1_stride;                                                                           \
                                                                                                                                  \
    uint8_t * restrict vtcm_weight_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;                               \
    uint8_t * restrict src1_data       = mmctx->vtcm_src1;                                                                        \
                                                                                                                                  \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                        \
    const uint32_t n_prefetch = kparams->n_prefetch;                                                                              \
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);                         \
                                                                                                                                  \
    const uint32_t tile_size = TILE_SIZE;                                                                                         \
    const uint32_t aligned_tile_size = hex_align_up(tile_size, 128);                                                              \
    uint32_t n_k_tiles_a = ne10 / 32;                                                                                             \
    uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;                                                    \
                                                                                                                                  \
    dma_queue * dma_queue = octx->ctx->dma[ith];                                                                                  \
                                                                                                                                  \
    hvx_mm_run_quant_task(mmctx, ith);                                                                                            \
                                                                                                                                  \
    for (uint32_t widx = 0; widx < n_weights; widx++) {                                                                           \
        const struct htp_tensor * restrict src_w = octx->src[widx];                                                               \
        const struct htp_tensor * restrict dst   = octx->dsts[widx];                                                              \
        if (!src_w || !dst) continue;                                                                                             \
                                                                                                                                  \
        const uint32_t ne00 = src_w->ne[0];                                                                                       \
        const uint32_t ne01 = src_w->ne[1];                                                                                       \
        const size_t dst_row_size = dst->nb[1];                                                                                   \
        const uint8_t * restrict src_w_row = (const uint8_t *) src_w->data;                                                       \
                                                                                                                                  \
        uint32_t n_k_tiles_w = ne00 / 32;                                                                                         \
        uint32_t tile_row_stride = n_k_tiles_w * tile_size;                                                                       \
                                                                                                                                  \
        const uint32_t src0_nrows = ne01 * src_w->ne[2] * src_w->ne[3];                                                           \
        uint32_t src0_nrows_per_thread = fastdiv(src0_nrows + nth - 1, &octx->ctx->n_threads_div);                                \
        src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);                                                          \
                                                                                                                                  \
        const uint32_t start_row = src0_nrows_per_thread * ith;                                                                   \
        const uint32_t end_row   = MIN(start_row + src0_nrows_per_thread, src0_nrows);                                            \
        if (start_row >= end_row) continue;                                                                                       \
                                                                                                                                  \
        uint32_t ct_start = start_row / 32;                                                                                       \
        uint32_t ct_end   = (end_row + 31) / 32;                                                                                  \
                                                                                                                                  \
        uint32_t push_ct = ct_start;                                                                                              \
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {                                                \
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_weight_ptr + d * tile_row_transfer_size_aligned,                          \
                           src_w_row + push_ct * tile_row_stride), aligned_tile_size, tile_size, tile_size, n_k_tiles_a);         \
        }                                                                                                                         \
                                                                                                                                  \
        for (uint32_t ct = ct_start; ct < ct_end; ct++) {                                                                         \
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;                                                                \
            int valid_rows = (int)ne01 - (int)(ct * 32);                                                                          \
            valid_rows = MIN(32, MAX(0, valid_rows));                                                                             \
                                                                                                                                  \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                \
            uint32_t ir1 = 0;                                                                                                     \
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {                                                                              \
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);                       \
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);                       \
                                                                                                                                  \
                float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));                                     \
                float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));                                     \
                float * dst_ptr0 = &dst_row0[ct * 32];                                                                            \
                float * dst_ptr1 = &dst_row1[ct * 32];                                                                            \
                                                                                                                                  \
                DOT_2X2(ne10, dst_ptr0, dst_ptr1, w_tile, src1_col0, src1_col1, valid_rows, NULL, NULL);                          \
            }                                                                                                                     \
                                                                                                                                  \
            for (; ir1 < src1_nrows; ++ir1) {                                                                                     \
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);                            \
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));                                          \
                float * dst_ptr = &dst_row[ct * 32];                                                                              \
                DOT_2X1(ne10, dst_ptr, w_tile, src1_col, valid_rows, NULL);                                                       \
            }                                                                                                                     \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);                                                                 \
                                                                                                                                  \
            if (push_ct < ct_end) {                                                                                               \
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src_w_row + push_ct * tile_row_stride),                 \
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);                                             \
                push_ct++;                                                                                                        \
            }                                                                                                                     \
        }                                                                                                                         \
    }                                                                                                                             \
}

MATMUL_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,   flat_vec_dot_q4_0_32x1)
MATMUL_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,   flat_vec_dot_q4_1_32x1)
MATMUL_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,   flat_vec_dot_q8_0_32x1)
MATMUL_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2,  flat_vec_dot_iq4nl_32x1)
MATMUL_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2,  flat_vec_dot_mxfp4_32x1)

#define QUANTIZE_IMPL(name, log_name, kernel_fn, dst_row_size_expr)                                        \
static void name(unsigned int nth, unsigned int ith, void * data) {                                        \
    struct htp_mm_context * mmctx = data;                                                                  \
    struct htp_ops_context * octx = mmctx->octx;                                                           \
    const struct htp_tensor * src = mmctx->act;                                                            \
    const uint32_t ne0 = src->ne[0];                                                                       \
    const uint32_t ne1 = src->ne[1];                                                                       \
    const uint32_t ne2 = src->ne[2];                                                                       \
    const uint32_t ne3 = src->ne[3];                                                                       \
    const uint32_t nrows = ne1 * ne2 * ne3;                                                                \
    const uint32_t nrows_per_thread = mmctx->n_quant_rows_per_thread;                                      \
                                                                                                           \
    const uint32_t ir_first = nrows_per_thread * ith;                                                      \
    if (ir_first >= nrows) {                                                                               \
        return;                                                                                            \
    }                                                                                                      \
                                                                                                           \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                                 \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                        \
                                                                                                           \
    uint8_t * restrict dst = mmctx->vtcm_src1;                                                             \
    const uint32_t ir_last = MIN(ir_first + nrows_per_thread, nrows);                                      \
    const size_t src_row_size = src->nb[1];                                                                \
    const size_t dst_row_size = (dst_row_size_expr);                                                       \
    const uint8_t * restrict src_data = (const uint8_t *) src->data + (src_row_size * ir_first);           \
    uint8_t * restrict dst_data = (uint8_t *) dst + (dst_row_size * ir_first);                             \
    uint8_t * restrict tmp_data = (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith);   \
    kernel_fn(src_data, dst_data, tmp_data, ne0, ir_last - ir_first, src_row_size, dst_row_size);          \
                                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, ir_first);                                         \
}

QUANTIZE_IMPL(quantize_f32_q8_0_tiled, "quantize-f32-q8_0_tiled", quantize_f32_q8_0_tiled_kernel, htp_mm_q8_0_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_tiled, "quantize-f32-q8_1_tiled", quantize_f32_q8_1_tiled_kernel, htp_mm_q8_1_tiled_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_0_flat,  "quantize-f32-q8_0_flat",  quantize_f32_q8_0_flat_kernel,  htp_mm_q8_0_flat_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_q8_1_flat,  "quantize-f32-q8_1_flat",  quantize_f32_q8_1_flat_kernel,  htp_mm_q8_1_flat_row_size(ne0))
QUANTIZE_IMPL(quantize_f32_f32_flat,   "quantize-f32-f32",        quantize_f32_f32_flat_kernel,   mmctx->vtcm_src1_stride)
QUANTIZE_IMPL(quantize_f32_f16_flat,   "quantize-f32-f16",        quantize_f32_f16_flat_kernel,   mmctx->vtcm_src1_stride)
QUANTIZE_IMPL(quantize_f16_f16_flat,   "quantize-f16-f16",        quantize_f16_f16_flat_kernel,   mmctx->vtcm_src1_stride)

static void quantize_f32_q8_0_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_q8_0_tiled_block_kernel(
        (const float *) src->data,
        mmctx->vtcm_src1,
        (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith),
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        src->nb[1],
        htp_mm_q8_0_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

static void quantize_f32_q8_1_tiled_block(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);

    const struct htp_tensor * src = mmctx->act;

    quantize_f32_q8_1_tiled_block_kernel(
        (const float *) src->data,
        mmctx->vtcm_src1,
        (uint8_t *) mmctx->vtcm_dst + (mmctx->vtcm_dst_size_per_thread * ith),
        src->ne[0],
        mmctx->quant_ib_first[ith],
        mmctx->quant_ib_last[ith],
        src->nb[1],
        htp_mm_q8_1_tiled_row_size(src->ne[0]),
        mmctx->quant_r[ith],
        mmctx->quant_c[ith]
    );

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, mmctx->quant_ib_first[ith]);
}

MATVEC_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x1)

MATVEC_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x1)
MATVEC_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x1)
MATVEC_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x1)
MATVEC_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x1)
MATVEC_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x1)


MATMUL_NX_2D_REPACKED_IMPL(q4_0,       576,  tiled_vec_dot_q4_0_32x2,  tiled_vec_dot_q4_0_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q4_1,       640,  tiled_vec_dot_q4_1_32x2,  tiled_vec_dot_q4_1_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q8_0,       1088, tiled_vec_dot_q8_0_32x2,  tiled_vec_dot_q8_0_32x1)
MATMUL_NX_2D_REPACKED_IMPL(iq4nl,      576,  tiled_vec_dot_iq4nl_32x2, tiled_vec_dot_iq4nl_32x1)
MATMUL_NX_2D_REPACKED_IMPL(mxfp4,      544,  tiled_vec_dot_mxfp4_32x2, tiled_vec_dot_mxfp4_32x1)

MATMUL_NX_2D_REPACKED_IMPL(q4_0_flat,  576,  flat_vec_dot_q4_0_32x2,   flat_vec_dot_q4_0_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q4_1_flat,  640,  flat_vec_dot_q4_1_32x2,   flat_vec_dot_q4_1_32x1)
MATMUL_NX_2D_REPACKED_IMPL(q8_0_flat,  1088, flat_vec_dot_q8_0_32x2,   flat_vec_dot_q8_0_32x1)
MATMUL_NX_2D_REPACKED_IMPL(iq4nl_flat, 576,  flat_vec_dot_iq4nl_32x2,  flat_vec_dot_iq4nl_32x1)
MATMUL_NX_2D_REPACKED_IMPL(mxfp4_flat, 544,  flat_vec_dot_mxfp4_32x2,  flat_vec_dot_mxfp4_32x1)

static void hvx_mm_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;  // src0 rows
    const uint32_t src1_nrows = ne11 * ne12 * ne13;  // src1 rows

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data     = mmctx->vtcm_src1;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;

    // Prefill vtcm with src0 rows
    if (src0_start_row < src0_end_row) {
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= (int)n_prefetch) {
                break;
            }
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        // Process src1 columns in pairs (2×2 tiling)
        uint32_t ir1 = 0;
        for (; ir1 + 1 < src1_nrows; ir1 += 2) {
            const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
            const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);
            float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));
            float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));
            mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
        }

        // Handle remaining src1 rows (fallback to 2×1)
        for (; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const int pr0 = (ir0 + n_prefetch);
        const int is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        uint32_t  ir0 = src0_end_row_x2;
        const int is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        #pragma unroll(2)
        for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
            const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
            float * restrict dst_row          = (float *) (dst->data + (ir1 * dst_row_size));
            mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }
    if (src2) {
        hvx_tensor_add_f32_grid(dst, src2, 0, src1_nrows, src0_start_row, src0_end_row, &kparams->div_ne12_ne1, &kparams->div_ne1);
    }
}

static void hvx_mv_2d(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const uint32_t src0_nrows = ne01;

    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const size_t dst_row_size  = nb1;
    const size_t src0_row_size = nb01;
    const size_t src1_row_size = nb11;

    const size_t src0_stride = mmctx->vtcm_src0_stride;
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * vtcm_dst_ptr  = mmctx->vtcm_dst  + mmctx->vtcm_dst_size_per_thread  * ith;
    uint8_t * vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * src1_data     = mmctx->vtcm_src1;

    float * tmp = (float *) vtcm_dst_ptr;

    const uint8_t * restrict src0_row = (const uint8_t *) src0->data;
    const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
    float * restrict dst_col          = (float *) dst->data;

    const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    // Prefill vtcm with 2x src0 rows
    if (src0_start_row < src0_end_row) {
        if (src2) {
            float * vtcm_src2_ptr = (float *) mmctx->vtcm_src2 + src0_start_row;
            const float * src2_ptr = (const float *) src2->data + src0_start_row;
            int slice_size = (int)src0_end_row - (int)src0_start_row;
            if (slice_size > 0) {
                dma_queue_push(dma_queue, dma_make_ptr(vtcm_src2_ptr, src2_ptr),
                               slice_size * sizeof(float), slice_size * sizeof(float), slice_size * sizeof(float), 1);
                dma_queue_pop_nowait(dma_queue);
            }
        }
        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint32_t is0 = (ir0 - src0_start_row);
            if (is0 >= n_prefetch) {
                break;
            }
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    // Process src0 rows
    for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_2x1(ne00, &tmp[ir0 - src0_start_row], ss0, ss0 + src0_stride, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

        // Prefetch next (n + vtcm_nrows) row
        const uint32_t pr0 = (ir0 + n_prefetch);
        const uint32_t is0 = (pr0 - src0_start_row) & prefetch_mask;
        if (pr0 < src0_end_row_x2) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }
    }

    // Process the last row (if any)
    if (src0_end_row != src0_end_row_x2) {
        const uint32_t ir0 = src0_end_row_x2;
        const uint32_t is0 = (ir0 - src0_start_row) & prefetch_mask;
        dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                       src0_stride, src0_row_size, src0_row_size, 1);
        const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        mmctx->vec_dot_1x1(ne00, &tmp[ir0 - src0_start_row], ss0, src1_col);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
    }

    int copy_cnt = src0_end_row - src0_start_row;
    if (copy_cnt > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
        if (src2) {
            hvx_add_f32_uaa((uint8_t *) &dst_col[src0_start_row],
                            (const uint8_t *) tmp,
                            (const uint8_t *) ((const float *) mmctx->vtcm_src2 + src0_start_row),
                            copy_cnt);
        } else {
            hvx_copy_f32_ua((uint8_t *) &dst_col[src0_start_row], (uint8_t *) tmp, copy_cnt);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, src0_end_row);
    }
}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * mmctx->mapping_stride + (i1)]

static void hvx_mm_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint32_t src0_nrows      = ne01;  // src0 rows per expert
    const uint32_t src1_nrows      = ne11;
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t n_ids = ids->ne[0];  // n_expert_used
    const uint32_t n_as  = ne02;        // n_expert

    const uint32_t *                matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows       = mmctx->matrix_rows;

    const size_t dst_row_size  = nb1;
    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const size_t src1_stride = mmctx->vtcm_src1_stride;

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) {
            continue;
        }

        const uint8_t * src0_row = (const uint8_t *) src0->data + cur_a * nb02;

        const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            for (uint32_t cid = 0; cid < cne1; ++cid) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                const int               rm1         = row_mapping.i1;  // expert idx
                const int               rm2         = row_mapping.i2;  // token idx

                const uint32_t ir1 = fastmodulo(rm1, ne11, &mmctx->mm_div_ne11);        // src1 row idx
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * ne11 + 0) * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (rm1 * nb1 + rm2 * nb2 + 0));

                mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }
}

static void hvx_mv_id(unsigned int nth, unsigned int ith, void * data) {
    htp_matmul_preamble;

    const struct htp_tensor * restrict ids = octx->src[2];

    const uint32_t src0_nrows      = ne01;  // src0 rows per expert
    const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    hvx_mm_run_quant_task(mmctx, ith);

    if (src0_start_row >= src0_end_row) {
        return;
    }

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    assert(ne13 % ne03 == 0);

    const size_t dst_row_size  = nb1;
    const size_t src1_row_size = htp_mm_q8_0_tiled_row_size(ne10);

    const uint32_t n_aids = src2->ne[0];  // num activated experts
    const uint32_t n_ids  = ne02;         // num experts

    // Per-thread VTCMs for all tensors
    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {  // for each expert
        const int32_t eid = *(const int32_t *) ((const uint8_t *) src2->data + ie1 * src2->nb[0]);
        if (eid < 0) {
            continue;
        }
        assert(eid < (int32_t) n_ids);

        const uint8_t * restrict src0_row = (const uint8_t *) src0->data + eid * nb02;
        const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
        float * restrict dst_row          = (float *) (dst->data + ie1 * nb1);

        const uint32_t tile_size = htp_mm_get_weight_tile_size(src0->type);
        const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src0->type);
        const uint32_t n_k_tiles_w = ne00 / 32;
        const uint32_t n_k_tiles_a = ne10 / 32;
        const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
        const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

        const uint32_t ct_start = src0_start_row / 32;
        const uint32_t ct_end   = (src0_end_row + 31) / 32;

        uint32_t push_ct = ct_start;
        for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                           aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
        }

        for (uint32_t ct = ct_start; ct < ct_end; ct++) {
            const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

            int valid_rows = (int)ne01 - (int)(ct * 32);
            valid_rows = MIN(32, MAX(0, valid_rows));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
            mmctx->vec_dot_32x1(ne10, &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

            if (push_ct < ct_end) {
                dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                push_ct++;
            }
        }
    }
}

static void hvx_mv_id_nx(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = (struct htp_mm_context *) data;
    struct htp_ops_context * octx = mmctx->octx;
    dma_queue * dma_queue         = octx->ctx->dma[ith];
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights      = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];

    hvx_mm_run_quant_task(mmctx, ith);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t n_aids = ids->ne[0];
    const uint32_t n_ids  = src0->ne[2];

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t ie1 = 0; ie1 < n_aids; ++ie1) {
        const int32_t eid = *(const int32_t *) ((const uint8_t *) ids->data + ie1 * ids->nb[0]);
        if (eid < 0) continue;
        assert(eid < (int32_t) n_ids);

        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;

            const uint32_t src0_nrows = src_w->ne[1];
            uint32_t src0_nrows_per_thread = fastdiv(src0_nrows + nth - 1, &octx->ctx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            const uint32_t src0_start_row = src0_nrows_per_thread * ith;
            const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
            if (src0_start_row >= src0_end_row) continue;

            const uint8_t * restrict src0_row = (const uint8_t *) src_w->data + eid * src_w->nb[2];
            const uint8_t * restrict src1_col = (const uint8_t *) src1_data;
            float * restrict dst_row = (float *) (dst->data + ie1 * dst->nb[1]);

            const uint32_t tile_size = htp_mm_get_weight_tile_size(src_w->type);
            const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src_w->type);
            const uint32_t n_k_tiles_w = src_w->ne[0] / 32;
            const uint32_t n_k_tiles_a = act->ne[0] / 32;
            const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

            const uint32_t ct_start = src0_start_row / 32;
            const uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t push_ct = ct_start;
            for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
                dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
            }

            for (uint32_t ct = ct_start; ct < ct_end; ct++) {
                const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

                int valid_rows = (int)src_w->ne[1] - (int)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (push_ct < ct_end) {
                    dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                                   aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                    push_ct++;
                }
            }
        }
    }
}

static void hvx_mm_id_nx(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = (struct htp_mm_context *) data;
    struct htp_ops_context * octx = mmctx->octx;
    dma_queue * dma_queue         = octx->ctx->dma[ith];
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights      = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];

    hvx_mm_run_quant_task(mmctx, ith);

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);

    const uint32_t n_as = src0->ne[2];

    const uint32_t * matrix_row_counts = mmctx->matrix_row_counts;
    const struct mmid_row_mapping * matrix_rows = mmctx->matrix_rows;

    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data = mmctx->vtcm_src1;

    for (uint32_t cur_a = 0; cur_a < n_as; ++cur_a) {
        const int32_t cne1 = matrix_row_counts[cur_a];
        if (cne1 == 0) continue;

        for (uint32_t p = 0; p < n_weights; ++p) {
            const struct htp_tensor * restrict src_w = octx->src[p];
            const struct htp_tensor * restrict dst   = octx->dsts[p];
            if (!src_w || !dst) continue;

            const uint32_t src0_nrows = src_w->ne[1];
            uint32_t src0_nrows_per_thread = fastdiv(src0_nrows + nth - 1, &octx->ctx->n_threads_div);
            src0_nrows_per_thread = hex_round_up(src0_nrows_per_thread, 32);

            const uint32_t src0_start_row = src0_nrows_per_thread * ith;
            const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
            if (src0_start_row >= src0_end_row) continue;

            const uint8_t * src0_row = (const uint8_t *) src_w->data + cur_a * src_w->nb[2];

            const uint32_t tile_size = htp_mm_get_weight_tile_size(src_w->type);
            const uint32_t aligned_tile_size = htp_mm_get_weight_aligned_tile_size(src_w->type);
            const uint32_t n_k_tiles_w = src_w->ne[0] / 32;
            const uint32_t n_k_tiles_a = act->ne[0] / 32;
            const uint32_t tile_row_stride = n_k_tiles_w * tile_size;
            const uint32_t tile_row_transfer_size_aligned = n_k_tiles_a * aligned_tile_size;

            const uint32_t ct_start = src0_start_row / 32;
            const uint32_t ct_end   = (src0_end_row + 31) / 32;

            uint32_t push_ct = ct_start;
            for (uint32_t d = 0; d < n_prefetch && push_ct < ct_end; d++, push_ct++) {
                dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + d * tile_row_transfer_size_aligned, src0_row + push_ct * tile_row_stride),
                               aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
            }

            for (uint32_t ct = ct_start; ct < ct_end; ct++) {
                const uint8_t * w_tile = dma_queue_pop(dma_queue).dst;

                int valid_rows = (int)src_w->ne[1] - (int)(ct * 32);
                valid_rows = MIN(32, MAX(0, valid_rows));

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ct);
                for (uint32_t cid = 0; cid < (uint32_t) cne1; ++cid) {
                    struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, cid);
                    const int rm1 = row_mapping.i1;
                    const int rm2 = row_mapping.i2;

                    const uint32_t ir1 = fastmodulo(rm1, act->ne[1], &mmctx->mm_div_ne11);
                    const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + (ir1 + rm2 * act->ne[1]) * src1_stride);
                    float * restrict dst_row = (float *) (dst->data + (rm1 * dst->nb[1] + rm2 * dst->nb[2]));

                    mmctx->vec_dot_32x1(act->ne[0], &dst_row[ct * 32], w_tile, src1_col, valid_rows, NULL);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ct);

                if (push_ct < ct_end) {
                    dma_queue_push(dma_queue, dma_make_ptr((uint8_t *)w_tile, src0_row + push_ct * tile_row_stride),
                                   aligned_tile_size, tile_size, tile_size, n_k_tiles_a);
                    push_ct++;
                }
            }
        }
    }
}

static int hvx_mm_init_vec_dot(struct htp_mm_context * mmctx, enum htp_data_type type) {
    switch (type) {
        case HTP_TYPE_Q4_0:
            mmctx->type         = "q4_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_0_32x1;
            return 0;
        case HTP_TYPE_Q4_1:
            mmctx->type         = "q4_1_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q4_1_32x1;
            return 0;
        case HTP_TYPE_Q8_0:
            mmctx->type         = "q8_0_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_q8_0_32x1;
            return 0;
        case HTP_TYPE_IQ4_NL:
            mmctx->type         = "iq4nl_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_iq4nl_32x1;
            return 0;
        case HTP_TYPE_MXFP4:
            mmctx->type         = "mxfp4_tiled-f32";
            mmctx->vec_dot_32x1 = tiled_vec_dot_mxfp4_32x1;
            return 0;
        default:
            return -1;
    }
}

static int hvx_mm_matmul(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = src1;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const uint32_t src0_nrows = ne01 * ne02 * ne03;
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4);

    // Compute src0_nrows_per_thread
    mmctx->src0_nrows_per_thread  = fastdiv(src0_nrows + octx->n_threads - 1, &octx->ctx->n_threads_div);
    if (is_repacked) {
        mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);
    } else {
        mmctx->src0_nrows_per_thread += (mmctx->src0_nrows_per_thread & 1); // round up to even
    }

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;
    size_t       src1_row_size = nb11;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);
    size_t       src1_row_size_padded;

    worker_callback_t quant_task_func;
    worker_callback_t matmul_job_func;
    uint32_t n_quant_tasks = 1;
    if (src1_nrows > 1) {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mm_2d;
        }
    } else {
        if (is_repacked) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mv_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mv_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mv_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mv_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            matmul_job_func = hvx_mv_2d;
        }
    }

    bool need_quant = true;

    switch (kparams->kernel_type) {
        case HTP_MM_KERNEL_HVX_F16_F16_VTCM:
            quant_task_func        = (src1->type == HTP_TYPE_F32) ? quantize_f32_f16_flat : quantize_f16_f16_flat;
            mmctx->type            = "f16-f16";
            mmctx->vec_dot_1x1     = vec_dot_f16_f16_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f16_f16_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f16_f16_aa_2x2;
            src1_row_size          = hex_round_up(ne10 * 2, 128);
            break;

        case HTP_MM_KERNEL_HVX_F16_F32_DDR:
            mmctx->type            = "f16-f32";
            mmctx->vec_dot_1x1     = vec_dot_f16_f32_uu_1x1;
            matmul_job_func        = hvx_mm_4d;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            need_quant             = false;
            quant_task_func        = NULL;
            src1_row_size          = nb11;
            break;

        case HTP_MM_KERNEL_HVX_F16_F16_DDR:
            mmctx->type            = "f16-f16";
            mmctx->vec_dot_1x1     = vec_dot_f16_f16_uu_1x1;
            matmul_job_func        = hvx_mm_4d;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            src1_row_size          = nb11;
            need_quant             = false;
            quant_task_func        = NULL;
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            quant_task_func        = quantize_f32_f32_flat;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = vec_dot_f32_f32_aa_1x1;
            mmctx->vec_dot_2x1     = vec_dot_f32_f32_aa_2x1;
            mmctx->vec_dot_2x2     = vec_dot_f32_f32_aa_2x2;
            src1_row_size          = hex_round_up(ne10 * 4, 128);
            break;

        case HTP_MM_KERNEL_HVX_F32_F32_DDR:
            quant_task_func        = NULL;
            mmctx->type            = "f32-f32";
            mmctx->vec_dot_1x1     = vec_dot_f32_f32_uu_1x1;
            mmctx->mm_div_ne12_ne1 = kparams->div_ne12_ne1;
            mmctx->mm_div_ne1      = kparams->div_ne1;
            mmctx->mm_div_r2       = kparams->div_r2;
            mmctx->mm_div_r3       = kparams->div_r3;
            src1_row_size          = nb11;
            need_quant             = false;
            matmul_job_func        = hvx_mm_4d;
            break;

        case HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT: {
            n_quant_tasks = MIN(src1_nrows, octx->n_threads);
            quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
            src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(ne10) : htp_mm_q8_0_flat_row_size(ne10);

            if (src1_nrows > 1) {
                switch (src0->type) {
                    case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_2d_repacked_q4_0_flat;   break;
                    case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_2d_repacked_q4_1_flat;   break;
                    case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_2d_repacked_q8_0_flat;   break;
                    case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_2d_repacked_iq4nl_flat;  break;
                    case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_2d_repacked_mxfp4_flat;  break;
                    default:              return HTP_STATUS_NO_SUPPORT;
                }
            } else {
                switch (src0->type) {
                    case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mv_2d_repacked_q4_0_flat;   break;
                    case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mv_2d_repacked_q4_1_flat;   break;
                    case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mv_2d_repacked_q8_0_flat;   break;
                    case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mv_2d_repacked_iq4nl_flat;  break;
                    case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mv_2d_repacked_mxfp4_flat;  break;
                    default:              return HTP_STATUS_NO_SUPPORT;
                }
            }
            break;
        }

        case HTP_MM_KERNEL_HVX_QUANT_BLOCK:
        case HTP_MM_KERNEL_HVX_QUANT_ROW:
        default:
            if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
                return HTP_STATUS_NO_SUPPORT;
            }

            const uint32_t qk = QK_Q8_0_TILED;
            const uint32_t nb = (ne10 + qk - 1) / qk;
            const uint32_t total_nb = src1_nrows * nb;

            if (src1_nrows < octx->n_threads) {
                n_quant_tasks = MIN(total_nb, octx->n_threads);
                quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
                for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
                    uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
                    uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
                    mmctx->quant_ib_first[ith] = ib_first;
                    mmctx->quant_ib_last[ith]  = ib_last;
                    mmctx->quant_r[ith]        = ib_first / nb;
                    mmctx->quant_c[ith]        = ib_first % nb;
                }
            } else {
                n_quant_tasks = MIN(src1_nrows, octx->n_threads);
                quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
            }
            src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);
            break;
    }

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, src1_nrows, octx->n_threads,
                                 dst_row_size, src0_row_size, src1_row_size, src2 ? src2->nb[1] : 0, kparams->n_prefetch, false, false);

    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_F32_F32_VTCM ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW ||
        kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    } else {
        mmctx->vtcm_src1_size_per_thread = fastdiv(L.src1_bytes, &octx->ctx->n_threads_div);
    }

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->ctx->n_threads_div);
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->ctx->n_threads_div);

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : L.total_bytes;

    FARF(HIGH, "matmul-%s : src0-vtcm-size %zu src1-vtcm-size %zu dst-vtcm-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.src1_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-%s : %ux%ux%ux%u * %ux%ux%ux%u-> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type, src0->ne[0],
         src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], dst->ne[0],
         dst->ne[1], dst->ne[2], dst->ne[3], src0->data, src1->data, dst->data);

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type,
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src2 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src2);
    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    if (need_quant) {
        mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
        mmctx->quant_task_func = quant_task_func;
        mmctx->n_quant_tasks = n_quant_tasks;
        atomic_init(&mmctx->quant_barrier, n_quant_tasks);
    } else {
        mmctx->quant_task_func = NULL;
        mmctx->n_quant_tasks = 0;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, octx->n_threads);

    return HTP_STATUS_OK;
}

static void hvx_mm_nx_2d(unsigned int nth, unsigned int ith, void * data) {
    struct htp_mm_context * mmctx = data;
    struct htp_ops_context * octx = mmctx->octx;
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;

    const struct htp_tensor * restrict act = octx->src[n_weights];
    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];
    const size_t src1_stride = mmctx->vtcm_src1_stride;

    uint8_t * restrict vtcm_src0_ptr = mmctx->vtcm_src0 + mmctx->vtcm_src0_size_per_thread * ith;
    uint8_t * restrict src1_data     = mmctx->vtcm_src1;

    dma_queue * dma_queue = octx->ctx->dma[ith];
    const uint32_t n_prefetch = kparams->n_prefetch;
    assert(n_prefetch >= 2 && n_prefetch <= HTP_MM_MAX_PREFETCH && (n_prefetch & (n_prefetch - 1)) == 0);
    const uint32_t prefetch_mask = n_prefetch - 1;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    hvx_mm_run_quant_task(mmctx, ith);

    for (uint32_t widx = 0; widx < n_weights; widx++) {
        const struct htp_tensor * restrict src_w = octx->src[widx];
        const struct htp_tensor * restrict dst   = octx->dsts[widx];
        if (!src_w || !dst) continue;

        const uint32_t ne00 = src_w->ne[0];
        const uint32_t ne01 = src_w->ne[1];
        const uint32_t src0_nrows = ne01 * src_w->ne[2] * src_w->ne[3];

        uint32_t src0_nrows_per_thread = fastdiv(src0_nrows + nth - 1, &octx->ctx->n_threads_div);
        src0_nrows_per_thread += (src0_nrows_per_thread & 1);

        const uint32_t src0_start_row  = src0_nrows_per_thread * ith;
        const uint32_t src0_end_row    = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);
        const uint32_t src0_end_row_x2 = src0_start_row + ((src0_end_row - src0_start_row) & ~1U);
        if (src0_start_row >= src0_end_row) continue;

        const size_t dst_row_size  = dst->nb[1];
        const size_t src0_row_size = src_w->nb[1];
        const size_t src0_stride   = hex_round_up(src0_row_size, 128);

        const uint8_t * restrict src0_row = (const uint8_t *) src_w->data;

        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const int is0 = (ir0 - src0_start_row);
            if (is0 >= (int)n_prefetch) break;
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 2);
        }

        for (uint32_t ir0 = src0_start_row; ir0 < src0_end_row_x2; ir0 += 2) {
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            uint32_t ir1 = 0;
            for (; ir1 + 1 < src1_nrows; ir1 += 2) {
                const uint8_t * restrict src1_col0 = (const uint8_t *) (src1_data + (ir1+0) * src1_stride);
                const uint8_t * restrict src1_col1 = (const uint8_t *) (src1_data + (ir1+1) * src1_stride);
                float * restrict dst_row0 = (float *) (dst->data + ((ir1+0) * dst_row_size));
                float * restrict dst_row1 = (float *) (dst->data + ((ir1+1) * dst_row_size));
                mmctx->vec_dot_2x2(ne00, &dst_row0[ir0], &dst_row1[ir0], ss0, ss0 + src0_stride, src1_col0, src1_col1);
            }
            for (; ir1 < src1_nrows; ++ir1) {
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));
                mmctx->vec_dot_2x1(ne00, &dst_row[ir0], ss0, ss0 + src0_stride, src1_col);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);

            const int pr0 = (ir0 + n_prefetch);
            const int is0 = (pr0 - src0_start_row) & prefetch_mask;
            if (pr0 < src0_end_row_x2) {
                dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + pr0 * src0_row_size),
                               src0_stride, src0_row_size, src0_row_size, 2);
            }
        }

        if (src0_end_row != src0_end_row_x2) {
            uint32_t ir0 = src0_end_row_x2;
            const int is0 = (ir0 - src0_start_row) & prefetch_mask;
            dma_queue_push(dma_queue, dma_make_ptr(vtcm_src0_ptr + is0 * src0_stride, src0_row + ir0 * src0_row_size),
                           src0_stride, src0_row_size, src0_row_size, 1);
            const uint8_t * ss0 = dma_queue_pop(dma_queue).dst;
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
            for (uint32_t ir1 = 0; ir1 < src1_nrows; ++ir1) {
                const uint8_t * restrict src1_col = (const uint8_t *) (src1_data + ir1 * src1_stride);
                float * restrict dst_row = (float *) (dst->data + (ir1 * dst_row_size));
                mmctx->vec_dot_1x1(ne00, &dst_row[ir0], ss0, src1_col);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ir0);
        }
    }
}

#define DEQUANTIZE_WORKER_LOOP_IMPL(SUFFIX)                                                     \
static void dequantize_tiled_worker_loop_##SUFFIX(unsigned int n, unsigned int i, void *data) { \
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;                         \
    struct htp_thread_trace * tr = &state->traces[i];                                           \
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                  \
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {      \
        int start = task_id * state->n_tiles_per_task;                                          \
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);              \
        dequantize_tiled_weight_to_fp16_task_##SUFFIX(state, start, end);                       \
    }                                                                                           \
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);                                   \
}

DEQUANTIZE_WORKER_LOOP_IMPL(q4_0)
DEQUANTIZE_WORKER_LOOP_IMPL(q4_1)
DEQUANTIZE_WORKER_LOOP_IMPL(iq4_nl)
DEQUANTIZE_WORKER_LOOP_IMPL(mxfp4)
DEQUANTIZE_WORKER_LOOP_IMPL(q8_0)

static void convert_f16_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;
    struct htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        convert_f16_weight_to_fp16_tiles_task(state, start, end);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_W_DEQUANT, i);
}

static void quantize_f32_worker_loop(unsigned int n, unsigned int i, void *data) {
    tiled_dequantize_state_t *state = (tiled_dequantize_state_t *)data;

    struct htp_thread_trace * tr = &state->traces[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);

    for (unsigned int task_id = i; task_id < (unsigned int)state->n_tasks; task_id += n) {
        int start = task_id * state->n_tiles_per_task;
        int end   = hex_smin(start + state->n_tiles_per_task, state->n_tot_tiles);
        quantize_f32_weight_to_fp16_tiles_task(state, start, end);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_QUANT, i);
}

static void transfer_output_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_task_state_t *st = (output_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = &st->traces[i];

    int start_chunk_idx = i * st->n_chunks_per_task;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        float        *dst      = st->dst      + chunk_idx * st->dst_stride;
        const float  *src2     = st->src2     ? (st->src2     + chunk_idx * st->src2_stride) : NULL;
        transfer_output_chunk_fp16_to_fp32(dst, src2, st->vtcm_src, chunk_idx, chunk_size, st->n_cols, st->dst_stride, st->src2_stride, st->dst_cols);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, start_chunk_idx);
}

typedef struct {
    struct htp_context            * ctx;
    struct htp_thread_trace       * traces;
    __fp16                        * dst;
    const float                   * src;
    const struct mmid_row_mapping * matrix_rows;
    float                         * vtcm_f32_act;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    size_t                          vtcm_f32_act_bytes_per_thread;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
} activation_transfer_task_state_t;

typedef struct {
    struct htp_context            * ctx;
    struct htp_thread_trace       * traces;
    __fp16                        * dst;
    const float                   * src;
    float                         * vtcm_f32_act;
    uint32_t                        n_rows;
    uint32_t                        k_block;
    uint32_t                        k_stride;
    uint32_t                        k_valid;
    uint32_t                        n_col_chunks;
    struct fastdiv_values           n_threads_div;
    size_t                          vtcm_f32_act_bytes;
    uint32_t                        dma_step_rows;
    uint32_t                        dma_step_rows_shift;
} activation_transfer_col_chunk_state_t;

static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        const float *restrict src,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_chunk_valid,
        uint32_t c_first,
        uint32_t c_len,
        float *thread_f32_act,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_ptr(thread_f32_act, src + c_first),
                       c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
    }
    // Push step 1
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride + c_first;
            float *next_buf = thread_f32_act + 1 * R * c_len;
            dma_queue_push(dma_q, dma_make_ptr(next_buf, next_src),
                           c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        float *curr_buf = thread_f32_act;

        if (r < n_rows) {
            curr_buf = (float *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        for (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            float *pair_buf = curr_buf + (p << 1) * c_len;
            bool r0_valid = ((row_idx + 0) < n_rows);
            bool r1_valid = ((row_idx + 1) < n_rows);

            transfer_activation_row_pair_fp32_to_fp16_col_chunk(
                vtcm_dst, pair_buf, pair_buf + c_len, row_idx, k_block, c_first, c_len, k_chunk_valid, r0_valid, r1_valid
            );
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride + c_first;
            dma_queue_push(dma_q, dma_make_ptr(curr_buf, next_src),
                           c_len * sizeof(float), k_stride * sizeof(float), k_chunk_valid * sizeof(float), nrows_to_fetch);
        }
    }
}

static void transfer_activation_chunk_fp32_to_fp16_col_chunk(
        __fp16 *restrict vtcm_dst,
        const float *restrict src,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t c_first,
        uint32_t c_len,
        uint32_t k_chunk_valid) {
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_TILE_N_ROWS);
    const uint32_t n_rows_tiled  = (n_rows / HTP_MM_TILE_N_ROWS) * HTP_MM_TILE_N_ROWS;

    uint32_t r = 0;

    #pragma unroll(2)
    for (r = 0; r < n_rows_tiled; r += 2) {
        const float *ptr_in0 = src + (r + 0) * k_stride + c_first;
        const float *ptr_in1 = src + (r + 1) * k_stride + c_first;

        transfer_activation_row_pair_fp32_to_fp16_col_chunk(
            vtcm_dst, ptr_in0, ptr_in1, r, k_block, c_first, c_len, k_chunk_valid, true, true
        );
    }

    for (; r < n_rows_padded; r += 2) {
        const bool row0_valid = r       < n_rows;
        const bool row1_valid = (r + 1) < n_rows;

        const float *ptr_in0 = row0_valid ? (src + (r + 0) * k_stride + c_first) : NULL;
        const float *ptr_in1 = row1_valid ? (src + (r + 1) * k_stride + c_first) : NULL;

        transfer_activation_row_pair_fp32_to_fp16_col_chunk(
            vtcm_dst, ptr_in0, ptr_in1, r, k_block, c_first, c_len, k_chunk_valid, row0_valid, row1_valid
        );
    }
}

static void transfer_activation_chunk_col_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_col_chunk_state_t *st = (activation_transfer_col_chunk_state_t *) data;
    struct htp_thread_trace * tr = &st->traces[i];

    uint32_t n_blocks = st->k_block / 32;
    uint32_t b_first = fastdiv(n_blocks * i, &st->n_threads_div);
    uint32_t b_last  = fastdiv(n_blocks * (i + 1), &st->n_threads_div);
    uint32_t c_first = b_first * 32;
    uint32_t c_last = b_last * 32;
    uint32_t c_len = c_last - c_first;

    if (c_len == 0) {
        return;
    }

    uint32_t k_chunk_valid = 0;
    if (st->k_valid > c_first) {
        k_chunk_valid = hex_smin(st->k_valid, c_last) - c_first;
    }

    __fp16 *dst = st->dst;
    const float *src = st->src;

    if (st->vtcm_f32_act) {
        size_t thread_scratch_bytes = hex_align_down(fastdiv(st->vtcm_f32_act_bytes, &st->n_threads_div), 128);
        float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * thread_scratch_bytes);

        transfer_activation_chunk_fp32_to_fp16_dma_pipelined_col_chunk(
            st->ctx->dma[i], dst, src, st->n_rows, st->k_block, st->k_stride, k_chunk_valid,
            c_first, c_len, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
        );
    } else {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, c_first);
        transfer_activation_chunk_fp32_to_fp16_col_chunk(
            dst, src, st->n_rows, st->k_block, st->k_stride, c_first, c_len, k_chunk_valid
        );
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, c_first);
    }
}

static void transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
        dma_queue *dma_q,
        __fp16 *restrict vtcm_dst,
        const float *restrict src,
        uint32_t n_rows,
        uint32_t k_block,
        uint32_t k_stride,
        uint32_t k_valid,
        float *thread_f32_act,
        struct htp_thread_trace *tr,
        uint32_t dma_step_rows,
        uint32_t dma_step_rows_shift) {

    const uint32_t R = dma_step_rows;
    const uint32_t n_rows_padded = hex_align_up(n_rows, HTP_MM_TILE_N_ROWS);

    const uint32_t n_steps = n_rows_padded >> dma_step_rows_shift;

    // Push step 0
    if (n_steps > 0 && n_rows > 0) {
        uint32_t nrows_to_fetch = hex_smin(n_rows, R);
        dma_queue_push(dma_q, dma_make_ptr(thread_f32_act, src),
                       k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
    }
    // Push step 1 (if valid)
    if (n_steps > 1) {
        uint32_t next_r = R * 1;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride;
            float *next_buf = thread_f32_act + 1 * R * k_block;
            dma_queue_push(dma_q, dma_make_ptr(next_buf, next_src),
                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
        }
    }
    for (uint32_t s = 0; s < n_steps; ++s) {
        uint32_t r = s << dma_step_rows_shift;
        float *curr_buf = thread_f32_act;

        if (r < n_rows) {
            curr_buf = (float *) dma_queue_pop(dma_q).dst;
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, r);
        for (uint32_t p = 0; p < (R >> 1); ++p) {
            uint32_t row_idx = r + (p << 1);
            float *pair_buf = curr_buf + (p << 1) * k_block;
            bool r0_valid = ((row_idx + 0) < n_rows);
            bool r1_valid = ((row_idx + 1) < n_rows);

            transfer_activation_row_pair_fp32_to_fp16(vtcm_dst, pair_buf, pair_buf + k_block, row_idx, k_block, k_valid, r0_valid, r1_valid);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, r);

        // Push step s + 2
        uint32_t next_s = s + 2;
        uint32_t next_r = next_s << dma_step_rows_shift;
        if (next_r < n_rows) {
            uint32_t nrows_to_fetch = hex_smin(n_rows - next_r, R);
            const float *next_src = src + next_r * k_stride;
            dma_queue_push(dma_q, dma_make_ptr(curr_buf, next_src),
                           k_block * sizeof(float), k_stride * sizeof(float), k_valid * sizeof(float), nrows_to_fetch);
        }
    }
}

static void transfer_activation_chunk_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_task_state_t *st = (activation_transfer_task_state_t *) data;

    struct htp_thread_trace * tr = &st->traces[i];

    for (unsigned int task_id = i; task_id < (unsigned int)st->n_tasks; task_id += n) {
        int    chunk_idx  = task_id * st->n_chunks_per_task;
        size_t chunk_size = hex_smin(st->n_tot_chunks - chunk_idx, st->n_chunks_per_task);

        __fp16      *dst = st->dst + chunk_idx * st->k_block;
        const float *src = st->src + chunk_idx * st->k_stride;

        if (st->vtcm_f32_act) {
            float *thread_f32_act = (float *)((char *)st->vtcm_f32_act + i * st->vtcm_f32_act_bytes_per_thread);
            transfer_activation_chunk_fp32_to_fp16_dma_pipelined(
                st->ctx->dma[i], dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid, thread_f32_act, tr, st->dma_step_rows, st->dma_step_rows_shift
            );
        } else {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
            transfer_activation_chunk_fp32_to_fp16(dst, src, chunk_size, st->k_block, st->k_stride, st->k_valid);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        }
    }
}

typedef struct {
    struct htp_thread_trace       * traces;
    const struct mmid_row_mapping * matrix_rows;
    __fp16                        * dst;
    const float                   * src;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        k_block;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    uint32_t                        ne11;
    struct fastdiv_values           ne11_div;
    size_t                          nb11;
    size_t                          nb12;
    uint32_t                        start_row;
    uint32_t                        cne1;
    uint32_t                        k_valid;
} activation_transfer_gathered_task_state_t;

typedef struct {
    struct htp_thread_trace       * traces;
    const struct mmid_row_mapping * matrix_rows;
    const __fp16                  * vtcm_src;
    float                         * dst;
    uint32_t                        n_tasks;
    uint32_t                        n_tot_chunks;
    uint32_t                        n_chunks_per_task;
    uint32_t                        n_cols;
    uint32_t                        cur_a;
    uint32_t                        mapping_stride;
    size_t                          dst_nb1;
    size_t                          dst_nb2;
    uint32_t                        start_row;
    uint32_t                        cne1;
} output_transfer_scattered_task_state_t;

static void transfer_activation_chunk_gathered_worker_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    int chunk_idx      = i;
    int chunk_size     = st->n_chunks_per_task;
    int vtcm_start_row = chunk_idx * chunk_size;
    int start_row      = st->start_row + vtcm_start_row;
    int n_rows         = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->ne11, &st->ne11_div, st->nb11, st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_activation_chunk_gathered_worker_flat_fn(unsigned int n, unsigned int i, void *data) {
    activation_transfer_gathered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int vtcm_start_row = chunk_idx * chunk_size;
    int start_row = st->start_row + vtcm_start_row;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
        transfer_activation_chunk_fp32_to_fp16_gathered_flat(
            st->dst, st->src, start_row, vtcm_start_row, n_rows, st->k_block,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->nb12, st->cne1, st->k_valid);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, chunk_idx);
    }
}

static void transfer_output_chunk_scattered_worker_fn(unsigned int n, unsigned int i, void *data) {
    output_transfer_scattered_task_state_t *st = data;
    struct htp_thread_trace * tr = &st->traces[i];
    int chunk_idx = i;
    int chunk_size = st->n_chunks_per_task;
    int vtcm_start_row = chunk_idx * chunk_size;
    int start_row = st->start_row + vtcm_start_row;
    int n_rows = hex_smin(st->cne1 - start_row, chunk_size);
    if (n_rows > 0) {
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
        transfer_output_chunk_fp16_to_fp32_scattered(
            st->dst, st->vtcm_src, start_row, vtcm_start_row, n_rows, st->n_cols,
            st->matrix_rows, st->cur_a, st->mapping_stride,
            st->dst_nb1, st->dst_nb2, st->cne1);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, chunk_idx);
    }
}

// --- Dispatchers and Public Entry Points ---

int op_matmul(struct htp_ops_context * octx) {
    return hvx_mm_matmul(octx);
}

static int hvx_mm_matmul_id(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    htp_matmul_tensors_preamble;
    const uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    const uint32_t src1_nrows           = mmctx->src1_nrows;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const struct htp_tensor * restrict ids = octx->src[2];
    const size_t src0_row_size = nb01;

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne10 + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (src1_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }
    size_t src1_row_size  = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(ne10) : htp_mm_q8_0_tiled_row_size(ne10);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, ne10, src1_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false);

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : L.total_bytes;

    FARF(HIGH, "matmul-id-%s : src0-spad-size %zu src1-spad-size %zu src2-spad-size 0 dst-spad-size %zu (%zu)\n", mmctx->type,
         L.src0_bytes, L.src1_bytes, L.dst_bytes, vtcm_size);

    FARF(HIGH, "matmul-id-%s : %ux%ux%ux%u * %ux%ux%ux%u (%ux%ux%ux%u) -> %ux%ux%ux%u (0x%p, 0x%p, 0x%p)\n", mmctx->type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], src0->data,
         src1->data, dst->data);

    // Make sure the reserved vtcm size is sufficient
    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-%s : current VTCM reservation %zu is too small, needed %zu\n", mmctx->type, octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src2 = NULL;
    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    octx->src1_spad.src  = NULL;
    octx->src0_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->ctx->n_threads_div);
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_src2_size_per_thread = 0;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->ctx->n_threads_div);

    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->quant_task_func = quant_task_func;
    mmctx->n_quant_tasks = n_quant_tasks;
    atomic_init(&mmctx->quant_barrier, n_quant_tasks);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    worker_pool_run_func(octx->ctx->worker_pool, hvx_mmid_task_func, mmctx, octx->n_threads);

    return HTP_STATUS_OK;
}
static int hvx_mm_matmul_id_nx(
    struct htp_ops_context * octx,
    struct htp_mm_context * mmctx,
    work_queue_func_t hvx_mmid_task_func
) {
    const uint32_t src0_row_size_padded = mmctx->src0_row_size_padded;
    const uint32_t src1_nrows           = mmctx->src1_nrows;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];
    const size_t src0_row_size = src0->nb[1];

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (act->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    work_queue_func_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (src1_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }
    size_t src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(act->ne[0]) : htp_mm_q8_0_tiled_row_size(act->ne[0]);

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], src1_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, true, false);

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-id-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    octx->src0_spad.src = NULL;
    octx->src1_spad.src = NULL;
    octx->src2_spad.src = NULL;
    octx->src3_spad.src = NULL;
    octx->dst_spad.src  = NULL;

    mmctx->vtcm_src0_stride = 0;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->ctx->n_threads_div);
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->ctx->n_threads_div);

    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->quant_task_func         = quant_task_func;
    mmctx->n_quant_tasks           = n_quant_tasks;
    atomic_init(&mmctx->quant_barrier, n_quant_tasks);

    FARF(HIGH, "matmul-id-nx: src0 %d:%d:%d type %s nrows %u, src1 %d:%d:%d nrows %u, vtcm %zu/%zu, threads %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], mmctx->type, src0->ne[1],
         act->ne[0], act->ne[1], act->ne[2], src1_nrows,
         L.total_bytes, octx->ctx->vtcm_size, octx->n_threads);

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    worker_pool_run_func(octx->ctx->worker_pool, hvx_mmid_task_func, mmctx, octx->n_threads);

    return HTP_STATUS_OK;
}

static inline void scan_expert_ids_n(
    const struct htp_tensor * ids,
    const uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    struct mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    const size_t ids_nb1 = ids->nb[1];
    const uint8_t * ids_data = (const uint8_t *) ids->data;

    for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
        const int32_t * row_ptr = (const int32_t *) (ids_data + iid1 * ids_nb1);
        for (uint32_t id = 0; id < n_ids; ++id) {
            const int32_t i02 = row_ptr[id];
            if (i02 < 0) {
                continue;
            }
            assert(i02 < n_as);

            if (matrix_rows) {
                matrix_rows[i02 * mapping_stride + counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
            }
            counts[i02] += 1;
        }
    }
}

static inline void scan_expert_ids(
    const struct htp_tensor * ids,
    uint32_t n_ids,
    uint32_t n_as,
    uint32_t * counts,
    struct mmid_row_mapping * matrix_rows,
    uint32_t mapping_stride
) {
    const size_t ids_nb0 = ids->nb[0];

    if (ids_nb0 == 4) {
        switch (n_ids) {
            case 8:  scan_expert_ids_n(ids, 8,     n_as, counts, matrix_rows, mapping_stride); break;
            case 4:  scan_expert_ids_n(ids, 4,     n_as, counts, matrix_rows, mapping_stride); break;
            case 2:  scan_expert_ids_n(ids, 2,     n_as, counts, matrix_rows, mapping_stride); break;
            default: scan_expert_ids_n(ids, n_ids, n_as, counts, matrix_rows, mapping_stride); break;
        }
    } else {
        // Strided fallback
        const size_t ids_nb1 = ids->nb[1];
        const uint8_t * ids_data = (const uint8_t *) ids->data;
        for (uint32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            const int32_t * row_ptr = (const int32_t *) (ids_data + iid1 * ids_nb1);
            for (uint32_t id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const uint8_t *) row_ptr + id * ids_nb0);
                if (i02 < 0) {
                    continue;
                }
                assert(i02 < n_as);

                if (matrix_rows) {
                    matrix_rows[i02 * mapping_stride + counts[i02]] = (struct mmid_row_mapping) { id, iid1 };
                }
                counts[i02] += 1;
            }
        }
    }
}

int op_matmul_id(struct htp_ops_context * octx) {
    htp_matmul_tensors_preamble;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = src1;

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    const struct htp_tensor * restrict ids = octx->src[2];

    const size_t src0_row_size = nb01;
    const size_t dst_row_size  = nb1;

    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = ne01;  // per expert
    const uint32_t src1_nrows = ne11 * ne12 * ne13;

    mmctx->src0_nrows_per_thread = fastdiv(src0_nrows + octx->n_threads - 1, &octx->ctx->n_threads_div);
    mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

    // row groups
    const int n_ids = ids->ne[0];  // n_expert_used
    const int n_as  = ne02;        // n_expert

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    struct mmid_row_mapping * matrix_rows = NULL;

    if (src1_nrows > 1) {
        const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
        assert(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((const void *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, NULL, 0);

        uint32_t max_count = hvx_reduce_max_i32((const uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_count > 0 ? max_count : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * sizeof(struct mmid_row_mapping);
        const size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        if (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            if (!mapping_buf) {
                return HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (struct mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * sizeof(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne11;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->src1_nrows           = src1_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    int s;
    if (hvx_mm_init_vec_dot(mmctx, src0->type) == 0) {
        s = hvx_mm_matmul_id(octx, mmctx, src1_nrows > 1 ? hvx_mm_id : hvx_mv_id);
    } else {
        s = HTP_STATUS_NO_SUPPORT;
    }

    if (mapping_buf != octx->ctx->ddr_spad_base) {
        free(mapping_buf);
    }

    return s;
}

int op_matmul_id_nx(struct htp_ops_context * octx) {
    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;
    const uint32_t n_weights = kparams->n_weights;
    const struct htp_tensor * restrict src0 = octx->src[0];
    const struct htp_tensor * restrict act  = octx->src[n_weights];
    const struct htp_tensor * restrict ids  = octx->src[n_weights + 1];

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act = act;

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    const uint32_t src0_nrows = src0->ne[1];
    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    mmctx->src0_nrows_per_thread = fastdiv(src0_nrows + octx->n_threads - 1, &octx->ctx->n_threads_div);
    mmctx->src0_nrows_per_thread = hex_round_up(mmctx->src0_nrows_per_thread, 32);

    const int n_ids = ids->ne[0];
    const int n_as  = src0->ne[2];

    uint8_t  * mapping_buf       = octx->ctx->ddr_spad_base;
    uint32_t   mapping_stride    = 1;
    uint32_t * matrix_row_counts = (uint32_t *) mapping_buf;
    struct mmid_row_mapping * matrix_rows = NULL;

    if (src1_nrows > 1) {
        const size_t matrix_row_counts_size = n_as * sizeof(uint32_t);
        assert(octx->ctx->ddr_spad_size >= matrix_row_counts_size);

        hex_l2fetch_block((const void *) ids->data, ids->ne[1] * ids->nb[1]);

        memset(matrix_row_counts, 0, matrix_row_counts_size);
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, NULL, 0);

        uint32_t max_count = hvx_reduce_max_i32((const uint8_t *) matrix_row_counts, n_as);
        mapping_stride = max_count > 0 ? max_count : 1;

        size_t matrix_row_map_size  = n_as * mapping_stride * sizeof(struct mmid_row_mapping);
        const size_t total_map_size = matrix_row_counts_size + matrix_row_map_size;

        if (total_map_size > octx->ctx->ddr_spad_size) {
            mapping_buf = memalign(128, total_map_size);
            if (!mapping_buf) {
                return HTP_STATUS_INTERNAL_ERR;
            }
        }

        matrix_row_counts = (uint32_t *) mapping_buf;
        matrix_rows       = (struct mmid_row_mapping *) (mapping_buf + matrix_row_counts_size);

        memset(matrix_row_counts, 0, n_as * sizeof(uint32_t));
        scan_expert_ids(ids, n_ids, n_as, matrix_row_counts, matrix_rows, mapping_stride);
    }

    mmctx->matrix_row_counts    = matrix_row_counts;
    mmctx->matrix_rows          = matrix_rows;
    mmctx->mapping_stride       = mapping_stride;
    mmctx->mm_div_ne11          = kparams->div_ne11;
    mmctx->src0_row_size_padded = src0_row_size_padded;
    mmctx->src1_nrows           = src1_nrows;

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    int s;
    if (hvx_mm_init_vec_dot(mmctx, src0->type) == 0) {
        s = hvx_mm_matmul_id_nx(octx, mmctx, src1_nrows > 1 ? hvx_mm_id_nx : hvx_mv_id_nx);
    } else {
        s = HTP_STATUS_NO_SUPPORT;
    }

    if (mapping_buf != octx->ctx->ddr_spad_base) {
        free(mapping_buf);
    }

    return s;
}
int op_matmul_nx(struct htp_ops_context * octx) {
    const struct htp_mm_kernel_params * kparams = (const struct htp_mm_kernel_params *) octx->kernel_params;

    struct htp_thread_trace * tr = &octx->ctx->trace[0];
    htp_trace_event_start(tr, HTP_TRACE_EVT_INIT, 0);

    const uint32_t n_weights = kparams->n_weights;

    const struct htp_tensor * restrict src0 = octx->src[0]; // first weight
    const struct htp_tensor * restrict act  = octx->src[n_weights]; // activation x

    bool is_repacked = (src0->type == HTP_TYPE_Q4_0 || src0->type == HTP_TYPE_Q4_1 ||
                        src0->type == HTP_TYPE_Q8_0 || src0->type == HTP_TYPE_IQ4_NL ||
                        src0->type == HTP_TYPE_MXFP4);

    struct htp_mm_context mmctx_struct = {0};
    struct htp_mm_context * mmctx = &mmctx_struct;
    mmctx->octx = octx;
    mmctx->act  = act;

    const uint32_t src1_nrows = act->ne[1] * act->ne[2] * act->ne[3];

    const size_t src0_row_size = src0->nb[1];
    const size_t src0_row_size_padded = hex_round_up(src0_row_size, 128);

    if (hvx_mm_init_vec_dot(mmctx, src0->type) != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (act->ne[0] + qk - 1) / qk;
    const uint32_t total_nb = src1_nrows * nb;

    worker_callback_t quant_task_func;
    uint32_t n_quant_tasks = 1;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_flat : quantize_f32_q8_0_flat;
    } else if (src1_nrows < octx->n_threads) {
        n_quant_tasks = MIN(total_nb, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled_block : quantize_f32_q8_0_tiled_block;
        for (uint32_t ith = 0; ith < n_quant_tasks; ++ith) {
            uint32_t ib_first = (total_nb * ith) / n_quant_tasks;
            uint32_t ib_last  = (total_nb * (ith + 1)) / n_quant_tasks;
            mmctx->quant_ib_first[ith] = ib_first;
            mmctx->quant_ib_last[ith]  = ib_last;
            mmctx->quant_r[ith]        = ib_first / nb;
            mmctx->quant_c[ith]        = ib_first % nb;
        }
    } else {
        n_quant_tasks = MIN(src1_nrows, octx->n_threads);
        quant_task_func = (src0->type == HTP_TYPE_Q4_1) ? quantize_f32_q8_1_tiled : quantize_f32_q8_0_tiled;
    }

    size_t src1_row_size;
    if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_flat_row_size(act->ne[0]) : htp_mm_q8_0_flat_row_size(act->ne[0]);
    } else {
        src1_row_size = (src0->type == HTP_TYPE_Q4_1) ? htp_mm_q8_1_tiled_row_size(act->ne[0]) : htp_mm_q8_0_tiled_row_size(act->ne[0]);
    }

    struct htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(&L, kparams->kernel_type, src0->type, act->ne[0], src1_nrows, octx->n_threads,
                                 0, src0_row_size, src1_row_size, 0, kparams->n_prefetch, false, true);

    size_t vtcm_size = kparams->vtcm_size > 0 ? (size_t)kparams->vtcm_size : L.total_bytes;

    if (octx->ctx->vtcm_size < vtcm_size) {
        FARF(ERROR, "matmul-nx: current VTCM reservation %zu is too small, needed %zu\n",
             octx->ctx->vtcm_size, vtcm_size);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = (uint8_t *) octx->ctx->vtcm_base;
    mmctx->vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src0);
    mmctx->vtcm_src1 = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src1);
    mmctx->vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);

    octx->src0_spad.src  = NULL;
    octx->src1_spad.src  = NULL;
    octx->src2_spad.src  = NULL;
    octx->src3_spad.src  = NULL;
    octx->dst_spad.src   = NULL;

    mmctx->vtcm_src0_stride = is_repacked ? 0 : src0_row_size_padded;
    mmctx->vtcm_src1_stride = src1_row_size;

    mmctx->vtcm_src0_size_per_thread = fastdiv(L.src0_bytes, &octx->ctx->n_threads_div);
    mmctx->vtcm_src1_size_per_thread = L.src1_bytes;
    mmctx->vtcm_dst_size_per_thread  = fastdiv(L.dst_bytes, &octx->ctx->n_threads_div);

    mmctx->n_quant_rows_per_thread = (src1_nrows + n_quant_tasks - 1) / n_quant_tasks;
    mmctx->quant_task_func = quant_task_func;
    mmctx->n_quant_tasks = n_quant_tasks;
    atomic_init(&mmctx->quant_barrier, n_quant_tasks);

    // Run fused matmul
    const uint32_t n_matmul_jobs = octx->n_threads;
    worker_callback_t matmul_job_func;
    if (is_repacked) {
        if (kparams->kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW_FLAT) {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_0_flat;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_1_flat;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q8_0_flat;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_nx_2d_repacked_iq4nl_flat;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_nx_2d_repacked_mxfp4_flat;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        } else {
            switch (src0->type) {
                case HTP_TYPE_Q4_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_0;   break;
                case HTP_TYPE_Q4_1:   matmul_job_func = hvx_mm_nx_2d_repacked_q4_1;   break;
                case HTP_TYPE_Q8_0:   matmul_job_func = hvx_mm_nx_2d_repacked_q8_0;   break;
                case HTP_TYPE_IQ4_NL: matmul_job_func = hvx_mm_nx_2d_repacked_iq4nl;  break;
                case HTP_TYPE_MXFP4:  matmul_job_func = hvx_mm_nx_2d_repacked_mxfp4;  break;
                default:              return HTP_STATUS_NO_SUPPORT;
            }
        }
    } else {
        matmul_job_func = hvx_mm_nx_2d;
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_INIT, 0);

    worker_pool_run_func(octx->ctx->worker_pool, matmul_job_func, mmctx, n_matmul_jobs);

    return HTP_STATUS_OK;
}
