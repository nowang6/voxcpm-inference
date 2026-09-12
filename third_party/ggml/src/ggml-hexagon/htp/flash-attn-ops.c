#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <assert.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hex-dma.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-flash-attn.h"
#include "htp-vtcm.h"
#include "work-queue.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-quant.h"

#include "flash-attn-ops.h"
#include "hvx-fa-kernels.h"

// Must be multiple of 32
#define FLASH_ATTN_BLOCK_SIZE (32 * 2)

struct htp_fa_context {
    const struct htp_ops_context * octx;

    struct fastdiv_values src0_div21;
    struct fastdiv_values src0_div1;

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;

    float scale;
    float max_bias;
    __fp16 logit_softcap;

    uint32_t n_head_log2;
    float m0;
    float m1;
    __fp16 slopes[512];

    uint32_t n_blocks;

    size_t size_q_row_padded;
    size_t size_k_row_padded;
    size_t size_v_row_padded;

    size_t size_k_block;
    size_t size_v_block;
    size_t size_m_block;

    uint32_t qrows;
    uint32_t qrows_per_thread;

    bool is_q_fp32;

    size_t size_q_block;
    size_t size_vkq_acc;

    uint8_t * spad_q;
    uint8_t * spad_k;
    uint8_t * spad_v;
    uint8_t * spad_m;
    uint8_t * spad_a;

    const struct htp_tensor * k;
    const struct htp_tensor * v;

    uint64_t t_start;
};

static void flash_attn_ext_f16_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * mask  = octx->src[3];
    const struct htp_tensor * sinks = octx->src[4];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];

    const uint32_t nek0 = k->ne[0];
    const uint32_t nek1 = k->ne[1];
    const uint32_t nek2 = k->ne[2];
    const uint32_t nek3 = k->ne[3];

    const uint32_t nev0 = v->ne[0];
    const uint32_t nev1 = v->ne[1];
    const uint32_t nev2 = v->ne[2];
    const uint32_t nev3 = v->ne[3];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbq3 = q->nb[3];

    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbk3 = k->nb[3];

    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];
    const uint32_t nbv3 = v->nb[3];

    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    // total rows in q
    const uint32_t nr = factx->qrows;
    const uint32_t dr = factx->qrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_queue * dma = octx->ctx->dma[ith];

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    const size_t size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    const size_t size_k_row = htp_tensor_get_row_size(k->type, DK);
    const size_t size_v_row = htp_tensor_get_row_size(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->size_q_block * ith;
    uint8_t * spad_k = factx->spad_k + factx->size_k_block * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->size_v_block * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (mask ? factx->size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->size_vkq_acc * ith;

    dma_cache m_cache;
    dma_cache_init(&m_cache, spad_m, factx->size_m_block, HVX_FA_DMA_CACHE_SIZE);

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &factx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &factx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &factx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &factx->broadcast_rv2);

        const __fp16 * mp_base = NULL;
        if (mask) {
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &factx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &factx->src3_div3);
            mp_base = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        // Precalculate next row variables if there is a next row
        bool has_next_ir = (ir + 1 < ir1);
        uint32_t next_ik2 = 0, next_ik3 = 0, next_iv2 = 0, next_iv3 = 0;
        const uint8_t * next_q_row_ptr = NULL;
        const __fp16 * next_mp_base = NULL;

        const uint8_t * next_k_src0 = NULL;
        const uint8_t * next_v_src0 = NULL;
        const uint8_t * next_m_src0 = NULL;
        uint32_t next_block_size0 = 0;

        const uint8_t * next_k_src1 = NULL;
        const uint8_t * next_v_src1 = NULL;
        const uint8_t * next_m_src1 = NULL;
        uint32_t next_block_size1 = 0;

        if (has_next_ir) {
            const uint32_t next_ir = ir + 1;
            const uint32_t next_iq3 = fastdiv(next_ir, &factx->src0_div21);
            const uint32_t next_iq2 = fastdiv(next_ir - next_iq3*neq2*neq1, &factx->src0_div1);
            const uint32_t next_iq1 = (next_ir - next_iq3*neq2*neq1 - next_iq2 * neq1);

            next_ik3 = fastdiv(next_iq3, &factx->broadcast_rk3);
            next_ik2 = fastdiv(next_iq2, &factx->broadcast_rk2);

            next_iv3 = fastdiv(next_iq3, &factx->broadcast_rv3);
            next_iv2 = fastdiv(next_iq2, &factx->broadcast_rv2);

            next_q_row_ptr = (const uint8_t *) q->data + (next_iq1*nbq1 + next_iq2*nbq2 + next_iq3*nbq3);

            if (mask) {
                const uint32_t next_im2 = fastmodulo(next_iq2, mask->ne[2], &factx->src3_div2);
                const uint32_t next_im3 = fastmodulo(next_iq3, mask->ne[3], &factx->src3_div3);
                next_mp_base = (const __fp16 *) ((const uint8_t *) mask->data + next_iq1*mask->nb[1] + next_im2*mask->nb[2] + next_im3*mask->nb[3]);
            }

            // Precalculate next K/V block 0 source pointers
            {
                const uint32_t ic_start = 0;
                next_block_size0 = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);
                next_k_src0 = (const uint8_t *) k->data + (ic_start*nbk1 + next_ik2*nbk2 + next_ik3*nbk3);
                next_v_src0 = (const uint8_t *) v->data + (ic_start*nbv1 + next_iv2*nbv2 + next_iv3*nbv3);
                if (mask) {
                    next_m_src0 = (const uint8_t *) (next_mp_base + ic_start);
                }
            }

            // Precalculate next K/V block 1 source pointers (if n_blocks > 1)
            if (factx->n_blocks > 1) {
                const uint32_t ic_start = 1 * FLASH_ATTN_BLOCK_SIZE;
                next_block_size1 = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);
                next_k_src1 = (const uint8_t *) k->data + (ic_start*nbk1 + next_ik2*nbk2 + next_ik3*nbk3);
                next_v_src1 = (const uint8_t *) v->data + (ic_start*nbv1 + next_iv2*nbv2 + next_iv3*nbv3);
                if (mask) {
                    next_m_src1 = (const uint8_t *) (next_mp_base + ic_start);
                }
            }
        }

        if (ir == ir0) {
            // Fetch Q row
            const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);
            dma_queue_push(dma, dma_make_ptr(spad_q, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);

            // Prefetch first two blocks
            for (uint32_t ib = 0; ib < MIN(factx->n_blocks, 2); ++ib) {
                const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

                // K
                const uint8_t * k_src = (const uint8_t *) k->data + (ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
                uint8_t * k_dst = spad_k + (ib % 2) * factx->size_k_block;
                dma_queue_push(dma, dma_make_ptr(k_dst, k_src), factx->size_k_row_padded, nbk1, size_k_row, current_block_size);

                // V
                const uint8_t * v_src = (const uint8_t *) v->data + (ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
                uint8_t * v_dst = spad_v + (ib % 2) * factx->size_v_block;
                dma_queue_push(dma, dma_make_ptr(v_dst, v_src), factx->size_v_row_padded, nbv1, size_v_row, current_block_size);

                // Mask
                if (mask) {
                    const uint8_t * m_src = (const uint8_t *) (mp_base + ic_start);
                    // Mask is 1D contiguous for this row
                    dma_cache_push(dma, &m_cache, m_src, current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
                }
            }
        }

        const uint32_t h = iq2; // head index
        const __fp16 slope = factx->slopes[h];

        HVX_Vector S_vec = hvx_vec_splat_f32(0.0f);
        HVX_Vector M_vec = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

        // Clear accumulator
        hvx_splat_f32_a(spad_a, 0, DV);
        float * VKQ32 = (float *) (spad_a + 0);

        uint8_t * q_ptr_vtcm = dma_queue_pop(dma).dst;
        if (factx->is_q_fp32) {
            hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);  // inplace convert f32 to f16
        }

        const HVX_Vector slope_vec = hvx_vec_splat_f16(slope);
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        const HVX_Vector v_cap     = (factx->logit_softcap != 0.0f) ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        const HVX_Vector vinf      = Q6_Vh_vsplat_R(0xFC00);
        const HVX_Vector vmin      = Q6_Vh_vsplat_R(0xFBFF);
        const HVX_Vector v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        const uint32_t stride_v2   = factx->size_v_row_padded * 2;
        for (uint32_t ib = 0; ib < factx->n_blocks; ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_base = dma_queue_pop(dma).dst; // K
            uint8_t * v_base = dma_queue_pop(dma).dst; // V
            __fp16  * m_base = mask ? dma_queue_pop(dma).dst : NULL; // M

            if (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                for (uint32_t r = 0; r < current_block_size; ++r) {
                    __fp16 * row_k = (__fp16 *)(k_base + r * factx->size_k_row_padded);
                    hvx_dequantize_row_q8_0_f16(row_k, row_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            if (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                for (uint32_t r = 0; r < current_block_size; ++r) {
                    __fp16 * row_v = (__fp16 *)(v_base + r * factx->size_v_row_padded);
                    hvx_dequantize_row_q8_0_f16(row_v, row_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_QK, ir);

            // Inner loop processing the block from VTCM
            // 1. Compute scores (64 elements FP16)
            HVX_Vector scores_f16 = Q6_V_vzero();
            if (current_block_size > 0) {
                HVX_Vector scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base, factx->size_k_row_padded, DK, factx->scale);
                HVX_Vector scores1 = (current_block_size > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base + 32 * factx->size_k_row_padded, factx->size_k_row_padded, DK, factx->scale) : Q6_V_vzero();
                scores_f16 = hvx_vec_f32_to_f16(scores0, scores1);
            }

            // 2. Softcap (in FP16)
            if (factx->logit_softcap != 0.0f) {
                scores_f16 = hvx_vec_tanh_f16(scores_f16);
                scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
            }

            HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(current_block_size * sizeof(__fp16));

            // 3. Mask (in FP16)
            if (mask) {
                HVX_Vector m_vals_f16 = *(const HVX_UVector *) m_base;
                HVX_VectorPred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                HVX_Vector m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, slope_vec);
                scores_f16 = Q6_V_vmux_QVV(q_tail_keep, hvx_vec_add_f16_f16(scores_f16, m_scaled), v_neg_inf);
            } else {
                scores_f16 = Q6_V_vmux_QVV(q_tail_keep, scores_f16, v_neg_inf);
            }

            // Compute block max in FP16
            HVX_Vector v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
            HVX_Vector v_max     = Q6_V_lo_W(hvx_vec_f16_to_f32(v_max_f16)); // splat block max in FP32
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, ir);

            if (ib + 1 == factx->n_blocks && has_next_ir) {
                // Queue next row's Q row!
                dma_queue_push(dma, dma_make_ptr(spad_q, next_q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);

                if (factx->n_blocks % 2 == 0) {
                    // Queue next row's block 0 (into buffer slot 0)
                    uint8_t * k_dst = spad_k + 0 * factx->size_k_block;
                    uint8_t * v_dst = spad_v + 0 * factx->size_v_block;

                    // K (block 0 of next row)
                    dma_queue_push(dma, dma_make_ptr(k_dst, next_k_src0), factx->size_k_row_padded, nbk1, size_k_row, next_block_size0);

                    // V (block 0 of next row)
                    dma_queue_push(dma, dma_make_ptr(v_dst, next_v_src0), factx->size_v_row_padded, nbv1, size_v_row, next_block_size0);

                    // Mask (block 0 of next row)
                    if (mask) {
                        dma_cache_push(dma, &m_cache, next_m_src0, next_block_size0 * 2, next_block_size0 * 2, next_block_size0 * 2, 1);
                    }
                }
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, ir);
            {
                // 4. Online Softmax Update
                HVX_Vector M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec);
                HVX_Vector diff_vec  = HVX_OP_SUB_F32(M_vec, M_new_vec);

                HVX_Vector diff_f16   = hvx_vec_f32_to_f16(diff_vec, diff_vec);
                HVX_Vector diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                HVX_Vector ms_f16     = hvx_vec_exp2_f16(diff_base2);
                HVX_Vector ms_vec     = Q6_V_lo_W(hvx_vec_f16_to_f32(ms_f16));

                M_vec = M_new_vec;

                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                // Compute P = exp2((S - M) * log2(e)) in FP16
                HVX_Vector v_m_vec_f16 = hvx_vec_f32_to_f16(M_vec, M_vec);
                HVX_Vector v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);

                HVX_Vector v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                HVX_Vector P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                P = Q6_V_vmux_QVV(q_tail_keep, P, Q6_V_vzero());

                // Convert P to FP32 to update the running sum S_vec
                HVX_VectorPair P_pair = hvx_vec_f16_to_f32(P);
                HVX_Vector P0 = Q6_V_lo_W(P_pair);
                HVX_Vector P1 = Q6_V_hi_W(P_pair);
                HVX_Vector p_sum_vec = hvx_vec_reduce_sum_f32(HVX_OP_ADD_F32(P0, P1));

                S_vec = HVX_OP_ADD_F32(HVX_OP_MUL_F32(S_vec, ms_vec), p_sum_vec);

                // 5. Accumulate V (F16 * F16 -> F32 accumulator)
                const uint8_t * v_ptr = v_base;

                for (uint32_t j = 0; j < current_block_size; j += 2) {
                    if (j + 1 == current_block_size) {
                        HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                        hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                        break;
                    }

                    HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                    HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                    hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->size_v_row_padded, S0, S1, DV);
                    v_ptr += stride_v2;
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, ir);

            // Issue DMA for next+1 block (if exists)
            if (ib + 2 < factx->n_blocks) {
                const uint32_t next_ib = ib + 2;
                const uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                // K
                const uint8_t * k_src = (const uint8_t *) k->data + (next_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
                dma_queue_push(dma, dma_make_ptr(k_base, k_src), factx->size_k_row_padded, nbk1, size_k_row, next_block_size);

                // V
                const uint8_t * v_src = (const uint8_t *) v->data + (next_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
                dma_queue_push(dma, dma_make_ptr(v_base, v_src), factx->size_v_row_padded, nbv1, size_v_row, next_block_size);

                // Mask
                if (mask) {
                    const uint8_t * m_src = (const uint8_t *) (mp_base + next_ic_start);
                    dma_cache_push(dma, &m_cache, m_src, next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                }
            }
        }

        if (has_next_ir) {
            if (factx->n_blocks % 2 == 0) {
                // Queue next row's block 1 (into buffer slot 1, if n_blocks > 1)
                if (factx->n_blocks > 1) {
                    uint8_t * k_dst = spad_k + 1 * factx->size_k_block;
                    uint8_t * v_dst = spad_v + 1 * factx->size_v_block;

                    // K (block 1 of next row)
                    dma_queue_push(dma, dma_make_ptr(k_dst, next_k_src1), factx->size_k_row_padded, nbk1, size_k_row, next_block_size1);

                    // V (block 1 of next row)
                    dma_queue_push(dma, dma_make_ptr(v_dst, next_v_src1), factx->size_v_row_padded, nbv1, size_v_row, next_block_size1);

                    // Mask (block 1 of next row)
                    if (mask) {
                        dma_cache_push(dma, &m_cache, next_m_src1, next_block_size1 * 2, next_block_size1 * 2, next_block_size1 * 2, 1);
                    }
                }
            } else {
                // Queue next row's block 0 (into buffer slot 0)
                {
                    uint8_t * k_dst = spad_k + 0 * factx->size_k_block;
                    uint8_t * v_dst = spad_v + 0 * factx->size_v_block;

                    // K (block 0 of next row)
                    dma_queue_push(dma, dma_make_ptr(k_dst, next_k_src0), factx->size_k_row_padded, nbk1, size_k_row, next_block_size0);

                    // V (block 0 of next row)
                    dma_queue_push(dma, dma_make_ptr(v_dst, next_v_src0), factx->size_v_row_padded, nbv1, size_v_row, next_block_size0);

                    // Mask (block 0 of next row)
                    if (mask) {
                        dma_cache_push(dma, &m_cache, next_m_src0, next_block_size0 * 2, next_block_size0 * 2, next_block_size0 * 2, 1);
                    }
                }

                // Queue next row's block 1 (into buffer slot 1, if n_blocks > 1)
                if (factx->n_blocks > 1) {
                    uint8_t * k_dst = spad_k + 1 * factx->size_k_block;
                    uint8_t * v_dst = spad_v + 1 * factx->size_v_block;

                    // K (block 1 of next row)
                    dma_queue_push(dma, dma_make_ptr(k_dst, next_k_src1), factx->size_k_row_padded, nbk1, size_k_row, next_block_size1);

                    // V (block 1 of next row)
                    dma_queue_push(dma, dma_make_ptr(v_dst, next_v_src1), factx->size_v_row_padded, nbv1, size_v_row, next_block_size1);

                    // Mask (block 1 of next row)
                    if (mask) {
                        dma_cache_push(dma, &m_cache, next_m_src1, next_block_size1 * 2, next_block_size1 * 2, next_block_size1 * 2, 1);
                    }
                }
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, ir);
        // sinks
        float M = hvx_vec_get_f32(M_vec);
        float S = hvx_vec_get_f32(S_vec);

        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float vs = 1.0f;

            if (s > M) {
                HVX_Vector diff_vec = hvx_vec_splat_f32(M - s);
                HVX_Vector ms_vec   = hvx_vec_exp_f32(diff_vec);
                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                float ms = hvx_vec_get_f32(ms_vec);
                S = S * ms + vs;
            } else {
                HVX_Vector diff_vec = hvx_vec_splat_f32(s - M);
                vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                S += vs;
            }
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, S_inv);

        // Store result
        // dst indices
        const uint32_t i1 = iq1;
        const uint32_t i2 = iq2;
        const uint32_t i3 = iq3;

        // dst is permuted: [DV, n_heads, n_tokens, n_seq]
        // head stride is nb[1], token stride is nb[2], batch stride is nb[3]
        uint8_t * dst_ptr = (uint8_t *) dst->data + i2 * dst->nb[1] + i1 * dst->nb[2] + i3 * dst->nb[3];

        if (dst->type == HTP_TYPE_F32) {
            hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        } else if (dst->type == HTP_TYPE_F16) {
            hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, ir);
    }
}
int op_flash_attn_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q    = octx->src[0];

    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * mask = octx->src[3];
    const struct htp_tensor * dst  = octx->dst;

    // Check support
    if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_fa_kernel_params * kparams = (const struct htp_fa_kernel_params *) octx->kernel_params;

    if (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED) {
        return HTP_STATUS_NO_SUPPORT;
    }

    struct htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.t_start = HAP_perf_get_qtimer_count();

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.broadcast_rk2 = kparams->broadcast_rk2;
    factx.broadcast_rk3 = kparams->broadcast_rk3;
    factx.broadcast_rv2 = kparams->broadcast_rv2;
    factx.broadcast_rv3 = kparams->broadcast_rv3;

    if (mask) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (kparams->is_q_fp32 != 0);
    factx.size_q_row_padded = kparams->u.hvx.size_q_row_padded;
    factx.size_k_row_padded = kparams->u.hvx.size_k_row_padded;
    factx.size_v_row_padded = kparams->u.hvx.size_v_row_padded;

    size_t size_q_block = factx.size_q_row_padded * 1; // single row for now
    factx.size_k_block = factx.size_k_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_v_block = factx.size_v_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_m_block = hex_round_up(FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16), 128);

    factx.n_blocks = kparams->n_kv_blocks;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.logit_softcap = (__fp16) kparams->logit_softcap;

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    const uint32_t n_head = q->ne[2];
    if (n_head > 512) {
        return HTP_STATUS_NO_SUPPORT;
    }
    for (uint32_t h = 0; h < n_head; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_slope(h, factx.n_head_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    factx.qrows = kparams->qrows;
    factx.qrows_per_thread = kparams->qrows_per_thread;

    size_t size_vkq_acc = hex_round_up(v->ne[0] * sizeof(float), 128); // VKQ32

    factx.size_q_block = size_q_block;
    factx.size_vkq_acc = size_vkq_acc;

    uint8_t * vtcm_cur = octx->ctx->vtcm_base;

    factx.spad_q = vtcm_seq_alloc(&vtcm_cur, size_q_block * octx->n_threads);
    factx.spad_k = vtcm_seq_alloc(&vtcm_cur, factx.size_k_block * 2 * octx->n_threads);
    factx.spad_v = vtcm_seq_alloc(&vtcm_cur, factx.size_v_block * 2 * octx->n_threads);
    factx.spad_m = vtcm_seq_alloc(&vtcm_cur, (mask ? factx.size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * octx->n_threads);
    factx.spad_a = vtcm_seq_alloc(&vtcm_cur, size_vkq_acc * octx->n_threads);

    if ((size_t) (vtcm_cur - octx->ctx->vtcm_base) > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        work_queue_run(octx->ctx->work_queue, flash_attn_ext_f16_thread, &factx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}
