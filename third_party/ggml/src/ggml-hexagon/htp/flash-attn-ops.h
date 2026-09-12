#ifndef HTP_FLASH_ATTN_OPS_H
#define HTP_FLASH_ATTN_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HVX_FA_DMA_CACHE_SIZE  128

#define HTP_FA_M_INITIAL_VAL  -10000.0f

enum htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX
};

struct htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  is_q_fp32;          // 1 = Q type is F32, 0 = F16
    uint8_t  is_dst_fp32;        // 1 = dst type is F32, 0 = F16
    uint8_t  n_threads;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_blocks;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    float    scale;
    float    max_bias;
    float    logit_softcap;
    uint32_t vtcm_size;

    uint32_t qrows;
    uint32_t qrows_per_thread;
    float    m0;
    float    m1;
    uint32_t n_head_log2;

    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

    union {
        struct {
            uint32_t size_q_row_padded;
            uint32_t size_k_row_padded;
            uint32_t size_v_row_padded;
            struct fastdiv_values src0_div21;
            struct fastdiv_values src0_div1;
        } hvx;
    } u;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#endif

#define FA_HVX_BLOCK_SIZE 64

static inline size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_mask, size_t n_threads) {
    const size_t size_q_row_padded = hex_round_up(DK * (is_q_fp32 ? 4 : 2), 128);
    const size_t size_k_row_padded = hex_round_up(DK * sizeof(__fp16), 128);
    const size_t size_v_row_padded = hex_round_up(DV * sizeof(__fp16), 128);

    const size_t size_q_block = size_q_row_padded * 1;
    const size_t size_k_block = size_k_row_padded * FA_HVX_BLOCK_SIZE;
    const size_t size_v_block = size_v_row_padded * FA_HVX_BLOCK_SIZE;
    const size_t size_m_block = hex_round_up(FA_HVX_BLOCK_SIZE * sizeof(__fp16), 128);
    const size_t size_vkq_acc = hex_round_up(DV * sizeof(float), 128);

    const size_t size_per_thread = size_q_block * 1
                                 + size_k_block * 2
                                 + size_v_block * 2
                                 + (has_mask ? size_m_block * HVX_FA_DMA_CACHE_SIZE : 0)
                                 + size_vkq_acc;

    return size_per_thread * n_threads;
}

#define FA_MIN_KV_BLOCKS 3

#ifdef __cplusplus
}
#endif

#endif /* HTP_FLASH_ATTN_OPS_H */
