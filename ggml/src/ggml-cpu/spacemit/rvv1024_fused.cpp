#include "rvv1024_fused.h"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <cstdlib>

// GCC's standard RVV header, NOT the SpacemiT/IME intrinsic interface.
#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
#    include <riscv_vector.h>
#endif

namespace spacemit_kernels::rvv::fused1024 {
bool hardware_vlen_supported() noexcept {
#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
    return __riscv_vlenb() == VLEN_BYTES;
#else
    return false;
#endif
}

void repack_q4_0_columns(const ggml_tensor * weight, void * dst, std::size_t capacity_bytes) {
    (void) weight;
    (void) dst;
    (void) capacity_bytes;
    GGML_ABORT("RVV1024: Q4_0 column-SoA repacking not implemented");
}

void repack_kv_q4_0_interleaved(const ggml_tensor * wk, const ggml_tensor * wv,
                               void * dst, std::size_t capacity_bytes) {
    (void) wk;
    (void) wv;
    (void) dst;
    (void) capacity_bytes;
    GGML_ABORT("RVV1024: interleaved K/V Q4_0 repacking not implemented");
}

void forward_kv_project(const ggml_compute_params * params, ggml_tensor * dst) {
    (void) params;
    (void) dst;
    GGML_ABORT("RVV1024: K/V projection and cache update not implemented");
}

bool prefill_kernel_ready() noexcept {
    // Hardware check ONLY. The graph builder MUST reject ordinary GGUF Q4_0
    // Q/W_O tensors until the model-lifetime packed-weight loader exists.
#if defined(__riscv_v) && defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    return hardware_vlen_supported();
#else
    return false;
#endif
}

// Three-head register-fused Q -> RoPE -> attention -> Q8 -> W_O.
// All Q/W_O weight accesses below are UNIT-STRIDE into packed column-SoA
// tensors; the original row-major Q4_0 path is intentionally removed.
// No such packed tensors are currently supplied by the graph builder; it must
// fall back to GGML until the packed-weight loader is integrated.
//
// The caller must supply the exact Qwen2 NEOX RoPE base and frequency scale.
// Other RoPE schemes (YaRN, per-dimension frequency tables, etc.) need a
// different coefficient provider. The GGML attention mask, not a guessed
// prefix+t+1, determines which KV-cache slots are visible to each token.
#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
[[maybe_unused]] static void fused_three_head_prefill_group(
        ggml_tensor * parts, int group) {
    constexpr size_t VL = 64;     // 64 FP32 lanes in m2 at VLEN=1024
    constexpr size_t WO_VL = WO_SUBTILE_OUTPUTS / 2; // 128 FP32 lanes in m4
    constexpr size_t QK = Q4_BLOCK_SIZE;
    constexpr int Q_BLOCKS = Q_BLOCKS_PER_ROW;
    constexpr float ATTN_SCALE = 0.08838834764831845f; // 1/sqrt(128)

    GGML_ASSERT(parts && parts->data && group >= 0 && group < N_Q_GROUPS);
    GGML_ASSERT(ggml_is_contiguous(parts) && parts->type == GGML_TYPE_F32 &&
                parts->ne[0] == HIDDEN_DIM && parts->ne[2] == N_Q_GROUPS);
    GGML_ASSERT(hardware_vlen_supported());

    // Sources match src/llama-rvv1024-scaffold.cpp's custom_4d args[].
    const ggml_tensor * prepared = parts->src[0]; // I8 prepared X/RoPE [2240, ntok]
    const ggml_tensor * wq  = parts->src[1];  // packed I8 [Q_PACKED_BYTES]
    const ggml_tensor * bq  = parts->src[2];  // optional Q bias [1536]
    const ggml_tensor * wo  = parts->src[3];  // packed I8 [WO_PACKED_BYTES]
    const ggml_tensor * kc  = parts->src[4];  // FP16 [128, 2, n_kv, 1]
    const ggml_tensor * vc  = parts->src[5];  // FP16 normal or V-transposed
    const ggml_tensor * mask= parts->src[6];  // F16/F32 [n_kv, ntok, 1, 1]
    const ggml_tensor * kcols= parts->src[8]; // optional [64 positions, 128 dims, 2 heads, blocks]
    GGML_ASSERT(prepared && wq && wo && kc && vc && mask);
    GGML_ASSERT(prepared->type == GGML_TYPE_I8 && ggml_is_contiguous(prepared) &&
                prepared->ne[0] == static_cast<int64_t>(sizeof(prefill_prepared_token)) &&
                prepared->ne[1] == parts->ne[1] && prepared->nb[0] == 1);
    // Packed blobs are byte-addressable tensors with exact ABI size.
    // Never reinterpret an ordinary GGML Q4_0 weight matrix as column-SoA.
    GGML_ASSERT(wq->type == GGML_TYPE_I8 && wo->type == GGML_TYPE_I8 &&
                wq->ne[0] == static_cast<int64_t>(Q_PACKED_BYTES) &&
                wo->ne[0] == static_cast<int64_t>(WO_PACKED_BYTES) &&
                wq->ne[1] == 1 && wq->ne[2] == 1 && wq->ne[3] == 1 &&
                wo->ne[1] == 1 && wo->ne[2] == 1 && wo->ne[3] == 1 &&
                wq->nb[0] == 1 && wo->nb[0] == 1 &&
                wq->data && wo->data);
    GGML_ASSERT(!bq || (bq->type == GGML_TYPE_F32 && bq->ne[0] == HIDDEN_DIM && bq->nb[0] == 4));
    GGML_ASSERT(kc->type == GGML_TYPE_F16 && vc->type == GGML_TYPE_F16 &&
                kc->ne[0] == HEAD_DIM && kc->ne[1] == N_KV_HEADS && kc->ne[3] == 1 &&
                kc->nb[0] == 2 && vc->ne[3] == 1);
    const int64_t n_kv = kc->ne[2];
    if (kcols) {
        GGML_ASSERT(kcols->type == GGML_TYPE_F16 && kcols->data &&
                    kcols->ne[0] == 64 && kcols->ne[1] == HEAD_DIM &&
                    kcols->ne[2] == N_KV_HEADS && kcols->ne[3] >= (n_kv + 63) / 64 &&
                    kcols->nb[0] == 2 && kcols->nb[1] == 128 &&
                    kcols->nb[2] == 16384 && kcols->nb[3] == 32768);
    }
    const bool v_trans = vc->nb[1] > vc->nb[2];
    GGML_ASSERT(((v_trans && vc->ne[0] == n_kv && vc->ne[1] == N_KV_HEADS && vc->ne[2] == HEAD_DIM) ||
                 (!v_trans && vc->ne[0] == HEAD_DIM && vc->ne[1] == N_KV_HEADS && vc->ne[2] == n_kv)) &&
                vc->nb[0] == 2 && n_kv > 0);
    GGML_ASSERT((mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) &&
                mask->ne[0] >= n_kv && mask->ne[1] >= prepared->ne[1] &&
                mask->nb[0] == (mask->type == GGML_TYPE_F16 ? 2u : 4u));

    const int first_head = static_cast<int>(group_first_q_head(group));
    const int kv_head = static_cast<int>(group_kv_head(group));

    const uint8_t * q_tile  = static_cast<const uint8_t *>(wq->data) + group * Q_GROUP_BYTES;
    const uint8_t * wo_tile = static_cast<const uint8_t *>(wo->data) + group * WO_GROUP_BYTES;

    // W_O linear projection: one packed load updates BOTH nibble halves.
    // Keep two FP32 output accumulators in RVV registers for each 256-row tile.
    const auto wo_block_both = [&](const uint8_t * packed_block,
                                   const int8_t * q8_bytes, float x_scale,
                                   vfloat32m4_t & acc_lo, vfloat32m4_t & acc_hi) {
        vint16m2_t isum_lo = __riscv_vmv_v_x_i16m2(0, WO_VL);
        vint16m2_t isum_hi = __riscv_vmv_v_x_i16m2(0, WO_VL);
        for (int k = 0; k < static_cast<int>(QK); ++k) {
            // Q8 was staged once, before the six W_O output subtiles.
            const int8_t xk = q8_bytes[k];
            const vuint8m1_t packed = __riscv_vle8_v_u8m1(
                packed_block + k * WO_COL_BYTES, WO_VL);
            const vint8m1_t lo8 = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vsub_vx_u8m1(__riscv_vand_vx_u8m1(packed, 15, WO_VL), 8, WO_VL));
            const vint8m1_t hi8 = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vsub_vx_u8m1(__riscv_vsrl_vx_u8m1(packed, 4, WO_VL), 8, WO_VL));
            isum_lo = __riscv_vwmacc_vx_i16m2(isum_lo, xk, lo8, WO_VL);
            isum_hi = __riscv_vwmacc_vx_i16m2(isum_hi, xk, hi8, WO_VL);
        }
        const float * scales = reinterpret_cast<const float *>(packed_block + QK * WO_COL_BYTES);
        acc_lo = __riscv_vfadd_vv_f32m4(acc_lo,
            __riscv_vfmul_vf_f32m4(
                __riscv_vfmul_vv_f32m4(__riscv_vfwcvt_f_x_v_f32m4(isum_lo, WO_VL),
                    __riscv_vle32_v_f32m4(scales, WO_VL), WO_VL), x_scale, WO_VL), WO_VL);
        acc_hi = __riscv_vfadd_vv_f32m4(acc_hi,
            __riscv_vfmul_vf_f32m4(
                __riscv_vfmul_vv_f32m4(__riscv_vfwcvt_f_x_v_f32m4(isum_hi, WO_VL),
                    __riscv_vle32_v_f32m4(scales + WO_VL, WO_VL), WO_VL), x_scale, WO_VL), WO_VL);
    };

    // Weight-stationary two-token W_O: ONE packed vector load supplies both
    // independent Q8 x Q4_0 dot products. Per-token Q8/weight scales and the
    // original FP32 accumulation order are preserved. This is not vwmacc.vv:
    // lanes remain output channels, while each token supplies one Q8 scalar.
    const auto wo_block_two_tokens = [&](const uint8_t * packed_block,
                                         const int8_t * q8_first, const int8_t * q8_second,
                                         float scale_first, float scale_second,
                                         vfloat32m4_t & first_lo, vfloat32m4_t & first_hi,
                                         vfloat32m4_t & second_lo, vfloat32m4_t & second_hi) {
        vint16m2_t sum0_lo = __riscv_vmv_v_x_i16m2(0, WO_VL);
        vint16m2_t sum0_hi = __riscv_vmv_v_x_i16m2(0, WO_VL);
        vint16m2_t sum1_lo = __riscv_vmv_v_x_i16m2(0, WO_VL);
        vint16m2_t sum1_hi = __riscv_vmv_v_x_i16m2(0, WO_VL);
        for (int k = 0; k < static_cast<int>(QK); ++k) {
            const int8_t x0 = q8_first[k];
            const int8_t x1 = q8_second[k];
            const vuint8m1_t packed = __riscv_vle8_v_u8m1(
                packed_block + k * WO_COL_BYTES, WO_VL);
            const vint8m1_t lo8 = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vsub_vx_u8m1(__riscv_vand_vx_u8m1(packed, 15, WO_VL), 8, WO_VL));
            const vint8m1_t hi8 = __riscv_vreinterpret_v_u8m1_i8m1(
                __riscv_vsub_vx_u8m1(__riscv_vsrl_vx_u8m1(packed, 4, WO_VL), 8, WO_VL));
            sum0_lo = __riscv_vwmacc_vx_i16m2(sum0_lo, x0, lo8, WO_VL);
            sum0_hi = __riscv_vwmacc_vx_i16m2(sum0_hi, x0, hi8, WO_VL);
            sum1_lo = __riscv_vwmacc_vx_i16m2(sum1_lo, x1, lo8, WO_VL);
            sum1_hi = __riscv_vwmacc_vx_i16m2(sum1_hi, x1, hi8, WO_VL);
        }
        const float * scales = reinterpret_cast<const float *>(packed_block + QK * WO_COL_BYTES);
        const vfloat32m4_t w_lo = __riscv_vle32_v_f32m4(scales, WO_VL);
        first_lo = __riscv_vfadd_vv_f32m4(first_lo,
            __riscv_vfmul_vf_f32m4(
                __riscv_vfmul_vv_f32m4(__riscv_vfwcvt_f_x_v_f32m4(sum0_lo, WO_VL),
                    w_lo, WO_VL), scale_first, WO_VL), WO_VL);
        second_lo = __riscv_vfadd_vv_f32m4(second_lo,
            __riscv_vfmul_vf_f32m4(
                __riscv_vfmul_vv_f32m4(__riscv_vfwcvt_f_x_v_f32m4(sum1_lo, WO_VL),
                    w_lo, WO_VL), scale_second, WO_VL), WO_VL);
        const vfloat32m4_t w_hi = __riscv_vle32_v_f32m4(scales + WO_VL, WO_VL);
        first_hi = __riscv_vfadd_vv_f32m4(first_hi,
            __riscv_vfmul_vf_f32m4(
                __riscv_vfmul_vv_f32m4(__riscv_vfwcvt_f_x_v_f32m4(sum0_hi, WO_VL),
                    w_hi, WO_VL), scale_first, WO_VL), WO_VL);
        second_hi = __riscv_vfadd_vv_f32m4(second_hi,
            __riscv_vfmul_vf_f32m4(
                __riscv_vfmul_vv_f32m4(__riscv_vfwcvt_f_x_v_f32m4(sum1_hi, WO_VL),
                    w_hi, WO_VL), scale_second, WO_VL), WO_VL);
    };

    // Load 64 FP16 K/V elements, supporting GGML's normal and transposed V.
    const auto f16_load64 = [&](const uint8_t * ptr, ptrdiff_t step) -> vfloat32m2_t {
        const vuint16m1_t bits = step == 2 ?
            __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(ptr), VL) :
            __riscv_vlse16_v_u16m1(reinterpret_cast<const uint16_t *>(ptr), step, VL);
        return __riscv_vfwcvt_f_f_v_f32m2(
            __riscv_vreinterpret_v_u16m1_f16m1(bits), VL);
    };

    // Same call-free scalar approximation used by simple-inference's fused
    // two-head kernel; x is nonpositive and relative error is checked later.
    const auto exp_neg = [&](float xval) -> float {
        if (xval < -85.0f) return 0.0f;
        if (xval >= 0.0f) return 1.0f;
        const float z = xval * 1.4426950408889634f;
        const int n = static_cast<int>(z < 0.0f ? z - 0.5f : z + 0.5f);
        const float r = xval - static_cast<float>(n) * 0.6931471805599453f;
        float p = 1.0f / 5040.0f;
        p = p*r + 1.0f/720.0f; p = p*r + 1.0f/120.0f;
        p = p*r + 1.0f/24.0f;  p = p*r + 1.0f/6.0f;
        p = p*r + 0.5f;        p = p*r + 1.0f;
        p = p*r + 1.0f;
        const uint32_t ebits = static_cast<uint32_t>(n + 127) << 23;
        float pow2;
        __builtin_memcpy(&pow2, &ebits, sizeof(pow2));
        return p * pow2;
    };

    // Directly turn 32 attention FP32 lanes into a Q8 register. FP16-round
    // the block scale as GGML Q8_0 does, without writing attention O to RAM.
    const auto quant32 = [&](vfloat32m2_t src, int high_half,
                             float & rounded_scale) -> vint8mf2_t {
        const vfloat32m2_t part = high_half ?
            __riscv_vslidedown_vx_f32m2(src, 32, QK) : src;
        const vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        const float maxabs = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m2_f32m1(
                __riscv_vfabs_v_f32m2(part, QK), seed, QK));
        const float d = maxabs / 127.0f;
        const vfloat16m1_t half_d = __riscv_vfncvt_f_f_w_f16m1(
            __riscv_vfmv_v_f_f32m2(d, 1), 1);
        rounded_scale = __riscv_vfmv_f_s_f32m2_f32(
            __riscv_vfwcvt_f_f_v_f32m2(half_d, 1));
        const float inv = d == 0.0f ? 0.0f : 1.0f / d;
        vint32m2_t q = __riscv_vfcvt_x_f_v_i32m2(
            __riscv_vfmul_vf_f32m2(part, inv, QK), QK);
        q = __riscv_vmax_vx_i32m2(q, -127, QK);
        q = __riscv_vmin_vx_i32m2(q, 127, QK);
        return __riscv_vncvt_x_x_w_i8mf2(
            __riscv_vncvt_x_x_w_i16m1(q, QK), QK);
    };

    // Experiment: project four tokens while each packed W_Q block is resident.
    // This is ordinary, cacheable per-worker stack storage (3 KiB), not a
    // hardware scratchpad or pinned L1. Attention and W_O remain two-token.
    alignas(64) _Float16 q_four[4][Q_HEADS_TILE][HEAD_DIM];

    const auto project_q_four = [&](int64_t first_token) {
        const prefill_prepared_token * input[4];
        for (int t = 0; t < 4; ++t) {
            const int64_t idx = first_token + t < prepared->ne[1]
                ? first_token + t : prepared->ne[1] - 1;
            input[t] = reinterpret_cast<const prefill_prepared_token *>(
                static_cast<const uint8_t *>(prepared->data) + idx * prepared->nb[1]);
        }
        const auto z = __riscv_vfmv_v_f_f32m2(0.0f, VL);
        const float * group_bias = bq
            ? static_cast<const float *>(bq->data) + first_head * HEAD_DIM : nullptr;

        // Process only one 128-output head at a time. All four tokens reuse
        // each 64-byte Q4 load, but other heads' FP32 sums are NOT live.
        for (int head = 0; head < Q_HEADS_TILE; ++head) {
            const float * head_bias = group_bias ? group_bias + head * HEAD_DIM : nullptr;
            const vfloat32m2_t bias_lo = head_bias ? __riscv_vle32_v_f32m2(head_bias, VL) : z;
            const vfloat32m2_t bias_hi = head_bias ? __riscv_vle32_v_f32m2(head_bias + VL, VL) : z;
            vfloat32m2_t q0_lo = bias_lo, q0_hi = bias_hi;
            vfloat32m2_t q1_lo = bias_lo, q1_hi = bias_hi;
            vfloat32m2_t q2_lo = bias_lo, q2_hi = bias_hi;
            vfloat32m2_t q3_lo = bias_lo, q3_hi = bias_hi;

            const auto add_block = [&](vfloat32m2_t &dst, vint16m1_t acc,
                                       vfloat32m2_t weight_scale, float x_scale) {
                dst = __riscv_vfadd_vv_f32m2(dst,
                    __riscv_vfmul_vf_f32m2(
                        __riscv_vfmul_vv_f32m2(
                            __riscv_vfwcvt_f_x_v_f32m2(acc, VL), weight_scale, VL),
                        x_scale, VL), VL);
            };

            for (int b = 0; b < Q_BLOCKS; ++b) {
                const uint8_t * block = q_tile + static_cast<size_t>(b) * Q_BLOCK_BYTES;
                vint16m1_t s0_lo = __riscv_vmv_v_x_i16m1(0, VL);
                vint16m1_t s0_hi = __riscv_vmv_v_x_i16m1(0, VL);
                vint16m1_t s1_lo = __riscv_vmv_v_x_i16m1(0, VL);
                vint16m1_t s1_hi = __riscv_vmv_v_x_i16m1(0, VL);
                vint16m1_t s2_lo = __riscv_vmv_v_x_i16m1(0, VL);
                vint16m1_t s2_hi = __riscv_vmv_v_x_i16m1(0, VL);
                vint16m1_t s3_lo = __riscv_vmv_v_x_i16m1(0, VL);
                vint16m1_t s3_hi = __riscv_vmv_v_x_i16m1(0, VL);

                for (int k = 0; k < static_cast<int>(QK); ++k) {
                    const uint8_t * col = block + k * Q_COL_BYTES + head * (HEAD_DIM / 2);
                    const vuint8mf2_t packed = __riscv_vle8_v_u8mf2(col, VL);
                    const vint8mf2_t lo8 = __riscv_vreinterpret_v_u8mf2_i8mf2(
                        __riscv_vsub_vx_u8mf2(
                            __riscv_vand_vx_u8mf2(packed, 15, VL), 8, VL));
                    const vint8mf2_t hi8 = __riscv_vreinterpret_v_u8mf2_i8mf2(
                        __riscv_vsub_vx_u8mf2(
                            __riscv_vsrl_vx_u8mf2(packed, 4, VL), 8, VL));
                    s0_lo = __riscv_vwmacc_vx_i16m1(s0_lo, input[0]->x8[b].qs[k], lo8, VL);
                    s0_hi = __riscv_vwmacc_vx_i16m1(s0_hi, input[0]->x8[b].qs[k], hi8, VL);
                    s1_lo = __riscv_vwmacc_vx_i16m1(s1_lo, input[1]->x8[b].qs[k], lo8, VL);
                    s1_hi = __riscv_vwmacc_vx_i16m1(s1_hi, input[1]->x8[b].qs[k], hi8, VL);
                    s2_lo = __riscv_vwmacc_vx_i16m1(s2_lo, input[2]->x8[b].qs[k], lo8, VL);
                    s2_hi = __riscv_vwmacc_vx_i16m1(s2_hi, input[2]->x8[b].qs[k], hi8, VL);
                    s3_lo = __riscv_vwmacc_vx_i16m1(s3_lo, input[3]->x8[b].qs[k], lo8, VL);
                    s3_hi = __riscv_vwmacc_vx_i16m1(s3_hi, input[3]->x8[b].qs[k], hi8, VL);
                }

                const float * scales = reinterpret_cast<const float *>(
                    block + QK * Q_COL_BYTES) + head * HEAD_DIM;
                const vfloat32m2_t w_lo = __riscv_vle32_v_f32m2(scales, VL);
                const vfloat32m2_t w_hi = __riscv_vle32_v_f32m2(scales + VL, VL);
                add_block(q0_lo, s0_lo, w_lo, input[0]->x8[b].scale);
                add_block(q0_hi, s0_hi, w_hi, input[0]->x8[b].scale);
                add_block(q1_lo, s1_lo, w_lo, input[1]->x8[b].scale);
                add_block(q1_hi, s1_hi, w_hi, input[1]->x8[b].scale);
                add_block(q2_lo, s2_lo, w_lo, input[2]->x8[b].scale);
                add_block(q2_hi, s2_hi, w_hi, input[2]->x8[b].scale);
                add_block(q3_lo, s3_lo, w_lo, input[3]->x8[b].scale);
                add_block(q3_hi, s3_hi, w_hi, input[3]->x8[b].scale);
            }

            // Preserve the original NEOX FP32 RoPE then one FP16 rounding.
            const auto store_rope = [&](int t, vfloat32m2_t lo, vfloat32m2_t hi) {
                const vfloat32m2_t cv = __riscv_vle32_v_f32m2(input[t]->c, VL);
                const vfloat32m2_t sv = __riscv_vle32_v_f32m2(input[t]->s, VL);
                const vfloat32m2_t rotated_lo = __riscv_vfsub_vv_f32m2(
                    __riscv_vfmul_vv_f32m2(lo, cv, VL),
                    __riscv_vfmul_vv_f32m2(hi, sv, VL), VL);
                const vfloat32m2_t rotated_hi = __riscv_vfadd_vv_f32m2(
                    __riscv_vfmul_vv_f32m2(hi, cv, VL),
                    __riscv_vfmul_vv_f32m2(lo, sv, VL), VL);
                __riscv_vse16_v_f16m1(q_four[t][head],
                    __riscv_vfncvt_f_f_w_f16m1(rotated_lo, VL), VL);
                __riscv_vse16_v_f16m1(q_four[t][head] + VL,
                    __riscv_vfncvt_f_f_w_f16m1(rotated_hi, VL), VL);
            };
            store_rope(0, q0_lo, q0_hi);
            store_rope(1, q1_lo, q1_hi);
            store_rope(2, q2_lo, q2_hi);
            store_rope(3, q3_lo, q3_hi);
        }
        // Make the 3 KiB Q scratch a true memory staging boundary. Long KV
        // scans may evict it, but attention holds each pair's Q in registers.
        __asm__ __volatile__("" : "+m"(q_four));
    };

    // The sole RVV1024 attention path: K-column QK, paired queries, BQ8.
#include "rvv1024_bq8.inc"


}
#endif // standard RVV available

// New single-token path. Existing BQ8 prefill implementation is unchanged.
#include "rvv1024_decode6_split4.inc"

void forward_q_attn_wo(ggml_tensor * parts, int ith, int nth, void * userdata) {
    GGML_ASSERT(parts && parts->data &&
                parts->type == GGML_TYPE_F32 && ggml_is_contiguous(parts) &&
                parts->ne[0] == HIDDEN_DIM && parts->ne[1] > 1 &&
                parts->ne[2] == N_Q_GROUPS && parts->ne[3] == 1);
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);
    (void) userdata; // RoPE parameters are consumed by the preparation node.

    // GGML invokes each node on its prefill threadpool. Every active worker
    // owns entire three-head groups, never individual heads. No concurrent
    // writes to the same [group, token, hidden] FP32 output slab.
    if (ith >= N_Q_GROUPS) {
        return;
    }
    GGML_ASSERT(prefill_kernel_ready()); // checks THIS worker's RVV width
#if defined(__riscv_v) && defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    /*if (ith == 0) {*/
        /*fprintf(stderr,*/
                /*"RVV1024 FUSED EXECUTING: tokens=%lld, threads=%d, vlenb=%zu\n",*/
                /*(long long) parts->ne[1], nth,*/
                /*(size_t) __riscv_vlenb());*/
    /*}*/
    for (int group = ith; group < N_Q_GROUPS; group += nth) {
        // This processes all prompt tokens for three Q heads together:
        // ONE shared K/V traversal, three softmax states, direct partial W_O.
        fused_three_head_prefill_group(parts, group);
    }
#else
    GGML_ABORT("RVV1024: fused prefill requires 1024-bit RVV with Zvfh");
#endif
}

} // namespace spacemit_kernels::rvv::fused1024
