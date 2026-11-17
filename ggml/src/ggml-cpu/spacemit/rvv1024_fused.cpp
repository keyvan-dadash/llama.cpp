#include "rvv1024_fused.h"

#include "ggml.h"

#include <cmath>
#include <cstring>

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
    // The current three-head kernel consumes ordinary GGML Q4_0 weights, so it
    // needs no offline repacker or TCM. The graph builder checks the restricted
    // Qwen2 model/RoPE/KV contract before selecting this experimental path.
#if defined(__riscv_v) && defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    return hardware_vlen_supported();
#else
    return false;
#endif
}

// One *actual* three-head RVV attention/W_O computation for a single group.
// Called by forward_q_attn_wo only for the explicitly enabled experimental
// path on compatible A100 workers. This correctness-first version reads
// ordinary GGML row-major Q4_0 weights; the future column-SoA repacker will
// replace its strided weight loads, not its three-head attention algorithm.
//
// The caller must supply the exact Qwen2 NEOX RoPE base and frequency scale.
// Other RoPE schemes (YaRN, per-dimension frequency tables, etc.) need a
// different coefficient provider. The GGML attention mask, not a guessed
// prefix+t+1, determines which KV-cache slots are visible to each token.
#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
[[maybe_unused]] static void fused_three_head_prefill_group(
        ggml_tensor * parts, int group, float rope_base, float rope_freq_scale) {
    constexpr size_t VL = 64;   // 64 FP32 lanes in m2 at VLEN=1024
    constexpr size_t QK = 32;   // GGML Q4_0 / Q8_0 block width
    constexpr int Q_BLOCKS = HIDDEN_DIM / QK;
    constexpr float ATTN_SCALE = 0.08838834764831845f; // 1/sqrt(128)

    GGML_ASSERT(parts && parts->data && group >= 0 && group < N_Q_GROUPS);
    GGML_ASSERT(ggml_is_contiguous(parts) && parts->type == GGML_TYPE_F32 &&
                parts->ne[0] == HIDDEN_DIM && parts->ne[2] == N_Q_GROUPS);
    GGML_ASSERT(rope_base > 0.0f && rope_freq_scale > 0.0f);
    GGML_ASSERT(hardware_vlen_supported());

    // Sources match src/llama-rvv1024-scaffold.cpp's custom_4d args[].
    const ggml_tensor * x   = parts->src[0];  // normed input [1536, ntok]
    const ggml_tensor * wq  = parts->src[1];  // original GGML Q4_0 [1536,1536]
    const ggml_tensor * bq  = parts->src[2];  // optional Q bias [1536]
    const ggml_tensor * wo  = parts->src[3];  // original GGML Q4_0 [1536,1536]
    const ggml_tensor * pos = parts->src[4];  // I32 token positions
    const ggml_tensor * kc  = parts->src[5];  // FP16 [128, 2, n_kv, 1]
    const ggml_tensor * vc  = parts->src[6];  // FP16 normal or V-transposed
    const ggml_tensor * mask= parts->src[7];  // F16/F32 [n_kv, ntok, 1, 1]
    GGML_ASSERT(x && wq && wo && pos && kc && vc && mask);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->ne[0] == HIDDEN_DIM && x->nb[0] == 4 &&
                x->ne[1] == parts->ne[1]);
    GGML_ASSERT(pos->type == GGML_TYPE_I32 && pos->ne[0] >= x->ne[1] && pos->nb[0] == 4);
    GGML_ASSERT(wq->type == GGML_TYPE_Q4_0 && wo->type == GGML_TYPE_Q4_0 &&
                wq->ne[0] == HIDDEN_DIM && wq->ne[1] == HIDDEN_DIM &&
                wo->ne[0] == HIDDEN_DIM && wo->ne[1] == HIDDEN_DIM &&
                wq->nb[0] == 18 && wo->nb[0] == 18);
    GGML_ASSERT(!bq || (bq->type == GGML_TYPE_F32 && bq->ne[0] == HIDDEN_DIM && bq->nb[0] == 4));
    GGML_ASSERT(kc->type == GGML_TYPE_F16 && vc->type == GGML_TYPE_F16 &&
                kc->ne[0] == HEAD_DIM && kc->ne[1] == N_KV_HEADS && kc->ne[3] == 1 &&
                kc->nb[0] == 2 && vc->ne[3] == 1);
    const int64_t n_kv = kc->ne[2];
    const bool v_trans = vc->ne[0] == n_kv && vc->ne[1] == N_KV_HEADS && vc->ne[2] == HEAD_DIM;
    GGML_ASSERT((v_trans || (vc->ne[0] == HEAD_DIM && vc->ne[1] == N_KV_HEADS && vc->ne[2] == n_kv)) &&
                vc->nb[0] == 2 && n_kv > 0);
    GGML_ASSERT((mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32) &&
                mask->ne[0] >= n_kv && mask->ne[1] >= x->ne[1] &&
                mask->nb[0] == (mask->type == GGML_TYPE_F16 ? 2u : 4u));

    const int first_head = static_cast<int>(group_first_q_head(group));
    const int kv_head = static_cast<int>(group_kv_head(group));

    // Read 64 distinct output rows of an ORIGINAL GGML Q4_0 matrix for one
    // 32-column input block, multiplying by one Q8_0 activation register.
    // Row stride comes from ggml_tensor::nb[1] (no temporary packed layout).
    const auto q4_dot64 = [&](const ggml_tensor * w, int row, int block,
                              vint8mf2_t q8, float x_scale) -> vfloat32m2_t {
        const uint8_t * base = static_cast<const uint8_t *>(w->data) +
                               static_cast<size_t>(row) * w->nb[1] +
                               static_cast<size_t>(block) * 18;
        const ptrdiff_t row_stride = static_cast<ptrdiff_t>(w->nb[1]);
        vint32m2_t sum = __riscv_vmv_v_x_i32m2(0, VL);
        for (int k = 0; k < static_cast<int>(QK); ++k) {
            const int8_t xk = __riscv_vmv_x_s_i8mf2_i8(
                __riscv_vslidedown_vx_i8mf2(q8, k, QK));
            const vuint8mf2_t packed = __riscv_vlse8_v_u8mf2(base + 2 + (k & 15), row_stride, VL);
            const vuint8mf2_t nibble = k < 16 ?
                __riscv_vand_vx_u8mf2(packed, 15, VL) :
                __riscv_vsrl_vx_u8mf2(packed, 4, VL);
            const vint8mf2_t signed4 = __riscv_vreinterpret_v_u8mf2_i8mf2(
                __riscv_vsub_vx_u8mf2(nibble, 8, VL));
            sum = __riscv_vwmacc_vx_i32m2(sum, xk,
                __riscv_vsext_vf2_i16m1(signed4, VL), VL);
        }
        const vuint16m1_t scale_bits = __riscv_vlse16_v_u16m1(
            reinterpret_cast<const uint16_t *>(base), row_stride, VL);
        const vfloat32m2_t w_scale = __riscv_vfwcvt_f_f_v_f32m2(
            __riscv_vreinterpret_v_u16m1_f16m1(scale_bits), VL);
        return __riscv_vfmul_vf_f32m2(
            __riscv_vfmul_vv_f32m2(
                __riscv_vfcvt_f_x_v_f32m2(sum, VL), w_scale, VL),
            x_scale, VL);
    };

    // Load 64 FP16 K/V elements, supporting GGML's normal and transposed V.
    const auto f16_load64 = [&](const uint8_t * ptr, ptrdiff_t step) -> vfloat32m2_t {
        const vuint16m1_t bits = step == 2 ?
            __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(ptr), VL) :
            __riscv_vlse16_v_u16m1(reinterpret_cast<const uint16_t *>(ptr), step, VL);
        return __riscv_vfwcvt_f_f_v_f32m2(
            __riscv_vreinterpret_v_u16m1_f16m1(bits), VL);
    };

    const auto sum64 = [&](vfloat32m2_t z) -> float {
        const vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        return __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m2_f32m1(z, zero, VL));
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

    // Input quantization and RoPE coefficients are temporary per-token data.
    // There is no addressable Q[3,128] or O[3,128] tensor in this function.
    struct InputBlock { int8_t qs[32]; float scale; };
    InputBlock x8[Q_BLOCKS];
    float c[64], s[64];

    for (int64_t token = 0; token < x->ne[1]; ++token) {
        const float * xv = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(x->data) + token * x->nb[1]);
        for (int block = 0; block < Q_BLOCKS; ++block) {
            float maximum = 0.0f;
            for (int k = 0; k < static_cast<int>(QK); ++k) {
                const float v = __builtin_fabsf(xv[block*QK + k]);
                if (v > maximum) maximum = v;
            }
            const float scale = maximum / 127.0f;
            const float inverse = scale == 0.0f ? 0.0f : 1.0f/scale;
            x8[block].scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
            for (int k = 0; k < static_cast<int>(QK); ++k) {
                int q = static_cast<int>(__builtin_nearbyintf(xv[block*QK + k] * inverse));
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                x8[block].qs[k] = static_cast<int8_t>(q);
            }
        }

        const int32_t position = *reinterpret_cast<const int32_t *>(
            static_cast<const uint8_t *>(pos->data) + token * pos->nb[0]);
        for (int i = 0; i < 64; ++i) {
            const float theta = static_cast<float>(position) *
                __builtin_powf(rope_base, -static_cast<float>(i) / 64.0f) * rope_freq_scale;
            c[i] = __builtin_cosf(theta);
            s[i] = __builtin_sinf(theta);
        }

        // Six 64-lane FP32 RVV values are exactly three complete Q heads.
        const float * bias = bq ? static_cast<const float *>(bq->data) + first_head * HEAD_DIM : nullptr;
        const auto zero = __riscv_vfmv_v_f_f32m2(0.0f, VL);
        vfloat32m2_t q00 = bias ? __riscv_vle32_v_f32m2(bias +   0, VL) : zero;
        vfloat32m2_t q01 = bias ? __riscv_vle32_v_f32m2(bias +  64, VL) : zero;
        vfloat32m2_t q10 = bias ? __riscv_vle32_v_f32m2(bias + 128, VL) : zero;
        vfloat32m2_t q11 = bias ? __riscv_vle32_v_f32m2(bias + 192, VL) : zero;
        vfloat32m2_t q20 = bias ? __riscv_vle32_v_f32m2(bias + 256, VL) : zero;
        vfloat32m2_t q21 = bias ? __riscv_vle32_v_f32m2(bias + 320, VL) : zero;
        for (int b = 0; b < Q_BLOCKS; ++b) {
            const vint8mf2_t a = __riscv_vle8_v_i8mf2(x8[b].qs, QK);
            const float scale = x8[b].scale;
            q00 = __riscv_vfadd_vv_f32m2(q00, q4_dot64(wq, first_head*128 +   0, b, a, scale), VL);
            q01 = __riscv_vfadd_vv_f32m2(q01, q4_dot64(wq, first_head*128 +  64, b, a, scale), VL);
            q10 = __riscv_vfadd_vv_f32m2(q10, q4_dot64(wq, first_head*128 + 128, b, a, scale), VL);
            q11 = __riscv_vfadd_vv_f32m2(q11, q4_dot64(wq, first_head*128 + 192, b, a, scale), VL);
            q20 = __riscv_vfadd_vv_f32m2(q20, q4_dot64(wq, first_head*128 + 256, b, a, scale), VL);
            q21 = __riscv_vfadd_vv_f32m2(q21, q4_dot64(wq, first_head*128 + 320, b, a, scale), VL);
        }

        // Qwen2 NEOX rotate-half RoPE: [first 64] rotate with [last 64].
        const vfloat32m2_t cv = __riscv_vle32_v_f32m2(c, VL);
        const vfloat32m2_t sv = __riscv_vle32_v_f32m2(s, VL);
        const auto rope = [&](vfloat32m2_t & lo, vfloat32m2_t & hi) {
            const vfloat32m2_t old = lo;
            lo = __riscv_vfsub_vv_f32m2(__riscv_vfmul_vv_f32m2(old, cv, VL),
                                         __riscv_vfmul_vv_f32m2(hi, sv, VL), VL);
            hi = __riscv_vfadd_vv_f32m2(__riscv_vfmul_vv_f32m2(hi, cv, VL),
                                         __riscv_vfmul_vv_f32m2(old, sv, VL), VL);
        };
        rope(q00, q01); rope(q10, q11); rope(q20, q21);

        // Online streaming softmax for THREE heads over ONE shared KV pass.
        // No per-head KV traversal, and no Q/O tensor materialization.
        float m0 = -__builtin_inff(), m1 = -__builtin_inff(), m2 = -__builtin_inff();
        float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;
        vfloat32m2_t o00=zero, o01=zero, o10=zero, o11=zero, o20=zero, o21=zero;
        for (int64_t kv = 0; kv < n_kv; ++kv) {
            const uint8_t * mp = static_cast<const uint8_t *>(mask->data) +
                                 token * mask->nb[1] + kv * mask->nb[0];
            float mask_score;
            if (mask->type == GGML_TYPE_F32) {
                mask_score = *reinterpret_cast<const float *>(mp);
            } else {
                const vuint16m1_t mh = __riscv_vle16_v_u16m1(
                    reinterpret_cast<const uint16_t *>(mp), 1);
                mask_score = __riscv_vfmv_f_s_f32m2_f32(
                    __riscv_vfwcvt_f_f_v_f32m2(
                        __riscv_vreinterpret_v_u16m1_f16m1(mh), 1));
            }
            if (mask_score == -__builtin_inff()) continue;

            const uint8_t * kp = static_cast<const uint8_t *>(kc->data) +
                                 kv * kc->nb[2] + kv_head * kc->nb[1];
            const vfloat32m2_t k0 = f16_load64(kp, 2);
            const vfloat32m2_t k1 = f16_load64(kp + 64*kc->nb[0], 2);
            const auto score = [&](vfloat32m2_t low, vfloat32m2_t high) -> float {
                const vfloat32m2_t products = __riscv_vfadd_vv_f32m2(
                    __riscv_vfmul_vv_f32m2(low, k0, VL),
                    __riscv_vfmul_vv_f32m2(high, k1, VL), VL);
                return sum64(products) * ATTN_SCALE + mask_score;
            };
            const float s0 = score(q00, q01);
            const float s1 = score(q10, q11);
            const float s2 = score(q20, q21);

            const uint8_t * vp = static_cast<const uint8_t *>(vc->data) +
                (v_trans ? kv * vc->nb[0] + kv_head * vc->nb[1] :
                           kv * vc->nb[2] + kv_head * vc->nb[1]);
            const ptrdiff_t v_step = static_cast<ptrdiff_t>(v_trans ? vc->nb[2] : vc->nb[0]);
            const vfloat32m2_t v0 = f16_load64(vp, v_step);
            const vfloat32m2_t v1 = f16_load64(vp + 64*v_step, v_step);
            const auto update = [&](float scoreval, float & m, float & l,
                                    vfloat32m2_t & out0, vfloat32m2_t & out1) {
                if (l == 0.0f) {
                    m = scoreval; l = 1.0f;
                    out0 = v0; out1 = v1;
                } else {
                    const float new_m = scoreval > m ? scoreval : m;
                    const float a = exp_neg(m - new_m);
                    const float b = exp_neg(scoreval - new_m);
                    l = l*a + b; m = new_m;
                    out0 = __riscv_vfadd_vv_f32m2(
                        __riscv_vfmul_vf_f32m2(out0, a, VL),
                        __riscv_vfmul_vf_f32m2(v0, b, VL), VL);
                    out1 = __riscv_vfadd_vv_f32m2(
                        __riscv_vfmul_vf_f32m2(out1, a, VL),
                        __riscv_vfmul_vf_f32m2(v1, b, VL), VL);
                }
            };
            update(s0,m0,l0,o00,o01); update(s1,m1,l1,o10,o11); update(s2,m2,l2,o20,o21);
        }
        GGML_ASSERT(l0 > 0.0f && l1 > 0.0f && l2 > 0.0f);
        o00 = __riscv_vfmul_vf_f32m2(o00, 1.0f/l0, VL);
        o01 = __riscv_vfmul_vf_f32m2(o01, 1.0f/l0, VL);
        o10 = __riscv_vfmul_vf_f32m2(o10, 1.0f/l1, VL);
        o11 = __riscv_vfmul_vf_f32m2(o11, 1.0f/l1, VL);
        o20 = __riscv_vfmul_vf_f32m2(o20, 1.0f/l2, VL);
        o21 = __riscv_vfmul_vf_f32m2(o21, 1.0f/l2, VL);

        // All twelve Q8 blocks come directly from the six O RVV registers.
        float d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11;
        const vint8mf2_t z0 =quant32(o00,0,d0),  z1 =quant32(o00,1,d1);
        const vint8mf2_t z2 =quant32(o01,0,d2),  z3 =quant32(o01,1,d3);
        const vint8mf2_t z4 =quant32(o10,0,d4),  z5 =quant32(o10,1,d5);
        const vint8mf2_t z6 =quant32(o11,0,d6),  z7 =quant32(o11,1,d7);
        const vint8mf2_t z8 =quant32(o20,0,d8),  z9 =quant32(o20,1,d9);
        const vint8mf2_t z10=quant32(o21,0,d10), z11=quant32(o21,1,d11);

        // Dense W_O: each group contributes to every hidden output dimension.
        // Distinct workers only write their own [group,token] partial-Y slab.
        float * y = reinterpret_cast<float *>(static_cast<uint8_t *>(parts->data) +
            group * parts->nb[2] + token * parts->nb[1]);
        const int first_block = first_head * HEAD_DIM / QK;
        for (int output = 0; output < HIDDEN_DIM; output += 64) {
            vfloat32m2_t accum = zero;
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 0,z0, d0), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 1,z1, d1), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 2,z2, d2), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 3,z3, d3), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 4,z4, d4), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 5,z5, d5), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 6,z6, d6), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 7,z7, d7), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 8,z8, d8), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+ 9,z9, d9), VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+10,z10,d10),VL);
            accum = __riscv_vfadd_vv_f32m2(accum,q4_dot64(wo,output,first_block+11,z11,d11),VL);
            __riscv_vse32_v_f32m2(y + output, accum, VL);
        }
    }
}
#endif // standard RVV available

void forward_q_attn_wo(ggml_tensor * parts, int ith, int nth, void * userdata) {
    GGML_ASSERT(parts && parts->data &&
                parts->type == GGML_TYPE_F32 && ggml_is_contiguous(parts) &&
                parts->ne[0] == HIDDEN_DIM && parts->ne[1] > 1 &&
                parts->ne[2] == N_Q_GROUPS && parts->ne[3] == 1);
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth && userdata);

    // The GGML custom callback forwards a pointer to the config bytes stored
    // in a graph-owned tensor's op_params. Copy to avoid aliasing assumptions.
    prefill_rope_config rope;
    std::memcpy(&rope, userdata, sizeof(rope));
    GGML_ASSERT(std::isfinite(rope.base) && rope.base > 0.0f &&
                std::isfinite(rope.freq_scale) && rope.freq_scale > 0.0f);

    // GGML invokes each node on its prefill threadpool. Every active worker
    // owns entire three-head groups, never individual heads. No concurrent
    // writes to the same [group, token, hidden] FP32 output slab.
    if (ith >= N_Q_GROUPS) {
        return;
    }
    GGML_ASSERT(prefill_kernel_ready()); // checks THIS worker's RVV width
#if defined(__riscv_v) && defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    for (int group = ith; group < N_Q_GROUPS; group += nth) {
        // This processes all prompt tokens for three Q heads together:
        // ONE shared K/V traversal, three softmax states, direct partial W_O.
        fused_three_head_prefill_group(parts, group, rope.base, rope.freq_scale);
    }
#else
    GGML_ABORT("RVV1024: fused prefill requires 1024-bit RVV with Zvfh");
#endif
}

} // namespace spacemit_kernels::rvv::fused1024
