#include "rvv1024_fused.h"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// GCC's standard RVV header, NOT the SpacemiT/IME intrinsic interface.
#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
#    include <riscv_vector.h>
#endif

namespace spacemit_kernels::rvv::fused1024 {
// Diagnostic-only timing: see LLAMA_RVV1024_PHASE_PROFILE below.
// The default path does not read counters or print anything.
#if defined(__riscv_v) && defined(__riscv_v_intrinsic)
namespace {

// Each record measures one three-Q-head group. Q+RoPE runs once per token
// pair; attention and Q8+W_O each run once per *real* token.
struct phase_local {
    uint64_t q_rope = 0;
    uint64_t attention = 0;
    uint64_t q8_wo = 0;
    uint64_t pairs = 0;
    uint64_t tokens = 0;
};

struct phase_shared {
    std::atomic<uint64_t> q_rope{0};
    std::atomic<uint64_t> attention{0};
    std::atomic<uint64_t> q8_wo{0};
    std::atomic<uint64_t> pairs{0};
    std::atomic<uint64_t> tokens{0};
};

// 0 = disabled; 1 = CLOCK_MONOTONIC_RAW nanoseconds (safe); 2 = rdcycle ticks.
static int phase_mode() {
    static const int mode = [] {
        const char * env = std::getenv("LLAMA_RVV1024_PHASE_PROFILE");
        if (env && std::strcmp(env, "ns") == 0)     return 1;
        if (env && std::strcmp(env, "cycles") == 0) return 2;
        return 0;
    }();
    return mode;
}

// In 'cycles' mode this is one inlined CSR read, not a function call.
// RISC-V Linux may forbid U-mode rdcycle (SIGILL); use 'ns' in that case.
static inline uint64_t phase_tick(int mode) {
    if (mode == 0) return 0;
    if (mode == 2) {
        uint64_t ticks;
        __asm__ __volatile__("rdcycle %0" : "=r"(ticks) :: "memory");
        return ticks;
    }
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

struct phase_report {
    phase_shared groups[N_Q_GROUPS];

    ~phase_report() {
        uint64_t total_q = 0, total_attn = 0, total_wo = 0;
        uint64_t total_pairs = 0, total_tokens = 0;
        const char * unit = phase_mode() == 2 ? "cycles" : "ns";
        for (int g = 0; g < N_Q_GROUPS; ++g) {
            const auto & s = groups[g];
            const uint64_t q = s.q_rope.load(std::memory_order_relaxed);
            const uint64_t a = s.attention.load(std::memory_order_relaxed);
            const uint64_t w = s.q8_wo.load(std::memory_order_relaxed);
            const uint64_t p = s.pairs.load(std::memory_order_relaxed);
            const uint64_t t = s.tokens.load(std::memory_order_relaxed);
            if (p == 0) continue;
            total_q += q; total_attn += a; total_wo += w;
            total_pairs += p; total_tokens += t;
            std::fprintf(stderr,
                "[RVV1024_PHASE] group=%d pairs=%llu tokens=%llu "
                "q_rope=%llu attention=%llu q8_wo=%llu (%s)\n",
                g, (unsigned long long) p, (unsigned long long) t,
                (unsigned long long) q, (unsigned long long) a,
                (unsigned long long) w, unit);
        }
        if (total_pairs == 0) return;
        const double sum = double(total_q) + double(total_attn) + double(total_wo);
        std::fprintf(stderr,
            "[RVV1024_PHASE] TOTAL unit=%s pairs=%llu tokens=%llu "
            "Q+RoPE=%.1f%% attention=%.1f%% Q8+W_O=%.1f%% "
            "q_rope=%llu attention=%llu q8_wo=%llu\n",
            unit, (unsigned long long) total_pairs, (unsigned long long) total_tokens,
            100.0 * double(total_q) / sum,
            100.0 * double(total_attn) / sum,
            100.0 * double(total_wo) / sum,
            (unsigned long long) total_q, (unsigned long long) total_attn,
            (unsigned long long) total_wo);
        std::fprintf(stderr,
            "[RVV1024_PHASE] These are summed worker timestamps over all layers "
            "and runs, NOT wall time; X/RoPE preparation and partial-Y "
            "reduction are excluded.\n");
    }
};

static phase_report & phase_stats() {
    static phase_report stats;
    return stats;
}

static void phase_record(int group, const phase_local & local) {
    phase_shared & s = phase_stats().groups[group];
    s.q_rope.fetch_add(local.q_rope, std::memory_order_relaxed);
    s.attention.fetch_add(local.attention, std::memory_order_relaxed);
    s.q8_wo.fetch_add(local.q8_wo, std::memory_order_relaxed);
    s.pairs.fetch_add(local.pairs, std::memory_order_relaxed);
    s.tokens.fetch_add(local.tokens, std::memory_order_relaxed);
}
} // namespace
#endif // RVV phase timing

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

    const int timing_mode = phase_mode();
    phase_local timing{};

    // One 32-input block, ONE head (128 outputs). Per input column a single
    // contiguous e8mf2 load (64 bytes) supplies both output halves via its
    // low/high nibbles. Three calls cover the 384 outputs of the Q group.
    // Processing one head at a time keeps only four Q accumulators live
    // across input blocks: two halves for each of the two tokens.
    // Two-token Q4_0 x Q8 microkernel: each 64-byte packed weight vector
    // is loaded and unpacked once, then used for both token accumulators.
    const auto q_head_block = [&](const uint8_t * packed_block, int head,
                                  const int8_t * input0_q8, float input0_scale,
                                  const int8_t * input1_q8, float input1_scale,
                                  vfloat32m2_t & q0_lo, vfloat32m2_t & q0_hi,
                                  vfloat32m2_t & q1_lo, vfloat32m2_t & q1_hi) {
        // Each 32-term dot product is bounded by 32 * 127 * 8 = 32512.
        // Accumulate in i16 as the fast RVV repack Q4_0 kernel does.
        vint16m1_t s0_lo = __riscv_vmv_v_x_i16m1(0, VL);
        vint16m1_t s0_hi = __riscv_vmv_v_x_i16m1(0, VL);
        vint16m1_t s1_lo = __riscv_vmv_v_x_i16m1(0, VL);
        vint16m1_t s1_hi = __riscv_vmv_v_x_i16m1(0, VL);
        for (int k = 0; k < static_cast<int>(QK); ++k) {
            const uint8_t * col = packed_block + k * Q_COL_BYTES + head * (HEAD_DIM / 2);
            const vuint8mf2_t packed = __riscv_vle8_v_u8mf2(col, VL);
            const vuint8mf2_t lowbits = __riscv_vsub_vx_u8mf2(
                __riscv_vand_vx_u8mf2(packed, 15, VL), 8, VL);
            const vuint8mf2_t highbits = __riscv_vsub_vx_u8mf2(
                __riscv_vsrl_vx_u8mf2(packed, 4, VL), 8, VL);
            const vint8mf2_t lo8 = __riscv_vreinterpret_v_u8mf2_i8mf2(lowbits);
            const vint8mf2_t hi8 = __riscv_vreinterpret_v_u8mf2_i8mf2(highbits);
            s0_lo = __riscv_vwmacc_vx_i16m1(s0_lo, input0_q8[k], lo8, VL);
            s0_hi = __riscv_vwmacc_vx_i16m1(s0_hi, input0_q8[k], hi8, VL);
            s1_lo = __riscv_vwmacc_vx_i16m1(s1_lo, input1_q8[k], lo8, VL);
            s1_hi = __riscv_vwmacc_vx_i16m1(s1_hi, input1_q8[k], hi8, VL);
        }
        const float * scales = reinterpret_cast<const float *>(
            packed_block + QK * Q_COL_BYTES) + head * HEAD_DIM;
        const vfloat32m2_t scale_lo = __riscv_vle32_v_f32m2(scales, VL);
        const vfloat32m2_t scale_hi = __riscv_vle32_v_f32m2(scales + 64, VL);
        q0_lo = __riscv_vfadd_vv_f32m2(q0_lo,
            __riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(__riscv_vfwcvt_f_x_v_f32m2(s0_lo, VL), scale_lo, VL),
                input0_scale, VL), VL);
        q0_hi = __riscv_vfadd_vv_f32m2(q0_hi,
            __riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(__riscv_vfwcvt_f_x_v_f32m2(s0_hi, VL), scale_hi, VL),
                input0_scale, VL), VL);
        q1_lo = __riscv_vfadd_vv_f32m2(q1_lo,
            __riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(__riscv_vfwcvt_f_x_v_f32m2(s1_lo, VL), scale_lo, VL),
                input1_scale, VL), VL);
        q1_hi = __riscv_vfadd_vv_f32m2(q1_hi,
            __riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(__riscv_vfwcvt_f_x_v_f32m2(s1_hi, VL), scale_hi, VL),
                input1_scale, VL), VL);
    };

    // W_O linear projection: one packed load updates BOTH nibble halves.
    // Keep two FP32 output accumulators in RVV registers for each 256-row tile.
    const auto wo_block_both = [&](const uint8_t * packed_block,
                                   vint8mf2_t q8, float x_scale,
                                   vfloat32m4_t & acc_lo, vfloat32m4_t & acc_hi) {
        vint16m2_t isum_lo = __riscv_vmv_v_x_i16m2(0, WO_VL);
        vint16m2_t isum_hi = __riscv_vmv_v_x_i16m2(0, WO_VL);
        for (int k = 0; k < static_cast<int>(QK); ++k) {
            const int8_t xk = __riscv_vmv_x_s_i8mf2_i8(
                __riscv_vslidedown_vx_i8mf2(q8, k, QK));
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

    // One pair at a time. The separate graph node already quantized both
    // inputs and prepared both sets of NEOX RoPE coefficients.
    for (int64_t token_pair = 0; token_pair < prepared->ne[1]; token_pair += 2) {
        const uint64_t t_q_begin = phase_tick(timing_mode);
        const auto * shared0 = reinterpret_cast<const prefill_prepared_token *>(
            static_cast<const uint8_t *>(prepared->data) + token_pair * prepared->nb[1]);
        const bool has_second = token_pair + 1 < prepared->ne[1];
        // For an odd final token, reuse the first input in the unused second
        // accumulator. We never execute attention/W_O for that dummy token.
        const auto * shared1 = has_second ?
            reinterpret_cast<const prefill_prepared_token *>(
                static_cast<const uint8_t *>(prepared->data) + (token_pair + 1) * prepared->nb[1]) : shared0;
        const float * bias = bq ? static_cast<const float *>(bq->data) + first_head * HEAD_DIM : nullptr;
        const auto zero = __riscv_vfmv_v_f_f32m2(0.0f, VL);

        // Finish each head's two-token Q projection, apply per-token RoPE,
        // and narrow immediately to FP16. No addressable Q scratch arrays.
        // FP16 Q for both tokens remains live for the two attention passes.
        const auto rope = [&](vfloat32m2_t & lo, vfloat32m2_t & hi,
                              vfloat32m2_t cv, vfloat32m2_t sv) {
            const vfloat32m2_t old = lo;
            lo = __riscv_vfsub_vv_f32m2(__riscv_vfmul_vv_f32m2(old, cv, VL),
                                         __riscv_vfmul_vv_f32m2(hi, sv, VL), VL);
            hi = __riscv_vfadd_vv_f32m2(__riscv_vfmul_vv_f32m2(hi, cv, VL),
                                         __riscv_vfmul_vv_f32m2(old, sv, VL), VL);
        };

        const auto project_head_pair = [&](int head,
                                           vfloat16m1_t & t0_lo, vfloat16m1_t & t0_hi,
                                           vfloat16m1_t & t1_lo, vfloat16m1_t & t1_hi) {
            const float * head_bias = bias ? bias + head * HEAD_DIM : nullptr;
            vfloat32m2_t q0_lo = head_bias ? __riscv_vle32_v_f32m2(head_bias, VL) : zero;
            vfloat32m2_t q0_hi = head_bias ? __riscv_vle32_v_f32m2(head_bias + 64, VL) : zero;
            vfloat32m2_t q1_lo = q0_lo;
            vfloat32m2_t q1_hi = q0_hi;
            for (int b = 0; b < Q_BLOCKS; ++b) {
                const uint8_t * block = q_tile + static_cast<size_t>(b) * Q_BLOCK_BYTES;
                q_head_block(block, head,
                    shared0->x8[b].qs, shared0->x8[b].scale,
                    shared1->x8[b].qs, shared1->x8[b].scale,
                    q0_lo, q0_hi, q1_lo, q1_hi);
            }

            // Preserve NEOX RoPE in FP32, then round Q to FP16 once.
            const vfloat32m2_t cv0 = __riscv_vle32_v_f32m2(shared0->c, VL);
            const vfloat32m2_t sv0 = __riscv_vle32_v_f32m2(shared0->s, VL);
            rope(q0_lo, q0_hi, cv0, sv0);
            t0_lo = __riscv_vfncvt_f_f_w_f16m1(q0_lo, VL);
            t0_hi = __riscv_vfncvt_f_f_w_f16m1(q0_hi, VL);

            const vfloat32m2_t cv1 = __riscv_vle32_v_f32m2(shared1->c, VL);
            const vfloat32m2_t sv1 = __riscv_vle32_v_f32m2(shared1->s, VL);
            rope(q1_lo, q1_hi, cv1, sv1);
            t1_lo = __riscv_vfncvt_f_f_w_f16m1(q1_lo, VL);
            t1_hi = __riscv_vfncvt_f_f_w_f16m1(q1_hi, VL);
        };

        vfloat16m1_t qh00, qh01, qh10, qh11, qh20, qh21;
        vfloat16m1_t qh00_second, qh01_second, qh10_second, qh11_second, qh20_second, qh21_second;
        project_head_pair(0, qh00, qh01, qh00_second, qh01_second);
        project_head_pair(1, qh10, qh11, qh10_second, qh11_second);
        project_head_pair(2, qh20, qh21, qh20_second, qh21_second);
        const uint64_t t_q_end = phase_tick(timing_mode);
        if (timing_mode) {
            timing.q_rope += t_q_end - t_q_begin;
            ++timing.pairs;
        }

        // First token: keep the existing attention, online softmax and W_O.
        const int64_t token = token_pair;

        // Online streaming softmax for THREE heads over ONE shared KV pass.
        // No per-head KV traversal, and no Q/O tensor materialization.
        float m0 = -__builtin_inff(), m1 = -__builtin_inff(), m2 = -__builtin_inff();
        float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;
        vfloat32m2_t o00=zero, o01=zero, o10=zero, o11=zero, o20=zero, o21=zero;
        // The same blockwise online-softmax recurrence as ggml's CPU tiled
        // FlashAttention, directly in this fused kernel. Only 3 x 64 FP32 scores
        // are materialized (768 bytes); Q and the unnormalized FP32 U vectors stay
        // in the existing register-fused path. No full KQ/O tensor is produced.
        constexpr int ATTN_KV_TILE = 64;
        for (int64_t kv_base = 0; kv_base < n_kv; kv_base += ATTN_KV_TILE) {
            const int tile_len = static_cast<int>(
                n_kv - kv_base < ATTN_KV_TILE ? n_kv - kv_base : ATTN_KV_TILE);
            float score0[ATTN_KV_TILE], score1[ATTN_KV_TILE], score2[ATTN_KV_TILE];
            float tile_m0 = -__builtin_inff();
            float tile_m1 = -__builtin_inff();
            float tile_m2 = -__builtin_inff();

            // Stage 1: compute all three Q.K score rows against one K tile.
            // A K vector is fetched only once per key for these three heads.
            for (int t = 0; t < tile_len; ++t) {
                const int64_t kv = kv_base + t;
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
                if (mask_score == -__builtin_inff()) {
                    score0[t] = score1[t] = score2[t] = -__builtin_inff();
                    continue;
                }

                const uint8_t * kp = static_cast<const uint8_t *>(kc->data) +
                                     kv * kc->nb[2] + kv_head * kc->nb[1];
                vfloat16m1_t k0 = __riscv_vreinterpret_v_u16m1_f16m1(
                    __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(kp), VL));
                const vfloat16m1_t k1 = __riscv_vreinterpret_v_u16m1_f16m1(
                    __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(kp + 64*kc->nb[0]), VL));
                const vfloat32m2_t dot0 = __riscv_vfwmacc_vv_f32m2(
                    __riscv_vfwmul_vv_f32m2(qh00, k0, VL), qh01, k1, VL);
                score0[t] = sum64(dot0) * ATTN_SCALE + mask_score;
                // Preserve the original score-order constraint in this experiment.
                __asm__ __volatile__("" : "+vr"(k0) : "f"(score0[t]));
                const vfloat32m2_t dot1 = __riscv_vfwmacc_vv_f32m2(
                    __riscv_vfwmul_vv_f32m2(qh10, k0, VL), qh11, k1, VL);
                score1[t] = sum64(dot1) * ATTN_SCALE + mask_score;
                __asm__ __volatile__("" : "+vr"(k0) : "f"(score1[t]));
                const vfloat32m2_t dot2 = __riscv_vfwmacc_vv_f32m2(
                    __riscv_vfwmul_vv_f32m2(qh20, k0, VL), qh21, k1, VL);
                score2[t] = sum64(dot2) * ATTN_SCALE + mask_score;
                if (score0[t] > tile_m0) tile_m0 = score0[t];
                if (score1[t] > tile_m1) tile_m1 = score1[t];
                if (score2[t] > tile_m2) tile_m2 = score2[t];
            }

            // A fully causally masked tile contributes nothing to m, l, or U.
            if (tile_m0 == -__builtin_inff() &&
                tile_m1 == -__builtin_inff() && tile_m2 == -__builtin_inff()) continue;

            // Stage 2: merge the tile maximum into the running online softmax.
            // Rescale U at most ONCE per KV tile, not once per KV position.
            const auto merge_max = [&](float tile_m, float & m, float & l,
                                       vfloat32m2_t & u0, vfloat32m2_t & u1) {
                if (tile_m == -__builtin_inff()) return;
                const float new_m = tile_m > m ? tile_m : m;
                if (l != 0.0f && new_m > m) {
                    const float alpha = exp_neg(m - new_m);
                    l *= alpha;
                    u0 = __riscv_vfmul_vf_f32m2(u0, alpha, VL);
                    u1 = __riscv_vfmul_vf_f32m2(u1, alpha, VL);
                }
                m = new_m;
            };
            merge_max(tile_m0, m0, l0, o00, o01);
            merge_max(tile_m1, m1, l1, o10, o11);
            merge_max(tile_m2, m2, l2, o20, o21);

            // Stage 3: P_tile * V_tile, accumulating the same unnormalized FP32 U.
            // The V cache is read once per key and shared by the three heads.
            for (int t = 0; t < tile_len; ++t) {
                if (score0[t] == -__builtin_inff() &&
                    score1[t] == -__builtin_inff() && score2[t] == -__builtin_inff()) continue;
                const float p0 = score0[t] == -__builtin_inff() ? 0.0f : exp_neg(score0[t] - m0);
                const float p1 = score1[t] == -__builtin_inff() ? 0.0f : exp_neg(score1[t] - m1);
                const float p2 = score2[t] == -__builtin_inff() ? 0.0f : exp_neg(score2[t] - m2);
                l0 += p0; l1 += p1; l2 += p2;

                const int64_t kv = kv_base + t;
                const uint8_t * vp = static_cast<const uint8_t *>(vc->data) +
                    (v_trans ? kv * vc->nb[0] + kv_head * vc->nb[1] :
                               kv * vc->nb[2] + kv_head * vc->nb[1]);
                const ptrdiff_t v_step = static_cast<ptrdiff_t>(v_trans ? vc->nb[2] : vc->nb[0]);
                const vfloat32m2_t v0 = f16_load64(vp, v_step);
                const vfloat32m2_t v1 = f16_load64(vp + 64*v_step, v_step);
                o00 = __riscv_vfmacc_vf_f32m2(o00, p0, v0, VL);
                o01 = __riscv_vfmacc_vf_f32m2(o01, p0, v1, VL);
                o10 = __riscv_vfmacc_vf_f32m2(o10, p1, v0, VL);
                o11 = __riscv_vfmacc_vf_f32m2(o11, p1, v1, VL);
                o20 = __riscv_vfmacc_vf_f32m2(o20, p2, v0, VL);
                o21 = __riscv_vfmacc_vf_f32m2(o21, p2, v1, VL);
            }
        }
        GGML_ASSERT(l0 > 0.0f && l1 > 0.0f && l2 > 0.0f);
        o00 = __riscv_vfmul_vf_f32m2(o00, 1.0f/l0, VL);
        o01 = __riscv_vfmul_vf_f32m2(o01, 1.0f/l0, VL);
        o10 = __riscv_vfmul_vf_f32m2(o10, 1.0f/l1, VL);
        o11 = __riscv_vfmul_vf_f32m2(o11, 1.0f/l1, VL);
        o20 = __riscv_vfmul_vf_f32m2(o20, 1.0f/l2, VL);
        o21 = __riscv_vfmul_vf_f32m2(o21, 1.0f/l2, VL);
        const uint64_t t_attention0_end = phase_tick(timing_mode);
        if (timing_mode) timing.attention += t_attention0_end - t_q_end;

        // All twelve Q8 blocks come directly from the six O RVV registers.
        float d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11;
        const vint8mf2_t z0 =quant32(o00,0,d0),  z1 =quant32(o00,1,d1);
        const vint8mf2_t z2 =quant32(o01,0,d2),  z3 =quant32(o01,1,d3);
        const vint8mf2_t z4 =quant32(o10,0,d4),  z5 =quant32(o10,1,d5);
        const vint8mf2_t z6 =quant32(o11,0,d6),  z7 =quant32(o11,1,d7);
        const vint8mf2_t z8 =quant32(o20,0,d8),  z9 =quant32(o20,1,d9);
        const vint8mf2_t z10=quant32(o21,0,d10), z11=quant32(o21,1,d11);

        // Select one of the twelve live Q8 register blocks per W_O input
        // block. No addressable O[384] scratch buffer or calls across live
        // Q8 values. Only each group's final partial-Y is stored.
        float * y = reinterpret_cast<float *>(static_cast<uint8_t *>(parts->data) +
            group * parts->nb[2] + token * parts->nb[1]);
        for (int yt = 0; yt < WO_SUBTILES; ++yt) {
            const uint8_t * subtile = wo_tile + static_cast<size_t>(yt) * WO_SUBTILE_BYTES;
            vfloat32m4_t acc_lo = __riscv_vfmv_v_f_f32m4(0.0f, WO_VL);
            vfloat32m4_t acc_hi = __riscv_vfmv_v_f_f32m4(0.0f, WO_VL);
            // Static Q8-register selection. No runtime switch or vector array.
            // Each call updates both output halves from one weight load per k.
            wo_block_both(subtile +  0 * WO_BLOCK_BYTES, z0,  d0,  acc_lo, acc_hi);
            wo_block_both(subtile +  1 * WO_BLOCK_BYTES, z1,  d1,  acc_lo, acc_hi);
            wo_block_both(subtile +  2 * WO_BLOCK_BYTES, z2,  d2,  acc_lo, acc_hi);
            wo_block_both(subtile +  3 * WO_BLOCK_BYTES, z3,  d3,  acc_lo, acc_hi);
            wo_block_both(subtile +  4 * WO_BLOCK_BYTES, z4,  d4,  acc_lo, acc_hi);
            wo_block_both(subtile +  5 * WO_BLOCK_BYTES, z5,  d5,  acc_lo, acc_hi);
            wo_block_both(subtile +  6 * WO_BLOCK_BYTES, z6,  d6,  acc_lo, acc_hi);
            wo_block_both(subtile +  7 * WO_BLOCK_BYTES, z7,  d7,  acc_lo, acc_hi);
            wo_block_both(subtile +  8 * WO_BLOCK_BYTES, z8,  d8,  acc_lo, acc_hi);
            wo_block_both(subtile +  9 * WO_BLOCK_BYTES, z9,  d9,  acc_lo, acc_hi);
            wo_block_both(subtile + 10 * WO_BLOCK_BYTES, z10, d10, acc_lo, acc_hi);
            wo_block_both(subtile + 11 * WO_BLOCK_BYTES, z11, d11, acc_lo, acc_hi);
            __riscv_vse32_v_f32m4(y + yt * WO_SUBTILE_OUTPUTS, acc_lo, WO_VL);
            __riscv_vse32_v_f32m4(y + yt * WO_SUBTILE_OUTPUTS + WO_VL, acc_hi, WO_VL);
        }
        const uint64_t t_wo0_end = phase_tick(timing_mode);
        if (timing_mode) {
            timing.q8_wo += t_wo0_end - t_attention0_end;
            ++timing.tokens;
        }
        // Second token: the same attention and W_O, but with its own RoPE
        // coefficients and cached Q projection. No two-iteration token loop.
        if (has_second) {
            const int64_t token = token_pair + 1;
            // Use token 2's Q vectors prepared above; no Q loads or RoPE here.
            const vfloat16m1_t qh00 = qh00_second;
            const vfloat16m1_t qh01 = qh01_second;
            const vfloat16m1_t qh10 = qh10_second;
            const vfloat16m1_t qh11 = qh11_second;
            const vfloat16m1_t qh20 = qh20_second;
            const vfloat16m1_t qh21 = qh21_second;
            // Online streaming softmax for THREE heads over ONE shared KV pass.
            // No per-head KV traversal, and no Q/O tensor materialization.
            float m0 = -__builtin_inff(), m1 = -__builtin_inff(), m2 = -__builtin_inff();
            float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f;
            vfloat32m2_t o00=zero, o01=zero, o10=zero, o11=zero, o20=zero, o21=zero;
            // The same blockwise online-softmax recurrence as ggml's CPU tiled
            // FlashAttention, directly in this fused kernel. Only 3 x 64 FP32 scores
            // are materialized (768 bytes); Q and the unnormalized FP32 U vectors stay
            // in the existing register-fused path. No full KQ/O tensor is produced.
            constexpr int ATTN_KV_TILE = 64;
            for (int64_t kv_base = 0; kv_base < n_kv; kv_base += ATTN_KV_TILE) {
                const int tile_len = static_cast<int>(
                    n_kv - kv_base < ATTN_KV_TILE ? n_kv - kv_base : ATTN_KV_TILE);
                float score0[ATTN_KV_TILE], score1[ATTN_KV_TILE], score2[ATTN_KV_TILE];
                float tile_m0 = -__builtin_inff();
                float tile_m1 = -__builtin_inff();
                float tile_m2 = -__builtin_inff();

                // Stage 1: compute all three Q.K score rows against one K tile.
                // A K vector is fetched only once per key for these three heads.
                for (int t = 0; t < tile_len; ++t) {
                    const int64_t kv = kv_base + t;
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
                    if (mask_score == -__builtin_inff()) {
                        score0[t] = score1[t] = score2[t] = -__builtin_inff();
                        continue;
                    }

                    const uint8_t * kp = static_cast<const uint8_t *>(kc->data) +
                                         kv * kc->nb[2] + kv_head * kc->nb[1];
                    vfloat16m1_t k0 = __riscv_vreinterpret_v_u16m1_f16m1(
                        __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(kp), VL));
                    const vfloat16m1_t k1 = __riscv_vreinterpret_v_u16m1_f16m1(
                        __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(kp + 64*kc->nb[0]), VL));
                    const vfloat32m2_t dot0 = __riscv_vfwmacc_vv_f32m2(
                        __riscv_vfwmul_vv_f32m2(qh00, k0, VL), qh01, k1, VL);
                    score0[t] = sum64(dot0) * ATTN_SCALE + mask_score;
                    // Preserve the original score-order constraint in this experiment.
                    __asm__ __volatile__("" : "+vr"(k0) : "f"(score0[t]));
                    const vfloat32m2_t dot1 = __riscv_vfwmacc_vv_f32m2(
                        __riscv_vfwmul_vv_f32m2(qh10, k0, VL), qh11, k1, VL);
                    score1[t] = sum64(dot1) * ATTN_SCALE + mask_score;
                    __asm__ __volatile__("" : "+vr"(k0) : "f"(score1[t]));
                    const vfloat32m2_t dot2 = __riscv_vfwmacc_vv_f32m2(
                        __riscv_vfwmul_vv_f32m2(qh20, k0, VL), qh21, k1, VL);
                    score2[t] = sum64(dot2) * ATTN_SCALE + mask_score;
                    if (score0[t] > tile_m0) tile_m0 = score0[t];
                    if (score1[t] > tile_m1) tile_m1 = score1[t];
                    if (score2[t] > tile_m2) tile_m2 = score2[t];
                }

                // A fully causally masked tile contributes nothing to m, l, or U.
                if (tile_m0 == -__builtin_inff() &&
                    tile_m1 == -__builtin_inff() && tile_m2 == -__builtin_inff()) continue;

                // Stage 2: merge the tile maximum into the running online softmax.
                // Rescale U at most ONCE per KV tile, not once per KV position.
                const auto merge_max = [&](float tile_m, float & m, float & l,
                                           vfloat32m2_t & u0, vfloat32m2_t & u1) {
                    if (tile_m == -__builtin_inff()) return;
                    const float new_m = tile_m > m ? tile_m : m;
                    if (l != 0.0f && new_m > m) {
                        const float alpha = exp_neg(m - new_m);
                        l *= alpha;
                        u0 = __riscv_vfmul_vf_f32m2(u0, alpha, VL);
                        u1 = __riscv_vfmul_vf_f32m2(u1, alpha, VL);
                    }
                    m = new_m;
                };
                merge_max(tile_m0, m0, l0, o00, o01);
                merge_max(tile_m1, m1, l1, o10, o11);
                merge_max(tile_m2, m2, l2, o20, o21);

                // Stage 3: P_tile * V_tile, accumulating the same unnormalized FP32 U.
                // The V cache is read once per key and shared by the three heads.
                for (int t = 0; t < tile_len; ++t) {
                    if (score0[t] == -__builtin_inff() &&
                        score1[t] == -__builtin_inff() && score2[t] == -__builtin_inff()) continue;
                    const float p0 = score0[t] == -__builtin_inff() ? 0.0f : exp_neg(score0[t] - m0);
                    const float p1 = score1[t] == -__builtin_inff() ? 0.0f : exp_neg(score1[t] - m1);
                    const float p2 = score2[t] == -__builtin_inff() ? 0.0f : exp_neg(score2[t] - m2);
                    l0 += p0; l1 += p1; l2 += p2;

                    const int64_t kv = kv_base + t;
                    const uint8_t * vp = static_cast<const uint8_t *>(vc->data) +
                        (v_trans ? kv * vc->nb[0] + kv_head * vc->nb[1] :
                                   kv * vc->nb[2] + kv_head * vc->nb[1]);
                    const ptrdiff_t v_step = static_cast<ptrdiff_t>(v_trans ? vc->nb[2] : vc->nb[0]);
                    const vfloat32m2_t v0 = f16_load64(vp, v_step);
                    const vfloat32m2_t v1 = f16_load64(vp + 64*v_step, v_step);
                    o00 = __riscv_vfmacc_vf_f32m2(o00, p0, v0, VL);
                    o01 = __riscv_vfmacc_vf_f32m2(o01, p0, v1, VL);
                    o10 = __riscv_vfmacc_vf_f32m2(o10, p1, v0, VL);
                    o11 = __riscv_vfmacc_vf_f32m2(o11, p1, v1, VL);
                    o20 = __riscv_vfmacc_vf_f32m2(o20, p2, v0, VL);
                    o21 = __riscv_vfmacc_vf_f32m2(o21, p2, v1, VL);
                }
            }
            GGML_ASSERT(l0 > 0.0f && l1 > 0.0f && l2 > 0.0f);
            o00 = __riscv_vfmul_vf_f32m2(o00, 1.0f/l0, VL);
            o01 = __riscv_vfmul_vf_f32m2(o01, 1.0f/l0, VL);
            o10 = __riscv_vfmul_vf_f32m2(o10, 1.0f/l1, VL);
            o11 = __riscv_vfmul_vf_f32m2(o11, 1.0f/l1, VL);
            o20 = __riscv_vfmul_vf_f32m2(o20, 1.0f/l2, VL);
            o21 = __riscv_vfmul_vf_f32m2(o21, 1.0f/l2, VL);
            const uint64_t t_attention1_end = phase_tick(timing_mode);
            if (timing_mode) timing.attention += t_attention1_end - t_wo0_end;

            // All twelve Q8 blocks come directly from the six O RVV registers.
            float d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11;
            const vint8mf2_t z0 =quant32(o00,0,d0),  z1 =quant32(o00,1,d1);
            const vint8mf2_t z2 =quant32(o01,0,d2),  z3 =quant32(o01,1,d3);
            const vint8mf2_t z4 =quant32(o10,0,d4),  z5 =quant32(o10,1,d5);
            const vint8mf2_t z6 =quant32(o11,0,d6),  z7 =quant32(o11,1,d7);
            const vint8mf2_t z8 =quant32(o20,0,d8),  z9 =quant32(o20,1,d9);
            const vint8mf2_t z10=quant32(o21,0,d10), z11=quant32(o21,1,d11);

            // Select one of the twelve live Q8 register blocks per W_O input
            // block. No addressable O[384] scratch buffer or calls across live
            // Q8 values. Only each group's final partial-Y is stored.
            float * y = reinterpret_cast<float *>(static_cast<uint8_t *>(parts->data) +
                group * parts->nb[2] + token * parts->nb[1]);
            for (int yt = 0; yt < WO_SUBTILES; ++yt) {
                const uint8_t * subtile = wo_tile + static_cast<size_t>(yt) * WO_SUBTILE_BYTES;
                vfloat32m4_t acc_lo = __riscv_vfmv_v_f_f32m4(0.0f, WO_VL);
                vfloat32m4_t acc_hi = __riscv_vfmv_v_f_f32m4(0.0f, WO_VL);
                // Static Q8-register selection. No runtime switch or vector array.
                // Each call updates both output halves from one weight load per k.
                wo_block_both(subtile +  0 * WO_BLOCK_BYTES, z0,  d0,  acc_lo, acc_hi);
                wo_block_both(subtile +  1 * WO_BLOCK_BYTES, z1,  d1,  acc_lo, acc_hi);
                wo_block_both(subtile +  2 * WO_BLOCK_BYTES, z2,  d2,  acc_lo, acc_hi);
                wo_block_both(subtile +  3 * WO_BLOCK_BYTES, z3,  d3,  acc_lo, acc_hi);
                wo_block_both(subtile +  4 * WO_BLOCK_BYTES, z4,  d4,  acc_lo, acc_hi);
                wo_block_both(subtile +  5 * WO_BLOCK_BYTES, z5,  d5,  acc_lo, acc_hi);
                wo_block_both(subtile +  6 * WO_BLOCK_BYTES, z6,  d6,  acc_lo, acc_hi);
                wo_block_both(subtile +  7 * WO_BLOCK_BYTES, z7,  d7,  acc_lo, acc_hi);
                wo_block_both(subtile +  8 * WO_BLOCK_BYTES, z8,  d8,  acc_lo, acc_hi);
                wo_block_both(subtile +  9 * WO_BLOCK_BYTES, z9,  d9,  acc_lo, acc_hi);
                wo_block_both(subtile + 10 * WO_BLOCK_BYTES, z10, d10, acc_lo, acc_hi);
                wo_block_both(subtile + 11 * WO_BLOCK_BYTES, z11, d11, acc_lo, acc_hi);
                __riscv_vse32_v_f32m4(y + yt * WO_SUBTILE_OUTPUTS, acc_lo, WO_VL);
                __riscv_vse32_v_f32m4(y + yt * WO_SUBTILE_OUTPUTS + WO_VL, acc_hi, WO_VL);
            }
            if (timing_mode) {
                timing.q8_wo += phase_tick(timing_mode) - t_attention1_end;
                ++timing.tokens;
            }
        }

    }
    if (timing_mode) phase_record(group, timing);
}
#endif // standard RVV available

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
