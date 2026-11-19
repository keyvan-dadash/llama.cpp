#pragma once

// Experimental GCC RVV 1,024-bit register-fused Qwen2 attention.
// Three-head prefill consuming packed column-SoA Q and W_O weights.
// Ordinary GGUF Q4_0 matrices MUST NOT be passed as packed weights.
// The packed-weight loader is a separate step; A100 validation is required.
//
// Source files live under spacemit/ for hardware organization, but use only
// standard GCC RISC-V Vector facilities; IME/vendor compiler extensions are
// not required. The dedicated CMake option is GGML_CPU_RVV1024_FUSED.

#include <cstddef>
#include <cstdint>

struct ggml_compute_params;
struct ggml_tensor;

namespace spacemit_kernels::rvv::fused1024 {

inline constexpr std::size_t VLEN_BYTES = 128;
inline constexpr int64_t HIDDEN_DIM     = 1536;
inline constexpr int64_t HEAD_DIM       = 128;
inline constexpr int64_t N_Q_HEADS      = 12;
inline constexpr int64_t N_KV_HEADS     = 2;
inline constexpr int64_t Q_HEADS_TILE   = 3;
inline constexpr int64_t N_Q_GROUPS     = N_Q_HEADS / Q_HEADS_TILE;
static_assert(N_Q_HEADS % Q_HEADS_TILE == 0, "Query heads must divide into groups");
static_assert(N_Q_HEADS / N_KV_HEADS == 6, "Expected Qwen2 GQA 6:1 mapping");

// Version-1 external weight ABI, little-endian (matches RVV1024 Python layout).
// A 64-byte e8mf2 unit-stride load contains 64 pairs of Q4_0 values;
// low nibble = first half of one head, high nibble = second half.
// Q: [4 groups][48 input blocks][32 input columns][3 heads][64 packed bytes]
//    followed per input block by 384 FP32 scales in Q-row order.
// W_O: [4 groups][6 output subtiles of 256 rows][12 input blocks]
//      [32 input columns][128 packed bytes] followed by 256 FP32 scales.
// KV (for a later fused-KV graph): [2 heads][48 input blocks]
//      [32 input columns][128 packed bytes], K low nibble, V high nibble,
//      followed by 128 K and then 128 V FP32 scales.
inline constexpr int64_t Q4_BLOCK_SIZE          = 32;
inline constexpr int64_t Q_BLOCKS_PER_ROW       = HIDDEN_DIM / Q4_BLOCK_SIZE;
inline constexpr int64_t WO_SUBTILE_OUTPUTS     = 256;
inline constexpr int64_t WO_SUBTILES            = HIDDEN_DIM / WO_SUBTILE_OUTPUTS;
inline constexpr int64_t WO_BLOCKS_PER_Q_GROUP  = Q_HEADS_TILE * HEAD_DIM / Q4_BLOCK_SIZE;
inline constexpr std::size_t Q_COL_BYTES        = Q_HEADS_TILE * (HEAD_DIM / 2); // 192
inline constexpr std::size_t Q_BLOCK_BYTES      = Q4_BLOCK_SIZE * Q_COL_BYTES +
                                                  Q_HEADS_TILE * HEAD_DIM * sizeof(float); // 7680
inline constexpr std::size_t Q_GROUP_BYTES      = Q_BLOCKS_PER_ROW * Q_BLOCK_BYTES; // 368640
inline constexpr std::size_t Q_PACKED_BYTES     = N_Q_GROUPS * Q_GROUP_BYTES; // 1474560
inline constexpr std::size_t KV_COL_BYTES       = HEAD_DIM; // 128
inline constexpr std::size_t KV_BLOCK_BYTES     = Q4_BLOCK_SIZE * KV_COL_BYTES +
                                                  2 * HEAD_DIM * sizeof(float); // 5120
inline constexpr std::size_t KV_GROUP_BYTES     = Q_BLOCKS_PER_ROW * KV_BLOCK_BYTES; // 245760
inline constexpr std::size_t KV_PACKED_BYTES    = N_KV_HEADS * KV_GROUP_BYTES; // 491520
inline constexpr std::size_t WO_COL_BYTES       = WO_SUBTILE_OUTPUTS / 2; // 128
inline constexpr std::size_t WO_BLOCK_BYTES     = Q4_BLOCK_SIZE * WO_COL_BYTES +
                                                  WO_SUBTILE_OUTPUTS * sizeof(float); // 5120
inline constexpr std::size_t WO_SUBTILE_BYTES   = WO_BLOCKS_PER_Q_GROUP * WO_BLOCK_BYTES; // 61440
inline constexpr std::size_t WO_GROUP_BYTES     = WO_SUBTILES * WO_SUBTILE_BYTES; // 368640
inline constexpr std::size_t WO_PACKED_BYTES    = N_Q_GROUPS * WO_GROUP_BYTES; // 1474560
static_assert(Q_BLOCK_BYTES == 7680 && Q_GROUP_BYTES == 368640 &&
              KV_BLOCK_BYTES == 5120 && KV_GROUP_BYTES == 245760 &&
              WO_BLOCK_BYTES == 5120 && WO_SUBTILE_BYTES == 61440 &&
              Q_PACKED_BYTES == 1474560 && WO_PACKED_BYTES == 1474560,
              "RVV1024 packed-weight ABI mismatch");

// Per-token, read-only preparation shared by all four Q-head groups.
// Produced by a separate GGML custom node, before fused Q/attention/W_O.
// Keep the original Q8 rounding/scaling and NEOX RoPE coefficients intact.
struct prefill_input_block {
    int8_t qs[Q4_BLOCK_SIZE];
    float scale;
};
struct prefill_prepared_token {
    prefill_input_block x8[Q_BLOCKS_PER_ROW];
    float c[HEAD_DIM / 2];
    float s[HEAD_DIM / 2];
};
static_assert(sizeof(prefill_input_block) == 36 && sizeof(prefill_prepared_token) == 2240,
              "RVV1024 prefill preparation ABI mismatch");


// Workers own disjoint [hidden, tokens] slabs of the [hidden, tokens, group]
// partial tensor. Group g covers Q heads [3*g, 3*g+3); its KV head is g/2.
inline constexpr int64_t group_first_q_head(int group) { return group * Q_HEADS_TILE; }
inline constexpr int64_t group_kv_head(int group) {
    return group_first_q_head(group) / (N_Q_HEADS / N_KV_HEADS);
}

// CPU backend extension procedure names, resolved from the GGML CPU registry.
inline constexpr const char * READY_PROC = "ggml_cpu_rvv1024_prefill_ready";
inline constexpr const char * FORWARD_PROC = "ggml_cpu_rvv1024_prefill_forward";
using prefill_ready_fn = bool (*)();
using prefill_forward_fn = void (*)(ggml_tensor * parts, int ith, int nth, void * userdata);

// Owned by a GGML graph-lifetime tensor's op_params (not a stack pointer).
// For the first Qwen2 prefill path we support NEOX RoPE without YaRN scaling.
struct prefill_rope_config {
    float base;
    float freq_scale;
};
static_assert(sizeof(prefill_rope_config) <= 64, "RoPE config must fit GGML op_params");

// Pure non-RVV reduction: each worker writes different output elements.
// Every group contributes to every Y coordinate; group slabs must never race.
inline void reduce_partials_f32(const float * parts, float * y, int64_t nelements,
                                int64_t groups, int ith, int nth) {
    for (int64_t i = ith; i < nelements; i += nth) {
        float sum = 0.0f;
        for (int64_t g = 0; g < groups; ++g) {
            sum += parts[g * nelements + i];
        }
        y[i] = sum;
    }
}

// Checks physical RVV width only. True does NOT mean the kernel is ready.
bool hardware_vlen_supported() noexcept;

// Model-load-time repacking, not a per-token operation.
// TODO: fix precise Q4_0 column-SoA layout and ownership/size contract.
void repack_q4_0_columns(const ggml_tensor * weight, void * dst, std::size_t capacity_bytes);

// TODO: interleave W_K and W_V for each KV group in the same Q4_0 column-SoA
// format, reusing the quantized input across both projections.
void repack_kv_q4_0_interleaved(const ggml_tensor * wk, const ggml_tensor * wv,
                               void * dst, std::size_t capacity_bytes);

// Planned graph node 1: X -> K/V projection -> K RoPE -> FP16 KV-cache writes.
// TODO: define GGML tensor source/side-effect contract and implement kernel.
void forward_kv_project(const ggml_compute_params * params, ggml_tensor * dst);

// Runtime hardware/compiler capability only, NOT proof of packed weights.
// The graph builder must also require packed Q/W_O source tensors and reject
// ordinary Q4_0. The packed loader is not included in this patch.
bool prefill_kernel_ready() noexcept;

// Signature EXACTLY matches ggml_custom_op_t. No fake ggml_compute_params.
// Called on each GGML batch worker with its logical ith/nth. The implementation
// owns scheduling group g = ith, ith+nth, ..., and writes ONLY partial[g].
// userdata points to a graph-lifetime prefill_rope_config byte sequence.
// No TCM, DMA, IME, or additional worker threads are involved.
void forward_q_attn_wo(ggml_tensor * parts, int ith, int nth, void * userdata);

} // namespace spacemit_kernels::rvv::fused1024
