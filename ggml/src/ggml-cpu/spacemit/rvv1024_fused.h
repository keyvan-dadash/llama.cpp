#pragma once

// Experimental GCC RVV 1,024-bit register-fused Qwen2 attention.
// Experimental three-head prefill using original GGML Q4_0 weights.
// Hardware/model checks gate this path; A100 numerical testing is required.
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

// Runtime hardware/compiler capability only. The graph builder independently
// rejects unsupported model, RoPE, KV-cache and attention configurations.
// This is still experimental and requires numerical validation on A100.
bool prefill_kernel_ready() noexcept;

// Signature EXACTLY matches ggml_custom_op_t. No fake ggml_compute_params.
// Called on each GGML batch worker with its logical ith/nth. The implementation
// owns scheduling group g = ith, ith+nth, ..., and writes ONLY partial[g].
// userdata points to a graph-lifetime prefill_rope_config byte sequence.
// No TCM, DMA, IME, or additional worker threads are involved.
void forward_q_attn_wo(ggml_tensor * parts, int ith, int nth, void * userdata);

} // namespace spacemit_kernels::rvv::fused1024
