#pragma once

// Experimental GCC RVV 1,024-bit register-fused Qwen2 attention.
// Template only: NO fused arithmetic, K/V interleaving or cache writes exist.
// These entry points are intentionally not wired to the llama graph yet.
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
inline constexpr int64_t Q_HEADS_TILE   = 2;

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

// Planned graph node 2: X -> Q projection -> Q RoPE -> online GQA attention
// -> Q8_0 quantization -> W_O projection -> FP32 Y, without materializing Q/O.
// TODO: match the current src/llama-rvv1024-scaffold.cpp node sources,
//       attach persistent packed weights, and implement RVV register fusion.
void forward_q_attn_wo(const ggml_compute_params * params, ggml_tensor * dst);

} // namespace spacemit_kernels::rvv::fused1024
