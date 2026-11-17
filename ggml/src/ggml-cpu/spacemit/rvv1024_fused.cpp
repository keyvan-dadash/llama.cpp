#include "rvv1024_fused.h"

#include "ggml.h"

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

void forward_q_attn_wo(const ggml_compute_params * params, ggml_tensor * dst) {
    (void) params;
    (void) dst;
    GGML_ABORT("RVV1024: register-fused Q/attention/W_O not implemented");
}

} // namespace spacemit_kernels::rvv::fused1024
