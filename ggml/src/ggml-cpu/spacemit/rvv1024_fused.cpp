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

bool prefill_kernel_ready() noexcept {
    // IMPORTANT: do not turn this on until the numerical kernel is implemented,
    // its packed-weight resources are alive, and full/hybrid Y is correct.
    return false;
}

void forward_q_attn_wo(ggml_tensor * parts, int ith, int nth, void * userdata) {
    (void) userdata;
    GGML_ASSERT(parts != nullptr && parts->data != nullptr &&
                parts->ne[0] == HIDDEN_DIM && parts->ne[2] == N_Q_GROUPS &&
                parts->ne[3] == 1);
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);
    // Check on the actual executing worker (not on the graph-building core).
    if (ith < N_Q_GROUPS && !hardware_vlen_supported()) {
        GGML_ABORT("RVV1024: worker %d is not on a 1024-bit RVV core", ith);
    }

    // One three-head group owns one independent FP32 output slab. On a
    // four-worker A100 pool, worker 0 owns Q[0..2], worker 1 Q[3..5], etc.
    // If fewer workers are available, each worker handles multiple groups.
    for (int group = ith; group < N_Q_GROUPS; group += nth) {
        const int64_t first_q = group_first_q_head(group);
        const int64_t kv_head = group_kv_head(group);
        float * y_partial = static_cast<float *>(parts->data) +
                            group * HIDDEN_DIM * parts->ne[1];
        (void) first_q;
        (void) kv_head;
        (void) y_partial;
        GGML_ABORT("RVV1024: three-head prefill math for group %d is not implemented", group);
    }
}

} // namespace spacemit_kernels::rvv::fused1024
