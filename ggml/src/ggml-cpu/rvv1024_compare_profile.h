#ifndef GGML_CPU_RVV1024_COMPARE_PROFILE_H
#define GGML_CPU_RVV1024_COMPARE_PROFILE_H

// Diagnostic-only profiler for Qwen2 attention paths in the CPU graph.
// Included only when GGML_USE_CPU_RISCV64_SPACEMIT is set.
// Set LLAMA_RVV1024_COMPARE_PROFILE=ns to enable it. Reported nanoseconds
// are SUMS OF WORKER EXECUTION TIME, not elapsed wall time. No barriers are
// included. Existing LLAMA_RVV1024_PHASE_PROFILE=ns measures the internal
// Q+RoPE / attention / Q8+Wo breakdown of rvv1024_partial_y.

#include <inttypes.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum rvv1024_cmp_bucket {
    RVV1024_CMP_WQ,
    RVV1024_CMP_WQ_BIAS,
    RVV1024_CMP_Q_ROPE,
    RVV1024_CMP_WK,
    RVV1024_CMP_WV,
    RVV1024_CMP_K_ROPE,
    RVV1024_CMP_FLASH_ATTN,
    RVV1024_CMP_WO,
    RVV1024_CMP_PREP,
    RVV1024_CMP_FUSED,
    RVV1024_CMP_REDUCE,
    RVV1024_CMP_COUNT,
};

static const char * const rvv1024_cmp_names[RVV1024_CMP_COUNT] = {
    "repack_Wq_matmul", "repack_Wq_bias", "repack_Q_rope",
    "common_Wk_matmul", "common_Wv_matmul", "common_K_rope",
    "repack_flash_attention", "repack_Wo_matmul",
    "fused_prepare_X_rope", "fused_Q_attention_Wo", "fused_reduce_partial_Y",
};

struct rvv1024_cmp_counter {
    atomic_uint_fast64_t ns;
    atomic_uint_fast64_t invocations; // per-worker node calls (NOT number of graph nodes)
};

static struct rvv1024_cmp_counter rvv1024_cmp_counters[RVV1024_CMP_COUNT];
static atomic_flag rvv1024_cmp_atexit_registered = ATOMIC_FLAG_INIT;

static uint64_t rvv1024_cmp_load_ns(enum rvv1024_cmp_bucket bucket) {
    return (uint64_t) atomic_load_explicit(&rvv1024_cmp_counters[bucket].ns, memory_order_relaxed);
}

static void rvv1024_cmp_report(void) {
    for (int i = 0; i < RVV1024_CMP_COUNT; ++i) {
        const uint64_t ns = rvv1024_cmp_load_ns((enum rvv1024_cmp_bucket) i);
        const uint64_t calls = (uint64_t) atomic_load_explicit(
            &rvv1024_cmp_counters[i].invocations, memory_order_relaxed);
        if (!calls) continue;
        fprintf(stderr, "[RVV1024_COMPARE] bucket=%s ns=%" PRIu64
                        " worker_ms=%.3f worker_calls=%" PRIu64 "\n",
                rvv1024_cmp_names[i], ns, ns * 1e-6, calls);
    }

    const uint64_t q = rvv1024_cmp_load_ns(RVV1024_CMP_WQ) +
                       rvv1024_cmp_load_ns(RVV1024_CMP_WQ_BIAS) +
                       rvv1024_cmp_load_ns(RVV1024_CMP_Q_ROPE);
    const uint64_t a = rvv1024_cmp_load_ns(RVV1024_CMP_FLASH_ATTN);
    const uint64_t w = rvv1024_cmp_load_ns(RVV1024_CMP_WO);
    const uint64_t p = rvv1024_cmp_load_ns(RVV1024_CMP_PREP);
    const uint64_t f = rvv1024_cmp_load_ns(RVV1024_CMP_FUSED);
    const uint64_t r = rvv1024_cmp_load_ns(RVV1024_CMP_REDUCE);
    fprintf(stderr, "[RVV1024_COMPARE] SUMMARY repack_Q_plus_RoPE_ns=%" PRIu64
                    " repack_attention_ns=%" PRIu64 " repack_Wo_ns=%" PRIu64
                    " fused_prepare_ns=%" PRIu64 " fused_kernel_ns=%" PRIu64
                    " fused_reduce_Y_ns=%" PRIu64 "\n", q, a, w, p, f, r);
    const double repack_selected = (double) q + (double) a + (double) w;
    const double fused_selected = (double) p + (double) f + (double) r;
    if (repack_selected > 0.0) {
        fprintf(stderr, "[RVV1024_COMPARE] SELECTED_REPACK Q_plus_RoPE=%.1f%%"
                        " attention=%.1f%% W_O=%.1f%%\n",
                100.0*q/repack_selected, 100.0*a/repack_selected, 100.0*w/repack_selected);
    }
    if (fused_selected > 0.0) {
        fprintf(stderr, "[RVV1024_COMPARE] SELECTED_FUSED prepare=%.1f%%"
                        " fused_kernel=%.1f%% reduce_partial_Y=%.1f%%\n",
                100.0*p/fused_selected, 100.0*f/fused_selected, 100.0*r/fused_selected);
    }
    fprintf(stderr, "[RVV1024_COMPARE] Sum of worker execution times over ALL"
                    " decode calls (including depth-fill), all layers and runs."
                    " Excludes graph barriers, task wait and unclassified ops."
                    " Percentages are of SELECTED operators only."
                    " Do not add fused_kernel to the internal PHASE totals.\n");
}

static inline bool rvv1024_cmp_enabled(void) {
    const char * env = getenv("LLAMA_RVV1024_COMPARE_PROFILE");
    return env && strcmp(env, "ns") == 0;
}

static inline void rvv1024_cmp_register_report(void) {
    if (!atomic_flag_test_and_set_explicit(&rvv1024_cmp_atexit_registered,
                                           memory_order_relaxed)) {
        atexit(rvv1024_cmp_report);
    }
}

static inline uint64_t rvv1024_cmp_tick(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ULL + (uint64_t) ts.tv_nsec;
}

static inline void rvv1024_cmp_accumulate(int bucket, uint64_t ns) {
    atomic_fetch_add_explicit(&rvv1024_cmp_counters[bucket].ns, ns, memory_order_relaxed);
    atomic_fetch_add_explicit(&rvv1024_cmp_counters[bucket].invocations, 1, memory_order_relaxed);
}

static inline bool rvv1024_cmp_weight_name(const char * name, const char * match) {
    return name && strstr(name, match) != NULL;
}

// Follow reshape / add / view nodes back to the Q/K projection. In Qwen2
// the RoPE input is a reshape of the Q or K projection (optionally with bias).
static inline int rvv1024_cmp_rope_source(const struct ggml_tensor * node) {
    const struct ggml_tensor * cur = node->src[0];
    for (int i = 0; cur && i < 12; ++i) {
        if (cur->op == GGML_OP_MUL_MAT && cur->src[0]) {
            if (rvv1024_cmp_weight_name(cur->src[0]->name, ".attn_q.weight")) return RVV1024_CMP_Q_ROPE;
            if (rvv1024_cmp_weight_name(cur->src[0]->name, ".attn_k.weight")) return RVV1024_CMP_K_ROPE;
            return -1;
        }
        cur = cur->src[0];
    }
    return -1;
}

static inline int rvv1024_cmp_classify(const struct ggml_tensor * node) {
    if (node->op == GGML_OP_MUL_MAT && node->src[0]) {
        const char * name = node->src[0]->name;
        if (rvv1024_cmp_weight_name(name, ".attn_q.weight"))      return RVV1024_CMP_WQ;
        if (rvv1024_cmp_weight_name(name, ".attn_k.weight"))      return RVV1024_CMP_WK;
        if (rvv1024_cmp_weight_name(name, ".attn_v.weight"))      return RVV1024_CMP_WV;
        if (rvv1024_cmp_weight_name(name, ".attn_output.weight")) return RVV1024_CMP_WO;
    }
    if (node->op == GGML_OP_ADD && node->src[1] &&
        rvv1024_cmp_weight_name(node->src[1]->name, ".attn_q.bias")) {
        return RVV1024_CMP_WQ_BIAS;
    }
    if (node->op == GGML_OP_ROPE) return rvv1024_cmp_rope_source(node);
    if (node->op == GGML_OP_FLASH_ATTN_EXT) return RVV1024_CMP_FLASH_ATTN;
    if (node->op == GGML_OP_CUSTOM) {
        if (strcmp(node->name, "rvv1024_prepared_x_rope") == 0) return RVV1024_CMP_PREP;
        if (strcmp(node->name, "rvv1024_partial_y") == 0)       return RVV1024_CMP_FUSED;
        if (strncmp(node->name,
            "rvv1024_attn_out-",
            sizeof("rvv1024_attn_out-") - 1) == 0)
            return RVV1024_CMP_REDUCE;
    }
    return -1;
}

#endif // GGML_CPU_RVV1024_COMPARE_PROFILE_H
