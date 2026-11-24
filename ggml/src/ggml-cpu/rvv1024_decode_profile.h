#pragma once

// Decode-only diagnostics. LLAMA_RVV1024_DECODE_PROFILE=ns (node wall time)
// or =detail (also worker times and optional RV64 rdtime subphases).
// Separate from the legacy prefill/attention comparator. No extra barriers.
#include <stdint.h>
#include <stdbool.h>

enum rvvd_phase {
    RVVD_Q8, RVVD_MAT_SETUP_SYNC, RVVD_MAT_CHUNKS,
    RVVD_QK, RVVD_SCORE_STORE, RVVD_SOFTMAX, RVVD_PV,
    RVVD_MERGE_Q8, RVVD_WO, RVVD_PHASE_COUNT,
};

#ifdef __cplusplus
extern "C" {
#endif
bool ggml_rvvd_prof_active(void);
void ggml_rvvd_prof_record(enum rvvd_phase phase, uint64_t ticks);
#ifdef __cplusplus
}

// Never call clock_gettime (or any ordinary ABI function) with the six QK
// accumulators live. rdtime is an elapsed-time counter, NOT a cycle counter.
// The memory clobber constrains memory motion, not all arithmetic scheduling.
static inline __attribute__((always_inline)) uint64_t rvvd_read_ticks(bool enabled) {
#if defined(__riscv) && __riscv_xlen == 64
    uint64_t t = 0;
    if (enabled) {
        __asm__ __volatile__("rdtime %0" : "=r"(t) :: "memory");
    }
    return t;
#else
    (void) enabled;
    return 0;
#endif
}

#else // implementation in ggml-cpu.c, after ggml-impl.h and compare profiler

#include <inttypes.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// All executed graph nodes are accounted for, including unknown operators.
enum rvvd_bucket {
    RVVD_FFN_GATE, RVVD_FFN_UP, RVVD_FFN_DOWN, RVVD_FFN_OTHER,
    RVVD_WQ, RVVD_WK, RVVD_WV, RVVD_ATTN, RVVD_WO_STD,
    RVVD_Q_BIAS, RVVD_ROPE, RVVD_CACHE, RVVD_KCOL,
    RVVD_PREP, RVVD_D6_Q, RVVD_D6_ATTN, RVVD_D6_WO, RVVD_D6_REDUCE,
    RVVD_NORM, RVVD_LOGITS, RVVD_OTHER_MATMUL, RVVD_OTHER, RVVD_COUNT,
};
static const char * const rvvd_names[RVVD_COUNT] = {
    "ffn_gate", "ffn_up", "ffn_down", "ffn_other",
    "Wq", "Wk", "Wv", "standard_attention", "Wo",
    "Q_bias", "rope", "KV_write", "K_column_write",
    "fused_prepare", "decode6_Q", "decode6_attention", "decode6_merge_Wo", "decode6_reduce_Y",
    "normalization", "logits", "other_matmul", "other",
};
static const char * const rvvd_phase_names[RVVD_PHASE_COUNT] = {
    "repack_Q8", "repack_setup_and_initial_sync", "repack_matmul_chunks",
    "QK", "score_store", "softmax_including_score_reload", "PV",
    "split_merge_and_Q8", "packed_Wo",
};

struct rvvd_counter {
    atomic_uint_fast64_t wall_ns, main_ns, worker_ns, nodes, worker_calls;
    atomic_uint_fast64_t sub_ticks[RVVD_PHASE_COUNT];
    atomic_uint_fast64_t sub_calls[RVVD_PHASE_COUNT];
};
static struct rvvd_counter rvvd_totals[RVVD_COUNT];
static atomic_uint_fast64_t rvvd_graph_ns, rvvd_graph_count;
static atomic_flag rvvd_registered = ATOMIC_FLAG_INIT;

// Per-worker detail accumulation: no atomics or calls in a live RVV tile.
static _Thread_local int rvvd_tls_bucket = -1;
static _Thread_local uint64_t rvvd_tls_ticks[RVVD_COUNT][RVVD_PHASE_COUNT];
static _Thread_local uint64_t rvvd_tls_calls[RVVD_COUNT][RVVD_PHASE_COUNT];

bool ggml_rvvd_prof_active(void) {
#if defined(__riscv) && __riscv_xlen == 64
    return rvvd_tls_bucket >= 0;
#else
    return false;
#endif
}
void ggml_rvvd_prof_record(enum rvvd_phase phase, uint64_t ticks) {
    if (rvvd_tls_bucket < 0 || phase < 0 || phase >= RVVD_PHASE_COUNT) return;
    rvvd_tls_ticks[rvvd_tls_bucket][phase] += ticks;
    rvvd_tls_calls[rvvd_tls_bucket][phase]++;
}

static inline bool rvvd_prefix(const char * s, const char * p) {
    return strncmp(s, p, strlen(p)) == 0;
}
static inline int rvvd_ffn_weight(const struct ggml_tensor * n) {
    if (n->op != GGML_OP_MUL_MAT || !n->src[0]) return -1;
    const char * w = n->src[0]->name;
    if (strstr(w, ".ffn_gate.weight")) return RVVD_FFN_GATE;
    if (strstr(w, ".ffn_up.weight"))   return RVVD_FFN_UP;
    if (strstr(w, ".ffn_down.weight")) return RVVD_FFN_DOWN;
    return -1;
}
static inline int rvvd_classify(const struct ggml_tensor * n) {
    int ffn = rvvd_ffn_weight(n);
    if (ffn >= 0) return ffn;
    if (n->op == GGML_OP_MUL_MAT && n->src[0]) {
        const char * w = n->src[0]->name;
        if (strstr(w, ".attn_q.weight"))      return RVVD_WQ;
        if (strstr(w, ".attn_k.weight"))      return RVVD_WK;
        if (strstr(w, ".attn_v.weight"))      return RVVD_WV;
        if (strstr(w, ".attn_output.weight")) return RVVD_WO_STD;
        if (strcmp(w, "output.weight") == 0 || rvvd_prefix(n->name, "result_output")) return RVVD_LOGITS;
        return RVVD_OTHER_MATMUL;
    }
    if (n->op == GGML_OP_FLASH_ATTN_EXT) return RVVD_ATTN;
    if (n->op == GGML_OP_ROPE) return RVVD_ROPE;
    if (n->op == GGML_OP_RMS_NORM) return RVVD_NORM;
    if (n->op == GGML_OP_SET_ROWS) return RVVD_CACHE;
    if (n->op == GGML_OP_ADD && n->src[1] && strstr(n->src[1]->name, ".attn_q.bias")) return RVVD_Q_BIAS;
    if (rvvd_prefix(n->name, "rvv1024_store_k_columns")) return RVVD_KCOL;
    if (rvvd_prefix(n->name, "rvv1024_prepared_x_rope")) return RVVD_PREP;
    if (rvvd_prefix(n->name, "rvv1024_decode6_q")) return RVVD_D6_Q;
    if (rvvd_prefix(n->name, "rvv1024_decode6_split_kv_stats")) return RVVD_D6_ATTN;
    if (rvvd_prefix(n->name, "rvv1024_decode6_partial_y")) return RVVD_D6_WO;
    if (rvvd_prefix(n->name, "rvv1024_decode6_attn_out")) return RVVD_D6_REDUCE;
    if (rvvd_prefix(n->name, "ffn_")) return RVVD_FFN_OTHER;
    return RVVD_OTHER;
}

// Qwen2 CPU graph: require a named FFN matmul and exactly one activation row
// for EVERY named FFN matmul. The final FFN alone is not sufficient, because
// output pruning can leave a single row in the last layer during prefill.
static bool rvvd_is_decode_graph(const struct ggml_cgraph * g) {
    bool seen = false;
    for (int i = 0; i < g->n_nodes; ++i) {
        const struct ggml_tensor * n = g->nodes[i];
        if (rvvd_ffn_weight(n) < 0) continue;
        const struct ggml_tensor * a = n->src[1];
        if (!a || a->ne[1] != 1 || a->ne[2] != 1 || a->ne[3] != 1) return false;
        seen = true;
    }
    return seen;
}

static uint64_t rvvd_timebase_hz(void) {
    const char * e = getenv("LLAMA_RVV1024_TIMEBASE_HZ");
    if (e && *e) return (uint64_t) strtoull(e, NULL, 10);
    // Device-tree integers are big endian. Accept 32- or 64-bit cells.
    FILE * f = fopen("/proc/device-tree/cpus/timebase-frequency", "rb");
    if (!f) return 0;
    unsigned char b[8];
    const size_t n = fread(b, 1, sizeof(b), f);
    fclose(f);
    if (n != 4 && n != 8) return 0;
    uint64_t hz = 0;
    for (size_t i = 0; i < n; ++i) hz = (hz << 8) | b[i];
    return hz;
}
static void rvvd_report(void) {
    const uint64_t graphs = atomic_load_explicit(&rvvd_graph_count, memory_order_relaxed);
    if (!graphs) return;
    const uint64_t graph_ns = atomic_load_explicit(&rvvd_graph_ns, memory_order_relaxed);
    const uint64_t hz = rvvd_timebase_hz();
    uint64_t selected_ns = 0;
    fprintf(stderr, "[RVVD] decode_cpu_graphs=%" PRIu64 " graph_ms=%.3f graph_ms_per_graph=%.6f timebase_hz=%" PRIu64 "\n",
            graphs, graph_ns * 1e-6, graph_ns * 1e-6 / graphs, hz);
    for (int b = 0; b < RVVD_COUNT; ++b) {
        const struct rvvd_counter * c = &rvvd_totals[b];
        const uint64_t nodes = atomic_load_explicit(&c->nodes, memory_order_relaxed);
        if (!nodes) continue;
        const uint64_t wall = atomic_load_explicit(&c->wall_ns, memory_order_relaxed);
        const uint64_t main = atomic_load_explicit(&c->main_ns, memory_order_relaxed);
        const uint64_t workers = atomic_load_explicit(&c->worker_ns, memory_order_relaxed);
        const uint64_t calls = atomic_load_explicit(&c->worker_calls, memory_order_relaxed);
        selected_ns += wall;
        fprintf(stderr, "[RVVD] bucket=%s nodes=%" PRIu64 " wall_ms=%.3f ms_per_graph=%.6f pct_graph=%.2f main_compute_ms=%.3f main_wait_ms=%.3f worker_ms=%.3f worker_calls=%" PRIu64 "\n",
                rvvd_names[b], nodes, wall*1e-6, wall*1e-6/graphs,
                graph_ns ? 100.0*wall/graph_ns : 0.0, main*1e-6,
                (wall >= main ? wall-main : 0)*1e-6, workers*1e-6, calls);
        for (int p = 0; p < RVVD_PHASE_COUNT; ++p) {
            const uint64_t pc = atomic_load_explicit(&c->sub_calls[p], memory_order_relaxed);
            if (!pc) continue;
            const uint64_t pt = atomic_load_explicit(&c->sub_ticks[p], memory_order_relaxed);
            fprintf(stderr, "[RVVD_DETAIL] bucket=%s phase=%s worker_ticks=%" PRIu64 " worker_calls=%" PRIu64,
                    rvvd_names[b], rvvd_phase_names[p], pt, pc);
            if (hz) fprintf(stderr, " worker_ms=%.3f worker_ms_per_graph=%.6f", 1000.0*pt/hz, 1000.0*pt/hz/graphs);
            fputc('\n', stderr);
        }
    }
    fprintf(stderr, "[RVVD] accounted_node_wall_ms=%.3f unaccounted_graph_ms=%.3f\n",
            selected_ns*1e-6, (graph_ns >= selected_ns ? graph_ns-selected_ns : 0)*1e-6);
    fprintf(stderr, "[RVVD] wall=worker0 timeline through existing node barrier; main_wait includes other workers, callback and barrier, not pure sync overhead. worker_ms and DETAIL are SUMS, not wall; worker_calls=0 in ns mode. Fused native ops charged to first node. Graph filter excludes multi-token FFN graphs; use --no-warmup and depths divisible by ubatch.\n");
}

struct rvvd_run {
    int mode; // 0=off, 1=wall only, 2=wall + workers + fine RV64 ticks
    bool primary;
    int pending;
    uint64_t graph_start, pending_start;
    uint64_t wall[RVVD_COUNT], main[RVVD_COUNT], worker[RVVD_COUNT];
    uint64_t nodes[RVVD_COUNT], calls[RVVD_COUNT];
};
static inline void rvvd_begin(struct rvvd_run * r, const struct ggml_cgraph * g, int ith) {
    r->mode = 0;
    const char * e = getenv("LLAMA_RVV1024_DECODE_PROFILE");
    const int mode = e && strcmp(e, "ns") == 0 ? 1 : e && strcmp(e, "detail") == 0 ? 2 : 0;
    if (!mode || !rvvd_is_decode_graph(g)) return;
    memset(r, 0, sizeof(*r));
    r->mode = mode;
    r->primary = ith == 0;
    r->pending = -1;
    if (mode == 2) {
        rvvd_tls_bucket = -1;
        memset(rvvd_tls_ticks, 0, sizeof(rvvd_tls_ticks));
        memset(rvvd_tls_calls, 0, sizeof(rvvd_tls_calls));
    }
    if (r->primary) {
        if (!atomic_flag_test_and_set_explicit(&rvvd_registered, memory_order_relaxed)) atexit(rvvd_report);
        r->graph_start = rvv1024_cmp_tick();
    }
}
static inline uint64_t rvvd_start(struct rvvd_run * r, const struct ggml_tensor * n, int * bucket) {
    *bucket = -1;
    if (!r->mode || (!r->primary && r->mode != 2)) return 0;
    *bucket = rvvd_classify(n);
    if (r->mode == 2) rvvd_tls_bucket = *bucket;
    return rvv1024_cmp_tick();
}
static inline void rvvd_compute_done(struct rvvd_run * r, int b, uint64_t start) {
    if (b < 0) return;
    const uint64_t ns = rvv1024_cmp_tick() - start;
    if (r->mode == 2) {
        rvvd_tls_bucket = -1;
        r->worker[b] += ns;
        r->calls[b]++;
    }
    if (r->primary) {
        r->main[b] += ns;
        r->pending = b;
        r->pending_start = start;
    }
}
static inline void rvvd_after_barrier(struct rvvd_run * r) {
    if (!r->mode || !r->primary || r->pending < 0) return;
    const int b = r->pending;
    r->wall[b] += rvv1024_cmp_tick() - r->pending_start;
    r->nodes[b]++;
    r->pending = -1;
}
static inline void rvvd_end(struct rvvd_run * r) {
    if (!r->mode) return;
    rvvd_after_barrier(r); // last node uses the existing final graph barrier
    if (r->primary) {
        atomic_fetch_add_explicit(&rvvd_graph_ns, rvv1024_cmp_tick()-r->graph_start, memory_order_relaxed);
        atomic_fetch_add_explicit(&rvvd_graph_count, 1, memory_order_relaxed);
    }
    for (int b = 0; b < RVVD_COUNT; ++b) {
        struct rvvd_counter * c = &rvvd_totals[b];
        if (r->primary && r->nodes[b]) {
            atomic_fetch_add_explicit(&c->wall_ns, r->wall[b], memory_order_relaxed);
            atomic_fetch_add_explicit(&c->main_ns, r->main[b], memory_order_relaxed);
            atomic_fetch_add_explicit(&c->nodes, r->nodes[b], memory_order_relaxed);
        }
        if (r->mode == 2 && r->calls[b]) {
            atomic_fetch_add_explicit(&c->worker_ns, r->worker[b], memory_order_relaxed);
            atomic_fetch_add_explicit(&c->worker_calls, r->calls[b], memory_order_relaxed);
            for (int p = 0; p < RVVD_PHASE_COUNT; ++p) {
                if (!rvvd_tls_calls[b][p]) continue;
                atomic_fetch_add_explicit(&c->sub_ticks[p], rvvd_tls_ticks[b][p], memory_order_relaxed);
                atomic_fetch_add_explicit(&c->sub_calls[p], rvvd_tls_calls[b][p], memory_order_relaxed);
            }
        }
    }
}
#endif // __cplusplus
