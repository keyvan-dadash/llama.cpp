#pragma once
// Included only by the C implementation of rvv1024_decode_profile.h.
// Linux perf_event_open, per-worker group snapshots. No changes to kernels.
// Scope: node (includes internal sync/quantization), or chunks (after sync).
#ifdef __linux__
#include <errno.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
extern long syscall(long number, ...);
#endif

#if defined(__linux__) && defined(SYS_perf_event_open)
#define RVVFP_WORKERS 64
struct rvvfp_counts {
    uint64_t attempted, valid, first, second, enabled, running;
};
struct rvvfp_atomic_counts {
    atomic_uint_fast64_t attempted, valid, first, second, enabled, running;
};
struct rvvfp_reading {
    uint64_t nr, enabled, running, value[2];
};
static struct rvvfp_atomic_counts rvvfp_totals[3][RVVFP_WORKERS];
static atomic_bool rvvfp_failed = ATOMIC_VAR_INIT(false);
static atomic_flag rvvfp_bad_config = ATOMIC_FLAG_INIT;
static _Thread_local int rvvfp_scope, rvvfp_ith, rvvfp_pending;
static _Thread_local int rvvfp_fd[2] = {-1, -1};
static _Thread_local bool rvvfp_start_ok;
static _Thread_local struct rvvfp_reading rvvfp_initial;
static _Thread_local struct rvvfp_counts rvvfp_local[3];

struct rvvfp_events {
    const char * name, * first, * second;
    uint64_t config[2];
};
static const struct rvvfp_events rvvfp_event_sets[] = {
    {"ipc", "cycles", "instructions", {PERF_COUNT_HW_CPU_CYCLES, PERF_COUNT_HW_INSTRUCTIONS}},
    {"cache", "cache_references", "cache_misses", {PERF_COUNT_HW_CACHE_REFERENCES, PERF_COUNT_HW_CACHE_MISSES}},
    {"branch", "branches", "branch_misses", {PERF_COUNT_HW_BRANCH_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_MISSES}},
    {"stall", "frontend_stall_cycles", "backend_stall_cycles", {PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, PERF_COUNT_HW_STALLED_CYCLES_BACKEND}},
};
static int rvvfp_get_scope(void) {
    const char * s = getenv("LLAMA_RVV1024_FFN_PERF");
    if (!s || !*s || strcmp(s, "0") == 0) return 0;
    if (strcmp(s, "node") == 0) return 1;
    if (strcmp(s, "chunks") == 0) return 2;
    return -1;
}
static const struct rvvfp_events * rvvfp_get_events(void) {
    const char * s = getenv("LLAMA_RVV1024_FFN_PERF_EVENTS");
    if (!s || !*s) return &rvvfp_event_sets[0];
    for (size_t i = 0; i < sizeof(rvvfp_event_sets)/sizeof(rvvfp_event_sets[0]); ++i) {
        if (strcmp(s, rvvfp_event_sets[i].name) == 0) return &rvvfp_event_sets[i];
    }
    return NULL;
}
static void rvvfp_close(void) {
    if (rvvfp_fd[0] >= 0) (void) ioctl(rvvfp_fd[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    for (int i = 1; i >= 0; --i) {
        if (rvvfp_fd[i] >= 0) close(rvvfp_fd[i]);
        rvvfp_fd[i] = -1;
    }
}
static void rvvfp_fail(const char * where, int err) {
    if (!atomic_exchange_explicit(&rvvfp_failed, true, memory_order_relaxed)) {
        fprintf(stderr, "[FFN_PERF] %s: %s; event pair unavailable or permission denied; check perf stat and perf_event_paranoid. Partial counts are not complete.\n", where, strerror(err));
    }
}
static void rvvfp_graph_begin(int ith) {
    rvvfp_scope = rvvfp_get_scope();
    rvvfp_pending = -1;
    rvvfp_start_ok = false;
    if (rvvfp_scope == 0) return;
    const struct rvvfp_events * e = rvvfp_get_events();
    if (rvvfp_scope < 0 || !e || ith < 0 || ith >= RVVFP_WORKERS) {
        if (!atomic_flag_test_and_set_explicit(&rvvfp_bad_config, memory_order_relaxed))
            fprintf(stderr, "[FFN_PERF] invalid scope/event set or worker index >=64; use node|chunks and ipc|cache|branch|stall\n");
        rvvfp_scope = 0;
        return;
    }
    rvvfp_ith = ith;
    memset(rvvfp_local, 0, sizeof(rvvfp_local));
    if (atomic_load_explicit(&rvvfp_failed, memory_order_relaxed)) return;
    for (int i = 0; i < 2; ++i) {
        struct perf_event_attr a;
        memset(&a, 0, sizeof(a));
        a.size = sizeof(a);
        a.type = PERF_TYPE_HARDWARE;
        a.config = e->config[i];
        a.disabled = i == 0;
        a.exclude_kernel = 1;
        a.exclude_hv = 1;
        a.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        // pid=0, cpu=-1: count this calling GGML worker, not all activity on a core.
        rvvfp_fd[i] = (int) syscall(SYS_perf_event_open, &a, 0, -1, i ? rvvfp_fd[0] : -1, PERF_FLAG_FD_CLOEXEC);
        if (rvvfp_fd[i] < 0) {
            rvvfp_fail(i ? e->second : e->first, errno);
            rvvfp_close();
            return;
        }
    }
    if (ioctl(rvvfp_fd[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) < 0) {
        rvvfp_fail("enable group", errno);
        rvvfp_close();
    }
}
static bool rvvfp_read(struct rvvfp_reading * s) {
    if (rvvfp_fd[0] < 0) return false;
    ssize_t n;
    do { n = read(rvvfp_fd[0], s, sizeof(*s)); } while (n < 0 && errno == EINTR);
    return n == (ssize_t) sizeof(*s) && s->nr == 2;
}
static void rvvfp_start(int scope, int b) {
    if (rvvfp_scope != scope || b < RVVD_FFN_GATE || b > RVVD_FFN_DOWN) return;
    rvvfp_pending = b;
    rvvfp_local[b].attempted++;
    rvvfp_start_ok = rvvfp_read(&rvvfp_initial);
}
static void rvvfp_stop(int scope, int b) {
    if (rvvfp_scope != scope || b < RVVD_FFN_GATE || b > RVVD_FFN_DOWN || rvvfp_pending != b) return;
    rvvfp_pending = -1;
    struct rvvfp_reading end;
    if (!rvvfp_start_ok || !rvvfp_read(&end)) return;
    const struct rvvfp_reading * s = &rvvfp_initial;
    if (end.enabled < s->enabled || end.running < s->running ||
        end.value[0] < s->value[0] || end.value[1] < s->value[1]) return;
    const uint64_t en = end.enabled - s->enabled, run = end.running - s->running;
    if (en == 0 || run == 0 || run > en) return;
    struct rvvfp_counts * c = &rvvfp_local[b];
    c->valid++;
    c->first += end.value[0] - s->value[0];
    c->second += end.value[1] - s->value[1];
    c->enabled += en;
    c->running += run;
}
static void rvvfp_graph_end(void) {
    if (!rvvfp_scope) return;
    rvvfp_close();
    for (int b = 0; b < 3; ++b) {
        const struct rvvfp_counts * l = &rvvfp_local[b];
        struct rvvfp_atomic_counts * t = &rvvfp_totals[b][rvvfp_ith];
#define RVVFP_ADD(f) atomic_fetch_add_explicit(&t->f, l->f, memory_order_relaxed)
        RVVFP_ADD(attempted); RVVFP_ADD(valid); RVVFP_ADD(first); RVVFP_ADD(second); RVVFP_ADD(enabled); RVVFP_ADD(running);
#undef RVVFP_ADD
    }
    rvvfp_scope = 0;
}
static void rvvfp_print(const struct rvvfp_events * e, int b, const char * label, const struct rvvfp_counts * c, uint64_t graphs) {
    if (!c->attempted) return;
    fprintf(stderr, "[FFN_PERF] bucket=%s %s samples=%" PRIu64 "/%" PRIu64,
        rvvd_names[b], label, c->valid, c->attempted);
    if (!c->valid) { fprintf(stderr, " unavailable\n"); return; }
    fprintf(stderr, " %s=%" PRIu64 " %s=%" PRIu64 " running_pct=%.3f",
        e->first, c->first, e->second, c->second, 100.0*c->running/c->enabled);
    if (c->first && strcmp(e->name, "ipc") == 0) fprintf(stderr, " IPC=%.4f", (double)c->second/c->first);
    if (c->first && (strcmp(e->name, "cache") == 0 || strcmp(e->name, "branch") == 0))
        fprintf(stderr, " miss_pct=%.4f", 100.0*c->second/c->first);
    if (graphs) fprintf(stderr, " %s_per_graph=%.3f %s_per_graph=%.3f", e->first, (double)c->first/graphs, e->second, (double)c->second/graphs);
    if (c->valid != c->attempted || c->running < c->enabled) fprintf(stderr, " raw_partial_counts");
    fputc('\n', stderr);
}
static void rvvfp_report(uint64_t graphs) {
    const int scope = rvvfp_get_scope();
    const struct rvvfp_events * e = rvvfp_get_events();
    if (scope <= 0 || !e) return;
    fprintf(stderr, "[FFN_PERF] scope=%s events=%s graphs=%" PRIu64 " user_only=1 per_thread=1; unscaled hardware counts, no outer node barrier; use separate uninstrumented timing\n", scope == 1 ? "node" : "chunks", e->name, graphs);
    for (int b = 0; b < 3; ++b) {
        struct rvvfp_counts sum = {0};
        for (int i = 0; i < RVVFP_WORKERS; ++i) {
            const struct rvvfp_atomic_counts * t = &rvvfp_totals[b][i];
            struct rvvfp_counts c;
#define RVVFP_LOAD(f) c.f = atomic_load_explicit(&t->f, memory_order_relaxed); sum.f += c.f
            RVVFP_LOAD(attempted); RVVFP_LOAD(valid); RVVFP_LOAD(first); RVVFP_LOAD(second); RVVFP_LOAD(enabled); RVVFP_LOAD(running);
#undef RVVFP_LOAD
            char label[32];
            snprintf(label, sizeof(label), "worker=%d", i);
            rvvfp_print(e, b, label, &c, graphs);
        }
        rvvfp_print(e, b, "worker=ALL", &sum, graphs);
    }
}
#else
static inline void rvvfp_graph_begin(int ith) { (void)ith; }
static inline void rvvfp_graph_end(void) {}
static inline void rvvfp_start(int scope, int b) { (void)scope; (void)b; }
static inline void rvvfp_stop(int scope, int b) { (void)scope; (void)b; }
static inline void rvvfp_report(uint64_t graphs) { (void)graphs; }
#endif
