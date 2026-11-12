// Experimental Qwen2 RVV1024 graph *scaffold*.
//
// This file deliberately does not implement the fused arithmetic yet. The normal
// Qwen2 path is unchanged unless LLAMA_RVV1024_GRAPH=1. In that graph-only mode,
// executing the fused node aborts loudly instead of silently returning wrong Y.
//
// Graph contract:
//   1. K/V use existing GGML projection, RoPE and cache-copy operations FOR NOW.
//   2. A KV-ready dependency node orders both cache writes before attention.
//   3. One future CPU kernel will consume X/W_Q/W_O/position/cache/mask and
//      produce FP32 Y directly, with Q, softmax state and O kept in RVV regs.
//   4. The backend implementation will later own the TCMHelper and DMA lifecycle.
//
// No custom hardware operations, manual TCM copies or multithreading here.

#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-model.h"

#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace {

static bool rvv1024_graph_requested() {
    const char * env = std::getenv("LLAMA_RVV1024_GRAPH");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

// Scheduling-only node: depends on *both* cache writes, producing an I32 token.
// The downstream custom op has GGML_MAX_SRC-limited inputs, so we merge the two
// cache-write dependencies here. A real fused K/V writer can replace this later.
static void rvv1024_kv_ready_fn(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    (void) userdata;
    if (ith == 0) {
        *static_cast<int32_t *>(dst->data) = 0;
    }
}

static void rvv1024_q_attn_wo_fn(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) dst;
    (void) ith;
    (void) nth;
    (void) userdata;
    GGML_ABORT("LLAMA_RVV1024_GRAPH=1 built the experimental graph, but the "
               "register-fused Q/attention/W_O kernel is not implemented yet; "
               "unset LLAMA_RVV1024_GRAPH to use the normal Qwen2 path");
}

} // namespace

ggml_tensor * llm_graph_context::build_attn_rvv1024_qwen2(
        const llama_layer & layer,
              ggml_tensor * x_norm,
              ggml_tensor * inp_pos,
  llm_graph_input_attn_kv * inp,
                      int il) const {
    if (!rvv1024_graph_requested()) {
        return nullptr;
    }

    // Narrow contract for the first structural experiment: Qwen2.5-Coder-1.5B,
    // Q4_0 projections, FP16 K/V, one sequence, no adapters or extra transforms.
    // Unsupported configurations must use the ordinary Qwen2 graph.
    if (n_embd != 1536 || n_head != 12 || n_head_kv != 2 ||
        n_embd_head_k != 128 || n_embd_head_v != 128 || n_rot != 128 ||
        n_tokens < 1 || ubatch.n_seqs_unq != 1 || cparams.training ||
        hparams.f_clamp_kqv > 0.0f || hparams.attn_soft_cap ||
        !loras->empty() ||
        !x_norm || x_norm->type != GGML_TYPE_F32 ||
        !inp || !inp->mctx || !inp->self_attn_mask.raw || inp->self_attn_mask.rows ||
        inp->mctx->type_k() != GGML_TYPE_F16 || inp->mctx->type_v() != GGML_TYPE_F16 ||
        !layer.wq || !layer.wk || !layer.wv || !layer.wo ||
        layer.wq->type != GGML_TYPE_Q4_0 || layer.wk->type != GGML_TYPE_Q4_0 ||
        layer.wv->type != GGML_TYPE_Q4_0 || layer.wo->type != GGML_TYPE_Q4_0 ||
        layer.wqkv || layer.wo_b || layer.wq_s || layer.wk_s || layer.wv_s || layer.wo_s ||
        (il < (int) inp->self_k_rot.size() && inp->self_k_rot[il]) ||
        (il < (int) inp->self_v_rot.size() && inp->self_v_rot[il])) {
        return nullptr;
    }

    // K/V construction deliberately uses original GGML ops for now. Keeping
    // these cache writes in the graph preserves normal cache-index handling.
    ggml_tensor * k_cur = build_lora_mm(layer.wk, x_norm);
    if (layer.wk_b) {
        k_cur = ggml_add(ctx0, k_cur, layer.wk_b);
    }
    k_cur = ggml_reshape_3d(ctx0, k_cur, 128, n_head_kv, n_tokens);
    k_cur = ggml_rope_ext(ctx0, k_cur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(k_cur, "rvv1024_Kcur", il);

    ggml_tensor * v_cur = build_lora_mm(layer.wv, x_norm);
    if (layer.wv_b) {
        v_cur = ggml_add(ctx0, v_cur, layer.wv_b);
    }
    v_cur = ggml_reshape_3d(ctx0, v_cur, 128, n_head_kv, n_tokens);
    cb(v_cur, "rvv1024_Vcur", il);

    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    auto * k_store = inp->mctx->cpy_k(ctx0, k_cur, inp->get_k_idxs(), il);
    auto * v_store = inp->mctx->cpy_v(ctx0, v_cur, inp->get_v_idxs(), il);
    ggml_build_forward_expand(gf, k_store);
    ggml_build_forward_expand(gf, v_store);

    // Both cache-copy operations are explicit dependencies. This I32 node
    // contains no K/V data: it only orders the downstream fused computation.
    ggml_tensor * cache_deps[] = { k_store, v_store };
    ggml_tensor * kv_ready = ggml_custom_4d(
            ctx0, GGML_TYPE_I32, 1, 1, 1, 1,
            cache_deps, 2, rvv1024_kv_ready_fn, 1, nullptr);
    ggml_set_name(kv_ready, "rvv1024_kv_ready");

    auto * k_cache = inp->mctx->get_k(ctx0, il);
    auto * v_cache = inp->mctx->get_v(ctx0, il);

    // Inputs: X, Wq, Q bias, Wo, positions, K cache, V cache, causal
    // mask, and the KV-ready dependency. 9 < GGML_MAX_SRC (10).
    // Future real kernel must also carry RoPE parameters using a graph-owned
    // configuration (not a pointer to stack or global mutable state).
    ggml_tensor * args[] = {
        x_norm, layer.wq, layer.wq_b, layer.wo, inp_pos,
        k_cache, v_cache, inp->self_attn_mask.raw, kv_ready,
    };
    ggml_tensor * y = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, n_embd, n_tokens, 1, 1,
            args, 9, rvv1024_q_attn_wo_fn, 1, nullptr);
    ggml_set_name(y, "rvv1024_q_attn_wo_UNIMPLEMENTED");
    cb(y, "rvv1024_attn_out", il);
    ggml_build_forward_expand(gf, y);
    return y;
}
