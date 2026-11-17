// Experimental Qwen2 RVV1024 graph *scaffold*.
//
// This file routes explicitly enabled prefill to an optional RVV CPU backend.
// Unsupported builds and model configurations continue on normal Qwen2.
//
// Graph contract:
//   1. K/V use existing GGML projection, RoPE and cache-copy operations FOR NOW.
//   2. A KV-ready dependency node orders both cache writes before attention.
//   3. CPU workers compute independent three-head FP32 partial-Y slabs.
//   4. A separate GGML node reduces those slabs into the final FP32 Y.
//
// No TCM, DMA, IME, custom hardware operations or new threads here.

#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "rvv1024_fused.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>

namespace {

static bool rvv1024_graph_requested() {
    const char * env = std::getenv("LLAMA_RVV1024_GRAPH");
    return env != nullptr && std::strcmp(env, "1") == 0;
}

// Only enable the graph once the CPU backend explicitly reports it can run.
// Registry lookup avoids hard-linking llama to an optional ggml-cpu symbol.
static ggml_backend_reg_t rvv1024_cpu_reg() {
    return ggml_backend_reg_by_name("CPU");
}

static bool rvv1024_backend_ready() {
    const auto reg = rvv1024_cpu_reg();
    if (reg == nullptr) {
        return false;
    }
    const auto ready = reinterpret_cast<spacemit_kernels::rvv::fused1024::prefill_ready_fn>(
            ggml_backend_reg_get_proc_address(reg, spacemit_kernels::rvv::fused1024::READY_PROC));
    const auto forward = ggml_backend_reg_get_proc_address(
            reg, spacemit_kernels::rvv::fused1024::FORWARD_PROC);
    return ready != nullptr && forward != nullptr && ready();
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

static void rvv1024_q_attn_wo_fn(ggml_tensor * parts, int ith, int nth, void * userdata) {
    const auto reg = rvv1024_cpu_reg();
    const auto forward = reg ? reinterpret_cast<spacemit_kernels::rvv::fused1024::prefill_forward_fn>(
            ggml_backend_reg_get_proc_address(reg, spacemit_kernels::rvv::fused1024::FORWARD_PROC)) : nullptr;
    if (!forward) {
        GGML_ABORT("RVV1024 CPU prefill backend disappeared after graph construction");
    }
    // EXACT custom callback ABI; the backend maps ith/nth onto three-head groups.
    forward(parts, ith, nth, userdata);
}

static void rvv1024_reduce_partials_fn(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) userdata;
    const ggml_tensor * parts = dst->src[0];
    GGML_ASSERT(parts && dst->type == GGML_TYPE_F32 && parts->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(parts));
    GGML_ASSERT(parts->ne[0] == dst->ne[0] && parts->ne[1] == dst->ne[1] &&
                parts->ne[2] == spacemit_kernels::rvv::fused1024::N_Q_GROUPS);
    GGML_ASSERT(nth > 0 && ith >= 0 && ith < nth);
    spacemit_kernels::rvv::fused1024::reduce_partials_f32(
            static_cast<const float *>(parts->data), static_cast<float *>(dst->data),
            dst->ne[0] * dst->ne[1], parts->ne[2], ith, nth);
}

} // namespace

ggml_tensor * llm_graph_context::build_attn_rvv1024_qwen2(
        const llama_layer & layer,
              ggml_tensor * x_norm,
              ggml_tensor * inp_pos,
  llm_graph_input_attn_kv * inp,
                      int il) const {
    // Experimental path. The CPU backend checks the RVV hardware; the model
    // and RoPE restrictions below keep unsupported configurations on GGML.
    if (!rvv1024_graph_requested() || !rvv1024_backend_ready()) {
        return nullptr;
    }

    // The packed-weight loader is not wired up yet. In particular, the
    // model's GGUF Q4_0 W_Q/W_O are NOT packed I8 column-SoA tensors.
    // Until model-lifetime packed tensors are supplied, reject them and use
    // the unmodified Qwen2 graph. Never pass row-major Q4_0 to the RVV kernel.
    // K/V projection intentionally stays on GGML during this phase.
    if (n_embd != 1536 || n_head != 12 || n_head_kv != 2 ||
        n_embd_head_k != 128 || n_embd_head_v != 128 || n_rot != 128 ||
        n_tokens <= 1 || ubatch.n_seqs_unq != 1 || cparams.training ||
        !cparams.causal_attn || rope_type != GGML_ROPE_TYPE_NEOX ||
        ext_factor != 0.0f || attn_factor != 1.0f ||
        !std::isfinite(freq_base) || freq_base <= 0.0f ||
        !std::isfinite(freq_scale) || freq_scale <= 0.0f ||
        hparams.f_max_alibi_bias != 0.0f ||
        hparams.f_clamp_kqv > 0.0f || hparams.attn_soft_cap ||
        !loras->empty() ||
        !x_norm || x_norm->type != GGML_TYPE_F32 ||
        !inp || !inp->mctx || !inp->self_attn_mask.raw || inp->self_attn_mask.rows ||
        inp->mctx->type_k() != GGML_TYPE_F16 || inp->mctx->type_v() != GGML_TYPE_F16 ||
        !layer.wq || !layer.wk || !layer.wv || !layer.wo ||
        layer.wq->type != GGML_TYPE_I8 || layer.wq->ne[0] != static_cast<int64_t>(spacemit_kernels::rvv::fused1024::Q_PACKED_BYTES) ||
        layer.wo->type != GGML_TYPE_I8 || layer.wo->ne[0] != static_cast<int64_t>(spacemit_kernels::rvv::fused1024::WO_PACKED_BYTES) ||
        layer.wk->type != GGML_TYPE_Q4_0 || layer.wv->type != GGML_TYPE_Q4_0 ||
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

    // Inputs: X, PACKED Wq, Q bias, PACKED Wo, positions, K cache, V cache,
    // causal mask, and the KV-ready dependency. 9 < GGML_MAX_SRC (10).
    // Future loader must supply graph-owned packed tensors at src[1]/src[3];
    // the original model weight tensors cannot be reused as these arguments.
    // GGML graph metadata owns the two RoPE scalars until graph execution
    // finishes. Do not pass an address of local variables as custom userdata.
    using spacemit_kernels::rvv::fused1024::prefill_rope_config;
    const prefill_rope_config rope_cfg = { freq_base, freq_scale };
    ggml_tensor * rope_params = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 2);
    static_assert(sizeof(rope_cfg) <= sizeof(rope_params->op_params), "RoPE config too large");
    std::memcpy(rope_params->op_params, &rope_cfg, sizeof(rope_cfg));
    ggml_tensor * args[] = {
        x_norm, layer.wq, layer.wq_b, layer.wo, inp_pos,
        k_cache, v_cache, inp->self_attn_mask.raw, kv_ready,
    };
    // [hidden, tokens, group, 1]. Each three-head group owns one nonoverlapping
    // slab. No atomics or concurrent additions into the final output.
    ggml_tensor * parts = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, n_embd, n_tokens,
            spacemit_kernels::rvv::fused1024::N_Q_GROUPS, 1,
            args, 9, rvv1024_q_attn_wo_fn, GGML_N_TASKS_MAX, rope_params->op_params);
    ggml_set_name(parts, "rvv1024_partial_y");

    // A separate graph node runs after all per-group workers have finished.
    ggml_tensor * reduce_args[] = { parts };
    ggml_tensor * y = ggml_custom_4d(
            ctx0, GGML_TYPE_F32, n_embd, n_tokens, 1, 1,
            reduce_args, 1, rvv1024_reduce_partials_fn, GGML_N_TASKS_MAX, nullptr);
    ggml_set_name(y, "rvv1024_attn_out");
    cb(y, "rvv1024_attn_out", il);
    ggml_build_forward_expand(gf, y);
    return y;
}
