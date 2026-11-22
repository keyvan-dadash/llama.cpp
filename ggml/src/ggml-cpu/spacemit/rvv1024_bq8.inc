// Experimental B_Q=8 x B_KV=64 hierarchical prefill.
// Included inside fused_three_head_prefill_group() after its RVV microkernels.
// Opt in with LLAMA_RVV1024_QUERY_TILE=8; default is four-token Q plus two-token attention/W_O.
// No TCM, DMA, KV projection, or GGUF ABI changes.
{
    static const bool enable_bq8 = [] {
        const char * e = std::getenv("LLAMA_RVV1024_QUERY_TILE");
        const bool enabled = e && std::strcmp(e, "8") == 0;
        if (enabled) {
            std::fprintf(stderr, "[RVV1024] experimental query-tile=8 KV-tile=64 enabled\n");
        }
        return enabled;
    }();
    if (enable_bq8) {
        constexpr int BQ = 8;
        constexpr int BK = 64;
        constexpr int H = static_cast<int>(Q_HEADS_TILE);

        // 22,080 bytes/core: U, Q and the final Q8 buffers are addressable
        // scratch. This is in addition to the original 3 KiB q_four array;
        // neither L1 nor L2 residency is guaranteed for RVV memory accesses.
        struct alignas(64) tile_scratch {
            float accum[BQ][H][HEAD_DIM]; // online-softmax U, FP32
            uint16_t q[BQ][H][HEAD_DIM];
            int8_t q8[BQ][WO_BLOCKS_PER_Q_GROUP][QK];
            float scales[BQ][WO_BLOCKS_PER_Q_GROUP];
            float m[BQ][H];
            float l[BQ][H];
        } s;
        static_assert(sizeof(tile_scratch) <= 24 * 1024,
                      "eight-query scratch unexpectedly large");

        const auto zero = __riscv_vfmv_v_f_f32m2(0.0f, VL);
        const ptrdiff_t v_step = static_cast<ptrdiff_t>(v_trans ? vc->nb[2] : vc->nb[0]);

        // Only changes the attention microkernel. Q projection (BQ=8),
        // W_O and the four three-head worker groups are unchanged.
        static const bool pair_queries = [] {
            const char * e = std::getenv("LLAMA_RVV1024_PAIR_QUERIES");
            const bool enabled = e && std::strcmp(e, "1") == 0;
            if (enabled) {
                std::fprintf(stderr,
                    "[RVV1024] paired attention: 2 queries x 3 heads, split U halves\n");
            }
            return enabled;
        }();

        // Only the opt-in path uses the producer-side, position-major K cache.
        // Six QK score vectors remain live: 2 queries x 3 heads x 64 KV lanes.
        static const bool kcol_qk = [] {
            const char * e = std::getenv("LLAMA_RVV1024_KCOL_QK");
            return e && std::strcmp(e, "1") == 0;
        }();
        static const bool kcol_qk_verify = [] {
            const char * e = std::getenv("LLAMA_RVV1024_KCOL_QK_VERIFY");
            return e && std::strcmp(e, "1") == 0;
        }();
        // 1: avoid six scalar FP16->FP32 conversions and a vector FP16->FP32
        // conversion for every head dimension. Use widening FP16 FMA directly.
        // 0: retain the exact previous KCOL QK arithmetic for A/B testing.
        static const bool kcol_qk_half_fma = [] {
            const char * e = std::getenv("LLAMA_RVV1024_KCOL_HALF_FMA");
            return e && std::strcmp(e, "1") == 0;
        }();
        // Diagnostic only: store a whole FP32 score vector to a tiny stack
        // buffer once, instead of 64 vslidedown+extract operations. This
        // intentionally loses the register-only score property, so OFF by
        // default; it isolates the cost of score extraction on the A100.
        static const bool kcol_qk_score_store = [] {
            const char * e = std::getenv("LLAMA_RVV1024_KCOL_SCORE_STORE");
            return e && std::strcmp(e, "1") == 0;
        }();
        if (kcol_qk) {
            GGML_ASSERT(pair_queries && kcols &&
                        "KCOL-QK needs LLAMA_RVV1024_PAIR_QUERIES=1 and K_COLUMNS=1");
            static std::atomic_flag printed = ATOMIC_FLAG_INIT;
            if (!printed.test_and_set(std::memory_order_relaxed)) {
                std::fprintf(stderr, "[RVV1024] KCOL-QK: six FP32 score vectors, "
                                     "64 KV lanes, zero QK dot reductions; "
                                     "half_fma=%d score_store=%d\n",
                                     int(kcol_qk_half_fma), int(kcol_qk_score_store));
            }
        }

        for (int64_t qb = 0; qb < prepared->ne[1]; qb += BQ) {
            const int active = static_cast<int>(
                prepared->ne[1] - qb < BQ ? prepared->ne[1] - qb : BQ);
            const uint64_t tq_begin = phase_tick(timing_mode);

            // Reuse the branch's existing four-token register-stationary W_Q
            // kernel TWICE. This keeps the new four-token arithmetic, RoPE,
            // FP16 rounding, and W_Q weight reuse identical to the control.
            // The lambda pads a short final group of four internally; only
            // copy real query-token results into the eight-query macro-tile.
            static_assert(sizeof(q_four[0]) == sizeof(s.q[0]),
                          "Q tile and four-token projection layouts differ");
            for (int base = 0; base < active; base += 4) {
                project_q_four(qb + base);
                const int count = active - base < 4 ? active - base : 4;
                for (int t = 0; t < count; ++t) {
                    // Bit-exact transfer of 384 FP16 elements (no conversion).
                    // 768 bytes per query; the destination is cacheable scratch.
                    std::memcpy(s.q[base + t], q_four[t], sizeof(s.q[0]));
                }
            }

            // W_Q output sums are entirely internal to project_q_four.
            // Initialize reusable FP32 backing as independent online-softmax
            // U vectors, one [head, 128] per query token.
            for (int t = 0; t < active; ++t) {
                for (int h = 0; h < H; ++h) {
                    s.m[t][h] = -__builtin_inff();
                    s.l[t][h] = 0.0f;
                    __riscv_vse32_v_f32m2(s.accum[t][h], zero, VL);
                    __riscv_vse32_v_f32m2(s.accum[t][h] + VL, zero, VL);
                }
            }
            const uint64_t tq_end = phase_tick(timing_mode);
            if (timing_mode) {
                timing.q_rope += tq_end - tq_begin;
                timing.pairs += static_cast<uint64_t>((active + 1) / 2);
            }

            // KEY OPTIMIZATION: KV-tile-major instead of query-major.
            // Each 64-position KV tile is reused by 8 queries in the same
            // worker, and by both three-Q-head workers sharing a KV head.
            // The inner register microkernel owns just ONE query's 3 heads;
            // FP32 (m,l,U) is spilled only once per query per KV tile.
            const int kv_step = kcol_qk ? 64 : BK;
            for (int64_t kb = 0; kb < n_kv; kb += kv_step) {
                const int len = static_cast<int>(
                    n_kv - kb < kv_step ? n_kv - kb : kv_step);
                for (int t = 0; t < active; t += (pair_queries ? 2 : 1)) {
                    // Six Q-head computations = 3 heads for each of TWO queries,
                    // not a new six-head ownership partition. K is loaded once
                    // per position for both queries; V is loaded once per half.
                    // Preserve the original 1-query kernel for the A/B control
                    // and the unmatched tail query of an odd token batch.
                    if (pair_queries && t + 1 < active) {
                        // In the KCOL path, this holds probabilities ONLY.
                        // Raw QK score vectors never enter addressable scratch.
                        alignas(64) float prob[2][H][BK];
                        float alpha[2][H];
                        float tile_max[2][H] = {
                            {-__builtin_inff(), -__builtin_inff(), -__builtin_inff()},
                            {-__builtin_inff(), -__builtin_inff(), -__builtin_inff()}
                        };
                        const auto get_mask_score = [&](int64_t tok, int64_t kv) -> float {
                            const uint8_t * mp = static_cast<const uint8_t *>(mask->data) +
                                tok * mask->nb[1] + kv * mask->nb[0];
                            if (mask->type == GGML_TYPE_F32) {
                                return *reinterpret_cast<const float *>(mp);
                            }
                            const vuint16m1_t mh = __riscv_vle16_v_u16m1(
                                reinterpret_cast<const uint16_t *>(mp), 1);
                            return __riscv_vfmv_f_s_f32m2_f32(
                                __riscv_vfwcvt_f_f_v_f32m2(
                                    __riscv_vreinterpret_v_u16m1_f16m1(mh), 1));
                        };
                        if (kcol_qk) {
                            // Vector axis = 64 KV positions, NOT 128 head dimensions.
                            // The six FP32m2 vectors contain the six score rows.
                            // Each contiguous K column is shared by all six heads.
                            // No vfredusum, scalar-score stores, or runtime K transpose.
                            const size_t lanes = static_cast<size_t>(len);
                            GGML_ASSERT(len > 0 && len <= 64 && (kb % 64) == 0);
                            const uint8_t * col_base =
                                static_cast<const uint8_t *>(kcols->data) +
                                (kb / 64) * kcols->nb[3] + kv_head * kcols->nb[2];

                            vfloat32m2_t s00 = __riscv_vfmv_v_f_f32m2(0.0f, lanes);
                            vfloat32m2_t s01 = s00, s02 = s00;
                            vfloat32m2_t s10 = s00, s11 = s00, s12 = s00;
                            if (kcol_qk_half_fma) {
                                // Each key column is 64 FP16 elements; each Q
                                // is a scalar FP16.  vfwmacc.vf performs a
                                // widened FP16*FP16 -> FP32 accumulation in
                                // one RVV instruction per score row.
                                // Use memcpy for bit reinterpretation: s.q is
                                // uint16_t storage, not an _Float16 array.
                                const auto q_half = [&](int qi, int h, int dim) -> _Float16 {
                                    _Float16 val;
                                    __builtin_memcpy(&val, &s.q[t + qi][h][dim], sizeof(val));
                                    return val;
                                };
                                for (int dim = 0; dim < HEAD_DIM; ++dim) {
                                    const uint16_t * kp = reinterpret_cast<const uint16_t *>(
                                        col_base + dim * kcols->nb[1]);
                                    const vfloat16m1_t kh = __riscv_vreinterpret_v_u16m1_f16m1(
                                        __riscv_vle16_v_u16m1(kp, lanes));
                                    s00 = __riscv_vfwmacc_vf_f32m2(s00, q_half(0, 0, dim), kh, lanes);
                                    s01 = __riscv_vfwmacc_vf_f32m2(s01, q_half(0, 1, dim), kh, lanes);
                                    s02 = __riscv_vfwmacc_vf_f32m2(s02, q_half(0, 2, dim), kh, lanes);
                                    s10 = __riscv_vfwmacc_vf_f32m2(s10, q_half(1, 0, dim), kh, lanes);
                                    s11 = __riscv_vfwmacc_vf_f32m2(s11, q_half(1, 1, dim), kh, lanes);
                                    s12 = __riscv_vfwmacc_vf_f32m2(s12, q_half(1, 2, dim), kh, lanes);
                                }
                            } else {
                                // Original KCOL microkernel for isolated A/B.
                                for (int dim = 0; dim < HEAD_DIM; ++dim) {
                                    const uint16_t * kp =
                                        reinterpret_cast<const uint16_t *>(
                                            col_base + dim * kcols->nb[1]);
                                    const vfloat16m1_t kh = __riscv_vreinterpret_v_u16m1_f16m1(
                                        __riscv_vle16_v_u16m1(kp, lanes));
                                    const vfloat32m2_t kf = __riscv_vfwcvt_f_f_v_f32m2(kh, lanes);
                                    s00 = __riscv_vfmacc_vf_f32m2(s00,
                                        ggml_fp16_to_fp32(s.q[t    ][0][dim]), kf, lanes);
                                    s01 = __riscv_vfmacc_vf_f32m2(s01,
                                        ggml_fp16_to_fp32(s.q[t    ][1][dim]), kf, lanes);
                                    s02 = __riscv_vfmacc_vf_f32m2(s02,
                                        ggml_fp16_to_fp32(s.q[t    ][2][dim]), kf, lanes);
                                    s10 = __riscv_vfmacc_vf_f32m2(s10,
                                        ggml_fp16_to_fp32(s.q[t + 1][0][dim]), kf, lanes);
                                    s11 = __riscv_vfmacc_vf_f32m2(s11,
                                        ggml_fp16_to_fp32(s.q[t + 1][1][dim]), kf, lanes);
                                    s12 = __riscv_vfmacc_vf_f32m2(s12,
                                        ggml_fp16_to_fp32(s.q[t + 1][2][dim]), kf, lanes);
                                }
                            }

                            // Add attention scaling and causal masks without
                            // reducing any individual QK dot product.
                            const auto mask_row = [&](int qi) -> vfloat32m2_t {
                                const uint8_t * mp =
                                    static_cast<const uint8_t *>(mask->data) +
                                    (qb + t + qi) * mask->nb[1] + kb * mask->nb[0];
                                if (mask->type == GGML_TYPE_F32) {
                                    return __riscv_vle32_v_f32m2(
                                        reinterpret_cast<const float *>(mp), lanes);
                                }
                                const vfloat16m1_t mh = __riscv_vreinterpret_v_u16m1_f16m1(
                                    __riscv_vle16_v_u16m1(
                                        reinterpret_cast<const uint16_t *>(mp), lanes));
                                return __riscv_vfwcvt_f_f_v_f32m2(mh, lanes);
                            };
                            const vfloat32m2_t mask0 = mask_row(0);
                            const vfloat32m2_t mask1 = mask_row(1);
                            const auto scaled = [&](vfloat32m2_t x, vfloat32m2_t m) {
                                return __riscv_vfadd_vv_f32m2(
                                    __riscv_vfmul_vf_f32m2(x, ATTN_SCALE, lanes),
                                    m, lanes);
                            };
                            s00 = scaled(s00, mask0);
                            s01 = scaled(s01, mask0);
                            s02 = scaled(s02, mask0);
                            s10 = scaled(s10, mask1);
                            s11 = scaled(s11, mask1);
                            s12 = scaled(s12, mask1);

                            const auto score_lane = [&](vfloat32m2_t x, int j) -> float {
                                return __riscv_vfmv_f_s_f32m2_f32(
                                    __riscv_vslidedown_vx_f32m2(
                                        x, static_cast<size_t>(j), 1));
                            };

                            // Optional one-time numerical test against the
                            // original contiguous FP16 K cache, including masks.
                            if (kcol_qk_verify && qb == 0 && t == 0 && kb == 0 && group == 0) {
                                const auto check = [&](int qi, int h, vfloat32m2_t x) {
                                    for (int j = 0; j < len; ++j) {
                                        const float got = score_lane(x, j);
                                        const float ms = get_mask_score(qb + t + qi, kb + j);
                                        if (ms == -__builtin_inff()) {
                                            GGML_ASSERT(got == -__builtin_inff());
                                            continue;
                                        }
                                        const uint8_t * ref_k =
                                            static_cast<const uint8_t *>(kc->data) +
                                            (kb + j) * kc->nb[2] + kv_head * kc->nb[1];
                                        float ref = 0.0f;
                                        for (int dim = 0; dim < HEAD_DIM; ++dim) {
                                            const uint16_t kb16 = *reinterpret_cast<const uint16_t *>(
                                                ref_k + dim * kc->nb[0]);
                                            ref = __builtin_fmaf(
                                                ggml_fp16_to_fp32(s.q[t + qi][h][dim]),
                                                ggml_fp16_to_fp32(kb16), ref);
                                        }
                                        ref = ref * ATTN_SCALE + ms;
                                        GGML_ASSERT(__builtin_isfinite(got) &&
                                                    __builtin_isfinite(ref) &&
                                                    __builtin_fabsf(got - ref) <=
                                                        0.002f + 0.001f * __builtin_fabsf(ref));
                                    }
                                };
                                check(0, 0, s00); check(0, 1, s01); check(0, 2, s02);
                                check(1, 0, s10); check(1, 1, s11); check(1, 2, s12);
                                static std::atomic_flag reported = ATOMIC_FLAG_INIT;
                                if (!reported.test_and_set(std::memory_order_relaxed)) {
                                    std::fprintf(stderr,
                                        "[RVV1024] KCOL-QK VERIFY PASS: %d scores checked "
                                        "(other layers checked silently)\n", 6 * len);
                                }
                            }

                            // Only SIX maximum reductions (not 6*len dot
                            // reductions). Convert each register-resident
                            // score row directly to scalar probabilities.
                            // prob[][][] is then used by the UNCHANGED PV stage.
                            const auto max_lane = [&](vfloat32m2_t x) -> float {
                                const vfloat32m1_t seed =
                                    __riscv_vfmv_v_f_f32m1(-__builtin_inff(), 1);
                                return __riscv_vfmv_f_s_f32m1_f32(
                                    __riscv_vfredmax_vs_f32m2_f32m1(x, seed, lanes));
                            };
                            bool any_valid = false;
                            const auto finalize = [&](int qi, int h, vfloat32m2_t scores) {
                                const float tm = max_lane(scores);
                                tile_max[qi][h] = tm;
                                any_valid = any_valid || tm != -__builtin_inff();
                                const float old_m = s.m[t + qi][h];
                                float new_m = old_m;
                                float scale = 1.0f;
                                float denom = s.l[t + qi][h];
                                if (tm != -__builtin_inff()) {
                                    if (tm > new_m) new_m = tm;
                                    if (denom != 0.0f && new_m > old_m) {
                                        scale = exp_neg(old_m - new_m);
                                        denom *= scale;
                                    }
                                }
                                alpha[qi][h] = scale;
                                s.m[t + qi][h] = new_m;
                                const auto add_score = [&](int j, float sc) {
                                    const float p = sc == -__builtin_inff() ?
                                        0.0f : exp_neg(sc - new_m);
                                    prob[qi][h][j] = p;
                                    denom += p;
                                };
                                if (kcol_qk_score_store) {
                                    // Experimental score export: 256 bytes per
                                    // head per tile, for A/B against slides.
                                    alignas(64) float scores_in_l1[64];
                                    __riscv_vse32_v_f32m2(scores_in_l1, scores, lanes);
                                    for (int j = 0; j < len; ++j) {
                                        add_score(j, scores_in_l1[j]);
                                    }
                                } else {
                                    // Original register-resident score path.
                                    for (int j = 0; j < len; ++j) {
                                        add_score(j, score_lane(scores, j));
                                    }
                                }
                                s.l[t + qi][h] = denom;
                            };
                            finalize(0, 0, s00); finalize(0, 1, s01); finalize(0, 2, s02);
                            finalize(1, 0, s10); finalize(1, 1, s11); finalize(1, 2, s12);
                            if (!any_valid) continue;
                        } else {
                        {
                            // QK phase: six Q vectors (12 f16m1 registers) and
                            // two K halves. The U accumulators are NOT live yet.
                            const vfloat16m1_t q000 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t][0], VL));
                            const vfloat16m1_t q001 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t][0] + VL, VL));
                            const vfloat16m1_t q010 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t][1], VL));
                            const vfloat16m1_t q011 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t][1] + VL, VL));
                            const vfloat16m1_t q020 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t][2], VL));
                            const vfloat16m1_t q021 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t][2] + VL, VL));
                            const vfloat16m1_t q100 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t + 1][0], VL));
                            const vfloat16m1_t q101 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t + 1][0] + VL, VL));
                            const vfloat16m1_t q110 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t + 1][1], VL));
                            const vfloat16m1_t q111 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t + 1][1] + VL, VL));
                            const vfloat16m1_t q120 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t + 1][2], VL));
                            const vfloat16m1_t q121 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(s.q[t + 1][2] + VL, VL));

                            const auto dot_score = [&](vfloat16m1_t qlo, vfloat16m1_t qhi,
                                                        vfloat16m1_t klo, vfloat16m1_t khi,
                                                        float mask_score) -> float {
                                return sum64(__riscv_vfwmacc_vv_f32m2(
                                    __riscv_vfwmul_vv_f32m2(qlo, klo, VL), qhi, khi, VL)) *
                                    ATTN_SCALE + mask_score;
                            };
                            for (int j = 0; j < len; ++j) {
                                const int64_t kv = kb + j;
                                const float mask0 = get_mask_score(qb + t, kv);
                                const float mask1 = get_mask_score(qb + t + 1, kv);
                                if (mask0 == -__builtin_inff() && mask1 == -__builtin_inff()) {
                                    for (int h = 0; h < H; ++h) {
                                        prob[0][h][j] = -__builtin_inff();
                                        prob[1][h][j] = -__builtin_inff();
                                    }
                                    continue;
                                }
                                const uint8_t * kp = static_cast<const uint8_t *>(kc->data) +
                                    kv * kc->nb[2] + kv_head * kc->nb[1];
                                vfloat16m1_t k0 = __riscv_vreinterpret_v_u16m1_f16m1(
                                    __riscv_vle16_v_u16m1(
                                        reinterpret_cast<const uint16_t *>(kp), VL));
                                const vfloat16m1_t k1 = __riscv_vreinterpret_v_u16m1_f16m1(
                                    __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(
                                        kp + VL * kc->nb[0]), VL));
                                if (mask0 != -__builtin_inff()) {
                                    prob[0][0][j] = dot_score(q000, q001, k0, k1, mask0);
                                    __asm__ __volatile__("" : "+vr"(k0) : "f"(prob[0][0][j]));
                                    prob[0][1][j] = dot_score(q010, q011, k0, k1, mask0);
                                    __asm__ __volatile__("" : "+vr"(k0) : "f"(prob[0][1][j]));
                                    prob[0][2][j] = dot_score(q020, q021, k0, k1, mask0);
                                } else {
                                    prob[0][0][j] = prob[0][1][j] = prob[0][2][j] = -__builtin_inff();
                                }
                                if (mask1 != -__builtin_inff()) {
                                    prob[1][0][j] = dot_score(q100, q101, k0, k1, mask1);
                                    __asm__ __volatile__("" : "+vr"(k0) : "f"(prob[1][0][j]));
                                    prob[1][1][j] = dot_score(q110, q111, k0, k1, mask1);
                                    __asm__ __volatile__("" : "+vr"(k0) : "f"(prob[1][1][j]));
                                    prob[1][2][j] = dot_score(q120, q121, k0, k1, mask1);
                                } else {
                                    prob[1][0][j] = prob[1][1][j] = prob[1][2][j] = -__builtin_inff();
                                }
                                for (int h = 0; h < H; ++h) {
                                    if (prob[0][h][j] > tile_max[0][h])
                                        tile_max[0][h] = prob[0][h][j];
                                    if (prob[1][h][j] > tile_max[1][h])
                                        tile_max[1][h] = prob[1][h][j];
                                }
                            }
                        }

                        bool any_valid = false;
                        for (int qi = 0; qi < 2; ++qi)
                            for (int h = 0; h < H; ++h)
                                any_valid = any_valid || tile_max[qi][h] != -__builtin_inff();
                        if (!any_valid) continue;

                        // Normalize each head's scores to the merged running
                        // maximum and turn score scratch into probabilities.
                        // l and m are updated only ONCE, not per U half.
                        for (int qi = 0; qi < 2; ++qi) {
                            for (int h = 0; h < H; ++h) {
                                const float tm = tile_max[qi][h];
                                const float old_m = s.m[t + qi][h];
                                float new_m = old_m;
                                float scale = 1.0f;
                                float denom = s.l[t + qi][h];
                                if (tm != -__builtin_inff()) {
                                    if (tm > new_m) new_m = tm;
                                    if (denom != 0.0f && new_m > old_m) {
                                        scale = exp_neg(old_m - new_m);
                                        denom *= scale;
                                    }
                                }
                                alpha[qi][h] = scale;
                                s.m[t + qi][h] = new_m;
                                for (int j = 0; j < len; ++j) {
                                    const float score = prob[qi][h][j];
                                    const float p = score == -__builtin_inff() ?
                                        0.0f : exp_neg(score - new_m);
                                    prob[qi][h][j] = p;
                                    denom += p;
                                }
                                s.l[t + qi][h] = denom;
                            }
                        }
                        } // legacy head-dimension-vectorized QK

                        // PV phase: one 64-element half of each of six U vectors.
                        // Six f32m2 accumulators = 12 physical RVV registers;
                        // V is one f32m2 (2 registers) at a time. The high and
                        // low halves never have to coexist in registers.
                        const auto accumulate_half = [&](int dim_offset) {
                            vfloat32m2_t u00 = __riscv_vle32_v_f32m2(s.accum[t][0] + dim_offset, VL);
                            vfloat32m2_t u01 = __riscv_vle32_v_f32m2(s.accum[t][1] + dim_offset, VL);
                            vfloat32m2_t u02 = __riscv_vle32_v_f32m2(s.accum[t][2] + dim_offset, VL);
                            vfloat32m2_t u10 = __riscv_vle32_v_f32m2(s.accum[t + 1][0] + dim_offset, VL);
                            vfloat32m2_t u11 = __riscv_vle32_v_f32m2(s.accum[t + 1][1] + dim_offset, VL);
                            vfloat32m2_t u12 = __riscv_vle32_v_f32m2(s.accum[t + 1][2] + dim_offset, VL);
                            u00 = __riscv_vfmul_vf_f32m2(u00, alpha[0][0], VL);
                            u01 = __riscv_vfmul_vf_f32m2(u01, alpha[0][1], VL);
                            u02 = __riscv_vfmul_vf_f32m2(u02, alpha[0][2], VL);
                            u10 = __riscv_vfmul_vf_f32m2(u10, alpha[1][0], VL);
                            u11 = __riscv_vfmul_vf_f32m2(u11, alpha[1][1], VL);
                            u12 = __riscv_vfmul_vf_f32m2(u12, alpha[1][2], VL);
                            for (int j = 0; j < len; ++j) {
                                const float p00 = prob[0][0][j], p01 = prob[0][1][j];
                                const float p02 = prob[0][2][j], p10 = prob[1][0][j];
                                const float p11 = prob[1][1][j], p12 = prob[1][2][j];
                                if (p00 == 0.0f && p01 == 0.0f && p02 == 0.0f &&
                                    p10 == 0.0f && p11 == 0.0f && p12 == 0.0f) continue;
                                const int64_t kv = kb + j;
                                const uint8_t * vp = static_cast<const uint8_t *>(vc->data) +
                                    (v_trans ? kv * vc->nb[0] + kv_head * vc->nb[1] :
                                               kv * vc->nb[2] + kv_head * vc->nb[1]);
                                const vfloat32m2_t v = f16_load64(
                                    vp + dim_offset * v_step, v_step);
                                u00 = __riscv_vfmacc_vf_f32m2(u00, p00, v, VL);
                                u01 = __riscv_vfmacc_vf_f32m2(u01, p01, v, VL);
                                u02 = __riscv_vfmacc_vf_f32m2(u02, p02, v, VL);
                                u10 = __riscv_vfmacc_vf_f32m2(u10, p10, v, VL);
                                u11 = __riscv_vfmacc_vf_f32m2(u11, p11, v, VL);
                                u12 = __riscv_vfmacc_vf_f32m2(u12, p12, v, VL);
                            }
                            __riscv_vse32_v_f32m2(s.accum[t][0] + dim_offset, u00, VL);
                            __riscv_vse32_v_f32m2(s.accum[t][1] + dim_offset, u01, VL);
                            __riscv_vse32_v_f32m2(s.accum[t][2] + dim_offset, u02, VL);
                            __riscv_vse32_v_f32m2(s.accum[t + 1][0] + dim_offset, u10, VL);
                            __riscv_vse32_v_f32m2(s.accum[t + 1][1] + dim_offset, u11, VL);
                            __riscv_vse32_v_f32m2(s.accum[t + 1][2] + dim_offset, u12, VL);
                        };
                        accumulate_half(0);
                        accumulate_half(static_cast<int>(VL));
                        continue;
                    }

                    // One-query fallback: original BQ8 attention body.
                    const int64_t token = qb + t;
                    alignas(64) float score[H][BK];
                    float tile_m0 = -__builtin_inff();
                    float tile_m1 = -__builtin_inff();
                    float tile_m2 = -__builtin_inff();
                    {
                        const vfloat16m1_t q00 = __riscv_vreinterpret_v_u16m1_f16m1(
                            __riscv_vle16_v_u16m1(s.q[t][0], VL));
                        const vfloat16m1_t q01 = __riscv_vreinterpret_v_u16m1_f16m1(
                            __riscv_vle16_v_u16m1(s.q[t][0] + VL, VL));
                        const vfloat16m1_t q10 = __riscv_vreinterpret_v_u16m1_f16m1(
                            __riscv_vle16_v_u16m1(s.q[t][1], VL));
                        const vfloat16m1_t q11 = __riscv_vreinterpret_v_u16m1_f16m1(
                            __riscv_vle16_v_u16m1(s.q[t][1] + VL, VL));
                        const vfloat16m1_t q20 = __riscv_vreinterpret_v_u16m1_f16m1(
                            __riscv_vle16_v_u16m1(s.q[t][2], VL));
                        const vfloat16m1_t q21 = __riscv_vreinterpret_v_u16m1_f16m1(
                            __riscv_vle16_v_u16m1(s.q[t][2] + VL, VL));

                        for (int j = 0; j < len; ++j) {
                            const int64_t kv = kb + j;
                            const uint8_t * mp = static_cast<const uint8_t *>(mask->data) +
                                token * mask->nb[1] + kv * mask->nb[0];
                            float mask_score;
                            if (mask->type == GGML_TYPE_F32) {
                                mask_score = *reinterpret_cast<const float *>(mp);
                            } else {
                                const vuint16m1_t mh = __riscv_vle16_v_u16m1(
                                    reinterpret_cast<const uint16_t *>(mp), 1);
                                mask_score = __riscv_vfmv_f_s_f32m2_f32(
                                    __riscv_vfwcvt_f_f_v_f32m2(
                                        __riscv_vreinterpret_v_u16m1_f16m1(mh), 1));
                            }
                            if (mask_score == -__builtin_inff()) {
                                score[0][j] = score[1][j] = score[2][j] = -__builtin_inff();
                                continue;
                            }
                            const uint8_t * kp = static_cast<const uint8_t *>(kc->data) +
                                kv * kc->nb[2] + kv_head * kc->nb[1];
                            vfloat16m1_t k0 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(kp), VL));
                            const vfloat16m1_t k1 = __riscv_vreinterpret_v_u16m1_f16m1(
                                __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(
                                    kp + VL * kc->nb[0]), VL));
                            score[0][j] = sum64(__riscv_vfwmacc_vv_f32m2(
                                __riscv_vfwmul_vv_f32m2(q00, k0, VL), q01, k1, VL)) *
                                ATTN_SCALE + mask_score;
                            // Keep the existing score-order compiler constraints.
                            __asm__ __volatile__("" : "+vr"(k0) : "f"(score[0][j]));
                            score[1][j] = sum64(__riscv_vfwmacc_vv_f32m2(
                                __riscv_vfwmul_vv_f32m2(q10, k0, VL), q11, k1, VL)) *
                                ATTN_SCALE + mask_score;
                            __asm__ __volatile__("" : "+vr"(k0) : "f"(score[1][j]));
                            score[2][j] = sum64(__riscv_vfwmacc_vv_f32m2(
                                __riscv_vfwmul_vv_f32m2(q20, k0, VL), q21, k1, VL)) *
                                ATTN_SCALE + mask_score;
                            if (score[0][j] > tile_m0) tile_m0 = score[0][j];
                            if (score[1][j] > tile_m1) tile_m1 = score[1][j];
                            if (score[2][j] > tile_m2) tile_m2 = score[2][j];
                        }
                    }
                    if (tile_m0 == -__builtin_inff() &&
                        tile_m1 == -__builtin_inff() &&
                        tile_m2 == -__builtin_inff()) continue;

                    float m0 = s.m[t][0], m1 = s.m[t][1], m2 = s.m[t][2];
                    float l0 = s.l[t][0], l1 = s.l[t][1], l2 = s.l[t][2];
                    vfloat32m2_t o00 = __riscv_vle32_v_f32m2(s.accum[t][0], VL);
                    vfloat32m2_t o01 = __riscv_vle32_v_f32m2(s.accum[t][0] + VL, VL);
                    vfloat32m2_t o10 = __riscv_vle32_v_f32m2(s.accum[t][1], VL);
                    vfloat32m2_t o11 = __riscv_vle32_v_f32m2(s.accum[t][1] + VL, VL);
                    vfloat32m2_t o20 = __riscv_vle32_v_f32m2(s.accum[t][2], VL);
                    vfloat32m2_t o21 = __riscv_vle32_v_f32m2(s.accum[t][2] + VL, VL);
                    const auto merge_max = [&](float tm, float & m, float & l,
                                               vfloat32m2_t & u0, vfloat32m2_t & u1) {
                        if (tm == -__builtin_inff()) return;
                        const float new_m = tm > m ? tm : m;
                        if (l != 0.0f && new_m > m) {
                            const float alpha = exp_neg(m - new_m);
                            l *= alpha;
                            u0 = __riscv_vfmul_vf_f32m2(u0, alpha, VL);
                            u1 = __riscv_vfmul_vf_f32m2(u1, alpha, VL);
                        }
                        m = new_m;
                    };
                    merge_max(tile_m0, m0, l0, o00, o01);
                    merge_max(tile_m1, m1, l1, o10, o11);
                    merge_max(tile_m2, m2, l2, o20, o21);

                    for (int j = 0; j < len; ++j) {
                        if (score[0][j] == -__builtin_inff() &&
                            score[1][j] == -__builtin_inff() &&
                            score[2][j] == -__builtin_inff()) continue;
                        const float p0 = score[0][j] == -__builtin_inff() ?
                            0.0f : exp_neg(score[0][j] - m0);
                        const float p1 = score[1][j] == -__builtin_inff() ?
                            0.0f : exp_neg(score[1][j] - m1);
                        const float p2 = score[2][j] == -__builtin_inff() ?
                            0.0f : exp_neg(score[2][j] - m2);
                        l0 += p0; l1 += p1; l2 += p2;
                        const int64_t kv = kb + j;
                        const uint8_t * vp = static_cast<const uint8_t *>(vc->data) +
                            (v_trans ? kv * vc->nb[0] + kv_head * vc->nb[1] :
                                       kv * vc->nb[2] + kv_head * vc->nb[1]);
                        const vfloat32m2_t v0 = f16_load64(vp, v_step);
                        const vfloat32m2_t v1 = f16_load64(vp + VL * v_step, v_step);
                        o00 = __riscv_vfmacc_vf_f32m2(o00, p0, v0, VL);
                        o01 = __riscv_vfmacc_vf_f32m2(o01, p0, v1, VL);
                        o10 = __riscv_vfmacc_vf_f32m2(o10, p1, v0, VL);
                        o11 = __riscv_vfmacc_vf_f32m2(o11, p1, v1, VL);
                        o20 = __riscv_vfmacc_vf_f32m2(o20, p2, v0, VL);
                        o21 = __riscv_vfmacc_vf_f32m2(o21, p2, v1, VL);
                    }
                    s.m[t][0] = m0; s.m[t][1] = m1; s.m[t][2] = m2;
                    s.l[t][0] = l0; s.l[t][1] = l1; s.l[t][2] = l2;
                    __riscv_vse32_v_f32m2(s.accum[t][0], o00, VL);
                    __riscv_vse32_v_f32m2(s.accum[t][0] + VL, o01, VL);
                    __riscv_vse32_v_f32m2(s.accum[t][1], o10, VL);
                    __riscv_vse32_v_f32m2(s.accum[t][1] + VL, o11, VL);
                    __riscv_vse32_v_f32m2(s.accum[t][2], o20, VL);
                    __riscv_vse32_v_f32m2(s.accum[t][2] + VL, o21, VL);
                }
            }
            const uint64_t ta_end = phase_tick(timing_mode);
            if (timing_mode) timing.attention += ta_end - tq_end;

            // Finalize online softmax and quantize directly from RVV registers;
            // no FP32 attention-output tensor is written. The normalization
            // runs in the Q8+W_O timing phase for this experimental path.
            for (int t = 0; t < active; ++t) {
                for (int h = 0; h < H; ++h) {
                    GGML_ASSERT(s.l[t][h] > 0.0f);
                    const float inv = 1.0f / s.l[t][h];
                    const vfloat32m2_t lo = __riscv_vfmul_vf_f32m2(
                        __riscv_vle32_v_f32m2(s.accum[t][h], VL), inv, VL);
                    const vfloat32m2_t hi = __riscv_vfmul_vf_f32m2(
                        __riscv_vle32_v_f32m2(s.accum[t][h] + VL, VL), inv, VL);
                    const int block = 4 * h;
                    __riscv_vse8_v_i8mf2(s.q8[t][block + 0],
                        quant32(lo, 0, s.scales[t][block + 0]), QK);
                    __riscv_vse8_v_i8mf2(s.q8[t][block + 1],
                        quant32(lo, 1, s.scales[t][block + 1]), QK);
                    __riscv_vse8_v_i8mf2(s.q8[t][block + 2],
                        quant32(hi, 0, s.scales[t][block + 2]), QK);
                    __riscv_vse8_v_i8mf2(s.q8[t][block + 3],
                        quant32(hi, 1, s.scales[t][block + 3]), QK);
                }
            }

            // W_O output-subtile-major for reuse of each 60 KiB group subtile
            // across four query pairs. Each actual inner W_O call still uses
            // register-resident FP32 partial sums and the existing microkernel.
            for (int yt = 0; yt < WO_SUBTILES; ++yt) {
                const uint8_t * ws = wo_tile + static_cast<size_t>(yt) * WO_SUBTILE_BYTES;
                for (int t = 0; t < active; t += 2) {
                    float * y0 = reinterpret_cast<float *>(
                        static_cast<uint8_t *>(parts->data) +
                        group * parts->nb[2] + (qb + t) * parts->nb[1]);
                    const bool has_second = t + 1 < active;
                    vfloat32m4_t a0 = __riscv_vfmv_v_f_f32m4(0.0f, WO_VL);
                    vfloat32m4_t a1 = __riscv_vfmv_v_f_f32m4(0.0f, WO_VL);
                    if (has_second) {
                        float * y1 = reinterpret_cast<float *>(
                            static_cast<uint8_t *>(parts->data) +
                            group * parts->nb[2] + (qb + t + 1) * parts->nb[1]);
                        vfloat32m4_t b0 = __riscv_vfmv_v_f_f32m4(0.0f, WO_VL);
                        vfloat32m4_t b1 = __riscv_vfmv_v_f_f32m4(0.0f, WO_VL);
                        for (int j = 0; j < WO_BLOCKS_PER_Q_GROUP; ++j) {
                            wo_block_two_tokens(ws + j * WO_BLOCK_BYTES,
                                s.q8[t][j], s.q8[t + 1][j],
                                s.scales[t][j], s.scales[t + 1][j],
                                a0, a1, b0, b1);
                        }
                        __riscv_vse32_v_f32m4(y1 + yt * WO_SUBTILE_OUTPUTS, b0, WO_VL);
                        __riscv_vse32_v_f32m4(y1 + yt * WO_SUBTILE_OUTPUTS + WO_VL, b1, WO_VL);
                    } else {
                        for (int j = 0; j < WO_BLOCKS_PER_Q_GROUP; ++j) {
                            wo_block_both(ws + j * WO_BLOCK_BYTES,
                                s.q8[t][j], s.scales[t][j], a0, a1);
                        }
                    }
                    __riscv_vse32_v_f32m4(y0 + yt * WO_SUBTILE_OUTPUTS, a0, WO_VL);
                    __riscv_vse32_v_f32m4(y0 + yt * WO_SUBTILE_OUTPUTS + WO_VL, a1, WO_VL);
                }
            }
            if (timing_mode) {
                timing.q8_wo += phase_tick(timing_mode) - ta_end;
                timing.tokens += static_cast<uint64_t>(active);
            }
        }
        if (timing_mode) phase_record(group, timing);
        return;
    }
}
