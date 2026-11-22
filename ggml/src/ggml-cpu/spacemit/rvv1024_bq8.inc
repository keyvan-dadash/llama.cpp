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
            for (int64_t kb = 0; kb < n_kv; kb += BK) {
                const int len = static_cast<int>(n_kv - kb < BK ? n_kv - kb : BK);
                for (int t = 0; t < active; ++t) {
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
