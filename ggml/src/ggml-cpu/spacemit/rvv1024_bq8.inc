// RVV1024 BQ8 attention: vectorize QK across 64 KV positions.
// Four queries x three heads share each K column; odd tails use the same path.
// Widening FP16 FMA accumulates up to 12 score vectors; PV runs as two-query pairs.
{
    // One RVV1024 path: BQ8, K columns and FP16 widening FMA.
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

        GGML_ASSERT(kcols && "RVV1024 requires a K-column cache");
        static std::atomic_flag once = ATOMIC_FLAG_INIT;
        if (!once.test_and_set(std::memory_order_relaxed)) {
            std::fprintf(stderr,
                "[RVV1024] BQ8 KCOL-QK: 4 queries x 3 heads, 64 KV lanes, "
                "2-query P-vector PV via vrgather.vx + vfmacc.vv\n");
        }

        for (int64_t qb = 0; qb < prepared->ne[1]; qb += BQ) {
            const int active = static_cast<int>(
                prepared->ne[1] - qb < BQ ? prepared->ne[1] - qb : BQ);

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
            // KEY OPTIMIZATION: KV-tile-major instead of query-major.
            // Each 64-position KV tile is reused by 8 queries in the same
            // worker, and by both three-Q-head workers sharing a KV head.
            // The QK microkernel shares each K column across up to FOUR queries;
            // the P/V microkernel processes only TWO queries at a time.
            for (int64_t kb = 0; kb < n_kv; kb += BK) {
                const int len = static_cast<int>(n_kv - kb < BK ? n_kv - kb : BK);
                for (int t = 0; t < active; t += 4) {
                    const int nq = active - t < 4 ? active - t : 4;
                    const bool has_second = nq > 1;
                    const bool has_third  = nq > 2;
                    const bool has_fourth = nq > 3;
                    {
                        // First pair stays in RVV P registers; stage only the
                        // deferred pair (2 x 3 heads x 64 FP32 = 1.5 KiB).
                        // This bounds PV register pressure independently of QK.
                        vfloat32m2_t p00, p01, p02, p10, p11, p12;
                        alignas(64) float deferred_p[2][H][BK];
                        float alpha[4][H];
                        {
                            // Vector axis = 64 KV positions, NOT 128 head dimensions.
                            // Up to twelve FP32m2 vectors hold QK scores.
                            // Each K-column load serves up to four queries.
                            // No per-key dot reductions, score exports, or K transpose.
                            const size_t lanes = static_cast<size_t>(len);
                            GGML_ASSERT(len > 0 && len <= 64 && (kb % 64) == 0);
                            const uint8_t * col_base =
                                static_cast<const uint8_t *>(kcols->data) +
                                (kb / 64) * kcols->nb[3] + kv_head * kcols->nb[2];

                            vfloat32m2_t s00 = __riscv_vfmv_v_f_f32m2(0.0f, lanes);
                            vfloat32m2_t s01 = s00, s02 = s00;
                            vfloat32m2_t s10 = s00, s11 = s00, s12 = s00;
                            vfloat32m2_t s20 = s00, s21 = s00, s22 = s00;
                            vfloat32m2_t s30 = s00, s31 = s00, s32 = s00;
                            // Load 64 FP16 K values once per dimension, then
                            // update up to twelve FP32 score vectors.
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
                                if (has_second) {
                                    s10 = __riscv_vfwmacc_vf_f32m2(s10, q_half(1, 0, dim), kh, lanes);
                                    s11 = __riscv_vfwmacc_vf_f32m2(s11, q_half(1, 1, dim), kh, lanes);
                                    s12 = __riscv_vfwmacc_vf_f32m2(s12, q_half(1, 2, dim), kh, lanes);
                                }
                                if (has_third) {
                                    s20 = __riscv_vfwmacc_vf_f32m2(s20, q_half(2, 0, dim), kh, lanes);
                                    s21 = __riscv_vfwmacc_vf_f32m2(s21, q_half(2, 1, dim), kh, lanes);
                                    s22 = __riscv_vfwmacc_vf_f32m2(s22, q_half(2, 2, dim), kh, lanes);
                                }
                                if (has_fourth) {
                                    s30 = __riscv_vfwmacc_vf_f32m2(s30, q_half(3, 0, dim), kh, lanes);
                                    s31 = __riscv_vfwmacc_vf_f32m2(s31, q_half(3, 1, dim), kh, lanes);
                                    s32 = __riscv_vfwmacc_vf_f32m2(s32, q_half(3, 2, dim), kh, lanes);
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
                            const auto scaled = [&](vfloat32m2_t x, vfloat32m2_t m) {
                                return __riscv_vfadd_vv_f32m2(
                                    __riscv_vfmul_vf_f32m2(x, ATTN_SCALE, lanes),
                                    m, lanes);
                            };
                            s00 = scaled(s00, mask0);
                            s01 = scaled(s01, mask0);
                            s02 = scaled(s02, mask0);
                            if (has_second) {
                                const vfloat32m2_t mask1 = mask_row(1);
                                s10 = scaled(s10, mask1);
                                s11 = scaled(s11, mask1);
                                s12 = scaled(s12, mask1);
                            }
                            if (has_third) {
                                const vfloat32m2_t mask2 = mask_row(2);
                                s20 = scaled(s20, mask2);
                                s21 = scaled(s21, mask2);
                                s22 = scaled(s22, mask2);
                            }
                            if (has_fourth) {
                                const vfloat32m2_t mask3 = mask_row(3);
                                s30 = scaled(s30, mask3);
                                s31 = scaled(s31, mask3);
                                s32 = scaled(s32, mask3);
                            }

                            // Vector form of the existing exp_neg() polynomial.
                            // Inputs are <= 0. A masked (-inf) score becomes zero;
                            // no probability leaves a vector register.
                            const auto exp_neg_vector = [&](vfloat32m2_t x) -> vfloat32m2_t {
                                const vbool16_t keep = __riscv_vmfgt_vf_f32m2_b16(x, -85.0f, lanes);
                                const vfloat32m2_t safe = __riscv_vfmax_vf_f32m2(x, -85.0f, lanes);
                                const vfloat32m2_t z = __riscv_vfmul_vf_f32m2(
                                    safe, 1.4426950408889634f, lanes);
                                // exp_neg() rounds the negative base-2 exponent
                                // away from zero; RTZ(z - 0.5) matches that.
                                const vint32m2_t n = __riscv_vfcvt_rtz_x_f_v_i32m2(
                                    __riscv_vfsub_vf_f32m2(z, 0.5f, lanes), lanes);
                                const vfloat32m2_t nf = __riscv_vfcvt_f_x_v_f32m2(n, lanes);
                                const vfloat32m2_t r = __riscv_vfsub_vv_f32m2(safe,
                                    __riscv_vfmul_vf_f32m2(nf, 0.6931471805599453f, lanes), lanes);
                                const auto horner = [&](vfloat32m2_t p, float c) {
                                    return __riscv_vfadd_vf_f32m2(
                                        __riscv_vfmul_vv_f32m2(p, r, lanes), c, lanes);
                                };
                                vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(1.0f / 5040.0f, lanes);
                                p = horner(p, 1.0f / 720.0f);
                                p = horner(p, 1.0f / 120.0f);
                                p = horner(p, 1.0f / 24.0f);
                                p = horner(p, 1.0f / 6.0f);
                                p = horner(p, 0.5f);
                                p = horner(p, 1.0f);
                                p = horner(p, 1.0f);
                                const vint32m2_t pow_bits = __riscv_vsll_vx_i32m2(
                                    __riscv_vadd_vx_i32m2(n, 127, lanes), 23, lanes);
                                const vfloat32m2_t pow2 =
                                    __riscv_vreinterpret_v_i32m2_f32m2(pow_bits);
                                const vfloat32m2_t result =
                                    __riscv_vfmul_vv_f32m2(p, pow2, lanes);
                                return __riscv_vmerge_vvm_f32m2(
                                    __riscv_vfmv_v_f_f32m2(0.0f, lanes), result, keep, lanes);
                            };
                            const auto max_lane = [&](vfloat32m2_t x) -> float {
                                const vfloat32m1_t seed =
                                    __riscv_vfmv_v_f_f32m1(-__builtin_inff(), 1);
                                return __riscv_vfmv_f_s_f32m1_f32(
                                    __riscv_vfredmax_vs_f32m2_f32m1(x, seed, lanes));
                            };
                            const auto sum_lane = [&](vfloat32m2_t x) -> float {
                                const vfloat32m1_t seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                                return __riscv_vfmv_f_s_f32m1_f32(
                                    __riscv_vfredusum_vs_f32m2_f32m1(x, seed, lanes));
                            };
                            bool any_valid = false;
                            const auto finalize = [&](int qi, int h, vfloat32m2_t scores)
                                    -> vfloat32m2_t {
                                const float tile_max = max_lane(scores);
                                if (tile_max == -__builtin_inff()) {
                                    // All 64 keys masked for this query/head.
                                    alpha[qi][h] = 1.0f;
                                    return __riscv_vfmv_v_f_f32m2(0.0f, lanes);
                                }
                                any_valid = true;
                                const float old_m = s.m[t + qi][h];
                                const float new_m = tile_max > old_m ? tile_max : old_m;
                                const float scale = (s.l[t + qi][h] != 0.0f && new_m > old_m) ?
                                    exp_neg(old_m - new_m) : 1.0f;
                                alpha[qi][h] = scale;
                                s.m[t + qi][h] = new_m;
                                const vfloat32m2_t weights = exp_neg_vector(
                                    __riscv_vfsub_vf_f32m2(scores, new_m, lanes));
                                s.l[t + qi][h] = s.l[t + qi][h] * scale + sum_lane(weights);
                                return weights;
                            };
                            p00 = finalize(0, 0, s00);
                            p01 = finalize(0, 1, s01);
                            p02 = finalize(0, 2, s02);
                            if (has_second) {
                                p10 = finalize(1, 0, s10);
                                p11 = finalize(1, 1, s11);
                                p12 = finalize(1, 2, s12);
                            }
                            if (has_third) {
                                __riscv_vse32_v_f32m2(deferred_p[0][0], finalize(2, 0, s20), lanes);
                                __riscv_vse32_v_f32m2(deferred_p[0][1], finalize(2, 1, s21), lanes);
                                __riscv_vse32_v_f32m2(deferred_p[0][2], finalize(2, 2, s22), lanes);
                            }
                            if (has_fourth) {
                                __riscv_vse32_v_f32m2(deferred_p[1][0], finalize(3, 0, s30), lanes);
                                __riscv_vse32_v_f32m2(deferred_p[1][1], finalize(3, 1, s31), lanes);
                                __riscv_vse32_v_f32m2(deferred_p[1][2], finalize(3, 2, s32), lanes);
                            }
                            if (!any_valid) continue;
                        }

                        // Reuse the same two-query PV kernel twice. First pair
                        // keeps P in registers; second reloads its staged P.
                        // Never have twelve P vectors and twelve U vectors live.
                        for (int pair = 0; pair < nq; pair += 2) {
                            const bool has_second = pair + 1 < nq;
                            if (pair != 0) {
                                const size_t lanes = static_cast<size_t>(len);
                                p00 = __riscv_vle32_v_f32m2(deferred_p[0][0], lanes);
                                p01 = __riscv_vle32_v_f32m2(deferred_p[0][1], lanes);
                                p02 = __riscv_vle32_v_f32m2(deferred_p[0][2], lanes);
                                if (has_second) {
                                    p10 = __riscv_vle32_v_f32m2(deferred_p[1][0], lanes);
                                    p11 = __riscv_vle32_v_f32m2(deferred_p[1][1], lanes);
                                    p12 = __riscv_vle32_v_f32m2(deferred_p[1][2], lanes);
                                }
                            }
                            // The PV accumulator operates on at most two
                            // queries (six heads) regardless of QK group size.
                            const auto accumulate_half = [&](int dim_offset) {
                                vfloat32m2_t u00 = __riscv_vle32_v_f32m2(s.accum[t + pair][0] + dim_offset, VL);
                                vfloat32m2_t u01 = __riscv_vle32_v_f32m2(s.accum[t + pair][1] + dim_offset, VL);
                                vfloat32m2_t u02 = __riscv_vle32_v_f32m2(s.accum[t + pair][2] + dim_offset, VL);
                                vfloat32m2_t u10 = has_second ? __riscv_vle32_v_f32m2(s.accum[t + pair + 1][0] + dim_offset, VL) : zero;
                                vfloat32m2_t u11 = has_second ? __riscv_vle32_v_f32m2(s.accum[t + pair + 1][1] + dim_offset, VL) : zero;
                                vfloat32m2_t u12 = has_second ? __riscv_vle32_v_f32m2(s.accum[t + pair + 1][2] + dim_offset, VL) : zero;
                                u00 = __riscv_vfmul_vf_f32m2(u00, alpha[pair][0], VL);
                                u01 = __riscv_vfmul_vf_f32m2(u01, alpha[pair][1], VL);
                                u02 = __riscv_vfmul_vf_f32m2(u02, alpha[pair][2], VL);
                                if (has_second) {
                                    u10 = __riscv_vfmul_vf_f32m2(u10, alpha[pair + 1][0], VL);
                                    u11 = __riscv_vfmul_vf_f32m2(u11, alpha[pair + 1][1], VL);
                                    u12 = __riscv_vfmul_vf_f32m2(u12, alpha[pair + 1][2], VL);
                                }
                                for (int j = 0; j < len; ++j) {
                                    const int64_t kv = kb + j;
                                    const uint8_t * vp = static_cast<const uint8_t *>(vc->data) +
                                        (v_trans ? kv * vc->nb[0] + kv_head * vc->nb[1] :
                                                   kv * vc->nb[2] + kv_head * vc->nb[1]);
                                    const vfloat32m2_t v = f16_load64(
                                        vp + dim_offset * v_step, v_step);
                                    // vrgather broadcasts P[j] across 64 output dimensions;
                                    // vector-vector FMA consumes it without extracting a scalar.
                                    u00 = __riscv_vfmacc_vv_f32m2(u00,
                                        __riscv_vrgather_vx_f32m2(p00, j, VL), v, VL);
                                    u01 = __riscv_vfmacc_vv_f32m2(u01,
                                        __riscv_vrgather_vx_f32m2(p01, j, VL), v, VL);
                                    u02 = __riscv_vfmacc_vv_f32m2(u02,
                                        __riscv_vrgather_vx_f32m2(p02, j, VL), v, VL);
                                    if (has_second) {
                                        u10 = __riscv_vfmacc_vv_f32m2(u10,
                                            __riscv_vrgather_vx_f32m2(p10, j, VL), v, VL);
                                        u11 = __riscv_vfmacc_vv_f32m2(u11,
                                            __riscv_vrgather_vx_f32m2(p11, j, VL), v, VL);
                                        u12 = __riscv_vfmacc_vv_f32m2(u12,
                                            __riscv_vrgather_vx_f32m2(p12, j, VL), v, VL);
                                    }
                                }
                                __riscv_vse32_v_f32m2(s.accum[t + pair][0] + dim_offset, u00, VL);
                                __riscv_vse32_v_f32m2(s.accum[t + pair][1] + dim_offset, u01, VL);
                                __riscv_vse32_v_f32m2(s.accum[t + pair][2] + dim_offset, u02, VL);
                                if (has_second) {
                                    __riscv_vse32_v_f32m2(s.accum[t + pair + 1][0] + dim_offset, u10, VL);
                                    __riscv_vse32_v_f32m2(s.accum[t + pair + 1][1] + dim_offset, u11, VL);
                                    __riscv_vse32_v_f32m2(s.accum[t + pair + 1][2] + dim_offset, u12, VL);
                                }
                            };
                            accumulate_half(0);
                            accumulate_half(static_cast<int>(VL));
                        }
                    }

                }
            }

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
        }
        return;
}
