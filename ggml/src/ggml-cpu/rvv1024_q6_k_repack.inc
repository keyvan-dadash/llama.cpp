// Included inside ggml::cpu::repack, after repack/gemv/gemm declarations.
// Experimental Q6_K output-parallel layout for VLEN=1024 and Zvfh.
// No extra bytes, no requantization, no change to the GGUF on disk.
#if defined(__riscv_v) && defined(__riscv_zvfh)

static constexpr int q6k_cols = 64;

struct q6k_group_64 {
    int8_t  scale[q6k_cols];
    uint8_t packed[4][3][q6k_cols]; // 4 groups of four input weights
};
struct q6k_block_64 {
    ggml_half d[q6k_cols];
    q6k_group_64 group[QK_K / 16];
};
static_assert(QK_K == 256, "Q6_K superblock size");
static_assert(sizeof(q6k_group_64) == 13 * q6k_cols, "Q6_K subgroup padding");
static_assert(sizeof(q6k_block_64) == q6k_cols * sizeof(block_q6_K), "Q6_K repack must preserve size");
static_assert(sizeof(block_q8_Kx4) == 4 * sizeof(block_q8_K), "Q8_K interleave size");

// Return the signed six-bit two's-complement encoding, not an int8 value.
static inline uint8_t q6k_bits(const block_q6_K & b, int k) {
    const int half = k / 128;
    const int quarter = (k % 128) / 32;
    const int lane = k % 32;
    const unsigned lo = (b.ql[half * 64 + (quarter & 1) * 32 + lane] >> (quarter >= 2 ? 4 : 0)) & 15u;
    const unsigned hi = (b.qh[half * 32 + lane] >> (2 * quarter)) & 3u;
    return (uint8_t) ((lo | (hi << 4)) ^ 32u);
}

// 4 x 6 bits -> 3 bytes. The top six bits of each byte contain one
// signed weight; the fourth weight occupies the three low two-bit fields.
static void q6k_pack_block(q6k_block_64 & out, const block_q6_K * src, int64_t stride) {
    for (int r = 0; r < q6k_cols; ++r) {
        const block_q6_K & b = src[r * stride];
        out.d[r] = b.d;
        for (int g = 0; g < QK_K / 16; ++g) {
            auto & group = out.group[g];
            group.scale[r] = b.scales[g];
            for (int j = 0; j < 4; ++j) {
                const int k = g * 16 + j * 4;
                const uint8_t q3 = q6k_bits(b, k + 3);
                for (int t = 0; t < 3; ++t) {
                    group.packed[j][t][r] = (uint8_t) ((q6k_bits(b, k + t) << 2) | ((q3 >> (2 * t)) & 3u));
                }
            }
        }
    }
}

// Q8_K GEMV and existing 4x1 Q8_K GEMM activation layouts.
static inline int8_t q6k_activation(const block_q8_K & a, int k, int /* row */) { return a.qs[k]; }
static inline int8_t q6k_activation(const block_q8_Kx4 & a, int k, int row) { return a.qs[4*k + row]; }
static inline float q6k_delta(const block_q8_K & a, int /* row */) { return a.d; }
static inline float q6k_delta(const block_q8_Kx4 & a, int row) { return a.d[row]; }

// 64 independent outputs: e8,mf2 -> e16,m1 -> e32,m2.
// Four partial i16 sums each contain only four products per scale group.
// Even with q8=-128 and q6=-32, |partial| <= 16384. The signed
// scale is applied using widening i16*i16 -> i32 MAC, not in i16.
// A whole superblock is bounded by 256*32*128*128 = 2^27.
template <typename Activation>
static inline void q6k_tile(int nb, float * dst, const q6k_block_64 * w,
                            const Activation * a, int row) {
    constexpr size_t vl = q6k_cols;
    vfloat32m2_t result = __riscv_vfmv_v_f_f32m2(0.0f, vl);
    for (int b = 0; b < nb; ++b) {
        vint32m2_t super = __riscv_vmv_v_x_i32m2(0, vl);
        for (int g = 0; g < QK_K / 16; ++g) {
            const auto & group = w[b].group[g];
            vint16m1_t s0 = __riscv_vmv_v_x_i16m1(0, vl);
            vint16m1_t s1 = s0;
            vint16m1_t s2 = s0;
            vint16m1_t s3 = s0;
            for (int j = 0; j < 4; ++j) {
                const int k = g * 16 + j * 4;
                const vuint8mf2_t p0 = __riscv_vle8_v_u8mf2(group.packed[j][0], vl);
                const vuint8mf2_t p1 = __riscv_vle8_v_u8mf2(group.packed[j][1], vl);
                const vuint8mf2_t p2 = __riscv_vle8_v_u8mf2(group.packed[j][2], vl);
                // Signed top-six-bit fields: one arithmetic shift each.
                const vint8mf2_t q0 = __riscv_vsra_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(p0), 2, vl);
                const vint8mf2_t q1 = __riscv_vsra_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(p1), 2, vl);
                const vint8mf2_t q2 = __riscv_vsra_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(p2), 2, vl);
                s0 = __riscv_vwmacc_vx_i16m1(s0, q6k_activation(a[b], k + 0, row), q0, vl);
                s1 = __riscv_vwmacc_vx_i16m1(s1, q6k_activation(a[b], k + 1, row), q1, vl);
                s2 = __riscv_vwmacc_vx_i16m1(s2, q6k_activation(a[b], k + 2, row), q2, vl);
                // Reconstruct q3 directly into bits 7:2; shifting p2 by 6
                // discards its top six bits, so it does not need a mask.
                const vuint8mf2_t lo = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(p0, 3, vl), 2, vl);
                const vuint8mf2_t mid = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(p1, 3, vl), 4, vl);
                const vuint8mf2_t hi = __riscv_vsll_vx_u8mf2(p2, 6, vl);
                const vuint8mf2_t p3 = __riscv_vor_vv_u8mf2(__riscv_vor_vv_u8mf2(lo, mid, vl), hi, vl);
                const vint8mf2_t q3 = __riscv_vsra_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(p3), 2, vl);
                s3 = __riscv_vwmacc_vx_i16m1(s3, q6k_activation(a[b], k + 3, row), q3, vl);
            }
            const vint16m1_t scale = __riscv_vsext_vf2_i16m1(__riscv_vle8_v_i8mf2(group.scale, vl), vl);
            super = __riscv_vwmacc_vv_i32m2(super, scale, s0, vl);
            super = __riscv_vwmacc_vv_i32m2(super, scale, s1, vl);
            super = __riscv_vwmacc_vv_i32m2(super, scale, s2, vl);
            super = __riscv_vwmacc_vv_i32m2(super, scale, s3, vl);
        }
        // Load FP16 bits without aliasing through a _Float16 pointer.
        const vuint16m1_t d_bits = __riscv_vle16_v_u16m1(w[b].d, vl);
        const vfloat32m2_t wd = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vreinterpret_v_u16m1_f16m1(d_bits), vl);
        const vfloat32m2_t d = __riscv_vfmul_vf_f32m2(wd, q6k_delta(a[b], row), vl);
        result = __riscv_vfmacc_vv_f32m2(result, __riscv_vfcvt_f_x_v_f32m2(super, vl), d, vl);
    }
    __riscv_vse32_v_f32m2(dst, result, vl);
}

// Use existing repack buffer, scheduler and Q8_K quantizer.
template <> int repack<block_q6_K, 1, 64>(ggml_tensor * t, const void * data, size_t data_size) {
    GGML_ASSERT(__riscv_vlenb() >= 128);
    GGML_ASSERT(t->ne[0] % QK_K == 0 && t->ne[1] % q6k_cols == 0);
    GGML_ASSERT(t->ne[2] == 1 && t->ne[3] == 1);
    const int64_t nb = t->ne[0] / QK_K;
    const int64_t tiles = t->ne[1] / q6k_cols;
    GGML_ASSERT(nb > 0 && tiles > 0);
    GGML_ASSERT(data_size == (size_t) tiles * nb * sizeof(q6k_block_64));
    // As with other CPU_REPACK formats, source data and destination differ.
    GGML_ASSERT(data != t->data);
    const auto * src = static_cast<const block_q6_K *>(data);
    auto * dst = static_cast<q6k_block_64 *>(t->data);
    for (int64_t tile = 0; tile < tiles; ++tile) {
        for (int64_t b = 0; b < nb; ++b) {
            q6k_pack_block(dst[tile*nb+b], src + tile*q6k_cols*nb+b, nb);
        }
    }
    std::fprintf(stderr, "[RVV1024] Q6_K 64x1 repack: %s, %lld x %lld, %zu bytes\n",
                 t->name, (long long) t->ne[0], (long long) t->ne[1], data_size);
    return 0;
}

template <> void gemv<block_q6_K, 1, 64, GGML_TYPE_Q8_K>(int n, float * s, size_t /* bs */, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(__riscv_vlenb() >= 128);
    GGML_ASSERT(n % QK_K == 0 && nc % q6k_cols == 0 && nr == 1);
    const int nb = n / QK_K;
    const auto * w = static_cast<const q6k_block_64 *>(vx);
    const auto * a = static_cast<const block_q8_K *>(vy);
    for (int x = 0; x < nc / q6k_cols; ++x) {
        q6k_tile(nb, s + x*q6k_cols, w + x*nb, a, 0);
    }
}

// Correctness-first multi-token fallback for the existing 4x1 Q8_K layout.
// Process one tile for four rows before advancing; no separate fast GEMM yet.
// The upstream scheduler handles 1-3 leftover activation rows via GEMV.
template <> void gemm<block_q6_K, 1, 64, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(__riscv_vlenb() >= 128);
    GGML_ASSERT(n % QK_K == 0 && nc % q6k_cols == 0 && nr % 4 == 0);
    const int nb = n / QK_K;
    const auto * w = static_cast<const q6k_block_64 *>(vx);
    const auto * a = static_cast<const block_q8_Kx4 *>(vy);
    for (int y = 0; y < nr / 4; ++y) {
        for (int x = 0; x < nc / q6k_cols; ++x) {
            for (int r = 0; r < 4; ++r) {
                q6k_tile(nb, s + (4*y+r)*bs+x*q6k_cols, w + x*nb, a + y*nb, r);
            }
        }
    }
}
#endif // __riscv_v && __riscv_zvfh
