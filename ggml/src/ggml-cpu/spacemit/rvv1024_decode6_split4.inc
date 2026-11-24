// Decode6 split-KV, one token, four workers:
//   Q projection: 4 workers x 3 heads, one packed weight shard each.
//   Attention: 4 workers x 6 heads; worker 2*kv_head + half owns half
//              of that KV head's history.  Outputs (U[128], m, l).
//   Merge + W_O: 4 workers x 3 heads; softmax-merge the two KV halves,
//                normalize, quantize Q8, then run one packed W_O shard.
//   Existing FP32 partial-Y reduction: 4 W_O output slabs -> final Y.
// Only standard RVV1.0/Zvfh; K/V cache update remains the normal GGML path.
#if defined(__riscv_v) && defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
namespace {
constexpr size_t D6_VL = 64;
constexpr size_t D6_WO_VL = 128;
constexpr int D6_BK_MAX = 128;
constexpr int D6_H = 6;
constexpr float D6_ATTN_SCALE = 0.08838834764831845f;
static_assert(D6_H == N_Q_HEADS / N_KV_HEADS && Q_HEADS_TILE == 3 &&
              N_KV_HEADS == 2 && DECODE6_WORKERS == 4 && DECODE6_STATE_STRIDE == 144,
              "decode6 split expects 12Q/2KV, 4 cores and 128D");

// Same polynomial as the prefill fused attention kernel; x must be <= 0.
static inline float d6_exp_neg(float x) {
    if (x < -85.0f) return 0.0f;
    if (x >= 0.0f) return 1.0f;
    const float z = x * 1.4426950408889634f;
    const int n = static_cast<int>(z < 0.0f ? z - 0.5f : z + 0.5f);
    const float r = x - static_cast<float>(n) * 0.6931471805599453f;
    float p = 1.0f / 5040.0f;
    p = p*r + 1.0f/720.0f; p = p*r + 1.0f/120.0f;
    p = p*r + 1.0f/24.0f;  p = p*r + 1.0f/6.0f;
    p = p*r + 0.5f; p = p*r + 1.0f; p = p*r + 1.0f;
    const uint32_t ebits = static_cast<uint32_t>(n + 127) << 23;
    float pow2;
    __builtin_memcpy(&pow2, &ebits, sizeof(pow2));
    return p * pow2;
}
static inline float d6_max(vfloat32m2_t v, size_t lanes) {
    const auto seed = __riscv_vfmv_v_f_f32m1(-__builtin_inff(), 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m2_f32m1(v, seed, lanes));
}
static inline float d6_sum(vfloat32m2_t v, size_t lanes) {
    const auto seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(v, seed, lanes));
}
static inline vfloat32m2_t d6_exp_neg_vec(vfloat32m2_t x, size_t lanes) {
    const vbool16_t keep = __riscv_vmfgt_vf_f32m2_b16(x, -85.0f, lanes);
    const auto safe = __riscv_vfmax_vf_f32m2(x, -85.0f, lanes);
    const auto z = __riscv_vfmul_vf_f32m2(safe, 1.4426950408889634f, lanes);
    const auto n = __riscv_vfcvt_rtz_x_f_v_i32m2(
        __riscv_vfsub_vf_f32m2(z, 0.5f, lanes), lanes);
    const auto nf = __riscv_vfcvt_f_x_v_f32m2(n, lanes);
    const auto r = __riscv_vfsub_vv_f32m2(safe,
        __riscv_vfmul_vf_f32m2(nf, 0.6931471805599453f, lanes), lanes);
    const auto horner = [&](vfloat32m2_t p, float c) -> vfloat32m2_t {
        return __riscv_vfadd_vf_f32m2(__riscv_vfmul_vv_f32m2(p, r, lanes), c, lanes);
    };
    auto p = __riscv_vfmv_v_f_f32m2(1.0f/5040.0f, lanes);
    p = horner(p,1.0f/720.0f);p = horner(p,1.0f/120.0f);
    p = horner(p,1.0f/24.0f);p = horner(p,1.0f/6.0f);
    p = horner(p,0.5f);p = horner(p,1.0f);p = horner(p,1.0f);
    const auto powbits = __riscv_vsll_vx_i32m2(
        __riscv_vadd_vx_i32m2(n, 127, lanes), 23, lanes);
    const auto pow2 = __riscv_vreinterpret_v_i32m2_f32m2(powbits);
    return __riscv_vmerge_vvm_f32m2(__riscv_vfmv_v_f_f32m2(0.0f, lanes),
        __riscv_vfmul_vv_f32m2(p, pow2, lanes), keep, lanes);
}
static inline vint8mf2_t d6_quant32(vfloat32m2_t src, int upper, float & scale) {
    constexpr size_t N=Q4_BLOCK_SIZE;
    const auto part = upper ? __riscv_vslidedown_vx_f32m2(src, N, N) : src;
    const auto seed = __riscv_vfmv_v_f_f32m1(0.0f,1);
    const float maxabs = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(__riscv_vfabs_v_f32m2(part,N),seed,N));
    const float d = maxabs / 127.0f;
    // Match prefill's Q8_0 scale rounding.
    const auto hd = __riscv_vfncvt_f_f_w_f16m1(__riscv_vfmv_v_f_f32m2(d,1),1);
    scale = __riscv_vfmv_f_s_f32m2_f32(__riscv_vfwcvt_f_f_v_f32m2(hd,1));
    const float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    auto qi = __riscv_vfcvt_x_f_v_i32m2(__riscv_vfmul_vf_f32m2(part,inverse,N),N);
    qi = __riscv_vmax_vx_i32m2(qi,-127,N);
    qi = __riscv_vmin_vx_i32m2(qi,127,N);
    return __riscv_vncvt_x_x_w_i8mf2(__riscv_vncvt_x_x_w_i16m1(qi,N),N);
}

static void d6_project_one_group(ggml_tensor * out, int group) {
    const auto * prepared = out->src[0];
    const auto * wq = out->src[1];
    const auto * bias = out->src[2];
    GGML_ASSERT(out && out->type == GGML_TYPE_F16 && ggml_is_contiguous(out) &&
                out->ne[0] == HEAD_DIM && out->ne[1] == N_Q_HEADS &&
                prepared && prepared->ne[1] == 1 && prepared->data &&
                wq && wq->type == GGML_TYPE_I8 && wq->ne[0] == static_cast<int64_t>(Q_PACKED_BYTES) && wq->data);
    const auto * x = reinterpret_cast<const prefill_prepared_token *>(prepared->data);
    const auto c = __riscv_vle32_v_f32m2(x->c,D6_VL);
    const auto s = __riscv_vle32_v_f32m2(x->s,D6_VL);
    const auto zero = __riscv_vfmv_v_f_f32m2(0.0f,D6_VL);
    const auto * wg = static_cast<const uint8_t *>(wq->data) + group*Q_GROUP_BYTES;
    for (int h = 0; h < Q_HEADS_TILE; ++h) {
        const int global_h = group*Q_HEADS_TILE + h;
        const float * hb = bias ? static_cast<const float *>(bias->data) + global_h*HEAD_DIM : nullptr;
        auto lo = hb ? __riscv_vle32_v_f32m2(hb,D6_VL) : zero;
        auto hi = hb ? __riscv_vle32_v_f32m2(hb+D6_VL,D6_VL) : zero;
        for (int b = 0; b < Q_BLOCKS_PER_ROW; ++b) {
            const auto * block = wg + static_cast<size_t>(b)*Q_BLOCK_BYTES;
            auto isumlo = __riscv_vmv_v_x_i16m1(0,D6_VL);
            auto isumhi = __riscv_vmv_v_x_i16m1(0,D6_VL);
            for (int k = 0; k < Q4_BLOCK_SIZE; ++k) {
                const auto * col = block + k*Q_COL_BYTES + h*(HEAD_DIM/2);
                const auto packed = __riscv_vle8_v_u8mf2(col,D6_VL);
                const auto wlo = __riscv_vreinterpret_v_u8mf2_i8mf2(
                    __riscv_vsub_vx_u8mf2(__riscv_vand_vx_u8mf2(packed,15,D6_VL),8,D6_VL));
                const auto whi = __riscv_vreinterpret_v_u8mf2_i8mf2(
                    __riscv_vsub_vx_u8mf2(__riscv_vsrl_vx_u8mf2(packed,4,D6_VL),8,D6_VL));
                isumlo = __riscv_vwmacc_vx_i16m1(isumlo,x->x8[b].qs[k],wlo,D6_VL);
                isumhi = __riscv_vwmacc_vx_i16m1(isumhi,x->x8[b].qs[k],whi,D6_VL);
            }
            const float * scales = reinterpret_cast<const float *>(block+Q4_BLOCK_SIZE*Q_COL_BYTES)+h*HEAD_DIM;
            const float sx=x->x8[b].scale;
            lo=__riscv_vfadd_vv_f32m2(lo,__riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(__riscv_vfwcvt_f_x_v_f32m2(isumlo,D6_VL),
                __riscv_vle32_v_f32m2(scales,D6_VL),D6_VL),sx,D6_VL),D6_VL);
            hi=__riscv_vfadd_vv_f32m2(hi,__riscv_vfmul_vf_f32m2(
                __riscv_vfmul_vv_f32m2(__riscv_vfwcvt_f_x_v_f32m2(isumhi,D6_VL),
                __riscv_vle32_v_f32m2(scales+D6_VL,D6_VL),D6_VL),sx,D6_VL),D6_VL);
        }
        const auto rotlo = __riscv_vfsub_vv_f32m2(
            __riscv_vfmul_vv_f32m2(lo,c,D6_VL),
            __riscv_vfmul_vv_f32m2(hi,s,D6_VL),D6_VL);
        const auto rothi = __riscv_vfadd_vv_f32m2(
            __riscv_vfmul_vv_f32m2(hi,c,D6_VL),
            __riscv_vfmul_vv_f32m2(lo,s,D6_VL),D6_VL);
        auto * dest = reinterpret_cast<_Float16 *>(static_cast<uint8_t *>(out->data) + global_h*out->nb[1]);
        __riscv_vse16_v_f16m1(dest,__riscv_vfncvt_f_f_w_f16m1(rotlo,D6_VL),D6_VL);
        __riscv_vse16_v_f16m1(dest+D6_VL,__riscv_vfncvt_f_f_w_f16m1(rothi,D6_VL),D6_VL);
    }
}


// BK128 uses m4 accumulators (128 historical positions per vector).
// Six heads x f32m4 = 24 registers, plus f16m2 K = 2 registers.
// QK scores are staged before softmax to avoid retaining six m4 score vectors
// alongside polynomial/mask temporaries. This is an explicit memory boundary.
static inline float d6_max128(vfloat32m4_t v, size_t lanes) {
    const auto seed = __riscv_vfmv_v_f_f32m1(-__builtin_inff(), 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(v, seed, lanes));
}
static inline float d6_sum128(vfloat32m4_t v, size_t lanes) {
    const auto seed = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(v, seed, lanes));
}
static inline vfloat32m4_t d6_exp_neg_vec128(vfloat32m4_t x, size_t lanes) {
    const vbool8_t keep = __riscv_vmfgt_vf_f32m4_b8(x, -85.0f, lanes);
    const auto safe = __riscv_vfmax_vf_f32m4(x, -85.0f, lanes);
    const auto z = __riscv_vfmul_vf_f32m4(safe, 1.4426950408889634f, lanes);
    const auto n = __riscv_vfcvt_rtz_x_f_v_i32m4(
        __riscv_vfsub_vf_f32m4(z, 0.5f, lanes), lanes);
    const auto nf = __riscv_vfcvt_f_x_v_f32m4(n, lanes);
    const auto r = __riscv_vfsub_vv_f32m4(safe,
        __riscv_vfmul_vf_f32m4(nf, 0.6931471805599453f, lanes), lanes);
    const auto horner = [&](vfloat32m4_t p, float c) -> vfloat32m4_t {
        return __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, lanes), c, lanes);
    };
    auto p = __riscv_vfmv_v_f_f32m4(1.0f/5040.0f, lanes);
    p = horner(p,1.0f/720.0f);p = horner(p,1.0f/120.0f);
    p = horner(p,1.0f/24.0f);p = horner(p,1.0f/6.0f);
    p = horner(p,0.5f);p = horner(p,1.0f);p = horner(p,1.0f);
    const auto powbits = __riscv_vsll_vx_i32m4(
        __riscv_vadd_vx_i32m4(n, 127, lanes), 23, lanes);
    const auto pow2 = __riscv_vreinterpret_v_i32m4_f32m4(powbits);
    return __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(0.0f, lanes),
        __riscv_vfmul_vv_f32m4(p, pow2, lanes), keep, lanes);
}

// Keep PV64 as the control: half the 128 output dimensions at a time,
// six f32m2 accumulator vectors and one converted V vector.
static void d6_pv64(const ggml_tensor * vc, int kv_head, bool v_trans,
                    ptrdiff_t v_step, int64_t kb, size_t lanes,
                    float u[D6_H][HEAD_DIM], const float p[D6_H][128],
                    const float alpha[D6_H]) {
    const auto update_half=[&](int off){
        auto u0=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(u[0]+off,D6_VL),alpha[0],D6_VL);
        auto u1=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(u[1]+off,D6_VL),alpha[1],D6_VL);
        auto u2=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(u[2]+off,D6_VL),alpha[2],D6_VL);
        auto u3=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(u[3]+off,D6_VL),alpha[3],D6_VL);
        auto u4=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(u[4]+off,D6_VL),alpha[4],D6_VL);
        auto u5=__riscv_vfmul_vf_f32m2(__riscv_vle32_v_f32m2(u[5]+off,D6_VL),alpha[5],D6_VL);
        for(size_t j=0;j<lanes;j++){
            if(p[0][j]==0.0f && p[1][j]==0.0f && p[2][j]==0.0f &&
               p[3][j]==0.0f && p[4][j]==0.0f && p[5][j]==0.0f) continue;
            const int64_t pos=kb+static_cast<int64_t>(j);
            const uint8_t * vp=static_cast<const uint8_t *>(vc->data)+
              (v_trans?pos*vc->nb[0]+kv_head*vc->nb[1]:pos*vc->nb[2]+kv_head*vc->nb[1]);
            const uint8_t * vv=vp+off*v_step;
            const auto vb=v_step==2?
                __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(vv),D6_VL):
                __riscv_vlse16_v_u16m1(reinterpret_cast<const uint16_t *>(vv),v_step,D6_VL);
            const auto v=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vreinterpret_v_u16m1_f16m1(vb),D6_VL);
            u0=__riscv_vfmacc_vf_f32m2(u0,p[0][j],v,D6_VL);
            u1=__riscv_vfmacc_vf_f32m2(u1,p[1][j],v,D6_VL);
            u2=__riscv_vfmacc_vf_f32m2(u2,p[2][j],v,D6_VL);
            u3=__riscv_vfmacc_vf_f32m2(u3,p[3][j],v,D6_VL);
            u4=__riscv_vfmacc_vf_f32m2(u4,p[4][j],v,D6_VL);
            u5=__riscv_vfmacc_vf_f32m2(u5,p[5][j],v,D6_VL);
        }
        __riscv_vse32_v_f32m2(u[0]+off,u0,D6_VL);
        __riscv_vse32_v_f32m2(u[1]+off,u1,D6_VL);
        __riscv_vse32_v_f32m2(u[2]+off,u2,D6_VL);
        __riscv_vse32_v_f32m2(u[3]+off,u3,D6_VL);
        __riscv_vse32_v_f32m2(u[4]+off,u4,D6_VL);
        __riscv_vse32_v_f32m2(u[5]+off,u5,D6_VL);
    };
    update_half(0);update_half(64);
}

// Optional independent experiment: keep all 128 V output dimensions in one
// f32m4 U vector per head (24 registers), plus V f32m4 and f16m2 temporaries.
// Tight register footprint; inspect GCC assembly for spilling on the A100.
static void d6_pv128(const ggml_tensor * vc, int kv_head, bool v_trans,
                     ptrdiff_t v_step, int64_t kb, size_t lanes,
                     float u[D6_H][HEAD_DIM], const float p[D6_H][128],
                     const float alpha[D6_H]) {
    constexpr size_t VL128=128;
    auto u0=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(u[0],VL128),alpha[0],VL128);
    auto u1=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(u[1],VL128),alpha[1],VL128);
    auto u2=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(u[2],VL128),alpha[2],VL128);
    auto u3=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(u[3],VL128),alpha[3],VL128);
    auto u4=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(u[4],VL128),alpha[4],VL128);
    auto u5=__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(u[5],VL128),alpha[5],VL128);
    for(size_t j=0;j<lanes;j++){
        if(p[0][j]==0.0f && p[1][j]==0.0f && p[2][j]==0.0f &&
           p[3][j]==0.0f && p[4][j]==0.0f && p[5][j]==0.0f) continue;
        const int64_t pos=kb+static_cast<int64_t>(j);
        const uint8_t * vp=static_cast<const uint8_t *>(vc->data)+
            (v_trans?pos*vc->nb[0]+kv_head*vc->nb[1]:pos*vc->nb[2]+kv_head*vc->nb[1]);
        const auto vb=v_step==2?
            __riscv_vle16_v_u16m2(reinterpret_cast<const uint16_t *>(vp),VL128):
            __riscv_vlse16_v_u16m2(reinterpret_cast<const uint16_t *>(vp),v_step,VL128);
        const auto v=__riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(vb),VL128);
        u0=__riscv_vfmacc_vf_f32m4(u0,p[0][j],v,VL128);
        u1=__riscv_vfmacc_vf_f32m4(u1,p[1][j],v,VL128);
        u2=__riscv_vfmacc_vf_f32m4(u2,p[2][j],v,VL128);
        u3=__riscv_vfmacc_vf_f32m4(u3,p[3][j],v,VL128);
        u4=__riscv_vfmacc_vf_f32m4(u4,p[4][j],v,VL128);
        u5=__riscv_vfmacc_vf_f32m4(u5,p[5][j],v,VL128);
    }
    __riscv_vse32_v_f32m4(u[0],u0,VL128);
    __riscv_vse32_v_f32m4(u[1],u1,VL128);
    __riscv_vse32_v_f32m4(u[2],u2,VL128);
    __riscv_vse32_v_f32m4(u[3],u3,VL128);
    __riscv_vse32_v_f32m4(u[4],u4,VL128);
    __riscv_vse32_v_f32m4(u[5],u5,VL128);
}

static void d6_attention_split(ggml_tensor * out, int worker) {
    const auto * q=out->src[0];
    const auto * kc=out->src[1];
    const auto * vc=out->src[2];
    const auto * mask=out->src[3];
    const auto * cols=out->src[5];
    const int kv_head=worker/2;
    const int half=worker%2;
    GGML_ASSERT(out && out->type==GGML_TYPE_F32 && ggml_is_contiguous(out) &&
        out->ne[0]==DECODE6_STATE_STRIDE && out->ne[1]==D6_H &&
        out->ne[2]==DECODE6_WORKERS &&
        q && q->type==GGML_TYPE_F16 && ggml_is_contiguous(q) &&
        q->ne[0]==HEAD_DIM && q->ne[1]==N_Q_HEADS &&
        kc && kc->type==GGML_TYPE_F16 && kc->ne[0]==HEAD_DIM && kc->ne[1]==N_KV_HEADS &&
        vc && vc->type==GGML_TYPE_F16 && mask && cols &&
        cols->type==GGML_TYPE_F16 && (cols->ne[0]==64 || cols->ne[0]==128) &&
        cols->ne[1]==HEAD_DIM && cols->ne[2]==N_KV_HEADS &&
        cols->nb[0]==2 && cols->nb[1]==2*cols->ne[0] &&
        mask->ne[1]>=1 && (mask->type==GGML_TYPE_F16 || mask->type==GGML_TYPE_F32));
    const int bk=static_cast<int>(cols->ne[0]);
    const int64_t n_kv=kc->ne[2];
    const int64_t blocks=(n_kv+bk-1)/bk;
    const int64_t split=((blocks+1)/2)*bk;
    const int64_t begin=half==0?0:split;
    const int64_t end=half==0?(split<n_kv?split:n_kv):n_kv;
    const bool v_trans=vc->nb[1]>vc->nb[2];
    const ptrdiff_t v_step=static_cast<ptrdiff_t>(v_trans?vc->nb[2]:vc->nb[0]);
    GGML_ASSERT(n_kv>0 && ((v_trans && vc->ne[0]==n_kv && vc->ne[1]==N_KV_HEADS && vc->ne[2]==HEAD_DIM) ||
                (!v_trans && vc->ne[0]==HEAD_DIM && vc->ne[1]==N_KV_HEADS && vc->ne[2]==n_kv)) &&
                mask->ne[0]>=n_kv && cols->ne[3]>=blocks &&
                cols->nb[2]==HEAD_DIM*cols->nb[1] && cols->nb[3]==N_KV_HEADS*cols->nb[2] &&
                mask->nb[0]==(mask->type==GGML_TYPE_F16?2u:4u));
    const auto * qall=reinterpret_cast<const _Float16 *>(q->data);
    alignas(64) float u[D6_H][HEAD_DIM]={};
    alignas(64) float p[D6_H][D6_BK_MAX];
    alignas(64) float scores128[D6_H][D6_BK_MAX];
    float m[D6_H],l[D6_H],alpha[D6_H];
    for(int h=0;h<D6_H;h++){m[h]=-__builtin_inff();l[h]=0.0f;}
    const auto read_q=[&](int h,int d)->_Float16 {
        return qall[(kv_head*D6_H+h)*HEAD_DIM+d];
    };
    // Read once per attention op: PV128 can be compared independently of KCOL width.
    const char * pv_env=std::getenv("GGML_RVV1024_PV128");
    const bool pv128=pv_env && std::strcmp(pv_env,"1")==0;
    const bool decode_prof = ggml_rvvd_prof_active();
    uint64_t prof_qk=0, prof_store=0, prof_softmax=0, prof_pv=0;
    for(int64_t kb=begin;kb<end;kb+=bk){
        const size_t lanes=static_cast<size_t>(end-kb<bk?end-kb:bk);
        const uint8_t * kbase=static_cast<const uint8_t *>(cols->data)+
            (kb/bk)*cols->nb[3]+kv_head*cols->nb[2];
        const uint8_t * mp=static_cast<const uint8_t *>(mask->data)+kb*mask->nb[0];
        bool any_valid=false;
        const uint64_t prof_t0 = rvvd_read_ticks(decode_prof);
        uint64_t prof_tqk = 0, prof_tstore = 0;
        if(bk==64){
            // Original BK64 path, same six f32m2 score accumulators.
            auto s0=__riscv_vfmv_v_f_f32m2(0.0f,lanes);
            vfloat32m2_t s1=s0,s2=s0,s3=s0,s4=s0,s5=s0;
            for(int d=0;d<HEAD_DIM;d++){
                const auto kh=__riscv_vreinterpret_v_u16m1_f16m1(
                    __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(kbase+d*cols->nb[1]),lanes));
                s0=__riscv_vfwmacc_vf_f32m2(s0,read_q(0,d),kh,lanes);
                s1=__riscv_vfwmacc_vf_f32m2(s1,read_q(1,d),kh,lanes);
                s2=__riscv_vfwmacc_vf_f32m2(s2,read_q(2,d),kh,lanes);
                s3=__riscv_vfwmacc_vf_f32m2(s3,read_q(3,d),kh,lanes);
                s4=__riscv_vfwmacc_vf_f32m2(s4,read_q(4,d),kh,lanes);
                s5=__riscv_vfwmacc_vf_f32m2(s5,read_q(5,d),kh,lanes);
            }
            prof_tqk = rvvd_read_ticks(decode_prof);
            prof_tstore = prof_tqk;
            const auto mv=mask->type==GGML_TYPE_F32?
                __riscv_vle32_v_f32m2(reinterpret_cast<const float *>(mp),lanes):
                __riscv_vfwcvt_f_f_v_f32m2(__riscv_vreinterpret_v_u16m1_f16m1(
                    __riscv_vle16_v_u16m1(reinterpret_cast<const uint16_t *>(mp),lanes)),lanes);
            const auto finalize=[&](int h,vfloat32m2_t scores){
                scores=__riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(scores,D6_ATTN_SCALE,lanes),mv,lanes);
                const float mx=d6_max(scores,lanes);
                if(mx==-__builtin_inff()){
                    alpha[h]=1.0f;
                    for(size_t j=0;j<lanes;j++)p[h][j]=0.0f;
                    return;
                }
                any_valid=true;
                const float nm=mx>m[h]?mx:m[h];
                const float a=(l[h]>0.0f&&nm>m[h])?d6_exp_neg(m[h]-nm):1.0f;
                alpha[h]=a;m[h]=nm;
                const auto w=d6_exp_neg_vec(__riscv_vfsub_vf_f32m2(scores,nm,lanes),lanes);
                __riscv_vse32_v_f32m2(p[h],w,lanes);
                l[h]=l[h]*a+d6_sum(w,lanes);
            };
            finalize(0,s0);finalize(1,s1);finalize(2,s2);
            finalize(3,s3);finalize(4,s4);finalize(5,s5);
        } else {
            // Six m4 QK accumulators; no transpose in the hot loop.
            // Explicit score stores free the 24-register live set before softmax.
            {
                auto s0=__riscv_vfmv_v_f_f32m4(0.0f,lanes);
                vfloat32m4_t s1=s0,s2=s0,s3=s0,s4=s0,s5=s0;
                for(int d=0;d<HEAD_DIM;d++){
                    const auto kh=__riscv_vreinterpret_v_u16m2_f16m2(
                        __riscv_vle16_v_u16m2(reinterpret_cast<const uint16_t *>(kbase+d*cols->nb[1]),lanes));
                    s0=__riscv_vfwmacc_vf_f32m4(s0,read_q(0,d),kh,lanes);
                    s1=__riscv_vfwmacc_vf_f32m4(s1,read_q(1,d),kh,lanes);
                    s2=__riscv_vfwmacc_vf_f32m4(s2,read_q(2,d),kh,lanes);
                    s3=__riscv_vfwmacc_vf_f32m4(s3,read_q(3,d),kh,lanes);
                    s4=__riscv_vfwmacc_vf_f32m4(s4,read_q(4,d),kh,lanes);
                    s5=__riscv_vfwmacc_vf_f32m4(s5,read_q(5,d),kh,lanes);
                }
                prof_tqk = rvvd_read_ticks(decode_prof);
                __riscv_vse32_v_f32m4(scores128[0],s0,lanes);
                __riscv_vse32_v_f32m4(scores128[1],s1,lanes);
                __riscv_vse32_v_f32m4(scores128[2],s2,lanes);
                __riscv_vse32_v_f32m4(scores128[3],s3,lanes);
                __riscv_vse32_v_f32m4(scores128[4],s4,lanes);
                __riscv_vse32_v_f32m4(scores128[5],s5,lanes);
                prof_tstore = rvvd_read_ticks(decode_prof);
            }
            const auto mv=mask->type==GGML_TYPE_F32?
                __riscv_vle32_v_f32m4(reinterpret_cast<const float *>(mp),lanes):
                __riscv_vfwcvt_f_f_v_f32m4(__riscv_vreinterpret_v_u16m2_f16m2(
                    __riscv_vle16_v_u16m2(reinterpret_cast<const uint16_t *>(mp),lanes)),lanes);
            for(int h=0;h<D6_H;h++){
                auto score=__riscv_vle32_v_f32m4(scores128[h],lanes);
                score=__riscv_vfadd_vv_f32m4(__riscv_vfmul_vf_f32m4(score,D6_ATTN_SCALE,lanes),mv,lanes);
                const float mx=d6_max128(score,lanes);
                if(mx==-__builtin_inff()){
                    alpha[h]=1.0f;
                    for(size_t j=0;j<lanes;j++)p[h][j]=0.0f;
                    continue;
                }
                any_valid=true;
                const float nm=mx>m[h]?mx:m[h];
                const float a=(l[h]>0.0f&&nm>m[h])?d6_exp_neg(m[h]-nm):1.0f;
                alpha[h]=a;m[h]=nm;
                const auto w=d6_exp_neg_vec128(__riscv_vfsub_vf_f32m4(score,nm,lanes),lanes);
                __riscv_vse32_v_f32m4(p[h],w,lanes);
                l[h]=l[h]*a+d6_sum128(w,lanes);
            }
        }
        const uint64_t prof_tsoft = rvvd_read_ticks(decode_prof);
        if (decode_prof) {
            prof_qk += prof_tqk - prof_t0;
            prof_store += prof_tstore - prof_tqk;
            prof_softmax += prof_tsoft - prof_tstore;
        }
        if(!any_valid) continue;
        if(pv128) d6_pv128(vc,kv_head,v_trans,v_step,kb,lanes,u,p,alpha);
        else      d6_pv64(vc,kv_head,v_trans,v_step,kb,lanes,u,p,alpha);
        if (decode_prof) prof_pv += rvvd_read_ticks(true) - prof_tsoft;
    }
    if (decode_prof) {
        ggml_rvvd_prof_record(RVVD_QK, prof_qk);
        if (bk==128) ggml_rvvd_prof_record(RVVD_SCORE_STORE, prof_store);
        ggml_rvvd_prof_record(RVVD_SOFTMAX, prof_softmax);
        ggml_rvvd_prof_record(RVVD_PV, prof_pv);
    }
    for(int h=0;h<D6_H;h++){
        auto * dst=reinterpret_cast<float *>(static_cast<uint8_t *>(out->data)+
            worker*out->nb[2]+h*out->nb[1]);
        __riscv_vse32_v_f32m2(dst,__riscv_vle32_v_f32m2(u[h],D6_VL),D6_VL);
        __riscv_vse32_v_f32m2(dst+64,__riscv_vle32_v_f32m2(u[h]+64,D6_VL),D6_VL);
        dst[HEAD_DIM]=m[h];dst[HEAD_DIM+1]=l[h];
    }
}

static void d6_merge_and_wo(ggml_tensor * out, int group) {
    const auto * states=out->src[0];
    const auto * wo=out->src[1];
    GGML_ASSERT(out && out->type==GGML_TYPE_F32 && ggml_is_contiguous(out) &&
      out->ne[0]==HIDDEN_DIM && out->ne[1]==1 && out->ne[2]==N_Q_GROUPS &&
      states && states->type==GGML_TYPE_F32 && ggml_is_contiguous(states) &&
      states->ne[0]==DECODE6_STATE_STRIDE && states->ne[1]==D6_H &&
      states->ne[2]==DECODE6_WORKERS && wo && wo->type==GGML_TYPE_I8 &&
      wo->ne[0]==static_cast<int64_t>(WO_PACKED_BYTES) && wo->data);
    const int kv_head=group/2;
    const int head_offset=(group%2)*Q_HEADS_TILE;
    alignas(64) int8_t q8[WO_BLOCKS_PER_Q_GROUP][Q4_BLOCK_SIZE];
    float scales[WO_BLOCKS_PER_Q_GROUP];
    const bool decode_prof = ggml_rvvd_prof_active();
    const uint64_t prof_t0 = rvvd_read_ticks(decode_prof);
    for(int h=0;h<Q_HEADS_TILE;h++){
        const int hh=head_offset+h;
        const auto * a=reinterpret_cast<const float *>(static_cast<const uint8_t *>(states->data)+
            (2*kv_head)*states->nb[2]+hh*states->nb[1]);
        const auto * b=reinterpret_cast<const float *>(static_cast<const uint8_t *>(states->data)+
            (2*kv_head+1)*states->nb[2]+hh*states->nb[1]);
        const float m0=a[HEAD_DIM],m1=b[HEAD_DIM];
        const float l0=a[HEAD_DIM+1],l1=b[HEAD_DIM+1];
        GGML_ASSERT(l0>0.0f || l1>0.0f);
        const float m=m0>m1?m0:m1;
        const float w0=l0>0.0f?d6_exp_neg(m0-m):0.0f;
        const float w1=l1>0.0f?d6_exp_neg(m1-m):0.0f;
        const float l=w0*l0+w1*l1;
        GGML_ASSERT(l>0.0f);
        for(int offset=0;offset<HEAD_DIM;offset+=64){
            const auto x0=__riscv_vle32_v_f32m2(a+offset,D6_VL);
            const auto x1=__riscv_vle32_v_f32m2(b+offset,D6_VL);
            const auto v=__riscv_vfmul_vf_f32m2(
                __riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(x0,w0,D6_VL),
                                       __riscv_vfmul_vf_f32m2(x1,w1,D6_VL),D6_VL),1.0f/l,D6_VL);
            const int base=h*(HEAD_DIM/Q4_BLOCK_SIZE)+offset/Q4_BLOCK_SIZE;
            __riscv_vse8_v_i8mf2(q8[base+0],d6_quant32(v,0,scales[base+0]),Q4_BLOCK_SIZE);
            __riscv_vse8_v_i8mf2(q8[base+1],d6_quant32(v,1,scales[base+1]),Q4_BLOCK_SIZE);
        }
    }
    const uint64_t prof_t1 = rvvd_read_ticks(decode_prof);
    // One three-head packed W_O shard per worker, exactly as in prefill.
    const uint8_t * wg=static_cast<const uint8_t *>(wo->data)+group*WO_GROUP_BYTES;
    float * y=reinterpret_cast<float *>(static_cast<uint8_t *>(out->data)+group*out->nb[2]);
    for(int yt=0;yt<WO_SUBTILES;yt++){
        auto lo=__riscv_vfmv_v_f_f32m4(0.0f,D6_WO_VL);
        auto hi=lo;
        for(int b=0;b<WO_BLOCKS_PER_Q_GROUP;b++){
            const auto * wb=wg+yt*WO_SUBTILE_BYTES+b*WO_BLOCK_BYTES;
            auto isumlo=__riscv_vmv_v_x_i16m2(0,D6_WO_VL);
            auto isumhi=__riscv_vmv_v_x_i16m2(0,D6_WO_VL);
            for(int k=0;k<Q4_BLOCK_SIZE;k++){
                const auto packed=__riscv_vle8_v_u8m1(wb+k*WO_COL_BYTES,D6_WO_VL);
                const auto wlo=__riscv_vreinterpret_v_u8m1_i8m1(
                    __riscv_vsub_vx_u8m1(__riscv_vand_vx_u8m1(packed,15,D6_WO_VL),8,D6_WO_VL));
                const auto whi=__riscv_vreinterpret_v_u8m1_i8m1(
                    __riscv_vsub_vx_u8m1(__riscv_vsrl_vx_u8m1(packed,4,D6_WO_VL),8,D6_WO_VL));
                isumlo=__riscv_vwmacc_vx_i16m2(isumlo,q8[b][k],wlo,D6_WO_VL);
                isumhi=__riscv_vwmacc_vx_i16m2(isumhi,q8[b][k],whi,D6_WO_VL);
            }
            const float * wscale=reinterpret_cast<const float *>(wb+Q4_BLOCK_SIZE*WO_COL_BYTES);
            lo=__riscv_vfadd_vv_f32m4(lo,__riscv_vfmul_vf_f32m4(
                __riscv_vfmul_vv_f32m4(__riscv_vfwcvt_f_x_v_f32m4(isumlo,D6_WO_VL),
                    __riscv_vle32_v_f32m4(wscale,D6_WO_VL),D6_WO_VL),scales[b],D6_WO_VL),D6_WO_VL);
            hi=__riscv_vfadd_vv_f32m4(hi,__riscv_vfmul_vf_f32m4(
                __riscv_vfmul_vv_f32m4(__riscv_vfwcvt_f_x_v_f32m4(isumhi,D6_WO_VL),
                    __riscv_vle32_v_f32m4(wscale+D6_WO_VL,D6_WO_VL),D6_WO_VL),scales[b],D6_WO_VL),D6_WO_VL);
        }
        __riscv_vse32_v_f32m4(y+yt*WO_SUBTILE_OUTPUTS,lo,D6_WO_VL);
        __riscv_vse32_v_f32m4(y+yt*WO_SUBTILE_OUTPUTS+D6_WO_VL,hi,D6_WO_VL);
    }
    if (decode_prof) {
        const uint64_t prof_t2 = rvvd_read_ticks(true);
        ggml_rvvd_prof_record(RVVD_MERGE_Q8, prof_t1 - prof_t0);
        ggml_rvvd_prof_record(RVVD_WO, prof_t2 - prof_t1);
    }
}
} // anonymous namespace
#endif // RVV+Zvfh

// Called through GGML CPU backend extension; all 3 stages use 4 workers.
void forward_decode6_q(ggml_tensor * out,int ith,int nth,void * userdata){
    (void)userdata;
#if defined(__riscv_v) && defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    GGML_ASSERT(hardware_vlen_supported() && nth>0 && ith>=0 && ith<nth);
    for(int g=ith;g<N_Q_GROUPS;g+=nth)d6_project_one_group(out,g);
#else
    (void)out;(void)ith;(void)nth;
    GGML_ABORT("RVV1024 decode Q requires RVV1024 + Zvfh");
#endif
}
void forward_decode6_attention(ggml_tensor * out,int ith,int nth,void * userdata){
    (void)userdata;
#if defined(__riscv_v) && defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    GGML_ASSERT(hardware_vlen_supported() && nth>0 && ith>=0 && ith<nth);
    if(ith==0){
        static std::atomic_flag once=ATOMIC_FLAG_INIT;
        if(!once.test_and_set(std::memory_order_relaxed)) {
            const auto * cols = out->src[5];
            const char * pv = std::getenv("GGML_RVV1024_PV128");
            std::fprintf(stderr,
                "[RVV1024] decode6-split4: 4 workers, H6 each, 2 history partitions/KV head, BK%lld PV%s\n",
                static_cast<long long>(cols ? cols->ne[0] : 0), pv && std::strcmp(pv, "1") == 0 ? "128" : "64");
        }
    }
    for(int g=ith;g<DECODE6_WORKERS;g+=nth)d6_attention_split(out,g);
#else
    (void)out;(void)ith;(void)nth;
    GGML_ABORT("RVV1024 decode attention requires RVV1024 + Zvfh");
#endif
}
void forward_decode6_wo(ggml_tensor * out,int ith,int nth,void * userdata){
    (void)userdata;
#if defined(__riscv_v) && defined(__riscv_v_intrinsic) && defined(__riscv_zvfh)
    GGML_ASSERT(hardware_vlen_supported() && nth>0 && ith>=0 && ith<nth);
    for(int g=ith;g<N_Q_GROUPS;g+=nth)d6_merge_and_wo(out,g);
#else
    (void)out;(void)ith;(void)nth;
    GGML_ABORT("RVV1024 decode W_O requires RVV1024 + Zvfh");
#endif
}
