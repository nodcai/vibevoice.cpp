// dfn_fast.cpp — hand-written single-frame DeepFilterNet3 forward.
// See dfn_fast.h. Every stage below mirrors one GGML builder in
// dfn.cpp; the comments name the builder + weight it corresponds to.
//
// Kernels have a NEON path (Apple silicon, Jetson) and a plain-C
// fallback. Numerically the two agree to float rounding; the NEON
// gate activations use ggml's polynomial expf instead of libm.

#include "dfn_fast.hpp"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define DFN_FAST_NEON 1
#else
#define DFN_FAST_NEON 0
#endif
#if DFN_FAST_NEON && defined(__ARM_FEATURE_DOTPROD)
#define DFN_FAST_DOTPROD 1
#else
#define DFN_FAST_DOTPROD 0
#endif

constexpr int kQBlock = 32; // int8 quantisation block

namespace {

constexpr int kC     = 64;  // conv channel width everywhere
constexpr int kH     = 256; // GRU hidden
constexpr int kErb   = 32;
constexpr int kSpec  = 96;
constexpr int kGates = 3 * kH;

using vec = std::vector<float>;

// (3 time × 3 freq) conv, one input channel → 64 output channels.
struct conv_tf {
    vec w; // [64][3 kt][3 kf]
    vec b; // [64] or empty
};
// Depthwise 1×3 (freq) → pointwise 64→64 + bias + relu.
struct dw_pw {
    vec dw; // [64][3]
    vec pw; // [64 oc][64 ic]
    vec b;  // [64]
};
// GroupedLinear: y[g*opg + o] = Σ_i w[(g*ipg + i)*opg + o] · x[g*ipg + i]
struct grouped {
    vec w;
    int G = 0, ipg = 0, opg = 0;
};
// ONNX GRU, linear_before_reset = 1. The two big matrices are kept in
// F16 on NEON (half the bytes streamed per frame; converted in
// registers) and in F32 otherwise.
struct gru_w {
#if DFN_FAST_NEON
    std::vector<ggml_fp16_t> W16; // [768][in]
    std::vector<ggml_fp16_t> R16; // [768][256]
#else
    vec W; // [768][in]
    vec R; // [768][256]
#endif
#if DFN_FAST_DOTPROD
    // Opt-in (DFN_GRU_INT8=1): int8 weights with one scale per 32-wide
    // block, dotted against block-quantised int8 activations via SDOT.
    // Quarter the bytes of F32 — the matvecs are L2-bandwidth bound —
    // at ~1e-3 relative error per gate pre-activation.
    std::vector<int8_t> W8, R8;
    vec                 Ws, Rs; // [rows][K/32] block scales
#endif
    vec Wb, Rb; // [768]
    int in = 0;
};
// Depthwise 1×1 skip conv + bias + relu (erb_dec convNp).
struct skip_dw {
    vec w, b; // [64]
};

} // namespace

struct dfn_fast_weights {
    bool gru_int8 = false; // GRU matvecs run on the int8 path (DFN_GRU_INT8=1)

    // ── encoder ───────────────────────────────────────────────────
    conv_tf enc_conv0;                    // Conv_282 / Conv_283
    dw_pw   enc_erb1, enc_erb2, enc_erb3; // erb_convN.0.weight + Conv_285/286, 288/289, 291/292
    conv_tf enc_df_conv0;                 // df_conv0.1.weight (group=2, no bias)
    vec     enc_df_pw, enc_df_pw_b;       // Conv_294 / Conv_295
    dw_pw   enc_df_conv1;                 // df_conv1.0.weight + Conv_297/298
    grouped enc_fc_emb;                   // df_fc_emb.0.weight  (32 groups, 96 → 16)
    grouped enc_lin_in, enc_lin_out;      // emb_gru.linear_in / linear_out
    gru_w   enc_gru;                      // emb_gru.gru

    // ── erb_dec ───────────────────────────────────────────────────
    grouped erb_lin_in, erb_lin_out;
    gru_w   erb_gru1, erb_gru2;
    skip_dw erb_skip3, erb_skip2, erb_skip1, erb_skip0; // Conv_288/289, 294/295, 300/301, 306/307
    dw_pw   erb_convt3;                                 // convt3.0.weight + Conv_291/292
    vec     erb_convt2_dw, erb_convt2_pw, erb_convt2_b; // convt2.0.weight + Conv_297/298
    vec     erb_convt1_dw, erb_convt1_pw, erb_convt1_b; // convt1.0.weight + Conv_303/304
    vec     erb_out_w;                                  // Conv_309  [64 ic][3 kf]
    float   erb_out_b = 0.0f;                           // Conv_310

    // ── df_dec ────────────────────────────────────────────────────
    grouped df_lin_in;                  // df_gru.linear_in (8 groups, 64 → 32)
    gru_w   df_gru1, df_gru2;           // df_gru.gru.gru / gru_1
    grouped df_skip;                    // df_skip.weight (16 groups, 32 → 16)
    grouped df_out;                     // df_out.0.weight (16 groups, 16 → 60)
    vec     df_alpha_w;                 // MatMul_321 [256]
    float   df_alpha_b = 0.0f;          // df_fc_a.0.bias
    vec     df_convp_w;                 // df_convp.1.weight [10 oc][32 icl][5 kt]
    vec     df_convp_pw, df_convp_pw_b; // Conv_268 [10][10], Conv_269 [10]
};

struct dfn_fast_scratch {
    vec pad;                    // 3 × (96 + 2) zero-bordered input rows
    vec e0, e1, e2, e3;         // 64×32, 64×16, 64×8, 64×8
    vec c0pre, c0, dff;         // 64×96, 64×96, 64×48
    vec tmp;                    // 64×96 depthwise / skip scratch
    vec flat, emb_in, combined; // 3072 (48×64), 512, 512
    vec x256, emb;              // 256, 512
    vec xlin, hlin;             // 768, 768
    vec gru_out, dec_a, dec_b;  // 512, 64×32, 64×32
    vec skip_add, df_out;       // 256, 960
    vec convp_pre, convp;       // 10×96, 10×96
    std::vector<const float*> rows; // B-row pointers for gemm_rows (≤ 160)
    std::vector<int8_t>       xq, hq; // int8 GRU inputs (256 each)
    vec                       xs, hs; // their block scales (256 / 32)
};

namespace {

// ── weight loading ───────────────────────────────────────────────

bool fetch(const std::map<std::string, ggml_tensor*>& tensors, const char* name, size_t expect, vec& out,
           std::string* err) {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second || !it->second->buffer) {
        if (err) *err = std::string("missing tensor ") + name;
        return false;
    }
    ggml_tensor* t = it->second;
    const size_t n = (size_t)ggml_nelements(t);
    if (n != expect) {
        if (err) *err = std::string("tensor ") + name + ": expected " + std::to_string(expect) + " elements, got " +
                        std::to_string(n);
        return false;
    }
    if (!ggml_is_contiguous(t)) {
        if (err) *err = std::string("tensor ") + name + " is not contiguous";
        return false;
    }
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int64_t)n);
    } else {
        if (err) *err = std::string("tensor ") + name + ": unsupported type " + ggml_type_name(t->type);
        return false;
    }
    return true;
}

// Symmetric int8 with one absmax scale per 32-wide block, row-major.
void quantize_rows(const vec& W, int rows, int K, std::vector<int8_t>& q, vec& scales) {
    const int nb = K / kQBlock;
    q.resize((size_t)rows * K);
    scales.resize((size_t)rows * nb);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const float* src = W.data() + (size_t)r * K + (size_t)b * kQBlock;
            float        amax = 0.0f;
            for (int i = 0; i < kQBlock; ++i) amax = std::max(amax, std::fabs(src[i]));
            const float d  = amax / 127.0f;
            const float id = d > 0.0f ? 1.0f / d : 0.0f;
            scales[(size_t)r * nb + b] = d;
            int8_t* dst = q.data() + (size_t)r * K + (size_t)b * kQBlock;
            for (int i = 0; i < kQBlock; ++i) dst[i] = (int8_t)lrintf(src[i] * id);
        }
    }
}

struct loader {
    const std::map<std::string, ggml_tensor*>& t;
    std::string* err;
    bool int8;
    bool ok = true;

    vec get(const char* name, size_t n) {
        vec v;
        if (ok && !fetch(t, name, n, v, err)) ok = false;
        return v;
    }
    float scalar(const char* name) {
        vec v = get(name, 1);
        return v.empty() ? 0.0f : v[0];
    }
    conv_tf conv3x3(const char* w, const char* b) {
        conv_tf c;
        c.w = get(w, (size_t)kC * 9);
        if (b) c.b = get(b, kC);
        return c;
    }
    dw_pw dwpw(const char* dw, const char* pw, const char* b) {
        dw_pw d;
        d.dw = get(dw, (size_t)kC * 3);
        d.pw = get(pw, (size_t)kC * kC);
        d.b  = get(b, kC);
        return d;
    }
    grouped grp(const char* name, int G, int ipg, int opg) {
        grouped g;
        g.G = G; g.ipg = ipg; g.opg = opg;
        g.w = get(name, (size_t)G * ipg * opg);
        return g;
    }
    gru_w gru(const char* prefix, int in) {
        gru_w g;
        g.in  = in;
        vec W = get((std::string(prefix) + ".W").c_str(), (size_t)kGates * in);
        vec R = get((std::string(prefix) + ".R").c_str(), (size_t)kGates * kH);
        vec B = get((std::string(prefix) + ".B").c_str(), (size_t)2 * kGates);
        if (!B.empty()) {
            g.Wb.assign(B.begin(), B.begin() + kGates);
            g.Rb.assign(B.begin() + kGates, B.end());
        }
#if DFN_FAST_DOTPROD
        if (int8 && ok) {
            quantize_rows(W, kGates, in, g.W8, g.Ws);
            quantize_rows(R, kGates, kH, g.R8, g.Rs);
        }
#endif
#if DFN_FAST_NEON
        g.W16.resize(W.size());
        g.R16.resize(R.size());
        ggml_fp32_to_fp16_row(W.data(), g.W16.data(), (int64_t)W.size());
        ggml_fp32_to_fp16_row(R.data(), g.R16.data(), (int64_t)R.size());
#else
        g.W = std::move(W);
        g.R = std::move(R);
#endif
        return g;
    }
    skip_dw skip(const char* w, const char* b) {
        skip_dw s;
        s.w = get(w, kC);
        s.b = get(b, kC);
        return s;
    }
};

// ── activations ──────────────────────────────────────────────────

inline float sigmoid_s(float x) { return 1.0f / (1.0f + expf(-x)); }

#if DFN_FAST_NEON
// expf on four lanes. Same polynomial as ggml's ggml_v_expf (ggml-cpu/
// vec.h, MIT); reproduced here so this file only depends on ggml's
// public headers. Max relative error ≈ 1 ulp-ish; handles overflow
// (→ +inf) and underflow (→ 0) like libm.
inline float32x4_t v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t  e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t  c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
                                    vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                                              vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u),
                                    u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) return vfmaq_f32(k, j, k);
    const uint32x4_t  d  = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}
inline float32x4_t v_sigmoid(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    return vdivq_f32(one, vaddq_f32(one, v_expf(vnegq_f32(x))));
}
// tanh(x) = 1 - 2 / (1 + exp(2x)); exp overflow → +inf → 1, underflow → -1.
inline float32x4_t v_tanh(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t two = vdupq_n_f32(2.0f);
    return vsubq_f32(one, vdivq_f32(two, vaddq_f32(one, v_expf(vmulq_f32(two, x)))));
}
#endif

// y = sigmoid(x) / tanh(x), n arbitrary.
void sigmoid_vec(int n, float* y, const float* x) {
    int i = 0;
#if DFN_FAST_NEON
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, v_sigmoid(vld1q_f32(x + i)));
#endif
    for (; i < n; ++i) y[i] = sigmoid_s(x[i]);
}
void tanh_vec(int n, float* y, const float* x) {
    int i = 0;
#if DFN_FAST_NEON
    for (; i + 4 <= n; i += 4) vst1q_f32(y + i, v_tanh(vld1q_f32(x + i)));
#endif
    for (; i < n; ++i) y[i] = tanhf(x[i]);
}

// ── small GEMM over row pointers ─────────────────────────────────
//
//   C[i][j] = (accumulate ? C[i][j] : bias[i]) + Σ_k A[i·lda + k] · Brow[k][j]
//
// with optional relu. `Brow[k]` points at row k of B (rows need not
// be equally spaced — that is what lets the 3×3 convs use shifted
// views of one padded buffer and df_convp mix history + current
// rows). Every conv in the network is one call of this.

#if DFN_FAST_NEON
// One R × (4·CV) register tile.
template <int R, int CV>
inline void gemm_tile(int K, const float* A, int lda, const float* const* Brow, int j, const float* bias, float* C,
                      int ldc, bool accumulate, bool relu) {
    float32x4_t acc[R][CV];
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < CV; ++c) {
            if (accumulate)
                acc[r][c] = vld1q_f32(C + (size_t)r * ldc + j + 4 * c);
            else
                acc[r][c] = vdupq_n_f32(bias ? bias[r] : 0.0f);
        }
    }
    int k = 0;
    // Four k at a time: one A vector load per row feeds four
    // lane-broadcast FMAs, so loads per FMA drop from 1/2 to ~1/3.
    for (; k + 4 <= K; k += 4) {
        float32x4_t av[R];
        for (int r = 0; r < R; ++r) av[r] = vld1q_f32(A + (size_t)r * lda + k);
        for (int kk = 0; kk < 4; ++kk) {
            const float* b = Brow[k + kk] + j;
            float32x4_t  bv[CV];
            for (int c = 0; c < CV; ++c) bv[c] = vld1q_f32(b + 4 * c);
            for (int r = 0; r < R; ++r) {
                switch (kk) {
                    case 0: for (int c = 0; c < CV; ++c) acc[r][c] = vfmaq_laneq_f32(acc[r][c], bv[c], av[r], 0); break;
                    case 1: for (int c = 0; c < CV; ++c) acc[r][c] = vfmaq_laneq_f32(acc[r][c], bv[c], av[r], 1); break;
                    case 2: for (int c = 0; c < CV; ++c) acc[r][c] = vfmaq_laneq_f32(acc[r][c], bv[c], av[r], 2); break;
                    default: for (int c = 0; c < CV; ++c) acc[r][c] = vfmaq_laneq_f32(acc[r][c], bv[c], av[r], 3); break;
                }
            }
        }
    }
    for (; k < K; ++k) {
        const float* b = Brow[k] + j;
        float32x4_t  bv[CV];
        for (int c = 0; c < CV; ++c) bv[c] = vld1q_f32(b + 4 * c);
        for (int r = 0; r < R; ++r) {
            const float32x4_t a = vld1q_dup_f32(A + (size_t)r * lda + k);
            for (int c = 0; c < CV; ++c) acc[r][c] = vfmaq_f32(acc[r][c], bv[c], a);
        }
    }
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < CV; ++c) {
            float32x4_t v = relu ? vmaxq_f32(acc[r][c], zero) : acc[r][c];
            vst1q_f32(C + (size_t)r * ldc + j + 4 * c, v);
        }
    }
}
#endif

void gemm_rows(int M, int N, int K, const float* A, int lda, const float* const* Brow, const float* bias, float* C,
               int ldc, bool accumulate, bool relu) {
    int i = 0;
#if DFN_FAST_NEON
    auto run_rows = [&](auto rows_tag, int i0) {
        constexpr int R = decltype(rows_tag)::value;
        const float*  a = A + (size_t)i0 * lda;
        const float*  b = bias ? bias + i0 : nullptr;
        float*        c = C + (size_t)i0 * ldc;
        int           j = 0;
        for (; j + 16 <= N; j += 16) gemm_tile<R, 4>(K, a, lda, Brow, j, b, c, ldc, accumulate, relu);
        for (; j + 4 <= N; j += 4) gemm_tile<R, 1>(K, a, lda, Brow, j, b, c, ldc, accumulate, relu);
        for (; j < N; ++j) {
            for (int r = 0; r < R; ++r) {
                float s = accumulate ? c[(size_t)r * ldc + j] : (b ? b[r] : 0.0f);
                for (int k = 0; k < K; ++k) s += a[(size_t)r * lda + k] * Brow[k][j];
                c[(size_t)r * ldc + j] = relu ? std::max(0.0f, s) : s;
            }
        }
    };
    for (; i + 4 <= M; i += 4) run_rows(std::integral_constant<int, 4>{}, i);
    for (; i < M; ++i) run_rows(std::integral_constant<int, 1>{}, i);
#else
    for (; i < M; ++i) {
        float*       c = C + (size_t)i * ldc;
        const float* a = A + (size_t)i * lda;
        if (!accumulate) {
            const float b0 = bias ? bias[i] : 0.0f;
            for (int j = 0; j < N; ++j) c[j] = b0;
        }
        for (int k = 0; k < K; ++k) {
            const float  ak = a[k];
            const float* bk = Brow[k];
            for (int j = 0; j < N; ++j) c[j] += ak * bk[j];
        }
        if (relu)
            for (int j = 0; j < N; ++j) c[j] = std::max(0.0f, c[j]);
    }
#endif
}

// ── GRU matrix–vector products ───────────────────────────────────
// y[j] = b[j] + Σ_k W[j][k] · x[k], j < rows. rows % 4 == 0, K % 16 == 0.

#if DFN_FAST_NEON
inline void cvt16x4(const ggml_fp16_t* p, float32x4_t& lo, float32x4_t& hi) {
    const float16x8_t h = vld1q_f16((const float16_t*)p);
    lo                  = vcvt_f32_f16(vget_low_f16(h));
    hi                  = vcvt_high_f32_f16(h);
}
void matvec_f16(const ggml_fp16_t* W, int rows, int K, const float* x, const float* b, float* y) {
    for (int j = 0; j < rows; j += 4) {
        float32x4_t acc[4][4];
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) acc[r][c] = vdupq_n_f32(0.0f);
        const ggml_fp16_t* w[4] = {W + (size_t)(j + 0) * K, W + (size_t)(j + 1) * K, W + (size_t)(j + 2) * K,
                                   W + (size_t)(j + 3) * K};
        for (int k = 0; k < K; k += 16) {
            const float32x4_t x0 = vld1q_f32(x + k), x1 = vld1q_f32(x + k + 4), x2 = vld1q_f32(x + k + 8),
                              x3 = vld1q_f32(x + k + 12);
            for (int r = 0; r < 4; ++r) {
                float32x4_t w0, w1, w2, w3;
                cvt16x4(w[r] + k, w0, w1);
                cvt16x4(w[r] + k + 8, w2, w3);
                acc[r][0] = vfmaq_f32(acc[r][0], w0, x0);
                acc[r][1] = vfmaq_f32(acc[r][1], w1, x1);
                acc[r][2] = vfmaq_f32(acc[r][2], w2, x2);
                acc[r][3] = vfmaq_f32(acc[r][3], w3, x3);
            }
        }
        for (int r = 0; r < 4; ++r) {
            const float32x4_t s = vaddq_f32(vaddq_f32(acc[r][0], acc[r][1]), vaddq_f32(acc[r][2], acc[r][3]));
            y[j + r]            = b[j + r] + vaddvq_f32(s);
        }
    }
}
#else
void matvec_f32(const float* W, int rows, int K, const float* x, const float* b, float* y) {
    for (int j = 0; j < rows; ++j) {
        const float* w  = W + (size_t)j * K;
        float        s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int k = 0; k < K; k += 4) {
            s0 += w[k] * x[k];
            s1 += w[k + 1] * x[k + 1];
            s2 += w[k + 2] * x[k + 2];
            s3 += w[k + 3] * x[k + 3];
        }
        y[j] = b[j] + (s0 + s1) + (s2 + s3);
    }
}
#endif

#if DFN_FAST_DOTPROD
// x → int8 with one scale per 32-wide block.
void quantize_vec(const float* x, int K, int8_t* q, float* scales) {
    for (int b = 0; b < K / kQBlock; ++b) {
        const float* src  = x + (size_t)b * kQBlock;
        float32x4_t  amax = vdupq_n_f32(0.0f);
        for (int i = 0; i < kQBlock; i += 4) amax = vmaxq_f32(amax, vabsq_f32(vld1q_f32(src + i)));
        const float d  = vmaxvq_f32(amax) / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        scales[b]      = d;
        const float32x4_t vid = vdupq_n_f32(id);
        int8_t*           dst = q + (size_t)b * kQBlock;
        for (int i = 0; i < kQBlock; i += 16) {
            const int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src + i), vid));
            const int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src + i + 4), vid));
            const int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src + i + 8), vid));
            const int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src + i + 12), vid));
            const int16x8_t p0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            const int16x8_t p1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            vst1q_s8(dst + i, vcombine_s8(vqmovn_s16(p0), vqmovn_s16(p1)));
        }
    }
}

// y[j] = b[j] + Σ_blocks Ws[j][b]·xs[b]·Σ_i W8[j][b·32+i]·xq[b·32+i]
void matvec_q8(const int8_t* W, const float* Ws, int rows, int K, const int8_t* xq, const float* xs, const float* b,
               float* y) {
    const int nb = K / kQBlock;
    for (int j = 0; j < rows; j += 4) {
        float32x4_t acc[4];
        for (int r = 0; r < 4; ++r) acc[r] = vdupq_n_f32(0.0f);
        for (int blk = 0; blk < nb; ++blk) {
            const int8x16_t x0 = vld1q_s8(xq + blk * kQBlock);
            const int8x16_t x1 = vld1q_s8(xq + blk * kQBlock + 16);
            const float     sx = xs[blk];
            for (int r = 0; r < 4; ++r) {
                const int8_t* w = W + (size_t)(j + r) * K + blk * kQBlock;
                int32x4_t     d = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w), x0);
                d               = vdotq_s32(d, vld1q_s8(w + 16), x1);
                acc[r]          = vfmaq_n_f32(acc[r], vcvtq_f32_s32(d), Ws[(size_t)(j + r) * nb + blk] * sx);
            }
        }
        for (int r = 0; r < 4; ++r) y[j + r] = b[j + r] + vaddvq_f32(acc[r]);
    }
}
#endif

// build_gru_step_lbr1 for one frame; `h` is updated in place.
//   z = σ(Wz·x + Wbz + Rz·h + Rbz)
//   r = σ(Wr·x + Wbr + Rr·h + Rbr)
//   n = tanh(Wn·x + Wbn + r ⊙ (Rn·h + Rbn))
//   h = n + z ⊙ (h - n)
void gru_step(const gru_w& g, dfn_fast_scratch* s, const float* x, float* h, float* xlin, float* hlin) {
#if DFN_FAST_NEON
#if DFN_FAST_DOTPROD
    if (!g.W8.empty()) {
        quantize_vec(x, g.in, s->xq.data(), s->xs.data());
        quantize_vec(h, kH, s->hq.data(), s->hs.data());
        matvec_q8(g.W8.data(), g.Ws.data(), kGates, g.in, s->xq.data(), s->xs.data(), g.Wb.data(), xlin);
        matvec_q8(g.R8.data(), g.Rs.data(), kGates, kH, s->hq.data(), s->hs.data(), g.Rb.data(), hlin);
    } else
#endif
    {
        matvec_f16(g.W16.data(), kGates, g.in, x, g.Wb.data(), xlin);
        matvec_f16(g.R16.data(), kGates, kH, h, g.Rb.data(), hlin);
    }
    for (int k = 0; k < kH; k += 4) {
        const float32x4_t z  = v_sigmoid(vaddq_f32(vld1q_f32(xlin + k), vld1q_f32(hlin + k)));
        const float32x4_t r  = v_sigmoid(vaddq_f32(vld1q_f32(xlin + kH + k), vld1q_f32(hlin + kH + k)));
        const float32x4_t n  = v_tanh(vfmaq_f32(vld1q_f32(xlin + 2 * kH + k), r, vld1q_f32(hlin + 2 * kH + k)));
        const float32x4_t hp = vld1q_f32(h + k);
        vst1q_f32(h + k, vfmaq_f32(n, z, vsubq_f32(hp, n)));
    }
#else
    (void)s;
    matvec_f32(g.W.data(), kGates, g.in, x, g.Wb.data(), xlin);
    matvec_f32(g.R.data(), kGates, kH, h, g.Rb.data(), hlin);
    for (int k = 0; k < kH; ++k) {
        const float z = sigmoid_s(xlin[k] + hlin[k]);
        const float r = sigmoid_s(xlin[kH + k] + hlin[kH + k]);
        const float n = tanhf(xlin[2 * kH + k] + r * hlin[2 * kH + k]);
        h[k]          = n + z * (h[k] - n);
    }
#endif
}

// ── grouped linear ───────────────────────────────────────────────
// build_grouped_linear (+ optional relu). Per group this is a
// (ipg → opg) matvec with the weight stored output-contiguous, so it
// runs as ipg rank-1 updates on an opg-wide register accumulator.

#if DFN_FAST_NEON
template <int OPG>
void grouped_lin_neon(const grouped& g, const float* x, float* y, bool relu) {
    constexpr int     Q    = OPG / 4;
    constexpr bool    two  = (2 * Q <= 16); // second accumulator set halves the FMA chain
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (int gi = 0; gi < g.G; ++gi) {
        float32x4_t acc[Q], acc2[two ? Q : 1];
        for (int q = 0; q < Q; ++q) acc[q] = zero;
        if (two)
            for (int q = 0; q < Q; ++q) acc2[q] = zero;
        const float* xg = x + (size_t)gi * g.ipg;
        const float* wg = g.w.data() + (size_t)gi * g.ipg * OPG;
        int          i  = 0;
        if (two) {
            for (; i + 2 <= g.ipg; i += 2) {
                const float32x4_t x0 = vdupq_n_f32(xg[i]);
                const float32x4_t x1 = vdupq_n_f32(xg[i + 1]);
                const float*      w0 = wg + (size_t)i * OPG;
                const float*      w1 = w0 + OPG;
                for (int q = 0; q < Q; ++q) {
                    acc[q]  = vfmaq_f32(acc[q], vld1q_f32(w0 + 4 * q), x0);
                    acc2[q] = vfmaq_f32(acc2[q], vld1q_f32(w1 + 4 * q), x1);
                }
            }
        }
        for (; i < g.ipg; ++i) {
            const float32x4_t xb = vdupq_n_f32(xg[i]);
            const float*      wr = wg + (size_t)i * OPG;
            for (int q = 0; q < Q; ++q) acc[q] = vfmaq_f32(acc[q], vld1q_f32(wr + 4 * q), xb);
        }
        float* yo = y + (size_t)gi * OPG;
        for (int q = 0; q < Q; ++q) {
            float32x4_t v = two ? vaddq_f32(acc[q], acc2[q]) : acc[q];
            vst1q_f32(yo + 4 * q, relu ? vmaxq_f32(v, zero) : v);
        }
    }
}
#endif

void grouped_lin(const grouped& g, const float* x, float* y, bool relu) {
#if DFN_FAST_NEON
    switch (g.opg) {
        case 16: grouped_lin_neon<16>(g, x, y, relu); return;
        case 32: grouped_lin_neon<32>(g, x, y, relu); return;
        case 60: grouped_lin_neon<60>(g, x, y, relu); return;
        default: break;
    }
#endif
    for (int gi = 0; gi < g.G; ++gi) {
        float*       yo = y + (size_t)gi * g.opg;
        const float* xg = x + (size_t)gi * g.ipg;
        for (int o = 0; o < g.opg; ++o) yo[o] = 0.0f;
        for (int i = 0; i < g.ipg; ++i) {
            const float  xi = xg[i];
            const float* wr = g.w.data() + ((size_t)gi * g.ipg + i) * g.opg;
            for (int o = 0; o < g.opg; ++o) yo[o] += xi * wr[o];
        }
        if (relu)
            for (int o = 0; o < g.opg; ++o) yo[o] = std::max(0.0f, yo[o]);
    }
}

// ── conv helpers ─────────────────────────────────────────────────

// Copy F-long rows into zero-bordered (F+2)-long rows.
inline void pad_rows(const float* x, int rows, int F, float* pad) {
    for (int r = 0; r < rows; ++r) {
        float* p = pad + (size_t)r * (F + 2);
        p[0]     = 0.0f;
        std::memcpy(p + 1, x + (size_t)r * F, (size_t)F * sizeof(float));
        p[F + 1] = 0.0f;
    }
}

// ggml_conv_2d(W ne=[3,3,1,64], x ne=[F,3,1,1], pW=1, pH=0) for T=1:
//   y[oc][f] = b[oc] + Σ_kt Σ_kf W[oc][kt][kf] · x[kt][f + kf - 1]
// `x3` holds the three input frames (t-2, t-1, t) as rows of F.
// [oc_begin, oc_end) lets the group=2 df_conv0 route each input
// channel to its 32 output channels.
void conv_tf3x3(const conv_tf& c, const float* x3, int F, dfn_fast_scratch* s, float* y, int oc_begin, int oc_end,
                bool relu) {
    pad_rows(x3, 3, F, s->pad.data());
    const int     P    = F + 2;
    const float** rows = s->rows.data();
    for (int kt = 0; kt < 3; ++kt)
        for (int kf = 0; kf < 3; ++kf) rows[kt * 3 + kf] = s->pad.data() + (size_t)kt * P + kf;
    gemm_rows(oc_end - oc_begin, F, 9, c.w.data() + (size_t)oc_begin * 9, 9, rows,
              c.b.empty() ? nullptr : c.b.data() + oc_begin, y + (size_t)oc_begin * F, F, false, relu);
}

// ggml_conv_2d(W ne=[1,1,64,64], x) + bias (+ relu):
//   y[oc][f] = b[oc] + Σ_ic W[oc][ic] · x[ic][f]
void pointwise(const float* W, const float* b, const float* x, int F, dfn_fast_scratch* s, float* y, bool relu) {
    const float** rows = s->rows.data();
    for (int ic = 0; ic < kC; ++ic) rows[ic] = x + (size_t)ic * F;
    gemm_rows(kC, F, kC, W, kC, rows, b, y, F, false, relu);
}

// ggml_conv_2d_dw(W ne=[3,1,1,64], x, sW=s, pW=1): per-channel 1×3
// freq conv. Fout = (Fin - 1) / s + 1.
void dw3(const float* w, const float* x, int Fin, int s, float* y, int Fout) {
    for (int c = 0; c < kC; ++c) {
        const float  w0 = w[c * 3 + 0], w1 = w[c * 3 + 1], w2 = w[c * 3 + 2];
        const float* xc = x + (size_t)c * Fin;
        float*       yc = y + (size_t)c * Fout;
        for (int o = 0; o < Fout; ++o) {
            const int i   = s * o - 1;
            float     acc = w1 * xc[i + 1];
            if (i >= 0) acc += w0 * xc[i];
            if (i + 2 < Fin) acc += w2 * xc[i + 2];
            yc[o] = acc;
        }
    }
}

// depthwise 1×3 → pointwise → bias → relu (erb_convN / df_conv1 / convt3)
void dw_pw_block(const dw_pw& l, const float* x, int Fin, int s, dfn_fast_scratch* sc, float* y) {
    const int Fout = (Fin - 1) / s + 1;
    dw3(l.dw.data(), x, Fin, s, sc->tmp.data(), Fout);
    pointwise(l.pw.data(), l.b.data(), sc->tmp.data(), Fout, sc, y, /*relu=*/true);
}

// build_skip_pointwise_dw: acc[c][f] += relu(x[c][f] · w[c] + b[c])
void skip_add_into(const skip_dw& s, const float* x, int F, float* acc) {
    for (int c = 0; c < kC; ++c) {
        const float  w  = s.w[c], b = s.b[c];
        const float* xc = x + (size_t)c * F;
        float*       ac = acc + (size_t)c * F;
        for (int f = 0; f < F; ++f) ac[f] += std::max(0.0f, xc[f] * w + b);
    }
}

// build_depthwise_conv_transpose_s2k3:
//   out[2i]   = w[1]·x[i]
//   out[2i+1] = w[2]·x[i] + w[0]·x[i+1]   (x[L] = 0)
void dw_convT_s2k3(const float* w, const float* x, int L, float* y) {
    for (int c = 0; c < kC; ++c) {
        const float  w0 = w[c * 3 + 0], w1 = w[c * 3 + 1], w2 = w[c * 3 + 2];
        const float* xc = x + (size_t)c * L;
        float*       yc = y + (size_t)c * 2 * L;
        for (int i = 0; i < L; ++i) {
            const float xn = (i + 1 < L) ? xc[i + 1] : 0.0f;
            yc[2 * i]      = w1 * xc[i];
            yc[2 * i + 1]  = w2 * xc[i] + w0 * xn;
        }
    }
}

} // namespace

// ── public API ───────────────────────────────────────────────────

dfn_fast_weights* dfn_fast_prepare(const std::map<std::string, ggml_tensor*>& tensors, std::string* err) {
    auto* w = new dfn_fast_weights();
    w->gru_int8 = DFN_FAST_DOTPROD && std::getenv("DFN_GRU_INT8") != nullptr;
    loader L{tensors, err, w->gru_int8};

    // encoder
    w->enc_conv0    = L.conv3x3("dfn.enc.onnx::Conv_282", "dfn.enc.onnx::Conv_283");
    w->enc_erb1     = L.dwpw("dfn.enc.erb_conv1.0.weight", "dfn.enc.onnx::Conv_285", "dfn.enc.onnx::Conv_286");
    w->enc_erb2     = L.dwpw("dfn.enc.erb_conv2.0.weight", "dfn.enc.onnx::Conv_288", "dfn.enc.onnx::Conv_289");
    w->enc_erb3     = L.dwpw("dfn.enc.erb_conv3.0.weight", "dfn.enc.onnx::Conv_291", "dfn.enc.onnx::Conv_292");
    w->enc_df_conv0 = L.conv3x3("dfn.enc.df_conv0.1.weight", nullptr);
    w->enc_df_pw    = L.get("dfn.enc.onnx::Conv_294", (size_t)kC * kC);
    w->enc_df_pw_b  = L.get("dfn.enc.onnx::Conv_295", kC);
    w->enc_df_conv1 = L.dwpw("dfn.enc.df_conv1.0.weight", "dfn.enc.onnx::Conv_297", "dfn.enc.onnx::Conv_298");
    w->enc_fc_emb   = L.grp("dfn.enc.df_fc_emb.0.weight", 32, 96, 16);
    w->enc_lin_in   = L.grp("dfn.enc.emb_gru.linear_in.0.weight", 16, 32, 16);
    w->enc_lin_out  = L.grp("dfn.enc.emb_gru.linear_out.0.weight", 16, 16, 32);
    w->enc_gru      = L.gru("dfn.enc.emb_gru.gru", kH);

    // erb_dec
    w->erb_lin_in    = L.grp("dfn.erb_dec.emb_gru.linear_in.0.weight", 16, 32, 16);
    w->erb_lin_out   = L.grp("dfn.erb_dec.emb_gru.linear_out.0.weight", 16, 16, 32);
    w->erb_gru1      = L.gru("dfn.erb_dec.emb_gru.gru", kH);
    w->erb_gru2      = L.gru("dfn.erb_dec.emb_gru.gru_1", kH);
    w->erb_skip3     = L.skip("dfn.erb_dec.onnx::Conv_288", "dfn.erb_dec.onnx::Conv_289");
    w->erb_skip2     = L.skip("dfn.erb_dec.onnx::Conv_294", "dfn.erb_dec.onnx::Conv_295");
    w->erb_skip1     = L.skip("dfn.erb_dec.onnx::Conv_300", "dfn.erb_dec.onnx::Conv_301");
    w->erb_skip0     = L.skip("dfn.erb_dec.onnx::Conv_306", "dfn.erb_dec.onnx::Conv_307");
    w->erb_convt3    = L.dwpw("dfn.erb_dec.convt3.0.weight", "dfn.erb_dec.onnx::Conv_291", "dfn.erb_dec.onnx::Conv_292");
    w->erb_convt2_dw = L.get("dfn.erb_dec.convt2.0.weight", (size_t)kC * 3);
    w->erb_convt2_pw = L.get("dfn.erb_dec.onnx::Conv_297", (size_t)kC * kC);
    w->erb_convt2_b  = L.get("dfn.erb_dec.onnx::Conv_298", kC);
    w->erb_convt1_dw = L.get("dfn.erb_dec.convt1.0.weight", (size_t)kC * 3);
    w->erb_convt1_pw = L.get("dfn.erb_dec.onnx::Conv_303", (size_t)kC * kC);
    w->erb_convt1_b  = L.get("dfn.erb_dec.onnx::Conv_304", kC);
    w->erb_out_w     = L.get("dfn.erb_dec.onnx::Conv_309", (size_t)kC * 3);
    w->erb_out_b     = L.scalar("dfn.erb_dec.onnx::Conv_310");

    // df_dec
    w->df_lin_in     = L.grp("dfn.df_dec.df_gru.linear_in.0.weight", 8, 64, 32);
    w->df_gru1       = L.gru("dfn.df_dec.df_gru.gru.gru", kH);
    w->df_gru2       = L.gru("dfn.df_dec.df_gru.gru.gru_1", kH);
    w->df_skip       = L.grp("dfn.df_dec.df_skip.weight", 16, 32, 16);
    w->df_out        = L.grp("dfn.df_dec.df_out.0.weight", 16, 16, 60);
    w->df_alpha_w    = L.get("dfn.df_dec.onnx::MatMul_321", kH);
    w->df_alpha_b    = L.scalar("dfn.df_dec.df_fc_a.0.bias");
    w->df_convp_w    = L.get("dfn.df_dec.df_convp.1.weight", (size_t)10 * 32 * 5);
    w->df_convp_pw   = L.get("dfn.df_dec.onnx::Conv_268", 100);
    w->df_convp_pw_b = L.get("dfn.df_dec.onnx::Conv_269", 10);

    if (!L.ok) {
        delete w;
        return nullptr;
    }
    return w;
}

void dfn_fast_free_weights(dfn_fast_weights* w) { delete w; }

dfn_fast_scratch* dfn_fast_scratch_create() {
    auto* s = new dfn_fast_scratch();
    s->pad.assign((size_t)3 * (kSpec + 2), 0.0f);
    s->e0.assign((size_t)kC * 32, 0.0f);
    s->e1.assign((size_t)kC * 16, 0.0f);
    s->e2.assign((size_t)kC * 8, 0.0f);
    s->e3.assign((size_t)kC * 8, 0.0f);
    s->c0pre.assign((size_t)kC * kSpec, 0.0f);
    s->c0.assign((size_t)kC * kSpec, 0.0f);
    s->dff.assign((size_t)kC * 48, 0.0f);
    s->tmp.assign((size_t)kC * kSpec, 0.0f);
    s->flat.assign((size_t)48 * kC, 0.0f);
    s->emb_in.assign(512, 0.0f);
    s->combined.assign(512, 0.0f);
    s->x256.assign(kH, 0.0f);
    s->emb.assign(512, 0.0f);
    s->xlin.assign(kGates, 0.0f);
    s->hlin.assign(kGates, 0.0f);
    s->gru_out.assign(512, 0.0f);
    s->dec_a.assign((size_t)kC * 32, 0.0f);
    s->dec_b.assign((size_t)kC * 32, 0.0f);
    s->skip_add.assign(kH, 0.0f);
    s->df_out.assign(960, 0.0f);
    s->convp_pre.assign((size_t)10 * kSpec, 0.0f);
    s->convp.assign((size_t)10 * kSpec, 0.0f);
    s->rows.assign(160, nullptr);
    s->xq.assign(kH, 0);
    s->hq.assign(kH, 0);
    s->xs.assign(kH / kQBlock, 0.0f);
    s->hs.assign(kH / kQBlock, 0.0f);
    return s;
}

void dfn_fast_scratch_free(dfn_fast_scratch* s) { delete s; }

void dfn_fast_forward(const dfn_fast_weights* w, dfn_fast_scratch* s, const dfn_fast_io& io) {
    float* h_enc  = io.gru_h + 0 * kH;
    float* h_erb1 = io.gru_h + 1 * kH;
    float* h_erb2 = io.gru_h + 2 * kH;
    float* h_df1  = io.gru_h + 3 * kH;
    float* h_df2  = io.gru_h + 4 * kH;

    // ═══ Encoder ════════════════════════════════════════════════
    // ERB branch (build_enc_erb_branch): feat_erb → e0 → e1 → e2 → e3
    conv_tf3x3(w->enc_conv0, io.feat_erb, kErb, s, s->e0.data(), 0, kC, /*relu=*/true);
    dw_pw_block(w->enc_erb1, s->e0.data(), 32, 2, s, s->e1.data()); // 32 → 16
    dw_pw_block(w->enc_erb2, s->e1.data(), 16, 2, s, s->e2.data()); // 16 → 8
    dw_pw_block(w->enc_erb3, s->e2.data(), 8, 1, s, s->e3.data());  //  8 → 8

    // DF branch (build_enc_df_branch): feat_spec → c0 → df_feat.
    // df_conv0 is group=2: re → oc 0..31, im → oc 32..63; no bias.
    conv_tf3x3(w->enc_df_conv0, io.feat_spec + 0 * 3 * kSpec, kSpec, s, s->c0pre.data(), 0, 32, false);
    conv_tf3x3(w->enc_df_conv0, io.feat_spec + 1 * 3 * kSpec, kSpec, s, s->c0pre.data(), 32, kC, false);
    pointwise(w->enc_df_pw.data(), w->enc_df_pw_b.data(), s->c0pre.data(), kSpec, s, s->c0.data(), /*relu=*/true);
    dw_pw_block(w->enc_df_conv1, s->c0.data(), kSpec, 2, s, s->dff.data()); // 96 → 48
    std::memcpy(io.c0_out, s->c0.data(), (size_t)kC * kSpec * sizeof(float));

    // Combine (build_enc_combine): df_feat (64×48) → freq-major flat →
    // GroupedLinear(32, 96→16) → relu → df_emb;  e3 (64×8) → flat;  add.
    for (int f = 0; f < 48; ++f)
        for (int c = 0; c < kC; ++c) s->flat[(size_t)f * kC + c] = s->dff[(size_t)c * 48 + f];
    grouped_lin(w->enc_fc_emb, s->flat.data(), s->emb_in.data(), /*relu=*/true);
    for (int f = 0; f < 8; ++f)
        for (int c = 0; c < kC; ++c)
            s->combined[(size_t)f * kC + c] = s->e3[(size_t)c * 8 + f] + s->emb_in[(size_t)f * kC + c];

    // emb_gru (build_enc_emb_gru): linear_in → relu → GRU → linear_out → relu
    grouped_lin(w->enc_lin_in, s->combined.data(), s->x256.data(), true);
    gru_step(w->enc_gru, s, s->x256.data(), h_enc, s->xlin.data(), s->hlin.data());
    grouped_lin(w->enc_lin_out, h_enc, s->emb.data(), true); // emb (512)

    // ═══ erb_dec ════════════════════════════════════════════════
    grouped_lin(w->erb_lin_in, s->emb.data(), s->x256.data(), true);
    gru_step(w->erb_gru1, s, s->x256.data(), h_erb1, s->xlin.data(), s->hlin.data());
    gru_step(w->erb_gru2, s, h_erb1, h_erb2, s->xlin.data(), s->hlin.data());
    grouped_lin(w->erb_lin_out, h_erb2, s->gru_out.data(), true); // emb_gru_out (512)

    // build_emb_gru_reshape_to_feat: gru_feat[c][f] = out[f*64 + c] (F=8)
    float* dec = s->dec_a.data();
    float* alt = s->dec_b.data();
    for (int c = 0; c < kC; ++c)
        for (int f = 0; f < 8; ++f) dec[(size_t)c * 8 + f] = s->gru_out[(size_t)f * kC + c];
    // + e3-skip, convt3 (dw 1×3, s=1) → 64×8
    skip_add_into(w->erb_skip3, s->e3.data(), 8, dec);
    dw_pw_block(w->erb_convt3, dec, 8, 1, s, alt);
    std::swap(dec, alt);
    // + e2-skip, convt2 (convT s2 k3: 8 → 16) → pw → relu
    skip_add_into(w->erb_skip2, s->e2.data(), 8, dec);
    dw_convT_s2k3(w->erb_convt2_dw.data(), dec, 8, s->tmp.data());
    pointwise(w->erb_convt2_pw.data(), w->erb_convt2_b.data(), s->tmp.data(), 16, s, alt, true);
    std::swap(dec, alt);
    // + e1-skip, convt1 (16 → 32) → pw → relu
    skip_add_into(w->erb_skip1, s->e1.data(), 16, dec);
    dw_convT_s2k3(w->erb_convt1_dw.data(), dec, 16, s->tmp.data());
    pointwise(w->erb_convt1_pw.data(), w->erb_convt1_b.data(), s->tmp.data(), 32, s, alt, true);
    std::swap(dec, alt);
    // + e0-skip, conv0_out (1×3 conv, 64 → 1) → sigmoid → mask (build_erb_dec_m)
    skip_add_into(w->erb_skip0, s->e0.data(), 32, dec);
    for (int f = 0; f < kErb; ++f) io.mask[f] = w->erb_out_b;
    for (int ic = 0; ic < kC; ++ic) {
        const float* xc = dec + (size_t)ic * kErb;
        const float  w0 = w->erb_out_w[ic * 3 + 0], w1 = w->erb_out_w[ic * 3 + 1], w2 = w->erb_out_w[ic * 3 + 2];
        for (int f = 0; f < kErb; ++f) {
            float acc = w1 * xc[f];
            if (f > 0) acc += w0 * xc[f - 1];
            if (f + 1 < kErb) acc += w2 * xc[f + 1];
            io.mask[f] += acc;
        }
    }
    sigmoid_vec(kErb, io.mask, io.mask);

    // ═══ df_dec ═════════════════════════════════════════════════
    // build_df_dec_no_convp
    grouped_lin(w->df_lin_in, s->emb.data(), s->x256.data(), true);
    gru_step(w->df_gru1, s, s->x256.data(), h_df1, s->xlin.data(), s->hlin.data());
    gru_step(w->df_gru2, s, h_df1, h_df2, s->xlin.data(), s->hlin.data());
    grouped_lin(w->df_skip, s->emb.data(), s->skip_add.data(), false);
    for (int k = 0; k < kH; ++k) s->skip_add[k] += h_df2[k];
    {
        float a = w->df_alpha_b;
        for (int k = 0; k < kH; ++k) a += w->df_alpha_w[k] * s->skip_add[k];
        io.alpha[0] = sigmoid_s(a);
    }
    grouped_lin(w->df_out, s->skip_add.data(), s->df_out.data(), false);
    tanh_vec(960, s->df_out.data(), s->df_out.data());

    // build_df_convp: causal 5-tap time conv over [c0_hist(4), c0], group=2
    // (oc 0..4 ← ic 0..31, oc 5..9 ← ic 32..63), then 1×1 10→10 + bias + relu.
    // Weight [oc][icl][kt] is a K = 32·5 row per oc; the B rows are the
    // matching (channel, tap) rows of c0_hist / c0.
    for (int g = 0; g < 2; ++g) {
        const float** rows = s->rows.data();
        for (int icl = 0; icl < 32; ++icl) {
            const int ic = 32 * g + icl;
            for (int kt = 0; kt < 4; ++kt) rows[icl * 5 + kt] = io.c0_hist + ((size_t)ic * 4 + kt) * kSpec;
            rows[icl * 5 + 4] = s->c0.data() + (size_t)ic * kSpec;
        }
        gemm_rows(5, kSpec, 160, w->df_convp_w.data() + (size_t)(5 * g) * 160, 160, rows, nullptr,
                  s->convp_pre.data() + (size_t)(5 * g) * kSpec, kSpec, false, false);
    }
    {
        const float** rows = s->rows.data();
        for (int ic = 0; ic < 10; ++ic) rows[ic] = s->convp_pre.data() + (size_t)ic * kSpec;
        gemm_rows(10, kSpec, 10, w->df_convp_pw.data(), 10, rows, w->df_convp_pw_b.data(), s->convp.data(), kSpec,
                  false, true);
    }
    // coefs[f][j] = tanh_out[f*10 + j] + convp[j][f]
    for (int f = 0; f < kSpec; ++f)
        for (int j = 0; j < 10; ++j)
            io.coefs[(size_t)f * 10 + j] = s->df_out[(size_t)f * 10 + j] + s->convp[(size_t)j * kSpec + f];
}
