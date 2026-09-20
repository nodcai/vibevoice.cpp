#include "acoustic_decoder_v2.hpp"

#include "backend.hpp"
#include "bench.hpp"
#include "common.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vv {

namespace {

constexpr int kTaps = 7;   // stem / head / mixer kernel width
constexpr int kCtx  = kTaps - 1;

// Read any float-ish tensor (F32 / F16 / quantized) into host f32.
std::vector<float> read_f32(const struct ggml_tensor* t) {
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        const auto* tt = ggml_get_type_traits(t->type);
        tt->to_float(raw.data(), out.data(), static_cast<int64_t>(n));
    }
    return out;
}

void write_f16(struct ggml_tensor* dst, const std::vector<float>& src) {
    std::vector<ggml_fp16_t> tmp(src.size());
    for (size_t i = 0; i < src.size(); ++i) tmp[i] = ggml_fp32_to_fp16(src[i]);
    ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size() * sizeof(ggml_fp16_t));
}

// Dense kernel [K, Ci, Co] (ggml ne) → tap-major planes [Ci, Co, K].
void relay_dense(const struct ggml_tensor* src_t, struct ggml_tensor* dst_t) {
    const int K  = static_cast<int>(src_t->ne[0]);
    const int Ci = static_cast<int>(src_t->ne[1]);
    const int Co = static_cast<int>(src_t->ne[2]);
    const auto src = read_f32(src_t);
    std::vector<float> dst(static_cast<size_t>(Ci) * Co * K);
    for (int k = 0; k < K; ++k)
        for (int ci = 0; ci < Ci; ++ci)
            for (int co = 0; co < Co; ++co)
                dst[static_cast<size_t>(k) * Ci * Co + static_cast<size_t>(co) * Ci + ci] =
                    src[static_cast<size_t>(co) * Ci * K + static_cast<size_t>(ci) * K + k];
    write_f16(dst_t, dst);
}

// Transposed kernel [K, Co, Ci] → [Ci, K·Co] with row r = k·Co + co.
void relay_transposed(const struct ggml_tensor* src_t, struct ggml_tensor* dst_t) {
    const int K  = static_cast<int>(src_t->ne[0]);
    const int Co = static_cast<int>(src_t->ne[1]);
    const int Ci = static_cast<int>(src_t->ne[2]);
    const auto src = read_f32(src_t);
    std::vector<float> dst(static_cast<size_t>(Ci) * K * Co);
    for (int ci = 0; ci < Ci; ++ci)
        for (int co = 0; co < Co; ++co)
            for (int k = 0; k < K; ++k)
                dst[(static_cast<size_t>(k) * Co + co) * Ci + ci] =
                    src[static_cast<size_t>(ci) * K * Co + static_cast<size_t>(co) * K + k];
    write_f16(dst_t, dst);
}

// Depthwise kernel [K, 1, C] → [C, K] (tap plane k at offset k·C).
void relay_depthwise(const struct ggml_tensor* src_t, struct ggml_tensor* dst_t) {
    const int K = static_cast<int>(src_t->ne[0]);
    const int C = static_cast<int>(src_t->ne[2]);
    const auto src = read_f32(src_t);
    std::vector<float> dst(static_cast<size_t>(C) * K);
    for (int c = 0; c < C; ++c)
        for (int k = 0; k < K; ++k)
            dst[static_cast<size_t>(k) * C + c] = src[static_cast<size_t>(c) * K + k];
    ggml_backend_tensor_set(dst_t, dst.data(), 0, dst.size() * sizeof(float));
}

struct ggml_tensor* rms_norm_c(struct ggml_context* g, struct ggml_tensor* x,
                               struct ggml_tensor* w, float eps) {
    x = ggml_rms_norm(g, x, eps);
    if (w) x = ggml_mul(g, x, w);
    return x;
}

}  // namespace

DecoderV2Weights::~DecoderV2Weights() { free(); }
void DecoderV2Weights::free() {
    if (buf) ggml_backend_buffer_free(buf);
    if (ctx) ggml_free(ctx);
    buf = nullptr; ctx = nullptr; ready = false;
}

DecoderV2State::~DecoderV2State() { free(); }
void DecoderV2State::free() {
    if (buf) ggml_backend_buffer_free(buf);
    if (ctx) ggml_free(ctx);
    buf = nullptr; ctx = nullptr; ready = false;
}

bool decoder_v2_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("VIBEVOICE_VAE_V2");
        return !(e && e[0] == '0');
    }();
    return on;
}

bool decoder_v2_prepare(const DecoderWeights& w, const AcousticConfig& cfg,
                        DecoderV2Weights* out) {
    if (!out || out->ready) return out && out->ready;
    if (!w.stem.kernel || !w.head.kernel) return false;
    if (w.stem.kernel->ne[0] != kTaps || w.head.kernel->ne[0] != kTaps) {
        VV_LOG_WARN("decoder_v2: stem/head kernel width %d/%d != %d - unsupported",
                    static_cast<int>(w.stem.kernel->ne[0]),
                    static_cast<int>(w.head.kernel->ne[0]), kTaps);
        return false;
    }

    const int n_stages = static_cast<int>(cfg.depths.size());
    if (n_stages < 1 || w.ups.size() + 1 != static_cast<size_t>(n_stages) ||
        w.stages.size() != static_cast<size_t>(n_stages)) {
        VV_LOG_WARN("decoder_v2: inconsistent stage count");
        return false;
    }

    out->n_stages = n_stages;
    out->c_in     = static_cast<int>(w.stem.kernel->ne[1]);
    out->C.assign(n_stages, 0);
    out->depth.assign(n_stages, 0);
    out->stride.assign(n_stages, 0);
    out->C[0] = static_cast<int>(w.stem.kernel->ne[2]);
    for (int s = 1; s < n_stages; ++s) {
        const auto& up = w.ups[s - 1];
        out->stride[s] = up.stride;
        out->C[s]      = static_cast<int>(up.kernel->ne[1]);
        if (up.kernel->ne[0] != 2 * up.stride) {
            VV_LOG_WARN("decoder_v2: stage %d kernel %d != 2*stride %d - unsupported",
                        s, static_cast<int>(up.kernel->ne[0]), up.stride);
            return false;
        }
        if (up.kernel->ne[2] != out->C[s - 1]) {
            VV_LOG_WARN("decoder_v2: stage %d C_in mismatch", s);
            return false;
        }
    }
    int n_blocks = 0;
    for (int s = 0; s < n_stages; ++s) {
        out->depth[s] = static_cast<int>(w.stages[s].size());
        n_blocks += out->depth[s];
    }
    if (w.head.kernel->ne[1] != out->C[n_stages - 1] || w.head.kernel->ne[2] != 1) {
        VV_LOG_WARN("decoder_v2: head shape mismatch");
        return false;
    }
    for (int s = 0; s < n_stages; ++s)
        for (const auto& b : w.stages[s])
            if (!b.mixer_kernel || b.mixer_kernel->ne[0] != kTaps ||
                b.mixer_kernel->ne[2] != out->C[s]) {
                VV_LOG_WARN("decoder_v2: block mixer kernel shape mismatch in stage %d", s);
                return false;
            }

    // ---- allocate re-laid tensors ----
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * static_cast<size_t>(4 + n_stages + n_blocks);
    ip.no_alloc = true;
    out->ctx = ggml_init(ip);
    if (!out->ctx) return false;

    out->stem_w = ggml_new_tensor_3d(out->ctx, GGML_TYPE_F16, out->c_in, out->C[0], kTaps);
    out->head_w = ggml_new_tensor_3d(out->ctx, GGML_TYPE_F16, out->C[n_stages - 1], 1, kTaps);
    out->tr_w.assign(n_stages, nullptr);
    out->tr_b.assign(n_stages, nullptr);
    for (int s = 1; s < n_stages; ++s) {
        out->tr_w[s] = ggml_new_tensor_2d(out->ctx, GGML_TYPE_F16, out->C[s - 1],
                                          static_cast<int64_t>(2) * out->stride[s] * out->C[s]);
        out->tr_b[s] = w.ups[s - 1].bias;
    }
    out->dw_w.clear();
    out->blocks.clear();
    for (int s = 0; s < n_stages; ++s)
        for (const auto& b : w.stages[s]) {
            out->dw_w.push_back(ggml_new_tensor_2d(out->ctx, GGML_TYPE_F32, out->C[s], kTaps));
            out->blocks.push_back(&b);
        }
    out->buf = ggml_backend_alloc_ctx_tensors(out->ctx, vv::backend());
    if (!out->buf) {
        VV_LOG_ERROR("decoder_v2: weight buffer allocation failed");
        out->free();
        return false;
    }

    BenchClock clk;
    relay_dense(w.stem.kernel, out->stem_w);
    relay_dense(w.head.kernel, out->head_w);
    for (int s = 1; s < n_stages; ++s) relay_transposed(w.ups[s - 1].kernel, out->tr_w[s]);
    for (size_t i = 0; i < out->blocks.size(); ++i)
        relay_depthwise(out->blocks[i]->mixer_kernel, out->dw_w[i]);

    out->stem_b     = w.stem.bias;
    out->head_b     = w.head.bias;
    out->final_norm = w.final_norm;
    out->ready      = true;
    VV_LOG_INFO("decoder_v2: ready (%d stages, %d blocks, %.1f MB re-laid, %.0f ms)",
                n_stages, n_blocks,
                static_cast<double>(ggml_backend_buffer_get_size(out->buf)) / 1e6, clk.ms());
    return true;
}

bool decoder_v2_state_init(const DecoderV2Weights& w, DecoderV2State* st) {
    if (!w.ready || !st) return false;
    st->free();
    const int n_blocks = static_cast<int>(w.blocks.size());
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * static_cast<size_t>(4 + w.n_stages + n_blocks);
    ip.no_alloc = true;
    st->ctx = ggml_init(ip);
    if (!st->ctx) return false;
    st->stem_state = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, w.c_in, kCtx);
    st->head_state = ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, w.C[w.n_stages - 1], kCtx);
    st->dw_state.clear();
    for (int s = 0; s < w.n_stages; ++s)
        for (int b = 0; b < w.depth[s]; ++b)
            st->dw_state.push_back(ggml_new_tensor_2d(st->ctx, GGML_TYPE_F32, w.C[s], kCtx));
    st->tr_state.assign(w.n_stages, nullptr);
    for (int s = 1; s < w.n_stages; ++s)
        st->tr_state[s] = ggml_new_tensor_3d(st->ctx, GGML_TYPE_F32, w.C[s], w.stride[s], 1);
    st->buf = ggml_backend_alloc_ctx_tensors(st->ctx, vv::backend());
    if (!st->buf) {
        VV_LOG_ERROR("decoder_v2: state buffer allocation failed");
        st->free();
        return false;
    }
    ggml_backend_buffer_clear(st->buf, 0);
    st->ready = true;
    return true;
}

void decoder_v2_state_reset(DecoderV2State& st) {
    if (st.buf) ggml_backend_buffer_clear(st.buf, 0);
}

namespace {

// One chunk of T latent frames. Reads and rewrites the conv states in-graph.
struct ggml_tensor* build_chunk(struct ggml_context* g,
                                const DecoderV2Weights& w, DecoderV2State& st,
                                struct ggml_tensor* inp, int T, float eps,
                                std::vector<struct ggml_tensor*>* state_writes) {
    // Concatenate the persistent left context in front of the chunk and
    // schedule its update from the chunk's tail.
    auto with_context = [&](struct ggml_tensor* x, struct ggml_tensor* state) {
        const int Tx = static_cast<int>(x->ne[1]);
        struct ggml_tensor* xin  = ggml_concat(g, state, x, 1);               // [C, 6+T]
        struct ggml_tensor* tail = ggml_view_2d(g, xin, xin->ne[0], kCtx, xin->nb[1],
                                                static_cast<size_t>(Tx) * xin->nb[1]);
        state_writes->push_back(ggml_cpy(g, tail, state));
        return xin;
    };
    auto shift = [&](struct ggml_tensor* xin, int k, int Tx) {
        return ggml_view_2d(g, xin, xin->ne[0], Tx, xin->nb[1],
                            static_cast<size_t>(k) * xin->nb[1]);
    };
    // Dense causal K=7 conv via tap-major planes: y = Σ_k W_k · shift_k(xin).
    auto dense7 = [&](struct ggml_tensor* xin, struct ggml_tensor* wt, struct ggml_tensor* bias, int Tx) {
        struct ggml_tensor* y = nullptr;
        for (int k = 0; k < kTaps; ++k) {
            struct ggml_tensor* wk = ggml_view_2d(g, wt, wt->ne[0], wt->ne[1], wt->nb[1],
                                                  static_cast<size_t>(k) * wt->nb[2]);
            struct ggml_tensor* m = ggml_mul_mat(g, wk, ggml_cont(g, shift(xin, k, Tx)));
            y = y ? ggml_add(g, y, m) : m;
        }
        if (bias) y = ggml_add(g, y, bias);
        return y;
    };

    // ---- stem ----
    struct ggml_tensor* h = dense7(with_context(inp, st.stem_state), w.stem_w, w.stem_b, T);

    int bi = 0;
    for (int s = 0; s < w.n_stages; ++s) {
        if (s > 0) {
            // Transposed conv, K = 2·stride: mul_mat then overlap-add of the
            // second half onto the next frame's first half. The last frame's
            // second half is carried to the next chunk (and trimmed at the end
            // of the stream, which is the causal trim_right = K - stride).
            const int Tc = static_cast<int>(h->ne[1]);
            const int sr = w.stride[s];
            const int Co = w.C[s];
            struct ggml_tensor* y  = ggml_mul_mat(g, w.tr_w[s], h);               // [2s·Co, Tc]
            y = ggml_reshape_3d(g, y, Co, 2 * sr, Tc);
            struct ggml_tensor* A  = ggml_view_3d(g, y, Co, sr, Tc, y->nb[1], y->nb[2], 0);
            struct ggml_tensor* B  = ggml_view_3d(g, y, Co, sr, Tc, y->nb[1], y->nb[2],
                                                  static_cast<size_t>(sr) * y->nb[1]);
            struct ggml_tensor* Bc = ggml_cont(g, B);                               // [Co, s, Tc]
            struct ggml_tensor* Bprev;
            if (Tc > 1) {
                struct ggml_tensor* Bhead = ggml_view_3d(g, Bc, Co, sr, Tc - 1, Bc->nb[1], Bc->nb[2], 0);
                Bprev = ggml_concat(g, st.tr_state[s], Bhead, 2);                 // [Co, s, Tc]
            } else {
                Bprev = st.tr_state[s];
            }
            struct ggml_tensor* Blast = ggml_view_3d(g, Bc, Co, sr, 1, Bc->nb[1], Bc->nb[2],
                                                     static_cast<size_t>(Tc - 1) * Bc->nb[2]);
            state_writes->push_back(ggml_cpy(g, Blast, st.tr_state[s]));
            struct ggml_tensor* out = ggml_add(g, ggml_cont(g, A), Bprev);
            h = ggml_reshape_2d(g, out, Co, static_cast<int64_t>(sr) * Tc);
            if (w.tr_b[s]) h = ggml_add(g, h, w.tr_b[s]);
        }
        for (int b = 0; b < w.depth[s]; ++b, ++bi) {
            const Block1DWeights& blk = *w.blocks[bi];
            const int Tc = static_cast<int>(h->ne[1]);
            // mixer: norm → depthwise causal conv → γ → residual
            struct ggml_tensor* x   = rms_norm_c(g, h, blk.norm, eps);
            struct ggml_tensor* xin = with_context(x, st.dw_state[bi]);
            struct ggml_tensor* conv = nullptr;
            for (int k = 0; k < kTaps; ++k) {
                struct ggml_tensor* wk = ggml_view_1d(g, w.dw_w[bi], w.C[s],
                                                      static_cast<size_t>(k) * w.dw_w[bi]->nb[1]);
                struct ggml_tensor* m = ggml_mul(g, shift(xin, k, Tc), wk);
                conv = conv ? ggml_add(g, conv, m) : m;
            }
            if (blk.mixer_bias) conv = ggml_add(g, conv, blk.mixer_bias);
            if (blk.gamma)      conv = ggml_mul(g, conv, blk.gamma);
            h = ggml_add(g, h, conv);
            // ffn: norm → up → GELU → down → γ → residual
            x = rms_norm_c(g, h, blk.ffn_norm, eps);
            x = ggml_mul_mat(g, blk.ffn_linear1, x);
            if (blk.ffn_linear1_b) x = ggml_add(g, x, blk.ffn_linear1_b);
            x = ggml_gelu(g, x);
            x = ggml_mul_mat(g, blk.ffn_linear2, x);
            if (blk.ffn_linear2_b) x = ggml_add(g, x, blk.ffn_linear2_b);
            if (blk.ffn_gamma)     x = ggml_mul(g, x, blk.ffn_gamma);
            h = ggml_add(g, h, x);
        }
    }

    if (w.final_norm) h = rms_norm_c(g, h, w.final_norm, eps);

    // ---- head: dense causal K=7, C_last → 1 ----
    const int Th = static_cast<int>(h->ne[1]);
    struct ggml_tensor* y = dense7(with_context(h, st.head_state), w.head_w, w.head_b, Th);
    return y;  // [1, T_audio]
}

}  // namespace

bool decoder_v2_decode(const DecoderV2Weights& w, DecoderV2State& st,
                       const float* latents, int n_frames,
                       std::vector<float>* audio_out, int chunk_frames) {
    if (!w.ready || !st.ready || !latents || !audio_out || n_frames <= 0) return false;
    if (chunk_frames <= 0) chunk_frames = 32;
    constexpr float kEps = 1e-5f;

    for (int f0 = 0; f0 < n_frames; f0 += chunk_frames) {
        const int T = std::min(chunk_frames, n_frames - f0);
        BenchClock clk;

        // Input lives in its own small allocated context so the graph
        // intermediates can be planned by gallocr with reuse.
        struct ggml_init_params ipi {};
        ipi.mem_size = ggml_tensor_overhead() * 4;
        ipi.no_alloc = true;
        struct ggml_context* ctx_in = ggml_init(ipi);
        struct ggml_tensor* inp = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, w.c_in, T);
        ggml_set_name(inp, "dec_v2_latent");
        ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx_in);
        if (!in_buf) { ggml_free(ctx_in); return false; }
        ggml_backend_tensor_set(inp, latents + static_cast<size_t>(f0) * w.c_in, 0,
                                sizeof(float) * static_cast<size_t>(w.c_in) * T);

        struct ggml_init_params ipg {};
        ipg.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
        ipg.no_alloc = true;
        struct ggml_context* g = ggml_init(ipg);
        std::vector<struct ggml_tensor*> writes;
        struct ggml_tensor* y = build_chunk(g, w, st, inp, T, kEps, &writes);
        ggml_set_name(y, "dec_v2_audio");
        struct ggml_cgraph* gf = ggml_new_graph_custom(g, 16384, false);
        ggml_build_forward_expand(gf, y);
        // State updates go after the output so every read precedes its write.
        for (auto* wr : writes) ggml_build_forward_expand(gf, wr);

        const bool ok = vv::compute_graph(gf);
        if (ok) {
            const size_t n = static_cast<size_t>(ggml_nelements(y));
            const size_t off = audio_out->size();
            audio_out->resize(off + n);
            ggml_backend_tensor_get(y, audio_out->data() + off, 0, sizeof(float) * n);
        }
        if (bench_enabled())
            std::fprintf(stderr, "[bench]   decoder_v2 chunk %d frames, %d nodes: %.1f ms\n",
                         T, ggml_graph_n_nodes(gf), clk.ms());
        ggml_free(g);
        ggml_backend_buffer_free(in_buf);
        ggml_free(ctx_in);
        if (!ok) {
            VV_LOG_ERROR("decoder_v2: compute failed");
            return false;
        }
    }
    return true;
}

}  // namespace vv
