// Stateful decoder (acoustic_decoder_v2) parity, self-contained.
//
// Builds a small random decoder (fp16-representable weights so both paths
// see identical numbers), then checks:
//   1. decoder_v2 whole-sequence == legacy decoder_forward (single shot)
//   2. decoder_v2 chunked (odd chunk size, continuing state) == whole
//   3. state reset gives the same output again
// No model files needed; runs in CI on whatever backend is active.
#include "acoustic_decoder_v2.hpp"
#include "acoustic_tokenizer.hpp"
#include "backend.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Rng {
    std::mt19937 gen{42};
    std::normal_distribution<float> nd{0.f, 1.f};
    // Value rounded through fp16 so F16 and F32 storage agree exactly.
    float next(float scale) {
        return ggml_fp16_to_fp32(ggml_fp32_to_fp16(nd(gen) * scale));
    }
};

void fill(struct ggml_tensor* t, Rng& rng, float scale, float offset = 0.f) {
    std::vector<float> v(static_cast<size_t>(ggml_nelements(t)));
    for (auto& x : v) x = rng.next(scale) + offset;
    ggml_backend_tensor_set(t, v.data(), 0, sizeof(float) * v.size());
}

double rel_rmse(const std::vector<float>& a, const std::vector<float>& b, double* max_abs) {
    double sd = 0, sr = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        sd += d * d; sr += static_cast<double>(a[i]) * a[i];
        m = std::max(m, std::fabs(d));
    }
    if (max_abs) *max_abs = m;
    return std::sqrt(sd / a.size()) / std::max(std::sqrt(sr / a.size()), 1e-12);
}

}  // namespace

int main() {
    vv::AcousticConfig cfg;
    cfg.channels = 1;
    cfg.vae_dim  = 8;
    cfg.ratios   = {2, 3};
    cfg.depths   = {2, 1, 1};
    cfg.eps      = 1e-5f;
    const std::vector<int> C = {16, 8, 4};
    const int n_stages = 3;

    // ---- random weights on the active backend ----
    struct ggml_init_params ip {};
    ip.mem_size = ggml_tensor_overhead() * 256;
    ip.no_alloc = true;
    struct ggml_context* wctx = ggml_init(ip);

    vv::DecoderWeights dw;
    auto t1 = [&](int n)             { return ggml_new_tensor_1d(wctx, GGML_TYPE_F32, n); };
    auto t2 = [&](int a, int b)      { return ggml_new_tensor_2d(wctx, GGML_TYPE_F32, a, b); };
    auto t3 = [&](int a, int b, int c) { return ggml_new_tensor_3d(wctx, GGML_TYPE_F32, a, b, c); };

    dw.stem.kernel = t3(7, cfg.vae_dim, C[0]); dw.stem.bias = t1(C[0]); dw.stem.stride = 1;
    dw.ups.resize(2);
    dw.ups[0].kernel = t3(2 * cfg.ratios[0], C[1], C[0]); dw.ups[0].bias = t1(C[1]); dw.ups[0].stride = cfg.ratios[0];
    dw.ups[1].kernel = t3(2 * cfg.ratios[1], C[2], C[1]); dw.ups[1].bias = t1(C[2]); dw.ups[1].stride = cfg.ratios[1];
    dw.stages.assign(n_stages, {});
    for (int s = 0; s < n_stages; ++s) {
        dw.stages[s].resize(cfg.depths[s]);
        for (auto& b : dw.stages[s]) {
            const int c = C[s];
            b.norm = t1(c); b.mixer_kernel = t3(7, 1, c); b.mixer_bias = t1(c); b.gamma = t1(c);
            b.ffn_norm = t1(c); b.ffn_linear1 = t2(c, 4 * c); b.ffn_linear1_b = t1(4 * c);
            b.ffn_linear2 = t2(4 * c, c); b.ffn_linear2_b = t1(c); b.ffn_gamma = t1(c);
        }
    }
    dw.final_norm = nullptr;
    dw.head.kernel = t3(7, C[2], 1); dw.head.bias = t1(1); dw.head.stride = 1;

    ggml_backend_buffer_t wbuf = vv::allocate_ctx_tensors(wctx);
    if (!wbuf) { std::fprintf(stderr, "FAIL: weight alloc\n"); return 1; }

    Rng rng;
    auto fill_conv = [&](vv::StridedConvWeights& w, int fan_in) {
        fill(w.kernel, rng, 0.8f / std::sqrt(static_cast<float>(fan_in)));
        fill(w.bias, rng, 0.05f);
    };
    fill_conv(dw.stem, 7 * cfg.vae_dim);
    fill_conv(dw.ups[0], C[0]);
    fill_conv(dw.ups[1], C[1]);
    fill_conv(dw.head, 7 * C[2]);
    for (int s = 0; s < n_stages; ++s)
        for (auto& b : dw.stages[s]) {
            const int c = C[s];
            fill(b.norm, rng, 0.1f, 1.0f); fill(b.ffn_norm, rng, 0.1f, 1.0f);
            fill(b.mixer_kernel, rng, 0.3f); fill(b.mixer_bias, rng, 0.05f);
            fill(b.gamma, rng, 0.2f, 0.5f); fill(b.ffn_gamma, rng, 0.2f, 0.5f);
            fill(b.ffn_linear1, rng, 0.8f / std::sqrt(static_cast<float>(c)));
            fill(b.ffn_linear1_b, rng, 0.05f);
            fill(b.ffn_linear2, rng, 0.8f / std::sqrt(4.0f * c));
            fill(b.ffn_linear2_b, rng, 0.05f);
        }

    // ---- random latents ----
    const int T = 23;
    std::vector<float> lat(static_cast<size_t>(T) * cfg.vae_dim);
    for (auto& v : lat) v = rng.next(1.0f);
    const int up_total = cfg.ratios[0] * cfg.ratios[1];

    // ---- legacy single-shot ----
    std::vector<float> ref;
    bool have_ref = false;
    // The legacy graph uses a left ggml_pad_ext, which GPU backends may not
    // implement (Metal aborts on it), so only compare on the CPU backend.
    if (ggml_backend_is_cpu(vv::backend())) {
        std::vector<float> packed(lat.size());
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < cfg.vae_dim; ++d)
                packed[static_cast<size_t>(d) * T + t] = lat[static_cast<size_t>(t) * cfg.vae_dim + d];
        struct ggml_init_params gp {};
        gp.mem_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
        gp.no_alloc = true;
        struct ggml_context* g = ggml_init(gp);
        struct ggml_tensor* z = ggml_new_tensor_3d(g, GGML_TYPE_F32, T, cfg.vae_dim, 1);
        struct ggml_tensor* y = vv::decoder_forward(g, z, dw, cfg);
        struct ggml_cgraph* gf = ggml_new_graph_custom(g, 8192, false);
        ggml_build_forward_expand(gf, y);
        ggml_backend_buffer_t buf = vv::allocate_ctx_tensors(g);
        if (buf) {
            ggml_backend_tensor_set(z, packed.data(), 0, sizeof(float) * packed.size());
            if (vv::compute_graph(gf)) {
                ref.assign(static_cast<size_t>(ggml_nelements(y)), 0.f);
                ggml_backend_tensor_get(y, ref.data(), 0, sizeof(float) * ref.size());
                have_ref = true;
            }
            ggml_backend_buffer_free(buf);
        }
        ggml_free(g);
    }
    if (!have_ref)
        std::fprintf(stderr, "note: legacy decoder not run on backend %s; skipping legacy comparison\n",
                     vv::backend_name());

    // ---- decoder_v2 ----
    vv::DecoderV2Weights v2;
    if (!vv::decoder_v2_prepare(dw, cfg, &v2)) { std::fprintf(stderr, "FAIL: prepare\n"); return 2; }
    vv::DecoderV2State st;
    if (!vv::decoder_v2_state_init(v2, &st)) { std::fprintf(stderr, "FAIL: state init\n"); return 3; }

    std::vector<float> whole;
    if (!vv::decoder_v2_decode(v2, st, lat.data(), T, &whole, /*chunk=*/64)) {
        std::fprintf(stderr, "FAIL: whole decode\n"); return 4;
    }
    if (whole.size() != static_cast<size_t>(T) * up_total) {
        std::fprintf(stderr, "FAIL: whole len %zu != %d\n", whole.size(), T * up_total); return 5;
    }

    // chunked, continuing state, odd chunk size that does not divide T
    vv::decoder_v2_state_reset(st);
    std::vector<float> chunked;
    for (int off = 0; off < T; off += 5) {
        const int n = std::min(5, T - off);
        if (!vv::decoder_v2_decode(v2, st, lat.data() + static_cast<size_t>(off) * cfg.vae_dim, n, &chunked, 64)) {
            std::fprintf(stderr, "FAIL: chunk at %d\n", off); return 6;
        }
    }
    // internal chunking path (chunk_frames < T) after a reset
    vv::decoder_v2_state_reset(st);
    std::vector<float> internal;
    if (!vv::decoder_v2_decode(v2, st, lat.data(), T, &internal, /*chunk=*/4)) {
        std::fprintf(stderr, "FAIL: internal chunked decode\n"); return 7;
    }

    int rc = 0;
    double m = 0;
    if (chunked.size() != whole.size() || internal.size() != whole.size()) {
        std::fprintf(stderr, "FAIL: chunked len %zu / internal %zu != whole %zu\n",
                     chunked.size(), internal.size(), whole.size());
        return 8;
    }
    const double r_chunk = rel_rmse(whole, chunked, &m);
    std::printf("decoder_v2 chunked vs whole:  rel_rmse=%.3e max_abs=%.3e\n", r_chunk, m);
    if (r_chunk > 5e-4) { std::fprintf(stderr, "FAIL: chunked diverges\n"); rc = 9; }
    const double r_int = rel_rmse(whole, internal, &m);
    std::printf("decoder_v2 internal vs whole: rel_rmse=%.3e max_abs=%.3e\n", r_int, m);
    if (r_int > 5e-4) { std::fprintf(stderr, "FAIL: internal chunking diverges\n"); rc = 10; }
    if (have_ref) {
        if (ref.size() != whole.size()) {
            std::fprintf(stderr, "FAIL: legacy len %zu != v2 %zu\n", ref.size(), whole.size());
            return 11;
        }
        const double r_ref = rel_rmse(ref, whole, &m);
        std::printf("decoder_v2 vs legacy:         rel_rmse=%.3e max_abs=%.3e\n", r_ref, m);
        if (r_ref > 5e-3) { std::fprintf(stderr, "FAIL: v2 diverges from legacy decoder\n"); rc = 12; }
    }

    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    if (rc == 0) std::printf("OK (backend %s)\n", vv::backend_name());
    return rc;
}
