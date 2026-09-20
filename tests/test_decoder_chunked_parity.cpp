// Decoder parity on real weights. Gated: needs a realtime-0.5b gguf via
// VIBEVOICE_TTS_MODEL. Compares the legacy chunked streaming decoder
// (run_decoder_chunk_streaming) against decode_latent_sequence, which now
// runs the stateful decoder_v2 when enabled - so this is the legacy-vs-v2
// gate on real weights. CPU only: the legacy graph needs a left pad that
// GPU backends do not implement.
#include "vibevoice.h"
#include "vibevoice_tts.hpp"
#include "acoustic_tokenizer.hpp"
#include "backend.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
    const char* model_env = std::getenv("VIBEVOICE_TTS_MODEL");
    if (!model_env || !*model_env) { std::fprintf(stderr, "skip: set VIBEVOICE_TTS_MODEL\n"); return 77; }
    if (!ggml_backend_is_cpu(vv::backend())) { std::fprintf(stderr, "skip: legacy decoder graph is CPU-only\n"); return 77; }
    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_env, &model)) { std::fprintf(stderr, "FAIL: load\n"); return 1; }
    if (model.variant != "realtime-0.5b") { std::fprintf(stderr, "FAIL: want realtime-0.5b got %s\n", model.variant.c_str()); return 2; }

    const int latent = model.cfg.latent;
    const int N = 48;  // frames (~6.4 s), multiple of the 6-frame window
    std::mt19937 rng(1234); std::normal_distribution<float> nd(0.f,1.f);
    std::vector<float> scaled((size_t)latent*N);
    // Real scaled latents have std ~5 (raw/0.196 - bias). Unit-variance input
    // decodes to near-silence where F16 accumulation noise dominates and the
    // relative error is meaningless, so match the real scale.
    for (auto& v : scaled) v = 5.0f * nd(rng);

    // Reference: single-shot decode. decode_latent_sequence takes per-frame
    // [latent] latents (frame-major, latent-fastest) and applies the ggml-order
    // packing internally (see vibevoice_tts.cpp) — the same packing the streaming
    // path applies per chunk. We feed the identical buffer to both paths so the
    // packing is applied consistently downstream.
    std::vector<float> ref = vv::decode_latent_sequence(model.cfg, model.w, scaled.data(), N);
    if (ref.empty()) { std::fprintf(stderr, "FAIL: single-shot decode empty\n"); return 3; }

    // Streaming: 6-frame windows, shared cache.
    std::vector<float> got; got.reserve(ref.size());
    vv::StreamingCache cache; cache.is_first_chunk = true;
    const int W = 6;
    for (int off = 0; off < N; off += W) {
        const int fn = std::min(W, N - off);
        std::vector<float> win_audio;
        const bool first = (off == 0), final = (off + fn >= N);
        if (!vv::run_decoder_chunk_streaming(model.cfg, model.w, scaled.data() + (size_t)off*latent,
                                             fn, cache, first, final, &win_audio)) {
            std::fprintf(stderr, "FAIL: stream chunk at %d\n", off); return 4;
        }
        got.insert(got.end(), win_audio.begin(), win_audio.end());
    }

    if (got.size() != ref.size()) { std::fprintf(stderr, "FAIL: len stream=%zu single=%zu\n", got.size(), ref.size()); return 5; }
    double sd=0, sr=0, maxabs=0;
    for (size_t i=0;i<ref.size();++i){ double d=(double)ref[i]-got[i]; sd+=d*d; sr+=(double)ref[i]*ref[i]; maxabs=std::max(maxabs,std::fabs(d)); }
    const double rmse=std::sqrt(sd/ref.size()), rref=std::sqrt(sr/ref.size());
    std::printf("decoder parity: max_abs=%.4f rmse/rms=%.4f\n", maxabs, rmse/std::max(rref,1e-12));
    if (rmse/std::max(rref,1e-12) > 0.01) { std::fprintf(stderr,"FAIL: decoder streaming diverges\n"); return 6; }
    return 0;
}
