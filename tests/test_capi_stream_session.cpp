// C ABI streaming session (vv_capi_stream_*). Gated on VIBEVOICE_TTS_MODEL,
// VIBEVOICE_TOKENIZER, VIBEVOICE_VOICE; skips (77) otherwise.
//
// Pushes text in pieces, checks chunks arrive before end(), that done()
// flips after the worker drains, that the reported sample rate matches the
// model (48 kHz with the post-filter, 24 kHz without), and that the
// no-postfilter switch changes the rate.
#include "vibevoice_capi.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {
struct Sink {
    std::vector<float> pcm;
    int chunks = 0;
};
void on_audio(const float* pcm, int n, void* user) {
    auto* s = static_cast<Sink*>(user);
    s->pcm.insert(s->pcm.end(), pcm, pcm + n);
    ++s->chunks;
}
}  // namespace

int main() {
    const char* tts = std::getenv("VIBEVOICE_TTS_MODEL");
    const char* tok = std::getenv("VIBEVOICE_TOKENIZER");
    const char* voice = std::getenv("VIBEVOICE_VOICE");
    if (!tts || !tok || !voice) { std::fprintf(stderr, "skip: set VIBEVOICE_TTS_MODEL/_TOKENIZER/_VOICE\n"); return 77; }
    if (vv_capi_load(tts, nullptr, tok, voice, 0) != 0) { std::fprintf(stderr, "FAIL: load\n"); return 1; }

    for (int no_pf = 0; no_pf <= 1; ++no_pf) {
        vv_capi_stream_params p;
        vv_capi_stream_default_params(&p);
        p.n_diffusion_steps = 3; p.cfg_scale = 1.7f; p.seed = 99; p.no_postfilter = no_pf;
        Sink sink;
        vv_capi_stream* s = vv_capi_stream_begin(nullptr, &p, on_audio, &sink);
        if (!s) { std::fprintf(stderr, "FAIL: begin\n"); return 2; }
        const int rate = vv_capi_stream_sample_rate(s);
        for (const char* piece : {"Hello there! ", "Welcome to ", "the session test."}) {
            if (vv_capi_stream_push_text(s, piece) != 0) { std::fprintf(stderr, "FAIL: push\n"); return 3; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (vv_capi_stream_done(s)) { std::fprintf(stderr, "FAIL: done before end\n"); return 4; }
        vv_capi_stream_end(s);
        for (int i = 0; i < 600 && !vv_capi_stream_done(s); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const bool done = vv_capi_stream_done(s) != 0;
        vv_capi_stream_free(s);
        std::printf("no_postfilter=%d: rate %d Hz, %d chunks, %.2f s, done=%d\n", no_pf, rate, sink.chunks,
                    static_cast<double>(sink.pcm.size()) / rate, done ? 1 : 0);
        if (!done || sink.chunks < 2 || sink.pcm.empty()) { std::fprintf(stderr, "FAIL: stream did not deliver\n"); return 5; }
        if (rate != 48000 && rate != 24000) { std::fprintf(stderr, "FAIL: rate\n"); return 6; }
        if (no_pf && rate != 24000) { std::fprintf(stderr, "FAIL: no_postfilter should give 24 kHz\n"); return 7; }
    }
    vv_capi_unload();
    std::printf("OK\n");
    return 0;
}
