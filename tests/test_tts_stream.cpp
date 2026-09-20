// Streaming session vs batch generate. Gated: needs VIBEVOICE_TTS_MODEL,
// VIBEVOICE_TOKENIZER, VIBEVOICE_VOICE (realtime-0.5b).
//
// Text pushed in pieces that split words must produce the same token
// windows as the whole text, so with the same seed the stream's chunks
// concatenate to exactly the batch waveform. Also checks done() and that
// abort() returns promptly.
#include "vibevoice_stream.hpp"
#include "vibevoice_tts.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main() {
    const char* model_env = std::getenv("VIBEVOICE_TTS_MODEL");
    const char* tok_env   = std::getenv("VIBEVOICE_TOKENIZER");
    const char* voice_env = std::getenv("VIBEVOICE_VOICE");
    if (!model_env || !*model_env || !tok_env || !*tok_env || !voice_env || !*voice_env) {
        std::fprintf(stderr, "skip: set VIBEVOICE_TTS_MODEL, VIBEVOICE_TOKENIZER, VIBEVOICE_VOICE\n");
        return 77;
    }
    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_env, &model)) { std::fprintf(stderr, "FAIL: load\n"); return 1; }
    if (!model.tokenizer.load_from_file(tok_env)) { std::fprintf(stderr, "FAIL: tokenizer\n"); return 2; }
    vv::VibeVoiceVoice voice;
    if (!vv::vibevoice_voice_load(voice_env, model, &voice)) { std::fprintf(stderr, "FAIL: voice\n"); return 3; }

    vv::VibeVoiceTTSParams p;
    p.voice = &voice; p.n_diffusion_steps = 3; p.cfg_scale = 1.7f; p.seed = 4242;

    const std::string text = "Hello there! Welcome to the demo, it should sound quite natural.";

    // batch
    std::vector<float> batch;
    if (vv::vibevoice_tts_generate(&model, text, p, &batch) != 0 || batch.empty()) {
        std::fprintf(stderr, "FAIL: batch generate\n"); return 4;
    }

    // stream, pushed in pieces that split words
    std::vector<float> streamed;
    int n_chunks = 0;
    vv::TtsStream st;
    if (!st.begin(&model, p, [&](const float* pcm, int n) {
            streamed.insert(streamed.end(), pcm, pcm + n);
            ++n_chunks;
        })) { std::fprintf(stderr, "FAIL: stream begin\n"); return 5; }
    const char* pieces[] = {"Hello th", "ere! Wel", "come to the de", "mo, it should", " sound quite natural."};
    for (const char* piece : pieces) {
        st.push_text(piece);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    if (st.done()) { std::fprintf(stderr, "FAIL: done before end()\n"); return 6; }
    st.end();
    st.join();
    if (!st.done() || st.result() != 0) { std::fprintf(stderr, "FAIL: stream rc=%d\n", st.result()); return 7; }

    std::printf("batch %zu samples, stream %zu samples in %d chunks\n", batch.size(), streamed.size(), n_chunks);
    if (batch.size() != streamed.size()) { std::fprintf(stderr, "FAIL: length mismatch\n"); return 8; }
    double max_abs = 0;
    for (size_t i = 0; i < batch.size(); ++i) max_abs = std::max(max_abs, static_cast<double>(std::fabs(batch[i] - streamed[i])));
    std::printf("max_abs diff = %.3e\n", max_abs);
    if (max_abs > 1e-5) { std::fprintf(stderr, "FAIL: stream diverges from batch\n"); return 9; }
    if (n_chunks < 2) { std::fprintf(stderr, "FAIL: expected several chunks\n"); return 10; }

    // abort returns promptly
    vv::TtsStream st2;
    int aborted_chunks = 0;
    if (!st2.begin(&model, p, [&](const float*, int) { ++aborted_chunks; })) return 11;
    st2.push_text("This is a long sentence that will be cut short before it is finished speaking, hopefully.");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const auto t0 = std::chrono::steady_clock::now();
    st2.abort();
    st2.join();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("abort joined in %.0f ms after %d chunks\n", ms, aborted_chunks);
    if (ms > 3000) { std::fprintf(stderr, "FAIL: abort too slow\n"); return 12; }
    std::printf("OK\n");
    return 0;
}
