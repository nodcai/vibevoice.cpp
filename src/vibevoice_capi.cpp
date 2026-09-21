// vibevoice_capi.cpp - flat C ABI implementation.
//
// Wraps the C++ orchestrators (vv::vibevoice_load, vv::vibevoice_tts_generate,
// vv::vibevoice_asr_transcribe) in a single global-state surface that
// LocalAI's purego backends consume (see vibevoice_capi.h).

#include "vibevoice_capi.h"

#include "audio_io.hpp"
#include "common.hpp"
#include "vibevoice_asr.hpp"
#include "vibevoice_stream.hpp"
#include "vibevoice_tts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

struct GlobalEngine {
    std::unique_ptr<vv::VibeVoiceModel> tts;
    std::unique_ptr<vv::VibeVoiceModel> asr;
    std::unique_ptr<vv::VibeVoiceVoice> voice;
    std::string                          voice_path_loaded;
    std::string                          ref_audio_path_loaded;  // 1.5b
    int                                  n_threads = 4;
    std::mutex                           mu;
};

GlobalEngine& engine() {
    static GlobalEngine g;
    return g;
}

// Release the models before the backend goes away. Registered after the
// first successful load, i.e. after the backend registered its own atexit,
// so this runs first: model buffers, then the backend, then ggml's device
// (ggml-metal asserts if buffers are still resident when its device is
// freed). A host that calls vv_capi_unload itself makes this a no-op.
void register_unload_at_exit() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::atexit([] {
            auto& g = engine();
            std::lock_guard<std::mutex> lk(g.mu);
            g.voice.reset();
            g.tts.reset();
            g.asr.reset();
        });
    });
}

bool ensure_voice_loaded(GlobalEngine& g, const char* voice_path) {
    if (!voice_path || !voice_path[0]) return g.voice != nullptr;
    if (g.voice && g.voice_path_loaded == voice_path) return true;
    if (!g.tts) {
        VV_LOG_ERROR("vv_capi: TTS model required to load a voice");
        return false;
    }
    auto v = std::make_unique<vv::VibeVoiceVoice>();
    if (!vv::vibevoice_voice_load(voice_path, *g.tts, v.get())) {
        VV_LOG_ERROR("vv_capi: voice load failed: %s", voice_path);
        return false;
    }
    g.voice              = std::move(v);
    g.voice_path_loaded  = voice_path;
    return true;
}

}  // namespace

extern "C" {

int vv_capi_load(const char* tts_model_path,
                 const char* asr_model_path,
                 const char* tokenizer_path,
                 const char* voice_path,
                 int         n_threads) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);

    const bool have_tok = tokenizer_path && tokenizer_path[0];
    if ((!tts_model_path || !tts_model_path[0]) &&
        (!asr_model_path || !asr_model_path[0])) {
        VV_LOG_ERROR("vv_capi_load: at least one of tts_model_path or asr_model_path is required");
        return -2;
    }

    g.tts.reset();
    g.asr.reset();
    g.voice.reset();
    g.voice_path_loaded.clear();
    g.n_threads = n_threads > 0 ? n_threads : 4;

    // Each VibeVoiceModel has its own Tokenizer member, but the tokenizer
    // backing data is loaded from the same gguf each time and `load_from_file`
    // refuses to run twice on a single Tokenizer instance. We use a fresh
    // Tokenizer per model — both load OK because each is a different
    // instance.
    if (tts_model_path && tts_model_path[0]) {
        auto m = std::make_unique<vv::VibeVoiceModel>();
        if (!vv::vibevoice_load(tts_model_path, m.get())) {
            VV_LOG_ERROR("vv_capi_load: TTS model load failed: %s", tts_model_path);
            return -3;
        }
        if (m->tokenizer.vocab_size() == 0) {
            if (!have_tok) { VV_LOG_ERROR("vv_capi_load: TTS model has no embedded tokenizer; tokenizer_path required"); return -7; }
            if (!m->tokenizer.load_from_file(tokenizer_path)) {
                VV_LOG_ERROR("vv_capi_load: TTS tokenizer load failed: %s", tokenizer_path);
                return -7;
            }
        }
        g.tts = std::move(m);
    }

    if (asr_model_path && asr_model_path[0]) {
        auto m = std::make_unique<vv::VibeVoiceModel>();
        if (!vv::vibevoice_load(asr_model_path, m.get())) {
            VV_LOG_ERROR("vv_capi_load: ASR model load failed: %s", asr_model_path);
            return -3;
        }
        if (m->tokenizer.vocab_size() == 0) {
            if (!have_tok) { VV_LOG_ERROR("vv_capi_load: ASR model has no embedded tokenizer; tokenizer_path required"); return -7; }
            if (!m->tokenizer.load_from_file(tokenizer_path)) {
                VV_LOG_ERROR("vv_capi_load: ASR tokenizer load failed: %s", tokenizer_path);
                return -7;
            }
        }
        g.asr = std::move(m);
    }

    if (voice_path && voice_path[0]) {
        if (!ensure_voice_loaded(g, voice_path)) return -3;
    }
    register_unload_at_exit();
    return 0;
}

int vv_capi_tts(const char*        text,
                const char*        voice_path,
                const char* const* ref_audio_paths,
                int                n_ref_audio_paths,
                const char*        dst_wav_path,
                int                n_diffusion_steps,
                float              cfg_scale,
                int                max_speech_frames,
                uint32_t           seed) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.tts)             return -3;
    if (!text || !dst_wav_path) return -2;

    const bool is_15b = (g.tts->variant == "1.5b");

    vv::VibeVoiceTTSParams p;
    p.cfg_scale          = cfg_scale > 0.0f ? cfg_scale : 1.3f;
    p.n_diffusion_steps  = n_diffusion_steps > 0 ? n_diffusion_steps : 20;
    p.max_speech_frames  = max_speech_frames > 0 ? max_speech_frames : 200;
    p.seed               = seed;
    p.verbose            = false;

    if (is_15b) {
        // 1.5B path: ref_audio_paths is required (per call or via load).
        if (n_ref_audio_paths > 0 && ref_audio_paths) {
            for (int i = 0; i < n_ref_audio_paths; ++i) {
                if (ref_audio_paths[i] && ref_audio_paths[i][0]) {
                    p.ref_audio_paths.emplace_back(ref_audio_paths[i]);
                }
            }
        }
        if (p.ref_audio_paths.empty() && !g.ref_audio_path_loaded.empty()) {
            p.ref_audio_paths.push_back(g.ref_audio_path_loaded);
        }
        if (p.ref_audio_paths.empty()) {
            VV_LOG_ERROR("vv_capi_tts: 1.5b model needs at least one "
                         "ref_audio_paths entry (per-call or via vv_capi_load)");
            return -2;
        }
    } else {
        // realtime-0.5b path: voice_path required (per call or via load).
        if (voice_path && voice_path[0]) {
            if (!ensure_voice_loaded(g, voice_path)) return -3;
        }
        if (!g.voice) {
            VV_LOG_ERROR("vv_capi_tts: no voice loaded — pass voice_path "
                         "here or to vv_capi_load");
            return -2;
        }
        p.voice = g.voice.get();
    }

    std::vector<float> samples;
    int rc = vv::vibevoice_tts_generate(g.tts.get(), text, p, &samples);
    if (rc != 0) {
        VV_LOG_ERROR("vv_capi_tts: generate rc=%d", rc);
        return rc;
    }

    vv_audio audio_out{};
    audio_out.samples     = samples.data();
    audio_out.n_samples   = samples.size();
    audio_out.sample_rate = vv::vibevoice_tts_output_sample_rate(*g.tts, p);
    audio_out.channels    = 1;
    return vv::save_wav_pcm16(dst_wav_path, audio_out);
}

int vv_capi_tts_sample_rate(void) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.tts) return 0;
    vv::VibeVoiceTTSParams p;
    return vv::vibevoice_tts_output_sample_rate(*g.tts, p);
}

int vv_capi_tts_stream(const char* text,
                       const char* voice_path,
                       int         n_diffusion_steps,
                       float       cfg_scale,
                       int         max_speech_frames,
                       uint32_t    seed,
                       vv_pcm_cb   on_pcm,
                       void*       user) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.tts)          return -3;
    if (!text || !on_pcm) return -2;

    // Realtime-0.5B only: the streaming decoder is not wired for the 1.5B
    // variant (see vibevoice_tts_generate_streaming). Report a distinct code.
    if (g.tts->variant == "1.5b") {
        VV_LOG_ERROR("vv_capi_tts_stream: streaming unsupported for 1.5b model");
        return -20;
    }

    vv::VibeVoiceTTSParams p;
    p.cfg_scale          = cfg_scale > 0.0f ? cfg_scale : 1.3f;
    p.n_diffusion_steps  = n_diffusion_steps > 0 ? n_diffusion_steps : 20;
    p.max_speech_frames  = max_speech_frames > 0 ? max_speech_frames : 200;
    p.seed               = seed;
    p.verbose            = false;

    if (voice_path && voice_path[0]) {
        if (!ensure_voice_loaded(g, voice_path)) return -3;
    }
    if (!g.voice) {
        VV_LOG_ERROR("vv_capi_tts_stream: no voice loaded — pass voice_path "
                     "here or to vv_capi_load");
        return -2;
    }
    p.voice = g.voice.get();

    // Reused int16 buffer. Convert each float window with the EXACT same
    // clamp + round as save_wav_pcm16 (audio_io.cpp) so the concatenated
    // callback stream is byte-identical to the WAV vv_capi_tts writes.
    std::vector<int16_t> buf;
    auto on_chunk = [&](const float* s, int n) -> bool {
        buf.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            float x = std::max(-1.0f, std::min(1.0f, s[i]));
            buf[static_cast<size_t>(i)] =
                static_cast<int16_t>(std::lround(x * 32767.0f));
        }
        // Non-zero from the caller means abort → stop generation cleanly.
        return on_pcm(buf.data(), n, user) == 0;
    };

    int rc = vv::vibevoice_tts_generate_streaming(g.tts.get(), text, p, on_chunk);
    if (rc != 0) {
        VV_LOG_ERROR("vv_capi_tts_stream: generate rc=%d", rc);
        return rc;
    }
    return 0;
}

void vv_capi_stream_default_params(vv_capi_stream_params* p) {
    if (!p) return;
    std::memset(p, 0, sizeof(*p));
    p->neg_condition_anchor = -1.0f;
}

}  // extern "C"

struct vv_capi_stream {
    vv::TtsStream st;
    vv_audio_cb   cb   = nullptr;
    void*         user = nullptr;
};

extern "C" {

vv_capi_stream* vv_capi_stream_begin(const char*                  voice_path,
                                     const vv_capi_stream_params* p,
                                     vv_audio_cb                  on_audio,
                                     void*                        user) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.tts || !on_audio) return nullptr;
    if (g.tts->variant == "1.5b") {
        VV_LOG_ERROR("vv_capi_stream_begin: streaming unsupported for 1.5b model");
        return nullptr;
    }
    if (voice_path && voice_path[0] && !ensure_voice_loaded(g, voice_path)) return nullptr;
    if (!g.voice) {
        VV_LOG_ERROR("vv_capi_stream_begin: no voice loaded");
        return nullptr;
    }
    vv_capi_stream_params d;
    vv_capi_stream_default_params(&d);
    if (!p) p = &d;
    vv::VibeVoiceTTSParams tp;
    tp.voice                     = g.voice.get();
    tp.n_diffusion_steps         = p->n_diffusion_steps > 0 ? p->n_diffusion_steps : 20;
    tp.cfg_scale                 = p->cfg_scale > 0.0f ? p->cfg_scale : 1.3f;
    tp.max_speech_frames         = p->max_speech_frames > 0 ? p->max_speech_frames : 200;
    tp.max_text_tokens           = p->max_text_tokens > 0 ? p->max_text_tokens : 1024;
    tp.seed                      = p->seed;
    tp.stream_first_chunk_frames = p->first_chunk_frames > 0 ? p->first_chunk_frames : 3;
    tp.stream_lead_chunk_frames  = p->lead_chunk_frames > 0 ? p->lead_chunk_frames : 0;
    tp.neg_condition_anchor      = p->neg_condition_anchor >= 0.0f ? p->neg_condition_anchor : 0.2f;
    tp.postfilter                = p->no_postfilter == 0;
    tp.trim_lead                 = p->no_trim == 0;
    auto* s = new vv_capi_stream();
    s->cb = on_audio;
    s->user = user;
    if (!s->st.begin(g.tts.get(), tp, [s](const float* pcm, int n) { s->cb(pcm, n, s->user); })) {
        delete s;
        return nullptr;
    }
    return s;
}

int vv_capi_stream_push_text(vv_capi_stream* s, const char* utf8) {
    if (!s || !utf8) return -2;
    if (s->st.done()) return -1;
    s->st.push_text(utf8);
    return 0;
}

int vv_capi_stream_end(vv_capi_stream* s) {
    if (!s) return -2;
    s->st.end();
    return 0;
}

void vv_capi_stream_abort(vv_capi_stream* s) {
    if (s) s->st.abort();
}

int vv_capi_stream_done(vv_capi_stream* s) {
    return (!s || s->st.done()) ? 1 : 0;
}

int vv_capi_stream_sample_rate(vv_capi_stream* s) {
    return s ? s->st.sample_rate() : 0;
}

void vv_capi_stream_free(vv_capi_stream* s) {
    if (!s) return;
    s->st.join();
    delete s;
}

int vv_capi_asr(const char* src_wav_path,
                char*       out_json,
                size_t      out_capacity,
                int         max_new_tokens) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    if (!g.asr)                    return -3;
    if (!src_wav_path || !out_json || out_capacity == 0) return -2;

    std::vector<float> audio;
    if (vv::load_wav_24k_mono(src_wav_path, &audio) != 0 || audio.empty()) {
        VV_LOG_ERROR("vv_capi_asr: failed to load wav: %s", src_wav_path);
        return -8;
    }

    vv::VibeVoiceASRParams p;
    p.max_new_tokens     = max_new_tokens > 0 ? max_new_tokens : 256;
    p.repetition_penalty = 1.0f;
    p.no_repeat_ngram    = 0;
    p.verbose            = false;

    std::string transcript;
    int rc = vv::vibevoice_asr_transcribe(g.asr.get(), audio, p, &transcript);
    if (rc != 0) {
        VV_LOG_ERROR("vv_capi_asr: transcribe rc=%d", rc);
        return rc;
    }
    if (transcript.empty()) {
        out_json[0] = '\0';
        return 0;
    }

    const size_t need = transcript.size() + 1;  // +1 for NUL
    if (need > out_capacity) {
        // Write the prefix that fits, NUL-terminate, and report the
        // required size negatively so the caller can grow + retry.
        std::memcpy(out_json, transcript.data(), out_capacity - 1);
        out_json[out_capacity - 1] = '\0';
        return -static_cast<int>(need);
    }
    std::memcpy(out_json, transcript.data(), transcript.size());
    out_json[transcript.size()] = '\0';
    return static_cast<int>(transcript.size());
}

void vv_capi_unload(void) {
    auto& g = engine();
    std::lock_guard<std::mutex> lk(g.mu);
    g.voice.reset();
    g.tts.reset();
    g.asr.reset();
    g.voice_path_loaded.clear();
}

const char* vv_capi_version(void) { return "vibevoice.cpp 0.1.0 (capi)"; }

int vv_capi_voice_clone(const char* /*src_wav_path*/,
                        const char* /*dst_voice_gguf_path*/,
                        int         /*with_cfg*/) {
    VV_LOG_ERROR("vv_capi_voice_clone: not supported. The realtime-0.5B "
                 "weights ship without encoders, so we cannot prep a voice "
                 "gguf at runtime. Load a 1.5B gguf and pass ref_audio_path "
                 "to vv_capi_tts for runtime voice cloning instead.");
    return -2;
}

}  // extern "C"
