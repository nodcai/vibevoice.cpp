// vibevoice_capi.h - flat C ABI for purego / dlopen integration.
//
// This is a *separate* header from vibevoice.h on purpose: it exposes a
// stateless, file-path-oriented surface that matches what
// LocalAI's go-purego backends expect (see backend/go/qwen3-tts-cpp/cpp/).
//
// Lifetime model: a single global engine, one load_model() per process,
// many tts() / asr() calls. Mirrors qwen3-tts-cpp exactly so a purego
// dlsym lookup and `purego.RegisterLibFunc` finds these symbols by name.
//
// Why a separate flat ABI instead of the existing vibevoice.h:
//   * No opaque pointers — purego pinning lifetimes is fiddly.
//   * No callee-allocated buffers — output via WAV path or caller-owned
//     char buffer.
//   * All return codes are int with 0 = success — every purego backend
//     in LocalAI expects exactly that.

#ifndef VIBEVOICE_CAPI_H
#define VIBEVOICE_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Loads the engine. Either or both model paths can be NULL:
//   tts_model_path  - realtime-0.5b gguf, required for vv_capi_tts.
//   asr_model_path  - asr-7b gguf,        required for vv_capi_asr.
//   tokenizer_path  - tokenizer gguf,     required for either.
//   voice_path      - voice gguf,         required for vv_capi_tts.
//                     Multiple voices: re-call with a different path,
//                     or pass NULL here and a per-call voice path to
//                     vv_capi_tts.
//   n_threads       - 0 → auto-detect.
// Returns 0 on success, non-zero error code otherwise. Idempotent —
// calling twice replaces the engine.
int vv_capi_load(const char* tts_model_path,
                 const char* asr_model_path,
                 const char* tokenizer_path,
                 const char* voice_path,
                 int         n_threads);

// Synthesize `text` into a 24 kHz mono WAV at `dst_wav_path`. The TTS
// path is selected by the loaded model's variant:
//
//   * realtime-0.5b -> uses `voice_path` (a pre-baked voice gguf).
//                      `ref_audio_paths` must be NULL / n_ref_audio == 0.
//   * 1.5b          -> uses `ref_audio_paths` — one WAV per speaker,
//                      24 kHz mono. `n_ref_audio_paths` is the number
//                      of distinct speakers (>= 1; the dialog in `text`
//                      can reference Speaker 0 .. n-1). `voice_path`
//                      must be NULL.
//
// `text` is either a plain sentence (single-speaker convenience —
// auto-wrapped as "Speaker 0: ...") or speaker-tagged dialog with one
// "Speaker N:" line per turn (passed through verbatim).
//
// `voice_path` may be NULL if already supplied to vv_capi_load.
// n_diffusion_steps == 0 -> 20, cfg_scale == 0 -> 1.3 (1.0 disables
// CFG), max_speech_frames == 0 -> 200, seed == 0 -> random.
int vv_capi_tts(const char*        text,
                const char*        voice_path,
                const char* const* ref_audio_paths,
                int                n_ref_audio_paths,
                const char*        dst_wav_path,
                int                n_diffusion_steps,
                float              cfg_scale,
                int                max_speech_frames,
                uint32_t           seed);

// Streaming counterpart of vv_capi_tts for the realtime-0.5B model. Each
// decoded audio window is converted to 24 kHz mono signed-16-bit PCM and handed
// to `on_pcm` as soon as it is produced. Concatenating every callback's samples
// yields exactly the PCM that vv_capi_tts writes to a WAV for the same
// text/voice/seed (same generate path, same float→int16 conversion).
//
//   text              - sentence or speaker-tagged dialog (see vv_capi_tts).
//   voice_path        - voice gguf; may be NULL if supplied to vv_capi_load.
//   n_diffusion_steps - 0 → 20. cfg_scale 0 → 1.3. max_speech_frames 0 → 200.
//   seed              - 0 → random.
//   on_pcm            - called per window; return non-zero to abort cleanly.
//   user              - opaque pointer forwarded to on_pcm.
//
// Realtime-0.5B only: returns -20 if a 1.5B model is loaded (unsupported).
// Returns 0 on success, non-zero error code otherwise.
typedef int (*vv_pcm_cb)(const int16_t* samples, int n_samples, void* user);
int vv_capi_tts_stream(const char* text,
                       const char* voice_path,
                       int         n_diffusion_steps,
                       float       cfg_scale,
                       int         max_speech_frames,
                       uint32_t    seed,
                       vv_pcm_cb   on_pcm,
                       void*       user);

// ---- Streaming session (realtime-0.5B) ----------------------------------
// Push text while it is being produced and receive audio per chunk from a
// worker thread. `on_audio` runs on that thread; `pcm` is valid only for the
// duration of the call. One session at a time per process: the worker owns
// the engine while it runs, so do not call vv_capi_tts / vv_capi_asr until
// vv_capi_stream_done() reports true or the session was freed.
typedef struct vv_capi_stream vv_capi_stream;
typedef void (*vv_audio_cb)(const float* pcm, int n_samples, void* user);

typedef struct {
    int      n_diffusion_steps;    /* 0 -> 20 */
    float    cfg_scale;            /* 0 -> 1.3 (1.0 disables CFG) */
    int      max_speech_frames;    /* 0 -> 200 (~27 s) */
    int      max_text_tokens;      /* 0 -> 1024 */
    uint32_t seed;                 /* 0 -> random */
    int      first_chunk_frames;   /* 0 -> 3; the dominant term in time-to-first-audio */
    int      lead_chunk_frames;    /* 0 -> off; hold this size until a chunk carried speech */
    float    neg_condition_anchor; /* < 0 -> 0.2; CFG negative-path anchor blend, 0..1 */
} vv_capi_stream_params;

// Fills `p` with the defaults above (all zero / -1 means "default").
void vv_capi_stream_default_params(vv_capi_stream_params* p);

// Starts a session with the loaded TTS model. `voice_path` may be NULL if a
// voice was given to vv_capi_load. Returns NULL on failure.
vv_capi_stream* vv_capi_stream_begin(const char*                  voice_path,
                                     const vv_capi_stream_params* p,
                                     vv_audio_cb                  on_audio,
                                     void*                        user);
// Append UTF-8 text. Non-blocking. Returns 0, or -1 after end().
int  vv_capi_stream_push_text(vv_capi_stream* s, const char* utf8);
// No more text. Non-blocking; the worker drains and finishes.
int  vv_capi_stream_end(vv_capi_stream* s);
// Stop as soon as possible; no further audio is delivered.
void vv_capi_stream_abort(vv_capi_stream* s);
// Non-zero once the worker has finished (all audio delivered or aborted).
int  vv_capi_stream_done(vv_capi_stream* s);
// Sample rate of the delivered chunks.
int  vv_capi_stream_sample_rate(vv_capi_stream* s);
// end() if needed, join the worker, release the session.
void vv_capi_stream_free(vv_capi_stream* s);

// Transcribe `src_wav_path` into a JSON string written into the caller-
// owned `out_json` buffer of size `out_capacity`. The JSON is the same
// shape the model produces, e.g.
//   [{"Start":0.0,"End":2.8,"Speaker":0,"Content":"…"}]
// Returns:
//    > 0 on success — the number of bytes written (excluding the NUL
//                     terminator). The buffer is always NUL-terminated.
//      0 if no transcription was produced.
//    < 0 on error (see vv_status in vibevoice.h for the enum).
//   If out_capacity is smaller than the produced JSON, returns the
//   negative of the required size and writes the prefix that fit.
int vv_capi_asr(const char* src_wav_path,
                char*       out_json,
                size_t      out_capacity,
                int         max_new_tokens);

// Free engine state. Optional — process exit also frees it.
void vv_capi_unload(void);

// Build / version info. Returns a pointer to a static string; do not free.
const char* vv_capi_version(void);

// Deprecated: voice-cloning via the realtime-0.5B + ASR-7B path is not
// supported; the public realtime weights ship without encoders. Load
// a 1.5B gguf and call vv_capi_tts with `ref_audio_path` instead.
int vv_capi_voice_clone(const char* src_wav_path,
                        const char* dst_voice_gguf_path,
                        int         with_cfg);

#ifdef __cplusplus
}
#endif

#endif  // VIBEVOICE_CAPI_H
