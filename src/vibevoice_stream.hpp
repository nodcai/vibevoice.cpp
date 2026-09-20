#ifndef VIBEVOICE_STREAM_HPP
#define VIBEVOICE_STREAM_HPP

// Streaming TTS session for the realtime-0.5B model.
//
// Text is pushed incrementally (as an LLM produces it) and audio comes
// back per chunk from a worker thread, so the first sound does not wait
// for the whole reply. The worker runs vibevoice_tts_run_realtime with a
// text source that tokenizes the stable prefix of what has been pushed
// so far: everything up to the last whitespace, because Qwen2's
// byte-level BPE folds a space into the following word (a trailing
// standalone space token would become a different token once the next
// word arrives and the generation would diverge). On end() the whole
// text plus a trailing "\n" is tokenized, exactly like the batch path.
//
// Requires a voice prompt. One session at a time per model: the worker
// owns the model's backend while it runs.

#include "vibevoice_tts.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace vv {

// Called from the worker thread for every decoded chunk. `pcm` is valid
// only for the duration of the call.
using TtsStreamAudioCb = std::function<void(const float* pcm, int n_samples)>;

class TtsStream {
public:
    TtsStream() = default;
    ~TtsStream();
    TtsStream(const TtsStream&)            = delete;
    TtsStream& operator=(const TtsStream&) = delete;

    // Starts the worker. `p.voice` must be set. Returns false on failure.
    bool begin(VibeVoiceModel* model, const VibeVoiceTTSParams& p, TtsStreamAudioCb on_audio);
    // Append UTF-8 text. Non-blocking. Ignored after end().
    void push_text(const std::string& utf8);
    // No more text will come. Non-blocking; the worker drains and finishes.
    void end();
    // Stop as soon as possible; no further audio is delivered.
    void abort();
    // True once the worker has finished (all audio delivered, or aborted).
    bool done() const;
    // Result of the generation loop (0 = ok), valid once done().
    int  result() const { return rc_; }
    // end() if needed, then join the worker.
    void join();

    // Output sample rate of the chunks this session delivers.
    int sample_rate() const { return sample_rate_; }

private:
    void worker();

    VibeVoiceModel*      model_ = nullptr;
    VibeVoiceTTSParams   params_;
    TtsStreamAudioCb     on_audio_;
    std::thread          thread_;
    mutable std::mutex   mu_;
    std::condition_variable cv_;
    std::string          text_;            // everything pushed so far
    size_t               handed_out_ = 0;  // tokens already given to the loop
    bool                 ended_  = false;
    bool                 done_   = false;
    std::atomic<bool>    aborted_{false};
    int                  rc_     = 0;
    int                  sample_rate_ = 24000;
};

}  // namespace vv

#endif  // VIBEVOICE_STREAM_HPP
