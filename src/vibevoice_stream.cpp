#include "vibevoice_stream.hpp"

#include "common.hpp"

#include <algorithm>

namespace vv {

TtsStream::~TtsStream() { join(); }

bool TtsStream::begin(VibeVoiceModel* model, const VibeVoiceTTSParams& p, TtsStreamAudioCb on_audio) {
    if (!model || !on_audio) return false;
    if (model->variant == "1.5b") {
        VV_LOG_ERROR("TtsStream: only the realtime-0.5b model streams");
        return false;
    }
    if (!p.voice) {
        VV_LOG_ERROR("TtsStream: a voice prompt is required");
        return false;
    }
    if (thread_.joinable()) return false;
    model_       = model;
    params_      = p;
    on_audio_    = std::move(on_audio);
    sample_rate_ = model->cfg.sample_rate;
    text_.clear();
    handed_out_ = 0;
    ended_ = done_ = false;
    aborted_.store(false);
    rc_ = 0;
    thread_ = std::thread([this] { worker(); });
    return true;
}

void TtsStream::push_text(const std::string& utf8) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (ended_) return;
        text_ += utf8;
    }
    cv_.notify_all();
}

void TtsStream::end() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        ended_ = true;
    }
    cv_.notify_all();
}

void TtsStream::abort() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        aborted_.store(true);
        ended_ = true;
    }
    cv_.notify_all();
}

bool TtsStream::done() const {
    std::lock_guard<std::mutex> lk(mu_);
    return done_;
}

void TtsStream::join() {
    end();
    if (thread_.joinable()) thread_.join();
}

void TtsStream::worker() {
    TtsStreamCtl ctl;
    ctl.pull_text = [this](int max_tokens, std::vector<int32_t>* out, bool* eof) {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            // Tokenize the whole accumulated text up to the stable cut, so the
            // token sequence matches what the batch path would produce for the
            // same text, and hand out only what has not been handed out yet.
            std::vector<int32_t> toks;
            if (ended_) {
                std::string t = text_;
                while (!t.empty() && (t.back() == ' ' || t.back() == '\t' || t.back() == '\r' || t.back() == '\n'))
                    t.pop_back();
                if (!t.empty()) toks = model_->tokenizer.encode(t + "\n");
            } else {
                // Cut AT the last whitespace, not after it: the space merges
                // into the next word's token once that word arrives.
                const size_t ws = text_.find_last_of(" \t\n");
                if (ws != std::string::npos && ws > 0)
                    toks = model_->tokenizer.encode(text_.substr(0, ws));
            }
            if (handed_out_ < toks.size()) {
                const size_t n = std::min(toks.size() - handed_out_, static_cast<size_t>(std::max(1, max_tokens)));
                out->insert(out->end(), toks.begin() + static_cast<long>(handed_out_),
                            toks.begin() + static_cast<long>(handed_out_ + n));
                handed_out_ += n;
                *eof = ended_ && handed_out_ >= toks.size();
                return;
            }
            if (ended_ || aborted_.load()) {
                *eof = true;
                return;
            }
            cv_.wait(lk);
        }
    };
    ctl.emit_audio = [this](const float* pcm, int n) {
        if (aborted_.load()) return false;
        if (pcm && n > 0) on_audio_(pcm, n);
        return true;
    };
    ctl.should_abort = [this] { return aborted_.load(); };

    const int rc = vibevoice_tts_run_realtime(model_, params_, ctl);
    {
        std::lock_guard<std::mutex> lk(mu_);
        rc_   = rc;
        done_ = true;
    }
    cv_.notify_all();
}

}  // namespace vv
