#ifndef VIBEVOICE_DFN_HPP
#define VIBEVOICE_DFN_HPP

// DeepFilterNet3 streaming speech enhancement, used as the TTS post-filter.
//
// VibeVoice realtime occasionally hallucinates a musical artefact in front
// of utterances that open with certain phrases ("Hello there", "Welcome").
// DFN3 is a small 48 kHz speech denoiser that removes it without touching
// the voice. Two stages per 10 ms frame: an ERB-band gain on the whole
// spectrum, then a complex "deep filter" over the lowest ~5 kHz.
//
// Only the hand-written per-frame forward (dfn_fast) is here; the ggml
// graph it was validated against is not part of this port. The weights
// come from the same gguf as the TTS model (scripts/merge_dfn_gguf.py),
// under `dfn.*`.
//
// All audio is 48 kHz mono float32.

#include "model_loader.hpp"

#include <vector>

struct dfn_fft_plan;
struct dfn_fast_weights;
struct dfn_fast_scratch;

namespace vv {

struct DfnModel {
    int sample_rate      = 48000;
    int frame_size       = 480;    // hop, 10 ms
    int fft_size         = 960;
    int lookahead_frames = 2;
    int n_erb            = 32;
    int df_bins          = 96;
    int df_order         = 5;

    std::vector<float> window;       // fft_size, Vorbis window
    std::vector<int>   erb_widths;   // n_erb
    std::vector<int>   erb_indices;  // n_erb
    dfn_fast_weights*  fast = nullptr;
    bool               ready = false;

    DfnModel() = default;
    DfnModel(const DfnModel&)            = delete;
    DfnModel& operator=(const DfnModel&) = delete;
    ~DfnModel();
    void free();
};

// Loads the filter from a gguf that carries `dfn.*` tensors (detected by
// `dfn.window`). Returns false, leaving *out unready, when the file has no
// filter; logs and returns false on a malformed one.
bool dfn_model_load(const ModelLoader& m, DfnModel* out);
bool dfn_model_present(const ModelLoader& m);

// Per-utterance state: STFT overlap-add, feature EMAs, GRU hidden states,
// spectrum and conv histories, and the lookahead delay line.
class DfnStream {
public:
    explicit DfnStream(const DfnModel& m);
    ~DfnStream();
    DfnStream(const DfnStream&)            = delete;
    DfnStream& operator=(const DfnStream&) = delete;

    void reset();
    // Run `n_frames` of silence through the filter and discard the output,
    // so the EMA and GRU state has converged away from its initial values
    // before real audio arrives. 100 frames (1 s) gives a clean start.
    void warmup(int n_frames);
    // Push `n` input samples; append whatever the filter can emit to *out.
    // The first 2 frames (20 ms) are held back as lookahead. `final_chunk`
    // flushes them and the overlap-add tail; `in` may then be null.
    void process(const float* in, int n, std::vector<float>* out, bool final_chunk);

private:
    void process_frame(const float* in_n_fft);

    const DfnModel&    m_;
    dfn_fft_plan*      fft_ = nullptr;
    dfn_fast_scratch*  scratch_ = nullptr;

    std::vector<float> in_buf_, frame_, spec_re_, spec_im_, ola_, out_queue_;
    std::vector<float> log_mag_mean_;      // n_erb
    bool               norm_seeded_ = false;
    std::vector<float> unit_norm_state_;   // df_bins
    bool               unit_norm_seeded_ = false;
    std::vector<float> df_hist_re_, df_hist_im_;
    int                df_hist_frames_ = 0;
    std::vector<float> gru_hidden_;        // 5 x 256
    std::vector<float> feat_erb_hist_;     // 2 x 32
    std::vector<float> feat_spec_hist_;    // 2 x 2 x 96
    std::vector<float> c0_hist_;           // 64 x 4 x 96
    int                frame_counter_ = 0;
    // per-frame work buffers
    std::vector<float> feat_spec_flat_, feh_, fsh_, coefs_flat_, c0_curr_;
    std::vector<float> erb_power_, log_feat_, gains_, coef_re_, coef_im_;
};

}  // namespace vv

#endif  // VIBEVOICE_DFN_HPP
