#include "dfn.hpp"

#include "common.hpp"
#include "dfn_fast.hpp"
#include "dfn_fft.hpp"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

namespace vv {

namespace {

// Normalisation constants, verified against libdf (deep_filter/src/lib.rs
// band_mean_norm_erb / band_unit_norm):
//   alpha = exp(-hop/sr / tau), hop 480, sr 48000, tau 1 s
//   ERB log-power mean state seeded from linspace(-60, -90) dB
//   per-bin |spec| EMA seeded from linspace(0.001, 0.0001)
constexpr float kEmaAlpha   = 0.99005f;
constexpr float kLogEps     = 1e-10f;
constexpr float kErbNormDiv = 40.0f;
constexpr float kMeanNormHi = -60.0f, kMeanNormLo = -90.0f;
constexpr float kUnitNormHi = 0.001f,  kUnitNormLo = 0.0001f;
constexpr int   kEncErb  = 32;
constexpr int   kEncSpec = 96;
constexpr int   kGruH    = 256;
constexpr int   kGruN    = 5;
constexpr int   kWarmupFrames = 1;   // see the alignment note in process_frame

// Vorbis window, what libdf uses (DFState::new): satisfies w^2[i] +
// w^2[i+N/2] = 1 for 50% overlap-add.
void build_window(std::vector<float>& w, int n) {
    w.resize(n);
    const double half = n / 2.0;
    for (int i = 0; i < n; ++i) {
        const double s = std::sin(0.5 * M_PI * (i + 0.5) / half);
        w[i] = static_cast<float>(std::sin(0.5 * M_PI * s * s));
    }
}

// Exact port of libdf::erb_fb: min 2 bins per band with borrow-forward.
void build_erb_table(std::vector<int>& widths, std::vector<int>& indices, int sr, int n_fft, int n_bands) {
    constexpr int kMinNbFreqs = 2;
    auto freq2erb = [](double f) { return 9.265 * std::log1p(f / (24.7 * 9.265)); };
    auto erb2freq = [](double e) { return 24.7 * 9.265 * (std::exp(e / 9.265) - 1.0); };
    const double freq_width = static_cast<double>(sr) / n_fft;
    const double erb_low = freq2erb(0.0), erb_high = freq2erb(sr / 2.0);
    const double step = (erb_high - erb_low) / n_bands;
    widths.assign(n_bands, 0);
    int prev_freq = 0, freq_over = 0;
    for (int i = 1; i <= n_bands; ++i) {
        const int fb = static_cast<int>(std::round(erb2freq(erb_low + i * step) / freq_width));
        int nb = fb - prev_freq - freq_over;
        if (nb < kMinNbFreqs) { freq_over = kMinNbFreqs - nb; nb = kMinNbFreqs; } else { freq_over = 0; }
        widths[i - 1] = nb;
        prev_freq = fb;
    }
    widths[n_bands - 1] += 1;
    int sum = 0;
    for (int w : widths) sum += w;
    const int too_large = sum - (n_fft / 2 + 1);
    if (too_large > 0) widths[n_bands - 1] -= too_large;
    indices.assign(n_bands, 0);
    int acc = 0;
    for (int i = 0; i < n_bands; ++i) { indices[i] = acc; acc += widths[i]; }
}

std::vector<float> read_f32(const struct ggml_tensor* t) {
    std::vector<float> out(static_cast<size_t>(ggml_nelements(t)));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(out.size());
        ggml_backend_tensor_get(t, tmp.data(), 0, ggml_nbytes(t));
        for (size_t i = 0; i < out.size(); ++i) out[i] = ggml_fp16_to_fp32(tmp[i]);
    }
    return out;
}

}  // namespace

DfnModel::~DfnModel() { free(); }
void DfnModel::free() {
    if (fast) dfn_fast_free_weights(fast);
    fast = nullptr;
    ready = false;
}

bool dfn_model_present(const ModelLoader& m) { return m.has("dfn.window"); }

bool dfn_model_load(const ModelLoader& m, DfnModel* out) {
    if (!out) return false;
    out->free();
    if (!dfn_model_present(m)) return false;
    out->sample_rate      = static_cast<int>(m.get_u32("dfn.sample_rate", 48000));
    out->frame_size       = static_cast<int>(m.get_u32("dfn.hop", 480));
    out->fft_size         = static_cast<int>(m.get_u32("dfn.n_fft", 960));
    out->lookahead_frames = static_cast<int>(m.get_u32("dfn.lookahead_frames", 2));
    out->n_erb            = static_cast<int>(m.get_u32("dfn.n_erb", 32));
    out->df_bins          = static_cast<int>(m.get_u32("dfn.df_bins", 96));
    out->df_order         = static_cast<int>(m.get_u32("dfn.df_order", 5));
    if (out->n_erb != kEncErb || out->df_bins != kEncSpec || out->df_order != 5 || out->lookahead_frames != 2) {
        VV_LOG_ERROR("dfn: unsupported geometry (n_erb %d, df_bins %d, order %d, lookahead %d)",
                     out->n_erb, out->df_bins, out->df_order, out->lookahead_frames);
        return false;
    }

    const struct ggml_tensor* win = m.tensor("dfn.window");
    if (win && ggml_nelements(win) == out->fft_size) out->window = read_f32(win);
    else build_window(out->window, out->fft_size);

    const struct ggml_tensor* tw = m.tensor("dfn.erb_widths");
    const struct ggml_tensor* ti = m.tensor("dfn.erb_indices");
    if (tw && ti && tw->type == GGML_TYPE_I32 && ti->type == GGML_TYPE_I32 &&
        ggml_nelements(tw) == out->n_erb && ggml_nelements(ti) == out->n_erb) {
        out->erb_widths.resize(out->n_erb);
        out->erb_indices.resize(out->n_erb);
        ggml_backend_tensor_get(tw, out->erb_widths.data(), 0, sizeof(int32_t) * out->n_erb);
        ggml_backend_tensor_get(ti, out->erb_indices.data(), 0, sizeof(int32_t) * out->n_erb);
    } else {
        build_erb_table(out->erb_widths, out->erb_indices, out->sample_rate, out->fft_size, out->n_erb);
    }

    std::map<std::string, ggml_tensor*> tensors;
    for (const auto& name : m.tensor_names())
        if (name.rfind("dfn.", 0) == 0) tensors[name] = m.tensor(name);
    std::string err;
    out->fast = dfn_fast_prepare(tensors, &err);
    if (!out->fast) {
        VV_LOG_ERROR("dfn: %s", err.c_str());
        return false;
    }
    out->ready = true;
    VV_LOG_INFO("dfn: post-filter ready (%d Hz, %d tensors)", out->sample_rate, static_cast<int>(tensors.size()));
    return true;
}

// ---------------------------------------------------------------------------

DfnStream::DfnStream(const DfnModel& m) : m_(m) {
    fft_     = dfn_fft_plan_create(m.fft_size);
    scratch_ = dfn_fast_scratch_create();
    const int n_bins = m.fft_size / 2 + 1;
    frame_.assign(m.fft_size, 0.0f);
    ola_.assign(m.fft_size, 0.0f);
    spec_re_.assign(n_bins, 0.0f);
    spec_im_.assign(n_bins, 0.0f);
    log_mag_mean_.assign(m.n_erb, 0.0f);
    unit_norm_state_.assign(m.df_bins, 0.0f);
    df_hist_frames_ = m.df_order + m.lookahead_frames;
    df_hist_re_.assign(static_cast<size_t>(df_hist_frames_) * n_bins, 0.0f);
    df_hist_im_.assign(static_cast<size_t>(df_hist_frames_) * n_bins, 0.0f);
    gru_hidden_.assign(static_cast<size_t>(kGruN) * kGruH, 0.0f);
    feat_erb_hist_.assign(static_cast<size_t>(2) * kEncErb, 0.0f);
    feat_spec_hist_.assign(static_cast<size_t>(2) * 2 * kEncSpec, 0.0f);
    c0_hist_.assign(static_cast<size_t>(4) * 64 * kEncSpec, 0.0f);
    feat_spec_flat_.assign(static_cast<size_t>(2) * kEncSpec, 0.0f);
    feh_.assign(static_cast<size_t>(3) * kEncErb, 0.0f);
    fsh_.assign(static_cast<size_t>(2) * 3 * kEncSpec, 0.0f);
    coefs_flat_.assign(static_cast<size_t>(10) * m.df_bins, 0.0f);
    c0_curr_.assign(static_cast<size_t>(64) * kEncSpec, 0.0f);
    reset();
}

DfnStream::~DfnStream() {
    if (fft_) dfn_fft_plan_free(fft_);
    if (scratch_) dfn_fast_scratch_free(scratch_);
}

void DfnStream::reset() {
    in_buf_.clear();
    std::fill(ola_.begin(), ola_.end(), 0.0f);
    out_queue_.clear();
    std::fill(log_mag_mean_.begin(), log_mag_mean_.end(), 0.0f);
    norm_seeded_ = false;
    std::fill(unit_norm_state_.begin(), unit_norm_state_.end(), 0.0f);
    unit_norm_seeded_ = false;
    std::fill(df_hist_re_.begin(), df_hist_re_.end(), 0.0f);
    std::fill(df_hist_im_.begin(), df_hist_im_.end(), 0.0f);
    std::fill(gru_hidden_.begin(), gru_hidden_.end(), 0.0f);
    std::fill(feat_erb_hist_.begin(), feat_erb_hist_.end(), 0.0f);
    std::fill(feat_spec_hist_.begin(), feat_spec_hist_.end(), 0.0f);
    std::fill(c0_hist_.begin(), c0_hist_.end(), 0.0f);
    frame_counter_ = 0;
}

void DfnStream::warmup(int n_frames) {
    if (n_frames <= 0) return;
    const int hop = m_.frame_size;
    std::vector<float> silence(static_cast<size_t>(hop) * n_frames, 0.0f);
    std::vector<float> sink;
    process(silence.data(), static_cast<int>(silence.size()), &sink, false);
    // Drop everything buffered beyond the lookahead delay so no warm-up
    // audio leaks into the first real output.
    const size_t keep = static_cast<size_t>(m_.lookahead_frames) * hop;
    if (out_queue_.size() > keep)
        out_queue_.erase(out_queue_.begin(), out_queue_.begin() + static_cast<long>(out_queue_.size() - keep));
}

void DfnStream::process(const float* in, int n, std::vector<float>* out, bool final_chunk) {
    const int n_fft = m_.fft_size, hop = m_.frame_size;
    if (in && n > 0) in_buf_.insert(in_buf_.end(), in, in + n);
    while (static_cast<int>(in_buf_.size()) >= n_fft) {
        process_frame(in_buf_.data());
        in_buf_.erase(in_buf_.begin(), in_buf_.begin() + hop);
    }
    if (final_chunk) {
        if (!in_buf_.empty()) {
            std::vector<float> padded(n_fft, 0.0f);
            std::memcpy(padded.data(), in_buf_.data(), in_buf_.size() * sizeof(float));
            process_frame(padded.data());
            in_buf_.clear();
        }
        out_queue_.insert(out_queue_.end(), ola_.begin(), ola_.begin() + hop);
        std::fill(ola_.begin(), ola_.end(), 0.0f);
    }
    const int delay = m_.lookahead_frames * hop;
    const int avail = static_cast<int>(out_queue_.size()) - (final_chunk ? 0 : delay);
    if (avail <= 0 || !out) return;
    out->insert(out->end(), out_queue_.begin(), out_queue_.begin() + avail);
    out_queue_.erase(out_queue_.begin(), out_queue_.begin() + avail);
}

// One analysis -> features -> network -> mask + deep filter -> synthesis frame.
void DfnStream::process_frame(const float* in_n_fft) {
    const int n_fft = m_.fft_size, hop = m_.frame_size, n_bins = n_fft / 2 + 1;
    for (int i = 0; i < n_fft; ++i) frame_[i] = in_n_fft[i] * m_.window[i];
    dfn_rfft(fft_, frame_.data(), spec_re_.data(), spec_im_.data());

    // Un-masked spectrum into the deep-filter history (oldest at 0).
    {
        const int nf = df_hist_frames_;
        std::memmove(df_hist_re_.data(), df_hist_re_.data() + n_bins, sizeof(float) * n_bins * (nf - 1));
        std::memmove(df_hist_im_.data(), df_hist_im_.data() + n_bins, sizeof(float) * n_bins * (nf - 1));
        std::memcpy(df_hist_re_.data() + static_cast<size_t>(n_bins) * (nf - 1), spec_re_.data(), sizeof(float) * n_bins);
        std::memcpy(df_hist_im_.data() + static_cast<size_t>(n_bins) * (nf - 1), spec_im_.data(), sizeof(float) * n_bins);
    }

    // ERB band power (libdf scales the spectrum by wnorm = 2*hop/N^2 before
    // its features; the masking spectrum itself is left at native scale).
    const float wnorm = 2.0f * hop / (static_cast<float>(n_fft) * n_fft);
    erb_power_.assign(m_.n_erb, 0.0f);
    for (int b = 0; b < m_.n_erb; ++b) {
        const int s = m_.erb_indices[b], e = std::min(n_bins, s + m_.erb_widths[b]);
        float acc = 0.0f;
        for (int k = s; k < e; ++k) acc += spec_re_[k] * spec_re_[k] + spec_im_[k] * spec_im_[k];
        erb_power_[b] = acc / std::max(1, e - s) * wnorm * wnorm;
    }
    log_feat_.resize(m_.n_erb);
    if (!norm_seeded_) {
        for (int b = 0; b < m_.n_erb; ++b) {
            const float t = m_.n_erb <= 1 ? 0.0f : static_cast<float>(b) / (m_.n_erb - 1);
            log_mag_mean_[b] = kMeanNormHi + t * (kMeanNormLo - kMeanNormHi);
        }
        norm_seeded_ = true;
    }
    for (int b = 0; b < m_.n_erb; ++b) {
        const float dB = 10.0f * std::log10(erb_power_[b] + kLogEps);
        log_mag_mean_[b] = (1.0f - kEmaAlpha) * dB + kEmaAlpha * log_mag_mean_[b];
        log_feat_[b] = (dB - log_mag_mean_[b]) / kErbNormDiv;
    }

    // Per-bin unit norm of the wnorm-scaled spectrum (libdf band_unit_norm).
    if (!unit_norm_seeded_) {
        for (int f = 0; f < kEncSpec; ++f) {
            const float t = static_cast<float>(f) / (kEncSpec - 1);
            unit_norm_state_[f] = kUnitNormHi + t * (kUnitNormLo - kUnitNormHi);
        }
        unit_norm_seeded_ = true;
    }
    std::fill(feat_spec_flat_.begin(), feat_spec_flat_.end(), 0.0f);
    for (int f = 0; f < kEncSpec && f < n_bins; ++f) {
        const float re = spec_re_[f] * wnorm, im = spec_im_[f] * wnorm;
        const float mag = std::sqrt(re * re + im * im);
        unit_norm_state_[f] = (1.0f - kEmaAlpha) * mag + kEmaAlpha * unit_norm_state_[f];
        const float denom = std::sqrt(std::max(1e-12f, unit_norm_state_[f]));
        feat_spec_flat_[f]            = re / denom;
        feat_spec_flat_[kEncSpec + f] = im / denom;
    }

    // Lookahead alignment: upstream shifts the feature stream forward by two
    // frames (pad_feat (-2, 2)); with our left-aligned framing that is one
    // frame here. The EMAs still run on it, the encoder does not see it.
    if (frame_counter_ < kWarmupFrames) {
        ++frame_counter_;
        return;
    }
    ++frame_counter_;

    // Stage the encoder inputs: [t-2, t-1, t] for feat_erb and, per re/im
    // channel, for feat_spec.
    std::memcpy(feh_.data(), feat_erb_hist_.data(), sizeof(float) * 2 * kEncErb);
    std::memcpy(feh_.data() + 2 * kEncErb, log_feat_.data(), sizeof(float) * kEncErb);
    for (int c = 0; c < 2; ++c) {
        for (int t = 0; t < 2; ++t)
            std::memcpy(fsh_.data() + (static_cast<size_t>(c) * 3 + t) * kEncSpec,
                        feat_spec_hist_.data() + (static_cast<size_t>(c) * 2 + t) * kEncSpec, sizeof(float) * kEncSpec);
        std::memcpy(fsh_.data() + (static_cast<size_t>(c) * 3 + 2) * kEncSpec,
                    feat_spec_flat_.data() + static_cast<size_t>(c) * kEncSpec, sizeof(float) * kEncSpec);
    }

    // Network.
    gains_.assign(m_.n_erb, 0.0f);
    float alpha = 0.0f;
    dfn_fast_io io{};
    io.feat_erb  = feh_.data();
    io.feat_spec = fsh_.data();
    io.c0_hist   = c0_hist_.data();
    io.gru_h     = gru_hidden_.data();
    io.mask      = gains_.data();
    io.coefs     = coefs_flat_.data();
    io.alpha     = &alpha;
    io.c0_out    = c0_curr_.data();
    dfn_fast_forward(m_.fast, scratch_, io);

    coef_re_.assign(static_cast<size_t>(m_.df_order) * m_.df_bins, 0.0f);
    coef_im_.assign(static_cast<size_t>(m_.df_order) * m_.df_bins, 0.0f);
    for (int d = 0; d < m_.df_order; ++d)
        for (int f = 0; f < m_.df_bins; ++f) {
            coef_re_[static_cast<size_t>(d) * m_.df_bins + f] = coefs_flat_[static_cast<size_t>(f) * 10 + 2 * d];
            coef_im_[static_cast<size_t>(d) * m_.df_bins + f] = coefs_flat_[static_cast<size_t>(f) * 10 + 2 * d + 1];
        }

    // Slide the histories by one frame.
    std::memcpy(feat_erb_hist_.data(), feat_erb_hist_.data() + kEncErb, sizeof(float) * kEncErb);
    std::memcpy(feat_erb_hist_.data() + kEncErb, log_feat_.data(), sizeof(float) * kEncErb);
    for (int c = 0; c < 2; ++c) {
        float* dst = feat_spec_hist_.data() + static_cast<size_t>(c) * 2 * kEncSpec;
        std::memmove(dst, dst + kEncSpec, sizeof(float) * kEncSpec);
        std::memcpy(dst + kEncSpec, feat_spec_flat_.data() + static_cast<size_t>(c) * kEncSpec, sizeof(float) * kEncSpec);
    }
    for (int c = 0; c < 64; ++c) {
        float* dst = c0_hist_.data() + static_cast<size_t>(c) * 4 * kEncSpec;
        std::memmove(dst, dst + kEncSpec, sizeof(float) * 3 * kEncSpec);
        std::memcpy(dst + 3 * kEncSpec, c0_curr_.data() + static_cast<size_t>(c) * kEncSpec, sizeof(float) * kEncSpec);
    }

    // The network's outputs are for the frame `lookahead` back: take that
    // spectrum from the history as the synthesis target.
    {
        const int t_idx = df_hist_frames_ - 1 - m_.lookahead_frames;
        std::memcpy(spec_re_.data(), df_hist_re_.data() + static_cast<size_t>(t_idx) * n_bins, sizeof(float) * n_bins);
        std::memcpy(spec_im_.data(), df_hist_im_.data() + static_cast<size_t>(t_idx) * n_bins, sizeof(float) * n_bins);
    }
    // Stage 1: ERB gains on every bin.
    for (int b = 0; b < m_.n_erb; ++b) {
        const int s = m_.erb_indices[b], e = std::min(n_bins, s + m_.erb_widths[b]);
        for (int k = s; k < e; ++k) { spec_re_[k] *= gains_[b]; spec_im_[k] *= gains_[b]; }
    }
    // Stage 2: deep filter replaces the low bins. Tap n multiplies the
    // history frame current + n - (order-1-lookahead).
    {
        const int current_idx = df_hist_frames_ - 1 - m_.lookahead_frames;
        const int past_off    = m_.df_order - 1 - m_.lookahead_frames;
        const int df_bins     = std::min(m_.df_bins, n_bins);
        for (int f = 0; f < df_bins; ++f) {
            float y_re = 0.0f, y_im = 0.0f;
            for (int d = 0; d < m_.df_order; ++d) {
                const int src = current_idx + d - past_off;
                if (src < 0 || src >= df_hist_frames_) continue;
                const float x_re = df_hist_re_[static_cast<size_t>(src) * n_bins + f];
                const float x_im = df_hist_im_[static_cast<size_t>(src) * n_bins + f];
                const float c_re = coef_re_[static_cast<size_t>(d) * m_.df_bins + f];
                const float c_im = coef_im_[static_cast<size_t>(d) * m_.df_bins + f];
                y_re += c_re * x_re - c_im * x_im;
                y_im += c_re * x_im + c_im * x_re;
            }
            spec_re_[f] = y_re;
            spec_im_[f] = y_im;
        }
    }

    // Synthesis + overlap-add.
    dfn_irfft(fft_, spec_re_.data(), spec_im_.data(), frame_.data());
    for (int i = 0; i < n_fft; ++i) ola_[i] += frame_[i] * m_.window[i];
    out_queue_.insert(out_queue_.end(), ola_.begin(), ola_.begin() + hop);
    std::memmove(ola_.data(), ola_.data() + hop, sizeof(float) * (n_fft - hop));
    std::memset(ola_.data() + (n_fft - hop), 0, sizeof(float) * hop);
}

}  // namespace vv
