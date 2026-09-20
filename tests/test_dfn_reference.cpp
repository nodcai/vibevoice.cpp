// DFN3 post-filter parity against a reference implementation, plus the
// 2x upsampler and the FFT as self-contained checks.
//
// The reference gate is gated on three env vars and skips (77) otherwise:
//   VIBEVOICE_DFN_MODEL   gguf carrying dfn.* tensors (the merged model or
//                         a plain deepfilternet3.gguf)
//   VIBEVOICE_DFN_IN_WAV  48 kHz mono 16-bit input
//   VIBEVOICE_DFN_REF_WAV the same input denoised by a reference
//                         implementation of the filter, 100 warm-up frames
// The reference tool runs its input through miniaudio's sinc resampler even
// at 48 -> 48 kHz, which delays the signal by one sample and adds ~1e-3 of
// passband ripple, with a startup transient in the first second. So the gate
// aligns the best lag within +-2 samples, skips the first second, and asks
// for < 3e-3 relative RMS and < 16 LSB max after that. Our port of the same
// filter code measured 3-7e-4 and 2-7 LSB.
#include "audio_io.hpp"
#include "dfn.hpp"
#include "dfn_fft.hpp"
#include "model_loader.hpp"
#include "resample2x.hpp"
#include "vibevoice.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int check_fft() {
    const int n = 960;
    dfn_fft_plan* p = dfn_fft_plan_create(n);
    if (!p) { std::fprintf(stderr, "FAIL: fft plan\n"); return 1; }
    std::vector<float> x(n), re(n / 2 + 1), im(n / 2 + 1), y(n);
    for (int i = 0; i < n; ++i) x[i] = std::sin(0.37 * i) + 0.5f * std::cos(1.9 * i + 0.3) + 0.1f * ((i * 7919) % 13 - 6);
    dfn_rfft(p, x.data(), re.data(), im.data());
    // naive DFT for a few bins
    double max_err = 0;
    for (int k : {0, 1, 7, 100, 333, 479, 480}) {
        double sr = 0, si = 0;
        for (int i = 0; i < n; ++i) { sr += x[i] * std::cos(2 * M_PI * k * i / n); si -= x[i] * std::sin(2 * M_PI * k * i / n); }
        max_err = std::max(max_err, std::fabs(sr - re[k]) + std::fabs(si - im[k]));
    }
    dfn_irfft(p, re.data(), im.data(), y.data());
    double rt = 0;
    for (int i = 0; i < n; ++i) rt = std::max(rt, static_cast<double>(std::fabs(y[i] - x[i])));
    dfn_fft_plan_free(p);
    std::printf("fft: max DFT err %.2e, roundtrip err %.2e\n", max_err, rt);
    return (max_err < 2e-3 && rt < 1e-5) ? 0 : 2;
}

int check_upsampler() {
    // A 1 kHz tone at 24 kHz must come out as a 1 kHz tone at 48 kHz with
    // unity gain and no image at 23 kHz.
    vv::Upsampler2x up;
    const int n = 4800;
    std::vector<float> x(n), y;
    for (int i = 0; i < n; ++i) x[i] = std::sin(2 * M_PI * 1000.0 * i / 24000.0);
    // push in odd-sized pieces to exercise the history
    for (int off = 0; off < n;) { const int c = std::min(n - off, 1 + (off * 31) % 700); up.process(x.data() + off, c, &y); off += c; }
    if (y.size() != static_cast<size_t>(2 * n)) { std::fprintf(stderr, "FAIL: upsampler length %zu\n", y.size()); return 3; }
    // steady-state rms over the last half, and the power in the image band
    double rms = 0, img_re = 0, img_im = 0, tone_re = 0, tone_im = 0;
    const int m = n;  // last n samples at 48k
    for (int i = m; i < 2 * n; ++i) {
        rms += y[i] * y[i];
        img_re += y[i] * std::cos(2 * M_PI * 23000.0 * i / 48000.0);  img_im += y[i] * std::sin(2 * M_PI * 23000.0 * i / 48000.0);
        tone_re += y[i] * std::cos(2 * M_PI * 1000.0 * i / 48000.0); tone_im += y[i] * std::sin(2 * M_PI * 1000.0 * i / 48000.0);
    }
    rms = std::sqrt(rms / m);
    const double tone = 2 * std::sqrt(tone_re * tone_re + tone_im * tone_im) / m;
    const double img  = 2 * std::sqrt(img_re * img_re + img_im * img_im) / m;
    std::printf("upsampler: rms %.4f (want 0.7071), tone %.4f, image %.2e (%.1f dB)\n", rms, tone, img, 20 * std::log10(img / tone));
    return (std::fabs(rms - 0.70711) < 1e-3 && std::fabs(tone - 1.0) < 2e-3 && img / tone < 3e-4) ? 0 : 4;
}

}  // namespace

int main() {
    int rc = check_fft();
    if (rc) return rc;
    rc = check_upsampler();
    if (rc) return rc;

    const char* model_env = std::getenv("VIBEVOICE_DFN_MODEL");
    const char* in_env    = std::getenv("VIBEVOICE_DFN_IN_WAV");
    const char* ref_env   = std::getenv("VIBEVOICE_DFN_REF_WAV");
    if (!model_env || !in_env || !ref_env) {
        std::printf("fft + upsampler OK; skipping the reference gate (set VIBEVOICE_DFN_MODEL/_IN_WAV/_REF_WAV)\n");
        return 77;
    }
    vv::ModelLoader loader;
    if (!loader.load(model_env)) { std::fprintf(stderr, "FAIL: load %s\n", model_env); return 5; }
    vv::DfnModel dfn;
    if (!vv::dfn_model_load(loader, &dfn)) { std::fprintf(stderr, "FAIL: dfn load\n"); return 6; }

    vv_audio in{}, ref{};
    if (vv_load_wav(in_env, &in) != VV_OK || vv_load_wav(ref_env, &ref) != VV_OK) { std::fprintf(stderr, "FAIL: wav load\n"); return 7; }
    if (in.sample_rate != 48000 || ref.sample_rate != 48000) { std::fprintf(stderr, "FAIL: wavs must be 48 kHz\n"); return 8; }

    vv::DfnStream st(dfn);
    st.warmup(100);
    std::vector<float> out;
    const int n = static_cast<int>(in.n_samples);
    for (int off = 0; off < n;) { const int c = std::min(n - off, 3000 + (off * 17) % 2000); st.process(in.samples + off, c, &out, false); off += c; }
    st.process(nullptr, 0, &out, true);

    // best lag in [-2, 2] by normalised correlation
    const long cmp = static_cast<long>(std::min(out.size(), ref.n_samples));
    int best_lag = 0; double best_c = -2;
    for (int lag = -2; lag <= 2; ++lag) {
        double dot = 0, na = 0, nb = 0;
        for (long i = 48000; i < cmp - 2; ++i) {
            const double x = out[static_cast<size_t>(i + lag)], y = ref.samples[i];
            dot += x * y; na += x * x; nb += y * y;
        }
        const double c = dot / std::sqrt(std::max(na * nb, 1e-24));
        if (c > best_c) { best_c = c; best_lag = lag; }
    }
    double max_abs = 0, sd = 0, sr = 0;
    for (long i = 48000; i < cmp - 2; ++i) {
        const double d = out[static_cast<size_t>(i + best_lag)] - ref.samples[i];
        max_abs = std::max(max_abs, std::fabs(d)); sd += d * d; sr += ref.samples[i] * (double)ref.samples[i];
    }
    const double rel = std::sqrt(sd / std::max(sr, 1e-12));
    std::printf("dfn reference: ours %zu samples, ref %zu; lag %d (corr %.7f); after 1 s: max_abs %.2e (%.1f LSB16), rel_rmse %.2e\n",
                out.size(), ref.n_samples, best_lag, best_c, max_abs, max_abs * 32768, rel);
    if (const char* dump = std::getenv("VIBEVOICE_DFN_OUT_WAV")) {
        vv_audio a{}; a.samples = out.data(); a.n_samples = out.size(); a.sample_rate = 48000; a.channels = 1;
        vv_save_wav(dump, &a);
    }
    const bool same_len = out.size() == ref.n_samples;
    vv_audio_free(&in); vv_audio_free(&ref);
    if (!same_len) { std::fprintf(stderr, "FAIL: length mismatch\n"); return 9; }
    if (max_abs * 32768 > 16.0 || rel > 3e-3) { std::fprintf(stderr, "FAIL: diverges from the reference\n"); return 10; }
    std::printf("OK\n");
    return 0;
}
