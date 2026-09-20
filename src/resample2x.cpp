#include "resample2x.hpp"

#include <cmath>

namespace vv {

namespace {

// Zeroth-order modified Bessel function, for the Kaiser window.
double bessel_i0(double x) {
    double sum = 1.0, term = 1.0;
    const double y = x * x / 4.0;
    for (int k = 1; k < 60; ++k) {
        term *= y / (static_cast<double>(k) * k);
        sum += term;
        if (term < 1e-14 * sum) break;
    }
    return sum;
}

}  // namespace

Upsampler2x::Upsampler2x(int taps_per_phase) : P_(taps_per_phase < 4 ? 4 : taps_per_phase) {
    // Prototype lowpass at the 2x rate: cutoff = 0.25 (cycles/sample),
    // length L = 2P, Kaiser beta 9 (~-90 dB stopband). Split into the two
    // output phases and normalise each to unity DC gain, which also
    // removes the x2 interpolation gain.
    const int    L    = 2 * P_;
    const double beta = 9.0;
    const double i0b  = bessel_i0(beta);
    std::vector<double> h(L);
    for (int m = 0; m < L; ++m) {
        const double t = m - (L - 1) / 2.0;
        const double sinc = (t == 0.0) ? 1.0 : std::sin(M_PI * 0.5 * t) / (M_PI * 0.5 * t);
        const double r = 2.0 * t / (L - 1);
        const double win = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;
        h[m] = sinc * win;
    }
    h0_.assign(P_, 0.0f);
    h1_.assign(P_, 0.0f);
    double s0 = 0, s1 = 0;
    for (int j = 0; j < P_; ++j) { s0 += h[2 * j]; s1 += h[2 * j + 1]; }
    for (int j = 0; j < P_; ++j) {
        h0_[j] = static_cast<float>(h[2 * j] / s0);
        h1_[j] = static_cast<float>(h[2 * j + 1] / s1);
    }
    reset();
}

void Upsampler2x::reset() { hist_.assign(static_cast<size_t>(P_ - 1), 0.0f); }

void Upsampler2x::process(const float* in, int n, std::vector<float>* out) {
    if (!in || n <= 0 || !out) return;
    // Work buffer: history followed by the new input, so every tap window
    // is contiguous. y[2k+phase] = sum_j h_phase[j] * x[k - j].
    std::vector<float> buf;
    buf.reserve(hist_.size() + static_cast<size_t>(n));
    buf.insert(buf.end(), hist_.begin(), hist_.end());
    buf.insert(buf.end(), in, in + n);
    const size_t base = out->size();
    out->resize(base + static_cast<size_t>(2) * n);
    float* y = out->data() + base;
    for (int k = 0; k < n; ++k) {
        // x[k - j] is buf[(P_-1) + k - j]
        const float* x = buf.data() + (P_ - 1) + k;
        float a0 = 0.0f, a1 = 0.0f;
        for (int j = 0; j < P_; ++j) {
            const float xv = x[-j];
            a0 += h0_[j] * xv;
            a1 += h1_[j] * xv;
        }
        y[2 * k]     = a0;
        y[2 * k + 1] = a1;
    }
    // Keep the last P_-1 inputs.
    hist_.assign(buf.end() - (P_ - 1), buf.end());
}

}  // namespace vv
