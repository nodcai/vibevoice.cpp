// dfn_fft.cpp — mixed-radix FFT used by the DFN3 streaming STFT.
//
// Real transforms of even length n go through one complex FFT of
// length n/2 (even samples in the real part, odd samples in the
// imaginary part) plus an O(n) split/merge pass — half the work of a
// full complex transform. Odd lengths fall back to a plain complex
// FFT. The complex FFT is a recursive decimation-in-time
// Cooley–Tukey over radices {2, 3, 5}; all twiddles — the length-n/2
// table, the split twiddles and the tiny per-radix DFT matrices — are
// computed once at plan creation, so the per-call path is pure
// multiply-adds.
//
// At DFN's n = 960 one forward + one inverse transform costs ~10 µs.

#include "dfn_fft.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

struct cpx {
    float re, im;
};

inline cpx cmul(cpx a, cpx b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
inline cpx cconj(cpx a) { return {a.re, -a.im}; }

// Factorise `n` into {5, 3, 2} (largest first so the small radices sit
// in the innermost stages). Returns false on any other prime factor.
bool factorise(int n, std::vector<int>& factors) {
    factors.clear();
    if (n <= 0) return false;
    for (int r : {5, 3, 2}) {
        while (n % r == 0) {
            factors.push_back(r);
            n /= r;
        }
    }
    return n == 1;
}

} // namespace

struct dfn_fft_plan {
    int  n    = 0;     // real length
    int  m    = 0;     // complex transform length (n/2 when even, else n)
    bool half = false; // real-via-half-length path

    std::vector<int> factors;      // of m
    std::vector<cpx> tw;           // W_m^k  (forward sign), k = 0..m-1
    std::vector<cpx> tw_split;     // W_n^k  (forward sign), k = 0..m   (half path only)
    std::vector<cpx> radix_tab[6]; // radix_tab[r][j*r + q] = e^{-2πi jq/r}

    // Per-call scratch, length m each.
    mutable std::vector<cpx> buf_in;
    mutable std::vector<cpx> buf_out;

    void fft_rec(const cpx* in, cpx* out, int len, int stride, bool inverse, int fi) const;
};

// Length-`r` DFT of `s` written to out[j * out_stride]. Table lookups
// only (r ≤ 5); radix-2 is special-cased since it needs no table.
static inline void small_dft(const cpx* s, int r, cpx* out, int out_stride, const cpx* tab, bool inverse) {
    if (r == 2) {
        out[0]          = {s[0].re + s[1].re, s[0].im + s[1].im};
        out[out_stride] = {s[0].re - s[1].re, s[0].im - s[1].im};
        return;
    }
    for (int j = 0; j < r; ++j) {
        cpx acc = s[0]; // tab[j*r + 0] == 1
        for (int q = 1; q < r; ++q) {
            cpx w = tab[j * r + q];
            if (inverse) w = cconj(w);
            acc.re += s[q].re * w.re - s[q].im * w.im;
            acc.im += s[q].re * w.im + s[q].im * w.re;
        }
        out[(size_t)j * out_stride] = acc;
    }
}

void dfn_fft_plan::fft_rec(const cpx* in, cpx* out, int len, int stride, bool inverse, int fi) const {
    const int  r   = factors[fi];
    const int  sub = len / r;
    const cpx* tab = radix_tab[r].data();
    cpx        s[5];

    if (sub == 1) {
        // Leaf: a bare radix-r DFT over strided input.
        for (int q = 0; q < r; ++q) s[q] = in[(size_t)q * stride];
        small_dft(s, r, out, 1, tab, inverse);
        return;
    }

    for (int q = 0; q < r; ++q)
        fft_rec(in + (size_t)q * stride, out + (size_t)q * sub, sub, stride * r, inverse, fi + 1);

    // Combine: X[j·sub + k] = Σ_q W_len^{kq} · W_r^{jq} · Y_q[k].
    // W_len^{kq} = W_m^{kq·stride}; kq·stride < m so no modulo needed.
    for (int k = 0; k < sub; ++k) {
        s[0] = out[k];
        for (int q = 1; q < r; ++q) {
            cpx w = tw[(size_t)k * q * stride];
            if (inverse) w = cconj(w);
            s[q] = cmul(out[(size_t)q * sub + k], w);
        }
        small_dft(s, r, out + k, sub, tab, inverse);
    }
}

extern "C" struct dfn_fft_plan* dfn_fft_plan_create(int n) {
    if (n <= 0) return nullptr;
    auto* p = new dfn_fft_plan();
    p->n    = n;
    p->half = (n % 2 == 0) && (n >= 4);
    p->m    = p->half ? n / 2 : n;
    if (!factorise(p->m, p->factors)) {
        delete p;
        return nullptr;
    }
    const int m = p->m;
    p->tw.resize(m);
    for (int k = 0; k < m; ++k) {
        double ang   = (-kTwoPi * k) / m;
        p->tw[k].re = (float)std::cos(ang);
        p->tw[k].im = (float)std::sin(ang);
    }
    if (p->half) {
        p->tw_split.resize(m + 1);
        for (int k = 0; k <= m; ++k) {
            double ang         = (-kTwoPi * k) / n;
            p->tw_split[k].re = (float)std::cos(ang);
            p->tw_split[k].im = (float)std::sin(ang);
        }
    }
    for (int r : {2, 3, 5}) {
        p->radix_tab[r].resize((size_t)r * r);
        for (int j = 0; j < r; ++j) {
            for (int q = 0; q < r; ++q) {
                double ang                    = (-kTwoPi * ((j * q) % r)) / r;
                p->radix_tab[r][j * r + q].re = (float)std::cos(ang);
                p->radix_tab[r][j * r + q].im = (float)std::sin(ang);
            }
        }
    }
    p->buf_in.resize(m);
    p->buf_out.resize(m);
    return p;
}

extern "C" void dfn_fft_plan_free(struct dfn_fft_plan* p) { delete p; }

extern "C" int dfn_fft_plan_n(const struct dfn_fft_plan* p) { return p ? p->n : 0; }

extern "C" void dfn_rfft(const struct dfn_fft_plan* p, const float* in, float* out_re, float* out_im) {
    if (!p || !in || !out_re || !out_im) return;
    const int n      = p->n;
    const int m      = p->m;
    cpx*      z      = p->buf_in.data();
    cpx*      Z      = p->buf_out.data();
    const int n_bins = n / 2 + 1;

    if (!p->half) {
        for (int i = 0; i < n; ++i) z[i] = {in[i], 0.0f};
        p->fft_rec(z, Z, m, 1, /*inverse=*/false, 0);
        for (int k = 0; k < n_bins; ++k) {
            out_re[k] = Z[k].re;
            out_im[k] = Z[k].im;
        }
        return;
    }

    // Pack even/odd samples into one complex sequence of length m.
    for (int i = 0; i < m; ++i) z[i] = {in[2 * i], in[2 * i + 1]};
    p->fft_rec(z, Z, m, 1, /*inverse=*/false, 0);

    // Split: E = FFT(even), O = FFT(odd) from Z, then X[k] = E + W_n^k·O.
    for (int k = 0; k <= m; ++k) {
        const cpx zk  = Z[k == m ? 0 : k];
        const cpx zmk = cconj(Z[(m - k) % m]);
        const cpx E   = {0.5f * (zk.re + zmk.re), 0.5f * (zk.im + zmk.im)};
        const cpx D   = {zk.re - zmk.re, zk.im - zmk.im};
        const cpx O   = {0.5f * D.im, -0.5f * D.re}; // D / (2i)
        const cpx WO  = cmul(p->tw_split[k], O);
        out_re[k]     = E.re + WO.re;
        out_im[k]     = E.im + WO.im;
    }
}

extern "C" void dfn_irfft(const struct dfn_fft_plan* p, const float* in_re, const float* in_im, float* out) {
    if (!p || !in_re || !in_im || !out) return;
    const int n      = p->n;
    const int m      = p->m;
    cpx*      Z      = p->buf_in.data();
    cpx*      z      = p->buf_out.data();
    const int n_bins = n / 2 + 1;

    if (!p->half) {
        for (int k = 0; k < n_bins; ++k) Z[k] = {in_re[k], in_im[k]};
        for (int k = n_bins; k < n; ++k) Z[k] = {in_re[n - k], -in_im[n - k]};
        p->fft_rec(Z, z, m, 1, /*inverse=*/true, 0);
        const float scale = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) out[i] = z[i].re * scale;
        return;
    }

    // Merge: E[k] = (X[k] + conj X[m-k]) / 2,  O[k] = (X[k] - conj X[m-k]) / 2 · W_n^{-k},
    // Z[k] = E[k] + i·O[k]. One inverse complex FFT of length m then
    // yields even samples in the real part, odd in the imaginary part.
    //
    // The DC and Nyquist bins of a real signal's spectrum are real; if
    // the caller hands us an imaginary part there (the deep filter's
    // complex coefficients do) it is ignored — the same convention as
    // a full-length complex inverse that keeps only the real output,
    // and as numpy's irfft.
    for (int k = 0; k < m; ++k) {
        const bool edge = (k == 0); // touches X[0] and X[m]
        const cpx  xk   = {in_re[k], edge ? 0.0f : in_im[k]};
        const cpx  xmk  = {in_re[m - k], edge ? 0.0f : -in_im[m - k]};
        const cpx E   = {0.5f * (xk.re + xmk.re), 0.5f * (xk.im + xmk.im)};
        const cpx D   = {0.5f * (xk.re - xmk.re), 0.5f * (xk.im - xmk.im)};
        const cpx O   = cmul(D, cconj(p->tw_split[k]));
        Z[k]          = {E.re - O.im, E.im + O.re};
    }
    p->fft_rec(Z, z, m, 1, /*inverse=*/true, 0);
    const float scale = 1.0f / (float)m;
    for (int i = 0; i < m; ++i) {
        out[2 * i]     = z[i].re * scale;
        out[2 * i + 1] = z[i].im * scale;
    }
}
