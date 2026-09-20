#ifndef VIBEVOICE_RESAMPLE2X_HPP
#define VIBEVOICE_RESAMPLE2X_HPP

// Stateful exact-ratio 2x upsampler (24 kHz -> 48 kHz for the DFN3
// post-filter). Windowed-sinc lowpass at a quarter of the output rate,
// evaluated as two polyphase branches so no zero-stuffed samples are ever
// multiplied. Unity passband gain; the filter's group delay is
// (taps/2 - 1) input samples, about 0.65 ms, which the stream absorbs.
//
// Only the fixed 1:2 ratio the post-filter needs: the general-purpose
// rational resampler this replaces (miniaudio) is 20k lines
// for one call site, and a decimation-gain bug in another project's
// polyphase resampler is exactly the kind of thing a fixed 2x path cannot
// have.

#include <vector>

namespace vv {

class Upsampler2x {
public:
    // `taps_per_phase` FIR taps per output phase (2x that in total).
    explicit Upsampler2x(int taps_per_phase = 32);

    // Push `n` input samples; appends 2n output samples to *out.
    void process(const float* in, int n, std::vector<float>* out);
    // Forget the history (new utterance).
    void reset();

    int taps_per_phase() const { return P_; }

private:
    int                P_;
    std::vector<float> h0_, h1_;   // polyphase taps: output 2k uses h0, 2k+1 uses h1
    std::vector<float> hist_;      // last P_-1 inputs, oldest first
};

}  // namespace vv

#endif  // VIBEVOICE_RESAMPLE2X_HPP
