#ifndef VIBEVOICE_ACOUSTIC_DECODER_V2_HPP
#define VIBEVOICE_ACOUSTIC_DECODER_V2_HPP

// Stateful σ-VAE acoustic decoder.
//
// Same network as decoder_forward (acoustic_tokenizer.cpp) — stem conv,
// (transposed-conv upsample → ConvNeXt Block1D stage) × N, head conv — but
// built from ops that are cheap on every ggml backend and that carry their
// left context across calls:
//
//   * dense causal conv K=7 (stem, head)   →  Σ_k mul_mat(W_k, shift_k(x))
//   * depthwise causal conv K=7 (mixer)    →  Σ_k mul(shift_k(x), w_k)
//   * transposed conv, K = 2·stride        →  one mul_mat [C_in → 2s·C_out]
//                                             + overlap-add of the two halves
//
// Every conv keeps its left context in a small persistent tensor that is
// concatenated in front of the chunk and rewritten inside the graph (the
// KV-cache pattern). Decoding a stream chunk by chunk therefore gives the
// same signal as decoding it all at once, without re-decoding any overlap.
//
// Why not ggml_conv_transpose_1d: on Metal it is either pathologically slow
// or never returns for the 3200× upsample this decoder does, and the left
// pad the causal convs need is not supported there at all.
//
// Layout inside this module is channels-first, [C, T] in ggml ne order, so a
// frame-major latent buffer ([n_frames][vae_dim] on the host) uploads as-is.
// Weights are re-laid once (decoder_v2_prepare); state is per stream.

#include "acoustic_tokenizer.hpp"
#include "ggml-backend.h"
#include "ggml.h"

#include <vector>

namespace vv {

struct DecoderV2Weights {
    ggml_context*         ctx = nullptr;   // owns the re-laid tensors' metadata
    ggml_backend_buffer_t buf = nullptr;   // owns their data

    int              n_stages = 0;
    int              c_in     = 0;         // latent dim (stem input channels)
    std::vector<int> C;                    // channels per stage
    std::vector<int> depth;                // Block1D count per stage
    std::vector<int> stride;               // upsample factor per stage (0 for stage 0)

    struct ggml_tensor*              stem_w = nullptr;  // [C_in, C0, 7] F16
    struct ggml_tensor*              head_w = nullptr;  // [C_last, 1, 7] F16
    std::vector<struct ggml_tensor*> tr_w;              // per stage s>=1: [C_{s-1}, 2·stride·C_s] F16
    std::vector<struct ggml_tensor*> dw_w;              // per block (flat): [C, 7] F32

    // Borrowed from DecoderWeights (not owned).
    struct ggml_tensor*                  stem_b = nullptr;
    struct ggml_tensor*                  head_b = nullptr;
    std::vector<struct ggml_tensor*>     tr_b;           // per stage
    std::vector<const Block1DWeights*>   blocks;         // flat, stage-major
    struct ggml_tensor*                  final_norm = nullptr;

    bool ready = false;

    DecoderV2Weights() = default;
    DecoderV2Weights(const DecoderV2Weights&)            = delete;
    DecoderV2Weights& operator=(const DecoderV2Weights&) = delete;
    ~DecoderV2Weights();
    void free();
};

struct DecoderV2State {
    ggml_context*         ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    struct ggml_tensor*              stem_state = nullptr;  // [C_in, 6]
    struct ggml_tensor*              head_state = nullptr;  // [C_last, 6]
    std::vector<struct ggml_tensor*> dw_state;              // per block: [C, 6]
    std::vector<struct ggml_tensor*> tr_state;              // per stage: [C_s, stride, 1]
    bool ready = false;

    DecoderV2State() = default;
    DecoderV2State(const DecoderV2State&)            = delete;
    DecoderV2State& operator=(const DecoderV2State&) = delete;
    ~DecoderV2State();
    void free();
};

// True unless VIBEVOICE_VAE_V2=0 is set (A/B switch for the original graph).
bool decoder_v2_enabled();

// Re-lay the decoder weights for this module. Returns false (leaving *out
// unready) if the decoder does not have the expected shape — every
// transposed conv must have K == 2·stride.
bool decoder_v2_prepare(const DecoderWeights& w, const AcousticConfig& cfg,
                        DecoderV2Weights* out);

// Allocate zeroed conv state for one stream. reset() zeroes it again.
bool decoder_v2_state_init(const DecoderV2Weights& w, DecoderV2State* st);
void decoder_v2_state_reset(DecoderV2State& st);

// Decode `n_frames` latents (frame-major [n_frames][c_in], already scaled)
// continuing from `st`, appending 24 kHz samples to *audio_out. Internally
// runs one graph per `chunk_frames` so the activation footprint is bounded.
bool decoder_v2_decode(const DecoderV2Weights& w, DecoderV2State& st,
                       const float* latents, int n_frames,
                       std::vector<float>* audio_out, int chunk_frames = 32);

}  // namespace vv

#endif  // VIBEVOICE_ACOUSTIC_DECODER_V2_HPP
