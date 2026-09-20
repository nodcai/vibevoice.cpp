// dfn_fast.hpp — hand-written single-frame (T=1) DeepFilterNet3 forward.
// The GGML graph it mirrors is not part of this port; the fast path is the
// only forward here.
//
// The streaming post-filter runs the whole network once per 10 ms
// hop. Expressed as a GGML graph that is ~250 tiny nodes — im2col,
// per-frame weight casts, permutes, thread barriers — and the actual
// arithmetic (≈3.3 M multiply-adds) is a small fraction of the wall
// clock. This file is the same network written as plain loops over
// weights that were converted to F32 and laid out once at load time.
//
// Layouts follow the GGML builders in `dfn.cpp` exactly (activation
// maps are `[channel][freq]`, C-contiguous) so the two paths can be
// compared frame-by-frame (`DFN_CHECK=1`). The GGML path stays as the
// reference for the golden tests and as a fallback (`DFN_GGML=1`).

#pragma once

#include "ggml.h"

#include <map>
#include <string>

struct dfn_fast_weights;
struct dfn_fast_scratch;

// Convert + re-layout the GGUF tensors. Returns NULL (and sets `err`)
// if any tensor is missing or has an unexpected element count.
dfn_fast_weights* dfn_fast_prepare(const std::map<std::string, ggml_tensor*>& tensors, std::string* err);
void              dfn_fast_free_weights(dfn_fast_weights* w);

// Per-stream intermediate buffers (allocated once, no per-frame heap).
dfn_fast_scratch* dfn_fast_scratch_create();
void              dfn_fast_scratch_free(dfn_fast_scratch* s);

struct dfn_fast_io {
    const float* feat_erb;  // [3][32]      frames t-2, t-1, t (log-ERB features)
    const float* feat_spec; // [2][3][96]   (re/im) × frames × bins, unit-normed
    const float* c0_hist;   // [64][4][96]  previous 4 frames of the encoder's c0
    float*       gru_h;     // [5][256]     in/out: enc, erb_dec#1, erb_dec#2, df_dec#1, df_dec#2
    float*       mask;      // [32]         ERB gains
    float*       coefs;     // [96][10]     DF coefs, (re, im) interleaved per tap
    float*       alpha;     // [1]
    float*       c0_out;    // [64][96]     this frame's c0 (caller pushes into c0_hist)
};

void dfn_fast_forward(const dfn_fast_weights* w, dfn_fast_scratch* s, const dfn_fast_io& io);
