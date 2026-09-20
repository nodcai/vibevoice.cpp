#ifndef VIBEVOICE_TTS_FRAME_GRAPH_HPP
#define VIBEVOICE_TTS_FRAME_GRAPH_HPP

// One ggml graph per speech frame for the realtime-0.5B TTS loop.
//
// The unfused loop launches, per frame, one diffusion-head graph per
// DPM-Solver++ step (two under CFG), then the connector, the positive
// TTS-LM step, the negative TTS-LM step and the EOS head, each as its own
// graph with a device round trip in between. This builds the whole frame
// as a single graph:
//
//   * every solver step with the CFG pair batched as two "frames" of the
//     diffusion head, the CFG blend and the solver update as graph ops
//   * the acoustic connector plus the speech type embedding
//   * the TTS-LM step for the positive and negative sequences as two
//     columns, so the layer weights are read once for both; each column
//     attends to its own resident K/V cache
//   * the EOS classifier on the positive hidden state
//
// Numerically the same as the unfused path modulo op ordering (frame 0
// latents agree to ~1e-6). VIBEVOICE_FUSED=0 restores the unfused loop.

#include "bench.hpp"
#include "diffusion_head.hpp"
#include "dpm_solver.hpp"
#include "qwen2.hpp"
#include "vibevoice_tts.hpp"

#include <vector>

namespace vv {

bool fused_frame_enabled();

struct FusedFrameInputs {
    const float* z0        = nullptr;   // [latent] initial Gaussian noise
    const float* cond_pos  = nullptr;   // [hidden] positive condition (TTS-LM last hidden)
    const float* cond_neg  = nullptr;   // [hidden] negative condition, or null for no CFG
    float        cfg_scale = 1.0f;
    const float* stype     = nullptr;   // [hidden] speech input-type embedding
};

struct FusedFrameOutputs {
    std::vector<float> latent;      // [latent] denoised speech latent
    std::vector<float> hidden_pos;  // [hidden] TTS-LM last hidden after consuming the frame
    std::vector<float> hidden_neg;  // [hidden] same for the negative sequence (empty if no CFG)
    float              eos_logit = 0.0f;
};

// Runs one frame. Advances kv_tlm.past_len (and kv_neg->past_len when CFG
// is on) by one on success. `bench` may be null.
bool run_fused_frame(const VibeVoiceConfig&      cfg,
                     const VibeVoiceWeights&     w,
                     const DiffusionHeadConfig&  dh_cfg,
                     const DPMSolverConfig&      solver_cfg,
                     const DPMSolverState&       solver_state,
                     ResidentKV&                 kv_tlm,
                     ResidentKV*                 kv_neg,
                     const FusedFrameInputs&     in,
                     FusedFrameOutputs*          out,
                     BenchTotals*                bench);

}  // namespace vv

#endif  // VIBEVOICE_TTS_FRAME_GRAPH_HPP
