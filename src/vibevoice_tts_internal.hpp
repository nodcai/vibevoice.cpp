#ifndef VIBEVOICE_TTS_INTERNAL_HPP
#define VIBEVOICE_TTS_INTERNAL_HPP

// Helpers shared between the TTS translation units (vibevoice_tts.cpp owns
// them; tts_realtime_loop.cpp consumes them). Not part of the public API.

#include "qwen2.hpp"
#include "vibevoice_tts.hpp"

#include <vector>

namespace vv {
namespace detail {

// One forward pass through a Qwen2 stack with a resident K/V cache.
// `inputs_embeds`: [hidden, n_new_tokens]; new K/V are written at
// kvs->past_len, which is then advanced.
bool run_qwen2_stack(struct ggml_context*                   ctx_ext,
                     const VibeVoiceConfig&                 cfg,
                     const std::vector<Qwen2LayerWeights>&  layers,
                     struct ggml_tensor*                    output_norm,
                     int                                    pos_start,
                     int                                    n_new_tokens,
                     const float*                           inputs_embeds,
                     ResidentKV*                            kvs,
                     std::vector<float>*                    out_hidden,
                     std::vector<float>*                    out_hidden_last);

// SpeechConnector: fc2(rmsnorm(fc1(x))). x: [latent, batch] → [hidden, batch].
std::vector<float> run_speech_connector(const VibeVoiceConfig&  cfg,
                                        const VibeVoiceWeights& w,
                                        const float*            x,
                                        int                     batch);

// sigmoid(fc2(relu(fc1(hidden)))).
float run_eos_classifier(const VibeVoiceConfig&  cfg,
                         const VibeVoiceWeights& w,
                         const float*            x_hidden);

// Adds tts_input_types[type_id] (0 = speech, 1 = text) into x [hidden * n_tokens].
void add_input_type_embedding(const VibeVoiceConfig&  cfg,
                              const VibeVoiceWeights& w,
                              int                     n_tokens,
                              int                     type_id,
                              float*                  x);

// Frame-major [n_frames][latent] → the legacy decoder's [latent][n_frames].
std::vector<float> pack_latents_ggml_order(const float* latents, int n_frames, int latent);

}  // namespace detail
}  // namespace vv

#endif  // VIBEVOICE_TTS_INTERNAL_HPP
