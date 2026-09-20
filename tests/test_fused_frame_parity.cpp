// Fused per-frame graph vs the unfused sampler, on real weights. Gated:
// needs VIBEVOICE_TTS_MODEL and VIBEVOICE_VOICE (realtime-0.5b).
//
// The latent produced by run_fused_frame depends only on (noise, cond,
// timesteps), so it is compared against dpm_solver_sample on the same
// inputs, with CFG off and on, for several step counts. The LM step inside
// the fused graph runs on an empty cache here; it does not influence the
// latent, and its own parity is covered by the frame-by-frame latent
// comparison of full runs (see the Phase 2 commit message).
#include "backend.hpp"
#include "dpm_solver.hpp"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "tts_frame_graph.hpp"
#include "vibevoice_tts.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
double rel_rmse(const std::vector<float>& a, const std::vector<float>& b) {
    double sd = 0, sr = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - b[i];
        sd += d * d; sr += static_cast<double>(a[i]) * a[i];
    }
    return std::sqrt(sd / a.size()) / std::max(std::sqrt(sr / a.size()), 1e-12);
}
}  // namespace

int main() {
    const char* model_env = std::getenv("VIBEVOICE_TTS_MODEL");
    const char* voice_env = std::getenv("VIBEVOICE_VOICE");
    if (!model_env || !*model_env || !voice_env || !*voice_env) {
        std::fprintf(stderr, "skip: set VIBEVOICE_TTS_MODEL and VIBEVOICE_VOICE\n");
        return 77;
    }
    vv::VibeVoiceModel model;
    if (!vv::vibevoice_load(model_env, &model)) { std::fprintf(stderr, "FAIL: load\n"); return 1; }
    if (model.variant != "realtime-0.5b") { std::fprintf(stderr, "FAIL: want realtime-0.5b\n"); return 2; }
    vv::VibeVoiceVoice voice;
    if (!vv::vibevoice_voice_load(voice_env, model, &voice)) { std::fprintf(stderr, "FAIL: voice\n"); return 3; }

    const auto& cfg = model.cfg;
    const auto& w   = model.w;
    const int latent = cfg.latent, hidden = cfg.hidden;

    vv::DiffusionHeadConfig dh_cfg;
    dh_cfg.hidden = hidden; dh_cfg.latent = latent; dh_cfg.head_layers = cfg.head_layers;
    dh_cfg.ffn_ratio = cfg.ffn_ratio; dh_cfg.eps = cfg.rms_norm_eps; dh_cfg.freq_size = 256;

    std::vector<float> speech_type(static_cast<size_t>(hidden), 0.0f);
    // tts_input_types row 0 (speech); its value does not affect the latent.

    std::mt19937 rng(777);
    std::normal_distribution<float> nd(0.f, 1.f);

    // The CPU backend picks different matmul kernels for different column
    // counts and rounds activations to F16 for F16 weights, so the batched
    // CFG pair legitimately differs from two single-column runs at the F16
    // level there. GPU backends use one kernel and agree to ~1e-6.
    const bool   cpu = ggml_backend_is_cpu(vv::backend());
    const double tol = cpu ? 5e-3 : 1e-4;

    int rc = 0;
    for (int steps : {1, 2, 3, 20}) {
        for (int with_cfg = 0; with_cfg <= 1; ++with_cfg) {
            if (with_cfg && !voice.has_neg) continue;
            vv::DPMSolverConfig scfg;
            scfg.num_train_timesteps = 1000; scfg.num_inference_steps = steps;
            scfg.solver_order = 2; scfg.lower_order_final = true;
            vv::DPMSolverState sstate;
            vv::dpm_solver_init(scfg, &sstate);

            std::vector<float> z0(latent);
            for (auto& v : z0) v = nd(rng);
            const float cfg_scale = with_cfg ? 1.7f : 1.0f;

            // unfused
            std::vector<float> x = z0;
            std::vector<float> cond_neg = with_cfg ? voice.neg_tlm_last_hidden : std::vector<float>{};
            if (vv::dpm_solver_sample(x, latent, 1, 1, voice.tlm_last_hidden, hidden,
                                      w.dh, dh_cfg, scfg, sstate, cond_neg, cfg_scale) != 0) {
                std::fprintf(stderr, "FAIL: dpm_solver_sample\n"); return 4;
            }

            // fused (empty caches; the LM step's result is not used)
            vv::ResidentKV kv_tlm, kv_neg;
            if (!kv_tlm.init(cfg.n_layers_tlm, cfg.head_dim, cfg.n_kv_heads, 4, GGML_TYPE_F16)) return 5;
            if (with_cfg && !kv_neg.init(cfg.n_layers_tlm, cfg.head_dim, cfg.n_kv_heads, 4, GGML_TYPE_F16)) return 5;
            vv::FusedFrameInputs in;
            in.z0 = z0.data(); in.cond_pos = voice.tlm_last_hidden.data();
            in.cond_neg = with_cfg ? voice.neg_tlm_last_hidden.data() : nullptr;
            in.cfg_scale = cfg_scale; in.stype = speech_type.data();
            vv::FusedFrameOutputs out;
            if (!vv::run_fused_frame(cfg, w, dh_cfg, scfg, sstate, kv_tlm, with_cfg ? &kv_neg : nullptr,
                                     in, &out, nullptr)) {
                std::fprintf(stderr, "FAIL: run_fused_frame\n"); return 6;
            }
            const double r = rel_rmse(x, out.latent);
            std::printf("steps=%2d cfg=%s  latent rel_rmse=%.2e  (eos_logit=%.3f, hidden[0]=%.4f)\n",
                        steps, with_cfg ? "on " : "off", r, out.eos_logit, out.hidden_pos[0]);
            if (!(r < tol)) { std::fprintf(stderr, "FAIL: exceeds %.1e\n", tol); rc = 10; }
        }
    }
    if (rc == 0) std::printf("OK (backend %s)\n", vv::backend_name());
    return rc;
}
