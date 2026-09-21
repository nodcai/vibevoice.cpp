// Realtime-0.5B generation loop with incremental text and per-chunk audio.
//
// One loop serves both entry points: vibevoice_tts_generate_streaming
// (whole text up front, chunks to a callback) and TtsStream (text pushed
// while an LLM produces it). The loop pulls text five tokens at a time
// through TtsStreamCtl::pull_text, which blocks until a window is available
// or the text has ended, and hands audio to TtsStreamCtl::emit_audio.
//
// Emit scheduling (each rule below was measured on an M4 Max):
//   * a small first chunk (3 frames, ~400 ms) for a low time-to-first-audio,
//     then the chunk grows by three halves up to 32 frames, so a chunk's
//     generation stays inside the previous chunk's playback near real time
//     while the fixed per-emit cost is still amortised
//   * the emit test runs after every frame, not once per six-frame window,
//     otherwise the first chunk can never be smaller than a window
//   * flush on silence: once four latents in a row are silent and real
//     frames are pending, they are emitted, because the model sometimes pads
//     half a second of silence before its EOS fires. A latent is silent when
//     its L2 norm is under an absolute floor (silence sits near 2, speech at
//     5-12 for this model), raised to 1.6x the lead-in's norm when frames 1-3
//     are themselves quiet - some voices start speaking immediately, and a
//     purely lead-relative threshold then classes everything as silent
//   * optional lead hold: keep the chunk small until a chunk has carried
//     speech, for callers that trim the lead-in and can only cut at a chunk
//     boundary
// The decoder is stateful, so each emit decodes only the new frames.

#include "acoustic_decoder_v2.hpp"
#include "backend.hpp"
#include "bench.hpp"
#include "common.hpp"
#include "dfn.hpp"
#include "resample2x.hpp"
#include "tts_frame_graph.hpp"
#include "vibevoice_tts.hpp"
#include "vibevoice_tts_internal.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace vv {

int vibevoice_tts_output_sample_rate(const VibeVoiceModel& model, const VibeVoiceTTSParams& p) {
    return (p.postfilter && model.dfn.ready) ? model.dfn.sample_rate : model.cfg.sample_rate;
}

using detail::add_input_type_embedding;
using detail::run_eos_classifier;
using detail::run_qwen2_stack;
using detail::run_speech_connector;

namespace {

// Embedding rows for `ids` from lm.tok_embd (F32 or F16), [hidden * n].
bool embed_tokens(const VibeVoiceModel& m, const int32_t* ids, int n, std::vector<float>* out) {
    const int hidden = m.cfg.hidden;
    struct ggml_tensor* te = m.w.lm_tok_embd;
    out->assign(static_cast<size_t>(hidden) * n, 0.0f);
    if (te->type == GGML_TYPE_F32) {
        const size_t row = sizeof(float) * hidden;
        for (int t = 0; t < n; ++t) {
            if (ids[t] < 0 || ids[t] >= m.cfg.vocab_size) return false;
            ggml_backend_tensor_get(te, out->data() + static_cast<size_t>(hidden) * t,
                                    row * static_cast<size_t>(ids[t]), row);
        }
        return true;
    }
    if (te->type == GGML_TYPE_F16) {
        const size_t row = sizeof(ggml_fp16_t) * hidden;
        std::vector<ggml_fp16_t> staged(hidden);
        for (int t = 0; t < n; ++t) {
            if (ids[t] < 0 || ids[t] >= m.cfg.vocab_size) return false;
            ggml_backend_tensor_get(te, staged.data(), row * static_cast<size_t>(ids[t]), row);
            for (int i = 0; i < hidden; ++i)
                (*out)[static_cast<size_t>(hidden) * t + i] = ggml_fp16_to_fp32(staged[i]);
        }
        return true;
    }
    // Quantized table (block-32 types keep rows self-contained): dequantize
    // each looked-up row through the type traits.
    const auto* tt = ggml_get_type_traits(te->type);
    if (!tt || !tt->to_float || hidden % ggml_blck_size(te->type) != 0) {
        VV_LOG_ERROR("lm.tok_embd unsupported dtype %s", ggml_type_name(te->type));
        return false;
    }
    const size_t row = ggml_row_size(te->type, hidden);
    std::vector<uint8_t> raw(row);
    for (int t = 0; t < n; ++t) {
        if (ids[t] < 0 || ids[t] >= m.cfg.vocab_size) return false;
        ggml_backend_tensor_get(te, raw.data(), row * static_cast<size_t>(ids[t]), row);
        tt->to_float(raw.data(), out->data() + static_cast<size_t>(hidden) * t, hidden);
    }
    return true;
}

}  // namespace

int vibevoice_tts_run_realtime(VibeVoiceModel*           model,
                               const VibeVoiceTTSParams& p,
                               const TtsStreamCtl&       ctl) {
    if (!model || !ctl.pull_text || !ctl.emit_audio) return -1;
    if (model->variant == "1.5b") {
        VV_LOG_ERROR("realtime loop: only the realtime-0.5b model is supported here");
        return -1;
    }
    if (!model->tokenizer.vocab_size()) {
        VV_LOG_ERROR("realtime loop: tokenizer not loaded");
        return -2;
    }
    const auto& cfg = model->cfg;
    const auto& w   = model->w;
    const int hidden = cfg.hidden;
    const int latent = cfg.latent;
    const int hd     = cfg.head_dim;
    const int n_kv   = cfg.n_kv_heads;

    constexpr int kTextWindow   = 5;
    constexpr int kSpeechWindow = 6;

    const bool use_cfg = p.voice && p.voice->has_neg && p.cfg_scale > 1.0f;
    auto should_abort = [&]() { return ctl.should_abort && ctl.should_abort(); };

    // ---- resident K/V caches (F16 unless VIBEVOICE_KV_F32=1) ----
    const ggml_type kv_type = [] {
        const char* e = std::getenv("VIBEVOICE_KV_F32");
        return (e && e[0] == '1') ? GGML_TYPE_F32 : GGML_TYPE_F16;
    }();
    const int max_text = std::max(16, p.max_text_tokens);
    const int max_lm_seq      = (p.voice ? p.voice->seq_lm     : 0) + max_text + 32;
    const int max_tlm_seq     = (p.voice ? p.voice->seq_tlm    : 0) + max_text + p.max_speech_frames + 32;
    const int max_neg_tlm_seq = (use_cfg ? p.voice->seq_neg_tlm : 0) + p.max_speech_frames + 32;
    ResidentKV kv_lm, kv_tlm, kv_neg_tlm;
    if (!kv_lm.init (cfg.n_layers_lm,  hd, n_kv, max_lm_seq,  kv_type)) return -5;
    if (!kv_tlm.init(cfg.n_layers_tlm, hd, n_kv, max_tlm_seq, kv_type)) return -5;
    if (use_cfg && !kv_neg_tlm.init(cfg.n_layers_tlm, hd, n_kv, max_neg_tlm_seq, kv_type)) return -5;

    auto upload_kv = [hd, n_kv](ResidentKV& dst, const std::vector<LayerKV>& src, int seq_len) {
        std::vector<ggml_fp16_t> tmp;
        for (size_t li = 0; li < src.size(); ++li) {
            const size_t per = static_cast<size_t>(hd) * n_kv * seq_len;
            if (dst.type == GGML_TYPE_F16) {
                tmp.resize(per);
                for (size_t i = 0; i < per; ++i) tmp[i] = ggml_fp32_to_fp16(src[li].k[i]);
                ggml_backend_tensor_set(dst.k[li], tmp.data(), 0, sizeof(ggml_fp16_t) * per);
                for (size_t i = 0; i < per; ++i) tmp[i] = ggml_fp32_to_fp16(src[li].v[i]);
                ggml_backend_tensor_set(dst.v[li], tmp.data(), 0, sizeof(ggml_fp16_t) * per);
            } else {
                ggml_backend_tensor_set(dst.k[li], src[li].k.data(), 0, sizeof(float) * per);
                ggml_backend_tensor_set(dst.v[li], src[li].v.data(), 0, sizeof(float) * per);
            }
        }
        dst.past_len = seq_len;
    };
    if (p.voice) {
        upload_kv(kv_lm,  p.voice->kv_lm,  p.voice->seq_lm);
        upload_kv(kv_tlm, p.voice->kv_tlm, p.voice->seq_tlm);
        if (use_cfg) upload_kv(kv_neg_tlm, p.voice->kv_neg_tlm, p.voice->seq_neg_tlm);
    }

    int lm_pos  = p.voice ? p.voice->seq_lm  : 0;
    int tlm_pos = p.voice ? p.voice->seq_tlm : 0;
    std::vector<float> tlm_hidden_last = p.voice ? p.voice->tlm_last_hidden
                                                 : std::vector<float>(static_cast<size_t>(hidden), 0.0f);
    int neg_tlm_pos = use_cfg ? p.voice->seq_neg_tlm : 0;
    std::vector<float> neg_tlm_hidden_last = use_cfg ? p.voice->neg_tlm_last_hidden : std::vector<float>{};
    // CFG negative-condition anchor: the negative path's hidden state drifts
    // as speech frames accumulate, audible as the voice losing energy over a
    // long reply. Blending the post-prefill snapshot back in holds it steady.
    const std::vector<float> neg_anchor = neg_tlm_hidden_last;
    const float anchor_w = std::min(1.0f, std::max(0.0f, p.neg_condition_anchor));
    std::vector<float> neg_cond(neg_tlm_hidden_last.size());

    // ---- diffusion ----
    DPMSolverConfig solver_cfg;
    solver_cfg.num_train_timesteps = 1000;
    solver_cfg.num_inference_steps = std::max(1, p.n_diffusion_steps);
    solver_cfg.solver_order        = 2;
    solver_cfg.lower_order_final   = true;
    DPMSolverState solver_state;
    dpm_solver_init(solver_cfg, &solver_state);
    DiffusionHeadConfig dh_cfg;
    dh_cfg.hidden = hidden; dh_cfg.latent = latent; dh_cfg.head_layers = cfg.head_layers;
    dh_cfg.ffn_ratio = cfg.ffn_ratio; dh_cfg.eps = cfg.rms_norm_eps; dh_cfg.freq_size = 256;
    std::mt19937 rng(p.seed ? p.seed : std::random_device{}());
    std::normal_distribution<float> norm(0.0f, 1.0f);

    // ---- decoder ----
    StreamingCache dec_cache;
    DecoderV2State dec_v2;
    const bool use_v2 = decoder_v2_enabled() &&
                        decoder_v2_prepare(w.at_dec, cfg.acoustic, &w.at_dec_v2) &&
                        decoder_v2_state_init(w.at_dec_v2, &dec_v2);
    const bool fused = fused_frame_enabled();
    std::vector<float> speech_type(static_cast<size_t>(hidden), 0.0f);
    add_input_type_embedding(cfg, w, 1, /*type=*/0, speech_type.data());
    BenchTotals bench;
    if (p.verbose)
        std::fprintf(stderr, "[tts] cfg=%s (%.2f, anchor %.2f) frames=%s kv=%s decoder=%s steps=%d\n",
                     use_cfg ? "on" : "off", static_cast<double>(p.cfg_scale), static_cast<double>(anchor_w),
                     fused ? "fused" : "unfused", ggml_type_name(kv_type),
                     use_v2 ? "v2" : "legacy", solver_cfg.num_inference_steps);

    // ---- emit scheduling ----
    std::vector<float> all_latents;
    all_latents.reserve(static_cast<size_t>(p.max_speech_frames) * latent);
    int  frames_emitted = 0;
    int  emits          = 0;
    bool first_emit     = true;
    const int   kMaxChunk = std::max(1, p.stream_max_chunk_frames);
    const int   first_chunk = std::min(kMaxChunk, std::max(1, p.stream_first_chunk_frames));
    const int   lead_chunk  = p.stream_lead_chunk_frames > 0 ? std::min(p.stream_lead_chunk_frames, first_chunk) : 0;
    int  emit_chunk    = lead_chunk > 0 ? lead_chunk : first_chunk;
    bool sound_shipped = false;
    constexpr int   kTailSilentFrames = 4;
    constexpr float kSilentNormRatio  = 1.6f;
    constexpr float kSilentNormFloor  = 3.5f;
    float silent_norm_ref     = 0.0f;
    int   silent_run          = 0;
    int   pending_sound       = 0;   // non-silent frames generated since the last emit
    int   pending_speech      = 0;   // same, but only once the silence reference exists
    constexpr int kWarmupSamples = 2400;   // decoder zero-state transient, 100 ms @ 24 kHz
    bool aborted = false;

    // ---- post-filter: 24 kHz -> 2x -> DFN3 at 48 kHz ----
    const bool use_pf = p.postfilter && model->dfn.ready;
    std::unique_ptr<DfnStream> pf;
    Upsampler2x up2x;
    std::vector<float> pf_in, pf_out;
    if (use_pf) {
        BenchScope bs(&bench, "dfn_warmup");
        pf = std::make_unique<DfnStream>(model->dfn);
        pf->warmup(100);
    }
    // ---- lead trim: runs on what the listener would hear (post-filtered) ----
    // The model puts digital silence in front of every reply (~150-800 ms),
    // or an artefact the post-filter turns into silence. Drop it until the
    // onset, keep 20 ms ahead of the onset so it is not a step, fade in 20 ms.
    const int   out_rate      = use_pf ? model->dfn.sample_rate : cfg.sample_rate;
    const float kSilence      = 1e-3f;                 // ~-60 dBFS; the lead is < -90, speech ~-16
    const int   kKeepAhead    = 20 * out_rate / 1000;
    const int   kFadeIn       = 20 * out_rate / 1000;
    const int   kLeadMax      = 1500 * out_rate / 1000;  // never swallow more than this
    int   lead_dropped = 0;
    bool  lead_done   = !p.trim_lead;
    int   fade_left   = 0;                             // samples of fade-in still to apply
    std::vector<float> lead_keep;                      // the retained lead (at most kLeadMax)
    std::vector<float> trim_buf;
    auto ship = [&](const float* pcm, int n) -> bool {
        if (n <= 0) return true;
        if (lead_done && fade_left <= 0) return ctl.emit_audio(pcm, n);
        trim_buf.clear();
        if (!lead_done) {
            int first = 0;
            while (first < n && std::fabs(pcm[first]) < kSilence) ++first;
            if (first >= n && lead_dropped + n <= kLeadMax) {
                // still the lead: retain it (for the keep-ahead, and so a reply
                // that never rises above the floor is delivered as silence at
                // the end rather than as nothing)
                lead_dropped += n;
                lead_keep.insert(lead_keep.end(), pcm, pcm + n);
                return true;
            }
            if (first >= n) first = 0;   // cap reached: deliver from here on, whatever it is
            lead_done = true;
            fade_left = kFadeIn;
            // keep-ahead: up to kKeepAhead samples before the onset, from
            // this chunk and, if needed, the remembered lead tail
            const int from_chunk = std::min(first, kKeepAhead);
            const int from_lead  = std::min(static_cast<int>(lead_keep.size()), kKeepAhead - from_chunk);
            trim_buf.insert(trim_buf.end(), lead_keep.end() - from_lead, lead_keep.end());
            trim_buf.insert(trim_buf.end(), pcm + first - from_chunk, pcm + n);
            lead_keep.clear();
        } else {
            trim_buf.assign(pcm, pcm + n);
        }
        // 20 ms fade-in from the first shipped sample
        const int done = kFadeIn - fade_left;
        for (int i = 0; i < static_cast<int>(trim_buf.size()) && fade_left > 0; ++i, --fade_left) {
            const float t = static_cast<float>(done + i) / kFadeIn;
            trim_buf[i] *= t * t;
        }
        return ctl.emit_audio(trim_buf.data(), static_cast<int>(trim_buf.size()));
    };
    // Everything the loop delivers goes through here.
    auto deliver = [&](const float* pcm24, int n, bool final_chunk) -> bool {
        if (!use_pf) return n > 0 ? ship(pcm24, n) : true;
        BenchScope bs(&bench, "postfilter");
        pf_in.clear();
        pf_out.clear();
        if (n > 0) up2x.process(pcm24, n, &pf_in);
        pf->process(pf_in.data(), static_cast<int>(pf_in.size()), &pf_out, final_chunk);
        if (pf_out.empty()) return true;
        return ship(pf_out.data(), static_cast<int>(pf_out.size()));
    };

    auto advance_chunk = [&]() {
        if (lead_chunk > 0 && !sound_shipped) {
            if (pending_speech == 0) return;      // still the lead-in: hold the size
            sound_shipped = true;                  // playback starts here: configured size, then grow
            emit_chunk = first_chunk;
            return;
        }
        emit_chunk = std::min(kMaxChunk, std::max(emit_chunk + 1, (emit_chunk * 3) / 2));
    };
    auto pending_frames = [&]() { return static_cast<int>(all_latents.size()) / latent - frames_emitted; };

    // Decode and deliver every frame not yet emitted. Returns false to stop.
    auto emit_pending = [&](bool final_chunk) -> bool {
        const int total = static_cast<int>(all_latents.size()) / latent;
        const int nf    = total - frames_emitted;
        if (nf <= 0) return true;
        BenchScope bs(&bench, "decode+emit");
        std::vector<float> scaled(static_cast<size_t>(nf) * latent);
        const float* src = all_latents.data() + static_cast<size_t>(frames_emitted) * latent;
        for (size_t i = 0; i < scaled.size(); ++i)
            scaled[i] = src[i] / cfg.speech_scaling - cfg.speech_bias;
        std::vector<float> audio;
        bool ok;
        if (use_v2) {
            ok = decoder_v2_decode(w.at_dec_v2, dec_v2, scaled.data(), nf, &audio);
        } else {
            ok = run_decoder_chunk_streaming(cfg, w, scaled.data(), nf, dec_cache, first_emit, final_chunk, &audio);
        }
        if (!ok) { VV_LOG_ERROR("decoder failed at frame %d", total); return false; }
        size_t off = 0;
        if (first_emit && p.trim_decoder_warmup)
            off = std::min(audio.size(), static_cast<size_t>(kWarmupSamples));
        first_emit = false;
        frames_emitted = total;
        ++emits;
        return deliver(audio.data() + off, static_cast<int>(audio.size() - off), false);
    };

    // ---- text ----
    std::vector<int32_t> text_ids;
    int  text_pos = 0;
    bool text_eof = false;
    // Block until `want` tokens are available or the text has ended.
    auto ensure_text = [&](int want) {
        while (!text_eof && static_cast<int>(text_ids.size()) < want) {
            std::vector<int32_t> more;
            bool eof = false;
            ctl.pull_text(want - static_cast<int>(text_ids.size()), &more, &eof);
            text_ids.insert(text_ids.end(), more.begin(), more.end());
            if (eof) text_eof = true;
            if (more.empty() && !eof) break;   // a non-blocking source with nothing yet
        }
    };

    int  total_frames = 0;
    bool finished     = false;
    std::vector<float> z(static_cast<size_t>(latent));

    while (!finished && total_frames < p.max_speech_frames) {
        if (should_abort()) { aborted = true; break; }
        ensure_text(text_pos + kTextWindow);
        if (text_ids.empty() && text_eof) break;    // nothing to say

        // ---- text window (up to 5 tokens) ----
        const int n_text = static_cast<int>(text_ids.size());
        if (text_pos < n_text) {
            BenchScope bs(&bench, "lm_text_window");
            const int win = std::min(kTextWindow, n_text - text_pos);
            if (lm_pos + win > kv_lm.max_seq || tlm_pos + win > kv_tlm.max_seq) {
                VV_LOG_ERROR("realtime loop: text exceeds max_text_tokens=%d", p.max_text_tokens);
                return -6;
            }
            std::vector<float> emb_win;
            if (!embed_tokens(*model, text_ids.data() + text_pos, win, &emb_win)) return -5;
            std::vector<float> lm_hidden;
            if (!run_qwen2_stack(nullptr, cfg, w.lm_layers, nullptr, lm_pos, win, emb_win.data(),
                                 &kv_lm, &lm_hidden, nullptr)) return -6;
            lm_pos += win;
            std::vector<float> tlm_in = lm_hidden;
            add_input_type_embedding(cfg, w, win, /*type=*/1, tlm_in.data());
            if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers, w.tlm_output_norm, tlm_pos, win, tlm_in.data(),
                                 &kv_tlm, nullptr, &tlm_hidden_last)) return -8;
            tlm_pos += win;
            text_pos += win;
            if (p.verbose) std::fprintf(stderr, "[tts] text window: %d/%d tokens\n", text_pos, n_text);
        }

        // ---- speech window (up to 6 frames or until EOS) ----
        const int sp_budget = std::min(kSpeechWindow, p.max_speech_frames - total_frames);
        for (int sp = 0; sp < sp_budget && !finished; ++sp) {
            if (should_abort()) { aborted = true; break; }
            const int frame = total_frames;
            for (auto& v : z) v = norm(rng);
            if (use_cfg) {
                for (size_t i = 0; i < neg_cond.size(); ++i)
                    neg_cond[i] = neg_tlm_hidden_last[i] * (1.0f - anchor_w) + neg_anchor[i] * anchor_w;
            }
            if (tlm_pos + 1 > kv_tlm.max_seq || (use_cfg && neg_tlm_pos + 1 > kv_neg_tlm.max_seq)) {
                VV_LOG_ERROR("realtime loop: KV cache full at frame %d", frame);
                return -6;
            }

            float eos = 0.0f;
            if (fused) {
                FusedFrameInputs fin;
                fin.z0 = z.data(); fin.cond_pos = tlm_hidden_last.data();
                fin.cond_neg = use_cfg ? neg_cond.data() : nullptr;
                fin.cfg_scale = p.cfg_scale; fin.stype = speech_type.data();
                FusedFrameOutputs fout;
                bool ok;
                {
                    BenchScope bs(&bench, "fused_frame");
                    ok = run_fused_frame(cfg, w, dh_cfg, solver_cfg, solver_state, kv_tlm,
                                         use_cfg ? &kv_neg_tlm : nullptr, fin, &fout, &bench);
                }
                if (!ok) { VV_LOG_ERROR("fused frame failed at frame %d", frame); return -9; }
                z = fout.latent;
                tlm_hidden_last = fout.hidden_pos;
                tlm_pos += 1;
                if (use_cfg) { neg_tlm_hidden_last = fout.hidden_neg; neg_tlm_pos += 1; }
                eos = 1.0f / (1.0f + std::exp(-fout.eos_logit));
            } else {
                int rc;
                {
                    BenchScope bs(&bench, "diffusion");
                    rc = dpm_solver_sample(z, latent, 1, 1, tlm_hidden_last, hidden, w.dh, dh_cfg,
                                           solver_cfg, solver_state, use_cfg ? neg_cond : std::vector<float>{},
                                           p.cfg_scale);
                }
                if (rc != 0) { VV_LOG_ERROR("dpm_solver_sample failed at frame %d", frame); return -9; }
                auto ac_embed = run_speech_connector(cfg, w, z.data(), 1);
                add_input_type_embedding(cfg, w, 1, /*type=*/0, ac_embed.data());
                {
                    BenchScope bs(&bench, "tlm_pos");
                    if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers, w.tlm_output_norm, tlm_pos, 1,
                                         ac_embed.data(), &kv_tlm, nullptr, &tlm_hidden_last)) return -10;
                    tlm_pos += 1;
                }
                if (use_cfg) {
                    BenchScope bs(&bench, "tlm_neg");
                    std::vector<float> ac_embed_neg(ac_embed);
                    if (!run_qwen2_stack(nullptr, cfg, w.tlm_layers, w.tlm_output_norm, neg_tlm_pos, 1,
                                         ac_embed_neg.data(), &kv_neg_tlm, nullptr, &neg_tlm_hidden_last)) return -10;
                    neg_tlm_pos += 1;
                }
                eos = run_eos_classifier(cfg, w, tlm_hidden_last.data());
            }
            all_latents.insert(all_latents.end(), z.begin(), z.end());
            ++total_frames;

            // Silence tracking on the latent just generated.
            {
                float ss = 0.0f;
                for (float v : z) ss += v * v;
                const float nrm = std::sqrt(ss);
                if (frame >= 1 && frame <= 3)   // the lead-in; frame 0 sometimes carries a click
                    silent_norm_ref = (silent_norm_ref == 0.0f) ? nrm : std::min(silent_norm_ref, nrm);
                float thresh = kSilentNormFloor;
                if (silent_norm_ref > 0.0f && silent_norm_ref < kSilentNormFloor)
                    thresh = std::max(thresh, kSilentNormRatio * silent_norm_ref);
                const bool silent = nrm < thresh;
                if (silent) {
                    ++silent_run;
                } else {
                    silent_run = 0;
                    ++pending_sound;
                    if (frame >= 1) ++pending_speech;   // frame 0 is never taken as speech
                }
            }
            if (p.verbose && (frame % 4 == 0 || eos > 0.5f))
                std::fprintf(stderr, "[tts] frame %d: eos=%.3f\n", frame, eos);
            if (eos > 0.5f) {
                if (p.verbose) std::fprintf(stderr, "[tts] EOS at frame %d\n", frame);
                finished = true;
                break;
            }

            // Emit test after every frame.
            if (pending_frames() >= emit_chunk) {
                if (!emit_pending(false)) { aborted = true; break; }
                advance_chunk();   // reads pending_speech, so before the reset
                pending_sound = pending_speech = 0;
            } else if (p.flush_on_silence && silent_run >= kTailSilentFrames && pending_sound > 0) {
                if (!emit_pending(false)) { aborted = true; break; }
                pending_sound = pending_speech = 0;
            }
        }
        if (aborted) break;
    }

    if (!aborted) {
        if (!emit_pending(true)) aborted = true;
        else if (use_pf && !deliver(nullptr, 0, true)) aborted = true;   // lookahead + overlap tail
        if (!aborted && !lead_done && !lead_keep.empty()) {
            // nothing ever rose above the floor: hand over what there is
            if (!ctl.emit_audio(lead_keep.data(), static_cast<int>(lead_keep.size()))) aborted = true;
            lead_keep.clear();
        }
    }
    if (bench_enabled()) {
        char title[128];
        std::snprintf(title, sizeof(title), "tts realtime: %d frames, %d emits, %d steps, backend %s%s",
                      total_frames, emits, solver_cfg.num_inference_steps, vv::backend_name(),
                      aborted ? " (aborted)" : "");
        bench.report(title);
    }
    if (const char* dump = std::getenv("VIBEVOICE_DUMP_LATENTS")) {
        if (FILE* f = std::fopen(dump, "wb")) {
            std::fwrite(all_latents.data(), sizeof(float), all_latents.size(), f);
            std::fclose(f);
        }
    }
    if (p.verbose)
        std::fprintf(stderr, "[tts] done: %d frames, %d emits%s\n", total_frames, emits, aborted ? " (aborted)" : "");
    return 0;
}

int vibevoice_tts_generate_streaming(VibeVoiceModel*           model,
                                     const std::string&        text,
                                     const VibeVoiceTTSParams& p,
                                     const vv_pcm_chunk_cb&    on_chunk) {
    if (!model || !on_chunk) return -1;
    if (model->variant == "1.5b") {
        VV_LOG_ERROR("vibevoice_tts_generate_streaming: streaming is only supported for realtime-0.5b models");
        return -1;
    }
    if (!model->tokenizer.vocab_size()) {
        VV_LOG_ERROR("vibevoice_tts_generate: tokenizer not loaded");
        return -2;
    }
    // A trailing "\n" terminates the user turn (mlx-audio convention); the
    // voice prefix ends mid-conversation and the model needs the separator to
    // start the speech reply.
    std::string t = text;
    while (!t.empty() && (t.back() == ' ' || t.back() == '\t' || t.back() == '\r' || t.back() == '\n')) t.pop_back();
    t += "\n";
    const std::vector<int32_t> ids = model->tokenizer.encode(t);
    if (ids.empty()) {
        VV_LOG_ERROR("tokenizer produced no tokens");
        return -3;
    }
    if (p.verbose) std::fprintf(stderr, "[tts] %zu input text tokens\n", ids.size());

    VibeVoiceTTSParams pp = p;
    pp.max_text_tokens = std::max(pp.max_text_tokens, static_cast<int>(ids.size()) + 8);
    size_t pos = 0;
    TtsStreamCtl ctl;
    ctl.pull_text = [&](int max_tokens, std::vector<int32_t>* out, bool* eof) {
        const size_t n = std::min(static_cast<size_t>(std::max(0, max_tokens)), ids.size() - pos);
        out->insert(out->end(), ids.begin() + static_cast<long>(pos), ids.begin() + static_cast<long>(pos + n));
        pos += n;
        *eof = pos >= ids.size();
    };
    ctl.emit_audio = [&](const float* pcm, int n) { return on_chunk(pcm, n); };
    return vibevoice_tts_run_realtime(model, pp, ctl);
}

}  // namespace vv
