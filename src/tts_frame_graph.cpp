#include "tts_frame_graph.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "rms_norm.hpp"
#include "rope.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace vv {

bool fused_frame_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("VIBEVOICE_FUSED");
        return !(e && e[0] == '0');
    }();
    return on;
}

namespace {

// Column `c` of a [d, n_cols] 2-D tensor as a 1-D view.
inline struct ggml_tensor* col1d(struct ggml_context* g, struct ggml_tensor* t, int c) {
    return ggml_view_1d(g, t, t->ne[0], static_cast<size_t>(c) * t->nb[1]);
}

}  // namespace

bool run_fused_frame(const VibeVoiceConfig&      cfg,
                     const VibeVoiceWeights&     w,
                     const DiffusionHeadConfig&  dh_cfg,
                     const DPMSolverConfig&      solver_cfg,
                     const DPMSolverState&       solver_state,
                     ResidentKV&                 kv_tlm,
                     ResidentKV*                 kv_neg,
                     const FusedFrameInputs&     in,
                     FusedFrameOutputs*          out,
                     BenchTotals*                bench) {
    if (!in.z0 || !in.cond_pos || !in.stype || !out) return false;
    const int M      = solver_cfg.num_inference_steps;
    const int latent = cfg.latent;
    const int hidden = cfg.hidden;
    const int freq   = dh_cfg.freq_size;
    if (static_cast<int>(solver_state.timesteps.size()) != M + 1 || M < 1) return false;

    const bool use_cfg = in.cond_neg != nullptr && kv_neg != nullptr && in.cfg_scale > 1.0f;
    const int  nc      = use_cfg ? 2 : 1;   // columns: positive (+ negative)

    Qwen2Hparams hp;
    hp.hidden_size    = hidden;
    hp.n_heads        = cfg.n_heads;
    hp.n_kv_heads     = cfg.n_kv_heads;
    hp.head_dim       = cfg.head_dim;
    hp.rope_theta     = cfg.rope_theta;
    hp.rms_norm_eps   = cfg.rms_norm_eps;
    hp.use_flash_attn = vv::backend_supports_flash_attn();
    const int hd = hp.head_dim, n_h = hp.n_heads, n_kv = hp.n_kv_heads;

    if (kv_tlm.past_len + 1 > kv_tlm.max_seq) { VV_LOG_ERROR("fused_frame: tlm KV cache full"); return false; }
    if (use_cfg && kv_neg->past_len + 1 > kv_neg->max_seq) { VV_LOG_ERROR("fused_frame: neg KV cache full"); return false; }

    BenchClock clk_all;

    // ---- inputs (own small allocated context) ----
    struct ggml_init_params ipi {};
    ipi.mem_size = ggml_tensor_overhead() * 8;
    ipi.no_alloc = true;
    struct ggml_context* ctx_in = ggml_init(ipi);
    struct ggml_tensor* z0    = ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, latent);
    struct ggml_tensor* cond  = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, hidden, nc);
    struct ggml_tensor* tsin  = ggml_new_tensor_2d(ctx_in, GGML_TYPE_F32, freq, M);
    struct ggml_tensor* stype = ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, hidden);
    struct ggml_tensor* pos   = ggml_new_tensor_1d(ctx_in, GGML_TYPE_I32, nc);
    ggml_backend_buffer_t in_buf = vv::allocate_ctx_tensors(ctx_in);
    if (!in_buf) { ggml_free(ctx_in); return false; }

    ggml_backend_tensor_set(z0, in.z0, 0, sizeof(float) * latent);
    ggml_backend_tensor_set(cond, in.cond_pos, 0, sizeof(float) * hidden);
    if (use_cfg) ggml_backend_tensor_set(cond, in.cond_neg, sizeof(float) * hidden, sizeof(float) * hidden);
    {
        std::vector<float> ts_v(static_cast<size_t>(freq) * M);
        for (int i = 0; i < M; ++i)
            timestep_sinusoidal(static_cast<float>(solver_state.timesteps[i]), freq,
                                ts_v.data() + static_cast<size_t>(i) * freq);
        ggml_backend_tensor_set(tsin, ts_v.data(), 0, sizeof(float) * ts_v.size());
    }
    ggml_backend_tensor_set(stype, in.stype, 0, sizeof(float) * hidden);
    {
        int32_t pv[2] = {kv_tlm.past_len, use_cfg ? kv_neg->past_len : 0};
        ggml_backend_tensor_set(pos, pv, 0, sizeof(int32_t) * nc);
    }

    // ---- graph ----
    struct ggml_init_params ipg {};
    ipg.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
    ipg.no_alloc = true;
    struct ggml_context* g = ggml_init(ipg);

    // 1. DPM-Solver++ with the CFG pair batched as two frames of the head.
    struct ggml_tensor* z       = z0;
    struct ggml_tensor* prev_x0 = nullptr;
    struct ggml_tensor* cond3   = ggml_reshape_3d(g, cond, hidden, nc, 1);
    for (int i = 0; i < M; ++i) {
        const int t = solver_state.timesteps[i];
        const int s = solver_state.timesteps[i + 1];
        const float a_t    = solver_state.alpha_t[t];
        const float sg_t   = solver_state.sigma_t[t];
        const float l_t    = solver_state.lambda_t[t];
        const float l_prev = (i > 0) ? solver_state.lambda_t[solver_state.timesteps[i - 1]] : 0.0f;
        float a_s, sg_s, l_s;
        if (s == -1) { a_s = 1.0f; sg_s = 0.0f; l_s = std::numeric_limits<float>::infinity(); }
        else         { a_s = solver_state.alpha_t[s]; sg_s = solver_state.sigma_t[s]; l_s = solver_state.lambda_t[s]; }

        struct ggml_tensor* noisy = (nc == 2) ? ggml_concat(g, z, z, 1) : z;
        noisy = ggml_reshape_3d(g, noisy, latent, nc, 1);
        struct ggml_tensor* tsv = ggml_view_2d(g, tsin, freq, 1, tsin->nb[1], static_cast<size_t>(i) * tsin->nb[1]);
        struct ggml_tensor* v   = diffusion_head_forward(g, noisy, cond3, tsv, w.dh, dh_cfg);  // [latent, nc, 1]
        v = ggml_reshape_2d(g, v, latent, nc);
        struct ggml_tensor* v_cfg;
        if (nc == 2) {
            // v_cfg = v_neg + cfg*(v_pos - v_neg) = (1-cfg)*v_neg + cfg*v_pos
            v_cfg = ggml_add(g, ggml_scale(g, col1d(g, v, 1), 1.0f - in.cfg_scale),
                                ggml_scale(g, col1d(g, v, 0), in.cfg_scale));
        } else {
            v_cfg = ggml_reshape_1d(g, v, latent);
        }
        // x0 = a_t*z - sigma_t*v
        struct ggml_tensor* x0 = ggml_sub(g, ggml_scale(g, z, a_t), ggml_scale(g, v_cfg, sg_t));

        const bool is_first = (i == 0);
        const bool is_last  = (i == M - 1);
        const int  order    = (is_first || (solver_cfg.lower_order_final && is_last)) ? 1 : solver_cfg.solver_order;
        if (s == -1) {
            z = x0;
        } else if (order == 1) {
            const float h = l_s - l_t;
            const float A = sg_s / sg_t;
            const float B = a_s * (std::exp(-h) - 1.0f);
            z = ggml_sub(g, ggml_scale(g, z, A), ggml_scale(g, x0, B));
        } else {
            const float h   = l_s - l_t;
            const float h_0 = l_t - l_prev;
            const float r0  = h_0 / h;
            const float A   = sg_s / sg_t;
            const float B   = a_s * (std::exp(-h) - 1.0f);
            const float Bd1 = 0.5f * B;
            struct ggml_tensor* D1 = ggml_scale(g, ggml_sub(g, x0, prev_x0), 1.0f / r0);
            z = ggml_sub(g, ggml_sub(g, ggml_scale(g, z, A), ggml_scale(g, x0, B)), ggml_scale(g, D1, Bd1));
        }
        prev_x0 = x0;
    }
    struct ggml_tensor* latent_out = z;
    ggml_set_name(latent_out, "ff_latent");
    ggml_set_output(latent_out);

    // 2. Acoustic connector + speech type embedding.
    struct ggml_tensor* e = ggml_mul_mat(g, w.ac_fc1_w, latent_out);
    if (w.ac_fc1_b) e = ggml_add(g, e, w.ac_fc1_b);
    e = ggml_rms_norm(g, e, 1e-6f);
    e = ggml_mul(g, e, w.ac_norm);
    e = ggml_mul_mat(g, w.ac_fc2_w, e);
    if (w.ac_fc2_b) e = ggml_add(g, e, w.ac_fc2_b);
    e = ggml_add(g, e, stype);

    // 3. TTS-LM step, positive and negative sequences as two columns.
    struct ggml_tensor* cur = (nc == 2) ? ggml_concat(g, e, e, 1) : e;   // [hidden, nc]
    const int n_layers = static_cast<int>(w.tlm_layers.size());
    std::vector<struct ggml_tensor*> writes;
    writes.reserve(static_cast<size_t>(n_layers) * 2 * nc);
    std::vector<size_t> writes_per_layer(n_layers + 1, 0);
    const float eps = hp.rms_norm_eps;
    for (int il = 0; il < n_layers; ++il) {
        const Qwen2LayerWeights& lw = w.tlm_layers[il];
        struct ggml_tensor* xn = rms_norm(g, cur, lw.attn_norm, eps);
        struct ggml_tensor* q  = ggml_mul_mat(g, lw.attn_q, xn);
        if (lw.attn_q_bias) q = ggml_add(g, q, lw.attn_q_bias);
        struct ggml_tensor* k  = ggml_mul_mat(g, lw.attn_k, xn);
        if (lw.attn_k_bias) k = ggml_add(g, k, lw.attn_k_bias);
        struct ggml_tensor* v  = ggml_mul_mat(g, lw.attn_v, xn);
        if (lw.attn_v_bias) v = ggml_add(g, v, lw.attn_v_bias);
        q = ggml_reshape_4d(g, q, hd, n_h,  nc, 1);
        k = ggml_reshape_4d(g, k, hd, n_kv, nc, 1);
        v = ggml_reshape_4d(g, v, hd, n_kv, nc, 1);
        q = ggml_rope_ext(g, q, pos, nullptr, hd, kRopeMode, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(g, k, pos, nullptr, hd, kRopeMode, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor* o_cols[2] = {nullptr, nullptr};
        for (int c = 0; c < nc; ++c) {
            ResidentKV& kvs = (c == 0) ? kv_tlm : *kv_neg;
            struct ggml_tensor* kr = kvs.k[il];
            struct ggml_tensor* vr = kvs.v[il];
            const int np = kvs.past_len;
            // this column's new K/V → resident slot np
            struct ggml_tensor* kc = ggml_view_4d(g, k, hd, n_kv, 1, 1, k->nb[1], k->nb[2], k->nb[3],
                                                  static_cast<size_t>(c) * k->nb[2]);
            struct ggml_tensor* vc = ggml_view_4d(g, v, hd, n_kv, 1, 1, v->nb[1], v->nb[2], v->nb[3],
                                                  static_cast<size_t>(c) * v->nb[2]);
            struct ggml_tensor* k_dst = ggml_view_4d(g, kr, hd, n_kv, 1, 1, kr->nb[1], kr->nb[2], kr->nb[3],
                                                     static_cast<size_t>(np) * kr->nb[2]);
            struct ggml_tensor* v_dst = ggml_view_4d(g, vr, hd, n_kv, 1, 1, vr->nb[1], vr->nb[2], vr->nb[3],
                                                     static_cast<size_t>(np) * vr->nb[2]);
            writes.push_back(ggml_cpy(g, kc, k_dst));
            writes.push_back(ggml_cpy(g, vc, v_dst));
            // full history [hd, n_kv, np+1] as strided views, permuted for attention
            struct ggml_tensor* k_full = ggml_view_4d(g, kr, hd, n_kv, np + 1, 1, kr->nb[1], kr->nb[2], kr->nb[3], 0);
            struct ggml_tensor* v_full = ggml_view_4d(g, vr, hd, n_kv, np + 1, 1, vr->nb[1], vr->nb[2], vr->nb[3], 0);
            struct ggml_tensor* qc = ggml_view_4d(g, q, hd, n_h, 1, 1, q->nb[1], q->nb[2], q->nb[3],
                                                  static_cast<size_t>(c) * q->nb[2]);
            struct ggml_tensor* q_p = ggml_permute(g, qc, 0, 2, 1, 3);
            struct ggml_tensor* k_p = ggml_permute(g, k_full, 0, 2, 1, 3);
            struct ggml_tensor* v_p = ggml_permute(g, v_full, 0, 2, 1, 3);
            o_cols[c] = qwen2_attention(g, q_p, k_p, v_p, /*mask=*/nullptr, hp);   // [n_h*hd, 1]
        }
        writes_per_layer[il + 1] = writes.size();
        struct ggml_tensor* o = (nc == 2) ? ggml_concat(g, o_cols[0], o_cols[1], 1) : o_cols[0];
        o = ggml_mul_mat(g, lw.attn_o, o);
        struct ggml_tensor* h  = ggml_add(g, cur, o);
        struct ggml_tensor* hn = rms_norm(g, h, lw.ffn_norm, eps);
        struct ggml_tensor* gt = ggml_mul_mat(g, lw.ffn_gate, hn);
        struct ggml_tensor* up = ggml_mul_mat(g, lw.ffn_up,   hn);
        struct ggml_tensor* f  = ggml_mul_mat(g, lw.ffn_down, ggml_swiglu_split(g, gt, up));
        cur = ggml_add(g, h, f);
    }
    if (w.tlm_output_norm) cur = rms_norm(g, cur, w.tlm_output_norm, eps);
    struct ggml_tensor* hidden_out = cur;
    ggml_set_name(hidden_out, "ff_hidden");
    ggml_set_output(hidden_out);

    // 4. EOS classifier on the positive column.
    struct ggml_tensor* eos = nullptr;
    if (w.eos_fc1_w && w.eos_fc2_w) {
        struct ggml_tensor* h0 = col1d(g, hidden_out, 0);
        struct ggml_tensor* x  = ggml_mul_mat(g, w.eos_fc1_w, h0);
        if (w.eos_fc1_b) x = ggml_add(g, x, w.eos_fc1_b);
        x = ggml_relu(g, x);
        x = ggml_mul_mat(g, w.eos_fc2_w, x);
        if (w.eos_fc2_b) x = ggml_add(g, x, w.eos_fc2_b);
        eos = x;
        ggml_set_name(eos, "ff_eos");
        ggml_set_output(eos);
    }

    struct ggml_cgraph* gf = ggml_new_graph_custom(g, 16384, false);
    // The K/V writes of layer il must be in the graph before layer il's
    // attention reads the resident tensor, and the attention of layer il is
    // only pulled in by a later expand - so expand the writes layer by layer
    // first (same rule as run_qwen2_stack / test_qwen2_resident).
    for (auto* wr : writes) ggml_build_forward_expand(gf, wr);
    ggml_build_forward_expand(gf, latent_out);
    ggml_build_forward_expand(gf, hidden_out);
    if (eos) ggml_build_forward_expand(gf, eos);

    // Debug: VIBEVOICE_FF_NOREUSE=1 gives every intermediate its own buffer
    // (no allocator reuse / in-place), to separate allocator effects from
    // kernel numerics when a backend disagrees with the unfused path.
    static const bool no_reuse = [] { const char* e = std::getenv("VIBEVOICE_FF_NOREUSE"); return e && e[0] == '1'; }();
    ggml_backend_buffer_t dbg_buf = no_reuse ? vv::allocate_ctx_tensors(g) : nullptr;
    const double build_ms = clk_all.ms();
    BenchClock clk_compute;
    const bool ok = vv::compute_graph(gf);
    if (ok) {
        out->latent.resize(latent);
        ggml_backend_tensor_get(latent_out, out->latent.data(), 0, sizeof(float) * latent);
        out->hidden_pos.resize(hidden);
        ggml_backend_tensor_get(hidden_out, out->hidden_pos.data(), 0, sizeof(float) * hidden);
        if (use_cfg) {
            out->hidden_neg.resize(hidden);
            ggml_backend_tensor_get(hidden_out, out->hidden_neg.data(), sizeof(float) * hidden, sizeof(float) * hidden);
        } else {
            out->hidden_neg.clear();
        }
        out->eos_logit = 0.0f;
        if (eos) ggml_backend_tensor_get(eos, &out->eos_logit, 0, sizeof(float));
        kv_tlm.past_len += 1;
        if (use_cfg) kv_neg->past_len += 1;
    } else {
        VV_LOG_ERROR("fused_frame: compute failed");
    }
    if (bench && bench_enabled()) {
        bench->add("ff_build+alloc", build_ms);
        bench->add("ff_compute", clk_compute.ms());
    }
    if (dbg_buf) ggml_backend_buffer_free(dbg_buf);
    ggml_free(g);
    ggml_backend_buffer_free(in_buf);
    ggml_free(ctx_in);
    return ok;
}

}  // namespace vv
