#include "qwen2.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "rms_norm.hpp"
#include "rope.hpp"

#include <cmath>

namespace vv {

namespace {

inline struct ggml_tensor* maybe_cont(struct ggml_context* ctx,
                                      struct ggml_tensor*  t) {
    return ggml_is_contiguous(t) ? t : ggml_cont(ctx, t);
}

// Y = matmul(W, X) + bias.  bias may be null.
inline struct ggml_tensor* linear(struct ggml_context* ctx,
                                  struct ggml_tensor*  W,
                                  struct ggml_tensor*  X,
                                  struct ggml_tensor*  bias) {
    struct ggml_tensor* y = ggml_mul_mat(ctx, W, X);
    if (bias) y = ggml_add(ctx, y, bias);
    return y;
}

// SwiGLU FFN: y = down( silu(gate(x)) * up(x) )
//
// Use ggml_swiglu_split which fuses silu(g) * u into a single op (saves
// the intermediate `silu_g` buffer + one extra memory pass; matches the
// pattern llama.cpp uses for Qwen2 in src/models/qwen2.cpp).
struct ggml_tensor* swiglu_ffn(struct ggml_context*     ctx,
                               struct ggml_tensor*      x,
                               const Qwen2LayerWeights& w) {
    struct ggml_tensor* g  = ggml_mul_mat(ctx, w.ffn_gate, x);
    struct ggml_tensor* u  = ggml_mul_mat(ctx, w.ffn_up,   x);
    struct ggml_tensor* gu = ggml_swiglu_split(ctx, g, u);
    return ggml_mul_mat(ctx, w.ffn_down, gu);
}

}  // namespace

bool qwen2_load_layer(const ModelLoader& m,
                      const std::string& prefix,
                      Qwen2LayerWeights* out) {
    if (!out) return false;
    auto get = [&](const std::string& name, bool required, struct ggml_tensor** dst) -> bool {
        struct ggml_tensor* t = m.tensor(prefix + name);
        if (!t) {
            if (required) {
                VV_LOG_ERROR("qwen2_load_layer: missing %s%s",
                             prefix.c_str(), name.c_str());
                return false;
            }
            return true;
        }
        *dst = t;
        return true;
    };
    bool ok = true;
    ok &= get("weight.attn_norm",    true,  &out->attn_norm);
    ok &= get("weight.attn_q",       true,  &out->attn_q);
    ok &= get("weight.attn_q_bias",  false, &out->attn_q_bias);
    ok &= get("weight.attn_k",       true,  &out->attn_k);
    ok &= get("weight.attn_k_bias",  false, &out->attn_k_bias);
    ok &= get("weight.attn_v",       true,  &out->attn_v);
    ok &= get("weight.attn_v_bias",  false, &out->attn_v_bias);
    ok &= get("weight.attn_o",       true,  &out->attn_o);
    ok &= get("weight.ffn_norm",     true,  &out->ffn_norm);
    ok &= get("weight.ffn_gate",     true,  &out->ffn_gate);
    ok &= get("weight.ffn_up",       true,  &out->ffn_up);
    ok &= get("weight.ffn_down",     true,  &out->ffn_down);
    return ok;
}

Qwen2LayerOutput qwen2_layer_forward(struct ggml_context*     ctx,
                                     struct ggml_tensor*      x,
                                     struct ggml_tensor*      pos,
                                     struct ggml_tensor*      mask,
                                     struct ggml_tensor*      k_past,
                                     struct ggml_tensor*      v_past,
                                     const Qwen2LayerWeights& w,
                                     const Qwen2Hparams&      hp) {
    const int hd     = hp.head_dim;
    const int n_h    = hp.n_heads;
    const int n_kv_h = hp.n_kv_heads;

    const int64_t n_tokens = x->ne[1];
    const int64_t n_batch  = x->ne[2] > 0 ? x->ne[2] : 1;

    // ---- attention pre-norm ----
    struct ggml_tensor* xn = rms_norm(ctx, x, w.attn_norm, hp.rms_norm_eps);

    // ---- q, k, v ----
    struct ggml_tensor* q = linear(ctx, w.attn_q, xn, w.attn_q_bias);
    struct ggml_tensor* k = linear(ctx, w.attn_k, xn, w.attn_k_bias);
    struct ggml_tensor* v = linear(ctx, w.attn_v, xn, w.attn_v_bias);

    // Reshape to [hd, n_h, seq, batch] and [hd, n_kv_h, seq, batch].
    q = ggml_reshape_4d(ctx, q, hd, n_h,    n_tokens, n_batch);
    k = ggml_reshape_4d(ctx, k, hd, n_kv_h, n_tokens, n_batch);
    v = ggml_reshape_4d(ctx, v, hd, n_kv_h, n_tokens, n_batch);

    // ---- RoPE on Q and K ----
    q = ggml_rope_ext(ctx, q, pos, /*freq_factors=*/nullptr,
                      hd, kRopeMode, /*n_ctx_orig=*/0,
                      hp.rope_theta, /*freq_scale=*/1.0f,
                      /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                      /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
    k = ggml_rope_ext(ctx, k, pos, /*freq_factors=*/nullptr,
                      hd, kRopeMode, 0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // ---- KV cache: concat past K/V along the sequence dim ----
    // K and V are both [hd, n_kv_h, n_tokens, B]. After concat along axis 2:
    // [hd, n_kv_h, past+n_tokens, B].
    struct ggml_tensor* k_full = k;
    struct ggml_tensor* v_full = v;
    if (k_past) {
        k_full = ggml_concat(ctx, k_past, k, /*dim=*/2);
    }
    if (v_past) {
        v_full = ggml_concat(ctx, v_past, v, /*dim=*/2);
    }

    // Permute for attention math (matches llama.cpp's build_attn_mha).
    //   q:      [hd, n_h,  seq,    b]    -> [hd, seq,    n_h, b]
    //   k_full: [hd, n_kv, seq_kv, b]    -> [hd, seq_kv, n_kv, b]
    //   v_full: same
    struct ggml_tensor* q_p = ggml_permute(ctx, q,      0, 2, 1, 3);
    struct ggml_tensor* k_p = ggml_permute(ctx, k_full, 0, 2, 1, 3);
    struct ggml_tensor* v_p = ggml_permute(ctx, v_full, 0, 2, 1, 3);

    struct ggml_tensor* o = qwen2_attention(ctx, q_p, k_p, v_p, mask, hp);
    if (n_batch > 1) {
        o = ggml_reshape_3d(ctx, o, n_h * hd, n_tokens, n_batch);
    }

    // Output projection.
    o = ggml_mul_mat(ctx, w.attn_o, o);

    // Residual.
    struct ggml_tensor* h = ggml_add(ctx, x, o);

    // ---- FFN ----
    struct ggml_tensor* hn = rms_norm(ctx, h, w.ffn_norm, hp.rms_norm_eps);
    struct ggml_tensor* f  = swiglu_ffn(ctx, hn, w);

    // Final residual.
    Qwen2LayerOutput out;
    out.y      = ggml_add(ctx, h, f);
    out.k_full = k_full;
    out.v_full = v_full;
    return out;
}

// qwen2_attention ------------------------------------------------------------

struct ggml_tensor* qwen2_attention(struct ggml_context* ctx,
                                    struct ggml_tensor*  q_p,
                                    struct ggml_tensor*  k_p,
                                    struct ggml_tensor*  v_p,
                                    struct ggml_tensor*  mask,
                                    const Qwen2Hparams&  hp) {
    const int     hd       = hp.head_dim;
    const int     n_h      = hp.n_heads;
    const int64_t n_tokens = q_p->ne[1];
    const int64_t n_batch  = q_p->ne[3] > 0 ? q_p->ne[3] : 1;
    const float   scale    = 1.0f / std::sqrt(static_cast<float>(hd));
    struct ggml_tensor* o = nullptr;
    if (hp.use_flash_attn) {
        struct ggml_tensor* k_fa = (k_p->type == GGML_TYPE_F32)
                                   ? ggml_cast(ctx, k_p, GGML_TYPE_F16) : k_p;
        struct ggml_tensor* v_fa = (v_p->type == GGML_TYPE_F32)
                                   ? ggml_cast(ctx, v_p, GGML_TYPE_F16) : v_p;
        o = ggml_flash_attn_ext(ctx, q_p, k_fa, v_fa, mask,
                                scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        o = ggml_cont_2d(ctx, o, n_h * hd, n_tokens * n_batch);
    } else {
        struct ggml_tensor* scores = ggml_mul_mat(ctx, k_p, q_p);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        struct ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, /*max_bias=*/0.0f);
        struct ggml_tensor* v_t  = maybe_cont(ctx, ggml_transpose(ctx, v_p));
        o = ggml_mul_mat(ctx, v_t, attn);
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont_2d(ctx, o, n_h * hd, n_tokens * n_batch);
    }
    return o;
}

// ResidentKV ----------------------------------------------------------------

bool ResidentKV::init(int n_layers, int hd, int n_kv, int max_seq_in, ggml_type type_in) {
    free();

    struct ggml_init_params p {};
    p.mem_size = ggml_tensor_overhead() * (2 * n_layers + 16);
    p.no_alloc = true;
    ctx = ggml_init(p);
    if (!ctx) return false;

    k.assign(n_layers, nullptr);
    v.assign(n_layers, nullptr);
    for (int i = 0; i < n_layers; ++i) {
        k[i] = ggml_new_tensor_4d(ctx, type_in, hd, n_kv, max_seq_in, 1);
        v[i] = ggml_new_tensor_4d(ctx, type_in, hd, n_kv, max_seq_in, 1);
        if (!k[i] || !v[i]) { free(); return false; }
    }
    buffer = vv::allocate_ctx_tensors(ctx);
    if (!buffer) { free(); return false; }

    max_seq  = max_seq_in;
    past_len = 0;
    type     = type_in;
    return true;
}

void ResidentKV::free() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
    k.clear();
    v.clear();
    max_seq  = 0;
    past_len = 0;
}

ResidentKV::~ResidentKV() { free(); }

// qwen2_layer_forward_resident ---------------------------------------------

Qwen2LayerOutputResident qwen2_layer_forward_resident(
    struct ggml_context*     ctx,
    struct ggml_tensor*      x,
    struct ggml_tensor*      pos,
    struct ggml_tensor*      mask,
    ResidentKV&              kvs,
    int                      layer_idx,
    const Qwen2LayerWeights& w,
    const Qwen2Hparams&      hp) {

    const int hd     = hp.head_dim;
    const int n_h    = hp.n_heads;
    const int n_kv_h = hp.n_kv_heads;

    const int64_t n_tokens = x->ne[1];
    const int64_t n_batch  = x->ne[2] > 0 ? x->ne[2] : 1;
    const int     past_len = kvs.past_len;
    const int     full_len = past_len + static_cast<int>(n_tokens);

    GGML_ASSERT(full_len <= kvs.max_seq);
    GGML_ASSERT(layer_idx >= 0 && layer_idx < static_cast<int>(kvs.k.size()));

    // ---- attention pre-norm + Q/K/V + RoPE on the new tokens only ----
    struct ggml_tensor* xn = rms_norm(ctx, x, w.attn_norm, hp.rms_norm_eps);
    struct ggml_tensor* q  = linear(ctx, w.attn_q, xn, w.attn_q_bias);
    struct ggml_tensor* k  = linear(ctx, w.attn_k, xn, w.attn_k_bias);
    struct ggml_tensor* v  = linear(ctx, w.attn_v, xn, w.attn_v_bias);

    q = ggml_reshape_4d(ctx, q, hd, n_h,    n_tokens, n_batch);
    k = ggml_reshape_4d(ctx, k, hd, n_kv_h, n_tokens, n_batch);
    v = ggml_reshape_4d(ctx, v, hd, n_kv_h, n_tokens, n_batch);

    q = ggml_rope_ext(ctx, q, pos, /*freq_factors=*/nullptr,
                      hd, kRopeMode, /*n_ctx_orig=*/0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, /*freq_factors=*/nullptr,
                      hd, kRopeMode, 0,
                      hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // ---- write new K/V into the resident buffer at offset past_len ----
    // Resident shape per layer: [hd, n_kv, max_seq, 1].
    // The destination view covers slots [past_len, past_len+n_tokens) on
    // the seq axis (axis 2).
    struct ggml_tensor* k_resident = kvs.k[layer_idx];
    struct ggml_tensor* v_resident = kvs.v[layer_idx];
    const size_t k_offset = static_cast<size_t>(past_len) * k_resident->nb[2];
    const size_t v_offset = static_cast<size_t>(past_len) * v_resident->nb[2];

    struct ggml_tensor* k_dst = ggml_view_4d(ctx, k_resident,
        hd, n_kv_h, n_tokens, n_batch,
        k_resident->nb[1], k_resident->nb[2], k_resident->nb[3], k_offset);
    struct ggml_tensor* v_dst = ggml_view_4d(ctx, v_resident,
        hd, n_kv_h, n_tokens, n_batch,
        v_resident->nb[1], v_resident->nb[2], v_resident->nb[3], v_offset);

    struct ggml_tensor* k_write = ggml_cpy(ctx, k, k_dst);
    struct ggml_tensor* v_write = ggml_cpy(ctx, v, v_dst);

    // ---- read full K/V (past + new) as a view of the resident tensor ----
    // The graph builder must add k_write/v_write to the forward expand
    // BEFORE these reads so the writes complete first; that's the caller's
    // job (we return the writes alongside `y`).
    struct ggml_tensor* k_full = ggml_view_4d(ctx, k_resident,
        hd, n_kv_h, full_len, n_batch,
        k_resident->nb[1], k_resident->nb[2], k_resident->nb[3], 0);
    struct ggml_tensor* v_full = ggml_view_4d(ctx, v_resident,
        hd, n_kv_h, full_len, n_batch,
        v_resident->nb[1], v_resident->nb[2], v_resident->nb[3], 0);

    // ---- attention (FA when supported, eager otherwise) ----
    struct ggml_tensor* q_p = ggml_permute(ctx, q,      0, 2, 1, 3);
    struct ggml_tensor* k_p = ggml_permute(ctx, k_full, 0, 2, 1, 3);
    struct ggml_tensor* v_p = ggml_permute(ctx, v_full, 0, 2, 1, 3);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    struct ggml_tensor* o = nullptr;

    if (hp.use_flash_attn) {
        struct ggml_tensor* k_fa = (k_p->type == GGML_TYPE_F32)
                                   ? ggml_cast(ctx, k_p, GGML_TYPE_F16) : k_p;
        struct ggml_tensor* v_fa = (v_p->type == GGML_TYPE_F32)
                                   ? ggml_cast(ctx, v_p, GGML_TYPE_F16) : v_p;
        o = ggml_flash_attn_ext(ctx, q_p, k_fa, v_fa, mask,
                                scale, /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        o = ggml_cont_2d(ctx, o, n_h * hd, n_tokens * n_batch);
    } else {
        struct ggml_tensor* scores = ggml_mul_mat(ctx, k_p, q_p);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        struct ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, /*max_bias=*/0.0f);
        struct ggml_tensor* v_t  = maybe_cont(ctx, ggml_transpose(ctx, v_p));
        o = ggml_mul_mat(ctx, v_t, attn);
        o = ggml_permute(ctx, o, 0, 2, 1, 3);
        o = ggml_cont_2d(ctx, o, n_h * hd, n_tokens * n_batch);
    }
    if (n_batch > 1) {
        o = ggml_reshape_3d(ctx, o, n_h * hd, n_tokens, n_batch);
    }

    o = ggml_mul_mat(ctx, w.attn_o, o);
    struct ggml_tensor* h  = ggml_add(ctx, x, o);
    struct ggml_tensor* hn = rms_norm(ctx, h, w.ffn_norm, hp.rms_norm_eps);
    struct ggml_tensor* f  = swiglu_ffn(ctx, hn, w);

    Qwen2LayerOutputResident out;
    out.y       = ggml_add(ctx, h, f);
    out.k_write = k_write;
    out.v_write = v_write;
    return out;
}

}  // namespace vv
