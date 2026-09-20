#ifndef VIBEVOICE_QWEN2_HPP
#define VIBEVOICE_QWEN2_HPP

// One Qwen2 transformer layer as a ggml graph builder.
//
// Layer formula:
//   h  = x + Attn(RMSNorm(x))
//   y  = h + FFN(RMSNorm(h))
// where Attn is GQA with RoPE on Q/K, and FFN is SwiGLU.
//
// Weights are stored in PyTorch-style layout (`[out, in]`) — when ggml loads
// them, ne[0] = in (the contiguous dim) and ne[1] = out, which is exactly
// what `ggml_mul_mat(W, X)` expects.

#include "ggml.h"
#include "ggml-backend.h"
#include "model_loader.hpp"

#include <string>
#include <vector>

namespace vv {

struct Qwen2Hparams {
    int   hidden_size       = 0;
    int   n_heads           = 0;
    int   n_kv_heads        = 0;
    int   head_dim          = 0;
    int   intermediate_size = 0;
    float rope_theta        = 1.0e6f;
    float rms_norm_eps      = 1.0e-6f;

    // When true, use ggml_flash_attn_ext instead of materializing the
    // [seq_kv, seq_q, n_h] attention scores tensor. Required for long
    // prefills (audio prefix) on backends that support it; the eager
    // mul_mat -> soft_max -> mul_mat path stays as the fallback.
    // Caller must build mask as F16 (FA contract); the eager path uses F32.
    bool  use_flash_attn    = false;
};

struct Qwen2LayerWeights {
    struct ggml_tensor* attn_norm   = nullptr;
    struct ggml_tensor* attn_q      = nullptr;
    struct ggml_tensor* attn_q_bias = nullptr;
    struct ggml_tensor* attn_k      = nullptr;
    struct ggml_tensor* attn_k_bias = nullptr;
    struct ggml_tensor* attn_v      = nullptr;
    struct ggml_tensor* attn_v_bias = nullptr;
    struct ggml_tensor* attn_o      = nullptr;
    struct ggml_tensor* ffn_norm    = nullptr;
    struct ggml_tensor* ffn_gate    = nullptr;
    struct ggml_tensor* ffn_up      = nullptr;
    struct ggml_tensor* ffn_down    = nullptr;
};

// Load layer weights from a gguf using a name prefix (e.g. "" or "blk.0.").
// Tensor names expected: prefix + "weight.attn_norm", "weight.attn_q",
// "weight.attn_q_bias", … (as produced by tests/dump_qwen2_reference.py).
bool qwen2_load_layer(const ModelLoader& m,
                      const std::string& prefix,
                      Qwen2LayerWeights* out);

struct Qwen2LayerOutput {
    struct ggml_tensor* y       = nullptr;  // [hidden, n_tokens, B]
    struct ggml_tensor* k_full  = nullptr;  // [head_dim, n_kv_heads, past+n_tokens, B]
    struct ggml_tensor* v_full  = nullptr;  // same shape
};

// ResidentKV holds K/V tensors that survive across multiple per-step
// graphs on the active backend's buffer. The classic pattern (see
// llama.cpp's llama-kv-cache-unified.cpp): allocate once at engine
// load, the per-step graph references the resident tensors via views
// and writes new rows via ggml_cpy. No per-step host round-trip, no
// per-step backend allocation for the K/V data.
//
// Lifetime is tied to the ggml_context that owns the tensor metadata;
// the backend buffer owns the data. Both are freed in `free()`.
struct ResidentKV {
    struct ggml_context*    ctx     = nullptr;
    ggml_backend_buffer_t   buffer  = nullptr;
    std::vector<struct ggml_tensor*> k;   // [hd, n_kv, max_seq, 1] per layer
    std::vector<struct ggml_tensor*> v;   // same
    int                     max_seq  = 0;
    int                     past_len = 0;
    ggml_type               type     = GGML_TYPE_F32;

    // `type` F16 halves the cache traffic; flash attention consumes the F16
    // strided views directly, and the per-step ggml_cpy converts on write.
    bool init(int n_layers, int hd, int n_kv, int max_seq, ggml_type type = GGML_TYPE_F32);
    void free();
    ~ResidentKV();
};

// Build a single Qwen2 layer's compute graph. `x` is [hidden, n_tokens, B];
// `pos` is an int32 vector of length n_tokens with ABSOLUTE positions; `mask`
// is the additive attention mask of shape [past+n_tokens, n_tokens, 1, 1].
//
// `k_past` / `v_past` may be null (no cache, prefill from position 0) or
// contain past KV of shape [head_dim, n_kv_heads, past_len, B]. In the
// cached case the new K/V are concatenated with the past along the sequence
// dim before attention.
//
// Returns the layer output AND the updated full K/V so the caller can use
// them as next-step `k_past`/`v_past`.
Qwen2LayerOutput qwen2_layer_forward(struct ggml_context*     ctx,
                                     struct ggml_tensor*      x,
                                     struct ggml_tensor*      pos,
                                     struct ggml_tensor*      mask,
                                     struct ggml_tensor*      k_past,
                                     struct ggml_tensor*      v_past,
                                     const Qwen2LayerWeights& w,
                                     const Qwen2Hparams&      hp);

// Persistent-KV variant. Instead of taking k_past/v_past as graph inputs
// the caller has to upload, this references the resident `kv` tensors
// directly via views and uses ggml_cpy to write the new K/V rows into
// `kv.k[layer_idx]` / `kv.v[layer_idx]` at offset `kv.past_len`.
//
// The caller is responsible for:
//   1. Building forward expand on the returned `k_write` / `v_write` cpy
//      tensors so the writes actually land in the resident buffer.
//   2. Updating `kv.past_len` after the graph has executed.
//
// Mask shape is [kv.past_len + n_tokens, n_tokens, 1, B] - it covers the
// full prefix the resident tensor will hold *after* this step's writes.
struct Qwen2LayerOutputResident {
    struct ggml_tensor* y       = nullptr;
    struct ggml_tensor* k_write = nullptr;  // ggml_cpy result; expand-on-graph
    struct ggml_tensor* v_write = nullptr;
};

// Attention core shared by the layer builders and the fused TTS frame graph.
//   q_p: [hd, n_tokens, n_h, B]      (permuted view is fine)
//   k_p, v_p: [hd, kv_len, n_kv, B]  (permuted views of the resident cache;
//             F16 views go to flash attention without a copy)
//   mask: additive [kv_len, n_tokens] (F16 for the FA path) or null
// Returns [n_h*hd, n_tokens*B], contiguous.
struct ggml_tensor* qwen2_attention(struct ggml_context* ctx,
                                    struct ggml_tensor*  q_p,
                                    struct ggml_tensor*  k_p,
                                    struct ggml_tensor*  v_p,
                                    struct ggml_tensor*  mask,
                                    const Qwen2Hparams&  hp);

Qwen2LayerOutputResident qwen2_layer_forward_resident(
    struct ggml_context*     ctx,
    struct ggml_tensor*      x,
    struct ggml_tensor*      pos,
    struct ggml_tensor*      mask,
    ResidentKV&              kv,
    int                      layer_idx,
    const Qwen2LayerWeights& w,
    const Qwen2Hparams&      hp);

}  // namespace vv

#endif  // VIBEVOICE_QWEN2_HPP
