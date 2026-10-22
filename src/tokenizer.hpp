#ifndef VIBEVOICE_TOKENIZER_HPP
#define VIBEVOICE_TOKENIZER_HPP

// Qwen2 byte-level BPE tokenizer.
//
// Loads vocab + merges from a `tokenizer.gguf` produced by
// scripts/convert_tokenizer.py.
//
// Pre-tokenizer is a hand-written approximation of the Qwen2 regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+
//   |\p{N}
//   |\s?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
//
// ASCII coverage is exact. Non-ASCII codepoints are heuristically classified
// (treated as Letter) — sufficient for CJK and accented Latin, but emoji and
// some symbols may diverge from HF. Replace `is_letter_cp`/`is_number_cp`
// with a full Unicode table to extend coverage.

#include "model_loader.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vv {

class Tokenizer {
public:
    Tokenizer();

    bool load(const ModelLoader& m);
    bool load_from_file(const std::string& path);

    // Deviations from the canonical Qwen2 encoding, measured to help the
    // realtime TTS model pronounce rare words (see AGENTS.md, "Gotchas").
    // Both default to off: encode(text) is the canonical encoding.
    struct EncodeOptions {
        // During BPE, refuse a merge whose merged token id is >= this value
        // (0 = off). Rare words that are a single very-high-id token come
        // out as more frequent sub-word pieces instead.
        int32_t max_merge_token_id = 0;
        // For text starting with an ASCII letter, encode the first pre-token
        // as if it followed a space and drop the space from the first
        // resulting token when that spaceless twin exists in the vocab
        // ("Crypt"+"ography" rather than "C"+"ryptography"). Otherwise the
        // normal encoding is kept.
        bool canonical_first_word = false;
        // When the text's last word (the last pre-token with a letter in
        // it) is a single token with id >= this, re-encode it with merges
        // capped at this id (0 = off). The realtime model rushes the
        // utterance-final word; a rare word that is one token comes out
        // garbled there ("cryptocurrency?" -> "cryptocur"), frequent
        // sub-word pieces survive. Only for text that is known to be
        // complete: a streamed prefix has no last word.
        int32_t last_word_max_merge_token_id = 0;
    };

    // Encode UTF-8 text to ids. Special tokens in the input string are
    // recognized and emitted as their reserved id (no BPE applied).
    std::vector<int32_t> encode(const std::string& text) const { return encode(text, EncodeOptions{}); }
    std::vector<int32_t> encode(const std::string& text, const EncodeOptions& opts) const;

    // Decode ids back to UTF-8 text. Special tokens are emitted as their
    // text form.
    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }

    size_t vocab_size() const { return tokens_.size(); }
    const std::string& token(int32_t id) const;

    // For tests/debugging.
    std::vector<std::string> pre_tokenize(const std::string& text) const;

private:
    void build_byte_maps();
    int  bpe_pretoken_to_ids(const std::string& pre_tok, int32_t max_merge_token_id,
                             std::vector<int32_t>& out) const;
    void bpe_first_word(const std::string& pre_tok, int32_t max_merge_token_id,
                        std::vector<int32_t>& out) const;

    std::vector<std::string>                tokens_;       // id -> displayable string
    std::vector<int32_t>                    token_type_;
    std::unordered_map<std::string, int32_t> token_id_;    // displayable string -> id

    // merges_: pair (a,b) -> rank (lower rank = applied earlier)
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
            std::hash<std::string> h;
            return h(p.first) * 0x9e3779b97f4a7c15ull ^ h(p.second);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merge_rank_;

    // Special tokens (must be matched verbatim before BPE).
    std::vector<std::pair<std::string, int32_t>> special_;  // sorted by length desc

    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;

    // GPT-2 byte-level maps.
    std::string byte_to_chstr_[256];   // each byte maps to a UTF-8 1-char string
    std::unordered_map<std::string, uint8_t> chstr_to_byte_;
};

}  // namespace vv

#endif  // VIBEVOICE_TOKENIZER_HPP
