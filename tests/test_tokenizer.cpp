// Tokenizer parity + encode-option tests.
//
// Loads the Qwen2 tokenizer from the first of:
//   ${VV_FIXTURES_DIR}/tokenizer.gguf            — produced by
//                                                  scripts/convert_tokenizer.py
//   $VIBEVOICE_TOKENIZER                         — the same file elsewhere
//   $VIBEVOICE_TTS_MODEL / $VIBEVOICE_MODEL      — a model gguf that embeds it
// and
//   ${VV_FIXTURES_DIR}/tokenizer_reference.jsonl — produced by
//                                                  tests/dump_tokenizer_reference.py
// then asserts our C++ tokenizer reproduces every reference id stream
// exactly, and that the realtime-TTS encode options (merge-id cap,
// canonical first word) split exactly the tokens they are meant to and
// nothing else.
//
// The test is skipped (returns 77, ctest "skip" code) if no tokenizer is
// found — this keeps it green on a fresh checkout where the user hasn't
// yet downloaded one.

#include "tokenizer.hpp"
#include "model_loader.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool file_ok(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Tiny ad-hoc JSON parser sufficient for our reference jsonl:
//   {"text": "...", "ids": [int, ...]}
struct Record {
    std::string text;
    std::vector<int32_t> ids;
};

bool parse_record(const std::string& line, Record* r) {
    auto t_pos = line.find("\"text\"");
    if (t_pos == std::string::npos) return false;
    auto colon = line.find(':', t_pos);
    auto q1    = line.find('"', colon);
    if (q1 == std::string::npos) return false;
    // Walk until matching unescaped quote.
    std::string s;
    size_t p = q1 + 1;
    while (p < line.size()) {
        char c = line[p];
        if (c == '\\' && p + 1 < line.size()) {
            char e = line[p + 1];
            switch (e) {
                case 'n':  s.push_back('\n'); p += 2; continue;
                case 't':  s.push_back('\t'); p += 2; continue;
                case 'r':  s.push_back('\r'); p += 2; continue;
                case '"':  s.push_back('"');  p += 2; continue;
                case '\\': s.push_back('\\'); p += 2; continue;
                case '/':  s.push_back('/');  p += 2; continue;
                case 'u':
                    if (p + 6 <= line.size()) {
                        unsigned cp = std::stoul(line.substr(p + 2, 4), nullptr, 16);
                        // emit UTF-8
                        if (cp < 0x80) s.push_back(static_cast<char>(cp));
                        else if (cp < 0x800) {
                            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        p += 6; continue;
                    }
                    s.push_back(e); p += 2; continue;
                default: s.push_back(e); p += 2; continue;
            }
        }
        if (c == '"') break;
        s.push_back(c);
        p++;
    }
    r->text = std::move(s);

    auto i_pos = line.find("\"ids\"", p);
    if (i_pos == std::string::npos) return false;
    auto lb = line.find('[', i_pos);
    auto rb = line.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;

    std::string body = line.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(body);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // strip whitespace
        size_t a = 0, b = tok.size();
        while (a < b && std::isspace(static_cast<unsigned char>(tok[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(tok[b - 1]))) --b;
        if (a == b) continue;
        r->ids.push_back(std::stoi(tok.substr(a, b - a)));
    }
    return true;
}

using Ids  = std::vector<int32_t>;
using Opts = vv::Tokenizer::EncodeOptions;

std::string fmt_ids(const Ids& ids) {
    std::string s;
    for (size_t k = 0; k < ids.size() && k < 30; ++k) s += std::to_string(ids[k]) + " ";
    return s;
}

// Linear scan for a displayable token string (space = "Ġ"); -1 if absent.
int32_t find_id(const vv::Tokenizer& tok, const std::string& displayable) {
    for (int32_t i = 0; i < static_cast<int32_t>(tok.vocab_size()); ++i)
        if (tok.token(i) == displayable) return i;
    return -1;
}

const std::string kSp = "\xC4\xA0";  // byte-level "Ġ" for ' '

int n_checks = 0, n_bad = 0;
void expect_ids(const char* what, const Ids& got, const Ids& want) {
    ++n_checks;
    if (got == want) return;
    ++n_bad;
    std::fprintf(stderr, "FAIL %s\n  want: %s\n  got:  %s\n", what, fmt_ids(want).c_str(), fmt_ids(got).c_str());
}
void expect_true(const char* what, bool ok) {
    ++n_checks;
    if (ok) return;
    ++n_bad;
    std::fprintf(stderr, "FAIL %s\n", what);
}

int run_parity(const vv::Tokenizer& tok, const std::string& jsonl, std::vector<Record>* records) {
    std::ifstream in(jsonl);
    std::string line;
    int n_total = 0, n_pass = 0, n_fail = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        Record r;
        if (!parse_record(line, &r)) {
            std::fprintf(stderr, "parse error on line: %s\n", line.c_str());
            return 3;
        }
        ++n_total;
        auto got = tok.encode(r.text);
        records->push_back(r);
        if (got == r.ids) {
            ++n_pass;
            continue;
        }
        ++n_fail;
        if (n_fail <= 5) {
            std::fprintf(stderr, "MISMATCH on text=%.80s\n", r.text.c_str());
            std::fprintf(stderr, "  expected (%zu): %s\n", r.ids.size(), fmt_ids(r.ids).c_str());
            std::fprintf(stderr, "  got      (%zu): %s\n", got.size(), fmt_ids(got).c_str());
        }
    }
    std::fprintf(stderr, "tokenizer parity: %d/%d pass, %d fail\n", n_pass, n_total, n_fail);
    return n_fail == 0 ? 0 : 4;
}

// Encode options. The generic checks run on any vocab; the concrete ones
// need the Qwen2 vocab (the realtime model's) and are skipped otherwise.
int run_options(const vv::Tokenizer& tok, const std::vector<Record>& records) {
    const int32_t kCap = 80000;
    Opts cap;   cap.max_merge_token_id = kCap;
    Opts canon; canon.canonical_first_word = true;
    Opts both = cap; both.canonical_first_word = true;
    Opts cap_all; cap_all.max_merge_token_id = static_cast<int32_t>(tok.vocab_size());

    std::vector<std::string> texts = {"", "\n", " ", "Hello world", " Cryptography", "(Cryptography)",
                                      "\"Cryptography\"", "1984 was", "<|object_ref_start|>Cryptography",
                                      "Cryptography is hard\n"};
    for (const auto& r : records) texts.push_back(r.text);

    for (const auto& t : texts) {
        const Ids plain = tok.encode(t);
        // Default options are byte-identical to the plain overload.
        expect_ids("default options == encode(text)", tok.encode(t, Opts{}), plain);
        // A cap above every id is a no-op.
        expect_ids("cap == vocab_size is a no-op", tok.encode(t, cap_all), plain);

        // The cap only ever yields ids below it (specials bypass BPE) and
        // never changes the decoded text.
        const Ids capped = tok.encode(t, cap);
        bool below = true;
        for (int32_t id : capped) below = below && (id < kCap || tok.token(id).rfind("<|", 0) == 0);
        expect_true("capped ids are all < cap (or special)", below);
        expect_true("cap preserves decoded text", tok.decode(capped) == tok.decode(plain));

        // The canonical first word is a no-op unless the text starts with
        // a letter, and never changes the decoded text.
        const Ids canonical = tok.encode(t, canon);
        const bool starts_letter = !t.empty() && std::isalpha(static_cast<unsigned char>(t[0]));
        if (!starts_letter) expect_ids("canonical first word is a no-op without a leading letter", canonical, plain);
        expect_true("canonical first word preserves decoded text", tok.decode(canonical) == tok.decode(plain));
    }

    // Common first words already split the mid-sentence way: no change.
    expect_ids("canonical first word leaves 'Hello world' alone", tok.encode("Hello world", canon), tok.encode("Hello world"));

    // A first word whose spaced token has no spaceless twin keeps its
    // plain encoding. Find one in the vocab: "Ġ" + letters, twin absent.
    {
        std::string found;
        for (int32_t i = 0; i < static_cast<int32_t>(tok.vocab_size()) && found.empty(); ++i) {
            const std::string& s = tok.token(i);
            if (s.size() < kSp.size() + 4 || s.compare(0, kSp.size(), kSp) != 0) continue;
            const std::string w = s.substr(kSp.size());
            bool letters = true;
            for (unsigned char c : w) letters = letters && std::isalpha(c);
            if (!letters) continue;
            if (find_id(tok, w) >= 0) continue;
            if (tok.encode(" " + w) != Ids{i}) continue;
            found = w;
        }
        if (found.empty()) {
            std::fprintf(stderr, "note: no spaced token without a spaceless twin in this vocab\n");
        } else {
            std::fprintf(stderr, "twin-less word: '%s'\n", found.c_str());
            expect_ids("missing spaceless twin keeps the plain encoding", tok.encode(found, canon), tok.encode(found));
        }
    }

    // Concrete Qwen2 cases (the realtime model's vocab).
    const int32_t id_cryptography  = find_id(tok, kSp + "cryptography");
    const int32_t id_crypt_sp      = find_id(tok, kSp + "crypt");
    const int32_t id_Crypt_sp      = find_id(tok, kSp + "Crypt");
    const int32_t id_Crypt         = find_id(tok, "Crypt");
    const int32_t id_ography       = find_id(tok, "ography");
    const int32_t id_C             = find_id(tok, "C");
    const int32_t id_ryptography   = find_id(tok, "ryptography");
    if (id_cryptography < 0 || id_crypt_sp < 0 || id_Crypt_sp < 0 || id_Crypt < 0 || id_ography < 0 ||
        id_C < 0 || id_ryptography < 0) {
        std::fprintf(stderr, "note: not the Qwen2 vocab; skipping the concrete cryptography cases\n");
    } else {
        std::fprintf(stderr, "qwen2: ' cryptography'=%d ' crypt'=%d ' Crypt'=%d 'Crypt'=%d 'ography'=%d\n",
                     id_cryptography, id_crypt_sp, id_Crypt_sp, id_Crypt, id_ography);
        expect_true("' cryptography' is a single token above the cap", id_cryptography >= kCap);
        expect_true("' crypt' and 'ography' are below the cap", id_crypt_sp < kCap && id_ography < kCap);
        expect_true("'Crypt' is the measured id 53367", id_Crypt == 53367);

        // Cap: splits the high-id token, leaves the rest of the sentence alone.
        expect_ids("plain ' cryptography'", tok.encode(" cryptography"), Ids{id_cryptography});
        expect_ids("cap splits ' cryptography'", tok.encode(" cryptography", cap), Ids{id_crypt_sp, id_ography});
        expect_ids("cap keeps a sentence of low-id tokens", tok.encode(" the quick brown fox jumps", cap),
                   tok.encode(" the quick brown fox jumps"));
        {
            Ids want = tok.encode(" the");
            want.push_back(id_crypt_sp); want.push_back(id_ography);
            const Ids tail = tok.encode(" lecture");
            want.insert(want.end(), tail.begin(), tail.end());
            expect_ids("cap splits only the rare word in a sentence", tok.encode(" the cryptography lecture", cap), want);
        }
        // The boundary is inclusive: an id equal to the cap is refused.
        Opts at_id;    at_id.max_merge_token_id    = id_cryptography;
        Opts above_id; above_id.max_merge_token_id = id_cryptography + 1;
        expect_ids("cap == id refuses the merge", tok.encode(" cryptography", at_id), Ids{id_crypt_sp, id_ography});
        expect_ids("cap == id + 1 keeps the merge", tok.encode(" cryptography", above_id), Ids{id_cryptography});

        // Canonical first word.
        expect_ids("plain 'Cryptography'", tok.encode("Cryptography"), Ids{id_C, id_ryptography});
        expect_ids("canonical 'Cryptography'", tok.encode("Cryptography", canon), Ids{id_Crypt, id_ography});
        {
            Ids want{id_Crypt, id_ography};
            const Ids tail = tok.encode(" is hard\n");
            want.insert(want.end(), tail.begin(), tail.end());
            expect_ids("canonical first word only touches the first word", tok.encode("Cryptography is hard\n", canon), want);
        }
        expect_ids("leading space: canonical is a no-op", tok.encode(" Cryptography", canon), tok.encode(" Cryptography"));
        expect_ids("leading punctuation: canonical is a no-op", tok.encode("(Cryptography)", canon), tok.encode("(Cryptography)"));
        expect_ids("leading special: canonical is a no-op", tok.encode("<|object_ref_start|>Cryptography", canon),
                   tok.encode("<|object_ref_start|>Cryptography"));

        // Both together, as the realtime path runs them.
        expect_ids("both: 'Cryptography'", tok.encode("Cryptography", both), Ids{id_Crypt, id_ography});
        {
            Ids want{id_crypt_sp, id_ography};
            const Ids tail = tok.encode(" is");
            want.insert(want.end(), tail.begin(), tail.end());
            expect_ids("both: ' cryptography is'", tok.encode(" cryptography is", both), want);
        }
    }

    std::fprintf(stderr, "encode options: %d checks, %d failed\n", n_checks, n_bad);
    return n_bad == 0 ? 0 : 5;
}

std::string resolve_tokenizer(const std::string& fixture_gguf) {
    if (file_ok(fixture_gguf)) return fixture_gguf;
    for (const char* env : {"VIBEVOICE_TOKENIZER", "VIBEVOICE_TTS_MODEL", "VIBEVOICE_MODEL"}) {
        const char* p = std::getenv(env);
        if (p && p[0] && file_ok(p)) return p;
    }
    return {};
}

}  // namespace

int main() {
#ifndef VV_FIXTURES_DIR
#  define VV_FIXTURES_DIR "tests/fixtures"
#endif
    const std::string fix    = VV_FIXTURES_DIR;
    const std::string jsonl  = fix + "/tokenizer_reference.jsonl";
    const std::string gguf   = resolve_tokenizer(fix + "/tokenizer.gguf");

    if (gguf.empty()) {
        std::fprintf(stderr,
                     "test_tokenizer: skipping — no tokenizer found.\n"
                     "  expected: %s/tokenizer.gguf (scripts/convert_tokenizer.py),\n"
                     "  or VIBEVOICE_TOKENIZER / VIBEVOICE_TTS_MODEL / VIBEVOICE_MODEL set.\n",
                     fix.c_str());
        return 77;  // ctest skip
    }

    vv::Tokenizer tok;
    if (!tok.load_from_file(gguf)) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", gguf.c_str());
        return 2;
    }

    std::vector<Record> records;
    int rc = 0;
    if (file_ok(jsonl)) {
        rc = run_parity(tok, jsonl, &records);
        if (rc != 0) return rc;
    } else {
        std::fprintf(stderr, "note: %s missing; parity not checked\n", jsonl.c_str());
    }
    return run_options(tok, records);
}
