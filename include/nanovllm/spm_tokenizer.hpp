#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <utility>

namespace spm {

struct Vocab {
    std::vector<std::string> piece;
  std::vector<float> score;
  int unk_id = 0;
  std::vector<std::string> specials;
};

static inline bool is_cont_byte(unsigned char b) {
    return (b & 0xC0) == 0x80;
}

static inline bool starts_with_meta(const std::string& p) {
    // U+2581 "Ã¢âÂ" UTF-8: 0xE2 0x96 0x81
    return p.size() >= 3 && (unsigned char)p[0] == 0xE2 &&
           (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81;
}

static inline std::string strip_meta(const std::string& p) {
    if (starts_with_meta(p) && p.size() >= 3) return p.substr(3);
    return p;
}

static inline std::string utf8_substr(const std::string& s, size_t start, size_t len) {
    size_t i = 0, pos = 0;
    while (i < s.size() && pos < start) {
        if (!is_cont_byte((unsigned char)s[i])) pos++;
        i++;
    }
    size_t j = i;
    while (j < s.size() && pos < start + len) {
        if (!is_cont_byte((unsigned char)s[j])) pos++;
        j++;
    }
    return s.substr(i, j - i);
}

static inline size_t utf8_len(const std::string& s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size();) {
        if (!is_cont_byte((unsigned char)s[i])) n++;
        i++;
    }
    return n;
}

// Minimal JSON helpers (tolerant, no exceptions)
static inline void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
}

static inline std::string parse_json_string(const std::string& s, size_t& i) {
    std::string out;
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') return out;
    i++; // skip quote
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[i + 1];
            switch (c) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'u': {
                    if (i + 6 <= s.size()) {
                        std::string hex = s.substr(i + 2, 4);
                        unsigned int cp = (unsigned int)std::strtoul(hex.c_str(), nullptr, 16);
                        if (cp < 0x80) out += (char)cp;
                        else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        }
                        i += 4;
                    }
                    break;
                }
                default: out += c; break;
            }
            i += 2;
        } else {
            out += s[i];
            i++;
        }
    }
    if (i < s.size()) i++; // skip closing quote
    return out;
}

static inline void parse_json_array_strings(const std::string& s, size_t& i, std::vector<std::string>& out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[') return;
    i++;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']') { i++; break; }
        if (i < s.size() && s[i] == '"') {
            out.push_back(parse_json_string(s, i));
        } else {
            i++;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') i++;
    }
}

static inline void parse_json_array_floats(const std::string& s, size_t& i, std::vector<float>& out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[') return;
    i++;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']') { i++; break; }
        size_t start = i;
        while (i < s.size() && s[i] != ',' && s[i] != ']' && s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r') i++;
        std::string num = s.substr(start, i - start);
        if (!num.empty()) out.push_back((float)std::strtod(num.c_str(), nullptr));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') i++;
    }
}

static inline void parse_json_array_objects(const std::string& s, size_t& i, std::vector<std::pair<int, std::string>>& out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[') return;
    i++;
    while (i < s.size()) {
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']') { i++; break; }
        if (i < s.size() && s[i] == '{') {
            int id = -1;
            std::string piece;
            i++;
            while (i < s.size() && s[i] != '}') {
                skip_ws(s, i);
                std::string key = parse_json_string(s, i);
                skip_ws(s, i);
                if (i < s.size() && s[i] == ':') i++;
                skip_ws(s, i);
                if (key == "id") {
                    size_t start = i;
                    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ' ' && s[i] != '\t') i++;
                    id = std::atoi(s.substr(start, i - start).c_str());
                } else if (key == "piece") {
                    piece = parse_json_string(s, i);
                } else {
                    // skip value
                    while (i < s.size() && s[i] != ',' && s[i] != '}') i++;
                }
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',') i++;
            }
            if (i < s.size()) i++; // skip }
            if (id >= 0) out.push_back({id, piece});
        } else {
            i++;
        }
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') i++;
    }
}

Vocab parse_vocab(const std::string& json_text) {
    Vocab v;
    size_t i = 0;
    skip_ws(json_text, i);
    if (i >= json_text.size() || json_text[i] != '{') return v;
    i++;
    while (i < json_text.size()) {
        skip_ws(json_text, i);
        if (i >= json_text.size() || json_text[i] == '}') break;
        std::string key = parse_json_string(json_text, i);
        skip_ws(json_text, i);
        if (i < json_text.size() && json_text[i] == ':') i++;
        skip_ws(json_text, i);
        if (key == "vocab") {
            // could be array of objects or array of strings
            if (i < json_text.size() && json_text[i] == '[') {
                // peek next non-ws char
                size_t j = i + 1;
                skip_ws(json_text, j);
                if (j < json_text.size() && json_text[j] == '{') {
                    std::vector<std::pair<int, std::string>> objs;
                    parse_json_array_objects(json_text, i, objs);
                    // sort by id
                    std::sort(objs.begin(), objs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                    for (auto& o : objs) {
                        if ((int)v.piece.size() <= o.first) v.piece.resize(o.first + 1);
                        v.piece[o.first] = o.second;
                    }
                } else {
                    parse_json_array_strings(json_text, i, v.piece);
                }
            }
        } else if (key == "scores") {
            parse_json_array_floats(json_text, i, v.score);
        } else if (key == "trainer_spec") {
            // parse special_tokens
            skip_ws(json_text, i);
            if (i < json_text.size() && json_text[i] == '{') {
                i++;
                while (i < json_text.size()) {
                    skip_ws(json_text, i);
                    if (i >= json_text.size() || json_text[i] == '}') break;
                    std::string k2 = parse_json_string(json_text, i);
                    skip_ws(json_text, i);
                    if (i < json_text.size() && json_text[i] == ':') i++;
                    skip_ws(json_text, i);
                    if (k2 == "special_tokens") {
                        if (i < json_text.size() && json_text[i] == '[') {
                            size_t j = i + 1;
                            skip_ws(json_text, j);
                            if (j < json_text.size() && json_text[j] == '{') {
                                std::vector<std::pair<int, std::string>> objs;
                                parse_json_array_objects(json_text, i, objs);
                                for (auto& o : objs) {
                                    v.specials.push_back(o.second);
                                    if ((int)v.piece.size() <= o.first) v.piece.resize(o.first + 1);
                                    v.piece[o.first] = o.second;
                                }
                            } else {
                                std::vector<std::string> strs;
                                parse_json_array_strings(json_text, i, strs);
                                for (auto& s : strs) v.specials.push_back(s);
                            }
                        }
                    } else {
                        // skip value
                        while (i < json_text.size() && json_text[i] != ',' && json_text[i] != '}') i++;
                    }
                    skip_ws(json_text, i);
                    if (i < json_text.size() && json_text[i] == ',') i++;
                }
                if (i < json_text.size() && json_text[i] == '}') i++;
            }
        } else {
            // skip value
            while (i < json_text.size() && json_text[i] != ',' && json_text[i] != '}') i++;
        }
        skip_ws(json_text, i);
        if (i < json_text.size() && json_text[i] == ',') i++;
    }
    // find unk_id
    for (size_t k = 0; k < v.piece.size(); k++) {
        if (v.piece[k] == "<unk>") { v.unk_id = (int)k; break; }
    }
    return v;
}

// Viterbi encode
std::vector<int> encode(const Vocab& v, const std::string& text) {
    std::vector<int> result;
    if (text.empty() || v.piece.empty()) return result;

    size_t n = utf8_len(text);
    if (n == 0) return result;

    // Build a map from piece string (without meta) to id for lookup
    // We'll do Viterbi: best_score[i] = best log-prob to reach position i
    // For each position, try all pieces that match text[i..i+len(piece)]

    std::vector<float> best(n + 1, -1e30f);
    std::vector<int> best_prev(n + 1, -1);
    std::vector<int> best_piece(n + 1, -1);
    best[0] = 0.0f;

    // Precompute byte offsets for each utf8 char position
    std::vector<size_t> byte_pos(n + 1, 0);
    size_t bi = 0;
    for (size_t ci = 0; ci < n; ci++) {
        byte_pos[ci] = bi;
        bi++;
        while (bi < text.size() && is_cont_byte((unsigned char)text[bi])) bi++;
    }
    byte_pos[n] = bi;

    for (size_t i = 0; i < n; i++) {
        if (best[i] <= -1e29f) continue;
        for (size_t pid = 0; pid < v.piece.size(); pid++) {
            const std::string& p = v.piece[pid];
            std::string stripped = strip_meta(p);
            if (stripped.empty()) continue;
            size_t plen = utf8_len(stripped);
            if (i + plen > n) continue;
            // compare bytes
            size_t sp = byte_pos[i];
            size_t ep = byte_pos[i + plen];
            if (ep - sp != stripped.size()) continue;
            if (text.compare(sp, stripped.size(), stripped) != 0) continue;
            float sc = (pid < v.score.size()) ? v.score[pid] : 0.0f;
            float val = best[i] + sc;
            if (val > best[i + plen]) {
                best[i + plen] = val;
                best_prev[i + plen] = (int)i;
                best_piece[i + plen] = (int)pid;
            }
        }
    }

    // Backtrack
    if (best[n] <= -1e29f) {
        // fallback: emit unk for each char
        for (size_t i = 0; i < n; i++) {
            result.push_back(v.unk_id);
        }
        return result;
    }

    std::vector<int> path;
    int cur = (int)n;
    while (cur > 0) {
        path.push_back(best_piece[cur]);
        cur = best_prev[cur];
    }
    std::reverse(path.begin(), path.end());

    // Handle meta-symbols: first piece gets no leading space, subsequent pieces
    // with meta get a space before them
    bool first = true;
    for (int pid : path) {
        const std::string& p = v.piece[pid];
        if (starts_with_meta(p)) {
            if (!first) {
                // emit space before
                // We represent space by emitting the piece as-is (with meta)
                // but the decode will handle it. For encode, we just push the id.
            }
            first = false;
        } else {
            first = false;
        }
        result.push_back(pid);
    }

    return result;
}

std::string decode(const Vocab& v, const std::vector<int>& ids) {
    std::string out;
    bool first = true;
    for (int id : ids) {
        if (id < 0 || id >= (int)v.piece.size()) continue;
        const std::string& p = v.piece[id];
        if (starts_with_meta(p)) {
            if (!first) out += ' ';
            out += strip_meta(p);
        } else {
            out += p;
        }
        first = false;
    }
    return out;
}

} // namespace spm

#ifdef SPM_TEST
#include <cassert>
#include <iostream>
int main() {
    // Build a minimal vocab for testing
    spm::Vocab v;
    // id 0: <unk>
    v.piece.push_back("<unk>");
    v.score.push_back(-10.0f);
    // id 1: Ã¢âÂhello
    v.piece.push_back(std::string("\xE2\x96\x81") + "hello");
    v.score.push_back(-1.0f);
    // id 2: Ã¢âÂa
    v.piece.push_back(std::string("\xE2\x96\x81") + "a");
    v.score.push_back(-2.0f);
    // id 3: Ã¢âÂb
    v.piece.push_back(std::string("\xE2\x96\x81") + "b");
    v.score.push_back(-2.0f);
    v.unk_id = 0;

    // Test 1: encode("hello") -> {1}
    {
        auto enc = spm::encode(v, "hello");
        assert(enc.size() == 1);
        assert(enc[0] == 1);
    }

    // Test 2: encode("a b") -> {2, 3}
    {
        auto enc = spm::encode(v, "a b");
        assert(enc.size() == 2);
        assert(enc[0] == 2);
        assert(enc[1] == 3);
    }

    // Test 3: decode({1}) == "hello"
    {
        auto dec = spm::decode(v, {1});
        assert(dec == "hello");
    }

    std::cout << "All SPM tests passed.\n";
    return 0;
}
#endif
