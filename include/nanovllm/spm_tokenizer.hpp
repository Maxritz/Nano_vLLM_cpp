#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <algorithm>

namespace spm {

inline bool is_digit(unsigned char c){ return c>='0' && c<='9'; }
inline bool is_punct(unsigned char c){ return (c>=33&&c<=47)||(c>=58&&c<=64)||(c>=91&&c<=96)||(c>=123&&c<=126); }
inline bool is_alpha_or_high(unsigned char c){ return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||c>=0x80; }

struct Vocab {
    std::vector<std::string> pieces;
    std::vector<int32_t> types;
    int unk_id = -1;
    int bos_id = -1;
    int eos_id = -1;
};

struct Spm {
    static Spm build(const Vocab& v) {
        Spm s;
        s.pieces = v.pieces;
        s.types = v.types;
        s.unk_id = v.unk_id;
        s.bos_id = v.bos_id;
        s.eos_id = v.eos_id;
        s.piece_to_id.clear();
        s.prefix_index.clear();
        s.byte_piece_map.clear();
        for (size_t i = 0; i < v.pieces.size(); ++i) {
            const std::string& p = v.pieces[i];
            s.piece_to_id[p] = static_cast<int>(i);
            int32_t t = (i < v.types.size()) ? v.types[i] : 1;
            if (t == 6) {
                // byte pseudo-piece: <0xHH>
                if (p.size() == 6 && p[0]=='<' && p[1]=='0' && p[2]=='x' && p[5]=='>') {
                    unsigned char byte_val = 0;
                    bool ok = true;
                    for (int k = 0; k < 2; ++k) {
                        char c = p[3+k];
                        byte_val <<= 4;
                        if (c>='0'&&c<='9') byte_val |= (c-'0');
                        else if (c>='A'&&c<='F') byte_val |= (c-'A'+10);
                        else if (c>='a'&&c<='f') byte_val |= (c-'a'+10);
                        else { ok = false; break; }
                    }
                    if (ok) {
                        s.byte_piece_map[byte_val] = static_cast<int>(i);
                    }
                }
            }
            if (p.size() >= 3) {
                std::string key(p.begin(), p.begin() + 3);
                s.prefix_index[key].push_back(static_cast<int>(i));
            } else {
                std::string key(p);
                s.prefix_index[key].push_back(static_cast<int>(i));
            }
        }
        s.has_byte_pieces = !s.byte_piece_map.empty();
        return s;
    }

    std::vector<int> encode(const std::string& text,
        const std::function<std::vector<std::string>(const std::string&)>& pre_split) const {
        std::vector<int> ids;
        std::vector<std::string> segments = pre_split(text);
        for (const std::string& seg : segments) {
            std::string s = seg;
            if (!s.empty() && s[0] == ' ') s = meta_space + s.substr(1);
            else s = meta_space + s;
            std::vector<int> seg_ids = encode_segment(s);
            ids.insert(ids.end(), seg_ids.begin(), seg_ids.end());
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string out;
        for (int id : ids) {
            if (id < 0 || id >= static_cast<int>(pieces.size())) continue;
            const std::string& p = pieces[id];
            int32_t t = (id < static_cast<int>(types.size())) ? types[id] : 1;
            if (t == 3 || t == 4) {
                out += p;
                continue;
            }
            if (t == 6 && p.size() == 6) {
                auto hx = [](char c) -> int { return (c <= '9') ? c - '0' : ((c | 32) - 'a' + 10); };
                out += static_cast<char>((hx(p[3]) << 4) | hx(p[4]));
                continue;
            }  // <0xHH> byte piece
            unsigned char c0 = static_cast<unsigned char>(p[0]), c1 = static_cast<unsigned char>(p[1]),
                           c2 = static_cast<unsigned char>(p[2]);
            if (p.size() >= 3 && c0 == 0xE2 && c1 == 0x96 && c2 == 0x81) {
                if (!out.empty() && out.back() != ' ') out += ' ';
                out += p.substr(3);
            } else {
                out += p;
            }
        }
        return out;
    }

private:
    std::vector<std::string> pieces;
    std::vector<int32_t> types;
    int unk_id = -1;
    int bos_id = -1;
    int eos_id = -1;
    std::unordered_map<std::string, int> piece_to_id;
    std::unordered_map<std::string, std::vector<int>> prefix_index;
    std::unordered_map<unsigned char, int> byte_piece_map;
    bool has_byte_pieces = false;
    static constexpr const char* meta_space = "\xE2\x96\x81";

    std::vector<int> encode_segment(const std::string& s) const {
        size_t n = s.size();
        std::vector<int> best(n + 1, std::numeric_limits<int>::max());
        std::vector<int> choice(n + 1, -1);
        best[0] = 0;
        for (size_t i = 0; i < n; ++i) {
            if (best[i] == std::numeric_limits<int>::max()) continue;
            std::string key;
            if (i + 3 <= n) {
                key.assign(s.data() + i, 3);
            } else {
                key.assign(s.data() + i, n - i);
            }
            auto it = prefix_index.find(key);
            if (it != prefix_index.end()) {
                for (int pid : it->second) {
                    const std::string& p = pieces[pid];
                    size_t plen = p.size();
                    if (i + plen <= n && s.compare(i, plen, p) == 0) {
                        int score = best[i] + 1;
                        if (score < best[i + plen] ||
                            (score == best[i + plen] && plen > pieces[choice[i + plen]].size()) ||
                            (score == best[i + plen] && plen == pieces[choice[i + plen]].size() && pid < choice[i + plen])) {
                            best[i + plen] = score;
                            choice[i + plen] = pid;
                        }
                    }
                }
            }
            // byte fallback
            if (has_byte_pieces) {
                unsigned char c = static_cast<unsigned char>(s[i]);
                auto bit = byte_piece_map.find(c);
                if (bit != byte_piece_map.end()) {
                    int pid = bit->second;
                    int score = best[i] + 1;
                    if (score < best[i + 1]) {
                        best[i + 1] = score;
                        choice[i + 1] = pid;
                    }
                }
            }
        }
        std::vector<int> result;
        if (best[n] == std::numeric_limits<int>::max()) {
            if (has_byte_pieces) {
                // walk segment; meta-space marker bytes are implicit, every other
                // byte maps to its <0xHH> piece (unk if absent).
                std::string meta(meta_space);
                for (size_t p = 0; p < n; ) {
                    if (s.compare(p, meta.size(), meta) == 0) { p += meta.size(); continue; }
                    auto bit = byte_piece_map.find(static_cast<unsigned char>(s[p]));
                    result.push_back(bit != byte_piece_map.end() ? bit->second : unk_id);
                    ++p;
                }
                return result;
            }
            if (unk_id >= 0) {
                result.push_back(unk_id);
            }
            return result;
        }
        size_t pos = n;
        while (pos > 0) {
            int pid = choice[pos];
            if (pid < 0) break;
            result.push_back(pid);
            pos -= pieces[pid].size();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }
};

inline std::vector<std::string> llama_regex_split(const std::string& text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    size_t n = text.size();
    while (i < n) {
        // optional leading space
        bool has_space = (i < n && text[i] == ' ');
        if (has_space) ++i;
        if (i >= n) break;
        // punctuation
        if (is_punct(text[i])) {
            std::string tok;
            if (has_space) tok += ' ';
            tok += text[i];
            tokens.push_back(tok);
            ++i;
            continue;
        }
        // alpha or high byte
        if (is_alpha_or_high(text[i])) {
            std::string tok;
            if (has_space) tok += ' ';
            while (i < n && is_alpha_or_high(text[i])) {
                tok += text[i];
                ++i;
            }
            tokens.push_back(tok);
            continue;
        }
        // digit
        if (is_digit(text[i])) {
            std::string tok;
            if (has_space) tok += ' ';
            while (i < n && is_digit(text[i])) {
                tok += text[i];
                ++i;
            }
            tokens.push_back(tok);
            continue;
        }
        // unknown byte: emit as single token
        std::string tok;
        if (has_space) tok += ' ';
        tok += text[i];
        tokens.push_back(tok);
        ++i;
    }
    return tokens;
}

inline bool is_punct(char c) {
    static const std::string punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return punct.find(c) != std::string::npos;
}

inline bool is_alpha_or_high(char c) {
    unsigned char uc = static_cast<unsigned char>(c);
    return (uc >= 0x80) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

inline bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

} // namespace spm

#ifdef SPM_TEST
#include <cassert>
#include <iostream>

int main() {
    using namespace spm;
    Vocab v;
    v.pieces = {"<unk>", "<s>", "</s>", "\xE2\x96\x81the", "\xE2\x96\x81" "cat", "\xE2\x96\x81" "dog", "the", "at"};
    v.types = {2, 3, 3, 1, 1, 1, 1, 1};
    v.unk_id = 0;
    v.bos_id = 1;
    v.eos_id = 2;
    Spm spm = Spm::build(v);

    // Test 1: roundtrip
    std::vector<int> ids = spm.encode("the cat", llama_regex_split);
    std::string decoded = spm.decode(ids);
    assert(decoded == "the cat");

    // Test 2: longest match for "the"
    std::vector<int> ids2 = spm.encode("the", llama_regex_split);
    assert(ids2.size() == 1);
    assert(ids2[0] == 3); // "\u2581the"

    // Test 3: unknown char with byte pieces
    Vocab v2;
    v2.pieces = {"<unk>", "<s>", "</s>", "<0x41>", "<0x42>"};
    v2.types = {2, 3, 3, 6, 6};
    v2.unk_id = 0;
    v2.bos_id = 1;
    v2.eos_id = 2;
    Spm spm2 = Spm::build(v2);
    std::vector<int> ids3 = spm2.encode("AB", llama_regex_split);
    assert(ids3.size() == 2);
    assert(ids3[0] == 3); // A
    assert(ids3[1] == 4); // B

    std::cout << "All tests passed.\n";
    return 0;
}
#endif
