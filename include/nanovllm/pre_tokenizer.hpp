#pragma once
#include <string>
#include <vector>
#include <regex>
#include <utility>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <cassert>

namespace ptok {

struct JVal {
    enum Type { NUL, STR, BOOL, ARR, OBJ } t = NUL;
    std::string s;
    bool b = false;
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;

    const JVal* at(const std::string& k) const {
        for (const auto& p : obj) if (p.first == k) return &p.second;
        return nullptr;
    }
    bool has(const std::string& k) const { return at(k) != nullptr; }
};

inline JVal parse_json(const std::string& str);

namespace detail {
    inline void skip_ws(const std::string& s, size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }
    inline std::string parse_str(const std::string& s, size_t& i) {
        std::string r;
        ++i; // skip "
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[i + 1];
                switch (c) {
                    case 'n': r += '\n'; break;
                    case 't': r += '\t'; break;
                    case 'r': r += '\r'; break;
                    case '\\': r += '\\'; break;
                    case '"': r += '"'; break;
                    case '/': r += '/'; break;
                    default: r += c; break;
                }
                i += 2;
            } else {
                r += s[i++];
            }
        }
        ++i; // skip "
        return r;
    }
}

inline JVal parse_json(const std::string& str) {
    size_t i = 0;
    std::function<JVal()> parse_val = [&]() -> JVal {
        detail::skip_ws(str, i);
        JVal v;
        if (i >= str.size()) return v;
        char c = str[i];
        if (c == '"') {
            v.t = JVal::STR;
            v.s = detail::parse_str(str, i);
        } else if (c == '{') {
            v.t = JVal::OBJ;
            ++i;
            detail::skip_ws(str, i);
            if (i < str.size() && str[i] == '}') { ++i; return v; }
            while (i < str.size()) {
                detail::skip_ws(str, i);
                std::string key = detail::parse_str(str, i);
                detail::skip_ws(str, i);
                if (i < str.size() && str[i] == ':') ++i;
                JVal val = parse_val();
                v.obj.emplace_back(key, val);
                detail::skip_ws(str, i);
                if (i < str.size() && str[i] == ',') ++i;
                else break;
            }
            if (i < str.size() && str[i] == '}') ++i;
        } else if (c == '[') {
            v.t = JVal::ARR;
            ++i;
            detail::skip_ws(str, i);
            if (i < str.size() && str[i] == ']') { ++i; return v; }
            while (i < str.size()) {
                v.arr.push_back(parse_val());
                detail::skip_ws(str, i);
                if (i < str.size() && str[i] == ',') ++i;
                else break;
            }
            if (i < str.size() && str[i] == ']') ++i;
        } else if (c == 't' || c == 'f') {
            v.t = JVal::BOOL;
            if (str.substr(i, 4) == "true") { v.b = true; i += 4; }
            else { v.b = false; i += 5; }
        } else if (c == 'n') {
            v.t = JVal::NUL;
            i += 4;
        } else {
            v.t = JVal::STR;
            while (i < str.size() && str[i] != ',' && str[i] != '}' && str[i] != ']' &&
                   !(str[i] == ' ' || str[i] == '\t' || str[i] == '\n' || str[i] == '\r')) {
                v.s += str[i++];
            }
        }
        return v;
    };
    return parse_val();
}

inline std::vector<std::string> pre_split(const JVal& pre_tokenizer, const std::string& text);

namespace detail {
    inline bool is_letter(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (static_cast<unsigned char>(c) >= 0x80);
    }
    inline bool is_digit(char c) { return c >= '0' && c <= '9'; }
    inline bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
    inline bool is_punct(char c) {
        return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') ||
               (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
    }

    // Translate POSIX character classes to ECMAScript-safe equivalents
    inline std::string translate_posix_classes(const std::string& regex_str) {
        std::string result;
        result.reserve(regex_str.size());
        for (size_t i = 0; i < regex_str.size(); ++i) {
            if (i + 1 < regex_str.size() && regex_str[i] == '[' && regex_str[i + 1] == ':') {
                // Find the closing [:
                size_t end = regex_str.find("[:", i + 2);
                if (end != std::string::npos) {
                    std::string cls = regex_str.substr(i + 2, end - (i + 2));
                    if (cls == "punct") {
                        result += "[!-/:-@\\[-`{-~]";
                        i = end + 1; // skip past [:
                    } else if (cls == "alpha") {
                        result += "[a-zA-Z\\x80-\\xff]";
                        i = end + 1;
                    } else if (cls == "digit") {
                        result += "[0-9]";
                        i = end + 1;
                    } else if (cls.rfind("^", 0) == 0) {
                        // Negated class [:^X:]
                        std::string inner = cls.substr(1);
                        if (inner == "punct") {
                            result += "[^!-/:-@\\[-`{-~]";
                        } else if (inner == "alpha") {
                            result += "[^a-zA-Z\\x80-\\xff]";
                        } else if (inner == "digit") {
                            result += "[^0-9]";
                        } else {
                            // Unknown negated class, keep as-is
                            result += "[";
                            result += regex_str[i];
                            result += regex_str[i + 1];
                            i = end + 1;
                            continue;
                        }
                        i = end + 1;
                    } else {
                        // Unknown class, keep as-is
                        result += regex_str[i];
                        result += regex_str[i + 1];
                        i = end + 1;
                        continue;
                    }
                } else {
                    result += regex_str[i];
                }
            } else {
                result += regex_str[i];
            }
        }
        return result;
    }

    inline std::vector<std::string> split_regex(const JVal& pt, const std::string& text) {
        const JVal* pat = pt.at("pattern");
        std::string regex_str;
        if (pat && pat->has("Regex")) regex_str = pat->at("Regex")->s;
        std::string behavior = pt.has("behavior") ? pt.at("behavior")->s : "Isolated";
        bool invert = pt.has("invert") ? pt.at("invert")->b : false;

        if (regex_str.empty()) return {text};

        // Translate POSIX classes to ECMAScript-safe equivalents
        regex_str = translate_posix_classes(regex_str);

        std::regex re(regex_str, std::regex::ECMAScript);
        std::vector<std::string> result;
        std::sregex_iterator it(text.begin(), text.end(), re), end;
        std::vector<std::pair<size_t, size_t>> matches;
        for (; it != end; ++it) {
            matches.emplace_back(it->position(), it->length());
        }

        if (behavior == "Isolated") {
            // HF Isolated: matched spans become tokens (whitespace-only match glues
            // to the next match, e.g. " " + "world" -> " world"). invert -> keep gaps.
            if (!invert) {
                for (size_t mi = 0; mi < matches.size(); ++mi) {
                    std::string tok = text.substr(matches[mi].first, matches[mi].second);
                    bool ws = !tok.empty() && std::all_of(tok.begin(), tok.end(),
                        [](char c){ return c==' '||c=='\t'||c=='\n'||c=='\r'; });
                    if (ws && mi + 1 < matches.size()) {
                        tok += text.substr(matches[mi+1].first, matches[mi+1].second);
                    }
                    if (!tok.empty()) result.push_back(tok);
                    if (ws) ++mi;
                }
            } else {
                size_t last = 0;
                for (auto& m : matches) {
                    if (m.first > last) result.push_back(text.substr(last, m.first - last));
                    last = m.first + m.second;
                }
                if (last < text.size()) result.push_back(text.substr(last));
            }
        } else if (behavior == "Merged") {
            size_t last = 0;
            for (auto& m : matches) {
                if (m.first > last) result.push_back(text.substr(last, m.first - last));
                last = m.first + m.second;
            }
            if (last < text.size()) result.push_back(text.substr(last));
            if (invert) {
                // Not well-defined; keep non-matches
            }
        } else if (behavior == "Removed") {
            size_t last = 0;
            for (auto& m : matches) {
                if (m.first > last) result.push_back(text.substr(last, m.first - last));
                last = m.first + m.second;
            }
            if (last < text.size()) result.push_back(text.substr(last));
            if (invert) {
                // Keep matches instead
                result.clear();
                for (auto& m : matches) result.push_back(text.substr(m.first, m.second));
            }
        }
        // Remove empty strings
        std::vector<std::string> filtered;
        for (auto& r : result) if (!r.empty()) filtered.push_back(r);
        return filtered;
    }

    inline std::vector<std::string> byte_level(const JVal& pt, const std::string& text) {
        bool add_prefix_space = pt.has("add_prefix_space") ? pt.at("add_prefix_space")->b : false;
        bool trim_offsets = pt.has("trim_offsets") ? pt.at("trim_offsets")->b : false;
        bool use_regex = pt.has("use_regex") ? pt.at("use_regex")->b : true;

        std::vector<std::string> result;
        if (!use_regex) {
            for (char c : text) result.push_back(std::string(1, c));
            return result;
        }

        size_t i = 0;
        while (i < text.size()) {
            if (is_space(text[i])) {
                size_t j = i;
                while (j < text.size() && is_space(text[j])) ++j;
                result.push_back(text.substr(i, j - i));
                i = j;
            } else if (is_letter(text[i])) {
                size_t j = i;
                while (j < text.size() && is_letter(text[j])) ++j;
                std::string seg = text.substr(i, j - i);
                if (add_prefix_space && (result.empty() || is_space(result.back()[0]))) {
                    seg = " " + seg;
                }
                result.push_back(seg);
                i = j;
            } else if (is_digit(text[i])) {
                size_t j = i;
                while (j < text.size() && is_digit(text[j])) ++j;
                std::string seg = text.substr(i, j - i);
                if (add_prefix_space && (result.empty() || is_space(result.back()[0]))) {
                    seg = " " + seg;
                }
                result.push_back(seg);
                i = j;
            } else {
                result.push_back(std::string(1, text[i]));
                ++i;
            }
        }
        return result;
    }

    inline std::vector<std::string> whitespace(const JVal& pt, const std::string& text) {
        std::vector<std::string> result;
        size_t i = 0;
        while (i < text.size()) {
            while (i < text.size() && is_space(text[i])) ++i;
            size_t j = i;
            while (j < text.size() && !is_space(text[j])) ++j;
            if (j > i) result.push_back(text.substr(i, j - i));
            i = j;
        }
        return result;
    }

    inline std::vector<std::string> whitespace_split(const JVal& pt, const std::string& text) {
        std::vector<std::string> result;
        size_t i = 0;
        while (i < text.size()) {
            while (i < text.size() && is_space(text[i])) ++i;
            size_t j = i;
            while (j < text.size() && !is_space(text[j])) ++j;
            if (j > i) result.push_back(text.substr(i, j - i));
            i = j;
            while (i < text.size() && is_space(text[i])) {
                size_t k = i;
                while (k < text.size() && is_space(text[k])) ++k;
                result.push_back(text.substr(i, k - i));
                i = k;
            }
        }
        return result;
    }

    inline std::vector<std::string> punctuation(const JVal& pt, const std::string& text) {
        std::vector<std::string> result;
        size_t i = 0;
        while (i < text.size()) {
            if (is_punct(text[i])) {
                result.push_back(std::string(1, text[i]));
                ++i;
            } else {
                size_t j = i;
                while (j < text.size() && !is_punct(text[j])) ++j;
                result.push_back(text.substr(i, j - i));
                i = j;
            }
        }
        return result;
    }

    inline std::vector<std::string> digits(const JVal& pt, const std::string& text) {
        bool individual = pt.has("individual") ? pt.at("individual")->b : false;
        bool combination = pt.has("combination") ? pt.at("combination")->b : false;
        std::vector<std::string> result;
        size_t i = 0;
        while (i < text.size()) {
            if (is_digit(text[i])) {
                size_t j = i;
                while (j < text.size() && is_digit(text[j])) ++j;
                if (individual) {
                    for (size_t k = i; k < j; ++k) result.push_back(std::string(1, text[k]));
                } else if (combination) {
                    result.push_back(text.substr(i, j - i));
                } else {
                    result.push_back(text.substr(i, j - i));
                }
                i = j;
            } else {
                result.push_back(std::string(1, text[i]));
                ++i;
            }
        }
        return result;
    }

    inline std::vector<std::string> metaspace(const JVal& pt, const std::string& text) {
        std::string replacement = pt.has("replacement") ? pt.at("replacement")->s : "A?A¢??";
        bool prepend = pt.has("prepend") ? pt.at("prepend")->b : true;
        bool keep = pt.has("keep") ? pt.at("keep")->b : false;

        std::vector<std::string> result;
        size_t i = 0;
        while (i < text.size()) {
            if (is_space(text[i])) {
                size_t j = i;
                while (j < text.size() && is_space(text[j])) ++j;
                if (keep) result.push_back(text.substr(i, j - i));
                i = j;
            } else {
                size_t j = i;
                while (j < text.size() && !is_space(text[j])) ++j;
                std::string seg = text.substr(i, j - i);
                if (prepend && !result.empty()) seg = replacement + seg;
                result.push_back(seg);
                i = j;
            }
        }
        return result;
    }
}

inline std::vector<std::string> pre_split(const JVal& pre_tokenizer, const std::string& text) {
    if (!pre_tokenizer.has("type")) return {text};
    std::string type = pre_tokenizer.at("type")->s;

    if (type == "Sequence") {
        std::vector<std::string> segments = {text};
        const JVal* children = pre_tokenizer.at("pretokenizers");
        if (children && children->t == JVal::ARR) {
            for (const auto& child : children->arr) {
                std::vector<std::string> new_segments;
                for (const auto& seg : segments) {
                    auto sub = pre_split(child, seg);
                    new_segments.insert(new_segments.end(), sub.begin(), sub.end());
                }
                segments = new_segments;
            }
        }
        return segments;
    } else if (type == "Split") {
        return detail::split_regex(pre_tokenizer, text);
    } else if (type == "ByteLevel") {
        return detail::byte_level(pre_tokenizer, text);
    } else if (type == "Whitespace") {
        return detail::whitespace(pre_tokenizer, text);
    } else if (type == "WhitespaceSplit") {
        return detail::whitespace_split(pre_tokenizer, text);
    } else if (type == "Punctuation") {
        return detail::punctuation(pre_tokenizer, text);
    } else if (type == "Digits") {
        return detail::digits(pre_tokenizer, text);
    } else if (type == "Metaspace") {
        return detail::metaspace(pre_tokenizer, text);
    }
    return {text};
}

} // namespace ptok

#ifdef PTOK_TEST
#include <cassert>
#include <iostream>

static ptok::JVal js(std::string x){ptok::JVal v; v.t=ptok::JVal::STR; v.s=std::move(x); return v;}
static ptok::JVal jo(std::vector<std::pair<std::string,ptok::JVal>> e){ptok::JVal v; v.t=ptok::JVal::OBJ; v.obj=std::move(e); return v;}
static ptok::JVal jb(bool x){ptok::JVal v; v.t=ptok::JVal::BOOL; v.b=x; return v;}
static ptok::JVal ja(std::vector<ptok::JVal> a){ptok::JVal v; v.t=ptok::JVal::ARR; v.arr=std::move(a); return v;}
int main() {
    using namespace ptok;

    // Test 1: Split Isolated
    {
        JVal pt;
        pt.t = JVal::OBJ;
        pt.obj = {
            {"type", js("Split")},
            {"pattern", jo({{"Regex", js(" ?[[:punct:]]|\\w+| ?")}})},
            {"behavior", js("Isolated")}
        };
        auto res = pre_split(pt, "Hello, world!");
        // Expected: ["Hello", ",", " world", "!"]
        assert(res.size() == 4);
        assert(res[0] == "Hello");
        assert(res[1] == ",");
        assert(res[2] == " world");
        assert(res[3] == "!");
    }

    // Test 2: Sequence[Split, Whitespace]
    {
        JVal split_pt;
        split_pt.t = JVal::OBJ;
        split_pt.obj = {
            {"type", js("Split")},
            {"pattern", jo({{"Regex", js("\\w+")}})},
            {"behavior", js("Isolated")}
        };
        JVal ws_pt;
        ws_pt.t = JVal::OBJ;
        ws_pt.obj = {{"type", js("Whitespace")}};

        JVal seq;
        seq.t = JVal::OBJ;
        seq.obj = {
            {"type", js("Sequence")},
            {"pretokenizers", ja({split_pt, ws_pt})}
        };
        auto res = pre_split(seq, "Hello, world!");
        assert(res.size() == 2);
        assert(res[0] == "Hello");
        assert(res[1] == "world");
    }

    // Test 3: Digits individual
    {
        JVal pt;
        pt.t = JVal::OBJ;
        pt.obj = {
            {"type", js("Digits")},
            {"individual", jb(true)}
        };
        auto res = pre_split(pt, "abc12");
        assert(res.size() == 5);
        assert(res[0] == "a");
        assert(res[1] == "b");
        assert(res[2] == "c");
        assert(res[3] == "1");
        assert(res[4] == "2");
    }

    // Test 4: Metaspace with prepend
    {
        JVal pt;
        pt.t = JVal::OBJ;
        pt.obj = {
            {"type", js("Metaspace")},
            {"replacement", js("_")},
            {"prepend", jb(true)}
        };
        auto res = pre_split(pt, "the quick brown");
        assert(res.size() == 3);
        assert(res[0] == "the");
        assert(res[1] == "_quick");
        assert(res[2] == "_brown");
    }

    // Test 5: Absent type passthrough
    {
        JVal pt;
        pt.t = JVal::OBJ;
        pt.obj = {{"type", js("UnknownType")}};
        auto res = pre_split(pt, "Hello world");
        assert(res.size() == 1);
        assert(res[0] == "Hello world");
    }

    std::cout << "All tests passed.\n";
    return 0;
}
#endif
