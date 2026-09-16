#pragma once

#include <string>
#include <vector>
#include <sstream>

namespace reason {

struct TagPair {
    std::string open;
    std::string close;
};

inline std::vector<TagPair>& _pairs() {
    static std::vector<TagPair> p = {
        {"", ""},
        {"[thinking]", "[/thinking]"}
    };
    return p;
}

inline void add_pair(TagPair p) {
    _pairs().push_back(p);
}

inline const std::vector<TagPair>& pairs() {
    return _pairs();
}

inline void reset_pairs() {
    _pairs() = {
        {"", ""},
        {"[thinking]", "[/thinking]"}
    };
}

// Find the earliest open tag in `text` (scanning all registered pairs),
// then the first close after that open, and return the text strictly between
// them, trimmed of surrounding whitespace. No nesting. Empty text or no pair
// -> "".
inline std::string extract_thinking(const std::string& text) {
    if (text.empty()) return "";
    size_t best_open = std::string::npos, best_close = std::string::npos;
    size_t best_open_len = 0, best_close_len = 0;
    for (const auto& tp : _pairs()) {
        if (tp.open.empty() && tp.close.empty()) continue;
        size_t o = text.find(tp.open);
        if (o == std::string::npos) continue;
        size_t c = text.find(tp.close, o + tp.open.size());
        if (c == std::string::npos) continue;
        if (best_open == std::string::npos || o < best_open) {
            best_open = o; best_close = c;
            best_open_len = tp.open.size(); best_close_len = tp.close.size();
        }
    }
    if (best_open == std::string::npos) return "";
    std::string inner = text.substr(best_open + best_open_len,
                                    best_close - (best_open + best_open_len));
    size_t a = 0, b = inner.size();
    while (a < b && std::isspace(static_cast<unsigned char>(inner[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(inner[b - 1]))) --b;
    return inner.substr(a, b - a);
}

// Strip ALL thinking blocks (open..close) from `text`, non-greedy per block,
// repeating until none remain. Tags themselves are removed.
inline std::string strip_thinking(const std::string& text) {
    if (text.empty()) return "";
    std::string result = text;
    bool changed = true;
    while (changed) {
        changed = false;
        size_t best_open = std::string::npos, best_close = std::string::npos;
        size_t best_open_len = 0, best_close_len = 0;
        for (const auto& tp : _pairs()) {
            if (tp.open.empty() && tp.close.empty()) continue;
            size_t o = result.find(tp.open);
            if (o == std::string::npos) continue;
            size_t c = result.find(tp.close, o + tp.open.size());
            if (c == std::string::npos) continue;
            if (best_open == std::string::npos || o < best_open) {
                best_open = o; best_close = c;
                best_open_len = tp.open.size(); best_close_len = tp.close.size();
            }
        }
        if (best_open != std::string::npos) {
            result.erase(best_open, (best_close + best_close_len) - best_open);
            changed = true;
        }
    }
    return result;
}

// Soft budget: keep at most `max_tokens` whitespace-separated tokens.
inline std::string budget_thinking(const std::string& thinking, int max_tokens) {
    if (thinking.empty() || max_tokens <= 0) return "";
    std::istringstream iss(thinking);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token && static_cast<int>(tokens.size()) < max_tokens) {
        tokens.push_back(token);
    }
    std::ostringstream oss;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) oss << " ";
        oss << tokens[i];
    }
    return oss.str();
}

} // namespace reason

#ifdef REAS_TEST
#include <cassert>
#include <iostream>

int main() {
    {
        std::string input = "pre [thinking] inner [/thinking] post";
        std::string result = reason::extract_thinking(input);
        assert(result == "inner");
    }
    {
        std::string input = "pre [thinking] inner [/thinking] post";
        std::string result = reason::strip_thinking(input);
        assert(result == "pre  post");
    }
    {
        std::string result = reason::budget_thinking("a b c d", 2);
        std::istringstream iss(result);
        int count = 0; std::string tok;
        while (iss >> tok) ++count;
        assert(count <= 2);
    }
    assert(reason::extract_thinking("") == "");
    assert(reason::strip_thinking("") == "");
    assert(reason::budget_thinking("", 5) == "");
    std::cout << "All REAS_TEST assertions passed.\n";
    return 0;
}
#endif