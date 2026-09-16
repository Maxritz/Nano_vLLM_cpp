#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace rag {

struct Doc {
    std::string id;
    std::string text;
};

class Index {
public:
    void add(Doc d) {
        docs_.push_back(std::move(d));
    }

    void build() {
        // Count document frequencies for each term.
        std::map<std::string, int> df;
        for (const auto& doc : docs_) {
            std::vector<std::string> seen;
            std::map<std::string, bool> seen_set;
            for (const auto& t : toks(doc.text)) {
                if (!seen_set.count(t)) {
                    seen_set[t] = true;
                    seen.push_back(t);
                }
            }
            for (const auto& t : seen) {
                df[t]++;
            }
        }

        // Compute average document length.
        double total_len = 0.0;
        for (const auto& doc : docs_) {
            total_len += static_cast<double>(toks(doc.text).size());
        }
        avgdl_ = docs_.empty() ? 0.0 : total_len / static_cast<double>(docs_.size());

        // Compute IDF using BM25 standard formula.
        idf_.clear();
        int N = static_cast<int>(docs_.size());
        for (const auto& [term, freq] : df) {
            double idf_val = std::log(1.0 + (static_cast<double>(N) - static_cast<double>(freq) + 0.5) /
                                           (static_cast<double>(freq) + 0.5));
            idf_[term] = idf_val;
        }

        // Pre-tokenize documents for scoring.
        doc_tokens_.clear();
        for (const auto& doc : docs_) {
            doc_tokens_.push_back(toks(doc.text));
        }
    }

    struct Hit {
        std::string id;
        double score;
    };

    std::vector<Hit> search(const std::string& query, int top_k) const {
        if (top_k <= 0 || docs_.empty() || query.empty()) {
            return {};
        }

        std::vector<std::string> q_tokens = toks(query);
        if (q_tokens.empty()) {
            return {};
        }

        std::vector<std::pair<double, size_t>> scores;
        for (size_t i = 0; i < docs_.size(); ++i) {
            double score = 0.0;
            const auto& tokens = doc_tokens_[i];
            std::map<std::string, int> term_freq;
            for (const auto& t : tokens) {
                term_freq[t]++;
            }

            double doc_len = static_cast<double>(tokens.size());
            for (const auto& qt : q_tokens) {
                auto it = term_freq.find(qt);
                if (it != term_freq.end()) {
                    double tf = static_cast<double>(it->second);
                    auto ii = idf_.find(qt);
                    double idf_val = (ii == idf_.end()) ? 0.0 : ii->second;
                    double denom = tf + k1_ * (1.0 - b_ + b_ * doc_len / avgdl_);
                    if (denom > 0.0) {
                        score += idf_val * ((tf * (k1_ + 1.0)) / denom);
                    }
                }
            }
            scores.emplace_back(score, i);
        }

        // Sort by score descending, then by index ascending for stability.
        std::sort(scores.begin(), scores.end(),
                  [](const std::pair<double, size_t>& a, const std::pair<double, size_t>& b) {
                      if (a.first != b.first) {
                          return a.first > b.first;
                      }
                      return a.second < b.second;
                  });

        std::vector<Hit> results;
        size_t count = std::min(static_cast<size_t>(top_k), scores.size());
        for (size_t i = 0; i < count; ++i) {
            results.push_back({docs_[scores[i].second].id, scores[i].first});
        }
        return results;
    }

    std::string stuff(const std::string& prompt, const std::string& query,
                      int top_k, size_t max_chars) const {
        std::vector<Hit> hits = search(query, top_k);
        if (hits.empty()) {
            return prompt;
        }

        std::string result = prompt;
        size_t current_len = result.size();

        // Process hits in reverse order so we can truncate oldest first.
        for (auto it = hits.rbegin(); it != hits.rend(); ++it) {
            const Hit& hit = *it;
            auto doc_it = std::find_if(docs_.begin(), docs_.end(),
                                       [&](const Doc& d) { return d.id == hit.id; });
            if (doc_it == docs_.end()) {
                continue;
            }

            std::string chunk = "\n\n[doc:" + hit.id + "]\n" + doc_it->text;
            if (current_len + chunk.size() <= max_chars) {
                result = chunk + result;
                current_len += chunk.size();
            } else {
                // Truncate this chunk to fit.
                size_t available = max_chars - current_len;
                if (available > 0) {
                    // We need to preserve the header part if possible.
                    std::string header = "\n\n[doc:" + hit.id + "]\n";
                    if (header.size() >= available) {
                        // Only fit part of the header.
                        result = header.substr(0, available) + result;
                        current_len = max_chars;
                    } else {
                        result = header + doc_it->text.substr(0, available - header.size()) + result;
                        current_len = max_chars;
                    }
                }
                break;
            }
        }

        return result;
    }

private:
    static std::vector<std::string> toks(const std::string& s) {
        std::vector<std::string> tokens;
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            if (c >= 0x80) {
                // UTF-8 multibyte sequence: keep the entire sequence as one token.
                std::string token;
                token += s[i];
                ++i;
                while (i < s.size()) {
                    unsigned char next = static_cast<unsigned char>(s[i]);
                    if ((next & 0xC0) == 0x80) {
                        token += s[i];
                        ++i;
                    } else {
                        break;
                    }
                }
                tokens.push_back(token);
            } else if (std::isalnum(c)) {
                // ASCII alphanumeric run.
                std::string token;
                while (i < s.size()) {
                    unsigned char ch = static_cast<unsigned char>(s[i]);
                    if (ch < 0x80 && std::isalnum(ch)) {
                        token += static_cast<char>(std::tolower(ch));
                        ++i;
                    } else {
                        break;
                    }
                }
                tokens.push_back(token);
            } else {
                // Skip non-alphanumeric ASCII characters.
                ++i;
            }
        }
        return tokens;
    }

    std::vector<Doc> docs_;
    std::vector<std::vector<std::string>> doc_tokens_;
    std::map<std::string, double> idf_;
    double avgdl_ = 0.0;
    static constexpr double k1_ = 1.2;
    static constexpr double b_ = 0.75;
};

} // namespace rag

#ifdef RAG_TEST
#include <cassert>
#include <iostream>

int main() {
    // Test 1: Exact-match doc ranks first.
    {
        rag::Index idx;
        idx.add({"doc1", "the quick brown fox"});
        idx.add({"doc2", "lazy dog runs slowly"});
        idx.add({"doc3", "the quick blue hare"});
        idx.build();

        auto hits = idx.search("quick brown fox", 3);
        assert(!hits.empty());
        assert(hits[0].id == "doc1");
    }

    // Test 2: max_chars cap respected.
    {
        rag::Index idx;
        idx.add({"doc1", "a b c d e f g h i j k l m n o p q r s t u v w x y z"});
        idx.build();

        std::string prompt = "PROMPT";
        std::string result = idx.stuff(prompt, "a b c", 1, 20);
        assert(result.size() <= 20);
    }

    // Test 3: UTF-8 text round-trips byte-identical through stuff().
    {
        rag::Index idx;
        std::string utf8_text = "cafÃ© rÃ©sumÃ© naÃ¯ve";
        idx.add({"doc1", utf8_text});
        idx.build();

        std::string prompt = "PROMPT";
        std::string result = idx.stuff(prompt, "cafÃ©", 1, 1000);
        // Check that the UTF-8 text appears byte-identical in the result.
        assert(result.find(utf8_text) != std::string::npos);
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
#endif
