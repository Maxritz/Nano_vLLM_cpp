// include/nanovllm/drafter.hpp
#pragma once

#include <vector>
#include <map>

namespace drafter {

// Model-free prompt-lookup + generated-text n-gram drafting (TokenSwift-style).
// No GPU, no model weights — pure data structure over token ids.
struct NgramIndex {
    // Build the lookup table from a token id sequence. For every n-gram (n>=2)
    // ending at position i, record what token followed it. Multiple occurrences
    // of the same n-gram keep the FIRST successor (stable, deterministic).
    void build(const std::vector<int>& tokens, int n) {
        if (n < 2) {
            n_ = 0;
            succ_.clear();
            return;
        }
        n_ = n;
        succ_.clear();
        if (static_cast<int>(tokens.size()) < n) return;
        for (size_t i = 0; i + n <= tokens.size(); ++i) {
            std::vector<int> gram(tokens.begin() + i, tokens.begin() + i + n - 1);
            int next = tokens[i + n - 1];
            if (succ_.find(gram) == succ_.end()) succ_[gram] = next;
        }
    }

    struct Match {
        bool found = false;
        int token = 0;
        std::vector<int> draft;
    };

    // Given the last (n-1) generated tokens, find the longest n-gram (starting
    // at n and shrinking) that has a recorded successor, and return it plus
    // the successor token. Returns {false, 0, {}} when nothing matches.
    Match suggest(const std::vector<int>& recent_tail) const {
        Match m;
        if (n_ < 2 || recent_tail.empty()) return m;
        for (int len = n_; len >= 2; --len) {
            if (static_cast<int>(recent_tail.size()) < len - 1) continue;
            std::vector<int> key(recent_tail.end() - (len - 1), recent_tail.end());
            auto it = succ_.find(key);
            if (it != succ_.end()) {
                m.found = true;
                m.token = it->second;
                m.draft = key;
                m.draft.push_back(m.token);
                return m;
            }
        }
        return m;
    }

    // Append `draft` tokens to `out` if they match the continuation predicted
    // by the index built on `corpus`. Stops at the first mismatch. Returns the
    // number of tokens appended.
    int extend(std::vector<int>& out, const std::vector<int>& draft,
               const std::vector<int>& corpus) const {
        int appended = 0;
        for (size_t i = 0; i < draft.size(); ++i) {
            if (i >= corpus.size()) break;
            if (draft[i] != corpus[i]) break;
            out.push_back(draft[i]);
            ++appended;
        }
        return appended;
    }

private:
    int n_ = 0;
    std::map<std::vector<int>, int> succ_;  // n-gram -> successor token
};

}  // namespace drafter

#ifdef DRAFT_TEST
#include <cassert>
#include <cstdio>

int main() {
    // (1) build+suggest on "1 2 3 1 2 4" with n=3: suggest({1,2}) -> token 3
    {
        drafter::NgramIndex idx;
        std::vector<int> corpus = {1, 2, 3, 1, 2, 4};
        idx.build(corpus, 3);
        auto m = idx.suggest({1, 2});
        assert(m.found);
        assert(m.token == 3);
        assert(m.draft.size() == 3);
        assert(m.draft[0] == 1 && m.draft[1] == 2 && m.draft[2] == 3);
    }
    // (2) extend() appends matching continuation and stops at mismatch
    {
        drafter::NgramIndex idx;
        std::vector<int> corpus = {1, 2, 3, 1, 2, 4};
        idx.build(corpus, 3);
        std::vector<int> out;
        std::vector<int> draft = {1, 2, 3, 9, 9};
        int n = idx.extend(out, draft, corpus);
        assert(n == 3);
        assert(out.size() == 3);
        assert(out[0] == 1 && out[1] == 2 && out[2] == 3);
    }
    // (3) n=1 build is a no-op (suggest always returns found=false)
    {
        drafter::NgramIndex idx;
        std::vector<int> corpus = {1, 2, 3, 1, 2, 4};
        idx.build(corpus, 1);
        auto m = idx.suggest({1, 2});
        assert(!m.found);
        assert(m.token == 0);
        assert(m.draft.empty());
    }
    std::printf("All drafter tests passed.\n");
    return 0;
}
#endif  // DRAFT_TEST