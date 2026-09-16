#pragma once

#include <vector>
#include <cstdint>
#include <cmath>

namespace rope {

// YaRN rope scaling. Given base_freq, pos, and scaling params, return the
// rotated frequency for a given dimension pair. Pure arithmetic, no deps.
//   N = head_dim, dim = the pair index (0..N/2-1), pos = position.
//   YaRN: lambda = 1 / (1 + 0.1*log2(pos/beta)*scaling)  (mrope-style)
//   returns the scaled angle = freq_base^(-2*dim/N) * lambda * pos
inline double yarn_angle(int head_dim, int dim, int64_t pos,
                         double theta_base = 500000.0, double scaling = 1.0,
                         double beta = 32.0, double freq_base = 10000.0) {
    if (pos == 0) return 0.0;
    double exponent = -2.0 * static_cast<double>(dim) / static_cast<double>(head_dim);
    double freq = std::pow(freq_base, exponent);
    double lambda = 1.0 / (1.0 + 0.1 * std::log2(static_cast<double>(pos) / beta) * scaling);
    return freq * lambda * static_cast<double>(pos);
}

// StreamingLLM eviction: given the current KV length and a sink+window
// budget, return the set of positions to KEEP (sink prefix + recent window).
// Positions are 0-based. Always keeps [0, sink) and the last `window` tokens.
inline std::vector<int64_t> streaming_keep(int64_t total_len, int64_t sink, int64_t window) {
    std::vector<int64_t> keep;
    if (total_len <= 0) return keep;
    for (int64_t i = 0; i < sink && i < total_len; ++i) keep.push_back(i);
    int64_t start = total_len - window;
    if (start < sink) start = sink;  // avoid overlap with sink
    for (int64_t i = start; i < total_len; ++i) keep.push_back(i);
    return keep;
}

}  // namespace rope

#ifdef ROPE_TEST
#include <cassert>
#include <iostream>

int main() {
    assert(rope::yarn_angle(8, 0, 0) == 0.0);
    auto result = rope::streaming_keep(10, 2, 3);
    std::vector<int64_t> expected = {0, 1, 7, 8, 9};
    assert(result == expected);
    std::cout << "All rope tests passed.\n";
    return 0;
}
#endif