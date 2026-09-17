#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

class Sampler {
 public:
  int sample(const float* logits, int vocab, double temperature) {
    if (vocab <= 0) return 0;
    if (temperature <= 0) {
      int best = 0;
      for (int i = 1; i < vocab; ++i)
        if (logits[i] > logits[best]) best = i;
      return best;
    }
    double max_logit = logits[0];
    for (int i = 1; i < vocab; ++i) max_logit = std::max(max_logit, static_cast<double>(logits[i]));
    std::vector<double> probs(vocab);
    double sum = 0.0;
    double inv_t = 1.0 / temperature;
    for (int i = 0; i < vocab; ++i) {
      probs[i] = std::exp((static_cast<double>(logits[i]) - max_logit) * inv_t);
      sum += probs[i];
    }
    double u = dist_(rng_) * sum;
    double cum = 0.0;
    for (int i = 0; i < vocab; ++i) {
      cum += probs[i];
      if (cum > u) return i;
    }
    return vocab - 1;
  }

  private:
   std::mt19937 rng_{std::random_device{}()};
   std::uniform_real_distribution<double> dist_{0.0, 1.0};
};

// CPU sampler shared by the Vulkan CLI paths (SAMP-1): greedy at temp<=0,
// else temperature + nucleus sampling. recent_ids/recent_len carry
// generated-token history; ids seen in the last rep_window entries have
// their scores divided by rep_penalty (applied pre-nucleus, i.e. penalized
// ids can drop out of top-p). rep_penalty == 1.0 skips this entirely.
inline int sample_row(const float* logits, int vocab, double temp, double top_p, std::mt19937& rng,
                      const int* recent_ids = nullptr, size_t recent_len = 0, double rep_penalty = 1.0,
                      int rep_window = 64) {
  if (temp <= 0) return (int)(std::max_element(logits, logits + vocab) - logits);
  std::vector<int> idx((size_t)vocab);
  for (int i = 0; i < vocab; ++i) idx[(size_t)i] = i;
  std::vector<float> sc((size_t)vocab);
  float mx = -1e30f;
  for (int i = 0; i < vocab; ++i) mx = std::max(mx, logits[i]);
  double sum = 0;
  for (int i = 0; i < vocab; ++i) { sc[(size_t)i] = std::exp((logits[i] - mx) / temp); sum += sc[(size_t)i]; }
  if (rep_penalty != 1.0 && rep_penalty > 0.0 && recent_ids != nullptr && recent_len > 0) {
    size_t w = rep_window > 0 ? std::min<size_t>((size_t)rep_window, recent_len) : recent_len;
    std::vector<int> seen;
    seen.reserve(w);
    bool changed = false;
    for (size_t k = recent_len - w; k < recent_len; ++k) {
      int id = recent_ids[k];
      if (id < 0 || id >= vocab) continue;
      if (std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
      seen.push_back(id);
      sc[(size_t)id] /= (float)rep_penalty;
      changed = true;
    }
    if (changed) {
      sum = 0;
      for (int i = 0; i < vocab; ++i) sum += sc[(size_t)i];
    }
  }
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return sc[(size_t)a] > sc[(size_t)b]; });
  double acc = 0;
  size_t keep = idx.size();
  for (size_t i = 0; i < idx.size(); ++i) {
    acc += sc[(size_t)idx[i]] / sum;
    if (acc >= top_p && i > 0) { keep = i + 1; break; }
  }
  double r = std::uniform_real_distribution<double>(0, 1)(rng) * acc, run = 0;
  for (size_t i = 0; i < keep; ++i) {
    run += sc[(size_t)idx[i]] / sum;
    if (r <= run) return idx[i];
  }
  return idx[keep - 1];
}

#ifdef SAMP_TEST
#include <cassert>
#include <cstdio>

// Spec: SAMP-1 windowed repetition penalty. Equality asserts are portable
// (paired runs consume the RNG identically, so equal scores => equal draws);
// the golden pair pins the penalty's distributional effect on this stdlib.
inline int samp_test_main() {
const float lg[4] = {10.0f, 9.0f, 0.0f, -5.0f};
// 1. Greedy = argmax, penalty ignored even when the top id is recent.
{ std::mt19937 r(42); int rec[2] = {0, 0};
  assert(sample_row(lg, 4, 0.0, 1.0, r, rec, 2, 1e20, 64) == 0); }
// 2. rep_penalty=1.0: skip path is stable (same seed => same draw) and in range.
{ std::mt19937 a(42), b(42); int rec[1] = {0};
  int t1 = sample_row(lg, 4, 1.0, 1.0, a, rec, 1, 1.0, 64);
  int t2 = sample_row(lg, 4, 1.0, 1.0, b, rec, 1, 1.0, 64);
  assert(t1 == t2 && t1 >= 0 && t1 < 4); }
// 3. Window slicing: recent=[0,0,1] window=1 penalizes exactly {1}, same as recent=[1].
{ std::mt19937 a(42), b(42); int r1[3] = {0, 0, 1}; int r2[1] = {1};
  assert(sample_row(lg, 4, 1.0, 1.0, a, r1, 3, 2.0, 1) ==
         sample_row(lg, 4, 1.0, 1.0, b, r2, 1, 2.0, 64)); }
// 4. rep_window=0 means whole history: identical scores to window=64 here.
{ std::mt19937 a(42), b(42); int rec[2] = {0, 1};
  assert(sample_row(lg, 4, 1.0, 1.0, a, rec, 2, 2.0, 0) ==
         sample_row(lg, 4, 1.0, 1.0, b, rec, 2, 2.0, 64)); }
// 5. Out-of-range recent ids are skipped: identical to the no-penalty run.
{ std::mt19937 a(42), b(42); int rec[2] = {-1, 99999};
  assert(sample_row(lg, 4, 1.0, 1.0, a, rec, 2, 2.0, 64) ==
         sample_row(lg, 4, 1.0, 1.0, b, rec, 1, 1.0, 64)); }
// 6. nullptr/empty history with penalty set: plain sampling, draw in range.
{ std::mt19937 r(42);
  int t = sample_row(lg, 4, 1.0, 1.0, r, nullptr, 0, 2.0, 64);
  assert(t >= 0 && t < 4); }
// 7. Golden pair (MSVC mt19937, seed 7): penalty moves mass off id 0.
{ std::mt19937 a(7), b(7); int rec[1] = {0};
  int plain = sample_row(lg, 4, 1.0, 1.0, a, rec, 1, 1.0, 64);
  int penal = sample_row(lg, 4, 1.0, 1.0, b, rec, 1, 5.0, 64);
  std::printf("[samp-golden] plain=%d penalized=%d\n", plain, penal);
  assert(plain == 0 && penal == 1); }
std::printf("All samp tests passed.\n");
return 0;
}
#endif  // SAMP_TEST
