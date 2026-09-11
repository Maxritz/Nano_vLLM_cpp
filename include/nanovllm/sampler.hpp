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
