#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "config.hpp"

enum class SequenceStatus { WAITING, RUNNING, FINISHED };

struct Sequence {
  static int block_size;
  static int counter;

  int seq_id = 0;
  SequenceStatus status = SequenceStatus::WAITING;
  std::vector<int> token_ids;
  int last_token = 0;
  int num_tokens = 0;
  int num_prompt_tokens = 0;
  int num_cached_tokens = 0;
  int num_scheduled_tokens = 0;
  bool is_prefill = true;
  std::vector<int> block_table;
  double temperature = 1.0;
  int max_tokens = 64;
  bool ignore_eos = false;

  Sequence() = default;
  Sequence(const std::vector<int>& ids, const SamplingParams& sp)
      : seq_id(counter++),
        token_ids(ids),
        last_token(ids.empty() ? 0 : ids.back()),
        num_tokens(static_cast<int>(ids.size())),
        num_prompt_tokens(static_cast<int>(ids.size())),
        temperature(sp.temperature),
        max_tokens(sp.max_tokens),
        ignore_eos(sp.ignore_eos) {
    if (ids.empty()) throw std::invalid_argument("empty prompt token ids");
  }

  bool is_finished() const { return status == SequenceStatus::FINISHED; }
  int num_completion_tokens() const { return num_tokens - num_prompt_tokens; }
  int num_blocks() const { return (num_tokens + block_size - 1) / block_size; }
  int last_block_num_tokens() const { return num_tokens - (num_blocks() - 1) * block_size; }

  std::vector<int> block(int i) const {
    int start = i * block_size;
    int end = std::min(start + block_size, num_tokens);
    return std::vector<int>(token_ids.begin() + start, token_ids.begin() + end);
  }

  void append_token(int t) {
    token_ids.push_back(t);
    last_token = t;
    ++num_tokens;
  }

  std::vector<int> prompt_token_ids() const {
    return std::vector<int>(token_ids.begin(), token_ids.begin() + num_prompt_tokens);
  }
  std::vector<int> completion_token_ids() const {
    return std::vector<int>(token_ids.begin() + num_prompt_tokens, token_ids.end());
  }
};

inline int Sequence::block_size = 256;
inline int Sequence::counter = 0;
