#pragma once

#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "block_manager.hpp"
#include "config.hpp"
#include "sequence.hpp"

class Scheduler {
 public:
  explicit Scheduler(const Config& config);

  bool is_finished() const { return waiting_.empty() && running_.empty(); }
  void add(std::shared_ptr<Sequence> seq);
  std::pair<std::vector<std::shared_ptr<Sequence>>, bool> schedule();
  void postprocess(std::vector<Sequence*>& seqs, const std::vector<int>& token_ids, bool is_prefill);

  BlockManager& block_manager() { return block_manager_; }

 private:
  void preempt(std::shared_ptr<Sequence> seq);

  int max_num_seqs_;
  int max_num_batched_tokens_;
  int eos_;
  int block_size_;
  BlockManager block_manager_;
  std::deque<std::shared_ptr<Sequence>> waiting_;
  std::deque<std::shared_ptr<Sequence>> running_;
};
