#include "nanovllm/scheduler.hpp"

#include <algorithm>
#include <cassert>
#include <stdexcept>

Scheduler::Scheduler(const Config& config)
    : max_num_seqs_(config.max_num_seqs),
      max_num_batched_tokens_(config.max_num_batched_tokens),
      eos_(config.eos),
      block_size_(config.kvcache_block_size),
      block_manager_(config.num_kvcache_blocks, config.kvcache_block_size) {}

void Scheduler::add(std::shared_ptr<Sequence> seq) { waiting_.push_back(std::move(seq)); }

void Scheduler::preempt(std::shared_ptr<Sequence> seq) {
  seq->status = SequenceStatus::WAITING;
  seq->is_prefill = true;
  block_manager_.deallocate(*seq);
  waiting_.push_front(std::move(seq));
}

std::pair<std::vector<std::shared_ptr<Sequence>>, bool> Scheduler::schedule() {
  std::vector<std::shared_ptr<Sequence>> scheduled;
  int num_batched_tokens = 0;

  while (!waiting_.empty() && static_cast<int>(scheduled.size()) < max_num_seqs_) {
    auto seq = waiting_.front();
    int remaining = max_num_batched_tokens_ - num_batched_tokens;
    if (remaining == 0) break;
    int num_cached_blocks = 0;
    int num_tokens = 0;
    if (seq->block_table.empty()) {
      num_cached_blocks = block_manager_.can_allocate(*seq);
      if (num_cached_blocks == -1) break;
      num_tokens = seq->num_tokens - num_cached_blocks * block_size_;
    } else {
      num_tokens = seq->num_tokens - seq->num_cached_tokens;
    }
    if (remaining < num_tokens && !scheduled.empty()) break;
    if (seq->block_table.empty()) block_manager_.allocate(*seq, num_cached_blocks);
    int sched = std::min(num_tokens, remaining);
    seq->num_scheduled_tokens = sched;
    num_batched_tokens += sched;
    if (seq->num_cached_tokens + seq->num_scheduled_tokens == seq->num_tokens) {
      seq->status = SequenceStatus::RUNNING;
      auto ptr = std::move(waiting_.front());
      waiting_.pop_front();
      scheduled.push_back(ptr);
      running_.push_back(std::move(ptr));
    } else {
      scheduled.push_back(seq);
    }
  }
  if (!scheduled.empty()) return {scheduled, true};

  while (!running_.empty() && static_cast<int>(scheduled.size()) < max_num_seqs_) {
    auto seq = std::move(running_.front());
    running_.pop_front();
    bool ok = true;
    while (!block_manager_.can_append(*seq)) {
      if (!running_.empty()) {
        auto victim = std::move(running_.back());
        running_.pop_back();
        preempt(std::move(victim));
      } else {
        preempt(std::move(seq));
        ok = false;
        break;
      }
    }
    if (!ok) break;
    seq->num_scheduled_tokens = 1;
    seq->is_prefill = false;
    block_manager_.may_append(*seq);
    scheduled.push_back(seq);
  }
  if (scheduled.empty()) {
    if (!waiting_.empty()) {
      auto retry = schedule();
      if (!retry.first.empty()) return retry;
    }
    throw std::runtime_error("scheduler could not make progress");
  }
  for (auto it = scheduled.rbegin(); it != scheduled.rend(); ++it) {
    running_.push_front(*it);
  }
  return {scheduled, false};
}

void Scheduler::postprocess(std::vector<Sequence*>& seqs, const std::vector<int>& token_ids, bool is_prefill) {
  if (token_ids.size() != seqs.size()) throw std::runtime_error("token_ids/seqs size mismatch in postprocess");
  for (size_t i = 0; i < seqs.size(); ++i) {
    Sequence& seq = *seqs[i];
    int token_id = token_ids[i];
    block_manager_.hash_blocks(seq);
    seq.num_cached_tokens += seq.num_scheduled_tokens;
    seq.num_scheduled_tokens = 0;
    if (is_prefill && seq.num_cached_tokens < seq.num_tokens) continue;
    seq.append_token(token_id);
    bool finish = (!seq.ignore_eos && token_id == eos_) || (seq.num_completion_tokens() == seq.max_tokens);
    if (finish) {
      seq.status = SequenceStatus::FINISHED;
      block_manager_.deallocate(seq);
      auto it = std::find_if(running_.begin(), running_.end(),
                             [&](const std::shared_ptr<Sequence>& p) { return p.get() == &seq; });
      if (it != running_.end()) running_.erase(it);
    }
  }
}
