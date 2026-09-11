#pragma once

#include <memory>
#include <vector>

#include "config.hpp"
#include "context.hpp"
#include "model.hpp"
#include "sampler.hpp"
#include "sequence.hpp"

class ModelRunner {
 public:
  explicit ModelRunner(Config config);

  void initialize();
  int allocate_kv_cache();
  std::vector<int> run(std::vector<Sequence*> seqs, bool is_prefill);

  const Config& config() const { return config_; }
  int num_kv_cache_blocks() const { return num_blocks_; }

 private:
  void prepare_block_tables(std::vector<Sequence*>& seqs, Context& ctx);
  void prepare_prefill(std::vector<Sequence*>& seqs, Context& ctx);
  void prepare_decode(std::vector<Sequence*>& seqs, Context& ctx);

  Config config_;
  Qwen3Model model_;
  Sampler sampler_;
  int num_blocks_ = 0;
  bool initialized_ = false;
};
