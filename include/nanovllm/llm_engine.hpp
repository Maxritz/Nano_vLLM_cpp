#pragma once

#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "tokenizer.hpp"
#include "scheduler.hpp"
#include "model_runner.hpp"
#include "sequence.hpp"

struct GenerateOutput {
  std::string text;
  std::vector<int> token_ids;
};

class LLMEngine {
 public:
  explicit LLMEngine(std::string model_path, int max_model_len = 4096);

  void add_request(const std::string& prompt, const SamplingParams& sp);
  void add_request(const std::vector<int>& prompt, const SamplingParams& sp);
  void add_chat_request(const std::string& user_prompt, const SamplingParams& sp);
  bool is_finished() const;
  std::pair<std::vector<std::pair<int, std::vector<int>>>, int> step();
  std::vector<GenerateOutput> generate(const std::vector<std::string>& prompts, const SamplingParams& sp);
  std::vector<GenerateOutput> generate(const std::vector<std::string>& prompts, const std::vector<SamplingParams>& sps);
  std::vector<GenerateOutput> generate_chat(const std::vector<std::string>& prompts, const std::vector<SamplingParams>& sps);
  std::vector<GenerateOutput> generate(const std::vector<std::vector<int>>& prompts, const std::vector<SamplingParams>& sps);
  int vocab_size() const { return config_.hf.vocab_size; }
  std::vector<int> encode_text(const std::string& text) const;

 private:
  Config config_;
  std::unique_ptr<ModelRunner> runner_;
  std::unique_ptr<Scheduler> scheduler_;
  Tokenizer tokenizer_;
};
