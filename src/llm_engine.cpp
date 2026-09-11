#include "nanovllm/llm_engine.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static std::string special_start() { return std::string("<") + '|' + "im_start|>"; }
static std::string special_end() { return std::string("<") + '|' + "im_end|>"; }

LLMEngine::LLMEngine(std::string model_path, int max_model_len) {
  config_.model = std::move(model_path);
  config_.max_model_len = max_model_len;
  config_.load_hf_config();
  tokenizer_.set_fallback_vocab_size(config_.hf.vocab_size);
  tokenizer_.load(config_.model);

  config_.eos = tokenizer_.eos_token_id();
  Sequence::block_size = config_.kvcache_block_size;
  runner_ = std::make_unique<ModelRunner>(config_);
  runner_->initialize();
  config_ = runner_->config();
  scheduler_ = std::make_unique<Scheduler>(config_);
}

void LLMEngine::add_request(const std::string& prompt, const SamplingParams& sp) {
  sp.validate();
  auto ids = tokenizer_.encode_text(prompt);
  if (ids.empty()) throw std::invalid_argument("empty prompt");
  if (static_cast<int>(ids.size()) + sp.max_tokens > config_.max_model_len)
    throw std::invalid_argument("prompt + max_tokens exceeds max_model_len");
  scheduler_->add(std::make_shared<Sequence>(std::move(ids), sp));
}

void LLMEngine::add_chat_request(const std::string& user_prompt, const SamplingParams& sp) {
  sp.validate();
  int im_start = tokenizer_.encode_special(special_start());
  int im_end = tokenizer_.encode_special(special_end());
  std::vector<int> ids;
  auto append = [&](const std::string& s) {
    auto part = tokenizer_.encode_text(s);
    ids.insert(ids.end(), part.begin(), part.end());
  };
if (im_start >= 0 && im_end >= 0) {
    ids.push_back(im_start);
    append("user\n" + user_prompt);
    ids.push_back(im_end);
    append("\n");
    ids.push_back(im_start);
    append("assistant\n");
  } else {
    append(user_prompt);
  }
  if (static_cast<int>(ids.size()) + sp.max_tokens > config_.max_model_len)
    throw std::invalid_argument("prompt + max_tokens exceeds max_model_len");
  scheduler_->add(std::make_shared<Sequence>(std::move(ids), sp));
}

void LLMEngine::add_request(const std::vector<int>& prompt, const SamplingParams& sp) {
  sp.validate();
  if (prompt.empty()) throw std::invalid_argument("empty prompt token ids");
  if (static_cast<int>(prompt.size()) + sp.max_tokens > config_.max_model_len)
    throw std::invalid_argument("prompt + max_tokens exceeds max_model_len");
  scheduler_->add(std::make_shared<Sequence>(prompt, sp));
}

bool LLMEngine::is_finished() const { return scheduler_->is_finished(); }

std::tuple<std::vector<std::pair<int, std::vector<int>>>, int, std::vector<std::pair<int, int>>> LLMEngine::step() {
  auto [seq_ptrs, is_prefill] = scheduler_->schedule();
  std::vector<Sequence*> seqs;
  if (seq_ptrs.empty()) return {{}, 0, {}};
  seqs.reserve(seq_ptrs.size());
  for (auto& p : seq_ptrs) seqs.push_back(p.get());
  int num_tokens = is_prefill ? 0 : -static_cast<int>(seqs.size());
  if (is_prefill) for (Sequence* s : seqs) num_tokens += s->num_scheduled_tokens;
  auto token_ids = runner_->run(seqs, is_prefill);
  scheduler_->postprocess(seqs, token_ids, is_prefill);
  std::vector<std::pair<int, std::vector<int>>> outputs;
  for (auto& p : seq_ptrs) {
    if (p->is_finished()) outputs.emplace_back(p->seq_id, p->completion_token_ids());
  }
  std::vector<std::pair<int, int>> new_tokens;
  if (!is_prefill) {
    for (size_t i = 0; i < seqs.size() && i < token_ids.size(); ++i)
      new_tokens.push_back({seqs[i]->seq_id, token_ids[i]});
  }
  return {outputs, num_tokens, new_tokens};
}

std::vector<GenerateOutput> LLMEngine::generate(const std::vector<std::string>& prompts, const SamplingParams& sp) {
  std::vector<SamplingParams> sps(prompts.size(), sp);
  return generate(prompts, sps);
}

std::vector<GenerateOutput> LLMEngine::generate(const std::vector<std::string>& prompts,
                                                const std::vector<SamplingParams>& sps) {
  if (sps.size() != prompts.size()) throw std::invalid_argument("prompts and sampling params size mismatch");
  for (size_t i = 0; i < prompts.size(); ++i) add_request(prompts[i], sps[i]);
  std::map<int, std::vector<int>> completed;
  while (!is_finished()) {
    auto [outs, _, news] = step();
    for (auto& o : outs) completed[o.first] = o.second;
  }
  std::vector<GenerateOutput> out;
  out.reserve(completed.size());
  for (auto& kv : completed) out.push_back({tokenizer_.decode_tokens(kv.second), kv.second});
  return out;
}

std::vector<GenerateOutput> LLMEngine::generate(const std::vector<std::vector<int>>& prompts,
                                                const std::vector<SamplingParams>& sps) {
  if (sps.size() != prompts.size()) throw std::invalid_argument("prompts and sampling params size mismatch");
  for (size_t i = 0; i < prompts.size(); ++i) add_request(prompts[i], sps[i]);
  std::map<int, std::vector<int>> completed;
  while (!is_finished()) {
    auto [outs, _, news] = step();
    for (auto& o : outs) completed[o.first] = o.second;
  }
  std::vector<GenerateOutput> out;
  out.reserve(completed.size());
  for (auto& kv : completed) out.push_back({tokenizer_.decode_tokens(kv.second), kv.second});
  return out;
}

std::vector<GenerateOutput> LLMEngine::generate_chat(const std::vector<std::string>& prompts,
                                                     const std::vector<SamplingParams>& sps) {
  if (sps.size() != prompts.size()) throw std::invalid_argument("prompts and sampling params size mismatch");
  for (size_t i = 0; i < prompts.size(); ++i) add_chat_request(prompts[i], sps[i]);
  std::map<int, std::vector<int>> completed;
  while (!is_finished()) {
    auto [outs, _, news] = step();
    for (auto& o : outs) completed[o.first] = o.second;
  }
  std::vector<GenerateOutput> out;
  out.reserve(completed.size());
  for (auto& kv : completed) out.push_back({tokenizer_.decode_tokens(kv.second), kv.second});
  return out;
}

std::vector<int> LLMEngine::encode_text(const std::string& text) const {
  return tokenizer_.encode_text(text);
}

std::string LLMEngine::decode_token(int id) const {
  return tokenizer_.decode_tokens({id});
}
