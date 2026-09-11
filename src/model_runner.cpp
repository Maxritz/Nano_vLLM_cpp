#include "nanovllm/model_runner.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

ModelRunner::ModelRunner(Config config) : config_(std::move(config)), model_(config_) {}

void ModelRunner::initialize() {
  if (initialized_) return;
  model_.load_weights();
  num_blocks_ = model_.allocate_kv_cache();
  config_.num_kvcache_blocks = num_blocks_;
  initialized_ = true;
}

int ModelRunner::allocate_kv_cache() {
  num_blocks_ = model_.allocate_kv_cache();
  config_.num_kvcache_blocks = num_blocks_;
  return num_blocks_;
}

static void check_block(int num_blocks, int block_id) {
  if (num_blocks <= 0) throw std::runtime_error("KV cache has not been allocated");
  if (block_id < 0 || block_id >= num_blocks) throw std::runtime_error("KV cache block id out of range");
}

void ModelRunner::prepare_block_tables(std::vector<Sequence*>& seqs, Context& ctx) {
  int max_blocks = 0;
  for (Sequence* seq : seqs) max_blocks = std::max(max_blocks, static_cast<int>(seq->block_table.size()));
  ctx.max_blocks = std::max(max_blocks, 1);
  ctx.block_tables.assign(seqs.size() * ctx.max_blocks, -1);
  for (size_t s = 0; s < seqs.size(); ++s) {
    for (size_t i = 0; i < seqs[s]->block_table.size(); ++i) {
      int block = seqs[s]->block_table[i];
      check_block(num_blocks_, block);
      ctx.block_tables[s * ctx.max_blocks + i] = block;
    }
  }
}

void ModelRunner::prepare_prefill(std::vector<Sequence*>& seqs, Context& ctx) {
  ctx.is_prefill = true;
  ctx.cu_seqlens_q = {0};
  ctx.cu_seqlens_k = {0};
  ctx.num_seqs = static_cast<int>(seqs.size());
  int bs = config_.kvcache_block_size;
  int max_slot = num_blocks_ * bs;
  for (int s = 0; s < ctx.num_seqs; ++s) {
    Sequence* seq = seqs[s];
    int start = seq->num_cached_tokens;
    int seqlen_q = seq->num_scheduled_tokens;
    int end = start + seqlen_q;
    int seqlen_k = end;
    if (seqlen_q <= 0) throw std::runtime_error("prefill cannot schedule zero tokens");
    if (start < 0 || end > seq->num_tokens) throw std::runtime_error("prefill token slice out of range");

    for (int i = 0; i < seqlen_q; ++i) {
      ctx.input_ids.push_back(seq->token_ids[start + i]);
      ctx.positions.push_back(start + i);
      ctx.query_seq.push_back(s);
      ctx.query_key_len.push_back(start + i + 1);
    }

    if (seq->block_table.empty()) {
      throw std::runtime_error("prefill sequence has no KV block table");
    }
    int need_blocks = (end + bs - 1) / bs;
    if (static_cast<int>(seq->block_table.size()) < need_blocks)
      throw std::runtime_error("prefill block table is too small");
    int start_block = start / bs;
    int end_block = need_blocks;
    for (int bi = start_block; bi < end_block; ++bi) {
      int block = seq->block_table[bi];
      check_block(num_blocks_, block);
      int slot_start = block * bs;
      if (bi == start_block) slot_start += start % bs;
      int slot_end;
      if (bi != end_block - 1) {
        slot_end = block * bs + bs;
      } else {
        slot_end = block * bs + end - bi * bs;
      }
      for (int slot = slot_start; slot < slot_end; ++slot) {
        if (slot < 0 || slot >= max_slot) throw std::runtime_error("KV slot mapping out of range");
        ctx.slot_mapping.push_back(slot);
      }
    }

    ctx.max_seqlen_q = std::max(ctx.max_seqlen_q, seqlen_q);
    ctx.max_seqlen_k = std::max(ctx.max_seqlen_k, seqlen_k);
    ctx.cu_seqlens_q.push_back(static_cast<int>(ctx.cu_seqlens_q.back() + seqlen_q));
    ctx.cu_seqlens_k.push_back(static_cast<int>(ctx.cu_seqlens_k.back() + seqlen_k));
  }
  for (size_t i = 1; i < ctx.cu_seqlens_q.size(); ++i) ctx.last_indices.push_back(ctx.cu_seqlens_q[i] - 1);
  ctx.num_tokens = static_cast<int>(ctx.input_ids.size());
  prepare_block_tables(seqs, ctx);
}

void ModelRunner::prepare_decode(std::vector<Sequence*>& seqs, Context& ctx) {
  ctx.is_prefill = false;
  ctx.num_seqs = static_cast<int>(seqs.size());
  int bs = config_.kvcache_block_size;
  int max_slot = num_blocks_ * bs;
  for (int s = 0; s < ctx.num_seqs; ++s) {
    Sequence* seq = seqs[s];
    int len = seq->num_tokens;
    if (len <= 0) throw std::runtime_error("decode sequence is empty");
    if (seq->block_table.empty()) throw std::runtime_error("decode sequence has no KV block table");
    int need_blocks = (len + bs - 1) / bs;
    if (static_cast<int>(seq->block_table.size()) < need_blocks)
      throw std::runtime_error("decode block table is too small");
    int block = seq->block_table.back();
    check_block(num_blocks_, block);
    int slot = block * bs + seq->last_block_num_tokens() - 1;
    if (slot < 0 || slot >= max_slot) throw std::runtime_error("KV slot mapping out of range");
    ctx.input_ids.push_back(seq->last_token);
    ctx.positions.push_back(len - 1);
    ctx.context_lens.push_back(len);
    ctx.query_seq.push_back(s);
    ctx.query_key_len.push_back(len);
    ctx.slot_mapping.push_back(slot);
    ctx.last_indices.push_back(s);
  }
  ctx.num_tokens = static_cast<int>(ctx.input_ids.size());
  prepare_block_tables(seqs, ctx);
}

std::vector<int> ModelRunner::run(std::vector<Sequence*> seqs, bool is_prefill) {
  if (seqs.empty()) return {};
  Context ctx;
  if (is_prefill) prepare_prefill(seqs, ctx);
  else prepare_decode(seqs, ctx);
  if (ctx.num_tokens == 0) throw std::runtime_error("prepared batch has no tokens");
  if (ctx.max_blocks == 0) throw std::runtime_error("context has no block tables");
  auto logits = model_.forward_logits(ctx);
  int vocab = config_.hf.vocab_size;
  if (vocab <= 0) throw std::runtime_error("invalid vocab size");
  if (logits.size() < static_cast<size_t>(ctx.num_seqs) * vocab)
    throw std::runtime_error("model returned fewer logits rows than scheduled sequences");
  std::vector<int> out(ctx.num_seqs);
  for (int i = 0; i < ctx.num_seqs; ++i) {
    out[i] = sampler_.sample(logits.data() + size_t(i) * vocab, vocab, seqs[i]->temperature);
    if (getenv("NANO_DEBUG")) {
      std::fprintf(stderr, "sample: is_prefill=%d row=%d input=%lld pos=%lld key_len=%d id=%d\n", (int)ctx.is_prefill,
                   i, (long long)ctx.input_ids[i], (long long)ctx.positions[i], ctx.query_key_len[i], out[i]);
      std::fflush(stderr);
    }
  }
  return out;
}
