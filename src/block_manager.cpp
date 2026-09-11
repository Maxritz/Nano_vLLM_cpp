#include "nanovllm/block_manager.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

static uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static bool same_tokens(const std::vector<int>& a, const std::vector<int>& b) {
  return a == b;
}

BlockManager::BlockManager(int num_blocks, int block_size) : block_size_(block_size) {
  blocks_.reserve(num_blocks);
  for (int i = 0; i < num_blocks; ++i) {
    Block b;
    b.block_id = i;
    blocks_.push_back(b);
    free_block_ids_.push_back(i);
  }
}

uint64_t BlockManager::compute_hash(const std::vector<int>& token_ids, int64_t prefix) {
  uint64_t h = (prefix == -1) ? 0x243f6a8885a308d3ULL : static_cast<uint64_t>(prefix);
  h ^= 0x1c6e4a8dc4b3f8a1ULL + token_ids.size() * 0x9e3779b97f4a7c15ULL;
  for (int t : token_ids) {
    h = splitmix64(h ^ static_cast<uint64_t>(static_cast<uint32_t>(t)));
  }
  return h;
}

int BlockManager::_allocate_block() {
  if (free_block_ids_.empty()) throw std::runtime_error("no free blocks");
  int id = free_block_ids_.front();
  free_block_ids_.pop_front();
  Block& b = blocks_[id];
  if (b.hash != 0 || b.has_hash) {
    auto it = hash_to_block_id_.find(b.hash);
    if (it != hash_to_block_id_.end() && it->second == id) hash_to_block_id_.erase(it);
  }
  b.reset();
  used_block_ids_.insert(id);
  return id;
}

void BlockManager::_deallocate_block(int block_id) {
  used_block_ids_.erase(block_id);
  free_block_ids_.push_back(block_id);
}

int BlockManager::can_allocate(const Sequence& seq) const {
  int64_t h = -1;
  int num_cached_blocks = 0;
  int num_new_blocks = seq.num_blocks();
  for (int i = 0; i + 1 < seq.num_blocks(); ++i) {
    std::vector<int> toks = seq.block(i);
    uint64_t uh = compute_hash(toks, h);
    h = static_cast<int64_t>(uh);
    auto it = hash_to_block_id_.find(uh);
    if (it == hash_to_block_id_.end()) break;
    int block_id = it->second;
    if (!same_tokens(blocks_[block_id].token_ids, toks)) break;
    ++num_cached_blocks;
    if (used_block_ids_.find(block_id) != used_block_ids_.end()) --num_new_blocks;
  }
  if (static_cast<int>(free_block_ids_.size()) < num_new_blocks) return -1;
  return num_cached_blocks;
}

void BlockManager::allocate(Sequence& seq, int num_cached_blocks) {
  if (!seq.block_table.empty()) throw std::runtime_error("allocate called with existing block table");
  int64_t h = -1;
  for (int i = 0; i < num_cached_blocks; ++i) {
    std::vector<int> toks = seq.block(i);
    uint64_t uh = compute_hash(toks, h);
    h = static_cast<int64_t>(uh);
    int block_id = hash_to_block_id_.at(uh);
    Block& block = blocks_[block_id];
    if (used_block_ids_.find(block_id) != used_block_ids_.end()) {
      block.ref_count += 1;
    } else {
      block.ref_count = 1;
      auto it = std::find(free_block_ids_.begin(), free_block_ids_.end(), block_id);
      if (it != free_block_ids_.end()) free_block_ids_.erase(it);
      used_block_ids_.insert(block_id);
    }
    seq.block_table.push_back(block_id);
  }
  for (int i = num_cached_blocks; i < seq.num_blocks(); ++i) {
    seq.block_table.push_back(_allocate_block());
  }
  seq.num_cached_tokens = num_cached_blocks * block_size_;
}

void BlockManager::deallocate(Sequence& seq) {
  for (auto it = seq.block_table.rbegin(); it != seq.block_table.rend(); ++it) {
    Block& block = blocks_[*it];
    block.ref_count -= 1;
    if (block.ref_count == 0) _deallocate_block(*it);
  }
  seq.num_cached_tokens = 0;
  seq.block_table.clear();
}

bool BlockManager::can_append(const Sequence& seq) const {
  bool need = (seq.num_tokens % block_size_) == 1;
  return static_cast<int>(free_block_ids_.size()) >= (need ? 1 : 0);
}

void BlockManager::may_append(Sequence& seq) {
  if ((seq.num_tokens % block_size_) == 1) {
    seq.block_table.push_back(_allocate_block());
  }
}

void BlockManager::hash_blocks(Sequence& seq) {
  int start = seq.num_cached_tokens / block_size_;
  int end = (seq.num_cached_tokens + seq.num_scheduled_tokens) / block_size_;
  if (start == end) return;
  int64_t h = (start > 0) ? static_cast<int64_t>(blocks_[seq.block_table[start - 1]].hash) : -1;
  for (int i = start; i < end; ++i) {
    Block& block = blocks_[seq.block_table[i]];
    std::vector<int> toks = seq.block(i);
    uint64_t uh = compute_hash(toks, h);
    h = static_cast<int64_t>(uh);
    block.update(uh, toks);
    hash_to_block_id_[uh] = block.block_id;
  }
}
