#pragma once

#include <cstdint>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "sequence.hpp"

struct Block {
  int block_id = -1;
  int ref_count = 0;
  uint64_t hash = 0;
  bool has_hash = false;
  std::vector<int> token_ids;

  void update(uint64_t h, const std::vector<int>& ids) {
    hash = h;
    has_hash = true;
    token_ids = ids;
  }
  void reset() {
    ref_count = 1;
    has_hash = false;
    hash = 0;
    token_ids.clear();
  }
};

class BlockManager {
 public:
  BlockManager(int num_blocks, int block_size);

  static uint64_t compute_hash(const std::vector<int>& token_ids, int64_t prefix = -1);

  int can_allocate(const Sequence& seq) const;
  void allocate(Sequence& seq, int num_cached_blocks);
  void deallocate(Sequence& seq);
  bool can_append(const Sequence& seq) const;
  void may_append(Sequence& seq);
  void hash_blocks(Sequence& seq);

  int free_count() const { return static_cast<int>(free_block_ids_.size()); }

 private:
  int _allocate_block();
  void _deallocate_block(int block_id);

  int block_size_;
  std::vector<Block> blocks_;
  std::unordered_map<uint64_t, int> hash_to_block_id_;
  std::deque<int> free_block_ids_;
  std::unordered_set<int> used_block_ids_;
};
