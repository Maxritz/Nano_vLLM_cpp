#pragma once

#include <cstdint>
#include <vector>

#include "common.hpp"

struct Context {
  bool is_prefill = false;
  std::vector<int64_t> input_ids;
  std::vector<int64_t> positions;
  std::vector<int32_t> cu_seqlens_q;
  std::vector<int32_t> cu_seqlens_k;
  std::vector<int32_t> slot_mapping;
  std::vector<int32_t> context_lens;
  std::vector<int32_t> block_tables;
  std::vector<int32_t> query_seq;
  std::vector<int32_t> query_key_len;
  std::vector<int32_t> last_indices;
  int32_t max_seqlen_q = 0;
  int32_t max_seqlen_k = 0;
  int32_t max_blocks = 0;
  int32_t num_tokens = 0;
  int32_t num_seqs = 0;

  DevVec<int64_t> d_input_ids;
  DevVec<int64_t> d_positions;
  DevVec<int32_t> d_cu_seqlens_q;
  DevVec<int32_t> d_cu_seqlens_k;
  DevVec<int32_t> d_slot_mapping;
  DevVec<int32_t> d_context_lens;
  DevVec<int32_t> d_block_tables;
  DevVec<int32_t> d_query_seq;
  DevVec<int32_t> d_query_key_len;
  DevVec<int32_t> d_last_indices;

  void clear() {
    is_prefill = false;
    input_ids.clear();
    positions.clear();
    cu_seqlens_q.clear();
    cu_seqlens_k.clear();
    slot_mapping.clear();
    context_lens.clear();
    block_tables.clear();
    query_seq.clear();
    query_key_len.clear();
    last_indices.clear();
    max_seqlen_q = max_seqlen_k = max_blocks = num_tokens = num_seqs = 0;
    d_input_ids.free();
    d_positions.free();
    d_cu_seqlens_q.free();
    d_cu_seqlens_k.free();
    d_slot_mapping.free();
    d_context_lens.free();
    d_block_tables.free();
    d_query_seq.free();
    d_query_key_len.free();
    d_last_indices.free();
  }

  void upload() {
    d_input_ids.assign(input_ids);
    d_positions.assign(positions);
    d_cu_seqlens_q.assign(cu_seqlens_q);
    d_cu_seqlens_k.assign(cu_seqlens_k);
    d_slot_mapping.assign(slot_mapping);
    d_context_lens.assign(context_lens);
    d_block_tables.assign(block_tables);
    d_query_seq.assign(query_seq);
    d_query_key_len.assign(query_key_len);
    d_last_indices.assign(last_indices);
  }
};
