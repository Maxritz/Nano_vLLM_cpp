#pragma once

#include <cstdint>
#include <vector>

#include "common.hpp"

void hip_matmul(const float* x, const float* w, int m, int n, int k, float* out);
void hip_matmul_u16(const float* x, const uint16_t* w, int m, int n, int k, float* out, bool w_bf16);
void hip_matmul_q8(const float* x, const uint8_t* w8, const uint16_t* ws, int m, int n, int k, float* out);
void hip_matmul_qk(const float* x, const uint8_t* wq, int m, int n, int k, float* out, int kind);
void hip_add_bias_inplace(float* x, const float* bias, int rows, int cols);

void hip_rms_norm(const float* x, const float* weight, float* out, int rows, int hidden, float eps);
void hip_rms_norm_add(const float* x, float* residual, const float* weight, float* out, int rows, int hidden, float eps);

void hip_silu_and_mul(const float* gate_up, float* out, int rows, int intermediate);

void hip_rope_inplace(float* data, const int64_t* positions, int tokens, int heads, int head_dim, int64_t stride,
                      const float* inv_freq);

void hip_store_kv(const float* key, const float* value, float* k_cache, float* v_cache, const int32_t* slot_mapping,
                  int tokens, int kv_heads, int head_dim);

void hip_paged_attention(const float* q, float* out, const float* k_cache, const float* v_cache,
                         const int32_t* query_seq, const int32_t* query_key_len, const int32_t* block_tables,
                         int tokens, int q_heads, int kv_heads, int head_dim, int block_size, int max_blocks,
                         float scale);

void hip_embedding(const int64_t* ids, const float* table, int tokens, int hidden, float* out);
void hip_embedding_u16(const int64_t* ids, const uint16_t* table, int tokens, int hidden, float* out, bool table_bf16);
void hip_embedding_q8(const int64_t* ids, const uint8_t* w8, const uint16_t* ws, int tokens, int hidden, float* out);
void hip_embedding_qk(const int64_t* ids, const uint8_t* wq, int tokens, int hidden, float* out, int kind);
void hip_gather_rows(const float* rows, const int32_t* indices, int rows_count, int cols, int out_rows, float* out);
void hip_copy_to_host(const float* dev, std::vector<float>& out, size_t n);
void hip_copy_from_host(const std::vector<float>& host, float* dev);
void hip_moe_route(const float* logits, int rows, int E, int K, bool norm, int32_t* idx, float* score);
void hip_moe_ffn(const float* x, const int32_t* idx, const float* score, const uint16_t* gu,
                 const uint16_t* dn, const int32_t* slot_of, int rows, int H, int I, int K,
                 bool bf16, float* out);
void hip_add_inplace(float* dst, const float* src, size_t n);
void hip_scale_rows_sigmoid(float* dst, const float* s, int rows, int cols);
