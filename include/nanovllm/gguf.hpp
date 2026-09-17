#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

// If `model` is a .gguf file, returns it; if it's a directory, returns the first .gguf
// inside (empty if none). nano-vllm accepts either form.
static inline std::string find_gguf_path(const std::string& model) {
  std::filesystem::path p(model);
  std::error_code ec;
  if (std::filesystem::is_directory(p, ec)) {
    for (const auto& ent : std::filesystem::directory_iterator(p, ec)) {
      if (ent.path().extension() == ".gguf") return ent.path().string();
    }
    return {};
  }
  if (p.extension() == ".gguf") return model;
  return {};
}

struct GGUFTensorMeta {
  std::string name;
  std::string dtype;      // "F32", "F16", "BF16", "Q8_0", ...
  std::vector<int64_t> shape;
  uint64_t offset = 0;    // from start of data section
};

// One fused K-quant segment: raw super-blocks of a single kind. Fused qkv holds
// three (q/k/v), fused gate_up two, single matrices one. Resolved by elem offset.
struct QKSeg {
  int kind = 0;            // ggml type id (2=Q4_0, 10=Q2_K, 12=Q4_K, 13=Q5_K, 14=Q6_K)
  size_t byte_off = 0;     // byte offset of this segment in the fused stream
  size_t elem_off = 0;     // element offset of this segment
};

// One GGUF metadata KV value. Only the field matching the value's type is meaningful
// (e.g. string arrays land in sarr, int arrays in iarr, floats in d/farr).
struct GGUFMetaValue {
  uint32_t type = 0;
  bool b = false;
  int64_t i = 0;
  double d = 0;
  std::string s;
  std::vector<std::string> sarr;
  std::vector<int64_t> iarr;
  std::vector<float> farr;
};

class GGUFLoader {
 public:
  void add_file(const std::string& path);
  bool contains(const std::string& name) const { return tensors_.find(name) != tensors_.end(); }
  const GGUFTensorMeta* tensor(const std::string& name) const;
  std::vector<std::string> names() const;
  const GGUFMetaValue* meta(const std::string& key) const;
  // Typed metadata accessors; return true only if the key exists with a matching type.
  bool meta_u32(const std::string& key, uint32_t& out) const;
  bool meta_i64(const std::string& key, int64_t& out) const;
  bool meta_f64(const std::string& key, double& out) const;
  bool meta_str(const std::string& key, std::string& out) const;
  bool meta_str_array(const std::string& key, std::vector<std::string>& out) const;
  bool meta_i32_array(const std::string& key, std::vector<int32_t>& out) const;
  bool meta_f32_array(const std::string& key, std::vector<float>& out) const;
  bool load_u16(const std::string& name, std::vector<uint16_t>& out, bool& bf16) const;
  bool load_float(const std::string& name, std::vector<float>& out) const;
  // Q8_0: block = [fp16 d][32 x int8], 34 bytes per 32 elements.
  bool load_q8_0(const std::string& name, std::vector<uint8_t>& out) const;
  // K-quants: raw super-blocks verbatim (QK_K=256 elements each). No conversion,
  // zero bloat: kernels dequantize inline. Returns false for unknown dtypes.
  // Supported ggml type ids: 12 (Q4_K, 144B/blk), 14 (Q6_K, 210B/blk).
  static size_t qk_block_bytes(int ggml_type);
  static constexpr int QK_K = 256;
  bool load_qk(const std::string& name, int ggml_type, size_t n_elements,
               std::vector<uint8_t>& out) const;
  // Host upcast for quant types with no GPU dequant kernel (Q5_0, IQ4_NL, Q3_K
  // as found in *_Q2_K files; plus Q4_0, Q4_1, IQ4_XS, Q2_K and the K-quants
  // Q4_K/Q6_K for the GGUF-MoE place path): dequantize on CPU. False otherwise.
  bool load_upcast_f16(const std::string& name, std::vector<uint16_t>& out) const;
  bool load_upcast_f32(const std::string& name, std::vector<float>& out) const;
  // Raw-block helpers (same table; also used by the GGUF-MoE place path and
  // test probes). upcast_* return 0 for unsupported dtypes.
  static size_t upcast_block_bytes(const std::string& dtype);
  static size_t upcast_block_vals(const std::string& dtype);
  static bool dequant_block_f32(const std::string& dtype, const uint8_t* blk, float* y);
  static bool dequant_blocks_f16(const std::string& dtype, const uint8_t* raw,
                                 size_t n_elements, uint16_t* y);

 private:
  std::string path_;
  uint64_t data_offset_ = 0;
  uint32_t alignment_ = 32;
  std::map<std::string, GGUFTensorMeta> tensors_;
  std::map<std::string, GGUFMetaValue> meta_;

  void read_tensor_data(const GGUFTensorMeta& meta, void* out, size_t bytes) const;
};