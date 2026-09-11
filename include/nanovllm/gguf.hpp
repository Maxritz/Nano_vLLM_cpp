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

 private:
  std::string path_;
  uint64_t data_offset_ = 0;
  uint32_t alignment_ = 32;
  std::map<std::string, GGUFTensorMeta> tensors_;
  std::map<std::string, GGUFMetaValue> meta_;

  void read_tensor_data(const GGUFTensorMeta& meta, void* out, size_t bytes) const;
};