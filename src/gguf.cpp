#include "nanovllm/gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "nanovllm/common.hpp"

namespace {

enum GGUFType {
  T_UINT8 = 0, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY,
  T_UINT64, T_INT64, T_FLOAT64,
};

enum GGMLType {
  M_F32 = 0, M_F16 = 1, M_Q4_0 = 2, M_Q4_1 = 3, M_Q5_0 = 6, M_Q5_1 = 7, M_Q8_0 = 8, M_Q8_1 = 9,
  M_Q2_K = 10, M_Q3_K = 11, M_Q4_K = 12, M_Q5_K = 13, M_Q6_K = 14, M_Q8_K = 15,
  M_BF16 = 30,
};

struct Reader {
  std::ifstream f;
  bool ok = true;
  explicit Reader(const std::string& path) : f(path, std::ios::binary) { if (!f) ok = false; }
  template <class T>
  bool read(T& v) { if (!f.read(reinterpret_cast<char*>(&v), sizeof(T))) { ok = false; return false; } return true; }
  bool read_bytes(void* dst, size_t n) { if (!f.read(reinterpret_cast<char*>(dst), n)) { ok = false; return false; } return true; }
  std::string read_string() {
    uint64_t len = 0;
    if (!read(len)) return {};
    std::string s(len, '\0');
    if (!read_bytes(&s[0], len)) return {};
    return s;
  }
};

uint32_t ggml_block_size(uint32_t t) {
  if (t == M_F32 || t == M_F16 || t == M_BF16) return 1;
  if (t == M_Q4_0 || t == M_Q4_1 || t == M_Q5_0 || t == M_Q5_1 || t == M_Q8_0) return 32;
  return 0;
}

size_t ggml_type_bytes(uint32_t t) {
  if (t == M_F32) return 4;
  if (t == M_F16 || t == M_BF16) return 2;
  if (t == M_Q8_0) return 34;
  if (t == M_Q4_0) return 18;
  if (t == M_Q4_1) return 20;
  if (t == M_Q5_0) return 22;
  if (t == M_Q5_1) return 24;
  return 0;
}

std::string ggml_type_name(uint32_t t) {
  switch (t) {
    case M_F32: return "F32";
    case M_F16: return "F16";
    case M_BF16: return "BF16";
    case M_Q8_0: return "Q8_0";
    case M_Q4_0: return "Q4_0";
    case M_Q4_1: return "Q4_1";
    case M_Q5_0: return "Q5_0";
    case M_Q5_1: return "Q5_1";
    case M_Q2_K: return "Q2_K";
    case M_Q3_K: return "Q3_K";
    case M_Q4_K: return "Q4_K";
    case M_Q5_K: return "Q5_K";
    case M_Q6_K: return "Q6_K";
    case M_Q8_K: return "Q8_K";
    default: return "UNKNOWN" + std::to_string(t);
  }
}

// Reads a single GGUF KV value of the given type into `v`.
void read_value(Reader& r, uint32_t type, GGUFMetaValue& v) {
  v.type = type;
  switch (type) {
    case T_UINT8: case T_INT8: case T_BOOL: {
      uint8_t b; if (r.read(b)) { v.b = b != 0; v.i = b; }
      break;
    }
    case T_UINT16: case T_INT16: {
      uint16_t b; if (r.read(b)) { v.i = b; v.d = b; }
      break;
    }
    case T_UINT32: case T_INT32: {
      uint32_t b; if (r.read(b)) { v.i = b; v.d = b; }
      break;
    }
    case T_UINT64: case T_INT64: {
      uint64_t b; if (r.read(b)) { v.i = static_cast<int64_t>(b); v.d = static_cast<double>(v.i); }
      break;
    }
    case T_FLOAT32: { float b; if (r.read(b)) v.d = b; break; }
    case T_FLOAT64: { double b; if (r.read(b)) v.d = b; break; }
    case T_STRING: { v.s = r.read_string(); break; }
    case T_ARRAY: {
      uint32_t elem_type = 0;
      uint64_t count = 0;
      r.read(elem_type);
      r.read(count);
      if (elem_type == T_STRING) {
        v.sarr.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count && r.ok; ++i) v.sarr.push_back(r.read_string());
      } else if (elem_type == T_FLOAT32) {
        v.farr.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count && r.ok; ++i) { float b; if (r.read(b)) v.farr.push_back(b); }
      } else if (elem_type == T_INT32) {
        v.iarr.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count && r.ok; ++i) { int32_t b; if (r.read(b)) v.iarr.push_back(b); }
      } else if (elem_type == T_UINT32) {
        v.iarr.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count && r.ok; ++i) { uint32_t b; if (r.read(b)) v.iarr.push_back(b); }
      } else if (elem_type == T_INT64 || elem_type == T_UINT64) {
        v.iarr.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count && r.ok; ++i) { int64_t b; if (r.read(b)) v.iarr.push_back(b); }
      } else if (elem_type == T_BOOL) {
        v.iarr.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count && r.ok; ++i) { uint8_t b; if (r.read(b)) v.iarr.push_back(b); }
      } else if (elem_type == T_FLOAT64) {
        for (uint64_t i = 0; i < count && r.ok; ++i) { double b; r.read(b); }
        v.iarr.resize(static_cast<size_t>(count), 0);  // not exposed; kept for sizing
      } else {
        // Unhandled array element type: skip raw bytes.
        r.ok = false;
      }
      break;
    }
    default: r.ok = false;
  }
}

}  // namespace

void GGUFLoader::add_file(const std::string& path) {
  Reader r(path);
  if (!r.ok) throw std::runtime_error("cannot open gguf file: " + path);
  uint32_t magic = 0, version = 0;
  uint64_t n_tensors = 0, n_kv = 0;
  if (!r.read(magic) || magic != 0x46554747u /* "GGUF" */)
    throw std::runtime_error("not a gguf file: " + path);
  r.read(version);
  r.read(n_tensors);
  r.read(n_kv);
  for (uint64_t i = 0; i < n_kv && r.ok; ++i) {
    std::string key = r.read_string();
    uint32_t type = 0;
    r.read(type);
    GGUFMetaValue v;
    read_value(r, type, v);
    if (r.ok) meta_[std::move(key)] = std::move(v);
  }
  std::vector<GGUFTensorMeta> metas;
  metas.reserve(n_tensors);
  for (uint64_t i = 0; i < n_tensors && r.ok; ++i) {
    GGUFTensorMeta m;
    m.name = r.read_string();
    uint32_t n_dims = 0;
    r.read(n_dims);
    m.shape.resize(n_dims);
    for (uint32_t j = 0; j < n_dims; ++j) r.read(m.shape[j]);
    uint32_t type = 0;
    r.read(type);
    m.dtype = ggml_type_name(type);
    r.read(m.offset);
    metas.push_back(std::move(m));
  }
  if (!r.ok) throw std::runtime_error("corrupt gguf header: " + path);
  uint64_t pos = r.f.tellg();
  auto it = meta_.find("general.alignment");
  if (it != meta_.end() && it->second.type == T_UINT32) alignment_ = static_cast<uint32_t>(it->second.i);
  // Data section starts at the file position aligned up to `alignment`.
  data_offset_ = (pos + alignment_ - 1) / alignment_ * alignment_;
  path_ = path;
  for (auto& m : metas) tensors_[m.name] = std::move(m);
}

const GGUFTensorMeta* GGUFLoader::tensor(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

std::vector<std::string> GGUFLoader::names() const {
  std::vector<std::string> out;
  out.reserve(tensors_.size());
  for (auto& kv : tensors_) out.push_back(kv.first);
  return out;
}

const GGUFMetaValue* GGUFLoader::meta(const std::string& key) const {
  auto it = meta_.find(key);
  return it == meta_.end() ? nullptr : &it->second;
}

bool GGUFLoader::meta_u32(const std::string& key, uint32_t& out) const {
  const GGUFMetaValue* v = meta(key);
  if (!v || (v->type != T_UINT32 && v->type != T_INT32 && v->type != T_UINT16 && v->type != T_UINT64)) return false;
  out = static_cast<uint32_t>(v->i);
  return true;
}

bool GGUFLoader::meta_i64(const std::string& key, int64_t& out) const {
  const GGUFMetaValue* v = meta(key);
  if (!v || (v->type != T_INT64 && v->type != T_INT32)) return false;
  out = v->i;
  return true;
}

bool GGUFLoader::meta_f64(const std::string& key, double& out) const {
  const GGUFMetaValue* v = meta(key);
  if (!v || (v->type != T_FLOAT32 && v->type != T_FLOAT64)) return false;
  out = v->d;
  return true;
}

bool GGUFLoader::meta_str(const std::string& key, std::string& out) const {
  const GGUFMetaValue* v = meta(key);
  if (!v || v->type != T_STRING) return false;
  out = v->s;
  return true;
}

bool GGUFLoader::meta_str_array(const std::string& key, std::vector<std::string>& out) const {
  const GGUFMetaValue* v = meta(key);
  if (!v || v->type != T_ARRAY || v->sarr.empty()) return false;
  out = v->sarr;
  return true;
}

bool GGUFLoader::meta_i32_array(const std::string& key, std::vector<int32_t>& out) const {
  const GGUFMetaValue* v = meta(key);
  if (!v || v->type != T_ARRAY || v->iarr.empty()) return false;
  out.assign(v->iarr.begin(), v->iarr.end());
  return true;
}

bool GGUFLoader::meta_f32_array(const std::string& key, std::vector<float>& out) const {
  const GGUFMetaValue* v = meta(key);
  if (!v || v->type != T_ARRAY || v->farr.empty()) return false;
  out = v->farr;
  return true;
}

void GGUFLoader::read_tensor_data(const GGUFTensorMeta& meta, void* out, size_t bytes) const {
  std::ifstream f(path_, std::ios::binary);
  if (!f) throw std::runtime_error("cannot reopen gguf file: " + path_);
  f.seekg(static_cast<std::streamoff>(data_offset_ + meta.offset));
  f.read(reinterpret_cast<char*>(out), bytes);
  if (!f) throw std::runtime_error("truncated tensor data for " + meta.name);
}

bool GGUFLoader::load_u16(const std::string& name, std::vector<uint16_t>& out, bool& bf16) const {
  const GGUFTensorMeta* m = tensor(name);
  if (!m) return false;
  if (m->dtype != "F16" && m->dtype != "BF16") return false;
  size_t n = 1;
  for (int64_t d : m->shape) n *= static_cast<size_t>(d);
  out.resize(n);
  read_tensor_data(*m, out.data(), n * 2);
  bf16 = (m->dtype == "BF16");
  return true;
}

bool GGUFLoader::load_float(const std::string& name, std::vector<float>& out) const {
  const GGUFTensorMeta* m = tensor(name);
  if (!m) return false;
  size_t n = 1;
  for (int64_t d : m->shape) n *= static_cast<size_t>(d);
  if (m->dtype == "F32") {
    out.resize(n);
    read_tensor_data(*m, out.data(), n * 4);
    return true;
  }
  if (m->dtype == "F16" || m->dtype == "BF16") {
    std::vector<uint16_t> raw(n);
    read_tensor_data(*m, raw.data(), n * 2);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = m->dtype == "BF16" ? bf16_to_float(raw[i]) : fp16_to_float(raw[i]);
    return true;
  }
  return false;
}

bool GGUFLoader::load_q8_0(const std::string& name, std::vector<uint8_t>& out) const {
  const GGUFTensorMeta* m = tensor(name);
  if (!m || m->dtype != "Q8_0") return false;
  size_t n = 1;
  for (int64_t d : m->shape) n *= static_cast<size_t>(d);
  size_t nblocks = (n + 31) / 32;
  out.resize(nblocks * 34);
  read_tensor_data(*m, out.data(), out.size());
  return true;
}

size_t GGUFLoader::qk_block_bytes(int ggml_type) {
  switch (ggml_type) {
    case 12: return 144;  // Q4_K
    case 13: return 176;  // Q5_K
    case 14: return 210;  // Q6_K
    default: return 0;
  }
}

bool GGUFLoader::load_qk(const std::string& name, int ggml_type, size_t n_elements,
                          std::vector<uint8_t>& out) const {
  size_t bb = qk_block_bytes(ggml_type);
  if (!bb || n_elements % QK_K != 0) return false;
  const GGUFTensorMeta* m = tensor(name);
  if (!m) return false;
  size_t n = 1;
  for (int64_t d : m->shape) n *= static_cast<size_t>(d);
  if (n != n_elements) return false;
  // Verify the on-disk dtype matches the requested kind (fail loud, never misread).
  if (m->dtype != ggml_type_name((uint32_t)ggml_type)) return false;
  out.resize(n_elements / QK_K * bb);
  read_tensor_data(*m, out.data(), out.size());
  return true;
}