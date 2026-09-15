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
  M_BF16 = 30, M_IQ4_NL = 20, M_IQ4_XS = 23,
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
    case M_IQ4_NL: return "IQ4_NL";
    case M_IQ4_XS: return "IQ4_XS";
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

namespace {

// Q5_0: [fp16 d][4B qh][32B qs(high nibbles in 2nd half)], 32 values.
void dequant_q5_0_block(const uint8_t* blk, float* y) {
  float d = fp16_to_float(uint16_t(blk[0] | (blk[1] << 8)));
  uint32_t qh = uint32_t(blk[2] | (blk[3] << 8) | (blk[4] << 16) | (blk[5] << 24));
  for (int i = 0; i < 32; ++i) {
    int ql = (i < 16) ? (blk[6 + i] & 0xF) : (blk[6 + i - 16] >> 4);
    int q = (ql | (int((qh >> i) & 1u) << 4)) - 16;
    y[i] = d * float(q);
  }
}

// IQ4_NL: [fp16 d][16B qs, low nibble first], 32 values, non-linear grid.
void dequant_iq4_nl_block(const uint8_t* blk, float* y) {
  static const int8_t kgrid[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                   1, 13, 25, 38, 53, 69, 89, 113};
  float d = fp16_to_float(uint16_t(blk[0] | (blk[1] << 8)));
  for (int i = 0; i < 32; ++i) {
    int nib = (i < 16) ? (blk[2 + i] & 0xF) : (blk[2 + i - 16] >> 4);
    y[i] = d * float(kgrid[nib]);
  }
}

// Q3_K: [32B hmask][64B qs][12B scales][fp16 d], 256 values.
void dequant_q3_k_block(const uint8_t* blk, float* y) {
  const uint8_t* hmask = blk;
  const uint8_t* qs = blk + 32;
  const uint8_t* scales = blk + 96;
  float d = fp16_to_float(uint16_t(blk[108] | (blk[109] << 8)));
  float dl[16];
  for (int j = 0; j < 16; ++j) {
    int l = (j < 8) ? (scales[j] & 0xF) : ((scales[j - 8] >> 4) & 0xF);
    int h = (scales[8 + (j & 3)] >> (2 * (j >> 2))) & 3;
    dl[j] = d * float(int8_t(l | (h << 4)) - 32);
  }
  for (int i = 0; i < 256; ++i) {
    int h2 = i >> 7, s2 = (i >> 5) & 3, b2 = i & 31;
    int ql = (qs[h2 * 32 + b2] >> (s2 * 2)) & 3;
    int qh = ((hmask[i & 31] >> (i >> 5)) & 1) ^ 1;
    y[i] = dl[i >> 4] * float(ql - (qh << 2));
  }
}

// Q4_0: [fp16 d][16B qs, low nibble first], 32 values, y = d*(q-8).
void dequant_q4_0_block(const uint8_t* blk, float* y) {
  float d = fp16_to_float(uint16_t(blk[0] | (blk[1] << 8)));
  for (int i = 0; i < 32; ++i) {
    int nib = (i < 16) ? (blk[2 + i] & 0xF) : (blk[2 + i - 16] >> 4);
    y[i] = d * float(nib - 8);
  }
}

// Q4_1: [fp16 d][fp16 m][16B qs, low nibble first], 32 values, y = d*q+m.
void dequant_q4_1_block(const uint8_t* blk, float* y) {
  float d = fp16_to_float(uint16_t(blk[0] | (blk[1] << 8)));
  float m = fp16_to_float(uint16_t(blk[2] | (blk[3] << 8)));
  for (int i = 0; i < 32; ++i) {
    int nib = (i < 16) ? (blk[4 + i] & 0xF) : (blk[4 + i - 16] >> 4);
    y[i] = d * float(nib) + m;
  }
}

// IQ4_XS: [fp16 d][u16 scales_h][4B scales_l][128B qs], 256 values.
// Group g (8 x 32 vals): scale = (sl | sh<<4)-32, qs via IQ4_NL grid.
void dequant_iq4_xs_block(const uint8_t* blk, float* y) {
  static const int8_t kgrid[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                   1, 13, 25, 38, 53, 69, 89, 113};
  float d = fp16_to_float(uint16_t(blk[0] | (blk[1] << 8)));
  uint32_t h = uint32_t(blk[2] | (blk[3] << 8));
  for (int g = 0; g < 8; ++g) {
    int sl = (blk[4 + (g >> 1)] >> ((g & 1) * 4)) & 0xF;
    int sh = (h >> (2 * g)) & 3;
    float dl = d * float((sl | (sh << 4)) - 32);
    for (int j = 0; j < 32; ++j) {
      int b = (j < 16) ? blk[8 + g * 16 + j] : blk[8 + g * 16 + j - 16];
      int nib = (j < 16) ? (b & 0xF) : (b >> 4);
      y[g * 32 + j] = dl * float(kgrid[nib]);
    }
  }
}

// Q2_K: [16B scales][64B qs][fp16 d][fp16 dmin], 256 values.
// Group g (16 x 16 vals): y = d*(s&15)*q - dmin*(s>>4).
void dequant_q2_k_block(const uint8_t* blk, float* y) {
  float d = fp16_to_float(uint16_t(blk[80] | (blk[81] << 8)));
  float dmin = fp16_to_float(uint16_t(blk[82] | (blk[83] << 8)));
  for (int g = 0; g < 16; ++g) {
    float dl = d * float(blk[g] & 0xF);
    float ml = dmin * float(blk[g] >> 4);
    for (int j = 0; j < 16; ++j) {
      int byte = (g >> 3) * 32 + (g & 1) * 16 + j;
      int q = (blk[16 + byte] >> (((g >> 1) & 3) * 2)) & 3;
      y[g * 16 + j] = dl * float(q) - ml;
    }
  }
}

// Q4_K: [fp16 d][fp16 dmin][12B scales][128B qs], 256 values (ggml layout:
// subblock j covers bytes (j>>1)*32, nibble plane (j&1); within the subblock
// byte l -> element 2l (low nibble plane) and byte l+16 -> element 2l+1).
void dequant_q4_k_block(const uint8_t* blk, float* y) {
  float d = fp16_to_float(uint16_t(blk[0] | (blk[1] << 8)));
  float dmin = fp16_to_float(uint16_t(blk[2] | (blk[3] << 8)));
  for (int j = 0; j < 8; ++j) {
    uint32_t sc, mn;
    if (j < 4) {
      sc = blk[4 + j] & 63;
      mn = blk[4 + j + 4] & 63;
    } else {
      sc = (blk[4 + j + 4] & 0xF) | ((blk[4 + j - 4] >> 6) << 4);
      mn = (blk[4 + j + 4] >> 4) | ((blk[4 + j] >> 6) << 4);
    }
    float d1 = d * float(sc), m1 = dmin * float(mn);
    int base = (j >> 1) * 32, sh = (j & 1) * 4;
    for (int l = 0; l < 16; ++l) {
      y[j * 32 + 2 * l] = d1 * float((blk[16 + base + l] >> sh) & 0xF) - m1;
      y[j * 32 + 2 * l + 1] = d1 * float((blk[16 + base + 16 + l] >> sh) & 0xF) - m1;
    }
  }
}

// Q6_K: [128B ql][64B qh][16 x int8 scales][fp16 d], 256 values
// (same index math as WL::mat Q6_K in tools/ref_top5.cpp + qk_w_q6).
void dequant_q6_k_block(const uint8_t* blk, float* y) {
  float d = fp16_to_float(uint16_t(blk[208] | (blk[209] << 8)));
  for (int half = 0; half < 2; ++half) {
    const uint8_t* ql = blk + half * 64;
    const uint8_t* qh = blk + 128 + half * 32;
    const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192 + half * 8);
    for (int l = 0; l < 32; ++l) {
      int is = l >> 4;
      int q1 = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
      int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
      int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
      int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
      y[half * 128 + l] = d * float(sc[is + 0]) * float(q1 - 32);
      y[half * 128 + l + 32] = d * float(sc[is + 2]) * float(q2 - 32);
      y[half * 128 + l + 64] = d * float(sc[is + 4]) * float(q3 - 32);
      y[half * 128 + l + 96] = d * float(sc[is + 6]) * float(q4 - 32);
    }
  }
}

struct Kind { const char* dtype; size_t blk_bytes; size_t blk_vals; void (*deq)(const uint8_t*, float*); };
static const Kind kUpcastKinds[] = {
    {"Q5_0", 22, 32, dequant_q5_0_block},
    {"IQ4_NL", 18, 32, dequant_iq4_nl_block},
    {"Q3_K", 110, 256, dequant_q3_k_block},
    {"Q4_0", 18, 32, dequant_q4_0_block},
    {"Q4_1", 20, 32, dequant_q4_1_block},
    {"IQ4_XS", 136, 256, dequant_iq4_xs_block},
    {"Q2_K", 84, 256, dequant_q2_k_block},
    {"Q4_K", 144, 256, dequant_q4_k_block},
    {"Q6_K", 210, 256, dequant_q6_k_block},
};

const Kind* find_upcast_kind(const std::string& dtype) {
  for (auto& c : kUpcastKinds)
    if (dtype == c.dtype) return &c;
  return nullptr;
}

}  // namespace

bool GGUFLoader::load_upcast_f32(const std::string& name, std::vector<float>& out) const {
  const GGUFTensorMeta* m = tensor(name);
  if (!m) return false;
  size_t n = 1;
  for (int64_t d : m->shape) n *= static_cast<size_t>(d);
  const Kind* k = find_upcast_kind(m->dtype);
  if (!k || n % k->blk_vals != 0) return false;
  std::vector<uint8_t> raw(n / k->blk_vals * k->blk_bytes);
  read_tensor_data(*m, raw.data(), raw.size());
  out.resize(n);
  for (size_t b = 0; b < n / k->blk_vals; ++b)
    k->deq(raw.data() + b * k->blk_bytes, out.data() + b * k->blk_vals);
  return true;
}

size_t GGUFLoader::upcast_block_bytes(const std::string& dtype) {
  const Kind* k = find_upcast_kind(dtype);
  return k ? k->blk_bytes : 0;
}

size_t GGUFLoader::upcast_block_vals(const std::string& dtype) {
  const Kind* k = find_upcast_kind(dtype);
  return k ? k->blk_vals : 0;
}

bool GGUFLoader::dequant_block_f32(const std::string& dtype, const uint8_t* blk, float* y) {
  const Kind* k = find_upcast_kind(dtype);
  if (!k) return false;
  k->deq(blk, y);
  return true;
}

bool GGUFLoader::dequant_blocks_f16(const std::string& dtype, const uint8_t* raw,
                                    size_t n_elements, uint16_t* y) {
  const Kind* k = find_upcast_kind(dtype);
  if (!k || n_elements % k->blk_vals != 0) return false;
  std::vector<float> tmp(k->blk_vals);
  for (size_t b = 0; b < n_elements / k->blk_vals; ++b) {
    k->deq(raw + b * k->blk_bytes, tmp.data());
    for (size_t i = 0; i < k->blk_vals; ++i)
      y[b * k->blk_vals + i] = float_to_fp16(tmp[i]);
  }
  return true;
}

bool GGUFLoader::load_upcast_f16(const std::string& name, std::vector<uint16_t>& out) const {
  std::vector<float> f;
  if (!load_upcast_f32(name, f)) return false;
  out.resize(f.size());
  for (size_t i = 0; i < f.size(); ++i) out[i] = float_to_fp16(f[i]);
  return true;
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