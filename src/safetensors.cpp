#include "nanovllm/safetensors.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sstream>

#include "nanovllm/common.hpp"

void SafetensorsLoader::add_directory(const std::string& dir) {
  std::vector<std::string> paths;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
      paths.push_back(entry.path().string());
    }
  }
  std::sort(paths.begin(), paths.end());
  for (const auto& p : paths) add_file(p);
}

void SafetensorsLoader::add_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open safetensors file: " + path);
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  if (!f) throw std::runtime_error("bad safetensors header in " + path);
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);
  if (!f) throw std::runtime_error("truncated safetensors header in " + path);
  Json root = JsonParser::parse(header);
  size_t file_index = files_.size();
  files_.push_back({path, 8 + header_len});
  for (const auto& kv : root.obj) {
    if (kv.first == "__metadata__" || !kv.second.is_object()) continue;
    TensorMeta tm;
    tm.file_index = file_index;
    const auto& e = kv.second;
    if (e.contains("dtype")) tm.dtype = e.at("dtype").as_string();
    if (e.contains("data_offsets") && e.at("data_offsets").is_array()) {
      const auto& offs = e.at("data_offsets").as_array();
      if (offs.size() == 2) {
        tm.start = offs[0].as_int64();
        tm.end = offs[1].as_int64();
      }
    }
    if (e.contains("shape") && e.at("shape").is_array()) {
      for (const auto& s : e.at("shape").as_array()) tm.shape.push_back(s.as_int64());
    }
    meta_[kv.first] = tm;
  }
}

std::vector<std::string> SafetensorsLoader::names() const {
  std::vector<std::string> out;
  out.reserve(meta_.size());
  for (auto& kv : meta_) out.push_back(kv.first);
  return out;
}

std::vector<int64_t> SafetensorsLoader::shape(const std::string& name) const {
  auto it = meta_.find(name);
  return it == meta_.end() ? std::vector<int64_t>{} : it->second.shape;
}

bool SafetensorsLoader::load_float(const std::string& name, std::vector<float>& out) const {
  auto it = meta_.find(name);
  if (it == meta_.end()) return false;
  const TensorMeta& tm = it->second;
  const File& file = files_[tm.file_index];
  std::ifstream f(file.path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot reopen safetensors file: " + file.path);
  size_t bytes = size_t(tm.end - tm.start);
  std::vector<uint8_t> raw(bytes);
  f.seekg(file.base + tm.start);
  f.read(reinterpret_cast<char*>(raw.data()), bytes);
  if (!f) throw std::runtime_error("truncated tensor data for " + name);

  size_t elems = bytes;
  if (tm.dtype == "F32" || tm.dtype == "I32") elems = bytes / 4;
  else if (tm.dtype == "F16" || tm.dtype == "BF16") elems = bytes / 2;
  else if (tm.dtype == "F64" || tm.dtype == "I64") elems = bytes / 8;
  else if (tm.dtype == "I8" || tm.dtype == "U8") elems = bytes;
  else throw std::runtime_error("unsupported dtype " + tm.dtype + " for " + name);

  out.resize(elems);
  if (tm.dtype == "F32" || tm.dtype == "I32") {
    std::memcpy(out.data(), raw.data(), bytes);
  } else if (tm.dtype == "F16") {
    for (size_t i = 0; i < elems; ++i) out[i] = fp16_to_float(reinterpret_cast<uint16_t*>(raw.data())[i]);
  } else if (tm.dtype == "BF16") {
    for (size_t i = 0; i < elems; ++i) out[i] = bf16_to_float(reinterpret_cast<uint16_t*>(raw.data())[i]);
  } else if (tm.dtype == "F64") {
    for (size_t i = 0; i < elems; ++i) out[i] = static_cast<float>(reinterpret_cast<double*>(raw.data())[i]);
  } else if (tm.dtype == "I64") {
    for (size_t i = 0; i < elems; ++i) out[i] = static_cast<float>(reinterpret_cast<int64_t*>(raw.data())[i]);
  } else if (tm.dtype == "I8") {
    for (size_t i = 0; i < elems; ++i) out[i] = static_cast<float>(static_cast<int8_t>(raw[i]));
  } else if (tm.dtype == "U8") {
    for (size_t i = 0; i < elems; ++i) out[i] = static_cast<float>(raw[i]);
  }
  return true;
}

bool SafetensorsLoader::load_u16(const std::string& name, std::vector<uint16_t>& out, bool& bf16) const {
  auto it = meta_.find(name);
  if (it == meta_.end()) return false;
  const TensorMeta& tm = it->second;
  if (tm.dtype != "F16" && tm.dtype != "BF16")
    throw std::runtime_error("load_u16 expected F16/BF16 but got " + tm.dtype + " for " + name);
  const File& file = files_[tm.file_index];
  std::ifstream f(file.path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot reopen safetensors file: " + file.path);
  size_t bytes = size_t(tm.end - tm.start);
  if (bytes % 2 != 0) throw std::runtime_error("odd byte count for " + name);
  out.resize(bytes / 2);
  f.seekg(file.base + tm.start);
  f.read(reinterpret_cast<char*>(out.data()), bytes);
  if (!f) throw std::runtime_error("truncated tensor data for " + name);
  bf16 = (tm.dtype == "BF16");
  return true;
}
