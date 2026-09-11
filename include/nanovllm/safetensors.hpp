#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"

struct TensorMeta {
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t start = 0;
  int64_t end = 0;
  size_t file_index = 0;
};

class SafetensorsLoader {
 public:
  void add_directory(const std::string& dir);
  void add_file(const std::string& path);
  bool contains(const std::string& name) const { return meta_.find(name) != meta_.end(); }
  std::vector<std::string> names() const;
  std::vector<int64_t> shape(const std::string& name) const;
  bool load_float(const std::string& name, std::vector<float>& out) const;
  bool load_u16(const std::string& name, std::vector<uint16_t>& out, bool& bf16) const;

 private:
  struct File {
    std::string path;
    uint64_t base = 0;
  };
  std::vector<File> files_;
  std::map<std::string, TensorMeta> meta_;
};
