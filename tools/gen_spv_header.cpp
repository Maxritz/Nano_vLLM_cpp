// Embed compiled SPIR-V (.spv) into a C header as static const uint32 arrays,
// plus a name->word-count table. Usage: gen_spv_header <spv_dir> <out_header>
// (C++ replacement for the old gen_spv_header.py; byte-identical output.)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: gen_spv_header <spv_dir> <out_header>\n");
    return 1;
  }
  std::vector<std::string> names;
  for (const auto& e : std::filesystem::directory_iterator(argv[1])) {
    std::string fn = e.path().filename().string();
    if (fn.size() > 4 && fn.compare(fn.size() - 4, 4, ".spv") == 0)
      names.push_back(fn.substr(0, fn.size() - 4));
  }
  std::sort(names.begin(), names.end());
  std::ofstream f(argv[2], std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", argv[2]);
    return 1;
  }
  f << "#pragma once\n#include <cstddef>\n"
       "/* Auto-generated: embedded SPIR-V for the Vulkan backend. */\n";
  std::vector<size_t> sizes;
  char buf[16];
  for (const auto& name : names) {
    std::ifstream sf(std::filesystem::path(argv[1]) / (name + ".spv"), std::ios::binary);
    std::vector<uint32_t> words((std::filesystem::file_size(
                                     std::filesystem::path(argv[1]) / (name + ".spv"))) /
                                4);
    sf.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
    f << "static const unsigned int " << name << "_spv[] = {\n";
    for (size_t i = 0; i < words.size(); i += 8) {
      f << "  ";
      for (size_t j = i; j < words.size() && j < i + 8; ++j) {
        std::snprintf(buf, sizeof(buf), "0x%08x,", words[j]);
        f << buf << (j + 1 < words.size() && j + 1 < i + 8 ? " " : "");
      }
      f << "\n";
    }
    f << "};\n";
    sizes.push_back(words.size());
  }
  f << "static const unsigned int spv_count(const char* n) {\n  if (!n) return 0;\n";
  for (size_t i = 0; i < names.size(); ++i)
    f << "  if (strcmp(n, \"" << names[i] << "\")==0) return " << sizes[i] << ";\n";
  f << "  return 0;\n}\n";
  f << "static const unsigned int* spv_for(const char* n) {\n  if (!n) return 0;\n";
  for (const auto& name : names)
    f << "  if (strcmp(n, \"" << name << "\")==0) return " << name << "_spv;\n";
  f << "  return 0;\n}\n";
  std::printf("embedded %zu shaders -> %s\n", names.size(), argv[2]);
  return 0;
}
