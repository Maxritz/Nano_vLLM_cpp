// gguf_info <file.gguf> [substr] [row]: list tensors (dtype + shape), filtered.
// With [row], also prints the host-upcast sum of that row for exact tensor names.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "nanovllm/gguf.hpp"

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: gguf_info <file.gguf> [substr] [row]\n"); return 1; }
  std::string filt = argc > 2 ? argv[2] : "";
  long row = argc > 3 ? std::atol(argv[3]) : -1;
  GGUFLoader g;
  g.add_file(argv[1]);
  for (auto& n : g.names()) {
    if (!filt.empty() && n.find(filt) == std::string::npos) continue;
    const GGUFTensorMeta* tm = g.tensor(n);
    std::printf("%s dtype=%s shape=[", n.c_str(), tm->dtype.c_str());
    for (size_t i = 0; i < tm->shape.size(); ++i)
      std::printf("%s%lld", i ? "x" : "", (long long)tm->shape[i]);
    std::printf("]");
    if (row >= 0 && n == filt && !tm->shape.empty()) {
      std::vector<float> up;
      size_t cols = (size_t)tm->shape[0];
      if (g.load_upcast_f32(n, up) && (size_t)(row + 1) * cols <= up.size()) {
        double s = 0;
        for (size_t i = 0; i < cols; ++i) s += up[(size_t)row * cols + i];
        std::printf(" rowsum[%ld]=%.6f vals=", row, s);
        for (size_t i = 0; i < cols && i < 256; ++i) std::printf("%.6f ", up[(size_t)row * cols + i]);
      } else {
        std::printf(" rowsum=UNAVAILABLE");
      }
    }
    std::printf("\n");
  }
  return 0;
}
