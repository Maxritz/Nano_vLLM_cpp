#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include "nanovllm/llm_engine.hpp"

static void usage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s <model_dir> [prompt] [--tokens 1,2,3] [--max-tokens N] [--temperature T] [--max-model-len N] [--chat]\n"
               "       %s --bench <model_dir> [--num-seqs N] [--max-tokens N] [--max-model-len N]\n",
               prog, prog);
}

static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> ids;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    ids.push_back(std::stoi(s.substr(p, q - p)));
    p = q + 1;
  }
  return ids;
}

int main(int argc, char** argv) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
#endif
  try {
    if (argc < 2) {
      usage(argv[0]);
      return 1;
    }
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.push_back(argv[i]);
    std::string model_dir;
    std::string prompt;
    std::vector<int> prompt_ids;
    int max_tokens = 64;
    double temperature = 0.6;
    int max_model_len = 4096;
    bool bench = false;
    bool chat = false;
    int bench_seqs = 4;

    for (size_t i = 0; i < args.size(); ++i) {
      const std::string& a = args[i];
      if (a == "--tokens" && i + 1 < args.size()) prompt_ids = parse_ids(args[++i]);
      else if (a == "--max-tokens" && i + 1 < args.size()) max_tokens = std::stoi(args[++i]);
      else if (a == "--temperature" && i + 1 < args.size()) temperature = std::stod(args[++i]);
      else if (a == "--max-model-len" && i + 1 < args.size()) max_model_len = std::stoi(args[++i]);
      else if (a == "--bench") bench = true;
      else if (a == "--chat") chat = true;
      else if (a == "--num-seqs" && i + 1 < args.size()) bench_seqs = std::stoi(args[++i]);
      else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
      else if (model_dir.empty()) model_dir = a;
      else if (prompt.empty()) prompt = a;
      else { usage(argv[0]); return 1; }
    }
    if (model_dir.empty()) {
      usage(argv[0]);
      return 1;
    }

    LLMEngine engine(model_dir, max_model_len);
    size_t free_mem = 0, total_mem = 0;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    std::fprintf(stderr, "VRAM after load: free %.1f GB / %.1f GB\n", free_mem / 1e9, total_mem / 1e9);
    auto t1 = std::chrono::steady_clock::now();
    if (getenv("NANO_DEBUG")) {
      auto dbg = engine.encode_text(prompt.empty() ? "Hello" : prompt);
      std::fprintf(stderr, "encode %zu:", dbg.size());
      for (int id : dbg) std::fprintf(stderr, " %d", id);
      std::fprintf(stderr, "\n");
      std::fflush(stderr);
    }
    SamplingParams sp;
    sp.temperature = temperature;
    sp.max_tokens = max_tokens;
    std::vector<GenerateOutput> outputs;
    if (bench) {
      std::vector<std::vector<int>> prompts;
      std::vector<SamplingParams> params;
      int vocab = engine.vocab_size();
      int span = std::max(1, vocab - 2);
      for (int i = 0; i < bench_seqs; ++i) {
        std::vector<int> ids(64);
        for (int t = 0; t < 64; ++t) ids[t] = 1 + ((i * 37 + t) % span);
        prompts.push_back(ids);
        SamplingParams p = sp;
        params.push_back(p);
      }
      outputs = engine.generate(prompts, params);
      auto t2 = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(t2 - t1).count();
      long long gen = 0;
      for (auto& o : outputs) gen += static_cast<long long>(o.token_ids.size());
      long long pp_toks = static_cast<long long>(prompts.size()) * 64;
      std::printf("total_seq: %d  prompt_tokens: %lld  generated_tokens: %lld\n", bench_seqs, pp_toks, gen);
      if (secs > 0.0) std::printf("PP: %lld tokens / %.3fs = %.1f tok/s\n", pp_toks, secs,
                                  pp_toks / secs);
      if (secs > 0.0 && gen > 0) std::printf("decode: %lld tokens / %.3fs = %.1f tok/s\n", gen, secs,
                                             gen / secs);
    } else if (!prompt_ids.empty()) {
      outputs = engine.generate(std::vector<std::vector<int>>{prompt_ids}, std::vector<SamplingParams>{sp});
    } else if (chat) {
      outputs = engine.generate_chat(std::vector<std::string>{prompt.empty() ? "Hello" : prompt}, std::vector<SamplingParams>{sp});
    } else {
      outputs = engine.generate(std::vector<std::string>{prompt.empty() ? "Hello" : prompt}, sp);
    }

    if (!bench) {
      const bool dbg = getenv("NANO_DEBUG") != nullptr;
      for (auto& out : outputs) {
        if (dbg) {
          std::fprintf(stderr, "token_ids:");
          for (int id : out.token_ids) std::fprintf(stderr, " %d", id);
          std::fprintf(stderr, "\n");
        }
        std::printf("%s\n", out.text.c_str());
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "Error: unknown exception\n");
    return 2;
  }
  return 0;
}
