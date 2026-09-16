#define NOMINMAX
#include "nanovllm/http_server.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include "nanovllm/config.hpp"
#include "nanovllm/tokenizer.hpp"
#include "nanovllm/vulkan_model.hpp"

static void usage(const char* prog) {
  std::fprintf(stderr, "Usage: %s <model_dir> [prompt] [--max-tokens N] [--max-model-len N]\n", prog);
  std::fprintf(stderr, "       [--chat] [--temperature T] [--top-p P]\n");
  std::fprintf(stderr, "       [--rep-penalty THETA] [--rep-window W]\n");
  std::fprintf(stderr, "       [--bench N] [--vram]\n");
  std::fprintf(stderr, "        <model_dir> --server [host] [port]\n");
  std::fprintf(stderr, "  Sampling defaults come from generation_config.json when present;\n");
  std::fprintf(stderr, "  explicit flags always win. --rep-penalty divides the logits of ids\n");
  std::fprintf(stderr, "  seen in the last W generated tokens (default 1.0 = off, --rep-window 64;\n");
  std::fprintf(stderr, "  --rep-window 0 = whole history). Greedy (temp<=0) is never penalized.\n");
}

// CPU sampler: greedy at temp<=0, else temperature + nucleus sampling.
// SAMP-1: recent_ids/recent_len carry generated-token history (main's `out`);
// ids seen in the last rep_window entries have their scores divided by
// rep_penalty (applied pre-nucleus, i.e. penalized ids can drop out of top-p).
// rep_penalty == 1.0 (default) skips this entirely: behavior is bit-identical.
static int sample_row(const float* logits, int vocab, double temp, double top_p, std::mt19937& rng,
                      const int* recent_ids = nullptr, size_t recent_len = 0, double rep_penalty = 1.0,
                      int rep_window = 64) {
  if (temp <= 0) return (int)(std::max_element(logits, logits + vocab) - logits);
  std::vector<int> idx((size_t)vocab);
  for (int i = 0; i < vocab; ++i) idx[(size_t)i] = i;
  std::vector<float> sc((size_t)vocab);
  float mx = -1e30f;
  for (int i = 0; i < vocab; ++i) mx = std::max(mx, logits[i]);
  double sum = 0;
  for (int i = 0; i < vocab; ++i) { sc[(size_t)i] = std::exp((logits[i] - mx) / temp); sum += sc[(size_t)i]; }
  if (rep_penalty != 1.0 && rep_penalty > 0.0 && recent_ids != nullptr && recent_len > 0) {
    size_t w = rep_window > 0 ? std::min<size_t>((size_t)rep_window, recent_len) : recent_len;
    std::vector<int> seen;
    seen.reserve(w);
    bool changed = false;
    for (size_t k = recent_len - w; k < recent_len; ++k) {
      int id = recent_ids[k];
      if (id < 0 || id >= vocab) continue;
      if (std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
      seen.push_back(id);
      sc[(size_t)id] /= (float)rep_penalty;
      changed = true;
    }
    if (changed) {
      sum = 0;
      for (int i = 0; i < vocab; ++i) sum += sc[(size_t)i];
    }
  }
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return sc[(size_t)a] > sc[(size_t)b]; });
  double acc = 0, cutoff = 0;
  size_t keep = idx.size();
  for (size_t i = 0; i < idx.size(); ++i) {
    acc += sc[(size_t)idx[i]] / sum;
    if (acc >= top_p && i > 0) { keep = i + 1; break; }
  }
  double r = std::uniform_real_distribution<double>(0, 1)(rng) * acc, run = 0;
  for (size_t i = 0; i < keep; ++i) {
    run += sc[(size_t)idx[i]] / sum;
    if (r <= run) return idx[i];
  }
  return idx[keep - 1];
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string model_dir;
    std::string prompt = "Hello";
    int max_tokens = 64;
    int max_model_len = 4096;
    bool server_mode = false;
    bool use_chat = false;
    double temperature = 0.0;
    double top_p = 1.0;
    bool temp_set = false, topp_set = false, maxtok_set = false;  // HF-1: CLI wins over generation_config.json
    double rep_penalty = 1.0;  // SAMP-1: 1.0 = off (current behavior exactly)
    int rep_window = 64;
    std::string server_host = "127.0.0.1";
    int server_port = 8080;
    std::string system_prompt;
    int bench_runs = 0;   // PERF-1: repeat generation N times, report averages
    bool vram_flag = false;

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--max-tokens" && i + 1 < argc) { max_tokens = std::atoi(argv[++i]); maxtok_set = true; }
      else if (a == "--max-model-len" && i + 1 < argc) max_model_len = std::atoi(argv[++i]);
      else if (a == "--chat") use_chat = true;
      else if (a == "--system" && i + 1 < argc) system_prompt = argv[++i];
      else if (a == "--temperature" && i + 1 < argc) { temperature = std::atof(argv[++i]); temp_set = true; }
      else if (a == "--top-p" && i + 1 < argc) { top_p = std::atof(argv[++i]); topp_set = true; }
      else if (a == "--rep-penalty" && i + 1 < argc) rep_penalty = std::atof(argv[++i]);
      else if (a == "--rep-window" && i + 1 < argc) rep_window = std::atoi(argv[++i]);
      else if (a == "--bench" && i + 1 < argc) bench_runs = std::atoi(argv[++i]);
      else if (a == "--vram") vram_flag = true;
      else if (a == "--server") server_mode = true;
      else if (server_mode && server_host == "127.0.0.1" && argv[i][0] != '-') {
        if (strchr(argv[i], '.')) server_host = argv[i];
        else server_port = std::atoi(argv[i]);
      }
      else if (model_dir.empty()) model_dir = a;
      else if (prompt == "Hello") prompt = a;
      else { usage(argv[0]); return 1; }
    }
    if (model_dir.empty()) { usage(argv[0]); return 1; }

    Config config;
    config.model = model_dir;
    config.max_model_len = max_model_len;
    config.load_hf_config();
    // HF-1: generation_config.json values are defaults only; explicit CLI wins.
    if (!temp_set && config.has_gen_temperature) temperature = config.gen_temperature;
    if (!topp_set && config.has_gen_top_p) top_p = config.gen_top_p;
    if (!maxtok_set && config.has_gen_max_tokens) max_tokens = config.gen_max_tokens;
    config.eos = -1;

    Tokenizer tok;
    tok.set_fallback_vocab_size(config.hf.vocab_size);
    tok.load(config.model);
    if (tok.eos_token_id() >= 0) config.eos = tok.eos_token_id();
    else if (config.has_gen_eos) config.eos = config.gen_eos;

    VulkanModel model(config);
    model.load_weights();
    model.alloc_expert_slots();
    int num_blocks = model.allocate_kv_cache();
    std::fprintf(stderr, "Loaded model, KV cache blocks: %d\n", num_blocks);
    if (vram_flag) {
#ifdef _WIN32
      PROCESS_MEMORY_COUNTERS pmc{};
      if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        std::fprintf(stderr, "[vram] working_set_mb=%.1f kv_blocks=%d vocab=%d hidden=%d layers=%d\n",
                     pmc.WorkingSetSize / 1048576.0, num_blocks, config.hf.vocab_size,
                     config.hf.hidden_size, config.hf.num_hidden_layers);
#else
      std::fprintf(stderr, "[vram] kv_blocks=%d vocab=%d hidden=%d layers=%d\n",
                   num_blocks, config.hf.vocab_size, config.hf.hidden_size, config.hf.num_hidden_layers);
#endif
    }

    std::vector<int> ids = tok.encode_text(prompt);
    if (server_mode) {
      // SERV-1: OpenAI-compatible loop. One request at a time (mutex guards
      // the shared KV cache; concurrent batching is SERV-2).
      std::mutex srv_mu;
      minihttp::Handler handler = [&](const minihttp::ChatRequest& req) -> std::string {
        std::lock_guard<std::mutex> lk(srv_mu);
        std::string text = req.last_user;
        if (tok.has_chat_template()) text = tok.apply_chat_template(text);
        std::vector<int> rids = tok.encode_text(text);
        if (rids.empty()) return "";
        double temp = req.temperature > 0 ? req.temperature : temperature;
        int mt = req.max_tokens > 0 ? req.max_tokens : max_tokens;
        int rbsize = config.kvcache_block_size;
        int rmax_blocks = 64;
        std::vector<int32_t> rbt(rmax_blocks, -1);
        for (size_t i = 0; i < rids.size(); ++i) rbt[i / rbsize] = (int32_t)(i / rbsize);
        std::mt19937 rrng((unsigned)std::chrono::steady_clock::now().time_since_epoch().count());
        VKContext ctx;
        int n = (int)rids.size();
        ctx.input_ids.assign(rids.begin(), rids.end());
        ctx.positions.resize(n);
        for (int i = 0; i < n; ++i) ctx.positions[i] = i;
        ctx.slot_mapping.resize(n);
        for (int i = 0; i < n; ++i) ctx.slot_mapping[i] = (int32_t)i;
        ctx.block_tables = rbt;
        ctx.query_seq = std::vector<int32_t>(n, 0);
        ctx.query_key_len.resize(n);
        for (int i = 0; i < n; ++i) ctx.query_key_len[i] = i + 1;
        ctx.last_indices = {n - 1};
        ctx.max_blocks = rmax_blocks;
        auto lg = model.forward_logits(ctx);
        std::vector<int> rout;
        int pos = n;
        int nxt = sample_row(&lg[0], config.hf.vocab_size, temp, top_p, rrng,
                             rout.data(), rout.size(), rep_penalty, rep_window);
        rout.push_back(nxt);
        std::string text_out = tok.decode_tokens({nxt});
        while ((int)rout.size() < mt) {
          VKContext d;
          d.input_ids = {rout.back()};
          d.positions = {pos};
          std::vector<int32_t> bt2(rmax_blocks, -1);
          for (int i = 0; i <= pos; ++i) bt2[i / rbsize] = (int32_t)(i / rbsize);
          d.block_tables = bt2;
          d.slot_mapping = {pos};
          d.query_seq = {0};
          d.query_key_len = {pos + 1};
          d.last_indices = {0};
          d.max_blocks = rmax_blocks;
          auto lg2 = model.forward_logits(d);
          int t = sample_row(&lg2[0], config.hf.vocab_size, temp, top_p, rrng,
                             rout.data(), rout.size(), rep_penalty, rep_window);
          if (t == config.eos) break;
          text_out += tok.decode_tokens({t});
          rout.push_back(t);
          ++pos;
        }
        return text_out;
      };
      minihttp::serve(server_port, model_dir, handler);
      return 0;
    }
    if (use_chat) {
      if (tok.has_chat_template()) {
        ids = tok.encode_text(tok.apply_chat_template(prompt, system_prompt));
      } else {
      int im_start = tok.encode_special("<|im_start|>");
      int im_end = tok.encode_special("<|im_end|>");
      if (im_start >= 0 && im_end >= 0) {
        std::vector<int> c;
        auto append = [&](const std::string& s) {
          auto part = tok.encode_text(s);
          c.insert(c.end(), part.begin(), part.end());
        };
        if (tok.chat_auto_system()) {
          c.push_back(im_start);
          append("system\nYou are a helpful assistant.");
          c.push_back(im_end);
          append("\n");
        }
        c.push_back(im_start);
        append(std::string("user\n") + prompt);
        c.push_back(im_end);
        append("\n");
        c.push_back(im_start);
        append("assistant\n");
        ids.swap(c);
      } else {
        std::fprintf(stderr, "warning: --chat requested but model has no im_start/im_end specials; using raw prompt\n");
      }
      }
    }
    if (ids.empty()) {
      std::fprintf(stderr, "Prompt encoded to empty token list.\n");
      return 1;
    }
    std::mt19937 rng(std::random_device{}());

    std::fprintf(stderr, "Prompt: \"%s\" -> %d tokens\n", prompt.c_str(), (int)ids.size());

    int bsize = config.kvcache_block_size;
    int max_blocks = 64;
    std::vector<int32_t> bt(max_blocks, -1);
    for (size_t i = 0; i < ids.size(); ++i) bt[i / bsize] = (int32_t)(i / bsize);

    std::vector<int> out;
    int pos = (int)ids.size();

    // PERF-1: optional --bench loop (fresh KV each run); timings averaged.
    int runs = bench_runs > 0 ? bench_runs : 1;
    double sum_ttft = 0, sum_dec = 0;
    int sum_ntok = 0, sum_nprompt = 0;
    for (int br = 0; br < runs; ++br) {
      out.clear();
      pos = (int)ids.size();
      if (br > 0) model.allocate_kv_cache();
      const bool quiet = br > 0;
      using clk = std::chrono::steady_clock;
      auto t0 = clk::now();
      clk::time_point t1 = t0;
      bool done = false;

    // Prefill
    {
      VKContext ctx;
      int n = (int)ids.size();
      ctx.input_ids.assign(ids.begin(), ids.end());
      ctx.positions.resize(n);
      for (int i = 0; i < n; ++i) ctx.positions[i] = i;
      ctx.slot_mapping.resize(n);
      for (int i = 0; i < n; ++i) ctx.slot_mapping[i] = (int32_t)i;
      ctx.block_tables = bt;
      ctx.query_seq = std::vector<int32_t>(n, 0);
      ctx.query_key_len.resize(n);
      for (int i = 0; i < n; ++i) ctx.query_key_len[i] = i + 1;
      ctx.last_indices = {n - 1};
      ctx.max_blocks = max_blocks;
      auto lg = model.forward_logits(ctx);
      const float* row = &lg[0];
      if (getenv("NANO_DEBUG")) {
        std::vector<int> idx(config.hf.vocab_size);
        for (int i = 0; i < config.hf.vocab_size; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return row[a] > row[b]; });
        std::fprintf(stderr, "gpu top5:");
        for (int k = 0; k < 5; ++k) std::fprintf(stderr, " %d(%.3f)", idx[k], row[idx[k]]);
        std::fprintf(stderr, "\n");
      }
      int nxt = sample_row(row, config.hf.vocab_size, temperature, top_p, rng, out.data(), out.size(),
                           rep_penalty, rep_window);
      t1 = clk::now();
      if (!quiet) std::fprintf(stderr, "prefill next=%d\n", nxt);
      if (!quiet) std::printf("%s", tok.decode_tokens({nxt}).c_str());
      out.push_back(nxt);
      done = (nxt == config.eos || (int)out.size() >= max_tokens);
      }  // end prefill
      if (!done) {
      // Decode loop
      while ((int)out.size() < max_tokens) {
      VKContext d;
      d.input_ids = {out.back()};
      d.positions = {pos};
      std::vector<int32_t> bt2(max_blocks, -1);
      for (int i = 0; i <= pos; ++i) bt2[i / bsize] = (int32_t)(i / bsize);
      d.block_tables = bt2;
      d.slot_mapping = {pos};
      d.query_seq = {0};
      d.query_key_len = {pos + 1};
      d.last_indices = {0};
      d.max_blocks = max_blocks;
      auto lg = model.forward_logits(d);
      const float* row = &lg[0];
      int t = sample_row(row, config.hf.vocab_size, temperature, top_p, rng, out.data(), out.size(), rep_penalty,
                         rep_window);
      if (t == config.eos) break;
      if (!quiet) std::printf("%s", tok.decode_tokens({t}).c_str());
      if (!quiet) std::fflush(stdout);
      out.push_back(t);
      ++pos;
    }
    }  // end decode (skipped when prefill finished the run)
    auto t2 = clk::now();
    sum_ttft += std::chrono::duration<double, std::milli>(t1 - t0).count();
    sum_dec += std::chrono::duration<double, std::milli>(t2 - t1).count();
    sum_ntok += (int)out.size() > 0 ? (int)out.size() - 1 : 0;
    sum_nprompt += (int)ids.size();
    }  // end bench loop
    if (!bench_runs) std::printf("\n");
    std::fprintf(stderr, "[perf] ttft_ms=%.2f decode_tok_s=%.2f prompt_tok_s=%.2f runs=%d\n",
                 sum_ttft / runs, sum_dec > 0 ? 1000.0 * sum_ntok / sum_dec : 0.0,
                 sum_ttft > 0 ? 1000.0 * sum_nprompt / sum_ttft : 0.0, runs);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr, "Error: unknown exception\n");
    return 2;
  }
}
