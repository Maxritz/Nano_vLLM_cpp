#define NOMINMAX
#include "nanovllm/http_server.hpp"
#include "nanovllm/profiles.hpp"
#include "nanovllm/bm25.hpp"
#include "nanovllm/mcp_client.hpp"
#include "nanovllm/reason.hpp"
#include "nanovllm/gen.hpp"
#include "nanovllm/drafter.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <memory>
#include <random>
#include <sstream>
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
#include "nanovllm/agent.hpp"

static void usage(const char* prog) {
  std::fprintf(stderr, "Usage: %s <model_dir> [prompt] [--max-tokens N] [--max-model-len N]\n", prog);
  std::fprintf(stderr, "       [--chat] [--repl] [--temperature T] [--top-p P]\n");
  std::fprintf(stderr, "       [--rep-penalty THETA] [--rep-window W] [--system S]\n");
  std::fprintf(stderr, "       [--bench N] [--vram]\n");
  std::fprintf(stderr, "       <model_dir> --server [host] [port]\n");
  std::fprintf(stderr, "  Sampling defaults come from generation_config.json when present;\n");
  std::fprintf(stderr, "  explicit flags always win. --rep-penalty divides the logits of ids\n");
  std::fprintf(stderr, "  seen in the last W generated tokens (default 1.0 = off, --rep-window 64;\n");
  std::fprintf(stderr, "  --rep-window 0 = whole history). Greedy (temp<=0) is never penalized.\n");
  std::fprintf(stderr, "  --repl: interactive chat loop reading stdin line by line (EOF to exit).\n");
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

// REPL-1: one generation turn over the shared KV cache. `out` is the running
// generated-token history (across turns) so the model keeps conversational
// context; `pos` is the current sequence length. Returns the decoded text and
// appends generated ids to `out`. Mirrors the inline prefill/decode loops above
// so behavior is identical (greedy at temp<=0, nucleus + windowed rep-penalty).
static std::string run_turn(VulkanModel& model, const Config& config, const Tokenizer& tok,
                            const std::vector<int>& prompt_ids, std::vector<int>& out, int& pos,
                            double temperature, double top_p, double rep_penalty, int rep_window,
                            int max_tokens, std::mt19937& rng,
                            const drafter::NgramIndex* draft = nullptr, int draft_k = 8,
                            int draft_n = 2) {
  int bsize = config.kvcache_block_size;
  int max_blocks = 64;
  std::vector<int32_t> bt(max_blocks, -1);
  // Absolute slots: append this turn's prompt after the retained history.
  for (size_t i = 0; i < prompt_ids.size(); ++i) {
    int slot = pos + (int)i;
    bt[slot / bsize] = (int32_t)(slot / bsize);
  }

  VKContext ctx;
  int n = (int)prompt_ids.size();
  ctx.input_ids.assign(prompt_ids.begin(), prompt_ids.end());
  ctx.positions.resize(n);
  for (int i = 0; i < n; ++i) ctx.positions[i] = pos + i;
  ctx.slot_mapping.resize(n);
  for (int i = 0; i < n; ++i) ctx.slot_mapping[i] = (int32_t)(pos + i);
  ctx.block_tables = bt;
  ctx.query_seq = std::vector<int32_t>(n, 0);
  ctx.query_key_len.resize(n);
  for (int i = 0; i < n; ++i) ctx.query_key_len[i] = pos + i + 1;
  ctx.last_indices = {n - 1};
  ctx.max_blocks = max_blocks;

  auto lg = model.forward_logits(ctx);
  const float* row = &lg[0];
  int nxt = sample_row(row, config.hf.vocab_size, temperature, top_p, rng,
                       out.data(), out.size(), rep_penalty, rep_window);
  pos += n;
  std::string text = tok.decode_tokens({nxt});
  out.push_back(nxt);
  while ((int)out.size() < max_tokens && nxt != config.eos) {
    // DEC-2: speculative decode. The drafter predicts K tokens from the last
    // n-1 generated tokens; we feed {last, d[0..K-1]} in ONE forward and
    // accept the matching prefix. Rejected draft slots sit beyond the next
    // read position (query_key_len = pos+accept+1), so no rollback is needed.
    // lg[i] predicts d[i] for i<K; lg[K] predicts after d[K-1].
    int vocab = config.hf.vocab_size;
    int accepted = 0;
    if (draft && draft_k > 0 && temperature <= 0.0) {
      std::vector<int> tail;
      int tn = std::min(draft_n - 1, (int)out.size());
      if (tn > 0) tail.assign(out.end() - tn, out.end());
      auto m = draft->suggest(tail);
      if (m.found && (int)m.draft.size() > 1) {
        std::vector<int> d = m.draft;  // [gram..., successor]
        int K = std::min(draft_k, (int)d.size());
        VKContext s;
        s.input_ids.reserve(1 + K);
        s.input_ids.push_back(out.back());
        for (int i = 0; i < K; ++i) s.input_ids.push_back(d[i]);
        int total = 1 + K;
        s.positions.resize(total);
        for (int i = 0; i < total; ++i) s.positions[i] = pos - 1 + i;
        s.slot_mapping.resize(total);
        for (int i = 0; i < total; ++i) s.slot_mapping[i] = (int32_t)(pos - 1 + i);
        std::vector<int32_t> bt3(max_blocks, -1);
        for (int i = 0; i <= pos - 1 + K; ++i) bt3[i / bsize] = (int32_t)(i / bsize);
        s.block_tables = bt3;
        s.query_seq = std::vector<int32_t>(total, 0);
        s.query_key_len.resize(total);
        for (int i = 0; i < total; ++i) s.query_key_len[i] = (pos - 1 + i) + 1;
        s.last_indices.resize(total);
        for (int i = 0; i < total; ++i) s.last_indices[i] = i;
        s.max_blocks = max_blocks;
        auto lg = model.forward_logits(s);
        const float* base = lg.data();
        for (int i = 0; i < K; ++i) {
          const float* r = base + (size_t)i * vocab;
          int arg = (int)(std::max_element(r, r + vocab) - r);
          if (arg != d[i]) { accepted = i; break; }
          accepted = i + 1;
        }
        if (getenv("NANO_DEBUG"))
          std::fprintf(stderr, "[T] draft K=%d accepted=%d\n", K, accepted);
        // Emit the accepted prefix, then sample the tail from the row that
        // disagreed (or lg[K] if all K matched).
        int emit_n = accepted;
        for (int i = 0; i < emit_n; ++i) {
          text += tok.decode_tokens({d[i]});
          out.push_back(d[i]); ++pos;
        }
        const float* prow = base + (size_t)accepted * vocab;
        int t = sample_row(prow, vocab, temperature, top_p, rng,
                           out.data(), out.size(), rep_penalty, rep_window);
        if (t == config.eos) break;
        text += tok.decode_tokens({t});
        out.push_back(t); ++pos;
        continue;
      }
    }
    // Baseline: one token per forward.
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
    auto lg2 = model.forward_logits(d);
    row = &lg2[0];
    int t = sample_row(row, vocab, temperature, top_p, rng,
                       out.data(), out.size(), rep_penalty, rep_window);
    if (t == config.eos) break;
    text += tok.decode_tokens({t});
    out.push_back(t);
    ++pos;
  }
  return text;
}

// REAS-1 / GEN-1: post-process a raw reply. --strip-thinking removes reasoning
// blocks; --json-shape enforces a JSON object shape, stopping the turn on the
// first rejected character. Returns the final text to print.
static std::string postprocess_reply(const std::string& raw, bool strip_think,
                                     const std::string& json_shape) {
  std::string out = raw;
  if (strip_think) out = reason::strip_thinking(out);
  if (json_shape.empty()) return out;
  std::unique_ptr<gen::Constraint> c = gen::json_shape(json_shape);
  if (!c) return out;
  std::string ok;
  for (char ch : out) {
    if (!c->allow(ch)) break;
    c->feed(ch);
    ok += ch;
  }
  return ok;
}

// AGENT-1: tool executor. `name=cmd` registers a shell command; the agent loop
// parses a tool call from the model's reply, runs the matching command with
// the JSON args piped to its stdin, and feeds stdout back as the next user
// turn. No tool found -> the reply is final text.
// REAS-1: --strip-thinking drops [thinking]..[/thinking] blocks from the reply
// before it is printed (model reasoning is discarded, only the answer shows).
// GEN-1: --json-shape a,b,c constrains the reply to a JSON object with exactly
// those keys; a rejected character stops the turn (the partial text is kept).
struct ToolExec {
  std::string name, cmd;
};
struct McpExec { std::string name; mcp::Client* client; };  // MCP-1: tool served by an MCP server
static std::string run_tool(const std::vector<ToolExec>& tools,
                            const std::vector<McpExec>& mcp_tools,
                            const agent::Call& call) {
  for (const auto& t : tools) {
    if (t.name == call.name) {
      std::string cmd = t.cmd + " 2>&1";
      FILE* p = _popen(cmd.c_str(), "w");
      if (!p) return std::string("[tool error] cannot run ") + t.name;
      if (!call.args.empty()) std::fputs(call.args.c_str(), p);
      std::fclose(p);
      // Re-run capturing output.
      p = _popen(cmd.c_str(), "r");
      if (!p) return std::string("[tool error] cannot run ") + t.name;
      std::string out; char buf[4096];
      while (std::fgets(buf, sizeof(buf), p)) out += buf;
      std::fclose(p);
      return out;
    }
  }
  for (const auto& m : mcp_tools) {
    if (m.name == call.name) {
      auto contents = m.client->call_tool(call.name, call.args);
      std::string out;
      for (const auto& c : contents) out += c.text;
      return out.empty() ? std::string("[tool error] empty MCP result") : out;
    }
  }
  return std::string("[tool error] unknown tool: ") + call.name;
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
    bool use_repl = false;
    double temperature = 0.0;
    double top_p = 1.0;
    bool temp_set = false, topp_set = false, maxtok_set = false;  // HF-1: CLI wins over generation_config.json
    bool rep_set = false, repwin_set = false, mml_set = false;
    std::string profile_name;  // PROF-1: --profile chat|long|bench|default
    double rep_penalty = 1.0;  // SAMP-1: 1.0 = off (current behavior exactly)
    int rep_window = 64;
    std::string server_host = "127.0.0.1";
    int server_port = 8080;
    std::string system_prompt;
    int bench_runs = 0;   // PERF-1: repeat generation N times, report averages
    bool vram_flag = false;
    std::string tools_spec;          // AGENT-1: "name=cmd,name=cmd,..."
    int max_iters = 0;               // AGENT-1: 0 = no tool loop
    std::string mcp_cmd;             // MCP-1: "name=cmd" served by an MCP server
    std::string rag_path;            // RAG-1: newline-delimited "id<TAB>text" docs
    int rag_top_k = 3;               // RAG-1: BM25 top-k docs per turn
    size_t rag_max_chars = 2048;     // RAG-1: context budget before the user turn
    bool strip_think = false;        // REAS-1: drop [thinking]..[/thinking] blocks
    std::string json_shape;          // GEN-1: constrain output to a JSON shape
    int draft_n = 0;                 // DEC-2: n-gram order (0 = off)
    int draft_k = 8;                 // DEC-2: draft length per step

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--max-tokens" && i + 1 < argc) { max_tokens = std::atoi(argv[++i]); maxtok_set = true; }
      else if (a == "--max-model-len" && i + 1 < argc) { max_model_len = std::atoi(argv[++i]); mml_set = true; }
      else if (a == "--chat") use_chat = true;
      else if (a == "--repl") use_repl = true;
      else if (a == "--system" && i + 1 < argc) system_prompt = argv[++i];
      else if (a == "--temperature" && i + 1 < argc) { temperature = std::atof(argv[++i]); temp_set = true; }
      else if (a == "--top-p" && i + 1 < argc) { top_p = std::atof(argv[++i]); topp_set = true; }
else if (a == "--rep-penalty" && i + 1 < argc) { rep_penalty = std::atof(argv[++i]); rep_set = true; }
      else if (a == "--rep-window" && i + 1 < argc) { rep_window = std::atoi(argv[++i]); repwin_set = true; }
      else if (a == "--profile" && i + 1 < argc) profile_name = argv[++i];
      else if (a == "--bench" && i + 1 < argc) bench_runs = std::atoi(argv[++i]);
      else if (a == "--vram") vram_flag = true;
      else if (a == "--tools" && i + 1 < argc) tools_spec = argv[++i];
      else if (a == "--max-iters" && i + 1 < argc) max_iters = std::atoi(argv[++i]);
      else if (a == "--mcp" && i + 1 < argc) mcp_cmd = argv[++i];
      else if (a == "--strip-thinking") strip_think = true;
      else if (a == "--json-shape" && i + 1 < argc) json_shape = argv[++i];
      else if (a == "--rag-docs" && i + 1 < argc) rag_path = argv[++i];
      else if (a == "--rag-top-k" && i + 1 < argc) rag_top_k = std::atoi(argv[++i]);
      else if (a == "--rag-chars" && i + 1 < argc) rag_max_chars = (size_t)std::atoi(argv[++i]);
      else if (a == "--draft-n" && i + 1 < argc) draft_n = std::atoi(argv[++i]);
      else if (a == "--draft-k" && i + 1 < argc) draft_k = std::atoi(argv[++i]);
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
    // PROF-1: --profile chat|long|bench|default sets generation + KV-cache
    // defaults. CLI flags already set on those fields win over the profile.
    if (!profile_name.empty()) {
      profile::Profile pr = profile::get(profile_name);
      if (!temp_set) temperature = pr.temperature;
      if (!topp_set) top_p = pr.top_p;
      if (!maxtok_set) max_tokens = pr.max_tokens;
      if (!rep_set) rep_penalty = pr.rep_penalty;
      if (!repwin_set) rep_window = pr.rep_window;
      if (!mml_set) max_model_len = pr.max_model_len;
      if (pr.use_chat) use_chat = true;
      if (!system_prompt.empty()) system_prompt = pr.system_prompt;
      if (getenv("NANO_DEBUG"))
        std::fprintf(stderr, "[T] profile=%s temp=%.2f top_p=%.2f max_tokens=%d rep=%.2f repwin=%d mml=%d chat=%d\n",
                     profile_name.c_str(), temperature, top_p, max_tokens, rep_penalty,
                     rep_window, max_model_len, (int)use_chat);
    }
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

    // RAG-1: BM25 retrieval index. --rag-docs path loads newline-delimited
    // "id<TAB>text" rows; each user turn is stuffed with the top-k hits
    // before tokenization (context budget = --rag-chars).
    rag::Index rag_idx;
    if (!rag_path.empty()) {
      std::ifstream rf(rag_path);
      std::string line;
      while (std::getline(rf, line)) {
        if (line.empty()) continue;
        std::string id, text;
        auto tab = line.find('\t');
        if (tab != std::string::npos) { id = line.substr(0, tab); text = line.substr(tab + 1); }
        else { id = line; text = line; }
        if (!id.empty()) rag_idx.add({id, text});
      }
      if (!rag_idx.empty()) {
        rag_idx.build();
        if (getenv("NANO_DEBUG"))
          std::fprintf(stderr, "[T] rag index built docs=%zu\n", rag_idx.size());
      }
    }

    VulkanModel model(config);
    model.load_weights();
    model.alloc_expert_slots();
    int num_blocks = model.allocate_kv_cache();
    std::fprintf(stderr, "Loaded model, KV cache blocks: %d\n", num_blocks);
    if (vram_flag) {
      std::fprintf(stderr, "%s\n", model.memory_report().c_str());
    }

    std::vector<int> ids = tok.encode_text(prompt);
    // DEC-2: model-free n-gram drafter (TokenSwift-style). Built once from the
    // prompt; suggest() uses only the index, so it stays valid across turns.
    drafter::NgramIndex drafter;
    if (draft_n >= 2 && !ids.empty()) {
      drafter.build(ids, draft_n);
      if (getenv("NANO_DEBUG"))
        std::fprintf(stderr, "[T] drafter n=%d built on %d prompt tokens\n", draft_n, (int)ids.size());
    }
    if (server_mode) {
      // SERV-1: OpenAI-compatible loop. One request at a time (mutex guards
      // the shared KV cache; concurrent batching is SERV-2).
      std::mutex srv_mu;
      minihttp::Handler handler = [&](const minihttp::ChatRequest& req) -> std::string {
        std::lock_guard<std::mutex> lk(srv_mu);
        std::string text = req.last_user;
        if (!rag_idx.empty()) text = rag_idx.stuff(text, text, rag_top_k, rag_max_chars);
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
    if (use_repl) {
      // REPL-1: interactive chat loop. Each line is a fresh turn; the shared
      // KV cache keeps conversational context across turns (out + pos persist).
      // EOF (Ctrl-D / Ctrl-Z) exits cleanly. No chat template -> raw prompt.
      // AGENT-1: --tools name=cmd,... + --max-iters N enables a tool-call loop:
      // after each reply we scan for a fenced/bare tool call, execute the
      // matching command with the JSON args on stdin, and feed stdout back as
      // the next user turn, up to N iterations.
      std::vector<ToolExec> tools;
      if (!tools_spec.empty()) {
        std::istringstream ts(tools_spec);
        std::string item;
        while (std::getline(ts, item, ',')) {
          auto eq = item.find('=');
          if (eq != std::string::npos && eq + 1 < item.size())
            tools.push_back({item.substr(0, eq), item.substr(eq + 1)});
        }
      }
      agent::Loop agloop;
      for (const auto& t : tools) {
        agent::Tool at;
        at.name = t.name;
        at.schema = "{}";
        agloop.add_tool(at);
      }
      // MCP-1: an MCP server process exposes tools over JSON-RPC on stdio.
      // --mcp "name=cmd" spawns it, lists its tools, and dispatches calls to it.
      std::unique_ptr<mcp::Client> mcp_client;
      std::vector<McpExec> mcp_tools;
      if (!mcp_cmd.empty()) {
        auto eq = mcp_cmd.find('=');
        std::string mname = eq != std::string::npos ? mcp_cmd.substr(0, eq) : mcp_cmd;
        std::string mcmd = eq != std::string::npos ? mcp_cmd.substr(eq + 1) : mcp_cmd;
        try {
          mcp_client = std::make_unique<mcp::Client>(mcmd);
          if (mcp_client->initialize()) {
            for (const auto& mt : mcp_client->list_tools()) {
              agent::Tool at;
              at.name = mt.name;
              at.schema = mt.schema.empty() ? "{}" : mt.schema;
              agloop.add_tool(at);
              mcp_tools.push_back({mt.name, mcp_client.get()});
              if (getenv("NANO_DEBUG"))
                std::fprintf(stderr, "[T] mcp tool=%s\n", mt.name.c_str());
            }
          }
        } catch (const std::exception& e) {
          std::fprintf(stderr, "[mcp] spawn failed: %s\n", e.what());
          mcp_client.reset();
        }
      }
      std::string sys_block = agloop.tools_system_block();

      std::vector<int> out;
      int pos = 0;
      std::mt19937 rng(std::random_device{}());
      std::string line;
      while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::vector<int> turn_ids;
        std::string prompt_text = line;
        if (!rag_idx.empty()) prompt_text = rag_idx.stuff(prompt_text, line, rag_top_k, rag_max_chars);
        if (!sys_block.empty()) prompt_text = sys_block + "\n" + prompt_text;
        if (tok.has_chat_template()) turn_ids = tok.encode_text(tok.apply_chat_template(prompt_text, system_prompt));
        else turn_ids = tok.encode_text(prompt_text);
        if (turn_ids.empty()) continue;
        std::string reply = run_turn(model, config, tok, turn_ids, out, pos,
                                     temperature, top_p, rep_penalty, rep_window,
                                     max_tokens, rng, &drafter, draft_k, draft_n);
        // AGENT-1: tool-call loop.
        int iters = 0;
        while (max_iters > 0 && iters < max_iters) {
          size_t next_off = 0;
          agent::Call c = agloop.find_call(reply, next_off);
          if (c.name.empty()) break;
          std::fprintf(stderr, "[tool] %s %s\n", c.name.c_str(), c.args.c_str());
          std::string result = run_tool(tools, mcp_tools, c);
          std::string feed = std::string("Tool ") + c.name + " returned:\n" + result;
          std::vector<int> fids;
          if (tok.has_chat_template()) fids = tok.encode_text(tok.apply_chat_template(feed, system_prompt));
          else fids = tok.encode_text(feed);
          if (fids.empty()) break;
          reply = run_turn(model, config, tok, fids, out, pos,
                           temperature, top_p, rep_penalty, rep_window,
                           max_tokens, rng);
          ++iters;
        }
        std::printf("%s\n", postprocess_reply(reply, strip_think, json_shape).c_str());
        std::fflush(stdout);
      }
      return 0;
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
