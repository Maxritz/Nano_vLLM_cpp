#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
#endif

#include "nanovllm/llm_engine.hpp"
#include "nanovllm/json.hpp"

static void usage(const char* prog) {
  std::fprintf(stderr,
               "Usage: %s <model_dir> [prompt] [--tokens 1,2,3] [--max-tokens N] [--temperature T] [--max-model-len N] [--chat]\n"
               "       %s --bench <model_dir> [--num-seqs N] [--max-tokens N] [--max-model-len N]\n"
               "       %s <model_dir> --server [host port]   (OpenAI-compatible HTTP/SSE server)\n",
               prog, prog, prog);
}

static std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += static_cast<char>(c);
        }
    }
  }
  return o;
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

// ---- OpenAI-compatible HTTP/SSE server (Windows/Winsock, single-threaded) ----
// A request is handled start-to-finish before the next is accepted (the engine
// is single-threaded). Streaming uses SSE (`data: ...\n\n` per chunk + `[DONE]`);
// non-stream returns one JSON object.

struct ServerReq {
  long max_tokens = 64;
  double temperature = 0.6;
  bool stream = false;
  bool chat = false;
  std::string prompt;
};

static ServerReq parse_req(const std::string& body, const std::string& path) {
  ServerReq r;
  r.chat = (path == "/v1/chat/completions");
  Json j = JsonParser::parse(body);
  if (j.contains("max_tokens")) r.max_tokens = j.at("max_tokens").as_int();
  if (j.contains("temperature")) r.temperature = j.at("temperature").as_number();
  if (j.contains("stream")) r.stream = j.at("stream").as_bool();
  if (r.chat) {
    std::string prompt;
    if (j.contains("messages") && j.at("messages").is_array()) {
      for (const auto& m : j.at("messages").as_array())
        prompt += (m.contains("role") ? m.at("role").as_string() : "user") +
                  std::string(": ") + (m.contains("content") ? m.at("content").as_string() : "") + "\n";
    }
    r.prompt = prompt;
  } else if (j.contains("prompt")) {
    const Json& p = j.at("prompt");
    if (p.is_array()) {
      const auto& a = p.as_array();
      r.prompt = a.empty() ? "" : a[0].as_string();
    } else {
      r.prompt = p.as_string();
    }
  }
  if (r.prompt.empty()) r.prompt = "Hello";
  return r;
}

static bool send_all(SOCKET c, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    int n = ::send(c, s.data() + off, static_cast<int>(s.size() - off), 0);
    if (n <= 0) return false;
    off += static_cast<size_t>(n);
  }
  return true;
}

static void respond_json(SOCKET c, int code, const std::string& body) {
  const char* reason = (code == 200) ? "OK" : (code == 400 ? "Bad Request" : "Internal Server Error");
  std::string resp = "HTTP/1.1 " + std::to_string(code) + " " + reason + "\r\n" +
                     "Content-Type: application/json\r\nConnection: close\r\n" +
                     "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  send_all(c, resp);
}

static std::string sse_chunk(const std::string& rid, long long created, const std::string& model,
                             const std::string& delta_inner, const std::string& finish) {
  std::string fr = finish.empty() ? "null" : ("\"" + finish + "\"");
  return "data: {\"id\":\"" + rid + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
         std::to_string(created) + ",\"model\":\"" + json_escape(model) +
         "\",\"choices\":[{\"index\":0,\"delta\":{" + delta_inner + "},\"finish_reason\":" + fr + "}]}\n\n";
}

struct GenResult {
  std::string finish = "length";
  int prompt_tokens = 0;
  int completion_tokens = 0;
};

static GenResult run_generation(LLMEngine& engine, const std::string& prompt, const SamplingParams& sp, bool chat,
                                const std::function<void(int, const std::string&)>& on_token) {
  GenResult res;
  res.prompt_tokens = static_cast<int>(engine.encode_text(prompt).size());
  const int eos = engine.eos_token();
  if (chat) engine.add_chat_request(prompt, sp);
  else engine.add_request(prompt, sp);
  while (!engine.is_finished()) {
    auto [outs, cnt, news] = engine.step();
    (void)outs; (void)cnt;
    for (const auto& pr : news) {
      int tok = pr.second;
      if (std::getenv("NV_PRINT_IDS")) std::fprintf(stderr, "%d ", tok);
      if (tok == eos) { res.finish = "stop"; return res; }
      std::string d = engine.decode_token(tok);
      ++res.completion_tokens;
      on_token(tok, d);
    }
  }
  return res;
}

static void handle_http(SOCKET c, LLMEngine& engine, const std::string& model_id) {
  char buf[8192];
  std::string req;
  while (req.find("\r\n\r\n") == std::string::npos) {
    int n = ::recv(c, buf, sizeof(buf), 0);
    if (n <= 0) return;
    req.append(buf, n);
    if (req.size() > 262144) { respond_json(c, 413, "{\"error\":{\"message\":\"payload too large\"}}"); return; }
  }
  const size_t he = req.find("\r\n\r\n");
  const std::string head = req.substr(0, he);
  std::string body = req.substr(he + 4);

  size_t cr = head.find("\r\n");
  std::string reqline = head.substr(0, cr == std::string::npos ? head.size() : cr);
  size_t sp1 = reqline.find(' '), sp2 = (sp1 == std::string::npos) ? std::string::npos : reqline.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) { respond_json(c, 400, "{\"error\":{\"message\":\"bad request\"}}"); return; }
  std::string method = reqline.substr(0, sp1);
  std::string path = reqline.substr(sp1 + 1, sp2 - sp1 - 1);
  size_t qpos = path.find('?');
  std::string ppath = qpos == std::string::npos ? path : path.substr(0, qpos);

  std::string low = head;
  std::transform(low.begin(), low.end(), low.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  int clen = 0;
  size_t pos = low.find("content-length:");
  if (pos != std::string::npos) {
    size_t vs = pos + 15, ve = low.find("\r\n", vs);
    clen = std::atoi(head.substr(vs, ve - vs).c_str());
  }
  while (static_cast<int>(body.size()) < clen) {
    int n = ::recv(c, buf, sizeof(buf), 0);
    if (n <= 0) return;
    body.append(buf, n);
    if (static_cast<int>(body.size()) >= clen) break;
  }
  body.resize(static_cast<size_t>(clen));

  if (method == "GET" && (ppath == "/v1/models" || ppath == "/models")) {
    std::string b = "{\"object\":\"list\",\"data\":[{\"id\":\"" + json_escape(model_id) +
                  "\",\"object\":\"model\",\"created\":" + std::to_string(static_cast<long>(std::time(nullptr))) +
                  ",\"owned_by\":\"nanollm\"}]}";
    respond_json(c, 200, b);
    return;
  }
  if (method != "POST") { respond_json(c, 404, "{\"error\":{\"message\":\"not found\"}}"); return; }
  if (ppath != "/v1/chat/completions" && ppath != "/v1/completions") {
    respond_json(c, 404, "{\"error\":{\"message\":\"not found\"}}");
    return;
  }

  // Tolerate a leading UTF-8 BOM emitted by some clients/editors.
  if (body.size() >= 3 && (unsigned char)body[0] == 0xEF && (unsigned char)body[1] == 0xBB &&
      (unsigned char)body[2] == 0xBF)
    body = body.substr(3);

  ServerReq r;
  try { r = parse_req(body, ppath); }
  catch (const std::exception& e) {
    respond_json(c, 400, std::string("{\"error\":{\"message\":\"") + json_escape(e.what()) + "\"}}");
    return;
  }

  const long long created = std::time(nullptr);
  const std::string rid = "chatcmpl-" + std::to_string((long long)(created * 1000) + ((uintptr_t)&r & 0xffff));
  const std::string& model = model_id;
  SamplingParams sp; sp.max_tokens = static_cast<int>(r.max_tokens); sp.temperature = r.temperature;

  if (r.stream) {
    std::string hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
                      "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if (!send_all(c, hdr)) return;
    const std::string field = r.chat ? "content" : "text";
    try {
      // OpenAI spec: the first chat chunk is role-only, content arrives in the next chunk.
      if (r.chat) send_all(c, sse_chunk(rid, created, model, "\"role\":\"assistant\"", ""));
      GenResult res = run_generation(engine, r.prompt, sp, r.chat,
        [&](int, const std::string& delta) {
          send_all(c, sse_chunk(rid, created, model, "\"" + field + "\":\"" + json_escape(delta) + "\"", ""));
        });
      send_all(c, sse_chunk(rid, created, model, "", res.finish));
      send_all(c, "data: [DONE]\n\n");
    } catch (const std::exception& e) {
      send_all(c, "event: error\ndata: {\"error\":{\"message\":\"" + json_escape(e.what()) + "\"}}\n\n");
    }
    return;
  }

  std::string text;
  try {
    GenResult res = run_generation(engine, r.prompt, sp, r.chat,
                                   [&](int, const std::string& delta) { text += delta; });
    std::string obj;
    if (r.chat) {
      obj = "{\"id\":\"" + rid + "\",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) +
            ",\"model\":\"" + json_escape(model) + "\","
            "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(text) +
            "\"},\"finish_reason\":\"" + res.finish + "\"}],"
            "\"usage\":{\"prompt_tokens\":" + std::to_string(res.prompt_tokens) +
            ",\"completion_tokens\":" + std::to_string(res.completion_tokens) +
            ",\"total_tokens\":" + std::to_string(res.prompt_tokens + res.completion_tokens) + "}}";
    } else {
      obj = "{\"id\":\"" + rid + "\",\"object\":\"text_completion\",\"created\":" + std::to_string(created) +
            ",\"model\":\"" + json_escape(model) + "\","
            "\"choices\":[{\"text\":\"" + json_escape(text) + "\",\"index\":0,\"finish_reason\":\"" + res.finish + "\"}],"
            "\"usage\":{\"prompt_tokens\":" + std::to_string(res.prompt_tokens) +
            ",\"completion_tokens\":" + std::to_string(res.completion_tokens) +
            ",\"total_tokens\":" + std::to_string(res.prompt_tokens + res.completion_tokens) + "}}";
    }
    respond_json(c, 200, obj);
  } catch (const std::exception& e) {
    respond_json(c, 500, std::string("{\"error\":{\"message\":\"") + json_escape(e.what()) + "\"}}");
  }
}

static int run_server(const std::string& model_dir, int max_model_len, int num_kv_blocks, double expert_budget, const std::string& host, int port) {
#ifdef _WIN32
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
    std::fprintf(stderr, "WSAStartup failed\n");
    return 1;
  }
#endif
  LLMEngine engine(model_dir, max_model_len, num_kv_blocks, expert_budget);
  std::string model_id = model_dir;
  if (!model_id.empty() && model_id.back() != '/' && model_id.back() != '\\') {
    size_t slash = model_id.find_last_of("/\\");
    model_id = (slash == std::string::npos) ? model_id : model_id.substr(slash + 1);
  }
  SOCKET s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET) { std::fprintf(stderr, "socket failed\n"); return 1; }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<u_short>(port));
  ::InetPton(AF_INET, host.c_str(), &addr.sin_addr);
  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
    std::fprintf(stderr, "bind failed on %s:%d\n", host.c_str(), port);
    return 1;
  }
  ::listen(s, 8);
  std::fprintf(stderr, "OpenAI-compatible server on http://%s:%d  (model=%s)\n", host.c_str(), port, model_id.c_str());
  std::fflush(stderr);
  while (true) {
    SOCKET client = ::accept(s, nullptr, nullptr);
    if (client == INVALID_SOCKET) continue;
    int on = 1;
    ::setsockopt(client, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&on), sizeof(on));
    handle_http(client, engine, model_id);
    ::closesocket(client);
  }
}

// ---- CLI / bench ----

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
    bool server_mode = false;
    std::string server_host = "127.0.0.1";
    int server_port = 8080;
    int bench_seqs = 4;
    int num_kv_blocks = 0;  // 0 = size from max_model_len
    double expert_budget = 0.0;  // 0 = auto (45% of free VRAM)

    for (size_t i = 0; i < args.size(); ++i) {
      const std::string& a = args[i];
      if (a == "--tokens" && i + 1 < args.size()) prompt_ids = parse_ids(args[++i]);
      else if (a == "--max-tokens" && i + 1 < args.size()) max_tokens = std::stoi(args[++i]);
      else if (a == "--temperature" && i + 1 < args.size()) temperature = std::stod(args[++i]);
      else if (a == "--max-model-len" && i + 1 < args.size()) max_model_len = std::stoi(args[++i]);
      else if (a == "--num-kv-blocks" && i + 1 < args.size()) num_kv_blocks = std::stoi(args[++i]);
      else if (a == "--expert-budget" && i + 1 < args.size()) expert_budget = std::stod(args[++i]);
      else if (a == "--bench") bench = true;
      else if (a == "--chat") chat = true;
      else if (a == "--num-seqs" && i + 1 < args.size()) bench_seqs = std::stoi(args[++i]);
      else if (a == "--server") {
        server_mode = true;
        if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) server_host = args[++i];
        if (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) server_port = std::stoi(args[++i]);
      }
      else if (a == "--host" && i + 1 < args.size()) server_host = args[++i];
      else if (a == "--port" && i + 1 < args.size()) server_port = std::stoi(args[++i]);
      else if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
      else if (model_dir.empty()) model_dir = a;
      else if (prompt.empty()) prompt = a;
      else { usage(argv[0]); return 1; }
    }
    if (model_dir.empty()) {
      usage(argv[0]);
      return 1;
    }
    if (server_mode) return run_server(model_dir, max_model_len, num_kv_blocks, expert_budget, server_host, server_port);

    LLMEngine engine(model_dir, max_model_len, num_kv_blocks, expert_budget);
    size_t free_mem = 0, total_mem = 0;
    HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
    std::fprintf(stderr, "VRAM after load: free %.1f GB / %.1f GB\n", free_mem / 1e9, total_mem / 1e9);
    if (std::getenv("NANO_DEBUG")) {
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
      int vocab = engine.vocab_size();
      int span = std::max(1, vocab - 2);
      for (int i = 0; i < bench_seqs; ++i) {
        std::vector<int> ids(64);
        for (int t = 0; t < 64; ++t) ids[t] = 1 + ((i * 37 + t) % span);
        engine.add_request(ids, sp);
      }
      double pp_secs = 0.0, tg_secs = 0.0;
      long long pp_toks = 0, gen = 0;
      while (!engine.is_finished()) {
        auto t0 = std::chrono::steady_clock::now();
        auto [outs_, cnt, news_] = engine.step();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (cnt > 0) {
          pp_toks += cnt;
          pp_secs += dt;
        } else if (cnt < 0) {
          gen += -cnt;
          tg_secs += dt;
        }
        (void)outs_; (void)news_;
      }
      std::printf("total_seq: %d  prompt_tokens: %lld  generated_tokens: %lld\n", bench_seqs, pp_toks, gen);
      if (pp_secs > 0.0) std::printf("PP: %lld tokens / %.3fs = %.1f tok/s\n", pp_toks, pp_secs, pp_toks / pp_secs);
      if (tg_secs > 0.0 && gen > 0) std::printf("decode: %lld tokens / %.3fs = %.1f tok/s\n", gen, tg_secs, gen / tg_secs);
    } else if (!prompt_ids.empty()) {
      outputs = engine.generate(std::vector<std::vector<int>>{prompt_ids}, std::vector<SamplingParams>{sp});
    } else if (chat) {
      outputs = engine.generate_chat(std::vector<std::string>{prompt.empty() ? "Hello" : prompt}, std::vector<SamplingParams>{sp});
    } else {
      outputs = engine.generate(std::vector<std::string>{prompt.empty() ? "Hello" : prompt}, sp);
    }

    if (!bench) {
      const bool dbg = std::getenv("NANO_DEBUG") != nullptr;
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
