#pragma once
// SERV-1: minimal OpenAI-compatible HTTP server (Windows winsock2 only).
// Must be included BEFORE <windows.h> in the TU (winsock2 first).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <functional>
#include <thread>
#include <vector>
#include <sstream>
#include <iostream>
#include <cstring>
#include <cstdio>

#pragma comment(lib, "ws2_32.lib")

namespace minihttp {

struct ChatRequest {
  std::string model;
  std::string last_user;
  double temperature = 0.0;
  int max_tokens = 0;
};

using Handler = std::function<std::string(const ChatRequest&)>;

inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

inline std::string build_chat_response(const std::string& model, const std::string& content) {
  std::ostringstream ss;
  ss << "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n"
     << "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"created\":0,\"model\":\""
     << json_escape(model) << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
     << json_escape(content) << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}}";
  return ss.str();
}

inline std::string build_webui() {
  const char* html =
      "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
      "<title>nanovllm</title>"
      "<style>body{font-family:sans-serif;max-width:720px;margin:2em auto;background:#111;color:#ddd}"
      "#log{border:1px solid #444;min-height:300px;padding:1em;white-space:pre-wrap;margin-bottom:1em}"
      ".u{color:#8cf}.a{color:#afa}input{width:78%}button{width:20%}</style></head><body>"
      "<h3>nanovllm chat</h3><div id=\"log\"></div>"
      "<input id=\"q\" placeholder=\"message...\"/><button onclick=\"send()\">Send</button>"
      "<script>async function send(){"
      "const q=document.getElementById('q');const t=q.value;q.value='';"
      "const log=document.getElementById('log');"
      "log.innerHTML+='<span class=u>You: '+t+'</span>\\n';"
      "const r=await fetch('/v1/chat/completions',{method:'POST',"
      "headers:{'Content-Type':'application/json'},"
      "body:JSON.stringify({messages:[{role:'user',content:t}],max_tokens:128})});"
      "const j=await r.json();"
      "log.innerHTML+='<span class=a>AI: '+j.choices[0].message.content+'</span>\\n';"
      "log.scrollTop=log.scrollHeight;}</script></body></html>";
  std::ostringstream ss;
  ss << "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close\r\n\r\n" << html;
  return ss.str();
}

inline std::string build_models_response(const std::string& model) {
  std::ostringstream ss;
  ss << "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n"
     << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(model)
     << "\",\"object\":\"model\",\"owned_by\":\"nanovllm\"}]}";
  return ss.str();
}

inline std::string http_status(int code, const char* text) {
  std::ostringstream ss;
  ss << "HTTP/1.1 " << code << " " << text << "\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
  return ss.str();
}

inline std::string json_extract_string(const std::string& body, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t i = body.find(pat);
  if (i == std::string::npos) return "";
  i = body.find(':', i + pat.size());
  if (i == std::string::npos) return "";
  if (++i >= body.size()) return "";
  while (i < body.size() && (body[i] == ' ' || body[i] == '\t')) i++;
  if (i >= body.size() || body[i] != '"') return "";
  i++;
  std::string val;
  while (i < body.size() && body[i] != '"') {
    if (body[i] == '\\' && i + 1 < body.size()) { val += body[i + 1]; i += 2; }
    else val += body[i++];
  }
  return val;
}

inline double json_extract_double(const std::string& body, const std::string& key, double def) {
  std::string pat = "\"" + key + "\"";
  size_t i = body.find(pat);
  if (i == std::string::npos) return def;
  i = body.find(':', i + pat.size());
  if (i == std::string::npos) return def;
  try { return std::stod(body.substr(i + 1)); } catch (...) { return def; }
}

inline int json_extract_int(const std::string& body, const std::string& key, int def) {
  std::string pat = "\"" + key + "\"";
  size_t i = body.find(pat);
  if (i == std::string::npos) return def;
  i = body.find(':', i + pat.size());
  if (i == std::string::npos) return def;
  try { return std::stoi(body.substr(i + 1)); } catch (...) { return def; }
}

inline std::string json_extract_last_user(const std::string& body) {
  size_t pos = 0;
  std::string last;
  while (true) {
    size_t rp = body.find("\"role\"", pos);
    if (rp == std::string::npos) break;
    size_t c0 = body.find(':', rp + 6);
    if (c0 == std::string::npos) break;
    size_t v0 = c0 + 1;
    while (v0 < body.size() && (body[v0] == ' ' || body[v0] == '\t')) v0++;
    if (v0 >= body.size() || body[v0] != '"') { pos = v0; continue; }
    v0++;
    std::string role;
    while (v0 < body.size() && body[v0] != '"') {
      if (body[v0] == '\\' && v0 + 1 < body.size()) { role += body[v0 + 1]; v0 += 2; }
      else role += body[v0++];
    }
    if (role != "user") { pos = v0; continue; }
    size_t cp = body.find("\"content\"", v0);
    if (cp == std::string::npos) { pos = v0; continue; }
    size_t cc = body.find(':', cp + 9);
    if (cc == std::string::npos) { pos = v0; continue; }
    size_t cs = cc + 1;
    while (cs < body.size() && (body[cs] == ' ' || body[cs] == '\t')) cs++;
    if (cs >= body.size() || body[cs] != '"') { pos = v0; continue; }
    cs++;
    std::string content;
    while (cs < body.size() && body[cs] != '"') {
      if (body[cs] == '\\' && cs + 1 < body.size()) { content += body[cs + 1]; cs += 2; }
      else content += body[cs++];
    }
    last = content;
    pos = cs;
  }
  return last;
}

inline void send_all(SOCKET s, const std::string& data) {
  size_t sent = 0;
  while (sent < data.size()) {
    int n = send(s, data.data() + sent, (int)(data.size() - sent), 0);
    if (n <= 0) break;
    sent += (size_t)n;
  }
}

inline void handle_client(SOCKET client_sock, const std::string& model_name, Handler h) {
  std::string err;
  do {
    char buf[65536];
    int total = 0;
    bool got_headers = false;
    while (total < (int)sizeof(buf) - 1) {
      int n = recv(client_sock, buf + total, (int)sizeof(buf) - 1 - total, 0);
      if (n <= 0) break;
      total += n;
      for (int k = (total - n > 3 ? total - n - 3 : 0); k + 3 < total; ++k) {
        if (buf[k] == '\r' && buf[k + 1] == '\n' && buf[k + 2] == '\r' && buf[k + 3] == '\n') {
          got_headers = true;
          break;
        }
      }
      if (got_headers) break;
    }
    if (total <= 0 || !got_headers) { err = http_status(400, "Bad Request"); break; }
    buf[total] = '\0';
    std::string req(buf, (size_t)total);
    size_t line_end = req.find("\r\n");
    if (line_end == std::string::npos) { err = http_status(400, "Bad Request"); break; }
    std::istringstream rl(req.substr(0, line_end));
    std::string method, path, version;
    rl >> method >> path >> version;
    size_t header_end = req.find("\r\n\r\n");
    std::string headers = req.substr(0, header_end);
    std::string body;
    size_t cl_pos = headers.find("Content-Length:");
    if (cl_pos != std::string::npos) {
      int cl = 0;
      try { cl = std::stoi(headers.substr(cl_pos + 15)); } catch (...) {}
      // read remaining body bytes if not yet received
      size_t body_start = header_end + 4;
      while ((int)(req.size() - body_start) < cl && (int)req.size() < 65535) {
        char extra[4096];
        int n = recv(client_sock, extra, sizeof(extra), 0);
        if (n <= 0) break;
        req.append(extra, (size_t)n);
      }
      if (cl > 0 && cl <= 65536 && body_start + (size_t)cl <= req.size())
        body = req.substr(body_start, (size_t)cl);
    }
    std::string response;
    if (method == "GET" && (path == "/" || path == "/index.html")) {
      response = build_webui();
    } else if (method == "GET" && path == "/v1/models") {
      response = build_models_response(model_name);
    } else if (method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions")) {
      ChatRequest cr;
      cr.model = json_extract_string(body, "model");
      cr.temperature = json_extract_double(body, "temperature", -1.0);
      cr.max_tokens = json_extract_int(body, "max_tokens", 0);
      cr.last_user = (path == "/v1/chat/completions") ? json_extract_last_user(body)
                                                      : json_extract_string(body, "prompt");
      if (cr.last_user.empty()) { err = http_status(400, "Bad Request"); break; }
      try {
        response = build_chat_response(cr.model.empty() ? model_name : cr.model, h(cr));
      } catch (...) { err = http_status(500, "Internal Server Error"); break; }
    } else {
      err = (method == "GET" || method == "POST") ? http_status(404, "Not Found")
                                                  : http_status(405, "Method Not Allowed");
      break;
    }
    send_all(client_sock, response);
    closesocket(client_sock);
    return;
  } while (false);
  if (!err.empty()) send_all(client_sock, err);
  closesocket(client_sock);
}

inline void serve(int port, const std::string& model_name, Handler h) {
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return;
  SOCKET listen_sock = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_sock == INVALID_SOCKET) return;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons((u_short)port);
  inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (bind(listen_sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
    closesocket(listen_sock);
    return;
  }
  listen(listen_sock, 8);
  std::fprintf(stderr, "minihttp listening on 127.0.0.1:%d (model %s)\n",
               port, model_name.c_str());
  while (true) {
    SOCKET client = accept(listen_sock, nullptr, nullptr);
    if (client == INVALID_SOCKET) continue;
    std::thread(handle_client, client, model_name, h).detach();
  }
}

}  // namespace minihttp
