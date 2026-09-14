#include "nanovllm/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "nanovllm/gguf.hpp"
#include "nanovllm/json.hpp"

static std::string encode_utf8(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

static bool chat_auto_injects_system(const std::string& tmpl) {
  if (tmpl.empty()) return false;
  if (tmpl.find("!= 'system'") != std::string::npos ||
      tmpl.find("!= \"system\"") != std::string::npos)
    return true;
  size_t pos = 0, count = 0;
  while ((pos = tmpl.find("You are", pos)) != std::string::npos) { ++count; pos += 7; }
  return count >= 2;
}

static std::vector<uint32_t> decode_utf8(const std::string& s) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    uint32_t cp = 0;
    int extra = 0;
    if (c < 0x80) {
      cp = c;
    } else if ((c & 0xE0) == 0xC0) {
      cp = c & 0x1F;
      extra = 1;
    } else if ((c & 0xF0) == 0xE0) {
      cp = c & 0x0F;
      extra = 2;
    } else if ((c & 0xF8) == 0xF0) {
      cp = c & 0x07;
      extra = 3;
    } else {
      cp = '?';
    }
    ++i;
    for (int k = 0; k < extra && i < s.size(); ++k, ++i) {
      cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
    }
    out.push_back(cp);
  }
  return out;
}

static std::vector<std::string> utf8_chars(const std::string& s) {
  std::vector<std::string> chars;
  for (uint32_t cp : decode_utf8(s)) chars.push_back(encode_utf8(cp));
  return chars;
}

static bool is_ascii_letter(uint32_t cp) {
  return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}
static bool is_ascii_digit(uint32_t cp) {
  return cp >= '0' && cp <= '9';
}
static bool is_ascii_space(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
}
static uint32_t lower_cp(uint32_t c) {
  return (c >= 'A' && c <= 'Z') ? c + 32 : c;
}

static std::string merge_key(const std::string& a, const std::string& b) {
  return a + '\x1F' + b;
}

bool Tokenizer::load(const std::string& model_dir) {
  loaded_ = false;
  vocab_.clear();
  id_to_piece_.clear();
  merge_rank_.clear();
  special_id_.clear();
  byte_to_codepoint_.assign(256, 0);
  codepoint_to_byte_.clear();
  eos_id_ = -1;

  byte_to_codepoint_.assign(256, 0);
  codepoint_to_byte_.clear();
  int remaining_idx = 0;
  for (int b = 0; b < 256; ++b) {
    bool safe = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b == 174 || b == 248;
    uint32_t cp;
    if (safe) {
      cp = static_cast<uint32_t>(b);
    } else {
      cp = static_cast<uint32_t>(256 + remaining_idx);
      ++remaining_idx;
    }
    byte_to_codepoint_[b] = cp;
    codepoint_to_byte_[cp] = static_cast<unsigned char>(b);
  }

  std::string path = model_dir + "/tokenizer.json";
  std::ifstream f(path, std::ios::binary);
  if (!f) return load_from_gguf(model_dir);
  std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  Json root = JsonParser::parse(text);

  if (root.contains("added_tokens")) {
    for (const auto& t : root.at("added_tokens").as_array()) {
      std::string content = t.at("content").as_string();
      int id = t.at("id").as_int();
      special_id_[content] = id;
      vocab_[content] = id;
      if (static_cast<int>(id_to_piece_.size()) <= id) id_to_piece_.resize(id + 1);
      id_to_piece_[id] = content;
    }
  }

  const Json& model = root.at("model");
  if (model.contains("vocab")) {
    for (const auto& kv : model.at("vocab").as_object()) {
      const std::string& piece = kv.first;
      int id = kv.second.as_int();
      vocab_[piece] = id;
      if (static_cast<int>(id_to_piece_.size()) <= id) id_to_piece_.resize(id + 1);
      id_to_piece_[id] = piece;
    }
  }

  if (model.contains("merges")) {
    const Json& merges = model.at("merges");
    if (merges.is_array()) {
      int rank = 0;
      for (const auto& m : merges.as_array()) {
        std::string a, b;
        if (m.is_array() && m.arr.size() >= 2) {
          a = m.arr[0].as_string();
          b = m.arr[1].as_string();
        } else {
          std::string line = m.as_string();
          size_t sp = line.find(' ');
          if (sp == std::string::npos) continue;
          a = line.substr(0, sp);
          b = line.substr(sp + 1);
        }
        merge_rank_[merge_key(a, b)] = rank++;
      }
    }
  }

  if (getenv("NANO_DEBUG")) {
    std::fprintf(stderr, "tokenizer vocab=%zu merges=%zu specials=%zu\n", vocab_.size(), merge_rank_.size(), special_id_.size());
    std::fflush(stderr);
  }
  auto set_eos = [&](const std::string& s) {
    auto it = special_id_.find(s);
    if (it != special_id_.end()) eos_id_ = it->second;
  };
  set_eos("<|endoftext|>");
  if (eos_id_ < 0) set_eos("<|im_end|>");
  std::string tc_path = model_dir + "/tokenizer_config.json";
  std::ifstream tc(tc_path, std::ios::binary);
  if (tc) {
    std::string t((std::istreambuf_iterator<char>(tc)), std::istreambuf_iterator<char>());
    Json root = JsonParser::parse(t);
    if (root.contains("chat_template")) {
      const Json& tmpl = root.at("chat_template");
      if (tmpl.is_string()) {
        chat_auto_system_ = chat_auto_injects_system(tmpl.as_string());
      }
    }
  }
  loaded_ = true;
  return true;
}

bool Tokenizer::load_from_gguf(const std::string& model_dir) {
  // No tokenizer.json: recover vocab + merges from the model's .gguf metadata.
  std::string gguf = find_gguf_path(model_dir);
  if (gguf.empty()) return false;
  GGUFLoader g;
  g.add_file(gguf);
  if (!g.meta_str_array("tokenizer.ggml.tokens", id_to_piece_)) return false;
  vocab_.reserve(id_to_piece_.size());
  for (size_t i = 0; i < id_to_piece_.size(); ++i) vocab_[id_to_piece_[i]] = static_cast<int>(i);

  std::vector<std::string> merges;
  if (g.meta_str_array("tokenizer.ggml.merges", merges)) {
    merge_rank_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
      size_t sp = merges[i].find(' ');
      if (sp == std::string::npos) continue;
      merge_rank_[merge_key(merges[i].substr(0, sp), merges[i].substr(sp + 1))] = static_cast<int>(i);
    }
  }

  // Qwen-style specials: token_type values 1=normal; add_imend/start etc. are CONTROL/USER_DEFINED.
  const int TYPE_CONTROL = 3, TYPE_USER_DEFINED = 4;
  std::vector<int32_t> types;
  if (g.meta_i32_array("tokenizer.ggml.token_type", types)) {
    for (size_t i = 0; i < types.size() && i < id_to_piece_.size(); ++i) {
      if (types[i] == TYPE_CONTROL || types[i] == TYPE_USER_DEFINED) special_id_[id_to_piece_[i]] = static_cast<int>(i);
    }
  }
  // last-resort specials by name (some Qwen GGUFs mark everything NORMAL)
  for (const char* s : {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_sep|>"}) {
    auto it = vocab_.find(s);
    if (it != vocab_.end()) special_id_.try_emplace(s, it->second);
  }
  uint32_t eos_id = 0;
  if (g.meta_u32("tokenizer.ggml.eos_token_id", eos_id)) eos_id_ = static_cast<int>(eos_id);
  if (eos_id_ < 0) {
    auto it = special_id_.find("<|endoftext|>");
    if (it != special_id_.end()) eos_id_ = it->second;
  }
  if (eos_id_ < 0) {
    auto it = special_id_.find("<|im_end|>");
    if (it != special_id_.end()) eos_id_ = it->second;
  }
  if (getenv("NANO_DEBUG")) {
    std::fprintf(stderr, "gguf tokenizer vocab=%zu merges=%zu specials=%zu eos=%d\n", id_to_piece_.size(),
                 merge_rank_.size(), special_id_.size(), eos_id_);
    std::fflush(stderr);
  }
  std::string tmpl;
  bool has_tmpl = g.meta_str("tokenizer.chat_template", tmpl);
  chat_auto_system_ = has_tmpl ? chat_auto_injects_system(tmpl) : false;
  loaded_ = true;
  return true;
}

int Tokenizer::encode_special(const std::string& special) const {
  auto it = special_id_.find(special);
  if (it == special_id_.end()) {
    auto vt = vocab_.find(special);
    return vt == vocab_.end() ? -1 : vt->second;
  }
  return it->second;
}

std::string Tokenizer::byte_to_codepoint_utf8(unsigned char b) const {
  return encode_utf8(byte_to_codepoint_[b]);
}

std::string Tokenizer::bytes_to_byte_level(const std::string& utf8_bytes) const {
  std::string out;
  for (unsigned char b : utf8_bytes) out += byte_to_codepoint_utf8(b);
  return out;
}

std::string Tokenizer::byte_level_to_bytes(const std::string& text) const {
  std::string out;
  for (uint32_t cp : decode_utf8(text)) {
    auto it = codepoint_to_byte_.find(cp);
    if (it != codepoint_to_byte_.end()) out.push_back(static_cast<char>(it->second));
    else out += encode_utf8(cp);
  }
  return out;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
  std::vector<std::string> raw;
  auto cps = decode_utf8(text);
  size_t i = 0;
  while (i < cps.size()) {
    if (cps[i] == '\'' && i + 1 < cps.size()) {
      const uint32_t contractions[][2] = {{'s',0},{'t',0},{'r','e'},{'v','e'},{'m',0},{'l','l'},{'d',0}};
      bool matched = false;
      for (const auto& c : contractions) {
        if (lower_cp(cps[i + 1]) == c[0]) {
          int take = c[1] == 0 ? 2 : 3;
          if (c[1] != 0 && i + 2 >= cps.size()) continue;
          if (c[1] != 0 && lower_cp(cps[i + 2]) != c[1]) continue;
          std::string s;
          for (int k = 0; k < take; ++k) s += encode_utf8(cps[i + k]);
          raw.push_back(s);
          i += take;
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }

    bool grouped = false;
    if (i + 1 < cps.size() && is_ascii_letter(cps[i + 1]) && cps[i] != '\r' && cps[i] != '\n') {
      size_t j = i + 1;
      while (j < cps.size() && is_ascii_letter(cps[j])) ++j;
      std::string s;
      for (size_t k = i; k < j; ++k) s += encode_utf8(cps[k]);
      raw.push_back(s);
      i = j;
      grouped = true;
    }

    if (!grouped) {
      size_t j = i;
      if (is_ascii_space(cps[j])) {
        while (j < cps.size() && is_ascii_space(cps[j])) ++j;
      } else if (is_ascii_letter(cps[j])) {
        ++j;
        while (j < cps.size() && is_ascii_letter(cps[j])) ++j;
      } else if (is_ascii_digit(cps[j])) {
        int count = 0;
        while (j < cps.size() && is_ascii_digit(cps[j]) && count < 3) {
          ++j;
          ++count;
        }
      } else {
        ++j;
      }
      std::string s;
      for (size_t k = i; k < j; ++k) s += encode_utf8(cps[k]);
      raw.push_back(s);
      i = j;
    }
  }

  std::vector<std::string> out;
  for (const auto& piece : raw) out.push_back(bytes_to_byte_level(piece));
  return out;
}

std::vector<int> Tokenizer::encode_piece(const std::string& piece) const {
  auto parts = utf8_chars(piece);
  if (parts.empty()) return {};

  while (parts.size() > 1) {
    int best_rank = -1;
    size_t best_i = 0;
    std::string best_a, best_b;
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      std::string key = merge_key(parts[i], parts[i + 1]);
      auto it = merge_rank_.find(key);
      if (it != merge_rank_.end() && (best_rank < 0 || it->second < best_rank)) {
        best_rank = it->second;
        best_i = i;
        best_a = parts[i];
        best_b = parts[i + 1];
      }
    }
    if (best_rank < 0) {
      if (getenv("NANO_DEBUG")) {
        std::fprintf(stderr, "no merge for parts=%zu", parts.size());
        for (auto& p : parts) std::fprintf(stderr, " [%s]", p.c_str());
        std::fprintf(stderr, "\n");
        if (parts.size() >= 2) {
          std::fprintf(stderr, "key first len %zu char %d\n", merge_key(parts[0], parts[1]).size(),
                       (int)(unsigned char)merge_key(parts[0], parts[1])[1]);
        }
        std::fflush(stderr);
      }
      break;
    }
    std::string merged = parts[best_i] + parts[best_i + 1];
    if (getenv("NANO_DEBUG")) {
      std::fprintf(stderr, "merge rank %d '%s'+'%s'->'%s'\n", best_rank, best_a.c_str(), best_b.c_str(),
                   merged.c_str());
      std::fflush(stderr);
    }
    parts[best_i] = merged;
    parts.erase(parts.begin() + best_i + 1);
  }

  std::vector<int> ids;
  for (const auto& p : parts) {
    auto it = vocab_.find(p);
    if (it != vocab_.end()) {
      ids.push_back(it->second);
    } else if (!p.empty()) {
      for (uint32_t cp : decode_utf8(p)) {
        auto single = vocab_.find(encode_utf8(cp));
        if (single != vocab_.end()) ids.push_back(single->second);
      }
    }
  }
  return ids;
}

std::vector<int> Tokenizer::encode_text(const std::string& text) const {
  if (!loaded_) {
    std::vector<int> ids;
    int mod = fallback_vocab_size_ > 0 ? fallback_vocab_size_ : 256;
    for (unsigned char b : text) ids.push_back(static_cast<int>(b) % mod);
    return ids;
  }

  std::vector<std::pair<std::string, int>> specials;
  for (const auto& kv : special_id_) specials.push_back({kv.first, kv.second});
  std::sort(specials.begin(), specials.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  std::vector<int> ids;
  size_t i = 0;
  while (i < text.size()) {
    bool found = false;
    for (const auto& sp : specials) {
      if (i + sp.first.size() <= text.size() && text.compare(i, sp.first.size(), sp.first) == 0) {
        ids.push_back(sp.second);
        i += sp.first.size();
        found = true;
        break;
      }
    }
    if (found) continue;
    size_t j = i + 1;
    while (j < text.size()) {
      bool special_here = false;
      for (const auto& sp : specials) {
        if (j + sp.first.size() <= text.size() && text.compare(j, sp.first.size(), sp.first) == 0) {
          special_here = true;
          break;
        }
      }
      if (special_here) break;
      ++j;
    }
    for (const auto& piece : pretokenize(text.substr(i, j - i))) {
      auto part = encode_piece(piece);
      ids.insert(ids.end(), part.begin(), part.end());
    }
    i = j;
  }
  return ids;
}

std::string Tokenizer::decode_tokens(const std::vector<int>& ids) const {
  std::string out;
  if (!loaded_) {
    for (int id : ids) out.push_back(id >= 32 && id < 127 ? static_cast<char>(id) : '?');
    return out;
  }
  for (int id : ids) {
    if (id < 0 || id >= static_cast<int>(id_to_piece_.size())) continue;
    const std::string& piece = id_to_piece_[id];
    auto special = special_id_.find(piece);
    if (special == special_id_.end()) out += byte_level_to_bytes(piece);
  }
  return out;
}
