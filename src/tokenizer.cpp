#include "nanovllm/tokenizer.hpp"
#include "nanovllm/chat_template.hpp"
#include "nanovllm/pre_tokenizer.hpp"

// TOK-2: apply an HF Split/Regex stage via the tested ptok implementation
// (POSIX classes translated; \p{...} patterns throw inside ptok and degrade
// to no-split for that piece instead of failing the load).
static std::vector<std::string> regex_split_piece(const std::string& pattern,
                                                  const std::string& behavior,
                                                  const std::string& text) {
  auto js = [](const std::string& x) { ptok::JVal v; v.t = ptok::JVal::STR; v.s = x; return v; };
  ptok::JVal pat; pat.t = ptok::JVal::OBJ; pat.obj = {{"Regex", js(pattern)}};
  ptok::JVal mk; mk.t = ptok::JVal::OBJ;
  mk.obj = {{"type", js("Split")}, {"pattern", pat}, {"behavior", js(behavior)}};
  try {
    return ptok::pre_split(mk, text);
  } catch (...) {
    return {text};
  }
}

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

// "<0x41>"-style SPM byte-fallback piece -> byte value. False for anything else.
static bool parse_byte_piece(const std::string& piece, int& byte_out) {
  if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' || (piece[2] != 'x' && piece[2] != 'X') ||
      piece[5] != '>') {
    return false;
  }
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  int hi = hex(piece[3]), lo = hex(piece[4]);
  if (hi < 0 || lo < 0) return false;
  byte_out = hi * 16 + lo;
  return true;
}

// SPM detokenize: U+2581 (E2 96 81) back to space; input is raw UTF-8.
static std::string spm_detokenize(const std::string& piece) {
  std::string out;
  out.reserve(piece.size());
  for (size_t i = 0; i < piece.size();) {
    if (i + 3 <= piece.size() && static_cast<unsigned char>(piece[i]) == 0xE2 &&
        static_cast<unsigned char>(piece[i + 1]) == 0x96 && static_cast<unsigned char>(piece[i + 2]) == 0x81) {
      out.push_back(' ');
      i += 3;
    } else {
      out.push_back(piece[i++]);
    }
  }
  return out;
}

bool Tokenizer::load(const std::string& model_dir) {
  loaded_ = false;
  vocab_.clear();
  id_to_piece_.clear();
  merge_rank_.clear();
  special_id_.clear();
  unigram_ = false;
  piece_score_.clear();
  piece_usable_.clear();
  byte_piece_id_.assign(256, -1);
  piece_type_.clear();
  have_piece_types_ = false;
  max_piece_bytes_ = 0;
  pretokenizer_.clear();
  pretok_warned_ = false;
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
  // Unigram vocab is an array of [piece, score] pairs (HF "Unigram" type).
  bool vocab_is_array = model.contains("vocab") && model.at("vocab").is_array();
  if (vocab_is_array) {
    int id = 0;
    for (const auto& e : model.at("vocab").as_array()) {
      if (!e.is_array() || e.as_array().size() < 2) continue;
      std::string piece = e.as_array()[0].as_string();
      float score = static_cast<float>(e.as_array()[1].as_number());
      vocab_[piece] = id;
      if (static_cast<int>(id_to_piece_.size()) <= id) id_to_piece_.resize(id + 1);
      id_to_piece_[id] = piece;
      if (static_cast<int>(piece_score_.size()) <= id) piece_score_.resize(id + 1, 0.0f);
      piece_score_[id] = score;
      ++id;
    }
  } else if (model.contains("vocab")) {
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

  // TOK-1: Unigram when explicitly typed, or merges absent with scores present.
  // (BPE models always carry merges, so the BPE path below stays byte-identical.)
  std::string model_type;
  if (model.contains("type") && model.at("type").is_string()) model_type = model.at("type").as_string();
  unigram_ = (model_type == "Unigram") || (merge_rank_.empty() && vocab_is_array);

  parse_pretokenizer(root);
  finalize_pieces();

  if (getenv("NANO_DEBUG")) {
    std::fprintf(stderr, "tokenizer vocab=%zu merges=%zu specials=%zu unigram=%d\n", vocab_.size(), merge_rank_.size(), special_id_.size(),
                 unigram_ ? 1 : 0);
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

  std::string tok_model;
  g.meta_str("tokenizer.ggml.model", tok_model);
  std::vector<float> scores;
  bool has_scores = g.meta_f32_array("tokenizer.ggml.scores", scores);
  if (has_scores) {
    piece_score_.assign(id_to_piece_.size(), 0.0f);
    for (size_t i = 0; i < scores.size() && i < piece_score_.size(); ++i) piece_score_[i] = scores[i];
  }

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
    have_piece_types_ = true;
    piece_type_.assign(types.begin(), types.end());
    for (size_t i = 0; i < types.size() && i < id_to_piece_.size(); ++i) {
      if (types[i] == TYPE_CONTROL || types[i] == TYPE_USER_DEFINED) special_id_[id_to_piece_[i]] = static_cast<int>(i);
    }
  }
  // TOK-1: SPM ("llama") and explicit "unigram" GGUFs carry scores but no
  // merges (verified on C:\models: Phi-3-mini, Tiny-LLM, tinyllama*). BPE
  // models (gpt2/gemma4) always have merges, keeping the BPE path identical.
  unigram_ = merge_rank_.empty() && (tok_model == "unigram" || tok_model == "llama" || has_scores);
  note_pretok_fallback("no pre_tokenizer config (GGUF metadata has none)");
  // last-resort specials by name (some Qwen GGUFs mark everything NORMAL)
  for (const char* s : {"<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_sep|>"}) {
    auto it = vocab_.find(s);
    if (it != vocab_.end()) special_id_.try_emplace(s, it->second);
  }
  finalize_pieces();
  uint32_t eos_id = 0;
  if (g.meta_u32("tokenizer.ggml.eos_token_id", eos_id)) eos_id_ = static_cast<int>(eos_id);
  uint32_t bid = 0;
  if (g.meta_u32("tokenizer.ggml.bos_token_id", bid)) bos_id_ = static_cast<int>(bid);
  if (const GGUFMetaValue* ab = g.meta("tokenizer.ggml.add_bos_token"))
    add_bos_ = ab->b;
  else
    add_bos_ = (tok_model == "llama");  // llama.cpp default when key absent
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
  if (has_tmpl) chat_template_ = tmpl;
  chat_auto_system_ = has_tmpl ? chat_auto_injects_system(tmpl) : false;
  loaded_ = true;
  return true;
}

std::string Tokenizer::apply_chat_template(const std::string& prompt, const std::string& system) const {
  if (chat_template_.empty()) return prompt;
  try {
    std::string eos;
    if (eos_id_ >= 0 && eos_id_ < static_cast<int>(id_to_piece_.size()))
      eos = id_to_piece_[(size_t)eos_id_];
    std::vector<std::pair<std::string, std::string>> conv;
    if (!system.empty()) conv.push_back({"system", system});
    conv.push_back({"user", prompt});
    std::vector<chtmpl::Msg> msgs;
    for (auto& kv : conv) msgs.push_back({kv.first, kv.second});
    return chtmpl::render(chat_template_, msgs, true, eos);
  } catch (...) {
    return prompt;  // malformed template: raw prompt, same as before
  }
}

void Tokenizer::note_pretok_fallback(const std::string& detail) {
  if (pretok_warned_) return;
  pretok_warned_ = true;
  std::fprintf(stderr, "tokenizer: %s; using built-in split\n", detail.c_str());
  std::fflush(stderr);
}

bool Tokenizer::collect_pre_stages(const Json& j, std::vector<PreStage>& out) {
  if (!j.is_object() || !j.contains("type") || !j.at("type").is_string()) return false;
  std::string t = j.at("type").as_string();
  if (t == "Sequence") {
    if (!j.contains("pretokenizers") || !j.at("pretokenizers").is_array()) return false;
    for (const auto& s : j.at("pretokenizers").as_array()) {
      if (!collect_pre_stages(s, out)) return false;
    }
    return true;
  }
  if (t == "ByteLevel") {
    PreStage s;
    s.kind = PreKind::ByteLevel;
    if (j.contains("add_prefix_space")) s.add_prefix_space = j.at("add_prefix_space").as_bool();
    out.push_back(s);
    return true;
  }
  if (t == "Split") {
    // Only Whitespace / single-char patterns; Regex and inverted splits fall back.
    if (j.contains("invert") && j.at("invert").as_bool()) return false;
    if (j.contains("behavior") && j.at("behavior").is_string() && j.at("behavior").as_string() == "Removed")
      return false;
    if (!j.contains("pattern")) return false;
    const Json& p = j.at("pattern");
    std::string pat;
    if (p.is_string()) {
      pat = p.as_string();
    } else if (p.is_object()) {
      if (p.contains("String") && p.at("String").is_string()) pat = p.at("String").as_string();
      else if (p.contains("string") && p.at("string").is_string()) pat = p.at("string").as_string();
      else if (p.contains("Regex") && p.at("Regex").is_string()) {
        PreStage s;
        s.kind = PreKind::Regex;
        s.pattern = p.at("Regex").as_string();
        if (j.contains("behavior") && j.at("behavior").is_string())
          s.behavior = j.at("behavior").as_string();
        out.push_back(s);
        return true;
      } else return false;  // unknown pattern object
    } else {
      return false;
    }
    PreStage s;
    if (pat == "Whitespace") {
      s.kind = PreKind::SplitWS;
    } else if (!pat.empty() && utf8_chars(pat).size() == 1) {
      s.kind = PreKind::SplitChar;
      s.ch = pat;
    } else {
      return false;
    }
    out.push_back(s);
    return true;
  }
  return false;
}

void Tokenizer::parse_pretokenizer(const Json& root) {
  pretokenizer_.clear();
  if (!root.contains("pre_tokenizer") || root.at("pre_tokenizer").is_null()) {
    note_pretok_fallback("no pre_tokenizer config");
    return;
  }
  std::vector<PreStage> stages;
  if (!collect_pre_stages(root.at("pre_tokenizer"), stages)) {
    pretokenizer_.clear();
    note_pretok_fallback("unsupported pre_tokenizer config");
    return;
  }
  pretokenizer_ = stages;
}

void Tokenizer::finalize_pieces() {
  size_t n = id_to_piece_.size();
  if (piece_score_.size() < n) piece_score_.resize(n, 0.0f);
  piece_usable_.assign(n, 0);
  byte_piece_id_.assign(256, -1);
  max_piece_bytes_ = 0;
  for (size_t i = 0; i < n; ++i) {
    const std::string& piece = id_to_piece_[i];
    if (piece.empty()) continue;
    int b = 0;
    if (parse_byte_piece(piece, b)) {
      byte_piece_id_[(size_t)b] = static_cast<int>(i);  // byte fallback only, never a Viterbi match
      continue;
    }
    if (special_id_.find(piece) != special_id_.end()) continue;
    if (have_piece_types_ && (i >= piece_type_.size() || piece_type_[i] != 1)) continue;  // 1 = NORMAL
    piece_usable_[i] = 1;
    if (piece.size() > max_piece_bytes_) max_piece_bytes_ = piece.size();
  }
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

static bool is_sep_byte(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// Split keeping separators: each separator run attaches to the following piece
// (GPT-2 style, so BPE merges on space-prefixed words still apply).
static std::vector<std::string> split_keep_ws(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    size_t j = i;
    while (j < n && is_sep_byte(s[j])) ++j;
    size_t k = j;
    while (k < n && !is_sep_byte(s[k])) ++k;
    if (k > j) {
      out.push_back(s.substr(i, k - i));
      i = k;
    } else {
      out.push_back(s.substr(i, j - i));
      i = j;
    }
  }
  return out;
}

static std::vector<std::string> split_keep_str(const std::string& s, const std::string& sep) {
  std::vector<std::string> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    size_t j = i;
    while (j < n && s.compare(j, sep.size(), sep) == 0) j += sep.size();
    size_t k = j;
    while (k < n && s.compare(k, sep.size(), sep) != 0) ++k;
    if (k > j) {
      out.push_back(s.substr(i, k - i));
      i = k;
    } else {
      out.push_back(s.substr(i, j - i));
      i = j;
    }
  }
  return out;
}

std::vector<std::string> Tokenizer::pretokenize_raw(const std::string& text) const {
  if (pretokenizer_.empty()) return default_split(text);
  std::string input = text;
  for (const auto& s : pretokenizer_) {
    if (s.kind == PreKind::ByteLevel && s.add_prefix_space) {
      if (!input.empty() && input[0] != ' ') input = " " + input;
      break;  // prefix-space applies once to the whole input
    }
  }
  std::vector<std::string> pieces = {input};
  for (const auto& s : pretokenizer_) {
    if (s.kind == PreKind::ByteLevel) continue;  // byte mapping happens in pretokenize()
    std::vector<std::string> next;
    for (const auto& p : pieces) {
      auto sp = (s.kind == PreKind::Regex) ? regex_split_piece(s.pattern, s.behavior, p)
                : ((s.kind == PreKind::SplitWS) ? split_keep_ws(p) : split_keep_str(p, s.ch));
      next.insert(next.end(), sp.begin(), sp.end());
    }
    pieces.swap(next);
  }
  return pieces;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
  auto raw = pretokenize_raw(text);
  if (unigram_) return raw;  // unigram works on raw UTF-8 (+ U+2581), not byte-level
  std::vector<std::string> out;
  out.reserve(raw.size());
  for (const auto& piece : raw) out.push_back(bytes_to_byte_level(piece));
  return out;
}

std::vector<std::string> Tokenizer::default_split(const std::string& text) const {
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
  for (const auto& piece : raw) out.push_back(piece);
  return out;
}

std::vector<int> Tokenizer::encode_unigram(const std::string& raw_piece) const {
  // SPM convention: space -> U+2581, then Viterbi best path over piece scores.
  // Byte-fallback pieces (<0x..>) only fire at dead ends (very low score), so
  // any real piece match wins and decode(encode(x)) == x holds.
  std::string s;
  s.reserve(raw_piece.size());
  for (char c : raw_piece) {
    if (c == ' ') s += "\xE2\x96\x81";
    else s.push_back(c);
  }
  size_t n = s.size();
  const double NEG = -1e100, BYTE_SCORE = -1e9;
  std::vector<double> best(n + 1, NEG);
  std::vector<int> blen(n + 1, -1), bid(n + 1, -1);
  best[0] = 0.0;
  for (size_t i = 0; i < n; ++i) {
    if (best[i] <= NEG / 2) continue;
    size_t maxl = std::min(max_piece_bytes_, n - i);
    for (size_t l = 1; l <= maxl; ++l) {
      auto it = vocab_.find(s.substr(i, l));
      if (it == vocab_.end()) continue;
      int id = it->second;
      if (id < 0 || static_cast<size_t>(id) >= piece_usable_.size() || !piece_usable_[static_cast<size_t>(id)])
        continue;
      double ns = best[i] + piece_score_[static_cast<size_t>(id)];
      if (ns > best[i + l]) {
        best[i + l] = ns;
        blen[i + l] = static_cast<int>(l);
        bid[i + l] = id;
      }
    }
    int fb = byte_piece_id_[static_cast<unsigned char>(s[i])];
    if (fb >= 0 && best[i] + BYTE_SCORE > best[i + 1]) {
      best[i + 1] = best[i] + BYTE_SCORE;
      blen[i + 1] = 1;
      bid[i + 1] = fb;
    }
  }
  std::vector<int> ids;
  size_t p = n;
  while (p > 0) {
    if (blen[p] <= 0) {
      --p;  // unreachable byte with no fallback representation: skip it
      continue;
    }
    ids.push_back(bid[p]);
    p -= static_cast<size_t>(blen[p]);
  }
  std::reverse(ids.begin(), ids.end());
  return ids;
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
    // SPM word-start rule: every word gets a meta-space prefix — the first piece of
    // a text span does too (nothing carried its leading ' '), but only if the text
    // before it did not already end in a space (else the space belongs to prev word).
    auto pieces = pretokenize(text.substr(i, j - i));
    bool need_meta = i == 0 || (j > 0 && j <= text.size() && text[j - 1] != ' ');
    for (const auto& piece0 : pieces) {
      const std::string* pp = &piece0;
      std::string metaed;
      if (unigram_ && need_meta && !piece0.empty() && piece0[0] != ' ') {
        metaed = " " + piece0;
        pp = &metaed;
      }
      need_meta = false;
      auto part = unigram_ ? encode_unigram(*pp) : encode_piece(*pp);
      ids.insert(ids.end(), part.begin(), part.end());
    }
    i = j;
  }
  if (add_bos_ && bos_id_ >= 0 && (ids.empty() || ids[0] != bos_id_))
    ids.insert(ids.begin(), bos_id_);
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
    if (special != special_id_.end()) continue;
    if (unigram_) {
      int b = 0;
      if (parse_byte_piece(piece, b)) {
        out.push_back(static_cast<char>(b));
        continue;
      }
      out += spm_detokenize(piece);
      continue;
    }
    out += byte_level_to_bytes(piece);
  }
  return out;
}
