#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "nanovllm/json.hpp"

struct SpecialToken {
  int id = -1;
  std::string content;
};

class Tokenizer {
 public:
  bool load(const std::string& model_dir);
  void set_fallback_vocab_size(int n) { fallback_vocab_size_ = n; }
  std::vector<int> encode_text(const std::string& text) const;
  std::string decode_tokens(const std::vector<int>& ids) const;
  int encode_special(const std::string& special) const;
  bool has_chat_template() const { return !chat_template_.empty(); }
  std::string apply_chat_template(const std::string& prompt, const std::string& system = "") const;
  int eos_token_id() const { return eos_id_; }
  int vocab_size() const { return static_cast<int>(id_to_piece_.size()); }
  bool loaded() const { return loaded_; }
  // True when the tokenizer uses SPM-Unigram Viterbi segmentation instead of
  // BPE merges (tokenizer.json type Unigram, or merges absent + scores present;
  // GGUF tokenizer.ggml.model "unigram"/"llama" without merges).
  bool is_unigram() const { return unigram_; }
  // True when the model's chat template auto-injects a default system turn
  // (Qwen2-era ChatML templates); tool-gated templates return false.
  bool chat_auto_system() const { return chat_auto_system_; }

 private:
  bool load_from_gguf(const std::string& model_dir);

  struct Piece {
    std::string s;
    uint32_t code = 0;
  };

  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_piece_;
  std::unordered_map<std::string, int> merge_rank_;
  std::unordered_map<std::string, int> special_id_;
  std::vector<uint32_t> byte_to_codepoint_;
  std::map<uint32_t, unsigned char> codepoint_to_byte_;
  int eos_id_ = -1;
  int bos_id_ = -1;   // GGUF tokenizer.ggml.bos_token_id
  bool add_bos_ = false;  // prepend bos at encode (llama.cpp: true for llama-arch when key absent)
  std::string chat_template_;  // GGUF tokenizer.chat_template (empty = none)
  int fallback_vocab_size_ = 0;
  bool loaded_ = false;
  bool chat_auto_system_ = false;

  // TOK-1 (SPM-Unigram Viterbi): true when merges are absent and segmentation
  // must use vocab scores (tokenizer.json Unigram / GGUF "unigram"/"llama").
  bool unigram_ = false;
  std::vector<float> piece_score_;  // parallel to id_to_piece_; 0 when unknown
  std::vector<char> piece_usable_;  // Viterbi candidates (normal, non-special)
  std::vector<int> byte_piece_id_;  // 256 entries, byte -> piece id (-1 absent)
  std::vector<int> piece_type_;  // GGUF token_type; empty = unknown (all normal)
  bool have_piece_types_ = false;
  size_t max_piece_bytes_ = 0;

  // TOK-2 (pre_tokenizer config from tokenizer.json). Empty = absent/unknown ->
  // built-in default split.
  enum class PreKind { ByteLevel, SplitWS, SplitChar, Regex };
  struct PreStage {
    PreKind kind = PreKind::ByteLevel;
    std::string ch;  // SplitChar separator (single char)
    std::string pattern;   // Regex (ECMAScript, POSIX classes OK; \p{...} degrades)
    std::string behavior = "Isolated";  // Split behavior passthrough
    bool add_prefix_space = false;
  };
  std::vector<PreStage> pretokenizer_;
  bool pretok_warned_ = false;

  std::string byte_to_codepoint_utf8(unsigned char b) const;
  std::string bytes_to_byte_level(const std::string& utf8_bytes) const;
  std::string byte_level_to_bytes(const std::string& text) const;
  std::vector<std::string> pretokenize(const std::string& text) const;
  std::vector<std::string> pretokenize_raw(const std::string& text) const;
  std::vector<std::string> default_split(const std::string& text) const;
  std::vector<int> encode_piece(const std::string& piece) const;
  std::vector<int> encode_unigram(const std::string& raw_piece) const;
  void parse_pretokenizer(const Json& root);
  bool collect_pre_stages(const Json& j, std::vector<PreStage>& out);
  void note_pretok_fallback(const std::string& detail);
  void finalize_pieces();  // builds piece_usable_/byte map/max len after load
};
