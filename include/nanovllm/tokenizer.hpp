#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

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
  int eos_token_id() const { return eos_id_; }
  int vocab_size() const { return static_cast<int>(id_to_piece_.size()); }
  bool loaded() const { return loaded_; }
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
  int fallback_vocab_size_ = 0;
  bool loaded_ = false;
  bool chat_auto_system_ = false;

  std::string byte_to_codepoint_utf8(unsigned char b) const;
  std::string bytes_to_byte_level(const std::string& utf8_bytes) const;
  std::string byte_level_to_bytes(const std::string& text) const;
  std::vector<std::string> pretokenize(const std::string& text) const;
  std::vector<int> encode_piece(const std::string& piece) const;
};
