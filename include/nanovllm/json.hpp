#pragma once

#include <cctype>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  bool is_null() const { return type == Type::Null; }
  bool is_object() const { return type == Type::Object; }
  bool is_array() const { return type == Type::Array; }
  bool contains(const std::string& k) const { return is_object() && obj.find(k) != obj.end(); }
  const Json& at(const std::string& k) const {
    static Json null_json;
    auto it = obj.find(k);
    return it == obj.end() ? null_json : it->second;
  }
  std::string as_string() const {
    if (type == Type::String) return str;
    throw std::runtime_error("json value is not a string");
  }
  double as_number() const {
    if (type == Type::Number) return num;
    if (type == Type::Bool) return b ? 1 : 0;
    throw std::runtime_error("json value is not a number");
  }
  int as_int() const { return static_cast<int>(as_number()); }
  int64_t as_int64() const { return static_cast<int64_t>(as_number()); }
  bool as_bool() const {
    if (type == Type::Bool) return b;
    if (type == Type::Number) return num != 0;
    if (type == Type::String) return str == "true";
    return false;
  }
  const std::vector<Json>& as_array() const { return arr; }
  const std::map<std::string, Json>& as_object() const { return obj; }
};

class JsonParser {
 public:
  static Json parse(const std::string& text) {
    JsonParser p(text);
    p.ws();
    Json v = p.value();
    p.ws();
    return v;
  }

 private:
  explicit JsonParser(const std::string& s) : s_(s) {}
  const std::string& s_;
  size_t i_ = 0;

  [[noreturn]] void fail(const char* msg) {
    throw std::runtime_error(std::string("JSON parse error near byte ") +
                             std::to_string(i_) + ": " + msg);
  }
  void ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  char peek() {
    ws();
    if (i_ >= s_.size()) fail("eof");
    return s_[i_];
  }
  void expect(char c) {
    ws();
    if (i_ >= s_.size() || s_[i_] != c) fail("unexpected character");
    ++i_;
  }
  Json value() {
    char c = peek();
    if (c == '{') return object();
    if (c == '[') return array();
    if (c == '"') { Json j; j.type = Json::Type::String; j.str = string(); return j; }
    if (c == 't' || c == 'f') return boolean();
    if (c == 'n') { expect_literal("null"); Json j; j.type = Json::Type::Null; return j; }
    return number();
  }
  void expect_literal(const char* lit) {
    size_t n = 0;
    while (lit[n]) ++n;
    if (i_ + n > s_.size() || s_.compare(i_, n, lit) != 0) fail("literal");
    i_ += n;
  }
  Json boolean() {
    Json j;
    j.type = Json::Type::Bool;
    if (peek() == 't') {
      expect_literal("true");
      j.b = true;
    } else {
      expect_literal("false");
      j.b = false;
    }
    return j;
  }
  Json number() {
    size_t start = i_;
    if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
    while (i_ < s_.size() && (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' ||
           s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '-' || s_[i_] == '+')) {
      ++i_;
    }
    if (start == i_) fail("number");
    Json j;
    j.type = Json::Type::Number;
    j.num = std::stod(s_.substr(start, i_ - start));
    return j;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (i_ < s_.size() && s_[i_] != '"') {
      char c = s_[i_++];
      if (c == '\\') {
        if (i_ >= s_.size()) fail("escape");
        char e = s_[i_++];
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          case 'u': {
            if (i_ + 4 > s_.size()) fail("unicode");
            unsigned int cp = std::stoul(s_.substr(i_, 4), nullptr, 16);
            i_ += 4;
            if (cp < 0x80) out.push_back(static_cast<char>(cp));
            else if (cp < 0x800) {
              out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
              out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
              out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
              out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
            break;
          }
          default: fail("escape");
        }
      } else {
        out.push_back(c);
      }
    }
    expect('"');
    return out;
  }
  Json array() {
    expect('[');
    Json j;
    j.type = Json::Type::Array;
    ws();
    if (i_ < s_.size() && s_[i_] == ']') { ++i_; return j; }
    while (true) {
      j.arr.push_back(value());
      char c = peek();
      if (c == ',') { ++i_; continue; }
      if (c == ']') { ++i_; break; }
      fail("array");
    }
    return j;
  }
  Json object() {
    expect('{');
    Json j;
    j.type = Json::Type::Object;
    ws();
    if (i_ < s_.size() && s_[i_] == '}') { ++i_; return j; }
    while (true) {
      ws();
      std::string k = string();
      expect(':');
      j.obj[k] = value();
      char c = peek();
      if (c == ',') { ++i_; continue; }
      if (c == '}') { ++i_; break; }
      fail("object");
    }
    return j;
  }
};
