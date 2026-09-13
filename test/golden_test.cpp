#include "golden_harness.hpp"

#include <cstdio>
#include <string>

static void parse_ties(const char* t, std::vector<int>& want, std::vector<int>& alt) {
  want.clear();
  alt.clear();
  if (!t) return;
  for (const char* p = t; *p;) {
    want.push_back(atoi(p));
    while (*p && *p != ',' && *p != '|') ++p;
    if (*p == '|') {
      ++p;
      alt.push_back(atoi(p));
      while (*p && *p != ',') ++p;
    } else {
      alt.push_back(-1);
    }
    if (*p == ',') ++p;
  }
}

static int check_prefix(const std::string& name, const std::vector<int>& got, const std::vector<int>& want,
                        const std::vector<int>& alt, int min_match) {
  int n = 0, m = (int)std::min(got.size(), want.size());
  for (; n < m; ++n) {
    if (got[n] == want[n]) continue;
    if (n < (int)alt.size() && alt[n] >= 0 && got[n] == alt[n]) continue;
    break;
  }
  if (n >= min_match) {
    std::printf("PASS %s (match %d/%d)\n", name.c_str(), n, (int)want.size());
    return 0;
  }
  std::printf("FAIL %s (match %d/%d)\n  want:", name.c_str(), n, (int)want.size());
  for (size_t i = 0; i < want.size(); ++i) {
    if (i < alt.size() && alt[i] >= 0) std::printf(" %d|%d", want[i], alt[i]);
    else std::printf(" %d", want[i]);
  }
  std::printf("\n  got :");
  for (int t : got) std::printf(" %d", t);
  std::printf("\n");
  return 1;
}

int main(int argc, char** argv) {
  const char* model = std::getenv("GOLDEN_MODEL");
  if (!model) model = "G:/VLLM-Models/Qwen2.5-7B-Instruct";
  int failures = 0;

  // Expected completion tokens (completion-only). Override per model, e.g.
  // GOLDEN_TOKENS="29051,29051,29051,29051". "a|b" documents an exact tie.
  // Defaults keep the Qwen2.5-7B refs.
  std::vector<int> want_plain, alt_plain, want_chat, alt_chat;
  parse_ties(std::getenv("GOLDEN_TOKENS"), want_plain, alt_plain);
  if (want_plain.empty()) { want_plain = {12095, 13}; alt_plain = {-1, -1}; }
  parse_ties(std::getenv("GOLDEN_TOKENS_CHAT"), want_chat, alt_chat);
  if (want_chat.empty()) { want_chat = {95456, 0, 21193, 374, 264}; alt_chat = {-1, -1, -1, -1, -1}; }

  failures += check_prefix("qwen25_plain",
                           greedy_ids_text(model, "The capital of France is", (int)want_plain.size(), 512),
                           want_plain, alt_plain, (int)want_plain.size());

  const char* chat_prompt = std::getenv("GOLDEN_CHAT_PROMPT");
  if (!chat_prompt) chat_prompt = "Write a Python function to reverse a string";
  failures += check_prefix("qwen25_chat",
                           greedy_ids_chat(model, chat_prompt,
                                           (int)want_chat.size(), 512),
                           want_chat, alt_chat, (int)want_chat.size());

  const char* d = std::getenv("GOLDEN_MODEL_DISTILLED");
  if (d) {
    failures += check_prefix("distilled_chat",
                             greedy_ids_chat(d, "Write a Python function to reverse a string", 8, 512),
                             {36142, 46537, 2698, 10217}, {}, 3);
  }

  std::printf(failures == 0 ? "ALL GOLDEN TESTS PASS\n" : "GOLDEN TESTS FAILED\n");
  return failures;
}