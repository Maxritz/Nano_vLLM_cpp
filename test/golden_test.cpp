#include "golden_harness.hpp"

#include <cstdio>
#include <string>

static int check_prefix(const std::string& name, const std::vector<int>& got, const std::vector<int>& want,
                        int min_match) {
  int n = prefix_match(got, want);
  if (n >= min_match) {
    std::printf("PASS %s (match %d/%d)\n", name.c_str(), n, (int)want.size());
    return 0;
  }
  std::printf("FAIL %s (match %d/%d)\n  want:", name.c_str(), n, (int)want.size());
  for (int t : want) std::printf(" %d", t);
  std::printf("\n  got :");
  for (int t : got) std::printf(" %d", t);
  std::printf("\n");
  return 1;
}

int main(int argc, char** argv) {
  const char* model = std::getenv("GOLDEN_MODEL");
  if (!model) model = "G:/VLLM-Models/Qwen2.5-7B-Instruct";
  int failures = 0;

  failures += check_prefix("qwen25_plain", greedy_ids_text(model, "The capital of France is", 8, 512), {12095, 13}, 2);

  failures += check_prefix("qwen25_chat",
                           greedy_ids_chat(model, "Write a Python function to reverse a string", 8, 512),
                           {95456, 0, 21193, 374, 264}, 4);

  const char* d = std::getenv("GOLDEN_MODEL_DISTILLED");
  if (d) {
    failures += check_prefix("distilled_chat",
                             greedy_ids_chat(d, "Write a Python function to reverse a string", 8, 512),
                             {36142, 46537, 2698, 10217}, 3);
  }

  std::printf(failures == 0 ? "ALL GOLDEN TESTS PASS\n" : "GOLDEN TESTS FAILED\n");
  return failures;
}