#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <string>

namespace {
struct HipBoot {
  HipBoot() {
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    const char* env = std::getenv("HIP_PATH");
    std::string bin = (env && *env) ? std::string(env) + "\\bin" : std::string(NANOVLLM_HIP_BIN);
    std::replace(bin.begin(), bin.end(), '/', '\\');
    while (!bin.empty() && bin.back() == '\\') bin.pop_back();
    wchar_t w[4096];
    if (MultiByteToWideChar(CP_ACP, 0, bin.c_str(), -1, w, 4096) > 0) AddDllDirectory(w);
  }
};
HipBoot hip_boot_;
}  // namespace
#endif
