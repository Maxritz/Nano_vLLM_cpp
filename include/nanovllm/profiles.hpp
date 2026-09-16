#pragma once

#include <string>
#include <vector>
#include <map>

namespace profile {

// A config profile unifies switches (like Edge0 prod presets). Header-only
// C++17, STL only.
struct Profile {
    std::string name;
    int max_model_len = 4096;
    int max_tokens = 64;
    double temperature = 0.0;
    double top_p = 1.0;
    double rep_penalty = 1.0;
    int rep_window = 64;
    int kvcache_block_size = 256;
    bool use_chat = false;
    std::string system_prompt;
};

// Built-in presets (Edge0-style): "default", "chat", "long", "bench".
inline Profile get(const std::string& name) {
    static const std::map<std::string, Profile> table = {
        {"default", Profile{"default", 4096, 64, 0.0, 1.0, 1.0, 64, 256, false, ""}},
        {"chat",   Profile{"chat", 4096, 64, 0.7, 0.95, 1.0, 64, 256, true,
                            "You are a helpful assistant."}},
        {"long",   Profile{"long", 16384, 512, 0.2, 0.9, 1.1, 128, 512, false, ""}},
        {"bench",  Profile{"bench", 2048, 32, 0.0, 1.0, 1.0, 32, 128, false, ""}}
    };
    auto it = table.find(name);
    if (it != table.end()) return it->second;
    return Profile{};
}

inline std::vector<std::string> names() {
    return {"default", "chat", "long", "bench"};
}

}  // namespace profile

#ifdef PROF_TEST
#include <cassert>
#include <iostream>

int main() {
    assert(profile::get("chat").use_chat == true);
    assert(profile::get("chat").temperature == 0.7);
    assert(profile::names().size() == 4);
    std::cout << "All profile tests passed.\n";
    return 0;
}
#endif