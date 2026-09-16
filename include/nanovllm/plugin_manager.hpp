#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <string>

#include "nanovllm/edge_plugin.h"

namespace edgeplug {

struct Policy {
    std::string name, source;
    edge_key_len_fn key_len = nullptr;
    edge_map_fn map = nullptr;
    HMODULE dll = nullptr;
};

namespace detail {

inline int64_t builtin_key_len(int64_t pos, int window, int sink) {
    if (window <= 0) {
        return pos + 1;
    }
    int64_t val = static_cast<int64_t>(sink) +
                  (static_cast<int64_t>(window) < (pos + 1 - sink)
                       ? static_cast<int64_t>(window)
                       : (pos + 1 - sink));
    return val < 0 ? 0 : val;
}

inline void builtin_map(int64_t pos, int64_t key_len, int window, int sink,
                        int block_size, int max_blocks,
                        int32_t* slot_out, int32_t* table_out) {
    (void)window;
    (void)sink;
    (void)max_blocks;
    *slot_out = static_cast<int32_t>(pos);
    for (int64_t i = 0; i < key_len; ++i) {
        table_out[i] = static_cast<int32_t>(i / block_size);
    }
    for (int64_t i = key_len; i < max_blocks; ++i) table_out[i] = -1;
}

}  // namespace detail

inline Policy builtin(const std::string& name) {
    Policy p;
    p.name = name;
    p.source = "builtin:" + name;
    p.key_len = &detail::builtin_key_len;
    p.map = &detail::builtin_map;
    p.dll = nullptr;
    return p;
}

inline Policy load(const std::string& dir, const std::string& name) {
    std::string primary = dir + "\\" + name + ".dll";
    std::string fallback = name + ".dll";

    HMODULE dll = LoadLibraryA(primary.c_str());
    if (dll == nullptr) {
        dll = LoadLibraryA(fallback.c_str());
    }

    if (dll == nullptr) {
        OutputDebugStringA(("edgeplug: LoadLibrary failed for " + primary +
                            " and " + fallback + "\n").c_str());
        return builtin(name);
    }

    auto desc_fn = reinterpret_cast<edge_plugin_desc_fn>(
        GetProcAddress(dll, "edge_plugin_desc"));
    if (desc_fn == nullptr) {
        OutputDebugStringA("edgeplug: GetProcAddress(edge_plugin_desc) failed\n");
        FreeLibrary(dll);
        return builtin(name);
    }

    const EdgeKvPolicy* desc = desc_fn();
    if (desc == nullptr) {
        OutputDebugStringA("edgeplug: edge_plugin_desc returned null\n");
        FreeLibrary(dll);
        return builtin(name);
    }

    if (desc->abi != EDGE_PLUGIN_ABI || desc->key_len == nullptr ||
        desc->map == nullptr) {
        OutputDebugStringA("edgeplug: ABI mismatch or missing function pointers\n");
        FreeLibrary(dll);
        return builtin(name);
    }

    Policy p;
    p.name = name;
    p.source = (dll != nullptr) ? primary : ("builtin:" + name);
    p.key_len = desc->key_len;
    p.map = desc->map;
    p.dll = dll;
    return p;
}

inline void unload(Policy& p) {
    if (p.dll != nullptr) {
        FreeLibrary(p.dll);
    }
    p.name.clear();
    p.source.clear();
    p.key_len = nullptr;
    p.map = nullptr;
    p.dll = nullptr;
}

}  // namespace edgeplug
