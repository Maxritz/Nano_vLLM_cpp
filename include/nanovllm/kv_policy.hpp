#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <string>
#include <vector>

#include "nanovllm/edge_plugin.h"

namespace kvpolicy {

struct Policy {
    std::string name, source;  // source = dll path or "builtin:streaming"
    edge_key_len_fn key_len = nullptr;
    edge_map_fn map = nullptr;
    HMODULE dll = nullptr;
};

// The streaming sink+window policy (the first DLL). Mirrors the ABI exactly:
//   key_len(pos, window, sink): window<=0 -> pos+1; else max(sink + min(window,
//     pos+1-sink), 0).  map: slot_out=pos; table_out[i]=i/block_size for
//     i<key_len, else -1 (tail-fill so max_blocks entries are always valid).
inline int64_t streaming_key_len(int64_t pos, int window, int sink) {
    if (window <= 0) return pos + 1;
    int64_t val = sink + (window < (pos + 1 - sink) ? window : (pos + 1 - sink));
    return val < 0 ? 0 : val;
}

inline void streaming_map(int64_t pos, int64_t key_len, int window, int sink,
                          int block_size, int max_blocks,
                          int32_t* slot_out, int32_t* table_out) {
    (void)window; (void)sink; (void)max_blocks;
    *slot_out = static_cast<int32_t>(pos);
    for (int64_t i = 0; i < key_len; ++i)
        table_out[i] = static_cast<int32_t>(i / block_size);
    for (int64_t i = key_len; i < max_blocks; ++i) table_out[i] = -1;
}

inline Policy builtin_streaming() {
    Policy p;
    p.name = "streaming";
    p.source = "builtin:streaming";
    p.dll = nullptr;
    p.key_len = &streaming_key_len;
    p.map = &streaming_map;
    return p;
}

// Load <dir>/<name>.dll else <name>.dll; validate desc && abi == EDGE_PLUGIN_ABI
// && key_len && map; on ANY failure return builtin_streaming(), never throw.
inline Policy load(const std::string& dir, const std::string& name) {
    Policy fallback = builtin_streaming();

    std::string path1 = dir + "\\" + name + ".dll";
    std::string path2 = name + ".dll";

    HMODULE dll = LoadLibraryA(path1.c_str());
    if (!dll) dll = LoadLibraryA(path2.c_str());
    if (!dll) return fallback;

    auto desc_fn = reinterpret_cast<edge_plugin_desc_fn>(
        GetProcAddress(dll, "edge_plugin_desc"));
    if (!desc_fn) { FreeLibrary(dll); return fallback; }

    const EdgeKvPolicy* desc = desc_fn();
    if (!desc || desc->abi != EDGE_PLUGIN_ABI ||
        desc->key_len == nullptr || desc->map == nullptr) {
        FreeLibrary(dll);
        return fallback;
    }

    Policy p;
    p.name = name;
    p.source = path1;
    p.dll = dll;
    p.key_len = desc->key_len;
    p.map = desc->map;
    return p;
}

inline void unload(Policy& p) {
    if (p.dll) { FreeLibrary(p.dll); p.dll = nullptr; }
    p.key_len = nullptr;
    p.map = nullptr;
}

}  // namespace kvpolicy

#ifdef KV_TEST
#include <cassert>
#include <iostream>

int main() {
    // Test 1: builtin_streaming().key_len(5, 3, 1) == 3
    auto bs = kvpolicy::builtin_streaming();
    assert(bs.key_len(5, 3, 1) == 3);
    // window<=0 -> identity
    assert(bs.key_len(5, -1, 1) == 6);
    // map tail-fills to -1
    int32_t slot = -99, tbl[4] = {9, 9, 9, 9};
    bs.map(2, 2, 3, 1, 1, 4, &slot, tbl);
    assert(slot == 2);
    assert(tbl[0] == 0 && tbl[1] == 1 && tbl[2] == -1 && tbl[3] == -1);

    // Test 2: load("nope","nope").name == builtin_streaming().name
    auto loaded = kvpolicy::load("nope", "nope");
    assert(loaded.name == bs.name);

    std::cout << "All kv_policy tests passed.\n";
    return 0;
}
#endif