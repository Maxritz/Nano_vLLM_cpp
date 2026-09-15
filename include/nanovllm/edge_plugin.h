// Edge plugin ABI: KV-cache policies (and future strategy plugins) as DLLs.
// Stable C ABI, versioned. Builtins in the engine mirror every interface so a
// missing/unloadable DLL silently falls back — the switches always work.
//
// A KV policy maps a logical (position, key_len) to physical cache slots and
// block tables. Single-sequence v1; the engine owns all Vulkan objects, the
// policy only computes index maps (pure arithmetic, no deps, trivially DLL-able).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EDGE_PLUGIN_ABI 1

// Returns windowed key length for `pos` (0-based) given sink+window, or
// pos+1 when window <= 0 (identity = unbounded cache).
typedef int64_t (*edge_key_len_fn)(int64_t pos, int window, int sink);

// Fill physical slot for `pos` and the block table covering [0, key_len).
// table has max_blocks entries; block_size matches the engine cache.
// Unused tail entries must be set to -1.
typedef void (*edge_map_fn)(int64_t pos, int64_t key_len, int window, int sink,
                            int block_size, int max_blocks,
                            int32_t* slot_out, int32_t* table_out);

typedef struct {
    int abi;                 // must equal EDGE_PLUGIN_ABI
    const char* name;        // e.g. "streaming"
    const char* (*describe)(void);
    edge_key_len_fn key_len;
    edge_map_fn map;
} EdgeKvPolicy;

// Every policy DLL exports exactly this symbol.
typedef const EdgeKvPolicy* (*edge_plugin_desc_fn)(void);

#ifdef __cplusplus
}
#endif
