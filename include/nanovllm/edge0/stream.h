// edge0-stream: out-of-core MoE expert streaming for Vulkan.
// SharedExpertCache: global LRU across layers (one VRAM slot budget for the whole model).
// PrefetchBuffer: cap-bounded staging for eagerly prefetched bundles.
#pragma once
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace e0 {

// Key: (layer, expert) packed into uint64_t.
struct LEKey {
    int layer, expert;
    uint64_t pack(int layer_shift) const {
        return (uint64_t)layer << layer_shift | (uint64_t)(expert & 0xFFFF);
    }
};

// Global LRU across all MoE layers: one shared memory budget for the whole model.
// A layer that thrashes evicts entries from other layers, keeping the
// whole-model VRAM budget flat. Entries are (layer, expert) → slot.
class SharedExpertCache {
public:
    explicit SharedExpertCache(int slots = 64)
        : slots_(std::max(0, slots)), lru_(std::max(0, slots), 0) {}
    // ponytail: per-layer slot ids alias in this global lru_; key lru_ by (layer,slot) if eviction quality matters

    // Returns the slot index for (layer, expert), or -1 if not cached.
    int get(int layer, int expert) const {
        uint64_t key = le_key(layer, expert);
        auto it = map_.find(key);
        return it == map_.end() ? -1 : (int)it->second;
    }

    // Insert a newly loaded expert into slot `slot`. Evicts the LRU entry if full.
    // Returns {evicted_layer, evicted_expert} (both -1 if no eviction).
    LEKey put(int layer, int expert, int slot, uint64_t clk) {
        uint64_t key = le_key(layer, expert);
        if (slot < 0) {  // eviction notice: drop the entry, no slot to stamp
            map_.erase(key);
            order_.erase(std::remove_if(order_.begin(), order_.end(),
                [key](const OrdEntry& e) { return e.first == key; }), order_.end());
            return {-1, -1};
        }
        if (slot >= (int)lru_.size()) lru_.resize(slot + 1, 0);
        lru_[slot] = clk;
        order_.erase(std::remove_if(order_.begin(), order_.end(),
            [key](const OrdEntry& e) { return e.first == key; }), order_.end());
        order_.push_back({key, clk});

        LEKey evicted{-1, -1};
        if ((int)map_.size() >= slots_) {
            // Find LRU entry to evict (smallest clock among all occupied)
            auto lru_it = std::min_element(order_.begin(), order_.end(),
                [](const OrdEntry& a, const OrdEntry& b) { return a.second < b.second; });
            if (lru_it != order_.end()) {
                uint64_t ek = lru_it->first;
                int el = layer_of(ek), ee = expert_of(ek);
                auto mit = map_.find(ek);
                int eslot = mit == map_.end() ? -1 : mit->second;
                map_.erase(ek);
                order_.erase(lru_it);
                if (eslot >= 0 && eslot < (int)lru_.size()) lru_[eslot] = 0;
                evicted = {el, ee};
            }
        }
        map_[key] = slot;
        return evicted;
    }

    void touch(int layer, int expert, uint64_t clk) {
        uint64_t key = le_key(layer, expert);
        auto it = map_.find(key);
        if (it == map_.end()) return;
        int slot = it->second;
        if (slot >= 0 && slot < (int)lru_.size()) lru_[slot] = clk;
        for (auto& e : order_)
            if (e.first == key) { e.second = clk; break; }
    }

    int size() const { return (int)map_.size(); }
    int capacity() const { return slots_; }
    void clear() { map_.clear(); order_.clear(); lru_.clear(); }

    // Host mirror of the device slot_of table, sized [num_layers][num_experts].
    // slot_of[layer][expert] = slot index, or -1 if not cached.
    struct HostMirror {
        int num_layers, num_experts;
        std::vector<int32_t> data;
        HostMirror(int nl, int ne) : num_layers(nl), num_experts(ne), data((size_t)nl * ne, -1) {}
        int32_t& at(int l, int e) { return data[(size_t)l * num_experts + e]; }
    };

private:
    static uint64_t le_key(int layer, int expert) {
        return (uint64_t)layer << 16 | (uint64_t)(expert & 0xFFFF);
    }
    static int layer_of(uint64_t k) { return (int)(k >> 16); }
    static int expert_of(uint64_t k) { return (int)(k & 0xFFFF); }
    struct OrdEntry { uint64_t first; uint64_t second; };
    int slots_;
    std::unordered_map<uint64_t, int> map_;     // (layer,expert) → slot
    std::vector<int> lru_;                       // slot → clock
    std::vector<OrdEntry> order_;                // LRU order tracking
};

// Cap-bounded buffer of eagerly prefetched bundles that have NOT yet been
// promoted into the LRU. `put` evicts the oldest entry when over capacity.
// `pop` removes a bundle when the layer actually consumes it.
class PrefetchBuffer {
public:
    explicit PrefetchBuffer(int cap = 48) : cap_(std::max(0, cap)) {}

    bool contains(uint64_t key) const {
        return buf_.find(key) != buf_.end();
    }
    void put(uint64_t key, uint64_t clk) {
        buf_[key] = clk;
        if (cap_ > 0 && (int)buf_.size() > cap_) {
            auto lru_it = std::min_element(buf_.begin(), buf_.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            buf_.erase(lru_it);
        }
    }
    void pop(uint64_t key) { buf_.erase(key); }
    void set_cap(int c) { cap_ = std::max(0, c); }

private:
    int cap_;
    std::unordered_map<uint64_t, uint64_t> buf_;  // key → clock
};

}  // namespace e0
