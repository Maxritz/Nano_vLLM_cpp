#pragma once

#include <map>
#include <memory>
#include <vector>

namespace cache {

// A KV block is identified by a logical slot id (the block table entry).
struct Block {
    int slot = -1;
    bool shared = false;
};

// Radix tree node: one edge per distinct next token, each edge leads to a
// child node holding the block that covers the transition.
struct Node {
    int token = -1;
    Block block;
    std::map<int, std::unique_ptr<Node>> children;
    Node* parent = nullptr;
};

class RadixCache {
public:
    // Insert a token sequence; returns the list of blocks (one per input
    // token) that cover it, creating shared nodes for any new prefix. Tokens
    // already present are reused (shared=true).
    std::vector<Block> insert(const std::vector<int>& tokens) {
        std::vector<Block> result;
        if (!root_) root_ = std::make_unique<Node>();
        Node* cur = root_.get();
        for (int tok : tokens) {
            auto it = cur->children.find(tok);
            if (it != cur->children.end()) {
                Node* child = it->second.get();
                Block b = child->block;
                b.shared = true;
                result.push_back(b);
                cur = child;
            } else {
                auto new_node = std::make_unique<Node>();
                new_node->token = tok;
                new_node->parent = cur;
                new_node->block.slot = next_slot_++;
                new_node->block.shared = false;
                Node* child = new_node.get();
                cur->children[tok] = std::move(new_node);
                result.push_back(child->block);
                cur = child;
            }
        }
        return result;
    }

    // Look up a sequence: returns the blocks covering the longest matching
    // prefix, or empty if nothing matches.
    std::vector<Block> lookup(const std::vector<int>& tokens) const {
        std::vector<Block> result;
        Node* cur = root_.get();
        for (int tok : tokens) {
            auto it = cur->children.find(tok);
            if (it == cur->children.end()) break;
            Node* child = it->second.get();
            result.push_back(child->block);
            cur = child;
        }
        return result;
    }

    // Number of nodes in the tree.
    size_t size() const { return count_nodes(root_.get()); }

    // Clear the tree (frees all nodes).
    void clear() {
        root_.reset();
        next_slot_ = 0;
    }

private:
    std::unique_ptr<Node> root_;
    int next_slot_ = 0;

    static size_t count_nodes(const Node* n) {
        if (!n) return 0;
        size_t cnt = 1;
        for (const auto& kv : n->children) cnt += count_nodes(kv.second.get());
        return cnt;
    }
};

}  // namespace cache