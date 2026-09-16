#pragma once

#include <memory>
#include <string>
#include <vector>
#include <set>
#include <cctype>

namespace gen {

// A soft constraint applied character-by-character at sample time. The engine
// calls allow(ch) for each candidate token's next character; returning false
// forces the sampler to skip that candidate. feed(ch) is called after a
// character is emitted so the constraint can advance its state; reset()
// clears it before a new sequence.
struct Constraint {
    virtual ~Constraint() = default;
    // true if character `ch` may be emitted at the current state.
    virtual bool allow(char ch) const = 0;
    // Advance state after `ch` is emitted.
    virtual void feed(char ch) {}
    // Reset before a new sequence.
    virtual void reset() {}
};

// A no-op constraint: every character is allowed.
struct AllowAll : Constraint {
    bool allow(char) const override { return true; }
};

namespace detail {

// Permissive JSON-object shape matcher. Enforces the brace / quote / colon /
// comma structure and the exact set of top-level keys (any order, each once),
// with string values. Does NOT constrain value contents. Accepts whitespace
// anywhere. After the closing '}' the matcher is done and rejects everything
// further (feed is a no-op, allow returns false) — call reset() to start again.
class JsonShapeConstraint : public Constraint {
public:
    explicit JsonShapeConstraint(const std::vector<std::string>& keys)
        : keys_(keys.begin(), keys.end()), nkeys_(keys.size()) {}

    bool allow(char ch) const override {
        if (done_) return ch == '}';  // object closed (or invalid): only closer valid
        if (std::isspace(static_cast<unsigned char>(ch))) return true;
        switch (state_) {
            case State::ExpectObjectStart:
                return ch == '{';
            case State::ExpectKeyOrEnd:
                return ch == '"' || ch == '}';
            case State::InKey:
                return true;  // '"' is the terminator, handled by feed()
            case State::ExpectColon:
                return ch == ':';
            case State::ExpectValueStart:
                return ch == '"';
            case State::InValue:
                return true;  // '"' is the terminator, handled by feed()
            case State::ExpectCommaOrEnd:
                return ch == ',' || ch == '}';
            default:
                return false;
        }
    }

    void feed(char ch) override {
        if (done_) return;
        if (std::isspace(static_cast<unsigned char>(ch))) return;
        switch (state_) {
            case State::ExpectObjectStart:
                if (ch == '{') state_ = State::ExpectKeyOrEnd;
                break;
            case State::ExpectKeyOrEnd:
                if (ch == '}') { done_ = true; }
                else if (ch == '"') { state_ = State::InKey; current_key_.clear(); }
                break;
            case State::InKey:
                if (ch == '"') state_ = State::ExpectColon;
                else current_key_ += ch;
                break;
            case State::ExpectColon:
                if (ch == ':') state_ = State::ExpectValueStart;
                break;
            case State::ExpectValueStart:
                if (ch == '"') state_ = State::InValue;
                break;
            case State::InValue:
                if (ch == '"') {
                    if (keys_.count(current_key_) && !seen_keys_.insert(current_key_).second) {
                        done_ = true;  // duplicate key -> invalid
                    } else {
                        state_ = State::ExpectCommaOrEnd;
                    }
                }
                break;
            case State::ExpectCommaOrEnd:
                if (ch == ',') state_ = State::ExpectKeyOrEnd;
                else if (ch == '}') done_ = true;
                break;
        }
    }

    void reset() override {
        state_ = State::ExpectObjectStart;
        done_ = false;
        current_key_.clear();
        seen_keys_.clear();
    }

private:
    enum class State {
        ExpectObjectStart,
        ExpectKeyOrEnd,
        InKey,
        ExpectColon,
        ExpectValueStart,
        InValue,
        ExpectCommaOrEnd,
    };
    std::set<std::string> keys_;
    size_t nkeys_ = 0;
    State state_ = State::ExpectObjectStart;
    bool done_ = false;
    std::string current_key_;
    std::set<std::string> seen_keys_;
};

} // namespace detail

// Build a Constraint matching a JSON object with exactly `keys` (comma-separated,
// any order, each once). Returns nullptr on bad input (empty, duplicate keys).
inline std::unique_ptr<Constraint> json_shape(const std::string& keys) {
    if (keys.empty()) return nullptr;
    std::vector<std::string> key_list;
    std::string current;
    for (char c : keys) {
        if (c == ',') {
            if (current.empty()) return nullptr;
            key_list.push_back(current);
            current.clear();
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        } else {
            current += c;
        }
    }
    if (!current.empty()) key_list.push_back(current);
    if (key_list.empty()) return nullptr;
    std::set<std::string> unique(key_list.begin(), key_list.end());
    if (unique.size() != key_list.size()) return nullptr;
    return std::make_unique<detail::JsonShapeConstraint>(key_list);
}

} // namespace gen

#ifdef GEN_TEST
#include <cassert>
#include <iostream>

int main() {
    auto c = gen::json_shape("a,b");
    assert(c != nullptr);
    gen::AllowAll a;
    assert(a.allow(0) == true);
    assert(gen::json_shape("") == nullptr);
    assert(gen::json_shape("a,a") == nullptr);

    // Shape walk: {"a":"x","b":"y"} accepted, garbage rejected.
    {
        auto m = gen::json_shape("a,b");
        const char* ok = "{\"a\":\"x\",\"b\":\"y\"}";
        for (const char* p = ok; *p; ++p) {
            assert(m->allow(*p));
            m->feed(*p);
        }
        assert(m->allow('z') == false);  // done: nothing further allowed
    }
    {
        auto m = gen::json_shape("a,b");
        const char* bad = "{\"a\":\"x\",\"a\":\"y\"}";
        for (const char* p = bad; *p; ++p) {
            if (!m->allow(*p)) { assert(false); }
            m->feed(*p);
        }
        assert(m->allow('z') == false);  // duplicate key -> done
    }
    std::cout << "All GEN_TEST assertions passed.\n";
    return 0;
}
#endif