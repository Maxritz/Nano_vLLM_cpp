#pragma once

#include <string>
#include <vector>
#include <functional>
#include <stdexcept>

namespace agent {

struct Tool {
    std::string name;
    std::string description;
    std::string schema;  // raw JSON
};

struct Call {
    std::string name;
    std::string args;
};

// Hand-rolled JSON: extract "name" and "arguments" from a tool-call object.
// Tolerant of missing/extra fields; never throws.
inline Call parse_call(const std::string& tool_obj_json) {
    Call result;
    if (tool_obj_json.empty()) return result;

    auto extract = [&](const std::string& key) -> std::string {
        std::string k = "\"" + key + "\"";
        size_t pos = tool_obj_json.find(k);
        if (pos == std::string::npos) return "";
        pos = tool_obj_json.find(':', pos + k.size());
        if (pos == std::string::npos) return "";
        pos++;
        while (pos < tool_obj_json.size() && (tool_obj_json[pos] == ' ' || tool_obj_json[pos] == '\t' ||
                                               tool_obj_json[pos] == '\n' || tool_obj_json[pos] == '\r')) {
            pos++;
        }
        if (pos >= tool_obj_json.size()) return "";
        if (tool_obj_json[pos] != '"') return "";
        pos++;
        std::string val;
        while (pos < tool_obj_json.size() && tool_obj_json[pos] != '"') {
            if (tool_obj_json[pos] == '\\' && pos + 1 < tool_obj_json.size()) {
                val += tool_obj_json[pos + 1];
                pos += 2;
            } else {
                val += tool_obj_json[pos];
                pos++;
            }
        }
        return val;
    };

    result.name = extract("name");
    result.args = extract("arguments");
    return result;
}

class Loop {
public:
    void add_tool(Tool t) {
        tools_.push_back(std::move(t));
    }

    const std::vector<Tool>& tools() const {
        return tools_;
    }

    // Render the tool list into the system prompt as a JSON-ish schema block.
    std::string tools_system_block() const {
        if (tools_.empty()) return "";
        std::string block = "Tools available:\n";
        for (const auto& t : tools_) {
            block += "- name: " + t.name + "\n";
            if (!t.description.empty()) block += "  description: " + t.description + "\n";
            if (!t.schema.empty()) block += "  schema: " + t.schema + "\n";
        }
        return block;
    }

    // Scan assistant text for a fenced ```json ... ``` or bare {"name":...}
    // block; return first parseable call, and the byte offset after it.
    Call find_call(const std::string& assistant_text, size_t& next_off) const {
        next_off = 0;
        Call empty_result;

        // Try fenced ```json ... ``` block first
        size_t fence_pos = assistant_text.find("```json");
        if (fence_pos != std::string::npos) {
            size_t content_start = fence_pos + 7; // skip ```json
            size_t end_fence = assistant_text.find("```", content_start);
            if (end_fence != std::string::npos) {
                std::string json_block = assistant_text.substr(content_start, end_fence - content_start);
                Call parsed = parse_call(json_block);
                if (!parsed.name.empty()) {
                    next_off = end_fence + 3;
                    return parsed;
                }
            }
        }

        // Try bare {"name":...} block
        size_t brace_pos = assistant_text.find("{");
        while (brace_pos != std::string::npos) {
            size_t end_brace = assistant_text.find('}', brace_pos);
            if (end_brace != std::string::npos) {
                std::string json_block = assistant_text.substr(brace_pos, end_brace - brace_pos + 1);
                Call parsed = parse_call(json_block);
                if (!parsed.name.empty()) {
                    next_off = end_brace + 1;
                    return parsed;
                }
            }
            brace_pos = assistant_text.find('{', brace_pos + 1);
        }

        return empty_result;
    }

private:
    std::vector<Tool> tools_;
};

} // namespace agent

#ifdef AGENT_TEST
#include <cassert>
#include <iostream>

int main() {
    {
        agent::Call c = agent::parse_call("{\"name\":\"foo\",\"arguments\":\"{}\"}");
        assert(c.name == "foo");
        assert(c.args == "{}");
    }
    {
        agent::Loop loop;
        std::string text = "Some text ```json {\"name\":\"bar\",\"arguments\":\"{\\\"x\\\":1}\"} ``` more text";
        size_t next_off = 0;
        agent::Call c = loop.find_call(text, next_off);
        assert(c.name == "bar");
        assert(next_off > 0);
        assert(next_off <= text.size());
    }
    std::cout << "All tests passed.\n";
    return 0;
}
#endif