#pragma once

#include <windows.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <sstream>
#include <atomic>
#include <mutex>

namespace mcp {

struct Tool {
    std::string name;
    std::string description;
    std::string schema; // raw JSON
};

struct Content {
    std::string type;
    std::string text;
};

class Client {
public:
    explicit Client(const std::string& cmd) {
        if (!spawn(cmd)) {
            throw std::runtime_error("Failed to spawn MCP server process");
        }
    }

    ~Client() {
        terminate();
    }

    Client(const Client&) = delete;
    Client& operator=(const Client&) = delete;

    Client(Client&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mtx_);
        hProcess_ = other.hProcess_;
        hStdinRead_ = other.hStdinRead_;
        hStdinWrite_ = other.hStdinWrite_;
        hStdoutRead_ = other.hStdoutRead_;
        hStdoutWrite_ = other.hStdoutWrite_;
        id_counter_ = other.id_counter_.load();
        running_ = other.running_.load();
        other.hProcess_ = nullptr;
        other.hStdinRead_ = nullptr;
        other.hStdinWrite_ = nullptr;
        other.hStdoutRead_ = nullptr;
        other.hStdoutWrite_ = nullptr;
        other.running_ = false;
    }

    Client& operator=(Client&& other) noexcept {
        if (this != &other) {
            terminate();
            std::lock_guard<std::mutex> lock(other.mtx_);
            hProcess_ = other.hProcess_;
            hStdinRead_ = other.hStdinRead_;
            hStdinWrite_ = other.hStdinWrite_;
            hStdoutRead_ = other.hStdoutRead_;
            hStdoutWrite_ = other.hStdoutWrite_;
            id_counter_ = other.id_counter_.load();
            running_ = other.running_.load();
            other.hProcess_ = nullptr;
            other.hStdinRead_ = nullptr;
            other.hStdinWrite_ = nullptr;
            other.hStdoutRead_ = nullptr;
            other.hStdoutWrite_ = nullptr;
            other.running_ = false;
        }
        return *this;
    }

    bool initialize(int timeout_ms = 10000) {
        try {
            std::string req = build_request("initialize", R"({"protocolVersion":"1.0","capabilities":{}})");
            if (!send_request(req)) return false;
            std::string resp = read_response(req_id(req), timeout_ms);
            return !resp.empty();
        } catch (...) {
            return false;
        }
    }

    std::vector<Tool> list_tools(int timeout_ms = 10000) {
        std::vector<Tool> tools;
        try {
            std::string req = build_request("tools/list", "{}");
            if (!send_request(req)) return tools;
            std::string resp = read_response(req_id(req), timeout_ms);
            if (resp.empty()) return tools;
            tools = parse_tools(resp);
        } catch (...) {
            return tools;
        }
        return tools;
    }

    std::vector<Content> call_tool(const std::string& name,
                                   const std::string& args_json,
                                   int timeout_ms = 30000) {
        std::vector<Content> contents;
        try {
            std::string params = R"({"name":")" + escape(name) + R"(","arguments":)" + args_json;
            std::string req = build_request("tools/call", params);
            if (!send_request(req)) return contents;
            std::string resp = read_response(req_id(req), timeout_ms);
            if (resp.empty()) return contents;
            contents = parse_contents(resp);
        } catch (...) {
            return contents;
        }
        return contents;
    }

private:
    HANDLE hProcess_ = nullptr;
    HANDLE hStdinRead_ = nullptr;
    HANDLE hStdinWrite_ = nullptr;
    HANDLE hStdoutRead_ = nullptr;
    HANDLE hStdoutWrite_ = nullptr;
    std::atomic<int> id_counter_{0};
    std::atomic<bool> running_{false};
    std::mutex mtx_;

    bool spawn(const std::string& cmd) {
        SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};

        if (!CreatePipe(&hStdinRead_, &hStdinWrite_, &sa, 0)) return false;
        if (!CreatePipe(&hStdoutRead_, &hStdoutWrite_, &sa, 0)) return false;

        SetHandleInformation(hStdinWrite_, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(hStdoutRead_, HANDLE_FLAG_INHERIT, 0);

        STARTUPINFOA si{sizeof(STARTUPINFOA)};
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = hStdinRead_;
        si.hStdOutput = hStdoutWrite_;
        si.hStdError = hStdoutWrite_;

        PROCESS_INFORMATION pi{0};
        std::string cmdline = cmd;

        if (!CreateProcessA(nullptr, &cmdline[0], nullptr, nullptr, TRUE,
                            0, nullptr, nullptr, &si, &pi)) {
            CloseHandle(hStdinRead_);
            CloseHandle(hStdinWrite_);
            CloseHandle(hStdoutRead_);
            CloseHandle(hStdoutWrite_);
            return false;
        }

        CloseHandle(hStdinRead_);
        CloseHandle(hStdoutWrite_);
        hProcess_ = pi.hProcess;
        running_ = true;
        return true;
    }

    void terminate() {
        if (hProcess_) {
            TerminateProcess(hProcess_, 0);
            CloseHandle(hProcess_);
            hProcess_ = nullptr;
        }
        if (hStdinWrite_) {
            CloseHandle(hStdinWrite_);
            hStdinWrite_ = nullptr;
        }
        if (hStdoutRead_) {
            CloseHandle(hStdoutRead_);
            hStdoutRead_ = nullptr;
        }
        running_ = false;
    }

    std::string build_request(const std::string& method, const std::string& params) {
        int id = ++id_counter_;
        std::ostringstream oss;
        oss << R"({"jsonrpc":"2.0","id":)" << id
            << R"(,"method":")" << method << R"(","params":)" << params << "}\n";
        return oss.str();
    }

    int req_id(const std::string& req) {
        size_t pos = req.find(R"("id":)");
        if (pos == std::string::npos) return 0;
        pos += 5;
        std::string num;
        while (pos < req.size() && isdigit(req[pos])) {
            num += req[pos++];
        }
        return num.empty() ? 0 : std::stoi(num);
    }

    bool send_request(const std::string& req) {
        DWORD written = 0;
        if (!WriteFile(hStdinWrite_, req.data(), (DWORD)req.size(), &written, nullptr)) {
            return false;
        }
        return written == req.size();
    }

    std::string read_response(int expected_id, int timeout_ms) {
        std::string buffer;
        auto start = std::chrono::steady_clock::now();
        while (true) {
            char ch;
            DWORD available = 0;
            if (PeekNamedPipe(hStdoutRead_, nullptr, 0, nullptr, &available, nullptr)) {
                if (available > 0) {
                    DWORD read = 0;
                    if (ReadFile(hStdoutRead_, &ch, 1, &read, nullptr) && read == 1) {
                        buffer += ch;
                        if (ch == '\n') {
                            if (validate_response(buffer, expected_id)) {
                                return buffer;
                            }
                            buffer.clear();
                        }
                    }
                }
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeout_ms) {
                return "";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return "";
    }

    bool validate_response(const std::string& resp, int expected_id) {
        std::string id_str = "\"id\":";
        size_t pos = resp.find(id_str);
        if (pos == std::string::npos) return false;
        pos += id_str.size();
        std::string num;
        while (pos < resp.size() && isdigit(resp[pos])) {
            num += resp[pos++];
        }
        if (num.empty()) return false;
        try {
            return std::stoi(num) == expected_id;
        } catch (...) {
            return false;
        }
    }

    std::vector<Tool> parse_tools(const std::string& resp) {
        std::vector<Tool> tools;
        size_t pos = resp.find("\"tools\":");
        if (pos == std::string::npos) return tools;
        pos += 7;
        size_t arr_start = resp.find('[', pos);
        size_t arr_end = resp.find(']', arr_start);
        if (arr_start == std::string::npos || arr_end == std::string::npos) return tools;

        std::string arr = resp.substr(arr_start + 1, arr_end - arr_start - 1);
        size_t idx = 0;
        while (idx < arr.size()) {
            size_t obj_start = arr.find('{', idx);
            if (obj_start == std::string::npos) break;
            int depth = 1;
            size_t i = obj_start + 1;
            while (i < arr.size() && depth > 0) {
                if (arr[i] == '{') depth++;
                else if (arr[i] == '}') depth--;
                i++;
            }
            if (depth != 0) break;
            std::string obj = arr.substr(obj_start, i - obj_start);
            Tool tool;
            tool.name = extract_json_string(obj, "name");
            tool.description = extract_json_string(obj, "description");
            tool.schema = extract_json_value(obj, "inputSchema");
            if (!tool.name.empty()) {
                tools.push_back(tool);
            }
            idx = i;
        }
        return tools;
    }

    std::vector<Content> parse_contents(const std::string& resp) {
        std::vector<Content> contents;
        size_t pos = resp.find("\"content\":");
        if (pos == std::string::npos) return contents;
        pos += 9;
        size_t arr_start = resp.find('[', pos);
        size_t arr_end = resp.find(']', arr_start);
        if (arr_start == std::string::npos || arr_end == std::string::npos) return contents;

        std::string arr = resp.substr(arr_start + 1, arr_end - arr_start - 1);
        size_t idx = 0;
        while (idx < arr.size()) {
            size_t obj_start = arr.find('{', idx);
            if (obj_start == std::string::npos) break;
            int depth = 1;
            size_t i = obj_start + 1;
            while (i < arr.size() && depth > 0) {
                if (arr[i] == '{') depth++;
                else if (arr[i] == '}') depth--;
                i++;
            }
            if (depth != 0) break;
            std::string obj = arr.substr(obj_start, i - obj_start);
            Content c;
            c.type = extract_json_string(obj, "type");
            c.text = extract_json_string(obj, "text");
            if (!c.type.empty()) {
                contents.push_back(c);
            }
            idx = i;
        }
        return contents;
    }

    std::string extract_json_string(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return "";
        pos += search.size();
        std::string result;
        while (pos < json.size()) {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                char next = json[pos + 1];
                if (next == 'n') result += '\n';
                else if (next == 't') result += '\t';
                else if (next == 'r') result += '\r';
                else if (next == '"') result += '"';
                else if (next == '\\') result += '\\';
                else result += next;
                pos += 2;
            } else if (json[pos] == '"') {
                break;
            } else {
                result += json[pos++];
            }
        }
        return result;
    }

    std::string extract_json_value(const std::string& json, const std::string& key) {
        std::string search = "\"" + key + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return "";
        pos += search.size();
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
        if (pos >= json.size()) return "";
        if (json[pos] == '{') {
            int depth = 1;
            size_t i = pos + 1;
            while (i < json.size() && depth > 0) {
                if (json[i] == '{') depth++;
                else if (json[i] == '}') depth--;
                i++;
            }
            if (depth == 0) return json.substr(pos, i - pos);
        } else if (json[pos] == '[') {
            int depth = 1;
            size_t i = pos + 1;
            while (i < json.size() && depth > 0) {
                if (json[i] == '[') depth++;
                else if (json[i] == ']') depth--;
                i++;
            }
            if (depth == 0) return json.substr(pos, i - pos);
        }
        return "";
    }

    std::string escape(const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c == '"') out += "\\\"";
            else if (c == '\\') out += "\\\\";
            else if (c == '\n') out += "\\n";
            else if (c == '\r') out += "\\r";
            else if (c == '\t') out += "\\t";
            else out += c;
        }
        return out;
    }
};

} // namespace mcp
