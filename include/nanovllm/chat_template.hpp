#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <sstream>

namespace chtmpl {

struct Msg {
    std::string role;
    std::string content;
};

namespace detail {

struct Var {
    enum Type { Str, Bool } type;
    std::string s;
    bool b;
    Var() : type(Bool), b(false) {}
    static Var fromStr(const std::string& v) { Var r; r.type = Var::Str; r.s = v; return r; }
    static Var fromBool(bool v) { Var r; r.type = Var::Bool; r.b = v; return r; }
    bool truthy() const {
        if (type == Bool) return b;
        return !s.empty();
    }
};

struct Ctx {
    const std::vector<Msg>* messages;
    bool add_generation_prompt;
    const Msg* msg;
    bool loop_first;
    bool loop_last;
    std::vector<std::pair<std::string, Var>> locals;
};

inline Var lookupVar(const Ctx& ctx, const std::string& name) {
    if (name == "messages") {
        Var r; r.type = Var::Bool; r.b = ctx.messages != nullptr && !ctx.messages->empty();
        return r;
    }
    if (name == "add_generation_prompt") return Var::fromBool(ctx.add_generation_prompt);
    if (name == "loop") {
        Var r; r.type = Var::Bool; r.b = true; return r;
    }
    if (name == "loop.first") return Var::fromBool(ctx.loop_first);
    if (name == "loop.last") return Var::fromBool(ctx.loop_last);
    if (name == "msg") {
        if (ctx.msg) return Var::fromStr(ctx.msg->content);
        return Var::fromStr("");
    }
    if (name == "message") {
        if (ctx.msg) return Var::fromStr(ctx.msg->content);
        return Var::fromStr("");
    }
    if (name == "message.role") {
        if (ctx.msg) return Var::fromStr(ctx.msg->role);
        return Var::fromStr("");
    }
    if (name == "message.content") {
        if (ctx.msg) return Var::fromStr(ctx.msg->content);
        return Var::fromStr("");
    }
    if (name == "msg.role") {
        if (ctx.msg) return Var::fromStr(ctx.msg->role);
        return Var::fromStr("");
    }
    if (name == "msg.content") {
        if (ctx.msg) return Var::fromStr(ctx.msg->content);
        return Var::fromStr("");
    }
    for (auto it = ctx.locals.rbegin(); it != ctx.locals.rend(); ++it) {
        if (it->first == name) return it->second;
    }
    Var r; r.type = Var::Bool; r.b = false; return r;
}

inline std::string evalExpr(const std::string& expr, const Ctx& ctx);

inline std::string evalPrimary(const std::string& e, const Ctx& ctx) {
    std::string s = e;
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    s = s.substr(a, b - a + 1);

    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
        return s.substr(1, s.size() - 2);
    }
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    if (s == "true") return "true";
    if (s == "false") return "false";
    if (s == "add_generation_prompt") {
        return ctx.add_generation_prompt ? "true" : "false";
    }
    if (s == "loop.first") return ctx.loop_first ? "true" : "false";
    if (s == "loop.last") return ctx.loop_last ? "true" : "false";
    if (s == "loop") return "true";
    if (s == "messages") return "";
    if (s == "msg" || s == "message") {
        return ctx.msg ? ctx.msg->content : "";
    }
    if (s == "msg.role" || s == "message.role") {
        return ctx.msg ? ctx.msg->role : "";
    }
    if (s == "msg.content" || s == "message.content") {
        return ctx.msg ? ctx.msg->content : "";
    }
    Var v = lookupVar(ctx, s);
    if (v.type == Var::Str) return v.s;
    return v.b ? "true" : "false";
}

inline std::string evalExpr(const std::string& expr, const Ctx& ctx) {
    std::string s = expr;
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    s = s.substr(a, b - a + 1);

    size_t ifPos = s.find(" if ");
    if (ifPos != std::string::npos) {
        size_t elsePos = s.find(" else ", ifPos + 4);
        if (elsePos != std::string::npos) {
            std::string x = s.substr(0, ifPos);
            std::string cond = s.substr(ifPos + 4, elsePos - (ifPos + 4));
            std::string y = s.substr(elsePos + 6);
            std::string condVal = evalExpr(cond, ctx);
            bool condTrue = (condVal == "true");
            return condTrue ? evalExpr(x, ctx) : evalExpr(y, ctx);
        }
    }

    size_t tilde = s.find('~');
    if (tilde != std::string::npos) {
        std::string left = s.substr(0, tilde);
        std::string right = s.substr(tilde + 1);
        return evalExpr(left, ctx) + evalExpr(right, ctx);
    }

    size_t eq = s.find("==");
    if (eq != std::string::npos) {
        std::string left = s.substr(0, eq);
        std::string right = s.substr(eq + 2);
        std::string lv = evalExpr(left, ctx);
        std::string rv = evalExpr(right, ctx);
        return lv == rv ? "true" : "false";
    }

    size_t neq = s.find("!=");
    if (neq != std::string::npos) {
        std::string left = s.substr(0, neq);
        std::string right = s.substr(neq + 2);
        std::string lv = evalExpr(left, ctx);
        std::string rv = evalExpr(right, ctx);
        return lv != rv ? "true" : "false";
    }

    size_t plus = s.find('+');
    if (plus != std::string::npos) {
        std::string left = s.substr(0, plus);
        std::string right = s.substr(plus + 1);
        return evalExpr(left, ctx) + evalExpr(right, ctx);
    }

    return evalPrimary(s, ctx);
}

inline bool evalCond(const std::string& cond, const Ctx& ctx) {
    std::string val = evalExpr(cond, ctx);
    return val == "true";
}

} // namespace detail

inline std::string render(const std::string& tmpl, const std::vector<Msg>& msgs, bool add_generation_prompt) {
    using namespace detail;
    std::string out;
    size_t i = 0;
    size_t n = tmpl.size();

    Ctx ctx;
    ctx.messages = &msgs;
    ctx.add_generation_prompt = add_generation_prompt;
    ctx.msg = nullptr;
    ctx.loop_first = false;
    ctx.loop_last = false;

    struct Parser {
        const std::string& tmpl;
        const std::vector<Msg>& msgs;
        bool add_generation_prompt;
        std::string& out;
        size_t& i;
        size_t n;

        Parser(const std::string& t, const std::vector<Msg>& m, bool agp, std::string& o, size_t& idx)
            : tmpl(t), msgs(m), add_generation_prompt(agp), out(o), i(idx), n(t.size()) {}

        void skipWs() {
            while (i < n && (tmpl[i] == ' ' || tmpl[i] == '\t' || tmpl[i] == '\r' || tmpl[i] == '\n')) i++;
        }

        std::string readUntil(char end) {
            std::string r;
            while (i < n && tmpl[i] != end) {
                r += tmpl[i++];
            }
            return r;
        }

        std::string readTag() {
            // Assumes we're positioned right after {%
            skipWs();
            std::string tag = readUntil('%');
            if (i < n && tmpl[i] == '%') i++;
            if (i < n && tmpl[i] == '}') i++;
            size_t te = tag.find_last_not_of(" \t\r\n");
            if (te != std::string::npos) tag = tag.substr(0, te + 1);
            size_t ts = tag.find_first_not_of(" \t\r\n");
            if (ts != std::string::npos) tag = tag.substr(ts);
            return tag;
        }

        void parseBlock(Ctx& ctx) {
            while (i < n) {
                if (tmpl[i] == '{') {
                    if (i + 1 < n && tmpl[i+1] == '{') {
                        i += 2;
                        std::string expr = readUntil('}');
                        if (i < n && tmpl[i] == '}') i++;
                        if (i < n && tmpl[i] == '}') i++;
                        std::string val = evalExpr(expr, ctx);
                        out += val;
                    } else if (i + 1 < n && tmpl[i+1] == '%') {
                        i += 2;
                        std::string tag = readTag();

                        if (tag.rfind("for ", 0) == 0) {
                            size_t inPos = tag.find(" in ");
                            if (inPos == std::string::npos) throw std::runtime_error("Malformed for tag: " + tag);
                            std::string varName = tag.substr(4, inPos - 4);
                            std::string iterName = tag.substr(inPos + 4);
                            size_t vts = varName.find_first_not_of(" \t");
                            size_t vte = varName.find_last_not_of(" \t");
                            varName = varName.substr(vts, vte - vts + 1);
                            size_t its = iterName.find_first_not_of(" \t");
                            size_t ite = iterName.find_last_not_of(" \t");
                            iterName = iterName.substr(its, ite - its + 1);

                            if (iterName == "messages") {
                                // Save position after the for tag
                                size_t bodyStart = i;
                                
                                // Find the matching endfor
                                size_t savedI = i;
                                size_t depth = 1;
                                while (i < n) {
                                    if (tmpl[i] == '{' && i + 1 < n && tmpl[i+1] == '%') {
                                        i += 2;
                                        std::string innerTag = readTag();
                                        if (innerTag.rfind("for ", 0) == 0) depth++;
                                        else if (innerTag == "endfor") {
                                            depth--;
                                            if (depth == 0) break;
                                        }
                                    } else {
                                        i++;
                                    }
                                }
                                size_t bodyEnd = i; // position at endfor tag
                                
                                // Now iterate
                                for (size_t mi = 0; mi < msgs.size(); mi++) {
                                    Ctx loopCtx = ctx;
                                    loopCtx.msg = &msgs[mi];
                                    loopCtx.loop_first = (mi == 0);
                                    loopCtx.loop_last = (mi == msgs.size() - 1);
                                    loopCtx.locals.push_back({varName, Var::fromStr(msgs[mi].content)});
                                    
                                    // Parse the body
                                    size_t bodyI = bodyStart;
                                    i = bodyStart;
                                    parseBlock(loopCtx);
                                }
                                
                                // Position i after endfor
                                i = bodyEnd;
                                // Skip past endfor tag (already consumed by readTag in the scan)
                                // Actually, readTag already advanced i past the endfor tag
                            } else {
                                throw std::runtime_error("Unsupported for iterator: " + iterName);
                            }
                        } else if (tag.rfind("if ", 0) == 0) {
                            std::string cond = tag.substr(3);
                            if (evalCond(cond, ctx)) {
                                parseBlock(ctx);
                            } else {
                                skipToTag("endif");
                            }
                        } else if (tag == "else") {
                            skipToTag("endif");
                        } else if (tag.rfind("elif ", 0) == 0) {
                            skipToTag("endif");
                        } else if (tag == "endif") {
                            return;
                        } else if (tag.rfind("set ", 0) == 0) {
                            std::string rest = tag.substr(4);
                            size_t eqPos = rest.find('=');
                            if (eqPos == std::string::npos) throw std::runtime_error("Malformed set: " + tag);
                            std::string varName = rest.substr(0, eqPos);
                            std::string expr = rest.substr(eqPos + 1);
                            size_t vts = varName.find_first_not_of(" \t");
                            size_t vte = varName.find_last_not_of(" \t");
                            varName = varName.substr(vts, vte - vts + 1);
                            std::string val = evalExpr(expr, ctx);
                            ctx.locals.push_back({varName, Var::fromStr(val)});
                        } else if (tag == "endfor") {
                            return;
                        } else {
                            throw std::runtime_error("Unknown tag: " + tag);
                        }
                    } else {
                        out += tmpl[i];
                        i++;
                    }
                } else {
                    out += tmpl[i];
                    i++;
                }
            }
        }

        void skipToTag(const std::string& tagName) {
            while (i < n) {
                if (tmpl[i] == '{' && i + 1 < n && tmpl[i+1] == '%') {
                    i += 2;
                    std::string tag = readTag();
                    if (tag == tagName) {
                        return;
                    }
                } else {
                    i++;
                }
            }
            throw std::runtime_error("Expected tag: " + tagName);
        }
    };

    Parser p(tmpl, msgs, add_generation_prompt, out, i);
    p.parseBlock(ctx);
    return out;
}

} // namespace chtmpl

#ifndef CHTMPL_NO_MAIN
#ifdef CHTMPL_TEST

#include <cassert>
#include <iostream>

int main() {
    using namespace chtmpl;

    // Test 1: Zephyr-style template
    {
        std::string tmpl = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}ASSISTANT: {% endif %}";
        std::vector<Msg> msgs = {
            {"user", "Hello"},
            {"assistant", "Hi there"}
        };
        std::string result = render(tmpl, msgs, true);
        std::string expected = "user: Hello\nassistant: Hi there\nASSISTANT: ";
        assert(result == expected);
    }

    // Test 2: ChatML template with add_generation_prompt true/false
    {
        std::string tmpl = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        std::vector<Msg> msgs = {
            {"user", "Hello"},
            {"assistant", "Hi"}
        };
        std::string resultTrue = render(tmpl, msgs, true);
        std::string resultFalse = render(tmpl, msgs, false);
        std::string expectedTrue = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n<|im_start|>assistant\n";
        std::string expectedFalse = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi<|im_end|>\n";
        assert(resultTrue == expectedTrue);
        assert(resultFalse == expectedFalse);
    }

    // Test 3: loop.last trailing newline
    {
        std::string tmpl = "{% for message in messages %}{{ message.content }}{% if loop.last %}\n{% endif %}{% endfor %}";
        std::vector<Msg> msgs = {
            {"user", "A"},
            {"user", "B"},
            {"user", "C"}
        };
        std::string result = render(tmpl, msgs, false);
        std::string expected = "ABC\n";
        assert(result == expected);
    }

    std::cout << "All tests passed.\n";
    return 0;
}

#endif // TEST
#endif // CHTMPL_NO_MAIN
