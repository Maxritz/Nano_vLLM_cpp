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
    std::string eos;  // bound to `eos_token` in templates
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
    if (name == "eos_token") return Var::fromStr(ctx.eos);
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

// Rewrite ident['key']/ident["key"] -> ident.key so bracket member access works.
inline std::string normBrackets(const std::string& in) {
    std::string out;
    for (size_t k = 0; k < in.size();) {
        size_t br = in.find('[', k);
        if (br == std::string::npos) { out += in.substr(k); break; }
        size_t q1 = br + 1;
        if (q1 < in.size() && (in[q1] == '\'' || in[q1] == '"')) {
            char qc = in[q1];
            size_t q2 = in.find(qc, q1 + 1);
            size_t cb = (q2 == std::string::npos) ? std::string::npos : in.find(']', q2 + 1);
            if (q2 != std::string::npos && cb == q2 + 1) {
                out += in.substr(k, br - k) + "." + in.substr(q1 + 1, q2 - q1 - 1);
                k = cb + 1;
                continue;
            }
        }
        out += in[k++];
    }
    return out;
}

inline std::string evalExpr(const std::string& expr, const Ctx& ctx);

inline std::string evalPrimary(const std::string& e, const Ctx& ctx) {
    std::string s = e;
    size_t a = s.find_first_not_of(" \t\r\n");
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    s = s.substr(a, b - a + 1);

    auto unescape = [](const std::string& t) {
        std::string o;
        for (size_t k = 0; k < t.size(); ++k) {
            if (t[k] == '\\' && k + 1 < t.size()) {
                char e = t[++k];
                o += (e == 'n') ? '\n' : (e == 't') ? '\t' : (e == 'r') ? '\r' : e;
            } else o += t[k];
        }
        return o;
    };
    if (s.size() >= 2 && s.front() == '\'' && s.back() == '\'') {
        return unescape(s.substr(1, s.size() - 2));
    }
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return unescape(s.substr(1, s.size() - 2));
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
    s = normBrackets(s);

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
    // top-level `or` of `and` chains (quote-aware split); falls back to single expr
    std::vector<std::vector<std::string>> ors;
    { char q = 0; size_t start = 0; std::vector<std::string> cur;
      auto flush = [&](size_t end) { cur.push_back(cond.substr(start, end - start)); };
      for (size_t k = 0; k < cond.size();) {
        char c = cond[k];
        if (q) { if (c == q) q = 0; ++k; continue; }
        if (c == '\'' || c == '"') { q = c; ++k; continue; }
        if (cond.compare(k, 4, " or ") == 0) { flush(k); ors.push_back(cur); cur.clear(); k += 4; start = k; continue; }
        ++k;
      }
      flush(cond.size()); ors.push_back(cur); }
    for (auto& ands : ors) {
      bool conj = true;
      for (auto& a : ands) {
        // split each `or`-arm on ` and `
        char q = 0; size_t start = 0;
        for (size_t k = 0; k <= a.size();) {
          char c = k < a.size() ? a[k] : 0;
          bool boundary = (k == a.size()) || (!q && a.compare(k, 5, " and ") == 0);
          if (!boundary) {
            if (q) { if (c == q) q = 0; }
            else if (c == '\'' || c == '"') q = c;
            ++k; continue;
          }
          if (evalExpr(a.substr(start, k - start), ctx) != "true") { conj = false; break; }
          k += 5; start = k;
        }
        if (!conj) break;
      }
      if (conj) return true;
    }
    return false;
}

} // namespace detail

inline std::string render(const std::string& tmpl, const std::vector<Msg>& msgs, bool add_generation_prompt,
                          const std::string& eos_token = "") {
    using namespace detail;
    std::string out;
    size_t i = 0;
    size_t n = tmpl.size();

    Ctx ctx;
    ctx.messages = &msgs;
    ctx.add_generation_prompt = add_generation_prompt;
    ctx.eos = eos_token;
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

        // Strip Jinja whitespace-control dashes + surrounding space for tag identity.
        static std::string normTag(std::string t) {
            size_t a = t.find_first_not_of(" \t\r\n-");
            if (a == std::string::npos) return "";
            size_t b = t.find_last_not_of(" \t\r\n-");
            return t.substr(a, b - a + 1);
        }
        static bool isWs(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
        void stripOutBack() { while (!out.empty() && isWs(out.back())) out.pop_back(); }

        void parseBlock(Ctx& ctx) {
            while (i < n) {
                if (tmpl[i] == '{') {
                    if (i + 1 < n && tmpl[i+1] == '{') {
                        i += 2;
                        if (i < n && tmpl[i] == '-') { stripOutBack(); i++; }
                        std::string expr = readUntil('}');
                        bool stripR = false;
                        { size_t e = expr.find_last_not_of(" \t\r\n");
                          if (e != std::string::npos && expr[e] == '-') {
                            expr.erase(e); stripR = true;
                            size_t e2 = expr.find_last_not_of(" \t\r\n");
                            expr = (e2 == std::string::npos) ? "" : expr.substr(0, e2 + 1);
                          } }
                        if (i < n && tmpl[i] == '}') i++;
                        if (i < n && tmpl[i] == '}') i++;
                        std::string val = evalExpr(expr, ctx);
                        out += val;
                        if (stripR) skipWs();
                    } else if (i + 1 < n && tmpl[i+1] == '%') {
                        i += 2;
                        if (i < n && tmpl[i] == '-') { stripOutBack(); i++; }
                        std::string tag = readTag();
                        bool stripR = false;
                        if (!tag.empty() && tag.back() == '-') {
                            tag.pop_back();
                            size_t e = tag.find_last_not_of(" \t\r\n");
                            tag = (e == std::string::npos) ? "" : tag.substr(0, e + 1);
                            stripR = true;
                        }
                        if (stripR) skipWs();

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
                                        std::string innerTag = normTag(readTag());
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
                                // Non-messages iterator (e.g. message.tool_calls): our Msg
                                // carries no lists, so the iterable is empty -> zero
                                // iterations (Jinja-correct). Skip to matching endfor.
                                { std::string iv = evalExpr(iterName, ctx);
                                  if (!(iv.empty() || iv == "false"))
                                    throw std::runtime_error("Unsupported non-empty for iterator: " + iterName); }
                                int depth = 1;
                                while (i < n && depth > 0) {
                                    if (tmpl[i] == '{' && i + 1 < n && tmpl[i+1] == '%') {
                                        i += 2;
                                        std::string t2 = normTag(readTag());
                                        if (t2.rfind("for ", 0) == 0) depth++;
                                        else if (t2 == "endfor") depth--;
                                    } else i++;
                                }
                            }
                        } else if (tag.rfind("if ", 0) == 0) {
                            parseIf(ctx, tag.substr(3));
                        } else if (tag == "else" || tag.rfind("elif ", 0) == 0 || tag == "endif") {
                            return;  // arm terminator: parseIf owns if-chains
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

        // Scan from i to the next elif/else/endif at depth 0 (nested if/for are
        // skipped). Leaves i at the tag START; term = normalized tag text,
        // termStripR = whether it closed with -%}. False at EOF.
        bool scanArm(std::string& term, bool& termStripR) {
            while (i < n) {
                if (tmpl[i] == '{' && i + 1 < n && tmpl[i+1] == '%') {
                    size_t save = i;
                    i += 2;
                    std::string tag = readTag();
                    std::string nt = normTag(tag);
                    if (nt.rfind("if ", 0) == 0 || nt.rfind("for ", 0) == 0) {
                        int depth = 1;
                        while (i < n && depth > 0) {
                            if (tmpl[i] == '{' && i + 1 < n && tmpl[i+1] == '%') {
                                i += 2;
                                std::string t2 = normTag(readTag());
                                if (t2.rfind("if ", 0) == 0 || t2.rfind("for ", 0) == 0) depth++;
                                else if (t2 == "endif" || t2 == "endfor") depth--;
                            } else i++;
                        }
                        continue;
                    }
                    if (nt.rfind("elif ", 0) == 0 || nt == "else" || nt == "endif") {
                        i = save; term = nt;
                        termStripR = (!tag.empty() && tag.back() == '-');
                        return true;
                    }
                } else i++;
            }
            return false;
        }

        // Full if/elif.../else/endif chain with correct arm selection.
        void parseIf(Ctx& ctx, const std::string& firstCond) {
            std::string cond = firstCond;
            for (;;) {
                size_t armStart = i;
                std::string term; bool termStripR = false;
                if (!scanArm(term, termStripR)) throw std::runtime_error("Unterminated if");
                i += 2; readTag();  // consume terminator
                if (termStripR) skipWs();
                if (term.rfind("elif ", 0) == 0) {
                    if (evalCond(cond, ctx)) {
                        i = armStart;
                        parseBlock(ctx);
                        skipToTag("endif");
                        return;
                    }
                    cond = term.substr(5);
                    continue;
                }
                if (term == "else") {
                    if (evalCond(cond, ctx)) { i = armStart; }
                    parseBlock(ctx);
                    return;
                }
                // endif
                if (evalCond(cond, ctx)) { i = armStart; parseBlock(ctx); }
                return;
            }
        }
        void skipToTag(const std::string& tagName) {
            int depth = 0;
            while (i < n) {
                if (tmpl[i] == '{' && i + 1 < n && tmpl[i+1] == '%') {
                    i += 2;
                    std::string tag = normTag(readTag());
                    if (tag.rfind("if ", 0) == 0 || tag.rfind("for ", 0) == 0) { depth++; continue; }
                    if (tag == "endif" || tag == "endfor") {
                        if (depth == 0) {
                            if (tag == tagName) return;
                            continue;  // stray close at depth 0: keep scanning
                        }
                        depth--;
                        continue;
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
