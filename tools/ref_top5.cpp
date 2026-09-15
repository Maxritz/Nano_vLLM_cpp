// Host fp32 prefill reference: exact-math adjudicator, no GPU, no Python.
// Mirrors Qwen3Model::forward_logits (including its qk-norm row quirk) in fp32.
// Usage: REF_MODEL=<path> [REF_PROMPT="..."] ref_top5
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "nanovllm/config.hpp"
#include "nanovllm/gguf.hpp"
#include "nanovllm/safetensors.hpp"
#include "nanovllm/tokenizer.hpp"

// Exact fp16/bf16 decoders (mirrors common.hpp; kept local so this tool stays
// plain C++ without HIP headers).
namespace {
inline float fp16_to_float(uint16_t h) {
  uint32_t sign = (uint32_t(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x03ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03ffu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    exp = exp - 15 + 127;
    bits = sign | (exp << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}
inline float bf16_to_float(uint16_t h) {
  uint32_t bits = uint32_t(h) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}
}  // namespace

namespace {
std::string ggmap(const std::string& n) {
  std::string s = n;
  auto rep = [&](const char* a, const char* b) {
    size_t p = s.find(a);
    if (p != std::string::npos) s.replace(p, strlen(a), b);
  };
  rep("model.embed_tokens.weight", "token_embd.weight");
  rep(".self_attn.q_proj.weight", ".attn_q.weight");
  rep(".self_attn.k_proj.weight", ".attn_k.weight");
  rep(".self_attn.v_proj.weight", ".attn_v.weight");
  rep(".self_attn.q_proj.bias", ".attn_q.bias");
  rep(".self_attn.k_proj.bias", ".attn_k.bias");
  rep(".self_attn.v_proj.bias", ".attn_v.bias");
  rep(".self_attn.o_proj.weight", ".attn_output.weight");
  rep(".mlp.gate_proj.weight", ".ffn_gate.weight");
  rep(".mlp.up_proj.weight", ".ffn_up.weight");
  rep(".mlp.down_proj.weight", ".ffn_down.weight");
  rep(".input_layernorm.weight", ".attn_norm.weight");
  rep(".post_attention_layernorm.weight", ".ffn_norm.weight");
  rep(".self_attn.q_norm.weight", ".attn_q_norm.weight");
  rep(".self_attn.k_norm.weight", ".attn_k_norm.weight");
  rep("model.norm.weight", "output_norm.weight");
  rep("lm_head.weight", "output.weight");
  rep("model.layers.", "blk.");
  return s;
}

struct WL {
  bool st_ = false, gg_ = false;
  SafetensorsLoader st;
  GGUFLoader gg;
  void open(const std::string& model) {
    namespace fs = std::filesystem;
    fs::path p(model);
    std::error_code ec;
    if (fs::is_directory(p, ec)) {
      bool has_st = false;
      for (auto& e : fs::directory_iterator(p, ec))
        if (e.path().extension() == ".safetensors") has_st = true;
      if (has_st) {
        st.add_directory(model);
        st_ = true;
        return;
      }
      std::string g = find_gguf_path(model);
      if (!g.empty()) {
        gg.add_file(g);
        gg_ = true;
        return;
      }
    }
    if (p.extension() == ".gguf") {
      gg.add_file(model);
      gg_ = true;
      return;
    }
    st.add_directory(model);
    st_ = true;
  }
  bool has(const std::string& n) const { return st_ ? st.contains(n) : gg.contains(ggmap(n)); }
  // Decoded fp32, row-major, exactly as the device kernels interpret the bytes.
  // K-quants decode here too (test oracle only; engines keep raw blocks).
  bool mat(const std::string& n, size_t n_elements, std::vector<float>& out) const {
    if (gg_) {
      std::vector<uint8_t> q8;
      if (gg.load_q8_0(ggmap(n), q8)) {
        size_t nblk = q8.size() / 34;
        out.resize(nblk * 32);
        for (size_t b = 0; b < nblk; ++b) {
          uint16_t sc;
          memcpy(&sc, &q8[b * 34], 2);
          float s = fp16_to_float(sc);
          for (int i = 0; i < 32; ++i) {
            int v = (int)q8[b * 34 + 2 + i];
            if (v > 127) v -= 256;
            out[b * 32 + i] = s * (float)v;
          }
        }
        return true;
      }
      for (int kind : {12, 13, 14}) {
        std::vector<uint8_t> qk;
        if (!gg.load_qk(ggmap(n), kind, n_elements, qk)) continue;
        size_t bb = GGUFLoader::qk_block_bytes(kind);
        out.resize(n_elements);
        for (size_t b = 0, nb = n_elements / 256; b < nb; ++b) {
           const uint8_t* blk = &qk[b * bb];
           if (kind == 12) {
            uint16_t dh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            uint16_t dmh = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
            float d = fp16_to_float(dh), dmin = fp16_to_float(dmh);
            for (int j = 0; j < 8; ++j) {
              uint32_t sc, mn;
              if (j < 4) {
                sc = blk[4 + j] & 63;
                mn = blk[4 + j + 4] & 63;
              } else {
                sc = (blk[4 + j + 4] & 0xF) | ((blk[4 + j - 4] >> 6) << 4);
                mn = (blk[4 + j + 4] >> 4) | ((blk[4 + j] >> 6) << 4);
              }
              float d1 = d * sc, m1 = dmin * mn;
              int base = (j >> 1) * 32, sh = (j & 1) * 4;
              for (int l = 0; l < 16; ++l) {
                out[b * 256 + j * 32 + 2 * l] = d1 * ((blk[16 + base + l] >> sh) & 0xF) - m1;
                out[b * 256 + j * 32 + 2 * l + 1] = d1 * ((blk[16 + base + 16 + l] >> sh) & 0xF) - m1;
              }
            }
          } else if (kind == 13) {  // Q5_K: [d][dmin][12B scales][32B qh][128B qs]
            uint16_t dhh = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            uint16_t dmh = (uint16_t)blk[2] | ((uint16_t)blk[3] << 8);
            float d = fp16_to_float(dhh), dmin = fp16_to_float(dmh);
            const uint8_t* ql = blk + 48;
            const uint8_t* qh = blk + 16;
            for (int group = 0; group < 4; ++group) {
              uint8_t u1 = 1u << (group * 2), u2 = 2u << (group * 2);
              for (int slot = 0; slot < 2; ++slot) {
                uint32_t sc, mn;
                int j = group * 2 + slot;
                if (j < 4) { sc = blk[4 + j] & 63; mn = blk[4 + j + 4] & 63; }
                else { sc = (blk[4 + j + 4] & 0xF) | ((blk[4 + j - 4] >> 6) << 4); mn = (blk[4 + j + 4] >> 4) | ((blk[4 + j] >> 6) << 4); }
                float d1 = d * sc, m1 = dmin * mn;
                for (int l = 0; l < 32; ++l) {
                  uint8_t qhl = qh[l];
                  uint8_t q = (slot == 0) ? (ql[group * 32 + l] & 0xF) : (ql[group * 32 + l] >> 4);
                  uint8_t hbit = (slot == 0) ? (qhl & u1) : (qhl & u2);
                   out[b * 256 + group * 64 + slot * 32 + l] = d1 * (q + (hbit ? 16 : 0)) - m1;
                 }
               }
             }
           } else {  // Q6_K
            uint16_t dh = (uint16_t)blk[208] | ((uint16_t)blk[209] << 8);
            float d = fp16_to_float(dh);
            for (int half = 0; half < 2; ++half) {
              const uint8_t* ql = blk + half * 64;
              const uint8_t* qh = blk + 128 + half * 32;
              const int8_t* sc = (const int8_t*)(blk + 192 + half * 8);
              for (int l = 0; l < 32; ++l) {
                int is = l >> 4;
                int q1 = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
                int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                out[b * 256 + half * 128 + l] = d * sc[is + 0] * (q1 - 32);
                out[b * 256 + half * 128 + l + 32] = d * sc[is + 2] * (q2 - 32);
                out[b * 256 + half * 128 + l + 64] = d * sc[is + 4] * (q3 - 32);
                out[b * 256 + half * 128 + l + 96] = d * sc[is + 6] * (q4 - 32);
              }
            }
          }
        }
        return true;
      }
      std::vector<float> up;
      if (gg.load_upcast_f32(ggmap(n), up)) { out.swap(up); return true; }
    }
    std::vector<uint16_t> u;
    bool bf = false;
    if (!(st_ ? st.load_u16(n, u, bf) : gg.load_u16(ggmap(n), u, bf))) return false;
    out.resize(u.size());
    for (size_t i = 0; i < u.size(); ++i) out[i] = bf ? bf16_to_float(u[i]) : fp16_to_float(u[i]);
    return true;
  }
  bool vec(const std::string& n, std::vector<float>& out) const {
    // F32 norm weights via load_float when available, else u16 decode.
    if (st_ ? st.load_float(n, out) : gg.load_float(ggmap(n), out)) return true;
    std::vector<uint16_t> u;
    bool bf = false;
    if (!(st_ ? st.load_u16(n, u, bf) : gg.load_u16(ggmap(n), u, bf))) return false;
    out.resize(u.size());
    for (size_t i = 0; i < u.size(); ++i) out[i] = bf ? bf16_to_float(u[i]) : fp16_to_float(u[i]);
    return true;
  }
};

void matvec(const std::vector<float>& W, const std::vector<float>& x, int n, int k,
            std::vector<float>& y) {
  y.assign(n, 0.0f);
  for (int j = 0; j < n; ++j) {
    double acc = 0;
    const float* w = &W[(size_t)j * k];
    for (int i = 0; i < k; ++i) acc += (double)w[i] * (double)x[i];
    y[j] = (float)acc;
  }
}
void rms_norm(const std::vector<float>& x, const std::vector<float>& w, int rows, int h, float eps,
              std::vector<float>& y) {
  y.resize((size_t)rows * h);
  for (int r = 0; r < rows; ++r) {
    double s = 0;
    for (int i = 0; i < h; ++i) s += (double)x[(size_t)r * h + i] * x[(size_t)r * h + i];
    float inv = (float)(1.0 / std::sqrt(s / h + eps));
    for (int i = 0; i < h; ++i) y[(size_t)r * h + i] = x[(size_t)r * h + i] * inv * w[i];
  }
}
}  // namespace

int main() {
  const char* model = std::getenv("REF_MODEL");
  if (!model) {
    std::fprintf(stderr, "set REF_MODEL\n");
    return 2;
  }
  const char* prompt = std::getenv("REF_PROMPT");
  if (!prompt) prompt = "The capital of France is";
  Config config;
  config.model = model;
  config.max_model_len = 512;
   config.load_hf_config();
   auto& hf = config.hf;
   int hidden = hf.hidden_size, inter = hf.intermediate_size, heads = hf.num_attention_heads,
      kvh = hf.num_key_value_heads, hd = hf.head_dim, nl = hf.num_hidden_layers;
  int qs = heads * hd, ks = kvh * hd;
  float eps = (float)hf.rms_norm_eps, scale = 1.0f / std::sqrt((float)hd);
  Tokenizer tok;
  tok.set_fallback_vocab_size(hf.vocab_size);
  tok.load(config.model);
  std::vector<int> ids = tok.encode_text(prompt);
  if (const char* ti = std::getenv("REF_TOKENS")) {
    ids.clear();
    for (const char* p = ti; *p;) { ids.push_back(atoi(p)); while (*p && *p != ',') ++p; if (*p == ',') ++p; }
  }
  if (std::getenv("REF_CHAT")) {
    int im_start = tok.encode_special("<|im_start|>");
    int im_end = tok.encode_special("<|im_end|>");
    std::vector<int> c;
    auto append = [&](const std::string& s) {
      auto part = tok.encode_text(s);
      c.insert(c.end(), part.begin(), part.end());
    };
    if (im_start >= 0 && im_end >= 0) {
      if (tok.chat_auto_system()) {
        c.push_back(im_start);
        append("system\nYou are a helpful assistant.");
        c.push_back(im_end);
        append("\n");
      }
      c.push_back(im_start);
      append(std::string("user\n") + prompt);
      c.push_back(im_end);
      append("\n");
      c.push_back(im_start);
      append("assistant\n");
    } else {
      append(prompt);
    }
    ids.swap(c);
  }
  int T = (int)ids.size();
  WL wl;
  wl.open(model);
  std::vector<float> embed;
  if (!wl.mat("model.embed_tokens.weight", (size_t)hf.vocab_size * hidden, embed)) {
    std::fprintf(stderr, "no embed\n");
    return 1;
  }
  std::vector<float> hidden_s((size_t)T * hidden), resid((size_t)T * hidden, 0.0f);
  for (int t = 0; t < T; ++t)
    memcpy(&hidden_s[(size_t)t * hidden], &embed[(size_t)ids[t] * hidden], hidden * 4);
  std::vector<float> invf(hd / 2);
  for (int i = 0; i < hd / 2; ++i)
    invf[i] = (float)(1.0 / std::pow(hf.rope_theta, (2.0 * i) / hd));
  std::vector<float> K((size_t)nl * T * ks), V((size_t)nl * T * ks);
  std::vector<float> norm, q, k, v, attn, tmp, gup, mlp;
  std::vector<float> inln, postln, qn, kn, o, ga, up, dn, fn;
  if (!wl.vec("model.norm.weight", fn)) {
    fn.assign(hidden, 1.0f);
  }
  auto ref_tag = [&](const std::string& stage, const std::vector<float>& h) {
    if (!std::getenv("NANO_PIPE")) return;
    double sum = 0; for (float x : h) sum += x;
    std::fprintf(stderr, "[PIPE ref %s] n=%zu sum=%.6f first=%.6f,%.6f,%.6f\n",
                 stage.c_str(), h.size(), sum, h[0], h[1], h[2]);
  };
  ref_tag("embed", hidden_s);
  for (int L = 0; L < nl; ++L) {
    std::string p = "model.layers." + std::to_string(L);
    wl.vec(p + ".input_layernorm.weight", inln);
    wl.vec(p + ".post_attention_layernorm.weight", postln);
    if (inln.empty()) inln.assign(hidden, 1.0f);
    if (postln.empty()) postln.assign(hidden, 1.0f);
    // rms_norm_add
    norm.resize((size_t)T * hidden);
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < hidden; ++i) resid[(size_t)t * hidden + i] += hidden_s[(size_t)t * hidden + i];
    rms_norm(resid, inln, T, hidden, eps, norm);
    std::vector<float> Wq, Wk, Wv, Wo, bq, bk, bv;
    wl.mat(p + ".self_attn.q_proj.weight", (size_t)qs * hidden, Wq);
    wl.mat(p + ".self_attn.k_proj.weight", (size_t)ks * hidden, Wk);
    wl.mat(p + ".self_attn.v_proj.weight", (size_t)ks * hidden, Wv);
    wl.mat(p + ".self_attn.o_proj.weight", (size_t)hidden * qs, Wo);
    wl.vec(p + ".self_attn.q_proj.bias", bq);
    wl.vec(p + ".self_attn.k_proj.bias", bk);
    wl.vec(p + ".self_attn.v_proj.bias", bv);
    if (L == 0 && std::getenv("NANO_PIPE")) {
      double wq=0,wk=0,wv=0; for(float x:Wq)wq+=x; for(float x:Wk)wk+=x; for(float x:Wv)wv+=x;
      double b0lo=0,b0hi=0,b1lo=0,b1hi=0;
      for(int i=0;i<128;i++){b0lo+=Wv[i];b0hi+=Wv[128+i];}
      for(int i=0;i<128;i++){b1lo+=Wv[256+i];b1hi+=Wv[384+i];}
      std::fprintf(stderr,"[PIPE refweight vsum=%.4f b0lo=%.4f b0hi=%.4f b1lo=%.4f b1hi=%.4f n=%zu\n",
        wv,b0lo,b0hi,b1lo,b1hi,Wv.size());
    }
    q.resize((size_t)T * qs);
    k.resize((size_t)T * ks);
    v.resize((size_t)T * ks);
    for (int t = 0; t < T; ++t) {
      std::vector<float> x(hidden);
      memcpy(x.data(), &norm[(size_t)t * hidden], hidden * 4);
      std::vector<float> yq, yk, yv;
      matvec(Wq, x, qs, hidden, yq);
      matvec(Wk, x, ks, hidden, yk);
      matvec(Wv, x, ks, hidden, yv);
      for (int i = 0; i < qs && i < (int)bq.size(); ++i) yq[i] += bq[i];
      for (int i = 0; i < ks && i < (int)bk.size(); ++i) yk[i] += bk[i];
      for (int i = 0; i < ks && i < (int)bv.size(); ++i) yv[i] += bv[i];
      memcpy(&q[(size_t)t * qs], yq.data(), yq.size() * 4);
      memcpy(&k[(size_t)t * ks], yk.data(), yk.size() * 4);
      memcpy(&v[(size_t)t * ks], yv.data(), yv.size() * 4);
    }
    bool has_qn = wl.vec(p + ".self_attn.q_norm.weight", qn) && !qn.empty();
    if (L == 0) { ref_tag("L0.q", q); ref_tag("L0.k", k); ref_tag("L0.v", v); }
    bool has_kn = wl.vec(p + ".self_attn.k_norm.weight", kn) && !kn.empty();
    if (has_qn) {
      std::vector<float> nq;
      rms_norm(q, qn, T * heads, hd, eps, nq);
      memcpy(q.data(), nq.data(), nq.size() * 4);
    }
    if (has_kn) {
      std::vector<float> nk;
      rms_norm(k, kn, T * kvh, hd, eps, nk);
      memcpy(k.data(), nk.data(), nk.size() * 4);
    }
    // rope
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < heads; ++h)
        for (int d = 0; d < hd / 2; ++d) {
          float fr = (float)t * invf[d], c = cosf(fr), s = sinf(fr);
          size_t b = (size_t)t * qs + h * hd;
          float x1 = q[b + d], x2 = q[b + d + hd / 2];
          q[b + d] = x1 * c - x2 * s;
          q[b + d + hd / 2] = x2 * c + x1 * s;
        }
      for (int h = 0; h < kvh; ++h)
        for (int d = 0; d < hd / 2; ++d) {
          float fr = (float)t * invf[d], c = cosf(fr), s = sinf(fr);
          size_t b = (size_t)t * ks + h * hd;
          float x1 = k[b + d], x2 = k[b + d + hd / 2];
          k[b + d] = x1 * c - x2 * s;
          k[b + d + hd / 2] = x2 * c + x1 * s;
        }
    }
    if (L == 0) { ref_tag("L0.qrope", q); ref_tag("L0.krope", k); }
    memcpy(&K[(size_t)L * T * ks], k.data(), k.size() * 4);
    memcpy(&V[(size_t)L * T * ks], v.data(), v.size() * 4);
    // causal attention
    attn.assign((size_t)T * qs, 0);
    int gqa = heads / kvh;
    std::vector<float> sc(T);
    for (int t = 0; t < T; ++t)
      for (int h = 0; h < heads; ++h) {
        int kh = h / gqa;
        double mx = -1e30;
        for (int u = 0; u <= t; ++u) {
          double dot = 0;
          for (int d = 0; d < hd; ++d)
            dot += (double)q[(size_t)t * qs + h * hd + d] * K[(size_t)L * T * ks + (size_t)u * ks + kh * hd + d];
          sc[u] = (float)(dot * scale);
          if (sc[u] > mx) mx = sc[u];
        }
        double se = 0;
        for (int u = 0; u <= t; ++u) se += std::exp(sc[u] - mx);
        for (int d = 0; d < hd; ++d) {
          double a = 0;
          for (int u = 0; u <= t; ++u)
            a += std::exp(sc[u] - mx) * V[(size_t)L * T * ks + (size_t)u * ks + kh * hd + d];
          attn[(size_t)t * qs + h * hd + d] = (float)(a / se);
        }
      }
     if (L == 0) ref_tag("L0.attn", attn);
     for (int t = 0; t < T; ++t) {
       std::vector<float> x(qs);
       memcpy(x.data(), &attn[(size_t)t * qs], qs * 4);
       std::vector<float> y;
       matvec(Wo, x, hidden, qs, y);
       memcpy(&hidden_s[(size_t)t * hidden], y.data(), y.size() * 4);
     }
     ref_tag("L" + std::to_string(L) + ".attnout", hidden_s);
     // post norm + mlp
    for (int t = 0; t < T; ++t)
      for (int i = 0; i < hidden; ++i) resid[(size_t)t * hidden + i] += hidden_s[(size_t)t * hidden + i];
    rms_norm(resid, postln, T, hidden, eps, norm);
    if (hf.num_experts > 0 && ((L + 1) % hf.decoder_sparse_step == 0)) {
      // Sparse layer: compute only the last token's MoE output (oracle feeds last-row logits).
      const int E = hf.num_experts, K = hf.num_experts_per_tok, I = hf.moe_intermediate_size;
      const int SI = hf.shared_expert_intermediate_size;
      std::vector<float> Wr;
      if (!wl.mat(p + ".mlp.gate.weight", (size_t)E * hidden, Wr))
        throw std::runtime_error("missing router " + p);
      std::vector<float> Sg, Su, Sd, Sgw;
      bool has_sh = (SI > 0 && !std::getenv("REF_NO_SHARED"));
      if (has_sh) {
        wl.mat(p + ".mlp.shared_expert.gate_proj.weight", (size_t)SI * hidden, Sg);
        wl.mat(p + ".mlp.shared_expert.up_proj.weight", (size_t)SI * hidden, Su);
        wl.mat(p + ".mlp.shared_expert.down_proj.weight", (size_t)hidden * SI, Sd);
        wl.mat(p + ".mlp.shared_expert_gate.weight", (size_t)hidden, Sgw);
        if (Sgw.size() != (size_t)hidden) Sgw.clear();
      }
      std::vector<std::vector<float>> Eg_c(E), Eu_c(E), Ed_c(E);
      std::vector<char> loaded(E, 0);
      for (int t = 0; t < T; ++t) {
      std::vector<float> x(hidden);
      memcpy(x.data(), &norm[(size_t)t * hidden], hidden * 4);
      std::vector<float> rl;
      matvec(Wr, x, E, hidden, rl);
      std::vector<int> perm(E);
      for (int i = 0; i < E; ++i) perm[i] = i;
      std::vector<double> pr2(E);
      {
        double m2 = -1e300, s2 = 0;
        for (int i = 0; i < E; ++i) m2 = std::max(m2, (double)rl[i]);
        for (int i = 0; i < E; ++i) { pr2[i] = std::exp((double)rl[i] - m2); s2 += pr2[i]; }
        for (int i = 0; i < E; ++i) pr2[i] /= s2;
      }
      std::vector<int> idx(K);
      std::vector<double> w(K);
      for (int i = 0; i < K; ++i) {
        int best = i;
        for (int j = i + 1; j < E; ++j) if (pr2[perm[j]] > pr2[perm[best]]) best = j;
        std::swap(perm[i], perm[best]);
        idx[i] = perm[i];
        w[i] = pr2[perm[i]];
      }
      if (hf.norm_topk_prob) {
        double s3 = 0;
        for (int i = 0; i < K; ++i) s3 += w[i];
        for (int i = 0; i < K; ++i) w[i] /= s3;
      }
      std::vector<float> y(hidden, 0.f);
      if (t == T - 1) {
        std::fprintf(stderr, "REF route L%d last:", L);
        for (int i = 0; i < K; ++i) std::fprintf(stderr, " %d(%.4f)", idx[i], w[i]);
        std::fprintf(stderr, "\n");
      }
      if (has_sh) {
        std::vector<float> gg, uu, mm;
        matvec(Sg, x, SI, hidden, gg);
        matvec(Su, x, SI, hidden, uu);
        mm.resize(SI);
        for (int i = 0; i < SI; ++i) mm[i] = gg[i] / (1.0f + std::exp(-gg[i])) * uu[i];
        std::vector<float> ys;
        matvec(Sd, mm, hidden, SI, ys);
        float gs = 0.f;
        for (int i = 0; i < (int)Sgw.size(); ++i) gs += Sgw[i] * x[i];
        float gsc = Sgw.empty() ? 1.0f : 1.0f / (1.0f + std::exp(-gs));
        for (int i = 0; i < hidden; ++i) y[i] += gsc * ys[i];
      }
      for (int k = 0; k < K; ++k) {
        int e = idx[k];
        if (!loaded[e]) {
          std::string pe = p + ".mlp.experts." + std::to_string(e) + ".";
          if (!wl.mat(pe + "gate_proj.weight", (size_t)I * hidden, Eg_c[e])) throw std::runtime_error("missing " + pe);
          wl.mat(pe + "up_proj.weight", (size_t)I * hidden, Eu_c[e]);
          wl.mat(pe + "down_proj.weight", (size_t)hidden * I, Ed_c[e]);
          loaded[e] = 1;
        }
        std::vector<float> gg, uu, mm;
        matvec(Eg_c[e], x, I, hidden, gg);
        matvec(Eu_c[e], x, I, hidden, uu);
        mm.resize(I);
        for (int i = 0; i < I; ++i) mm[i] = gg[i] / (1.0f + std::exp(-gg[i])) * uu[i];
        std::vector<float> ys;
        matvec(Ed_c[e], mm, hidden, I, ys);
        for (int i = 0; i < hidden; ++i) y[i] += (float)w[k] * ys[i];
      }
      memcpy(&hidden_s[(size_t)t * hidden], y.data(), hidden * 4);
      if (t == T - 1 && L < 2) { double am = 0; for (float v : y) am = std::max(am, (double)std::fabs(v)); std::fprintf(stderr, "REF moe L%d |y|_max=%.4f\n", L, am); }
      }
    } else {
    std::vector<float> Wg, Wu, Wd;
    wl.mat(p + ".mlp.gate_proj.weight", (size_t)inter * hidden, Wg);
    wl.mat(p + ".mlp.up_proj.weight", (size_t)inter * hidden, Wu);
    wl.mat(p + ".mlp.down_proj.weight", (size_t)hidden * inter, Wd);
    for (int t = 0; t < T; ++t) {
      std::vector<float> x(hidden);
      memcpy(x.data(), &norm[(size_t)t * hidden], hidden * 4);
      std::vector<float> g, u;
      matvec(Wg, x, inter, hidden, g);
      matvec(Wu, x, inter, hidden, u);
      std::vector<float> m(inter);
      for (int i = 0; i < inter; ++i) m[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
      std::vector<float> y;
      matvec(Wd, m, hidden, inter, y);
      memcpy(&hidden_s[(size_t)t * hidden], y.data(), y.size() * 4);
     }
     }
     ref_tag("L" + std::to_string(L) + ".mlpout", hidden_s);
  }
  for (int i = 0; i < hidden; ++i) resid[(size_t)(T - 1) * hidden + i] += hidden_s[(size_t)(T - 1) * hidden + i];
  std::vector<float> last(hidden), nl2(hidden);
  memcpy(last.data(), &resid[(size_t)(T - 1) * hidden], hidden * 4);
  rms_norm(last, fn, 1, hidden, eps, nl2);
  std::vector<float> lm;
  if (hf.tie_word_embeddings)
    lm = embed;
  else if (!wl.mat("lm_head.weight", (size_t)hf.vocab_size * hidden, lm))
    lm = embed;
  std::vector<float> logits;
  matvec(lm, nl2, hf.vocab_size, hidden, logits);
  std::vector<int> top(5, -1);
  std::vector<float> tv(5, -1e30f);
  for (int i = 0; i < hf.vocab_size; ++i)
    if (logits[i] > tv[4]) {
      tv[4] = logits[i];
      top[4] = i;
      for (int kk = 3; kk >= 0; --kk)
        if (tv[kk + 1] > tv[kk]) {
          std::swap(tv[kk], tv[kk + 1]);
          std::swap(top[kk], top[kk + 1]);
        }
    }
   std::printf("ref top5:");
   for (int k = 0; k < 5; ++k) std::printf(" %d(%.3f)", top[k], tv[k]);
   std::printf("\n");
   return 0;
}
