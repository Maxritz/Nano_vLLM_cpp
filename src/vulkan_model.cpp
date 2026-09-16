#define NOMINMAX
#include "nanovllm/vulkan_model.hpp"
#include "nanovllm/common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

static void warm_pages(const uint8_t* data, size_t bytes) {
    if (!data || bytes == 0) return;
#ifdef _WIN32
    HMODULE hKernel = GetModuleHandleA("kernel32.dll");
    if (hKernel) {
        using PrefetchFunc = void (*)(void*, size_t, unsigned);
        if (auto fn = (PrefetchFunc)GetProcAddress(hKernel, "PrefetchVirtualMemory")) {
            fn(const_cast<uint8_t*>(data), bytes, 2);
            return;
        }
    }
#endif
    const size_t page = 4096;
    for (size_t off = 0; off < bytes; off += page)
        (void)data[off];
}

namespace {
// GGUF-MoE quantized experts: stacked raw K-quant blocks per layer, kept
// quantized on host; dequantized per expert into ms.stage at place time.
// (VulkanMoeState has no quantized members and its header is owned elsewhere,
// so layer blobs live here, keyed by layer; cleared on every load_weights.)
struct GgufMoeQ {
    std::string dtype;  // "Q4_K" / "Q6_K"
    int kind = 0;       // ggml type id (12/14)
    std::vector<uint8_t> gate, up, down;  // stacked [E, I, H] raw blocks
};
std::map<int, GgufMoeQ> g_gguf_moe;
// GGUF MoE metadata (config.hpp fill_from_gguf reads dense fields only and is
// owned elsewhere: MoE counts are parsed here, nowhere else).
void apply_gguf_moe_meta(const GGUFLoader& g, HFConfig& hf) {
    std::string arch;
    if (!g.meta_str("general.architecture", arch) || arch.empty()) return;
    uint32_t u = 0;
    if (g.meta_u32(arch + ".expert_count", u)) hf.num_experts = static_cast<int>(u);
    if (g.meta_u32(arch + ".expert_used_count", u)) hf.num_experts_per_tok = static_cast<int>(u);
    if (g.meta_u32(arch + ".expert_feed_forward_length", u)) hf.moe_intermediate_size = static_cast<int>(u);
    if (g.meta_u32(arch + ".expert_shared_feed_forward_length", u))
        hf.shared_expert_intermediate_size = static_cast<int>(u);
}
}  // namespace

void VulkanModel::moe_place(VLayerWeights::VulkanMoeState& ms, int layer,
                            const std::vector<int32_t>& h_idx, int hidden, int I) {
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    std::vector<int32_t> uniq = h_idx;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    if ((int)uniq.size() > ms.slots)
        throw std::runtime_error("moe_place: unique experts exceed slots (chunking invariant broken)");
    const size_t gu_n = (size_t)2 * I * hidden, dn_n = (size_t)I * hidden;
    bool dirty = false;
    for (int32_t e : uniq) {
        if (e < 0 || e >= ms.E) continue;
        if (ms.h_slot_of[e] >= 0) {
            ms.h_lru[ms.h_slot_of[e]] = ++ms.clock;
            expert_cache_.touch(layer, e, ++ms.clock);
            continue;
        }
        // Check global cache: if pages are already warm, skip warm_pages
        if (expert_cache_.get(layer, e) < 0) {
            // Not in global cache — this is a new expert, warm pages
        }
        int slot = -1;
        for (int s = 0; s < ms.slots; ++s)
            if (ms.h_slot_exp[s] < 0) { slot = s; break; }
        if (slot < 0) {
            slot = 0;
            for (int s = 1; s < ms.slots; ++s)
                if (ms.h_lru[s] < ms.h_lru[slot]) slot = s;
            ms.h_slot_of[ms.h_slot_exp[slot]] = -1;
            expert_cache_.put(layer, ms.h_slot_exp[slot], -1, ++ms.clock);
            ms.evictions++;
        }
        uint16_t* sg = ms.stage.data();
        uint16_t* sd = sg + gu_n;
        auto qit = g_gguf_moe.find(layer);
        if (qit != g_gguf_moe.end()) {
            // GGUF MoE: experts stay quantized; dequant per-expert into stage.
            const GgufMoeQ& q = qit->second;
            size_t nb = dn_n / GGUFLoader::upcast_block_vals(q.dtype);
            size_t stride = nb * GGUFLoader::upcast_block_bytes(q.dtype);
            warm_pages(q.gate.data() + (size_t)e * stride, stride);
            if (!GGUFLoader::dequant_blocks_f16(q.dtype, q.gate.data() + (size_t)e * stride, dn_n, sg) ||
                !GGUFLoader::dequant_blocks_f16(q.dtype, q.up.data() + (size_t)e * stride, dn_n, sg + dn_n) ||
                !GGUFLoader::dequant_blocks_f16(q.dtype, q.down.data() + (size_t)e * stride, dn_n, sd))
                throw std::runtime_error("moe_place: GGUF expert dequant failed");
        } else if (ms.stacked) {
            std::memcpy(sg, ms.s_gu + (size_t)e * gu_n, gu_n * 2);
            warm_pages((const uint8_t*)(ms.s_dn + (size_t)e * dn_n), dn_n * sizeof(uint16_t));
            std::memcpy(sd, ms.s_dn + (size_t)e * dn_n, dn_n * 2);
        } else {
            warm_pages((const uint8_t*)ms.f_g[e], dn_n * sizeof(uint16_t));
            std::memcpy(sg, ms.f_g[e], dn_n * 2);
            warm_pages((const uint8_t*)ms.f_u[e], dn_n * sizeof(uint16_t));
            std::memcpy(sg + dn_n, ms.f_u[e], dn_n * 2);
            warm_pages((const uint8_t*)ms.f_d[e], dn_n * sizeof(uint16_t));
            std::memcpy(sd, ms.f_d[e], dn_n * 2);
        }
        // Async: transfer runs while the host places remaining experts; the
        // submit_wait after the moe_ffn loop orders + reclaims them.
        ms.slot_gu.upload_at_async((VkDeviceSize)slot * gu_n * sizeof(uint16_t), sg, (VkDeviceSize)(gu_n * sizeof(uint16_t)));
        ms.slot_dn.upload_at_async((VkDeviceSize)slot * dn_n * sizeof(uint16_t), sd, (VkDeviceSize)(dn_n * sizeof(uint16_t)));
        ms.h_slot_of[e] = slot;
        ms.h_slot_exp[slot] = e;
        ms.h_lru[slot] = ++ms.clock;
        expert_cache_.put(layer, e, slot, ++ms.clock);
        ms.loads++;
        dirty = true;
    }
    if (dirty)
        ms.slot_of.upload(ms.h_slot_of.data(),
                          (VkDeviceSize)(ms.h_slot_of.size() * sizeof(int32_t)));
}

VulkanModel::VulkanModel(const Config& config) : config_(config) {
    config_.validate();
    const auto& hf = config_.hf;
    q_size_  = hf.num_attention_heads * hf.head_dim;
    kv_size_ = hf.num_key_value_heads * hf.head_dim;
    qkv_size_ = q_size_ + 2 * kv_size_;
    kv_head_stride_ = hf.num_key_value_heads * hf.head_dim;
    block_size_ = config_.kvcache_block_size;
    layers_.resize(hf.num_hidden_layers);
    std::vector<float> inv_freq(hf.head_dim / 2);
    for (int i = 0; i < hf.head_dim / 2; ++i)
        inv_freq[i] = static_cast<float>(1.0 / std::pow(hf.rope_theta, (2.0 * i) / hf.head_dim));
    dev_ = VulkanBackend::Create(true);
    inv_freq_ = dev_->make(inv_freq);
}
VulkanModel::~VulkanModel() = default;

std::string VulkanModel::map_name(const std::string& n) {
    std::string s = n;
    auto rep = [&](const std::string& a, const std::string& b) {
        size_t p = s.find(a); if (p != std::string::npos) s.replace(p, a.size(), b);
    };
    // GGUF MoE (qwen2-style blk.* names, see mkmicro_moe_gguf.py) + nested prefix.
    rep("model.language_model.layers.", "blk.");
    rep("model.language_model.embed_tokens.weight", "token_embd.weight");
    rep("model.language_model.norm.weight", "output_norm.weight");
    rep(".mlp.gate.weight", ".ffn_gate_inp.weight");
    rep(".mlp.experts.gate_exps.weight", ".ffn_gate_exps.weight");
    rep(".mlp.experts.up_exps.weight", ".ffn_up_exps.weight");
    rep(".mlp.experts.down_exps.weight", ".ffn_down_exps.weight");
    rep(".mlp.shared_expert.gate_proj.weight", ".ffn_gate_shexp.weight");
    rep(".mlp.shared_expert.up_proj.weight", ".ffn_up_shexp.weight");
    rep(".mlp.shared_expert.down_proj.weight", ".ffn_down_shexp.weight");
    rep(".mlp.shared_expert_gate.weight", ".ffn_shexp_gate.weight");
    rep("model.embed_tokens.weight", "token_embd.weight");
    rep(".self_attn.q_proj.weight", ".attn_q.weight");
    rep(".self_attn.k_proj.weight", ".attn_k.weight");
    rep(".self_attn.v_proj.weight", ".attn_v.weight");
    rep(".self_attn.o_proj.weight", ".attn_output.weight");
    rep(".self_attn.q_proj.bias", ".attn_q.bias");
    rep(".self_attn.k_proj.bias", ".attn_k.bias");
    rep(".self_attn.v_proj.bias", ".attn_v.bias");
    rep(".mlp.gate_proj.weight", ".ffn_gate.weight");
    rep(".mlp.up_proj.weight", ".ffn_up.weight");
    rep(".mlp.down_proj.weight", ".ffn_down.weight");
    rep(".mlp.gate_up_proj.weight", ".ffn_gate_up.weight");
    rep(".input_layernorm.weight", ".attn_norm.weight");
    rep(".post_attention_layernorm.weight", ".ffn_norm.weight");
    rep(".self_attn.q_norm.weight", ".attn_q_norm.weight");
    rep(".self_attn.k_norm.weight", ".attn_k_norm.weight");
    rep("model.norm.weight", "output_norm.weight");
    rep("lm_head.weight", "output.weight");
    rep("model.layers.", "blk.");
    return s;
}
void VulkanModel::load_model_dir(const std::string& model_dir) {
    std::filesystem::path p(model_dir);
    if (p.has_extension() && p.extension() == ".gguf") { gg_loader_.add_file(model_dir); gguf_ = true; st_ = false; return; }
    std::error_code ec;
    if (std::filesystem::is_directory(p, ec)) {
        bool has_st = false;
        for (auto& e : std::filesystem::directory_iterator(p, ec))
            if (e.path().extension() == ".safetensors") has_st = true;
        if (has_st) { st_loader_.add_directory(model_dir); st_ = true; gguf_ = false; return; }
        std::string gg = find_gguf_path(model_dir);
        if (!gg.empty()) { gg_loader_.add_file(gg); gguf_ = true; st_ = false; return; }
    }
    st_loader_.add_directory(model_dir); st_ = true; gguf_ = false;
}
bool VulkanModel::contains(const std::string& n) const { return st_ ? st_loader_.contains(n) : gg_loader_.contains(map_name(n)); }

std::string VulkanModel::moe_key_prefix(int layer) const {
    // Resolve the MLP key prefix for this layer. Handles:
    //   Qwen3-MoE:        model.layers.N.mlp
    //   Qwen3.5-MoE full: language_model.model.layers.N.mlp (or .mlp.switch_mlp)
    //   Qwen3.5-MoE tiny: model.language_model.layers.N.mlp
    //   Gemma/Llama MoE:  model.layers.N.mlp
    // Uses mlp.gate.weight as probe so mixed-attention layers (linear_attn) work.
    // GGUF MoE (qwen2-style blk.*): probe GGUF directly, return the HF spelling
    // so downstream mp-relative names map back to blk.* via map_name.
    if (gguf_ && gg_loader_.contains("blk." + std::to_string(layer) + ".ffn_gate_inp.weight"))
        return "model.layers." + std::to_string(layer) + ".mlp";
    std::string p = "model.layers." + std::to_string(layer);
    if (contains(p + ".mlp.gate.weight")) return p + ".mlp";
    p = "language_model.model.layers." + std::to_string(layer);
    if (contains(p + ".mlp.gate.weight")) {
        if (contains(p + ".mlp.switch_mlp.gate.weight")) return p + ".mlp.switch_mlp";
        return p + ".mlp";
    }
    p = "model.language_model.layers." + std::to_string(layer);
    if (contains(p + ".mlp.gate.weight")) {
        if (contains(p + ".mlp.switch_mlp.gate.weight")) return p + ".mlp.switch_mlp";
        return p + ".mlp";
    }
    return "model.layers." + std::to_string(layer) + ".mlp";
}
bool VulkanModel::load_float(const std::string& n, std::vector<float>& out) const {
    if (st_) return st_loader_.load_float(n, out);
    return gg_loader_.load_float(map_name(n), out);
}
bool VulkanModel::load_u16(const std::string& n, std::vector<uint16_t>& out, bool& bf16) const {
    if (st_) return st_loader_.load_u16(n, out, bf16);
    if (gg_loader_.load_u16(map_name(n), out, bf16)) return true;
    // Host upcast for quants with no GPU dequant kernel (Q5_0/IQ4_NL/Q3_K plus
    // Q4_0/Q4_1/IQ4_XS/Q2_K/Q4_K/Q6_K, e.g. *_Q2_K files and GGUF MoE experts).
    if (gg_loader_.load_upcast_f16(map_name(n), out)) { bf16 = false; return true; }
    return false;
}
bool VulkanModel::load_q8(const std::string& n, std::vector<uint8_t>& out) const { return gguf_ && gg_loader_.load_q8_0(map_name(n), out); }
bool VulkanModel::load_qk(const std::string& n, int kind, size_t n_elements,
                           std::vector<uint8_t>& out) const {
  return gguf_ && gg_loader_.load_qk(map_name(n), kind, n_elements, out);
}
bool VulkanModel::load_optional(const std::string& n, std::vector<float>& out) const { return load_float(n, out); }

void VulkanModel::upload_q8_block(const std::vector<uint8_t>& fused, VMatrix& m) {
    size_t nblk = fused.size() / 34;
    std::vector<uint8_t> w8(nblk * 32); std::vector<uint16_t> ws(nblk);
    for (size_t b = 0; b < nblk; ++b) { std::memcpy(&ws[b], &fused[b * 34], 2); std::memcpy(&w8[b * 32], &fused[b * 34 + 2], 32); }
    m.q8 = dev_->make(w8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m.qsc = dev_->make(ws, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m.is_q8 = true; m.bf16 = false;
}
void VulkanModel::upload_qk_block(const std::vector<uint8_t>& raw, int kind, VMatrix& m) {
    m.qk = dev_->make(raw, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m.qk_segs = {{kind, 0, 0}};
    m.q8.free(); m.qsc.free(); m.f16.free(); m.is_q8 = false;
}
void VulkanModel::upload_u16_block(const std::vector<uint16_t>& src, VMatrix& m) {
    m.f16 = dev_->make(src, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m.q8.free(); m.qsc.free(); m.qk.free(); m.qk_segs.clear(); m.is_q8 = false;
}
// Q8_0 (34B blocks: fp16 scale + 32 int8) -> F16 host buffer, for mixed-precision fusion.
static bool q8_to_u16(const std::vector<uint8_t>& q8, std::vector<uint16_t>& u) {
    if (q8.empty() || q8.size() % 34) return false;
    u.resize(q8.size() / 34 * 32);
    for (size_t b = 0, nb = q8.size() / 34; b < nb; ++b) {
        uint16_t sc; std::memcpy(&sc, &q8[b * 34], 2);
        float s = fp16_to_float(sc);
        for (int i = 0; i < 32; ++i)
            u[b * 32 + i] = float_to_fp16(s * float(static_cast<int8_t>(q8[b * 34 + 2 + i])));
    }
    return true;
}
void VulkanModel::load_matrix(const std::string& name, VMatrix& m, bool required, size_t n_elements) {
    std::vector<uint8_t> q8;
    if (load_q8(name, q8)) { upload_q8_block(q8, m); m.f16.free(); m.qk.free(); m.qk_segs.clear(); return; }
    for (int kind : {12, 13, 14}) {
        if (kind == 14 && std::getenv("NANO_F16Q6")) continue;  // DBG tag: force Q6_K via F16 upcast
        std::vector<uint8_t> qk;
         if (load_qk(name, kind, n_elements, qk)) { upload_qk_block(qk, kind, m); return; }
    }
    std::vector<uint16_t> u; bool bf16 = false;
    if (load_u16(name, u, bf16)) { upload_u16_block(u, m); m.bf16 = bf16; return; }
    if (required) throw std::runtime_error("missing required weight: " + name);
}
void VulkanModel::load_matrix_host(const std::string& n, std::vector<uint16_t>& u16, bool& bf16,
                                   std::vector<uint8_t>& q8, bool& is_q8, std::vector<uint8_t>& qk,
                                   int& qk_kind, size_t n_elements, bool required) {
    u16.clear(); q8.clear(); qk.clear(); bf16 = false; is_q8 = false; qk_kind = 0;
    if (load_q8(n, q8)) { is_q8 = true; return; }
    for (int kind : {12, 13, 14}) {
         if (load_qk(n, kind, n_elements, qk)) { qk_kind = kind; return; }
    }
    if (load_u16(n, u16, bf16)) { is_q8 = false; return; }
    if (required) throw std::runtime_error("missing required weight: " + n);
}

void VulkanModel::load_weights() {
    load_model_dir(config_.model);
    g_gguf_moe.clear();
    {
       auto names = st_ ? st_loader_.names() : gg_loader_.names();
       bool has_moe = false;
       for (auto& n : names)
         if (n.find("exps") != std::string::npos || n.find(".experts.") != std::string::npos ||
             n.find("gate_inp") != std::string::npos || n.find("shared_expert") != std::string::npos ||
             (n.find("moe") != std::string::npos && n.find("_norm") == std::string::npos)) {
           has_moe = true; break;
         }
        auto& hf = config_.hf;
        if (has_moe && gguf_)
            apply_gguf_moe_meta(gg_loader_, config_.hf);  // MoE counts live in GGUF KV
        if (has_moe && hf.num_experts <= 0)
           throw std::runtime_error("MoE weights present but config has num_experts=0");
        for (auto& n : names)
          if (n.find("ssm_") != std::string::npos || n.find(".ssm.") != std::string::npos ||
              n.find("mamba") != std::string::npos)
            throw std::runtime_error("SSM/mamba hybrid layers are unsupported (file: " + n + ")");
    }
    if (!contains("model.embed_tokens.weight") && !contains("model.language_model.embed_tokens.weight"))
        throw std::runtime_error("model.embed_tokens.weight not found");
    model_prefix_ = contains("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";
    auto& hf = config_.hf;
    int hidden = hf.hidden_size, inter = hf.intermediate_size;
    load_matrix(model_prefix_ + "embed_tokens.weight", embed_, true, size_t(hf.vocab_size) * hidden);
    for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
        auto& lw = layers_[layer];
        std::string p = model_prefix_ + "layers." + std::to_string(layer);
        bool has_attn = contains(p + ".self_attn.qkv_proj.weight") ||
                        contains(p + ".self_attn.q_proj.weight");
        if (has_attn) {
        if (contains(p + ".self_attn.qkv_proj.weight")) {
            load_matrix(p + ".self_attn.qkv_proj.weight", lw.qkv, true, size_t(qkv_size_) * hidden);
        } else {
            std::vector<uint16_t> q_u,k_u,v_u; bool qbf=false,kbf=false,vbf=false;
            std::vector<uint8_t> q_q8,k_q8,v_q8; bool q8q=false,k8q=false,v8q=false;
            std::vector<uint8_t> q_qk,k_qk,v_qk; int qkind=0,kkind=0,vkind=0;
            load_matrix_host(p + ".self_attn.q_proj.weight", q_u,qbf,q_q8,q8q,q_qk,qkind,size_t(q_size_)*hidden,true);
            load_matrix_host(p + ".self_attn.k_proj.weight", k_u,kbf,k_q8,k8q,k_qk,kkind,size_t(kv_size_)*hidden,true);
            load_matrix_host(p + ".self_attn.v_proj.weight", v_u,vbf,v_q8,v8q,v_qk,vkind,size_t(kv_size_)*hidden,true);
            if (q8q&&k8q&&v8q) {
                std::vector<uint8_t> f; f.insert(f.end(),q_q8.begin(),q_q8.end());
                f.insert(f.end(),k_q8.begin(),k_q8.end()); f.insert(f.end(),v_q8.begin(),v_q8.end());
                upload_q8_block(f, lw.qkv);
            } else if (qkind&&kkind&&vkind) {
                size_t qe = size_t(q_size_)*hidden, ke = size_t(kv_size_)*hidden;
                if (qe%256 || ke%256)
                    throw std::runtime_error("K-quant q/k/v split off super-block boundary");
                size_t qb = qe/256*GGUFLoader::qk_block_bytes(qkind);
                size_t kb = ke/256*GGUFLoader::qk_block_bytes(kkind);
                size_t vb = ke/256*GGUFLoader::qk_block_bytes(vkind);
                 std::vector<uint8_t> f; f.insert(f.end(),q_qk.begin(),q_qk.end());
                 f.insert(f.end(),k_qk.begin(),k_qk.end()); f.insert(f.end(),v_qk.begin(),v_qk.end());
                  upload_qk_block(f, qkind, lw.qkv);
                  lw.qkv.qk_segs = {{qkind,0,0},{kkind,qb,qe},{vkind,qb+kb,qe+ke}};
            } else {
                // Mixed precisions (Q8 vs K-quant vs upcast/F16): fuse all as F16 host-side.
                std::vector<uint16_t> f;
                auto fuse_u16 = [&](const std::string& n, std::vector<uint16_t>& u,
                                    const std::vector<uint8_t>& q8b) {
                    if (u.empty()) { bool b = false; load_u16(n, u, b); }
                    if (u.empty() && !q8b.empty()) q8_to_u16(q8b, u);
                    if (u.empty())
                        throw std::runtime_error("mixed-precision q/k/v needs F16 for " + n);
                    f.insert(f.end(), u.begin(), u.end());
                };
                fuse_u16(p + ".self_attn.q_proj.weight", q_u, q_q8);
                fuse_u16(p + ".self_attn.k_proj.weight", k_u, k_q8);
                fuse_u16(p + ".self_attn.v_proj.weight", v_u, v_q8);
                lw.qkv.bf16 = qbf && kbf && vbf && q_q8.empty() && k_q8.empty() && v_q8.empty()
                              && !qkind && !kkind && !vkind;
                    if (std::getenv("NANO_PIPE") && layer == 0) {
                      auto hf16=[](uint16_t h){uint32_t sign=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff;float val;
                        if(e==0)val=(float)m/16777216.0f*(sign?-1.f:1.f);
                        else if(e==31)val=m?NAN:(sign?-INFINITY:INFINITY);
                        else{uint32_t z=((sign?1u:0u)<<31)|((e+127-15)<<23)|(m<<13);std::memcpy(&val,&z,4);}return val;};
                      size_t vs=(size_t)kv_size_*hidden; double vsum=0; for(uint16_t h:v_u) vsum+=hf16(h);
                      auto bs=[&](int bi){double s=0;for(int i=0;i<256;i++)s+=hf16(v_u[(size_t)bi*256+i]);return s;};
                      double qbs=0; for(int i=0;i<256;i++)qbs+=hf16(q_u[i]);
                      std::fprintf(stderr,"[PIPE load0 vsum=%.4f b0=%.4f b1=%.4f b2=%.4f b1023=%.4f b2047=%.4f\n",
                        vsum, bs(0), bs(1), bs(2), bs(1023), bs(2047));
                      std::fprintf(stderr,"[PIPE loadq q_u n=%zu qfirst=%.6f,%.6f,%.6f,%.6f,%.6f,%.6f qblk0=%.4f\n",
                        q_u.size(),(double)hf16(q_u[0]),(double)hf16(q_u[1]),(double)hf16(q_u[2]),
                        (double)hf16(q_u[3]),(double)hf16(q_u[4]),(double)hf16(q_u[5]),qbs);
                      { std::vector<uint8_t> rb; if(load_qk(p+".self_attn.v_proj.weight",12,vs,rb)) {
                          std::fprintf(stderr,"[PIPE raw2 "); for(int i=0;i<16;i++)std::fprintf(stderr,"%02x ",rb[2*144+i]);
                          std::fprintf(stderr,"] d=%.6f dmin=%.6f\n",(double)hf16((uint16_t)(rb[2*144]|(rb[2*144+1]<<8))),(double)hf16((uint16_t)(rb[2*144+2]|(rb[2*144+3]<<8)))); } }
                    }
                    upload_u16_block(f, lw.qkv);
            }
        }
        std::vector<float> qb,kb,vb;
        if (load_optional(p + ".self_attn.q_proj.bias", qb) &&
             load_optional(p + ".self_attn.k_proj.bias", kb) &&
             load_optional(p + ".self_attn.v_proj.bias", vb)) {
             std::vector<float> b; b.insert(b.end(),qb.begin(),qb.end());
             b.insert(b.end(),kb.begin(),kb.end()); b.insert(b.end(),vb.begin(),vb.end());
             lw.qkv_bias = dev_->make(b);
         }
        load_matrix(p + ".self_attn.o_proj.weight", lw.o, true, size_t(hidden) * q_size_);
        }  // end has_attn
        if (hf.num_experts > 0 && hf.layer_is_moe(layer)) {
            auto& ms = lw.moe;
            const int H = hidden, E = hf.num_experts, K = hf.num_experts_per_tok, I = hf.moe_intermediate_size;
            ms.E = E; ms.K = K; ms.I = I;
            ms.router_kind = hf.is_sigmoid_group() ? 1 : 0;
            ms.n_group = hf.n_group; ms.topk_group = hf.topk_group;
            ms.routed_scaling = hf.routed_scaling; ms.norm_topk = hf.norm_topk_prob;
            std::vector<uint16_t> rt;
            bool b = false;
            std::string mp = moe_key_prefix(layer) + ".";
            if (!load_u16(mp + "gate.weight", rt, b) || rt.size() != (size_t)E * H)
                throw std::runtime_error("missing/short router weights for layer " + std::to_string(layer));
            ms.router.f16 = dev_->make(rt); ms.router.bf16 = b; ms.bf16 = b;
            const int SI = hf.shared_expert_intermediate_size;
            if (SI > 0) {
                std::vector<uint16_t> g, u, d; bool b1=false, b2=false, b3=false;
                if (!load_u16(mp + "shared_expert.gate_proj.weight", g, b1) ||
                    !load_u16(mp + "shared_expert.up_proj.weight", u, b2) ||
                    !load_u16(mp + "shared_expert.down_proj.weight", d, b3))
                    throw std::runtime_error("missing shared expert weights for layer " + std::to_string(layer));
                std::vector<uint16_t> gu; gu.reserve(g.size()+u.size());
                gu.insert(gu.end(), g.begin(), g.end()); gu.insert(gu.end(), u.begin(), u.end());
                ms.sh_gu.f16 = dev_->make(gu); ms.sh_gu.bf16 = b1&&b2;
                ms.sh_dn.f16 = dev_->make(d); ms.sh_dn.bf16 = b3;
                std::vector<uint16_t> gt; bool bg=false;
                if (load_u16(mp + "shared_expert_gate.weight", gt, bg) && (int)gt.size() == H) {
                    ms.sh_gate.f16 = dev_->make(gt); ms.sh_gate.bf16 = bg;
                }
            }
            ms.stacked = contains(mp + "experts.gate_up_proj") && contains(mp + "experts.down_proj");
            if (gguf_ && !ms.stacked) {
                // GGUF MoE: stacked Q4_K/Q6_K experts stay quantized on host;
                // dequantized per expert into ms.stage at place time.
                static const char* kProjs[3] = {
                    "experts.gate_exps.weight", "experts.up_exps.weight", "experts.down_exps.weight"};
                std::vector<std::vector<uint8_t>> raws(3);
                GgufMoeQ qb;
                for (int pi = 0; pi < 3; ++pi) {
                    std::string hn = mp + kProjs[pi];
                    const GGUFTensorMeta* tm = gg_loader_.tensor(map_name(hn));
                    if (!tm) throw std::runtime_error("missing GGUF MoE experts tensor: " + hn);
                    int kind = (tm->dtype == "Q4_K") ? 12 : (tm->dtype == "Q6_K") ? 14 : 0;
                    if (!kind) throw std::runtime_error("unsupported GGUF MoE expert dtype " + tm->dtype);
                    if (qb.kind && qb.kind != kind)
                        throw std::runtime_error("mixed expert quant kinds in layer " + std::to_string(layer));
                    qb.kind = kind;
                    qb.dtype = tm->dtype;
                    size_t need = (size_t)E * I * H;
                    if (need % GGUFLoader::QK_K != 0)
                        throw std::runtime_error("GGUF MoE expert size off super-block boundary");
                    if (!load_qk(hn, kind, need, raws[pi]))
                        throw std::runtime_error("short GGUF MoE experts tensor: " + hn);
                }
                qb.gate = std::move(raws[0]); qb.up = std::move(raws[1]); qb.down = std::move(raws[2]);
                g_gguf_moe[layer] = std::move(qb);
                ms.f_g.assign(E, nullptr); ms.f_u.assign(E, nullptr); ms.f_d.assign(E, nullptr);
            } else if (ms.stacked) {
                size_t ne = 0; bool nf = false;
                const uint8_t* raw_gu = st_loader_.mapped(mp + "experts.gate_up_proj", ne, nf);
                if (!raw_gu || ne != (size_t)E * 2 * I * H * sizeof(uint16_t)) throw std::runtime_error("bad stacked gate_up_proj for layer " + std::to_string(layer));
                const uint8_t* raw_dn = st_loader_.mapped(mp + "experts.down_proj", ne, nf);
                if (!raw_dn || ne != (size_t)E * H * I * sizeof(uint16_t)) throw std::runtime_error("bad stacked down_proj for layer " + std::to_string(layer));
                ms.s_gu = reinterpret_cast<const uint16_t*>(raw_gu);
                ms.s_dn = reinterpret_cast<const uint16_t*>(raw_dn);
            } else {
                ms.f_g.resize(E); ms.f_u.resize(E); ms.f_d.resize(E);
                for (int e = 0; e < E; ++e) {
                    std::string pe = mp + "experts." + std::to_string(e) + ".";
                    size_t ne = 0; bool nf = false;
                    ms.f_g[e] = reinterpret_cast<const uint16_t*>(st_loader_.mapped(pe + "gate_proj.weight", ne, nf));
                    if (!ms.f_g[e] || ne != (size_t)I * H * sizeof(uint16_t)) throw std::runtime_error("missing expert gate " + pe);
                    ms.f_u[e] = reinterpret_cast<const uint16_t*>(st_loader_.mapped(pe + "up_proj.weight", ne, nf));
                    if (!ms.f_u[e] || ne != (size_t)I * H * sizeof(uint16_t)) throw std::runtime_error("missing expert up " + pe);
                    ms.f_d[e] = reinterpret_cast<const uint16_t*>(st_loader_.mapped(pe + "down_proj.weight", ne, nf));
                    if (!ms.f_d[e] || ne != (size_t)H * I * sizeof(uint16_t)) throw std::runtime_error("missing expert down " + pe);
                }
            }
            if (ms.router_kind == 1) {
                std::vector<float> eb;
                if (load_optional(mp + "gate.expert_bias", eb) && (int)eb.size() == E) {
                    ms.ex_bias = dev_->make(eb);
                } else {
                    ms.ex_bias = dev_->make(std::vector<float>(E, 0.0f));
                }
            }
        } else if (contains(p + ".mlp.gate_up_proj.weight")) {
            load_matrix(p + ".mlp.gate_up_proj.weight", lw.gate_up, true, size_t(2 * inter) * hidden);
        } else {
            std::vector<uint16_t> g_u,u_u; bool gbf=false,ubf=false;
            std::vector<uint8_t> g_q8,u_q8; bool gq=false,uq=false;
            std::vector<uint8_t> g_qk,u_qk; int gkind=0,ukind=0;
            load_matrix_host(p + ".mlp.gate_proj.weight", g_u,gbf,g_q8,gq,g_qk,gkind,size_t(inter)*hidden,true);
            load_matrix_host(p + ".mlp.up_proj.weight", u_u,ubf,u_q8,uq,u_qk,ukind,size_t(inter)*hidden,true);
            if (gq&&uq) {
                std::vector<uint8_t> f; f.insert(f.end(),g_q8.begin(),g_q8.end()); f.insert(f.end(),u_q8.begin(),u_q8.end());
                upload_q8_block(f, lw.gate_up);
            } else if (gkind&&ukind&&gkind==ukind) {
                if ((size_t(inter)*hidden)%256)
                    throw std::runtime_error("K-quant gate/up split off super-block boundary");
                size_t gb = size_t(inter)*hidden/256*GGUFLoader::qk_block_bytes(gkind);
                std::vector<uint8_t> f; f.insert(f.end(),g_qk.begin(),g_qk.end()); f.insert(f.end(),u_qk.begin(),u_qk.end());
                upload_qk_block(f, gkind, lw.gate_up);
                lw.gate_up.qk_segs = {{gkind,0,0},{ukind,gb,size_t(inter)*hidden}};
            } else {
                // Mixed precisions: fuse all as F16 host-side.
                std::vector<uint16_t> f;
                auto fuse_u16 = [&](const std::string& n, std::vector<uint16_t>& u,
                                    const std::vector<uint8_t>& q8b) {
                    if (u.empty()) { bool b = false; load_u16(n, u, b); }
                    if (u.empty() && !q8b.empty()) q8_to_u16(q8b, u);
                    if (u.empty())
                        throw std::runtime_error("mixed-precision gate/up needs F16 for " + n);
                    f.insert(f.end(), u.begin(), u.end());
                };
                fuse_u16(p + ".mlp.gate_proj.weight", g_u, g_q8);
                fuse_u16(p + ".mlp.up_proj.weight", u_u, u_q8);
                upload_u16_block(f, lw.gate_up);
                lw.gate_up.bf16 = gbf && ubf && g_q8.empty() && u_q8.empty() && !gkind && !ukind;
            }
        }
        if (!(hf.num_experts > 0 && hf.layer_is_moe(layer)))
            load_matrix(p + ".mlp.down_proj.weight", lw.down, true, size_t(hidden) * inter);
    std::vector<float> ln(hidden, 1.0f);
    (void)load_optional(p + ".input_layernorm.weight", ln); lw.input_ln = dev_->make(ln);
    ln.assign(hidden, 1.0f);
    (void)load_optional(p + ".post_attention_layernorm.weight", ln); lw.post_ln = dev_->make(ln);
    if (!hf.attention_bias) {
      std::vector<float> qn; (void)load_optional(p + ".self_attn.q_norm.weight", qn); if (!qn.empty()) lw.q_norm = dev_->make(qn);
      std::vector<float> kn; (void)load_optional(p + ".self_attn.k_norm.weight", kn); if (!kn.empty()) lw.k_norm = dev_->make(kn);
    }
    }
    std::vector<float> fn(hf.hidden_size, 1.0f);
    load_optional(model_prefix_ + "norm.weight", fn);
    final_norm_ = dev_->make(fn);
    tie_lm_head_ = hf.tie_word_embeddings;
    if (!tie_lm_head_ && contains("lm_head.weight"))
        load_matrix("lm_head.weight", lm_head_, false, size_t(hf.vocab_size) * hidden);
    ready_ = true;
    dev_->submit_wait();
}

void VulkanModel::alloc_expert_slots() {
    auto& hf = config_.hf;
    if (hf.num_experts <= 0) return;
    int nl = 0;
    for (int l = 0; l < hf.num_hidden_layers; ++l)
        if (hf.layer_is_moe(l)) ++nl;
    if (!nl) return;
    const size_t per_expert = (size_t)2 * hf.moe_intermediate_size * hf.hidden_size * sizeof(uint16_t);
    const size_t total = (size_t)nl * hf.num_experts * per_expert;
    size_t budget = config_.expert_budget_gb > 0 ? (size_t)(config_.expert_budget_gb * 1e9) : (size_t)(2.0 * 1e9);
    if (budget > total) budget = total;
    int slots = (int)(budget / ((size_t)nl * per_expert));
    if (slots < hf.num_experts_per_tok + 2) slots = hf.num_experts_per_tok + 2;
    if (slots > hf.num_experts) slots = hf.num_experts;
    // Global cache: allow up to 2x per-layer slots across all layers (CPU page-cache budget).
    expert_cache_ = e0::SharedExpertCache(std::max(slots * nl, slots * 2));
    prefetch_buf_ = e0::PrefetchBuffer(std::max(48, slots * nl / 4));
    for (auto& lw : layers_) {
        auto& ms = lw.moe;
        if (!ms.E) continue;
        ms.slots = slots;
        ms.slot_gu.alloc((VkDeviceSize)slots * 2 * ms.I * hf.hidden_size * sizeof(uint16_t));
        ms.slot_dn.alloc((VkDeviceSize)slots * ms.I * hf.hidden_size * sizeof(uint16_t));
        ms.h_slot_of.assign(ms.E, -1);
        ms.h_slot_exp.assign(slots, -1);
        ms.h_lru.assign(slots, 0);
        ms.slot_of = dev_->make(ms.h_slot_of);
        ms.stage.resize((size_t)3 * ms.I * hf.hidden_size);
    }
    std::fprintf(stderr, "MoE: %d sparse layers, %d experts (top-%d), %d VRAM slots/layer, %.2f GB budget\n",
                 nl, hf.num_experts, hf.num_experts_per_tok, slots, budget / 1e9);
}

void VulkanModel::prefetch_experts() {
    // Overlap with the NEXT forward call: wait for the previous prefetch (if any),
    // then kick a new one on a worker. The future's destructor joins at model teardown.
    if (prefetch_fut_.valid()) prefetch_fut_.wait();
    if (config_.hf.num_experts <= 0) return;
    prefetch_fut_ = std::async(std::launch::async, [this] { do_prefetch_experts(); });
}

void VulkanModel::do_prefetch_experts() {
    auto& hf = config_.hf;
    if (hf.num_experts <= 0) return;
    std::lock_guard<std::mutex> lk(prefetch_mu_);
    const size_t gu_bytes = (size_t)2 * hf.moe_intermediate_size * hf.hidden_size * sizeof(uint16_t);
    const size_t dn_bytes = (size_t)hf.moe_intermediate_size * hf.hidden_size * sizeof(uint16_t);
    for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
        if (!hf.layer_is_moe(layer)) continue;
        auto& ms = layers_[layer].moe;
        if (!ms.E) continue;
        for (int e = 0; e < ms.E; ++e) {
            if (ms.h_slot_of[e] < 0) continue;
            if (ms.stacked) {
                warm_pages((const uint8_t*)(ms.s_gu + (size_t)e * 2 * ms.I * hf.hidden_size), gu_bytes);
                warm_pages((const uint8_t*)(ms.s_dn + (size_t)e * ms.I * hf.hidden_size), dn_bytes);
            } else {
                if (ms.f_g[e]) warm_pages((const uint8_t*)ms.f_g[e], dn_bytes);
                if (ms.f_u[e]) warm_pages((const uint8_t*)ms.f_u[e], dn_bytes);
                if (ms.f_d[e]) warm_pages((const uint8_t*)ms.f_d[e], dn_bytes);
            }
        }
    }
}

int VulkanModel::estimate_kv_cache_blocks() const {
    int max_needed = (config_.max_model_len + block_size_ - 1) / block_size_ + 64;
    return config_.num_kvcache_blocks > 0 ? config_.num_kvcache_blocks : std::min(64, max_needed);
}
int VulkanModel::allocate_kv_cache() {
    int num_blocks = config_.num_kvcache_blocks > 0 ? config_.num_kvcache_blocks : estimate_kv_cache_blocks();
    if (num_blocks <= 0) throw std::runtime_error("cannot allocate KV cache: no free blocks");
    auto& hf = config_.hf;
    kv_layer_stride_ = (size_t)num_blocks * block_size_ * kv_head_stride_;
    num_blocks_ = num_blocks;
    config_.num_kvcache_blocks = num_blocks;
    // Per-layer k/v slices (each well under the 2GB VkBuffer maxBufferSize limit
    // on AMD drivers) instead of one giant monolithic kv_cache_ buffer.
    k_cache_.resize(hf.num_hidden_layers);
    v_cache_.resize(hf.num_hidden_layers);
    VkDeviceSize per_layer = (VkDeviceSize)kv_layer_stride_ * sizeof(float);
    for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
        k_cache_[layer].alloc(per_layer);
        v_cache_[layer].alloc(per_layer);
    }
    dev_->submit_wait();
    return num_blocks;
}

std::vector<float> VulkanModel::forward_logits(const VKContext& ctx) {    if (!ready_) throw std::runtime_error("model weights not loaded");
    if (k_cache_.empty()) throw std::runtime_error("kv cache not allocated");
    auto& hf = config_.hf;
    int rows = (int)ctx.input_ids.size();
    int hidden = hf.hidden_size, inter = hf.intermediate_size;
    int heads = hf.num_attention_heads, kv_heads = hf.num_key_value_heads;
    int head_dim = hf.head_dim;
    float eps = (float)hf.rms_norm_eps;
    float scale = 1.0f / std::sqrt((float)head_dim);
    if (head_dim > 256) throw std::runtime_error("head_dim > 256 unsupported by attention kernel");

    VBuf d_ids = dev_->make(ctx.input_ids);
    VBuf d_pos = dev_->make(ctx.positions);
    VBuf d_slot; if (!ctx.slot_mapping.empty()) d_slot = dev_->make(ctx.slot_mapping);
    VBuf d_qseq; if (!ctx.query_seq.empty()) d_qseq = dev_->make(ctx.query_seq);
    VBuf d_qlen; if (!ctx.query_key_len.empty()) d_qlen = dev_->make(ctx.query_key_len);
    VBuf d_tbl;  if (!ctx.block_tables.empty()) d_tbl = dev_->make(ctx.block_tables);
    VBuf d_last; if (!ctx.last_indices.empty()) d_last = dev_->make(ctx.last_indices);

    VBuf hidden_dev, residual_dev, norm_dev, q_dev, k_dev, v_dev, attn_dev, gate_dev, mlp_dev;
    const VkDeviceSize F4 = sizeof(float);
    hidden_dev.alloc((VkDeviceSize)rows * hidden * F4);
    residual_dev.alloc((VkDeviceSize)rows * hidden * F4);
    norm_dev.alloc((VkDeviceSize)rows * hidden * F4);
    q_dev.alloc((VkDeviceSize)rows * q_size_ * F4);
    k_dev.alloc((VkDeviceSize)rows * kv_size_ * F4);
    v_dev.alloc((VkDeviceSize)rows * kv_size_ * F4);
    attn_dev.alloc((VkDeviceSize)rows * q_size_ * F4);
    gate_dev.alloc((VkDeviceSize)rows * 2 * std::max(inter, hf.shared_expert_intermediate_size) * F4);
    mlp_dev.alloc((VkDeviceSize)rows * std::max(inter, hf.shared_expert_intermediate_size) * F4);
    { std::vector<float> zeros((size_t)rows * hidden, 0.0f);  // HIP zeroes scratch at alloc; residual is read-before-write
      residual_dev.upload(zeros.data(), (VkDeviceSize)zeros.size() * F4); }

    const bool pipe_dbg = std::getenv("NANO_PIPE") != nullptr;
    auto pipe_tag = [&](const char* stage, const VBuf& b, size_t floats) {
        if (!pipe_dbg) return;
        dev_->submit_wait();
        std::vector<float> v(std::min<size_t>(floats, (size_t)rows * hidden));
        b.download(v.data(), v.size() * F4);
        double sum = 0.0; for (float x : v) sum += x;
        std::fprintf(stderr, "[PIPE vk %s] n=%zu sum=%.6f first=%.6f,%.6f,%.6f\n",
                     stage, v.size(), sum, v[0], v[1], v[2]);
    };
    dev_->embedding(d_ids, embed_, hidden_dev, rows, hidden);
    pipe_tag("embed", hidden_dev, (size_t)rows * hidden);
    for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
        auto& lw = layers_[layer];
        dev_->rms_norm_add(hidden_dev, residual_dev, lw.input_ln, norm_dev, rows, hidden, eps);
        if (lw.qkv.f16.buffer || lw.qkv.q8.buffer) {
        dev_->matmul(norm_dev, lw.qkv, rows, q_size_, hidden, q_dev, 0u);
        dev_->matmul(norm_dev, lw.qkv, rows, kv_size_, hidden, k_dev, (uint32_t)(q_size_ * hidden));
        dev_->matmul(norm_dev, lw.qkv, rows, kv_size_, hidden, v_dev, (uint32_t)((q_size_ + kv_size_) * hidden));
        if (lw.qkv_bias.nbytes) {
            dev_->add_bias_inplace(q_dev, lw.qkv_bias, rows, q_size_, 0u);
            dev_->add_bias_inplace(k_dev, lw.qkv_bias, rows, kv_size_, (uint32_t)q_size_);
            dev_->add_bias_inplace(v_dev, lw.qkv_bias, rows, kv_size_, (uint32_t)(q_size_ + kv_size_));
        }
        if (lw.q_norm.nbytes) dev_->rms_norm(q_dev, lw.q_norm, q_dev, rows * heads, head_dim, eps);
        if (lw.k_norm.nbytes) dev_->rms_norm(k_dev, lw.k_norm, k_dev, rows * kv_heads, head_dim, eps);
        if (layer == 0) { pipe_tag("L0.q", q_dev, (size_t)rows * q_size_); pipe_tag("L0.k", k_dev, (size_t)rows * kv_size_); pipe_tag("L0.v", v_dev, (size_t)rows * kv_size_); }
        dev_->rope(q_dev, d_pos, inv_freq_, rows, heads, head_dim, q_size_);
        dev_->rope(k_dev, d_pos, inv_freq_, rows, kv_heads, head_dim, kv_size_);
        if (layer == 0) { pipe_tag("L0.qrope", q_dev, (size_t)rows * q_size_); pipe_tag("L0.krope", k_dev, (size_t)rows * kv_size_); }
        int d = kv_heads * head_dim;
        if (d_slot.buffer && !ctx.slot_mapping.empty())
            dev_->store_kv(k_dev, v_dev, k_cache_[layer], v_cache_[layer], d_slot, kv_heads, head_dim, rows * d, 0u, 0u);
        if (d_qseq.buffer && !ctx.query_seq.empty()) {
            // ATTN-1: single sw_start per dispatch from the longest sequence
            // (exact for the usual single-request case); per-layer pattern flag.
            int sw_start = 0;
            bool use_sw = hf.sliding_window > 0 &&
                (hf.sliding_window_pattern.empty() ||
                 (layer < (int)hf.sliding_window_pattern.size() && hf.sliding_window_pattern[layer]));
            if (use_sw && !ctx.query_key_len.empty()) {
                int maxlen = 0;
                for (auto v : ctx.query_key_len) maxlen = std::max(maxlen, (int)v);
                if (maxlen > hf.sliding_window) sw_start = maxlen - hf.sliding_window;
            }
            dev_->paged_attention_ex(q_dev, attn_dev, k_cache_[layer], v_cache_[layer], d_qseq, d_qlen, d_tbl,
                                     rows, heads, kv_heads, head_dim, block_size_, ctx.max_blocks, scale,
                                     sw_start, (float)hf.attn_logit_softcapping, 0u, 0u);
        }
        if (layer == 0) pipe_tag("L0.attn", attn_dev, (size_t)rows * q_size_);
        dev_->matmul(attn_dev, lw.o, rows, hidden, q_size_, hidden_dev);
        pipe_tag(("L" + std::to_string(layer) + ".attnout").c_str(), hidden_dev, (size_t)rows * hidden);
        }  // end has_attn
        dev_->rms_norm_add(hidden_dev, residual_dev, lw.post_ln, norm_dev, rows, hidden, eps);
        if (lw.moe.E > 0) {
            auto& ms = lw.moe;
            const int H = hidden, E = ms.E, K = ms.K, I = ms.I;
            VBuf route_dev, idx_dev, score_dev, shs_dev, exp_dev;
            route_dev.alloc((VkDeviceSize)rows * E * F4);
            idx_dev.alloc((VkDeviceSize)rows * K * sizeof(int32_t));
            score_dev.alloc((VkDeviceSize)rows * K * F4);
            dev_->matmul(norm_dev, ms.router, rows, E, H, route_dev);
            // Fused route+FFN needs SOFTMAX_TOPK routing with bounded K/E; the
            // fused kernel sizes shared idx/score by SPEC_K (sigmoid path and
            // oversize shapes keep the classic two-dispatch path).
            const bool use_fused =
                ms.router_kind == 0 && K <= 64 && E <= 256 && !std::getenv("MOE_NO_FUSED");
            std::vector<int32_t> h_idx;
            if (!use_fused) {
                if (ms.router_kind == 0) {
                    dev_->moe_route(route_dev, idx_dev, score_dev, rows, E, K, ms.norm_topk);
                } else {
                    int bias_off = ms.ex_bias.nbytes ? 0 : -1;
                    dev_->moe_route_sigmoid(route_dev, idx_dev, score_dev, ms.ex_bias,
                                            rows, E, K, ms.n_group, ms.topk_group,
                                            ms.norm_topk, ms.routed_scaling, bias_off);
                }
                // Host needs routing indices before placing experts: flush the queue first
                // (original code read idx_dev back while the dispatch was still queued).
                dev_->submit_wait();
                h_idx.assign((size_t)rows * K, 0);
                idx_dev.download(h_idx.data(), (VkDeviceSize)h_idx.size() * sizeof(int32_t));
            }
             const int SI = hf.shared_expert_intermediate_size;
             if (SI > 0 && !std::getenv("MOE_NO_SHARED")) {
                dev_->matmul(norm_dev, ms.sh_gu, rows, 2 * SI, H, gate_dev);
                dev_->silu_and_mul(gate_dev, mlp_dev, rows, SI);
                dev_->matmul(mlp_dev, ms.sh_dn, rows, H, SI, hidden_dev);
                if (ms.sh_gate.f16.nbytes) {
                    shs_dev.alloc((VkDeviceSize)rows * F4);
                    dev_->matmul(norm_dev, ms.sh_gate, rows, 1, H, shs_dev);
                    dev_->scale_sigmoid(hidden_dev, shs_dev, rows, H);
                }
            } else {
                std::vector<float> zeros((size_t)rows * H, 0.0f);
                hidden_dev.upload(zeros.data(), (VkDeviceSize)zeros.size() * F4);
            }
            const int chunk = std::max(1, ms.slots / K);
            if (use_fused) {
                exp_dev.alloc((VkDeviceSize)rows * H * F4);
                bool have_idx = false;
                for (int r0 = 0; r0 < rows; r0 += chunk) {
                    int rc = std::min(chunk, rows - r0);
                    // Optimistic: route+FFN against currently resident slots.
                    dev_->moe_route_ffn(norm_dev, route_dev, idx_dev, score_dev,
                                        ms.slot_gu, ms.slot_dn, ms.slot_of, exp_dev,
                                        r0, rc, H, I, K, E, ms.bf16, ms.norm_topk);
                    dev_->submit_wait();
                    if (!have_idx) {
                        // Routing is deterministic in logits: one readback serves
                        // all chunks and any fixup re-runs.
                        h_idx.assign((size_t)rows * K, 0);
                        idx_dev.download(h_idx.data(), (VkDeviceSize)h_idx.size() * sizeof(int32_t));
                        have_idx = true;
                    }
                    std::vector<int32_t> sub(h_idx.begin() + (size_t)r0 * K,
                                             h_idx.begin() + (size_t)(r0 + rc) * K);
                    bool miss = false;
                    for (int32_t e : sub)
                        if (e < 0 || e >= E || ms.h_slot_of[e] < 0) { miss = true; break; }
                    if (miss) {
                        moe_place(ms, layer, sub, H, I);
                        dev_->moe_route_ffn(norm_dev, route_dev, idx_dev, score_dev,
                                            ms.slot_gu, ms.slot_dn, ms.slot_of, exp_dev,
                                            r0, rc, H, I, K, E, ms.bf16, ms.norm_topk);
                    }
                }
                dev_->add_buf(hidden_dev, exp_dev, rows * H);
            } else {
                for (int r0 = 0; r0 < rows; r0 += chunk) {
                    int rc = std::min(chunk, rows - r0);
                    std::vector<int32_t> sub(h_idx.begin() + (size_t)r0 * K,
                                             h_idx.begin() + (size_t)(r0 + rc) * K);
                    moe_place(ms, layer, sub, H, I);
                    dev_->moe_ffn(norm_dev, idx_dev, score_dev, ms.slot_gu, ms.slot_dn,
                                  ms.slot_of, hidden_dev, rc, H, I, K, ms.bf16);
                }
            }
            dev_->submit_wait();
        } else {
            dev_->matmul(norm_dev, lw.gate_up, rows, 2 * inter, hidden, gate_dev);
            dev_->silu_and_mul(gate_dev, mlp_dev, rows, inter);
            dev_->matmul(mlp_dev, lw.down, rows, hidden, inter, hidden_dev);
        }
        pipe_tag(("L" + std::to_string(layer) + ".mlpout").c_str(), hidden_dev, (size_t)rows * hidden);
    }
    prefetch_experts();
    dev_->rms_norm_add(hidden_dev, residual_dev, final_norm_, norm_dev, rows, hidden, eps);
    int out_rows = (int)ctx.last_indices.size();
    VBuf selected_dev; selected_dev.alloc((VkDeviceSize)out_rows * hidden * F4);
    dev_->gather_rows(norm_dev, d_last, selected_dev, hidden, out_rows * hidden);
    VBuf logits_dev; logits_dev.alloc((VkDeviceSize)out_rows * hf.vocab_size * F4);
    VMatrix& lm = tie_lm_head_ ? embed_ : lm_head_;
    dev_->matmul(selected_dev, lm, out_rows, hf.vocab_size, hidden, logits_dev);
    std::vector<float> out((size_t)out_rows * hf.vocab_size);
    dev_->submit_wait();
    logits_dev.download(out.data(), (VkDeviceSize)out.size() * sizeof(float));
    // ATTN-2: Gemma final-logit softcap (host-side; exact, negligible cost).
    if (hf.final_logit_softcapping > 0) {
        float c = (float)hf.final_logit_softcapping;
        for (float& x : out) x = std::tanh(x / c) * c;
    }
    return out;
}

std::string VulkanModel::memory_report() const {
  std::ostringstream ss;
  const VkRT* rt = dev_ ? dev_->rt() : nullptr;
  if (!rt) return "vram: no runtime\n";
  const VkPhysicalDeviceMemoryProperties& m = rt->memProps;
  ss << "vram: device=" << dev_->props().deviceName
     << " heaps=" << m.memoryHeapCount;
  for (uint32_t i = 0; i < m.memoryHeapCount; ++i) {
    const VkMemoryHeap& h = m.memoryHeaps[i];
    ss << " heap" << i << "=" << (h.size >> 20) << "MB";
    if (h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ss << "device-local";
#ifdef VK_MEMORY_HEAP_HOST_VISIBLE_BIT
    if (h.flags & VK_MEMORY_HEAP_HOST_VISIBLE_BIT) ss << "host-visible";
#endif
#ifdef VK_MEMORY_HEAP_HOST_COHERENT_BIT
    if (h.flags & VK_MEMORY_HEAP_HOST_COHERENT_BIT) ss << "coherent";
#endif
#ifdef VK_MEMORY_HEAP_HOST_CACHED_BIT
    if (h.flags & VK_MEMORY_HEAP_HOST_CACHED_BIT) ss << "cached";
#endif
    if (h.flags & ~0xFu) ss << "flags=0x" << std::hex << (h.flags & ~0xFu) << std::dec;
  }
  ss << " memTypes=" << m.memoryTypeCount;
  // KV cache usage: one VBuf per layer for k and v.
  size_t kv_bytes = 0;
  for (const auto& b : k_cache_) kv_bytes += b.nbytes;
  for (const auto& b : v_cache_) kv_bytes += b.nbytes;
  ss << " kv_cache=" << (kv_bytes >> 20) << "MB"
     << " blocks=" << num_blocks_ << " block_size=" << block_size_
     << " layers=" << (int)k_cache_.size();
  return ss.str();
}
