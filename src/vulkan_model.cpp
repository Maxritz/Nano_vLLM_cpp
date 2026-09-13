#include "nanovllm/vulkan_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

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
bool VulkanModel::load_float(const std::string& n, std::vector<float>& out) const {
    if (st_) return st_loader_.load_float(n, out);
    return gg_loader_.load_float(map_name(n), out);
}
bool VulkanModel::load_u16(const std::string& n, std::vector<uint16_t>& out, bool& bf16) const {
    if (st_) return st_loader_.load_u16(n, out, bf16);
    return gg_loader_.load_u16(map_name(n), out, bf16);
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
void VulkanModel::load_matrix(const std::string& name, VMatrix& m, bool required, size_t n_elements) {
    std::vector<uint8_t> q8;
    if (load_q8(name, q8)) { upload_q8_block(q8, m); m.f16.free(); m.qk.free(); m.qk_segs.clear(); return; }
    for (int kind : {12, 13, 14}) {
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
    { // MoE weights would otherwise be silently ignored (garbage output). Fail loud.
      auto names = st_ ? st_loader_.names() : gg_loader_.names();
      for (auto& n : names)
        if (n.find("exps") != std::string::npos || n.find(".experts.") != std::string::npos ||
            n.find("gate_inp") != std::string::npos || n.find("shared_expert") != std::string::npos ||
            n.find("moe") != std::string::npos)
          throw std::runtime_error("MoE models are not supported by this dense engine");
    }
    if (!contains("model.embed_tokens.weight")) throw std::runtime_error("model.embed_tokens.weight not found");
    auto& hf = config_.hf;
    int hidden = hf.hidden_size, inter = hf.intermediate_size;
    load_matrix("model.embed_tokens.weight", embed_, true, size_t(hf.vocab_size) * hidden);
    for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
        auto& lw = layers_[layer];
        std::string p = "model.layers." + std::to_string(layer);
        if (contains(p + ".self_attn.qkv_proj.weight")) {
            load_matrix(p + ".self_attn.qkv_proj.weight", lw.qkv, true, size_t(qkv_size_) * hidden);
        } else {
            std::vector<uint16_t> q_u,k_u,v_u; bool qbf=false,kbf=false,vbf=false;
            std::vector<uint8_t> q_q8,k_q8,v_q8; bool q8q=false,k8q=false,v8q=false;
            std::vector<uint8_t> q_qk,k_qk,v_qk; int qkind=0,kkind=0,vkind=0;
            load_matrix_host(p + ".self_attn.q_proj.weight", q_u,qbf,q_q8,q8q,q_qk,qkind,size_t(q_size_)*hidden,true);
            load_matrix_host(p + ".self_attn.k_proj.weight", k_u,kbf,k_q8,k8q,k_qk,kkind,size_t(kv_size_)*hidden,true);
            load_matrix_host(p + ".self_attn.v_proj.weight", v_u,vbf,v_q8,v8q,v_qk,vkind,size_t(kv_size_)*hidden,true);
            if (q8q||k8q||v8q) {
                if(!(q8q&&k8q&&v8q)) throw std::runtime_error("mixed Q8 q/k/v unsupported");
                std::vector<uint8_t> f; f.insert(f.end(),q_q8.begin(),q_q8.end());
                f.insert(f.end(),k_q8.begin(),k_q8.end()); f.insert(f.end(),v_q8.begin(),v_q8.end());
                upload_q8_block(f, lw.qkv);
            } else if (qkind||kkind||vkind) {
                if (!(qkind && kkind && vkind))
                    throw std::runtime_error("mixed K-quant/non-K-quant q/k/v unsupported");
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
                std::vector<uint16_t> f; f.insert(f.end(),q_u.begin(),q_u.end());
                f.insert(f.end(),k_u.begin(),k_u.end()); f.insert(f.end(),v_u.begin(),v_u.end());
                upload_u16_block(f, lw.qkv); lw.qkv.bf16 = qbf && kbf && vbf;
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
        if (contains(p + ".mlp.gate_up_proj.weight")) {
            load_matrix(p + ".mlp.gate_up_proj.weight", lw.gate_up, true, size_t(2 * inter) * hidden);
        } else {
            std::vector<uint16_t> g_u,u_u; bool gbf=false,ubf=false;
            std::vector<uint8_t> g_q8,u_q8; bool gq=false,uq=false;
            std::vector<uint8_t> g_qk,u_qk; int gkind=0,ukind=0;
            load_matrix_host(p + ".mlp.gate_proj.weight", g_u,gbf,g_q8,gq,g_qk,gkind,size_t(inter)*hidden,true);
            load_matrix_host(p + ".mlp.up_proj.weight", u_u,ubf,u_q8,uq,u_qk,ukind,size_t(inter)*hidden,true);
            if (gq||uq) {
                if(!(gq&&uq)) throw std::runtime_error("mixed Q8 gate/up unsupported");
                std::vector<uint8_t> f; f.insert(f.end(),g_q8.begin(),g_q8.end()); f.insert(f.end(),u_q8.begin(),u_q8.end());
                upload_q8_block(f, lw.gate_up);
            } else if (gkind||ukind) {
                if (!(gkind && gkind==ukind))
                    throw std::runtime_error("mixed K-quant gate/up needs split dispatch (unsupported)");
                if ((size_t(inter)*hidden)%256)
                    throw std::runtime_error("K-quant gate/up split off super-block boundary");
                size_t gb = size_t(inter)*hidden/256*GGUFLoader::qk_block_bytes(gkind);
                std::vector<uint8_t> f; f.insert(f.end(),g_qk.begin(),g_qk.end()); f.insert(f.end(),u_qk.begin(),u_qk.end());
                upload_qk_block(f, gkind, lw.gate_up);
                lw.gate_up.qk_segs = {{gkind,0,0},{ukind,gb,size_t(inter)*hidden}};
            } else {
                std::vector<uint16_t> f; f.insert(f.end(),g_u.begin(),g_u.end()); f.insert(f.end(),u_u.begin(),u_u.end());
                upload_u16_block(f, lw.gate_up); lw.gate_up.bf16 = gbf && ubf;
            }
        }
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
    (void)load_optional("model.norm.weight", fn);
    final_norm_ = dev_->make(fn);
    tie_lm_head_ = hf.tie_word_embeddings;
    if (!tie_lm_head_ && contains("lm_head.weight"))
        load_matrix("lm_head.weight", lm_head_, false, size_t(hf.vocab_size) * hidden);
    ready_ = true;
    dev_->submit_wait();
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

std::vector<float> VulkanModel::forward_logits(const VKContext& ctx) {
    if (!ready_) throw std::runtime_error("model weights not loaded");
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
    gate_dev.alloc((VkDeviceSize)rows * 2 * inter * F4);
    mlp_dev.alloc((VkDeviceSize)rows * inter * F4);
    { std::vector<float> zeros((size_t)rows * hidden, 0.0f);  // HIP zeroes scratch at alloc; residual is read-before-write
      residual_dev.upload(zeros.data(), (VkDeviceSize)zeros.size() * F4); }

    dev_->embedding(d_ids, embed_, hidden_dev, rows, hidden);
    for (int layer = 0; layer < hf.num_hidden_layers; ++layer) {
        auto& lw = layers_[layer];
        dev_->rms_norm_add(hidden_dev, residual_dev, lw.input_ln, norm_dev, rows, hidden, eps);
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
        dev_->rope(q_dev, d_pos, inv_freq_, rows, heads, head_dim, q_size_);
        dev_->rope(k_dev, d_pos, inv_freq_, rows, kv_heads, head_dim, kv_size_);
        int d = kv_heads * head_dim;
        if (d_slot.buffer && !ctx.slot_mapping.empty())
            dev_->store_kv(k_dev, v_dev, k_cache_[layer], v_cache_[layer], d_slot, kv_heads, head_dim, rows * d, 0u, 0u);
        if (d_qseq.buffer && !ctx.query_seq.empty())
            dev_->paged_attention(q_dev, attn_dev, k_cache_[layer], v_cache_[layer], d_qseq, d_qlen, d_tbl,
                                  rows, heads, kv_heads, head_dim, block_size_, ctx.max_blocks, scale, 0u, 0u);
        dev_->matmul(attn_dev, lw.o, rows, hidden, q_size_, hidden_dev);
        dev_->rms_norm_add(hidden_dev, residual_dev, lw.post_ln, norm_dev, rows, hidden, eps);
        dev_->matmul(norm_dev, lw.gate_up, rows, 2 * inter, hidden, gate_dev);
        dev_->silu_and_mul(gate_dev, mlp_dev, rows, inter);
        dev_->matmul(mlp_dev, lw.down, rows, hidden, inter, hidden_dev);
    }
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
    return out;
}
