#include "nanovllm/vulkan_backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#define VKC(call) do { VkResult _r=(call); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "VK FAIL %s:%d rc=%d\n", __FILE__, __LINE__, (int)_r); std::exit(1); } } while(0)

static VkRuntime* g_rt = nullptr;
static VkDevice    g_dev = VK_NULL_HANDLE;
static VkQueue     g_q = VK_NULL_HANDLE;
static VkCommandBuffer g_cmd = VK_NULL_HANDLE;

static uint32_t ru(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

// ---------- VBuf ----------
VBuf::VBuf(VBuf&& o) noexcept : buffer(o.buffer), memory(o.memory), nbytes(o.nbytes) {
    o.buffer = VK_NULL_HANDLE; o.memory = VK_NULL_HANDLE; o.nbytes = 0;
}
VBuf& VBuf::operator=(VBuf&& o) noexcept {
    if (this != &o) free();
    buffer = o.buffer; memory = o.memory; nbytes = o.nbytes;
    o.buffer = VK_NULL_HANDLE; o.memory = VK_NULL_HANDLE; o.nbytes = 0;
    return *this;
}
VBuf::~VBuf() { free(); }
void VBuf::alloc(VkDeviceSize bytes, VkBufferUsageFlags usage) {
    free();
    if (bytes <= 0) return;
    nbytes = bytes;
    VkResult _r = vkr_malloc(g_rt, bytes, usage, &buffer, &memory);
    if (_r != VK_SUCCESS) {
        std::fprintf(stderr, "VK ALLOC FAIL bytes=%lld usage=0x%llx rc=%d\n",
                     (long long)bytes, (unsigned long long)usage, (int)_r);
        std::exit(1);
    }
}
void VBuf::free() {
    if (buffer && memory) vkr_free(g_rt, buffer, memory);
    buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; nbytes = 0;
}
void VBuf::upload(const void* host, VkDeviceSize bytes) {
    VKC(vkr_upload(g_rt, g_cmd, g_q, host, buffer, 0, bytes));
}
void VBuf::download(void* host, VkDeviceSize bytes) const {
    VKC(vkr_download(g_rt, g_cmd, g_q, buffer, 0, host, bytes));
}
uint32_t VBuf::words() const { return uint32_t(nbytes / sizeof(uint32_t)); }

// ---------- VulkanBackend ----------
VulkanBackend::VulkanBackend(VkInstance inst, VkPhysicalDevice pd, VkPhysicalDeviceProperties props,
                             VkDevice dev, VkRuntime* rt, uint32_t qfi, bool is_amd, bool is_rdna4)
    : inst_(inst), pd_(pd), props_(props), dev_(dev), rt_(rt), q_(VK_NULL_HANDLE), qfi_(qfi),
      is_amd_(is_amd), is_rdna4_(is_rdna4) {
    g_rt = rt_; g_dev = dev_;
    vkGetDeviceQueue(dev_, qfi_, 0, &q_);
    g_q = q_;

    VKC(vkr_create_command_pool(rt_, qfi_, &pool_));
    VkCommandBufferAllocateInfo ca{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ca.commandPool = pool_; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VKC(vkAllocateCommandBuffers(dev_, &ca, &cmd_));
    g_cmd = cmd_;
    VkFenceCreateInfo fc{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VKC(vkCreateFence(dev_, &fc, nullptr, &fence_));
    VKC(vkr_create_pipeline_cache(dev_, &pcache_));

    VkDescriptorSetLayoutBinding bind[8]{};
    for (int i = 0; i < 8; ++i) {
        bind[i].binding = i; bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].descriptorCount = 1; bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorBindingFlags bflags[8];
    for (int i = 0; i < 8; ++i) bflags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    bfc.bindingCount = 8; bfc.pBindingFlags = bflags;
    VkDescriptorSetLayoutCreateInfo dsci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dsci.pNext = &bfc;
    dsci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
    dsci.bindingCount = 8; dsci.pBindings = bind;
    VKC(vkCreateDescriptorSetLayout(dev_, &dsci, nullptr, &dset_layout_));

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 64 };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &dset_layout_;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VKC(vkCreatePipelineLayout(dev_, &plci, nullptr, &pipe_layout_));

    push_fn_ = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(dev_, "vkCmdPushDescriptorSetKHR");
    if (!push_fn_) { std::fprintf(stderr, "VK: push descriptor unavailable\n"); std::exit(1); }
}

VulkanBackend::~VulkanBackend() {
    vkDeviceWaitIdle(dev_);
    for (auto& kv : pipes_) vkDestroyPipeline(dev_, kv.second.pipe, nullptr);
    vkDestroyPipelineLayout(dev_, pipe_layout_, nullptr);
    vkDestroyDescriptorSetLayout(dev_, dset_layout_, nullptr);
    vkDestroyPipelineCache(dev_, pcache_, nullptr);
    vkDestroyFence(dev_, fence_, nullptr);
    vkDestroyCommandPool(dev_, pool_, nullptr);
    vkr_destroy_runtime(rt_);
    vkDestroyDevice(dev_, nullptr);
    vkDestroyInstance(inst_, nullptr);
}

static VkShaderModule make_shader(VkDevice dev, const unsigned int* spv, size_t nwords) {
    VkShaderModuleCreateInfo si{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    si.codeSize = nwords * sizeof(uint32_t); si.pCode = spv;
    VkShaderModule m; VKC(vkCreateShaderModule(dev, &si, nullptr, &m));
    return m;
}

VkPipeline VulkanBackend::get_pipe(const char* name) {
    auto it = pipes_.find(name);
    if (it != pipes_.end()) return it->second.pipe;
    ensure_pipeline(name);
    return pipes_[name].pipe;
}

void VulkanBackend::ensure_pipeline(const char* name) {
    const unsigned int* spv = spv_for(name);
    unsigned int nwords = spv_count(name);
    if (!spv || !nwords) { std::fprintf(stderr, "VK: no spirv for %s\n", name); std::exit(1); }
    VkShaderModule mod = make_shader(dev_, spv, nwords);
    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.layout = pipe_layout_;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main";
    VkPipeline pipe;
    VKC(vkCreateComputePipelines(dev_, pcache_, 1, &cpi, nullptr, &pipe));
    vkDestroyShaderModule(dev_, mod, nullptr);
    pipes_[name] = PipeEntry{ pipe };
}

void VulkanBackend::begin_if_needed() {
    if (begun_) return;
    VkCommandBufferBeginInfo bb{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(cmd_, &bb));
    begun_ = true;
}

void VulkanBackend::storage_barrier() {
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         1, &mb, 0, nullptr, 0, nullptr);
}

void VulkanBackend::dispatch(const char* name, const void* pc, size_t pc_size,
                             const BufBind* bufs, size_t nbufs, uint32_t gx, uint32_t gy, uint32_t gz) {
    begin_if_needed();
    VkPipeline p = get_pipe(name);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    VkDescriptorBufferInfo infos[8];
    VkWriteDescriptorSet writes[8];
    for (size_t i = 0; i < nbufs; ++i) {
        infos[i].buffer = bufs[i].buffer; infos[i].offset = bufs[i].offset;
        infos[i].range = bufs[i].range ? bufs[i].range : VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = nullptr; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].dstSet = VK_NULL_HANDLE; writes[i].dstBinding = bufs[i].binding;
        writes[i].dstArrayElement = 0; writes[i].descriptorCount = 1; writes[i].pBufferInfo = &infos[i];
    }
    push_fn_(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout_, 0, (uint32_t)nbufs, writes);
    if (pc && pc_size) vkCmdPushConstants(cmd_, pipe_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
    vkCmdDispatch(cmd_, gx, gy, gz);
    storage_barrier();
}

void VulkanBackend::submit_wait() {
    if (!begun_) return;
    VKC(vkEndCommandBuffer(cmd_));
    begun_ = false;
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_;
    VKC(vkQueueSubmit(q_, 1, &si, fence_));
    VKC(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
    VKC(vkResetFences(dev_, 1, &fence_));
    VKC(vkResetCommandBuffer(cmd_, 0));
}

std::unique_ptr<VulkanBackend> VulkanBackend::Create(bool force_device_local) {
    VkApplicationInfo ai{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.pApplicationName = "nanovllm_vulkan"; ai.apiVersion = VK_API_VERSION_1_4;
     VkValidationFeaturesEXT vvf{ VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT };
     VkValidationFeatureEnableEXT vfe = VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT;
     vvf.enabledValidationFeatureCount = 1; vvf.pEnabledValidationFeatures = &vfe;
     VkInstanceCreateInfo ic{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
     ic.pNext = &vvf;
     ic.pApplicationInfo = &ai;
    // Enable validation layers (best-effort: ignored if layer absent).
    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    ic.enabledLayerCount = std::getenv("VK_NO_VALIDATION") ? 0u : 1u;
    ic.ppEnabledLayerNames = std::getenv("VK_NO_VALIDATION") ? nullptr : layers;
    VkInstance inst; VKC(vkCreateInstance(&ic, nullptr, &inst));
    uint32_t npd = 0; vkEnumeratePhysicalDevices(inst, &npd, nullptr);
    if (npd < 1) { std::fprintf(stderr, "VK: no physical device\n"); std::exit(1); }
    std::vector<VkPhysicalDevice> pds(npd);
    vkEnumeratePhysicalDevices(inst, &npd, pds.data());
    int pick = 0;
    for (uint32_t i = 0; i < npd; ++i) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && p.vendorID == 0x1002u) { pick = (int)i; break; }
    }
    VkPhysicalDevice pd = pds[pick];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qps(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qps.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i) if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { std::fprintf(stderr, "VK: no compute queue\n"); std::exit(1); }
    bool is_amd = (props.vendorID == 0x1002u || props.vendorID == 0x1023u);
    bool is_rdna4 = is_amd && (props.apiVersion >= VK_API_VERSION_1_4 || props.deviceID == 0x7550);
    (void)force_device_local; // vkr_malloc routes STORAGE -> device-local on AMD already
    VkDevice dev; VKC(vkr_create_device(pd, qfi, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qfi, 0, &q);
    VkRuntime* rt; VKC(vkr_create_runtime(pd, dev, q, &rt));
    auto* b = new VulkanBackend(inst, pd, props, dev, rt, qfi, is_amd, is_rdna4);
    g_rt = rt; g_dev = dev;
    if (is_rdna4) std::fprintf(stderr, "VK: RDNA4 detected (device-local VRAM enforced)\n");
    return std::unique_ptr<VulkanBackend>(b);
}

// ---------- op dispatchers (bit-exact mirrors of hip_ops.hip; GLSL math matches) ----------
void VulkanBackend::matmul(VBuf& x, VBuf& w, int m, int n, int k, VBuf& y, bool bf16, uint32_t w_off) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    struct PC { int m, n, k; int bf16; uint32_t w_off; } pc{ m, n, k, bf16 ? 1 : 0, w_off };
    VkDeviceSize wbyte = (VkDeviceSize)w_off * 2;
    BufBind bufs[3] = { {0, x.buffer, 0, 0}, {1, w.buffer, wbyte, 0}, {2, y.buffer, 0, 0} };
    dispatch("matmul", &pc, sizeof(pc), bufs, 3, ru(n,16)/16, ru(m,16)/16, 1);
}
void VulkanBackend::matmul_q8(VBuf& x, VBuf& w8, VBuf& ws, int m, int n, int k, VBuf& y, uint32_t w_off, uint32_t ws_off) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    struct PC { int m, n, k; uint32_t w_off; uint32_t ws_off; } pc{ m, n, k, w_off, ws_off };
    BufBind bufs[4] = { {0, x.buffer, 0, 0}, {1, w8.buffer, (VkDeviceSize)w_off, 0},
                        {2, ws.buffer, (VkDeviceSize)ws_off * 2, 0}, {3, y.buffer, 0, 0} };
    dispatch("matmul_q8", &pc, sizeof(pc), bufs, 4, ru(n,16)/16, ru(m,16)/16, 1);
}
void VulkanBackend::matmul(VBuf& x, VMatrix& w, int m, int n, int k, VBuf& y, uint32_t w_off) {
    if (!w.qk_segs.empty()) {
        const QKSeg* seg = &w.qk_segs[0];
        for (auto& s : w.qk_segs)
            if (w_off >= (uint32_t)s.elem_off) seg = &s;
         size_t bb = seg->kind == 12 ? 144 : (seg->kind == 13 ? 176 : 210);
         if (!bb || (w_off - (uint32_t)seg->elem_off) % 256) {
             std::fprintf(stderr, "VK: K-quant row split off super-block boundary\n"); std::exit(1);
         }
         VkDeviceSize wbyte_dbg = (VkDeviceSize)(seg->byte_off + (w_off - (uint32_t)seg->elem_off) / 256 * bb);
         if (k == 3584 && n == 512) std::fprintf(stderr, "VKDBG kmatmul n=%d k=%d w_off=%u seg.kind=%d seg.byte_off=%zu seg.elem_off=%zu bb=%zu wbyte=%llu\n", n, k, w_off, seg->kind, seg->byte_off, seg->elem_off, bb, (unsigned long long)wbyte_dbg);
         matmul_qk(x, w.qk, m, n, k, y, seg->kind, wbyte_dbg);
    } else if (w.is_q8) matmul_q8(x, w.q8, w.qsc, m, n, k, y, w_off, w_off / 32);
    else matmul(x, w.f16, m, n, k, y, w.bf16, w_off);
}
void VulkanBackend::matmul_qk(VBuf& x, VBuf& wq, int m, int n, int k, VBuf& y, int kind, VkDeviceSize wbyte_off) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    struct PC { int m, n, k; int kind; uint32_t woff; } pc{ m, n, k, kind, (uint32_t)wbyte_off };
    BufBind bufs[3] = { {0, x.buffer, 0, 0}, {1, wq.buffer, 0, 0}, {2, y.buffer, 0, 0} };
    std::fprintf(stderr, "VKMATQkk kind=%d n=%d k=%d woff=%u\n", kind, n, k, (uint32_t)wbyte_off);
    dispatch("matmul_qk", &pc, sizeof(pc), bufs, 3, ru(n,16)/16, ru(m,16)/16, 1);
}
void VulkanBackend::rms_norm(VBuf& x, VBuf& w, VBuf& y, int rows, int hidden, float eps) {
    struct PC { int rows, hidden; float eps; } pc{ rows, hidden, eps };
    BufBind bufs[3] = { {0, x.buffer, 0, 0}, {1, w.buffer, 0, 0}, {2, y.buffer, 0, 0} };
    dispatch("rms_norm", &pc, sizeof(pc), bufs, 3, rows, 1, 1);
}
void VulkanBackend::rms_norm_add(VBuf& x, VBuf& residual, VBuf& w, VBuf& y, int rows, int hidden, float eps) {
    struct PC { int rows, hidden; float eps; } pc{ rows, hidden, eps };
    BufBind bufs[4] = { {0, x.buffer, 0, 0}, {1, residual.buffer, 0, 0}, {2, w.buffer, 0, 0}, {3, y.buffer, 0, 0} };
    dispatch("rms_norm_add", &pc, sizeof(pc), bufs, 4, rows, 1, 1);
}
void VulkanBackend::add_bias_inplace(VBuf& x, VBuf& bias, int rows, int cols, uint32_t row_off) {
    struct PC { int rows, cols; } pc{ rows, cols };
    VkDeviceSize ob = (VkDeviceSize)row_off * sizeof(float);
    BufBind bufs[3] = { {0, x.buffer, 0, 0}, {1, bias.buffer, ob, 0}, {2, x.buffer, 0, 0} };
    dispatch("add_bias", &pc, sizeof(pc), bufs, 3, rows, 1, 1);
}
void VulkanBackend::silu_and_mul(VBuf& gup, VBuf& y, int rows, int inter) {
    struct PC { int rows, inter; } pc{ rows, inter };
    BufBind bufs[2] = { {0, gup.buffer, 0, 0}, {1, y.buffer, 0, 0} };
    dispatch("silu_and_mul", &pc, sizeof(pc), bufs, 2, ru(rows*inter,256)/256, 1, 1);
}
void VulkanBackend::rope(VBuf& data, VBuf& pos, VBuf& inv_freq, int tokens, int heads, int head_dim, int64_t stride) {
    int total = tokens * heads * (head_dim/2); if (total <= 0) return;
    struct PC { int tokens, heads, head_dim; int64_t stride; } pc{ tokens, heads, head_dim, stride };
    BufBind bufs[3] = { {0, data.buffer, 0, 0}, {1, pos.buffer, 0, 0}, {2, inv_freq.buffer, 0, 0} };
    dispatch("rope", &pc, sizeof(pc), bufs, 3, ru(total,256)/256, 1, 1);
}
void VulkanBackend::store_kv(VBuf& key, VBuf& val, VBuf& k_cache, VBuf& v_cache, VBuf& slot_map,
                             int kv_heads, int head_dim, int total_tokens, VkDeviceSize k_off, VkDeviceSize v_off) {
    struct PC { int kv_heads, head_dim, total; } pc{ kv_heads, head_dim, total_tokens };
    BufBind bufs[5] = { {0, key.buffer, 0, 0}, {1, val.buffer, 0, 0},
                        {2, k_cache.buffer, k_off * sizeof(float), 0}, {3, v_cache.buffer, v_off * sizeof(float), 0},
                        {4, slot_map.buffer, 0, 0} };
    dispatch("store_kv", &pc, sizeof(pc), bufs, 5, ru(total_tokens,256)/256, 1, 1);
}
void VulkanBackend::paged_attention(VBuf& q, VBuf& y, VBuf& k_cache, VBuf& v_cache, VBuf& qseq, VBuf& qlen,
                                    VBuf& bt, int tokens, int q_heads, int kv_heads, int head_dim,
                                    int block_size, int max_blocks, float scale, VkDeviceSize k_off, VkDeviceSize v_off) {
    if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0) return;
    struct PC { int tokens, qh, kvh, hd, bs, mb; float scale; } pc{ tokens, q_heads, kv_heads, head_dim, block_size, max_blocks, scale };
    BufBind bufs[7] = { {0, q.buffer, 0, 0}, {1, y.buffer, 0, 0},
                        {2, k_cache.buffer, k_off * sizeof(float), 0}, {3, v_cache.buffer, v_off * sizeof(float), 0},
                        {4, qseq.buffer, 0, 0}, {5, qlen.buffer, 0, 0}, {6, bt.buffer, 0, 0} };
    dispatch("paged_attention", &pc, sizeof(pc), bufs, 7, tokens, q_heads, 1);
}
void VulkanBackend::embedding(VBuf& ids, VMatrix& w, VBuf& y, int tokens, int hidden) {
    int total = tokens * hidden; if (total <= 0) return;
    if (!w.qk_segs.empty()) {
        embedding_qk(ids, w.qk, y, tokens, hidden, w.qk_segs[0].kind);
    } else if (w.is_q8) {
        struct PC2 { int hidden, total; } pc{ hidden, total };
        BufBind bufs[4] = { {0, ids.buffer, 0, 0}, {1, w.q8.buffer, 0, 0},
                            {2, w.qsc.buffer, 0, 0}, {3, y.buffer, 0, 0} };
        dispatch("embedding_q8", &pc, sizeof(pc), bufs, 4, ru(total,256)/256, 1, 1);
    } else {
        struct PC2 { int hidden, total, bf16; } pc{ hidden, total, w.bf16 ? 1 : 0 };
        BufBind bufs[3] = { {0, ids.buffer, 0, 0}, {1, w.f16.buffer, 0, 0}, {2, y.buffer, 0, 0} };
        dispatch("embedding_f16", &pc, sizeof(pc), bufs, 3, ru(total,256)/256, 1, 1);
    }
}
void VulkanBackend::embedding_qk(VBuf& ids, VBuf& wq, VBuf& y, int tokens, int hidden, int kind) {
    int total = tokens * hidden; if (total <= 0) return;
    struct PC2 { int hidden, total, kind; } pc{ hidden, total, kind };
    BufBind bufs[3] = { {0, ids.buffer, 0, 0}, {1, wq.buffer, 0, 0}, {2, y.buffer, 0, 0} };
    dispatch("embedding_qk", &pc, sizeof(pc), bufs, 3, ru(total,256)/256, 1, 1);
}
void VulkanBackend::gather_rows(VBuf& rows, VBuf& idx, VBuf& y, int cols, int total) {
    if (total <= 0 || cols <= 0) return;
    struct PC { int rows, cols, total; } pc{ total / cols, cols, total };
    BufBind bufs[3] = { {0, rows.buffer, 0, 0}, {1, idx.buffer, 0, 0}, {2, y.buffer, 0, 0} };
    dispatch("gather_rows", &pc, sizeof(pc), bufs, 3, ru(total,256)/256, 1, 1);
}
