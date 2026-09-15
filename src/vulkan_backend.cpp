#include "nanovllm/vulkan_backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <unordered_map>
#include <vector>

#define VKC(call) do { VkResult _r=(call); if (_r != VK_SUCCESS) { \
    std::fprintf(stderr, "VK FAIL %s:%d rc=%d\n", __FILE__, __LINE__, (int)_r); std::exit(1); } } while(0)

static VkDevice               g_dev = VK_NULL_HANDLE;
static VkQueue              g_q = VK_NULL_HANDLE;
static uint32_t             g_qfi = 0;
static VkPhysicalDeviceMemoryProperties g_memProps{};
static VkCommandBuffer      g_cmd = VK_NULL_HANDLE;
static VkCommandPool        g_xferPool = VK_NULL_HANDLE;
static VkCommandBuffer      g_xferCmd = VK_NULL_HANDLE;
static VkFence              g_xferFence = VK_NULL_HANDLE;
// Async-transfer state (P2-3): dedicated transfer queue when the device has a
// transfer-capable family distinct from compute; else g_xq == g_q (same-queue
// FIFO keeps ordering, no semaphore needed).
static VkQueue              g_xq = VK_NULL_HANDLE;
static uint32_t             g_xqfi = 0;
struct PendingXfer { VkBuffer staging; VkDeviceMemory mem; VkCommandBuffer cmd; VkSemaphore sem; };
static std::vector<PendingXfer> g_xfer_pending;
static uint32_t             g_share_idx[2] = {0, 0};
static void apply_sharing(VkBufferCreateInfo& bci) {
    if (g_xq != VK_NULL_HANDLE && g_xqfi != g_qfi) {
        g_share_idx[0] = g_qfi; g_share_idx[1] = g_xqfi;
        bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bci.queueFamilyIndexCount = 2;
        bci.pQueueFamilyIndices = g_share_idx;
    }
}

static uint32_t ru(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

// ── memory-type helpers ────────────────────────────────────────────────────
static bool try_mem_type(uint32_t typeFilter, VkMemoryPropertyFlags props, uint32_t& out) {
    for (uint32_t i = 0; i < g_memProps.memoryTypeCount; ++i)
        if ((typeFilter & (1u << i)) && (g_memProps.memoryTypes[i].propertyFlags & props) == props) {
            out = i; return true;
        }
    return false;
}
static uint32_t find_mem_type(uint32_t typeFilter, VkMemoryPropertyFlags props) {
    uint32_t idx;
    if (try_mem_type(typeFilter, props, idx)) return idx;
    std::fprintf(stderr, "VK: no memory type (filter=0x%x props=0x%x)\n", typeFilter, (unsigned)props);
    std::exit(1);
}

// ── buffer helpers (replace VAiT vkruntime malloc/free/upload/download) ────
static void vkrt_malloc(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer* buffer, VkDeviceMemory* memory) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = bytes;
    bci.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    apply_sharing(bci);
    VKC(vkCreateBuffer(g_dev, &bci, nullptr, buffer));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, *buffer, &mr);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    uint32_t idx;
    if (!try_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, idx) &&
        !try_mem_type(mr.memoryTypeBits,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, idx)) {
        std::fprintf(stderr, "VK: no device-local or host-visible memory for %lld bytes\n", (long long)bytes);
        std::exit(1);
    }
    mai.memoryTypeIndex = idx;
    VKC(vkAllocateMemory(g_dev, &mai, nullptr, memory));
    VKC(vkBindBufferMemory(g_dev, *buffer, *memory, 0));
}

static void vkrt_free(VkBuffer buffer, VkDeviceMemory memory) {
    if (buffer)  vkDestroyBuffer(g_dev, buffer, nullptr);
    if (memory)  vkFreeMemory(g_dev, memory, nullptr);
}

static void xfer_stage(const void* host, VkDeviceSize bytes, VkBuffer* staging, VkDeviceMemory* mem) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    apply_sharing(bci);
    VKC(vkCreateBuffer(g_dev, &bci, nullptr, staging));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, *staging, &mr);
    uint32_t idx = find_mem_type(mr.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = idx;
    VKC(vkAllocateMemory(g_dev, &mai, nullptr, mem));
    vkBindBufferMemory(g_dev, *staging, *mem, 0);
    void* mapped;
    VKC(vkMapMemory(g_dev, *mem, 0, bytes, 0, &mapped));
    std::memcpy(mapped, host, (size_t)bytes);
    vkUnmapMemory(g_dev, *mem);
}

// Fire-and-forget upload on the transfer queue; ordered before later compute
// via queue FIFO (same family) or a signal semaphore consumed by submit_wait.
static void vkrt_upload_async(const void* host, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes) {
    VkBuffer staging; VkDeviceMemory mem;
    xfer_stage(host, bytes, &staging, &mem);
    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = g_xferPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VKC(vkAllocateCommandBuffers(g_dev, &cai, &cb));
    VkCommandBufferBeginInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(cb, &cbi));
    VkBufferCopy bc{ 0, offset, bytes };
    vkCmdCopyBuffer(cb, staging, buffer, 1, &bc);
    VKC(vkEndCommandBuffer(cb));
    VkSemaphore sem = VK_NULL_HANDLE;
    if (g_xqfi != g_qfi) {
        VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VKC(vkCreateSemaphore(g_dev, &sci, nullptr, &sem));
    }
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    if (sem) { si.signalSemaphoreCount = 1; si.pSignalSemaphores = &sem; }
    VKC(vkQueueSubmit(g_xq, 1, &si, VK_NULL_HANDLE));
    g_xfer_pending.push_back({staging, mem, cb, sem});
}

// Free async-transfer resources; call after the compute fence proves the
// transfer was consumed (same-family FIFO) or waited on (cross-family sem).
static void xfer_drain() {
    for (auto& p : g_xfer_pending) {
        if (p.sem) vkDestroySemaphore(g_dev, p.sem, nullptr);
        vkFreeCommandBuffers(g_dev, g_xferPool, 1, &p.cmd);
        vkFreeMemory(g_dev, p.mem, nullptr);
        vkDestroyBuffer(g_dev, p.staging, nullptr);
    }
    g_xfer_pending.clear();
}

static void vkrt_upload(const void* host, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes) {
    VkBuffer staging; VkDeviceMemory stagingMem;
    xfer_stage(host, bytes, &staging, &stagingMem);
    VkCommandBufferBeginInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(g_xferCmd, &cbi));
    VkBufferCopy bc{ 0, offset, bytes };
    vkCmdCopyBuffer(g_xferCmd, staging, buffer, 1, &bc);
    VKC(vkEndCommandBuffer(g_xferCmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &g_xferCmd;
    VKC(vkQueueSubmit(g_xq, 1, &si, g_xferFence));
    VKC(vkWaitForFences(g_dev, 1, &g_xferFence, VK_TRUE, UINT64_MAX));
    vkResetFences(g_dev, 1, &g_xferFence);
    vkResetCommandBuffer(g_xferCmd, 0);
    vkFreeMemory(g_dev, stagingMem, nullptr);
    vkDestroyBuffer(g_dev, staging, nullptr);
}

static void vkrt_download(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize bytes, void* host) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    apply_sharing(bci);
    VkBuffer staging;
    VKC(vkCreateBuffer(g_dev, &bci, nullptr, &staging));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, staging, &mr);
    uint32_t idx = find_mem_type(mr.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = idx;
    VkDeviceMemory stagingMem;
    VKC(vkAllocateMemory(g_dev, &mai, nullptr, &stagingMem));
    vkBindBufferMemory(g_dev, staging, stagingMem, 0);
    VkCommandBufferBeginInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(g_xferCmd, &cbi));
    VkBufferCopy bc{ offset, 0, bytes };
    vkCmdCopyBuffer(g_xferCmd, buffer, staging, 1, &bc);
    VKC(vkEndCommandBuffer(g_xferCmd));
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &g_xferCmd;
    VKC(vkQueueSubmit(g_xq, 1, &si, g_xferFence));
    VKC(vkWaitForFences(g_dev, 1, &g_xferFence, VK_TRUE, UINT64_MAX));
    vkResetFences(g_dev, 1, &g_xferFence);
    vkResetCommandBuffer(g_xferCmd, 0);
    void* mapped;
    VKC(vkMapMemory(g_dev, stagingMem, 0, bytes, 0, &mapped));
    std::memcpy(host, mapped, (size_t)bytes);
    vkUnmapMemory(g_dev, stagingMem);
    vkFreeMemory(g_dev, stagingMem, nullptr);
    vkDestroyBuffer(g_dev, staging, nullptr);
}

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
    vkrt_malloc(bytes, usage, &buffer, &memory);
}
void VBuf::free() {
    vkrt_free(buffer, memory);
    buffer = VK_NULL_HANDLE; memory = VK_NULL_HANDLE; nbytes = 0;
}
void VBuf::upload(const void* host, VkDeviceSize bytes) {
    vkrt_upload(host, buffer, 0, bytes);
}
void VBuf::upload_at(VkDeviceSize offset, const void* host, VkDeviceSize bytes) {
    vkrt_upload(host, buffer, offset, bytes);
}
void VBuf::upload_at_async(VkDeviceSize offset, const void* host, VkDeviceSize bytes) {
    vkrt_upload_async(host, buffer, offset, bytes);
}
void VBuf::download(void* host, VkDeviceSize bytes) const {
    vkrt_download(buffer, 0, bytes, host);
}
uint32_t VBuf::words() const { return uint32_t(nbytes / sizeof(uint32_t)); }

// ---------- VulkanBackend ----------
VulkanBackend::VulkanBackend(VkInstance inst, VkPhysicalDevice pd, VkPhysicalDeviceProperties props,
                             VkDevice dev, VkRT* rt, uint32_t qfi, bool is_amd, bool is_rdna4)
    : inst_(inst), pd_(pd), props_(props), dev_(dev), rt_(rt), q_(VK_NULL_HANDLE), qfi_(qfi),
      is_amd_(is_amd), is_rdna4_(is_rdna4) {
    g_dev = dev_;
    g_memProps = rt_->memProps;
    g_qfi = qfi_;
    vkGetDeviceQueue(dev_, qfi_, 0, &q_);
    g_q = q_;

    VkCommandPoolCreateInfo cpi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.queueFamilyIndex = qfi_;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VKC(vkCreateCommandPool(dev_, &cpi, nullptr, &pool_));

    VkCommandPoolCreateInfo xp{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    xp.queueFamilyIndex = g_xqfi;  // transfer family (== compute family if no dedicated one)
    xp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VKC(vkCreateCommandPool(dev_, &xp, nullptr, &g_xferPool));
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = g_xferPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VKC(vkAllocateCommandBuffers(dev_, &cai, &g_xferCmd));
    VkFenceCreateInfo fc{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VKC(vkCreateFence(dev_, &fc, nullptr, &g_xferFence));

    VkCommandBufferAllocateInfo ca{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ca.commandPool = pool_; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    VKC(vkAllocateCommandBuffers(dev_, &ca, &cmd_));
    g_cmd = cmd_;
    VKC(vkCreateFence(dev_, &fc, nullptr, &fence_));
    VkPipelineCacheCreateInfo pci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    VKC(vkCreatePipelineCache(dev_, &pci, nullptr, &pcache_));

    VkDescriptorSetLayoutBinding bind[8]{};
    for (int i = 0; i < 8; ++i) {
        bind[i].binding = i; bind[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bind[i].descriptorCount = 1; bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorBindingFlags bflags[8];
    for (int i = 0; i < 8; ++i) bflags[i] = 0;  // push descriptor layout: no per-binding flags
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
    if (!push_fn_) { std::fprintf(stderr, "VK: push descriptor unavailable — enable VK_KHR_push_descriptor\n"); std::exit(1); }
}

VulkanBackend::~VulkanBackend() {
    vkDeviceWaitIdle(dev_);
    xfer_drain();
    for (auto& kv : pipes_) vkDestroyPipeline(dev_, kv.second.pipe, nullptr);
    vkDestroyPipelineLayout(dev_, pipe_layout_, nullptr);
    vkDestroyDescriptorSetLayout(dev_, dset_layout_, nullptr);
    vkDestroyPipelineCache(dev_, pcache_, nullptr);
    vkDestroyFence(dev_, fence_, nullptr);
    vkDestroyCommandPool(dev_, pool_, nullptr);
    vkDestroyCommandPool(dev_, g_xferPool, nullptr);
    vkDestroyFence(dev_, g_xferFence, nullptr);
    delete rt_;
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

VkPipeline VulkanBackend::get_pipe_spec(const char* spv_name, const char* key,
                                        const VkSpecializationMapEntry* entries, uint32_t n_entries,
                                        const void* data, size_t data_size) {
    auto it = pipes_.find(key);
    if (it != pipes_.end()) return it->second.pipe;
    ensure_pipeline_spec(spv_name, key, entries, n_entries, data, data_size);
    return pipes_[key].pipe;
}

void VulkanBackend::ensure_pipeline_spec(const char* spv_name, const char* key,
                                         const VkSpecializationMapEntry* entries, uint32_t n_entries,
                                         const void* data, size_t data_size) {
    const unsigned int* spv = spv_for(spv_name);
    unsigned int nwords = spv_count(spv_name);
    if (!spv || !nwords) { std::fprintf(stderr, "VK: no spirv for %s\n", spv_name); std::exit(1); }
    VkShaderModule mod = make_shader(dev_, spv, nwords);
    VkSpecializationInfo spec{};
    spec.mapEntryCount = n_entries;
    spec.pMapEntries = entries;
    spec.dataSize = data_size;
    spec.pData = data;
    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.layout = pipe_layout_;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main";
    cpi.stage.pSpecializationInfo = &spec;
    VkPipeline pipe;
    VKC(vkCreateComputePipelines(dev_, pcache_, 1, &cpi, nullptr, &pipe));
    vkDestroyShaderModule(dev_, mod, nullptr);
    pipes_[key] = PipeEntry{ pipe };
}

void VulkanBackend::dispatch_bound(VkPipeline p,
                                  const void* pc, size_t pc_size,
                                  const BufBind* bufs, size_t nbufs, uint32_t gx, uint32_t gy, uint32_t gz) {
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
    dispatch_bound(get_pipe(name), pc, pc_size, bufs, nbufs, gx, gy, gz);
}

void VulkanBackend::dispatch_spec(const char* spv_name, const char* key,
                                  const VkSpecializationMapEntry* entries, uint32_t n_entries,
                                  const void* data, size_t data_size,
                                  const void* pc, size_t pc_size,
                                  const BufBind* bufs, size_t nbufs, uint32_t gx, uint32_t gy, uint32_t gz) {
    begin_if_needed();
    dispatch_bound(get_pipe_spec(spv_name, key, entries, n_entries, data, data_size),
                   pc, pc_size, bufs, nbufs, gx, gy, gz);
}

void VulkanBackend::submit_wait() {
    if (!begun_) return;
    VKC(vkEndCommandBuffer(cmd_));
    begun_ = false;
    // Cross-family async uploads signal here: compute waits before reading them.
    // Same-family uploads need no wait (queue FIFO already orders them).
    std::vector<VkSemaphore> waits;
    std::vector<VkPipelineStageFlags> stages;
    for (auto& p : g_xfer_pending)
        if (p.sem) { waits.push_back(p.sem); stages.push_back(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT); }
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = (uint32_t)waits.size();
    si.pWaitSemaphores = waits.data();
    si.pWaitDstStageMask = stages.data();
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd_;
    VKC(vkQueueSubmit(q_, 1, &si, fence_));
    VKC(vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
    VKC(vkResetFences(dev_, 1, &fence_));
    VKC(vkResetCommandBuffer(cmd_, 0));
    xfer_drain();
}

std::unique_ptr<VulkanBackend> VulkanBackend::Create(bool force_device_local) {
    VkApplicationInfo ai{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.pApplicationName = "nanovllm_vulkan"; ai.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo ic{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ic.pApplicationInfo = &ai;
    // Best-effort validation layer: enable only if present and not suppressed.
    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    if (!std::getenv("VK_NO_VALIDATION")) {
        uint32_t nLayers = 0;
        vkEnumerateInstanceLayerProperties(&nLayers, nullptr);
        std::vector<VkLayerProperties> avail(nLayers);
        vkEnumerateInstanceLayerProperties(&nLayers, avail.data());
        for (const auto& l : avail)
            if (strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                ic.enabledLayerCount = 1; ic.ppEnabledLayerNames = layers; break;
            }
    }

    VkInstance inst; VKC(vkCreateInstance(&ic, nullptr, &inst));
    uint32_t npd = 0; vkEnumeratePhysicalDevices(inst, &npd, nullptr);
    if (npd < 1) { std::fprintf(stderr, "VK: no physical device\n"); std::exit(1); }
    std::vector<VkPhysicalDevice> pds(npd);
    vkEnumeratePhysicalDevices(inst, &npd, pds.data());

    // Generic device selection: prefer discrete GPU of any vendor, fall back to first.
    int pick = -1;
    for (uint32_t i = 0; i < npd; ++i) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pick = (int)i; break; }
        if (pick < 0) pick = (int)i;
    }
    if (pick < 0) { std::fprintf(stderr, "VK: no suitable physical device\n"); std::exit(1); }

    VkPhysicalDevice pd = pds[pick];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qps(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qps.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i) if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    if (qfi == UINT32_MAX) { std::fprintf(stderr, "VK: no compute queue\n"); std::exit(1); }
    // Dedicated transfer queue when the device exposes a transfer-capable
    // family distinct from compute (async DMA overlap); else shared FIFO.
    uint32_t xqfi = qfi;
    for (uint32_t i = 0; i < nq; ++i)
        if (i != qfi && (qps[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) { xqfi = i; break; }

    // Query device extensions
    uint32_t nExt = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &nExt, nullptr);
    std::vector<VkExtensionProperties> exts(nExt);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &nExt, exts.data());
    bool hasFloat16Int8 = false, hasPushDescriptor = false, hasScalarLayout = false;
    for (const auto& e : exts) {
        if (strcmp(e.extensionName, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) == 0) hasFloat16Int8 = true;
        if (strcmp(e.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) hasPushDescriptor = true;
        if (strcmp(e.extensionName, VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME) == 0) hasScalarLayout = true;
    }

    // Query features: Vulkan 1.1 (16-bit storage) + 1.2 (8-bit storage, float16/int8) + core
    VkPhysicalDeviceVulkan11Features vk11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    VkPhysicalDeviceVulkan12Features vk12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    VkPhysicalDeviceFeatures2 feats2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    vk12.pNext = nullptr;
    vk11.pNext = &vk12;
    feats2.pNext = &vk11;
    vkGetPhysicalDeviceFeatures2(pd, &feats2);
    bool hasShaderInt64 = feats2.features.shaderInt64;
    bool hasStorage16 = vk11.storageBuffer16BitAccess;
    bool hasStorage8 = vk12.storageBuffer8BitAccess;

    // Assemble device extension list (enable only what's available)
    std::vector<const char*> devExts;
    if (hasPushDescriptor) devExts.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    if (hasScalarLayout) devExts.push_back(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
    if (hasFloat16Int8) devExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);

    std::fprintf(stderr, "VK: selected %s (vendor=0x%x, api=%u.%u)\n",
                 props.deviceName, props.vendorID,
                 VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion));

    float qprio[2] = {1.0f, 1.0f};
    VkDeviceQueueCreateInfo qcis[2]{};
    qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qcis[0].queueFamilyIndex = qfi; qcis[0].queueCount = 1; qcis[0].pQueuePriorities = &qprio[0];
    uint32_t nqci = 1;
    if (xqfi != qfi) {
        qcis[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qcis[1].queueFamilyIndex = xqfi; qcis[1].queueCount = 1; qcis[1].pQueuePriorities = &qprio[1];
        nqci = 2;
    }

    // Enable: 16-bit storage (fp16/bf16 weights) + 8-bit storage (Q8 path) + shaderInt64/16
    VkPhysicalDeviceVulkan11Features dev11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
    dev11.storageBuffer16BitAccess = hasStorage16 ? VK_TRUE : VK_FALSE;
    dev11.uniformAndStorageBuffer16BitAccess = hasStorage16 ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan12Features dev12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    dev12.storageBuffer8BitAccess = hasStorage8 ? VK_TRUE : VK_FALSE;
    dev12.uniformAndStorageBuffer8BitAccess = hasStorage8 ? VK_TRUE : VK_FALSE;
    dev12.shaderFloat16 = vk12.shaderFloat16;
    dev12.shaderInt8 = vk12.shaderInt8;

    VkPhysicalDeviceFeatures devCoreFeats{};
    devCoreFeats.shaderInt64 = hasShaderInt64 ? VK_TRUE : VK_FALSE;
    devCoreFeats.shaderInt16 = feats2.features.shaderInt16 ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = nqci; dci.pQueueCreateInfos = qcis;
    dci.enabledExtensionCount = (uint32_t)devExts.size();
    dci.ppEnabledExtensionNames = devExts.data();
    dci.pEnabledFeatures = &devCoreFeats;
    dev12.pNext = nullptr;
    dev11.pNext = &dev12;
    dci.pNext = &dev11;

    (void)force_device_local;
    VkDevice dev; VKC(vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q; vkGetDeviceQueue(dev, qfi, 0, &q);
    g_xqfi = xqfi;
    g_xq = q;
    if (xqfi != qfi) {
        vkGetDeviceQueue(dev, xqfi, 0, &g_xq);
        std::fprintf(stderr, "VK: dedicated transfer queue family %u\n", xqfi);
    }

    auto* rt = new VkRT;
    rt->device = dev; rt->physDev = pd; rt->queue = q; rt->qfi = qfi;
    vkGetPhysicalDeviceMemoryProperties(pd, &rt->memProps);
    g_memProps = rt->memProps;

    auto* b = new VulkanBackend(inst, pd, props, dev, rt, qfi,
                                props.vendorID == 0x1002u || props.vendorID == 0x1023u,
                                (props.vendorID == 0x1002u || props.vendorID == 0x1023u) && props.apiVersion >= VK_API_VERSION_1_4);
    g_dev = dev;
    return std::unique_ptr<VulkanBackend>(b);
}

// ── op dispatchers (bit-exact mirrors of hip_ops.hip; GLSL math matches) ─────
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
        VkDeviceSize wbyte = (VkDeviceSize)(seg->byte_off + (w_off - (uint32_t)seg->elem_off) / 256 * bb);
        matmul_qk(x, w.qk, m, n, k, y, seg->kind, wbyte);
    } else if (w.is_q8) matmul_q8(x, w.q8, w.qsc, m, n, k, y, w_off, w_off / 32);
    else matmul(x, w.f16, m, n, k, y, w.bf16, w_off);
}
void VulkanBackend::matmul_qk(VBuf& x, VBuf& wq, int m, int n, int k, VBuf& y, int kind, VkDeviceSize wbyte_off) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    struct PC { int m, n, k; int kind; uint32_t woff; } pc{ m, n, k, kind, (uint32_t)wbyte_off };
    BufBind bufs[3] = { {0, x.buffer, 0, 0}, {1, wq.buffer, 0, 0}, {2, y.buffer, 0, 0} };
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
void VulkanBackend::scale_sigmoid(VBuf& x, VBuf& s, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    struct PC { int rows, cols; } pc{ rows, cols };
    BufBind bufs[2] = { {0, x.buffer, 0, 0}, {1, s.buffer, 0, 0} };
    dispatch("scale_sigmoid", &pc, sizeof(pc), bufs, 2, rows, 1, 1);
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
    paged_attention_ex(q, y, k_cache, v_cache, qseq, qlen, bt, tokens, q_heads, kv_heads, head_dim,
                       block_size, max_blocks, scale, 0, 0.0f, k_off, v_off);
}
void VulkanBackend::paged_attention_ex(VBuf& q, VBuf& y, VBuf& k_cache, VBuf& v_cache, VBuf& qseq, VBuf& qlen,
                                    VBuf& bt, int tokens, int q_heads, int kv_heads, int head_dim,
                                    int block_size, int max_blocks, float scale, int sw_start, float attn_softcap,
                                    VkDeviceSize k_off, VkDeviceSize v_off) {
    if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0) return;
    struct PC { int tokens, qh, kvh, hd, bs, mb; float scale; int sw_start; float softcap; } pc{ tokens, q_heads, kv_heads, head_dim, block_size, max_blocks, scale, sw_start, attn_softcap };
    static_assert(sizeof(PC) == 36, "paged_attention PC must stay 36B (pipe_layout_ allows 64B)");
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

void VulkanBackend::moe_route(VBuf& logits, VBuf& idx, VBuf& score, int rows, int E, int K, bool norm) {
    if (rows <= 0 || E <= 0 || K <= 0) return;
    struct PC { int rows, E, K, norm; } pc{ rows, E, K, norm ? 1 : 0 };
    BufBind bufs[3] = { {0, logits.buffer, 0, 0}, {1, idx.buffer, 0, 0}, {2, score.buffer, 0, 0} };
    dispatch("moe_route", &pc, sizeof(pc), bufs, 3, rows, 1, 1);
}

void VulkanBackend::moe_route_sigmoid(VBuf& logits, VBuf& idx, VBuf& score, VBuf& bias,
                                      int rows, int E, int K, int n_group, int topk_group,
                                      bool norm, float routed_scaling, int bias_off) {
    if (rows <= 0 || E <= 0 || K <= 0 || n_group <= 0 || topk_group <= 0) return;
    struct PC { int rows, E, K, n_group, topk_group, norm, bias_off; float routed_scaling; }
    pc{ rows, E, K, n_group, topk_group, norm ? 1 : 0, bias_off, routed_scaling };
    BufBind bufs[4] = { {0, logits.buffer, 0, 0}, {1, idx.buffer, 0, 0},
                         {2, score.buffer, 0, 0}, {3, bias.buffer, 0, 0} };
    dispatch("moe_route_sigmoid", &pc, sizeof(pc), bufs, 4, rows, 1, 1);
}

void VulkanBackend::moe_ffn(VBuf& x, VBuf& idx, VBuf& score, VBuf& gu, VBuf& dn,
                            VBuf& slot_of, VBuf& out, int rows, int H, int I, int K, bool bf16) {
    if (rows <= 0 || H <= 0 || I <= 0 || K <= 0) return;
    struct PC { int rows, H, I, K, bf16; } pc{ rows, H, I, K, bf16 ? 1 : 0 };
    BufBind bufs[7] = {
        {0, x.buffer, 0, 0}, {1, idx.buffer, 0, 0}, {2, score.buffer, 0, 0},
        {3, gu.buffer, 0, 0}, {4, dn.buffer, 0, 0}, {5, slot_of.buffer, 0, 0},
        {6, out.buffer, 0, 0}
    };
    // Size shared memory (ys[I], res[H]) to the model, not compile-time maxima.
    struct Spec { int H, I; } spec{ H, I };
    VkSpecializationMapEntry entries[2] = {{0, 0, sizeof(int)}, {1, sizeof(int), sizeof(int)}};
    char key[64];
    std::snprintf(key, sizeof(key), "moe_ffn#%dx%d", H, I);
    dispatch_spec("moe_ffn", key, entries, 2, &spec, sizeof(spec),
                  &pc, sizeof(pc), bufs, 7, rows, 1, 1);
}

void VulkanBackend::moe_route_ffn(VBuf& x, VBuf& logits, VBuf& idx, VBuf& score, VBuf& gu, VBuf& dn,
                                  VBuf& slot_of, VBuf& out, int r0, int rows, int H, int I, int K, int E,
                                  bool bf16, bool norm) {
    if (rows <= 0 || H <= 0 || I <= 0 || K <= 0 || E <= 0) return;
    struct PC { int r0, rows, H, I, K, E, bf16, norm; }
    pc{ r0, rows, H, I, K, E, bf16 ? 1 : 0, norm ? 1 : 0 };
    BufBind bufs[8] = {
        {0, x.buffer, 0, 0}, {1, logits.buffer, 0, 0}, {2, idx.buffer, 0, 0}, {3, score.buffer, 0, 0},
        {4, gu.buffer, 0, 0}, {5, dn.buffer, 0, 0}, {6, slot_of.buffer, 0, 0}, {7, out.buffer, 0, 0}
    };
    struct Spec { int H, I, K; } spec{ H, I, K };
    VkSpecializationMapEntry entries[3] = {{0, 0, sizeof(int)}, {1, sizeof(int), sizeof(int)},
                                           {2, 2 * sizeof(int), sizeof(int)}};
    char key[64];
    std::snprintf(key, sizeof(key), "moe_route_ffn#%dx%dx%d", H, I, K);
    dispatch_spec("moe_route_ffn", key, entries, 3, &spec, sizeof(spec),
                  &pc, sizeof(pc), bufs, 8, rows, 1, 1);
}

void VulkanBackend::add_buf(VBuf& a, VBuf& b, int n) {
    if (n <= 0) return;
    struct PC { int n; } pc{ n };
    BufBind bufs[2] = {{0, a.buffer, 0, 0}, {1, b.buffer, 0, 0}};
    dispatch("add_buf", &pc, sizeof(pc), bufs, 2, (uint32_t)((n + 255) / 256), 1, 1);
}
