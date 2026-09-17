// Raw device-memory bandwidth probe. Allocates a large buffer, dispatches
// bw_probe with coalesced uint32 reads, and reports effective GB/s. This is the
// ceiling any weight-streaming kernel can hope to approach.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "nanovllm/vulkan_backend.hpp"

int main() {
    auto dev = VulkanBackend::Create();
    const size_t MB = 1u << 20;
    const size_t bytes = 256 * MB;          // 256MiB, >> LLC
    const uint32_t nwords = (uint32_t)(bytes / 4);
    const uint32_t lanes = 64 * 512;        // 512 workgroups x 64 lanes = 32768
    const uint32_t passes = 4;              // amortize launch
    VBuf src; src.alloc(bytes);
    { std::vector<uint32_t> h(nwords, 0x12345678u); src.upload(h.data(), bytes); }
    VBuf dst; dst.alloc(4 * 512);

    struct PC { uint32_t nwords; uint32_t stride; uint32_t passes; } pc{ nwords, lanes, passes };
    VulkanBackend::BufBind bufs[2] = { {0, src.buffer, 0, 0}, {1, dst.buffer, 0, 0} };

    // warmup
    dev->dispatch("bw_probe", &pc, sizeof(pc), bufs, 2, 512, 1, 1);
    dev->submit_wait();

    auto t0 = std::chrono::steady_clock::now();
    dev->dispatch("bw_probe", &pc, sizeof(pc), bufs, 2, 512, 1, 1);
    dev->submit_wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gbs = (double)bytes * passes / (ms / 1000.0) / 1e9;
    std::printf("bw_probe: %zu MiB x %u passes, %u lanes in %.2f ms -> %.2f GB/s\n",
                bytes / MB, passes, lanes, ms, gbs);
    return 0;
}
