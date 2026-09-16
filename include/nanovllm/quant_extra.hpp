#pragma once

#include <cstdint>
#include <vector>
#include <cstring>

namespace quant {

// Host dequantizers for quant formats with no GPU kernel yet. These return
// fp32 vectors the loader can upload. No GPU code here. The block layouts
// below are the documented wire formats; the fp8/fp16 scale decoding is a
// simple fixed-point approximation (good enough for host upcast), not a
// bit-exact IEEE decode — the GPU kernel path is separate.

// MXFP4 (GPT-OSS style): blocks of 32 values. Each block: 1 byte E8M0-ish
// scale + 16 bytes of packed 4-bit nibbles (2 per byte).
//   value = (nibble - 7) * scale,  scale = scale_u8 * 2^-2
inline size_t block_bytes_mxfp4() { return 17; }

inline bool dequant_mxfp4(const uint8_t* raw, size_t n_values,
                          std::vector<float>& out) {
    if (n_values % 32 != 0 || raw == nullptr) return false;
    size_t n_blocks = n_values / 32;
    out.resize(n_values);
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t* block = raw + b * block_bytes_mxfp4();
        float scale = static_cast<float>(block[0]) * 0.25f;  // * 2^-2
        const uint8_t* nib = block + 1;
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = nib[i];
            int lo = byte & 0x0F;
            int hi = (byte >> 4) & 0x0F;
            out[b * 32 + i * 2 + 0] = static_cast<float>(lo - 7) * scale;
            out[b * 32 + i * 2 + 1] = static_cast<float>(hi - 7) * scale;
        }
    }
    return true;
}

// AWQ: 19-byte blocks — [scale:2 fp16][zero:1 int8][q:16 bytes, 32 int4].
//   value = (q - zero) * scale
struct AwqBlock { uint16_t scale; uint8_t zero; uint8_t q[16]; };

inline size_t block_bytes_awq() { return 19; }

inline bool dequant_awq(const uint8_t* raw, size_t n_groups,
                        std::vector<float>& out) {
    if (raw == nullptr) return false;
    out.resize(n_groups * 32);
    for (size_t g = 0; g < n_groups; ++g) {
        const uint8_t* blk = raw + g * block_bytes_awq();
        uint16_t scale_u16;
        std::memcpy(&scale_u16, blk, 2);
        float scale = static_cast<float>(scale_u16) / 256.0f;  // fp16 approx
        uint8_t z = blk[2];
        const uint8_t* q = blk + 3;
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = q[i];
            int lo = byte & 0x0F;
            int hi = (byte >> 4) & 0x0F;
            out[g * 32 + i * 2 + 0] = static_cast<float>(lo - z) * scale;
            out[g * 32 + i * 2 + 1] = static_cast<float>(hi - z) * scale;
        }
    }
    return true;
}

// GPTQ: 19-byte blocks — [q:16 bytes, 32 int4][scale:2 fp16][zero:1 int4 nibble].
//   value = (q - zero_nibble) * scale
inline size_t block_bytes_gptq() { return 19; }

inline bool dequant_gptq(const uint8_t* raw, size_t n_groups,
                         std::vector<float>& out) {
    if (raw == nullptr) return false;
    out.resize(n_groups * 32);
    for (size_t g = 0; g < n_groups; ++g) {
        const uint8_t* blk = raw + g * block_bytes_gptq();
        const uint8_t* q = blk;
        uint16_t scale_u16;
        std::memcpy(&scale_u16, blk + 16, 2);
        float scale = static_cast<float>(scale_u16) / 256.0f;
        uint8_t z_byte = blk[18];
        int z_lo = z_byte & 0x0F;
        int z_hi = (z_byte >> 4) & 0x0F;
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = q[i];
            int lo = byte & 0x0F;
            int hi = (byte >> 4) & 0x0F;
            out[g * 32 + i * 2 + 0] = static_cast<float>(lo - z_lo) * scale;
            out[g * 32 + i * 2 + 1] = static_cast<float>(hi - z_hi) * scale;
        }
    }
    return true;
}

}  // namespace quant

#ifdef QUANT_TEST
#include <cassert>
#include <cstdio>

int main() {
    assert(quant::block_bytes_mxfp4() == 17);
    assert(quant::block_bytes_awq() == 19);
    assert(quant::block_bytes_gptq() == 19);

    // dequant_mxfp4 of one block: scale_u8=4 -> scale=1.0, all-zero nibbles
    // -> 32 values all == -7.0
    uint8_t block[17] = {0};
    block[0] = 4;
    std::vector<float> out;
    assert(quant::dequant_mxfp4(block, 32, out));
    assert(out.size() == 32);
    for (int i = 0; i < 32; ++i) assert(out[i] == -7.0f);

    // AWQ: one group, scale=256 -> scale=1.0, zero=0, all-zero q -> 32 zeros
    uint8_t awq[19] = {0};
    awq[0] = 1; awq[1] = 1;  // scale_u16 = 256
    std::vector<float> aout;
    assert(quant::dequant_awq(awq, 1, aout));
    assert(aout.size() == 32);
    for (int i = 0; i < 32; ++i) assert(aout[i] == 0.0f);

    std::printf("All quant tests passed.\n");
    return 0;
}
#endif