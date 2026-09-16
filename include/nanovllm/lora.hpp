#pragma once

#include <string>
#include <vector>
#include <cmath>

#include "nanovllm/gguf.hpp"

namespace lora {

// A LoRA delta is two low-rank matrices A (r x in) and B (out x r) applied as
//   W' = W + (alpha / r) * (B @ A)
// Edge0 defines the contract: lora_A/B are the two tensors, alpha the scalar.
struct Adapter {
    std::string name;
    double alpha = 0.0;
    int r = 0;
    std::vector<float> A;  // row-major, r rows x in cols  (A[row*in + col])
    std::vector<float> B;  // row-major, out rows x r cols (B[out*r + col])
    int in_dim = 0, out_dim = 0;
};

// Load a GGUF file's "lora.*" tensors into an Adapter. Uses the project's own
// GGUFLoader (no hand-rolled parser). Returns an Adapter with r>0 on success,
// empty Adapter (r==0) on any failure — never throws.
inline Adapter load_from_gguf(const std::string& path) {
    Adapter ad;
    try {
        GGUFLoader g;
        g.add_file(path);

        const GGUFTensorMeta* ma = g.tensor("lora.A");
        const GGUFTensorMeta* mb = g.tensor("lora.B");
        if (!ma || !mb) return ad;
        if (ma->shape.size() != 2 || mb->shape.size() != 2) return ad;

        ad.r = static_cast<int>(ma->shape[0]);
        ad.in_dim = static_cast<int>(ma->shape[1]);
        ad.out_dim = static_cast<int>(mb->shape[0]);
        int r_b = static_cast<int>(mb->shape[1]);
        if (ad.r <= 0 || ad.in_dim <= 0 || ad.out_dim <= 0 || r_b != ad.r) return ad;

        ad.A.resize(static_cast<size_t>(ad.r) * ad.in_dim);
        ad.B.resize(static_cast<size_t>(ad.out_dim) * ad.r);
        bool ba = g.load_float("lora.A", ad.A);
        bool bb = g.load_float("lora.B", ad.B);
        if (!ba || !bb) return ad;

        double alpha = 0.0;
        if (g.meta_f64("lora.alpha", alpha)) {
            ad.alpha = alpha;
        } else if (const GGUFMetaValue* mv = g.meta("lora.alpha")) {
            ad.alpha = mv->d;  // fall back to whatever scalar type was stored
        } else {
            return ad;  // alpha required
        }
        return ad;
    } catch (...) {
        return ad;  // r == 0 on any failure — never throws
    }
}

// Apply the adapter to a weight buffer in place: W'[i*out+j] += (alpha/r) *
// sum_k B[i*r+k] * A[k*in+j]. out/in must match adapter dims; if not, no-op.
inline void apply(const Adapter& ad, float* W, int out, int in) {
    if (ad.r <= 0 || ad.in_dim <= 0 || ad.out_dim <= 0) return;
    if (out != ad.out_dim || in != ad.in_dim) return;
    if (W == nullptr) return;
    if (ad.A.empty() || ad.B.empty()) return;

    double scale = ad.alpha / static_cast<double>(ad.r);
    for (int i = 0; i < ad.out_dim; ++i) {
        for (int j = 0; j < ad.in_dim; ++j) {
            double sum = 0.0;
            for (int k = 0; k < ad.r; ++k) {
                sum += static_cast<double>(ad.B[static_cast<size_t>(i) * ad.r + k]) *
                       static_cast<double>(ad.A[static_cast<size_t>(k) * ad.in_dim + j]);
            }
            W[static_cast<size_t>(i) * in + j] += static_cast<float>(scale * sum);
        }
    }
}

}  // namespace lora

#ifdef LORA_TEST
#include <cassert>
#include <cstdio>
#include <fstream>

int main() {
    // Test 1: load_from_gguf on a nonexistent file returns r==0
    {
        lora::Adapter ad = lora::load_from_gguf("nonexistent_file.gguf");
        assert(ad.r == 0);
    }

    // Test 2: apply() with a hand-built 2x2 adapter (r=1, alpha=1) adds exactly B@A to W
    {
        lora::Adapter ad;
        ad.name = "test";
        ad.alpha = 1.0;
        ad.r = 1;
        ad.in_dim = 2;
        ad.out_dim = 2;
        ad.A = {1.0f, 2.0f};  // 1x2
        ad.B = {3.0f, 4.0f};  // 2x1

        float W[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        lora::apply(ad, W, 2, 2);

        // B @ A = [[3*1, 3*2], [4*1, 4*2]] = [[3, 6], [4, 8]]
        assert(W[0] == 3.0f);
        assert(W[1] == 6.0f);
        assert(W[2] == 4.0f);
        assert(W[3] == 8.0f);
    }

    // Test 3: dimension mismatch (out/in != adapter) leaves W unchanged
    {
        lora::Adapter ad;
        ad.name = "test";
        ad.alpha = 1.0;
        ad.r = 1;
        ad.in_dim = 2;
        ad.out_dim = 2;
        ad.A = {1.0f, 2.0f};
        ad.B = {3.0f, 4.0f};

        float W[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float W_orig[6];
        std::memcpy(W_orig, W, sizeof(W));
        lora::apply(ad, W, 3, 2);  // wrong dims
        for (int i = 0; i < 6; ++i) assert(W[i] == W_orig[i]);
    }

    std::printf("All lora tests passed.\n");
    return 0;
}
#endif