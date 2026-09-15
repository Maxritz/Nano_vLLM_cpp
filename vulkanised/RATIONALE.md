# Why gguf2vk exists, and what it takes to actually run its output

## The original motivation

MLX (Apple's ML framework) gets a real performance edge on Apple Silicon from
one specific thing: CPU and GPU share the same physical memory, so there's no
copy cost moving data between them. That's a hardware/memory-topology trick,
not a model architecture trick — any model runs faster under MLX on Apple
Silicon regardless of its shape.

Vulkan compute doesn't have that same free win on discrete GPUs (there's still
a PCIe hop between CPU and VRAM on hardware like the RDNA2/RDNA4 cards this
project targets). But Vulkan does have a different, real lever: **model
architecture choices that are deliberately shaped around GPU tiling and
subgroup width**, rather than shaped by whatever a training run happened to
pick. That's what this project chases — not copying MLX's specific trick, but
applying the same underlying idea (co-design the data layout with the
hardware's actual execution model) to the piece of the stack Vulkan can
influence: **how weight tensors are laid out in memory**.

Motivation for choosing Vulkan specifically, rather than staying on
ROCm/HIP or CUDA: portability. A model converted this way should, in
principle, run near-optimally on AMD, Nvidia, and Intel Arc GPUs without
separate vendor-tuned forks — Vulkan compute is the one API all three support
natively. That only holds if the tile shape and shader design stay grounded
in things that are true across vendors (subgroup width of 32 is common,
though not universal — see the caveat in the integration spec).

## What "Vulkan-native" means concretely here

Two independent pieces, each solving a different problem:

1. **Architecture-level spec** (`vulkan-native-moe-spec.md`, produced earlier
   in this project): head_dim, GQA ratio, hidden_size divisibility rules
   chosen so attention/FFN shapes tile cleanly against a subgroup-32 GPU.
   This is a constraint on *which models* are good candidates — it can't be
   fixed by conversion, only satisfied by the model's original training
   shape. `compat_check.py` enforces this.

2. **Tensor relayout** (`relayout.py` + the rest of the converter): given a
   model that already satisfies #1, reorder each 2-D weight tensor's bytes
   from plain row-major into fixed-size tiles, so a GPU compute shader can
   read a tile with coalesced, subgroup-aligned memory access instead of
   striding across the full matrix. This is what the converter actually does
   — a mechanical, automatable transformation, unlike #1.

## The size tradeoff (and where it comes from)

The converter also requantizes every tensor to a single block-32 format
(`Q8_0` or `Q4_0`, your choice via `--format`), because block-32 is the size
this project's tiling scheme is aligned to (32 = common GPU subgroup width).

This has a real, non-negligible cost: source models quantized with
llama.cpp's mixed K-quant schemes (Q3_K_M, Q4_K_M, etc.) average somewhere
around 3.5–4.5 bits/element by mixing several different block formats per
tensor based on which layers benefit most from precision. Q8_0 is 8.5
bits/element flat, roughly doubling file size. Q4_0 (added later, via
`--format q4_0`) gets close to the original average again, at some added
quantization error and slightly more shader unpack work.

**The real tradeoff underneath the size number:** K-quant's per-tensor mixed
precision is a genuinely good idea for size/quality — it's just also the
thing that makes the bit-packing so gnarly (packed 6-bit sub-block scales,
different schemes per format) that writing a correct, simple GPU shader
for it is real, nontrivial work. Q8_0/Q4_0 trade some of that size efficiency
for a shader that's honestly simple to get right. That trade should be made
consciously, not by default — hence the format choice existing at all,
and this document naming the tradeoff explicitly rather than only mentioning
the tiling benefit.

**Native mode keeps the original size** (`--format native`, built after this
was first documented as a future path): relayout the *original* K-quant (or
legacy) blocks into tile order without changing their bit width — tiling at
block granularity (32 or 256 elements/block, whatever the source format
uses) rather than after flattening to individual fp32 elements. This avoids
the size penalty and adds zero quantization error, since nothing is decoded
or re-encoded — only whole blocks move. It even handles Q2_K, which this
project's `dequant.py` explicitly refuses to decode, because native mode
never needs to interpret a block's contents, only know its byte size.

The cost moves entirely to the shader side: reading a tiled K-quant block on
the GPU still means implementing that format's native unpack math (the same
packed-scale complexity `dequant.py` went through on the Python side) inside
the compute shader, rather than the trivial Q8_0 case. Whether that tradeoff
is worth it depends on whether you value file size or shader simplicity more
for a given deployment — both modes exist so that's a deliberate choice, not
a default.

## What it takes for a runtime to actually run the output

The converter's output file is **not** loadable by a stock GGUF reader in any
meaningful way, even though it reuses GGUF's binary shell and standard Q8_0/
Q4_0 tensor types. A stock reader (llama.cpp, gguf-py, anything else) would
successfully parse the header and tensor table, then read each tensor's bytes
in the wrong order — silently wrong output, not a crash, since nothing about
the file format itself is invalid.

The `vulkan_native.*` metadata keys exist specifically to mark this: any
reader that doesn't recognize them and proceeds anyway is reading a file it
doesn't actually understand.

**To run this file correctly, a runtime needs:**
1. To detect `vulkan_native.layout == "tiled"` and read the tile shape +
   format from metadata (never hardcode 32×8 — that's this converter's
   default, not a fixed constant).
2. A matmul/GEMM implementation whose *weight-reading* path knows how to
   translate a logical (row, col) index into the tiled byte layout — see
   `LLAMA_CPP_INTEGRATION_SPEC.md` for the exact index math and where this
   would plug into llama.cpp specifically.
3. Standard Q8_0/Q4_0 dequant logic once the tiled offset is resolved — the
   quantization format itself is unmodified, only the element ordering is
   nonstandard.

**Where this actually belongs long-term:** VAiSt (this project's own
from-scratch Vulkan compute stack) is the natural home, since it's already
Vulkan-only and under active development here — no upstream review process,
no need to convince llama.cpp maintainers a new op and shader are worth
merging. The llama.cpp integration spec exists because it was asked for
explicitly and is a legitimate path to a much larger existing user base, but
it's real, multi-week engineering (a new ggml op, new shaders, backend
wiring, validation against the existing `mul_mat` path) — not a converter
tweak. Treat "convert a file" and "get an engine to execute the tiled
layout" as two separate projects with two separate timelines.

## Honest confidence status of the pieces involved

- **Architecture spec (head_dim/GQA/hidden_size rules):** design choices,
  not something to "verify" — they're constraints this project chose, backed
  by reasoning about subgroup alignment, not external ground truth.
- **GGUF parsing, legacy quant dequant (F32/F16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0):**
  unit-tested against hand-computed byte vectors. Trustworthy.
- **K-quant dequant (Q4_K/Q6_K/Q3_K/Q5_K):** implemented from public ggml
  struct layouts, internally self-consistent (diff-tested against a
  pre-vectorization oracle, Q3_K's packing checked via algebraic round-trip),
  but **not cross-checked against an independent llama.cpp reference dump.**
  Treat converted weights from these formats as provisional.
- **Q2_K:** not implemented at all — refuses loudly rather than guessing.
- **Requant (Q8_0/Q4_0 encode), relayout (element-level tiling for q8_0/q4_0):**
  vectorized, diff-tested against original loop-based versions, round-trip-verified
  for losslessness (relayout) and bounded error (requant, which is lossy by design).
- **Native block relayout (`--format native`):** byte-exact round-trip verified
  (no quantization error is possible in this path — it never decodes anything).
  Zero-padding safety (an all-zero padding block decodes to exactly zero) is
  confirmed for every implemented dequant format and reasoned to hold for Q2_K
  by the same scale-zeroing pattern, though Q2_K's specific layout hasn't been
  checked directly since it's never decoded.
- **llama.cpp integration:** not built. The spec above is a plan, not a
  progress report.
