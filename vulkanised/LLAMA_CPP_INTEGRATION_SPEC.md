# llama.cpp Integration Spec: Vulkan-Native Tiled Tensor Support

**Audience:** an AI coding agent (or engineer) working directly in the llama.cpp
source tree. This spec describes what to build, where, and how to validate it.
It does not include llama.cpp source snippets verbatim (file layouts and exact
signatures shift between versions) — start each step by reading the current
version of the referenced file before writing code against it.

**Prerequisite reading for the agent:** `RATIONALE.md` in this same folder —
explains *why* this exists (Vulkan portability motivation, tiling scheme, size
tradeoffs) before diving into *how*.

---

## 0. What already exists (from the gguf2vk converter)

- A converter (`convert.py` + friends) that reads a standard GGUF file and
  produces a **GGUF-shell** output file: same outer binary format, but 2-D
  weight tensors have been reordered into fixed-size tiles instead of plain
  row-major order.
- **Two distinct tiling modes** — the shader side needs to handle both, since
  the file's own metadata says which one was used and real-world usage so
  far has favored the second:
  - **Element-tiled (`--format q8_0` or `q4_0`):** tensor is dequantized,
    tiled at the individual-element level, then requantized to a uniform
    Q8_0 or Q4_0 format. Tile shape is in **elements** (`tile_rows` ×
    `tile_cols`). Simple shader unpack (Q8_0: one scale + direct int8 read;
    Q4_0: one scale + nibble split), but real size/quality cost (~2.1x size
    for Q8_0, some added quantization error for either).
  - **Block-tiled (`--format native`):** NO dequant/requant — the tensor
    keeps its original per-tensor quant format (Q3_K, Q4_K, Q5_K, Q6_K,
    Q4_0, Q5_0, Q8_0, whatever the source file used), and only whole
    **quantization blocks** get reordered into tiles. Tile shape is in
    **blocks** (`tile_rows` × `tile_block_cols`), not elements. Zero size
    penalty, zero added quantization error — but the shader must implement
    that tensor's *native* unpack math (the same packed-scale bit
    manipulation `dequant.py` had to work through on the Python side) rather
    than a single simple Q8_0/Q4_0 case.
  - **In practice, native mode is the one that's actually been used and
    validated on a real multi-GB model** (a 40-layer Q3_K_M model, verified
    byte-exact against the original via `verify_native.py`) — treat it as
    the primary target, and element-tiled Q8_0/Q4_0 as the simpler
    bootstrap case to get the tiling *mechanics* right before tackling
    K-quant's unpack complexity in GLSL.
- The output file carries these metadata keys so any reader can detect the
  tiling and its parameters — **check `vulkan_native.format` first to know
  which of the two tiling schemes below applies**:
  - `vulkan_native.layout` (string) — currently always `"tiled"`
  - `vulkan_native.tile_rows` (uint32) — rows per tile, both modes
  - `vulkan_native.tile_cols` (uint32) — **element-tiled modes only**
    (elements per tile column)
  - `vulkan_native.tile_block_cols` (uint32) — **native mode only**
    (quantization blocks per tile column)
  - `vulkan_native.format` (string) — `"q8_0"`, `"q4_0"`, or `"native"`
  - `vulkan_native.source` (string) — original filename, informational only
- Tile layout convention (must match exactly): matrix padded to a multiple
  of tile shape, tiles emitted in row-major tile order, each tile internally
  row-major. Identical convention for both modes — only the *unit* being
  tiled differs (individual elements vs. whole quantization blocks). See
  `relayout.py` (element-level) and `native_relayout.py` (block-level).
- **`verify_native.py`** exists specifically to confirm a native-mode output
  file untiles back to byte-identical original tensor data — run this
  against any test file used for validation in step 6 below, as a sanity
  check independent of whatever the shader eventually produces.

**This spec is entirely about the read/execute side — nothing here needs to
touch the converter.**

---

## 1. Minimum viable goal

Make llama.cpp's Vulkan backend able to load a `vulkan_native`-tagged GGUF
file (either tiling mode) and run inference on it correctly, using a matmul
kernel that reads the tiled layout directly (no untiling pass on the CPU at
load time — that would throw away the entire performance point of tiling).

**Suggested build order, given native mode is the real target:** get the
element-tiled Q8_0 case working first (simplest possible shader — proves the
tiling index math and backend wiring work at all), then extend to native
mode's block-tiled Q4_K/Q6_K/Q3_K/Q5_K cases one format at a time, using
`dequant.py`'s Python implementations as the reference for each format's bit
layout.

**Explicitly out of scope for v1:** MoE routing, non-2D tensor tiling, tile
shapes other than what's declared in the file's metadata.

---

## 2. Model-loading side

**Files to look at:** `src/llama-model-loader.cpp`, `src/llama-model.cpp` (or
current equivalent paths — these move between versions).

**What to do:**
1. When loading GGUF metadata, check for `vulkan_native.layout`. If present
   and equal to `"tiled"`, read `vulkan_native.tile_rows`/`tile_cols` and
   `vulkan_native.format`.
2. Tag every 2-D tensor in the model with this tiling info (a flag + the two
   tile dimensions) so later graph-building code knows to route it
   differently. How this tagging is threaded through depends on llama.cpp's
   current tensor metadata structures — find wherever per-tensor flags
   already live (e.g. quantization type is already tracked per-tensor; tile
   info is a sibling piece of metadata) and extend it there rather than
   inventing a parallel side-table.
3. **Reject at load time, not at inference time**, if `vulkan_native.layout`
   is present but the active backend isn't Vulkan — this format only makes
   sense for the Vulkan path (see §5).

---

## 3. New ggml operation for tiled matmul

**Files to look at:** `ggml/include/ggml.h` (op enum + op-specific params
struct), `ggml/src/ggml.c` (op registration, shape inference).

**What to do:**
1. Add a new op — e.g. `GGML_OP_MUL_MAT_TILED` — distinct from the existing
   `GGML_OP_MUL_MAT`. Don't try to overload the existing op with a hidden
   flag; a distinct op keeps backend dispatch and shape-inference logic
   clean and auditable.
2. Op parameters need to carry `tile_rows`, `tile_cols`, and the padded
   (post-tiling) shape of the tensor, since the logical shape and the
   physical (padded-to-tile) shape differ whenever the matrix dims aren't
   multiples of the tile shape.
3. Shape-inference function: output shape is the same as standard
   `mul_mat` (this op only changes how the weight operand is read, not the
   math it computes) — logical shape in, logical shape out. The tiling is
   an implementation detail of *reading* the weight tensor, invisible to the
   rest of the graph.

---

## 4. Vulkan shader work (the actual core of this project)

**Files to look at:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp` and the shader
source directory alongside it (GLSL sources compiled to SPIR-V at build
time — check the current build script for how existing shaders get compiled
and registered).

**What to do:**
1. Write a new compute shader (GLSL) that performs matrix-vector or
   matrix-matrix multiplication where the **weight matrix operand** is read
   from the tiled layout instead of row-major. The **activation operand**
   stays in normal row-major layout — only the pre-converted weight tensors
   are tiled.
2. **Tile index math differs by mode** — branch on `vulkan_native.format`:
   - *Element-tiled (q8_0/q4_0):* given logical (row, col), `tile_row = row
     / tile_rows`, `tile_col = col / tile_cols`, offset within the tiled
     stream follows directly from there (element granularity).
   - *Block-tiled (native):* given logical (row, col), first find which
     **block** that column falls in (`block_col = col / block_size`, where
     `block_size` is 32 for legacy formats or 256 for K-quants — read from
     the tensor's own ggml type, not hardcoded), then tile-index at the
     *block* level: `tile_row = row / tile_rows`, `tile_block_col =
     block_col / tile_block_cols`. The element itself lives at a fixed
     offset within its block (`col % block_size`), same as it would in a
     non-tiled version of that same quant format — tiling only changes
     which block-shaped chunk of bytes to seek to, not how to read within it.
3. **Dequant logic once the right block's bytes are located:**
   - `q8_0`/`q4_0` (element-tiled path): trivial — one scale, direct
     int8 read (Q8_0) or nibble split (Q4_0). Good first target to prove the
     tiling mechanics work before tackling K-quants.
   - `q3_k`/`q4_k`/`q5_k`/`q6_k` (native path — the one real files actually
     use): port the bit-unpacking directly from `dequant.py`'s numpy
     implementations of each format. These aren't simple — packed 6-bit
     sub-block scales spread across 12 bytes per super-block, in some cases
     (Q3_K) with an additional cross-byte bit-shuffling scheme. Treat each
     format as its own real porting task, not a variation on a theme; the
     Python implementations are the tested, working reference for the exact
     bit layout — port the arithmetic faithfully rather than re-deriving it
     from ggml source from scratch, since that's exactly the process that
     took real iteration (see `dequant.py`'s confidence notes) to get right
     even in Python.
   - `q2_k`: no reference exists anywhere in this project — `dequant.py`
     explicitly never implemented it. If a file needing Q2_K support shows
     up, that bit-packing scheme needs to be worked out from scratch (ggml's
     `block_q2_K` struct layout, similar in spirit to Q3_K but not
     implemented here) before a shader can be written for it — flag this
     rather than guessing at the layout, same reasoning `dequant.py` used to
     refuse it originally.
4. Reuse the workgroup/subgroup tiling parameters that already exist in
   llama.cpp's current `mul_mat` shaders where possible (local size,
   coop-matrix usage if the target GPU supports it) — the point of this
   project's tile shape (32 rows matching subgroup width) is specifically to
   make this reuse natural, not to invent a parallel shader architecture from
   scratch.

---

## 5. Backend wiring

**File:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

1. Register the new shader (pipeline creation, following the existing
   pattern for other ops in this file — same push-constant setup, same
   descriptor-set binding pattern already used for `mul_mat`).
2. Add `GGML_OP_MUL_MAT_TILED` to whatever dispatch switch/table routes ops
   to their Vulkan implementation.
3. **CPU fallback:** every ggml op needs to work across backends, or be
   explicitly gated to Vulkan-only with a clear error on other backends
   (check how existing Vulkan-only extensions in this codebase handle that —
   there's precedent, follow it rather than inventing new gating logic).
   Given this format's entire purpose is Vulkan-specific memory access
   patterns, an explicit "Vulkan only" error is more honest than writing a
   slow, pointless CPU implementation.

---

## 6. Validation (do this before trusting any of the above)

1. **Start with `verify_native.py`.** Before writing any GLSL, run
   `python3 verify_native.py original.gguf converted.gguf` on a native-mode
   conversion — confirms the converter's own output is byte-exact against
   the source, independent of anything about llama.cpp. If this fails, the
   bug is in the converter, not the shader you're about to write; fix it
   there first.
2. **Golden comparison test.** Take a small model, convert it with
   `gguf2vk`'s converter, run inference on both the original (standard
   `mul_mat`) and tiled (`GGML_OP_MUL_MAT_TILED`) versions with identical
   input, and diff the logits.
   - **Element-tiled (q8_0/q4_0):** expect a small but nonzero difference —
     same order of magnitude as comparing two different standard quant
     formats of the same model, since requantization adds its own error.
   - **Native mode:** expect logits to match much more closely, since the
     only source of numerical difference is floating-point operation
     *order* (tiled traversal vs. row-major traversal can sum in a
     different sequence) — not quantization error, because none was added.
     A native-mode mismatch bigger than floating-point-order noise points
     at a real shader bug, not expected quantization behavior.
3. **Isolate the two error sources.** Before comparing against the original
   model, first run `verify_native.py` (native mode) or manually reverse
   `relayout.from_tiled` + dequant (element-tiled mode) to confirm the
   *file* is correct — this isolates "did the converter work" from "did the
   shader read the tiles correctly." If the Python-side check passes but
   the shader output doesn't match, the bug is in the shader's index math
   or dequant, not in the converter.
4. **Shape edge cases.** Test a model where at least one weight matrix's
   dimensions aren't multiples of `tile_rows`/`tile_cols` (or
   `tile_block_cols` for native mode), to exercise the zero-padding path in
   both the converter and the shader's bounds handling.

---

## 7. Known gaps to flag back to the user, not silently work around

- If `dequant.py`/`ggml_types.py` in the converter show a quant type as
  `not_implemented` or `best_effort` for the source model being converted,
  that's a converter-side concern (see `RATIONALE.md`) — not something to
  patch around from the llama.cpp side.
- If the target GPU's subgroup size isn't 32 (some non-AMD/Nvidia hardware),
  the tile shape's alignment assumptions may not hold as cleanly — flag this
  rather than assuming the 32×8 default is universally optimal; the tile
  shape is a converter-time parameter (`--tile-rows`/`--tile-cols`), not
  fixed by this spec.
