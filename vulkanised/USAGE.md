# gguf2vk — Usage Guide

GGUF → Vulkan-native (tiled, block-32) converter and tester. Pure Python + numpy, no other dependencies, no llama.cpp/gguf-py.

## Files

| File | Role |
|---|---|
| `gguf_reader.py` | Parses GGUF binary format from scratch — metadata, tensor table, raw bytes. |
| `ggml_types.py` | Block size / byte size table per ggml quant type. Computes how many bytes each tensor occupies. |
| `compat_check.py` | Gate that checks a model's architecture (head_dim, GQA ratio, hidden_size) against the Vulkan-native spec before touching weights. |
| `dequant.py` | Converts quantized bytes → fp32 numpy arrays. See coverage table below. |
| `requant.py` | Re-quantizes fp32 → block-32 Q8_0 or Q4_0, chosen via `--format`. |
| `relayout.py` | Reorders 2-D weight matrices into GPU-tile-friendly order at the element level (used by q8_0/q4_0 modes). |
| `native_relayout.py` | Reorders 2-D weight matrices' existing quantized BLOCKS into tile order without decoding them — used by `--format native`. |
| `writer.py` | Writes the converted tensors out as a GGUF-shell file. |
| `convert.py` | Orchestrator — runs the full pipeline end to end. **This is the one you actually run.** |
| `test_gguf2vk.py` | Test suite — unit tests, round-trip tests, and integration tests against real files. |

All 9 files must stay in the same folder — they import each other.

## Install

```bash
pip install numpy
```

That's the only dependency.

## Quick start

```bash
# 1. Sanity-check the toolkit itself
python3 test_gguf2vk.py your-model.gguf

# 2. Check if a model's architecture fits the spec (metadata only, fast, no weight math)
python3 compat_check.py your-model.gguf

# 3. Convert it
python3 convert.py your-model.gguf output.gguf
```

## `convert.py` — the converter

```bash
python3 convert.py <input.gguf> <output.gguf> [options]
```

| Flag | Default | What it does |
|---|---|---|
| `--tile-rows N` | 32 | Rows per relayout tile |
| `--tile-cols N` | 8 | Columns per relayout tile |
| `--force` | off | Convert even if `compat_check` fails — for testing the pipeline on any model, spec-compliant or not |

**What happens on a run:**
1. Parses the file, runs the compat check, prints the report
2. If it fails and you didn't pass `--force`, stops there — no output file
3. Otherwise, dequantizes every tensor, relayouts 2-D weight matrices, requantizes to block-32, writes the result
4. Prints a per-tensor `OK` / `SKIPPED` / `ERROR` line, then a summary count

**Read the summary count.** `57/57 converted, 0 skipped` means a complete file. Anything less means the output is a partial conversion — check which tensors were skipped and why before using it.

## Quant format coverage

| Format | Status | Confidence |
|---|---|---|
| F32, F16 | ✅ implemented | Verified |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | ✅ implemented | Verified — unit-tested against hand-computed byte vectors |
| Q4_K, Q6_K, Q3_K, Q5_K | ✅ implemented | Best-effort — structurally sound and self-consistency-tested (diff-tested against a pre-vectorization oracle; Q3_K's scale packing checked via algebraic round-trip), math not cross-checked against an independent llama.cpp reference dump |
| Q2_K | ❌ not implemented | Refuses loudly (`NotImplementedError`) rather than guessing at the bit-packing |

**Practical upshot:** models quantized as Q4_0, Q5_0, Q5_1, Q8_0, or any mix of Q3_K/Q4_K/Q5_K/Q6_K (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, etc.) will convert fully — confirmed end to end on a real Q3_K_M 14B model (243/243 tensors, 0 skipped). Only Q2_K remains a genuine gap; models using it will convert norm/F32 tensors and skip Q2_K weight matrices, with the summary count showing it plainly.

## Output format choice

`convert.py` supports three modes, chosen with `--format`:

| Format | Bits/element | Quant error | Notes |
|---|---|---|---|
| `q8_0` (default) | 8.5 | Some (int8 requant) | Simplest shader unpack (one scale, direct int8 read). Roughly 2.1x the size of a typical mixed-K-quant source file. |
| `q4_0` | 4.5 | More (4-bit requant) | Close to original file size. Slightly more shader work (nibble unpack) than Q8_0. |
| `native` | Unchanged — whatever the source file used per tensor | **None** | No dequant/requant at all — reorders each tensor's existing quantized blocks into GPU tiles, keeping the original bit width and format exactly as shipped. Zero size penalty, zero added error. Works even on Q2_K, since nothing is decoded, only whole blocks moved. The tradeoff moves entirely to the shader: reading a tiled K-quant block still means implementing that format's native unpack math on the GPU side. |

```bash
python3 convert.py input.gguf output-q4.gguf --format q4_0
python3 convert.py input.gguf output-native.gguf --format native
```

`native` mode uses `--tile-block-cols` instead of `--tile-cols` (tile width in *blocks*, not elements — a block is 32 or 256 elements depending on the source format, so "tile width in elements" isn't a meaningful unit once you stop dequantizing). Default is 1.

See `RATIONALE.md` for the full size/complexity tradeoff discussion.

## `compat_check.py` — the spec gate

Run standalone to check a model without converting anything:

```bash
python3 compat_check.py your-model.gguf
```

Checks:
- `head_dim` (embedding_length ÷ head_count) is 64 or 128
- GQA ratio (head_count ÷ head_count_kv) is a power of 2
- `embedding_length` divides evenly by 256
- If MoE metadata is present, expert count is within the coarse-grained cap

`PASS` means safe to convert. `FAIL` lists the specific rule(s) that broke — that's an architecture mismatch, not something a format conversion can fix (would need finetuning/retraining into a compliant shape instead).

## `test_gguf2vk.py` — the test suite

```bash
python3 test_gguf2vk.py [model1.gguf model2.gguf ...]
```

No args → auto-picks up any `.gguf` files in `/mnt/user-data/uploads/`. With args → tests against exactly the files you name.

Four tiers, run every time:
1. Legacy dequant — hand-verified math, always runs regardless of files provided
2. Requant/relayout round-trip — synthetic data, always runs
3. K-quant structural check — only runs if a provided file actually contains Q4_K/Q6_K tensors; says so and skips (not fails) otherwise
4. Full integration — runs `convert.py --force` against every provided file end to end, confirms it doesn't crash and produces a non-empty output

Exit code 0 = everything passed, 1 = something failed. Safe to wire into a CI check.

## Reading `writer.py` output

**The output file is not a drop-in replacement for the input.** It reuses GGUF's binary shell (same header/metadata/tensor-table format) purely because it's a convenient, already-defined container — but tensor data inside is tile-relayouted and requantized, so stock llama.cpp or any standard GGUF loader will read it and produce garbage. It's meant to be loaded by a Vulkan-native reader that knows the `vulkan_native.*` metadata tags (`vulkan_native.layout`, `vulkan_native.tile_rows`, `vulkan_native.tile_cols`) and undoes the tiling the same way `relayout.from_tiled()` does.

## Common issues

**"FAIL — head_dim = X, spec requires one of (64, 128)"**
Architecture mismatch, not a bug. Model wasn't shaped for GPU-tile-friendly dims. Use `--force` if you just want to test the pipeline mechanically, or pick a different source model for a real conversion.

**Lots of `SKIPPED` lines**
Model uses Q2_K somewhere — the only genuinely unimplemented format. Check the skip messages to confirm. Re-quantize the source model with a different quant type if you control that step, or accept a partial conversion.

**Old behavior after updating a file**
Classic stale-copy issue — confirm the file you're running actually has the change (`grep` for a distinctive string from the fix) before assuming the fix didn't work.
