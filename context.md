# Context / Handoff

Last updated: 2026-09-18 (commit f85099c). Owner goal: close the Vulkan decode gap
to llama.cpp on an Intel iGPU laptop.

## Where we are

| kernel | decode t/s (qwen2.5-0.5b-q2_k) |
|---|---|
| baseline tiled 16x16 | 7.6 |
| scalar per-column GEMV | 11.9 |
| **cooperative tiled GEMV (committed)** | **12.4** |
| llama.cpp Vulkan, same GPU | 53.4 |

The gap is kernel quality, not hardware. llama.cpp Vulkan reaches 53.4 t/s on
this exact iGPU with **no cooperative matrix** (vulkaninfo confirmed absent) —
subgroup ops + coalescing + packed loads only.

## Ground truth (build llama.cpp on this laptop)

Builds at `C:/Users/rina0423/Desktop/Ultrallama/` (a fork; also has ispc,
The-Forge, llm20_engine, control-plane dirs):

- `build_cpu/bin/Release/llama-bench.exe` — CPU: 23.6 t/s tg8, 105.9 t/s pp4
- `build_vk/bin/Release/llama-bench.exe` — Vulkan: **53.4 t/s tg8**, 162.6 t/s pp4

```
llama-bench.exe -m C:/models/qwen2.5-0.5b-instruct-q2_k.gguf -p 4 -n 8 -dev vulkan0
```
Note: `-b vulkan` does NOT work; it's `-dev vulkan0`.

Reference kernels to study:
`C:/Users/rina0423/Desktop/Ultrallama/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q2_k.comp`
and `mul_mat_vec_base.glsl`.

## Hardware

`Get-CimInstance Win32_VideoController` → only "Intel(R) Graphics", driver
32.0.101.8724, AdapterRAM 2147479552, deviceType INTEGRATED_GPU,
apiVersion 1.4.344. **No AMD dGPU on this box** —
`C:\Users\rina0423\Downloads\vulkan-amd-rdna4-perf-fix.patch` (RX 9070 XT)
is for a different machine.

Present: subgroupQuadOperationsInAllStages, shaderFloat16, VK_KHR_8bit_storage,
VK_KHR_16bit_storage, VK_KHR_shader_subgroup_extended_types,
VK_EXT_subgroup_size_control, shaderSubgroupFloat16Atomics=false.
**Absent:** VK_KHR_cooperative_matrix.

## CRITICAL: the bandwidth-ceiling mistake

Earlier reasoning: "24.5 GB/s UMA ceiling caps a 2B model at ~20 t/s" — **WRONG**.
llama.cpp Vulkan sits at ~21 GB/s effective (53.4 t/s x 390 MiB = ~85% of that
ceiling) while my kernels extracted only ~16% (~4 GB/s). The bandwidth IS there;
my kernels just don't capture it. **Do not rule out a target with ceiling math
without cross-checking against llama.cpp.**

## CRITICAL: measurement methodology

Sequential 5-run batches are **thermally confounded** on this laptop — medians
drifted 10.35 → 9.83 → 7.62 → 6.82 with ZERO code change. Every perf claim must
use interleaved A/B/A/B/A over 5–10 rounds.

Two earlier conclusions ("packed loads are 26% slower", "GEMV is a wash") were
thermal artifacts, disproved by interleaving.

Saved per-config binaries in `build_vk/Release/` enable interleaving:
`nano_scalar.exe` (16x4 scalar), `nano_packed.exe` (16x4 packed),
`nano_32x8.exe`, `nano_16x16.exe`, `nano_tiled.exe` (cooperative tiled).

## Q8 decode GEMV (the win this session)

`shaders/vulkan/matmul_q8_gemv.comp` — llama.cpp/ThunderKittens structure:
one output column per workgroup, 16 lanes cooperatively read one 32-byte Q8_0
block as 16 consecutive uint16s (coalesced burst), 16 blocks in flight per
workgroup (256 threads) to hide DRAM latency, register accumulators across the
whole K walk, single subgroupAdd + shared-memory reduce.

The flaw it fixes: the per-column GEMV had 16 output columns each streaming its
own weight column independently → instantaneous reads 896 bytes apart = scattered,
one wasted cache line per column.

Gate in `src/vulkan_backend.cpp` (`matmul_q8`, ~line 695):
`gemv = m <= 4 && ru(n,16)/16 >= 128`. Tiled path needs **n** workgroups
(one per output column), not ru(n,16)/16.

## Q8 on-GPU weight layout

`upload_q8_block` (`src/vulkan_model.cpp:261`) strips the fp16 scale header:
w8 = n*k int8 (stride **32** per block, NOT 34), ws = separate fp16 scales,
one per block. Weight is [n][k] row-major → column c owns blocks
[c*nb_total, (c+1)*nb_total). Two readers: `matmul_q8.comp` (tiled prefill) and
the GEMV. Rebasing the upload would break the tiled kernel, so sign is absorbed
in-shader.

Correct unsigned identity for int8 (Q8 is two's-complement, NOT biased):
q = u for u<128, q = u−256 for u>=128, so
`sum(x*q) = sum(x*u) − 256 * sum_{u>=128}(x)`.
The q = u−128 version is WRONG (instant tell: garbage logits).

## Dispatch anatomy (NANO_DISPATCH env)

388 dispatches/token for qwen2.5-0.5b: 144 matmul (f16), 72 add_bias,
49 rms_norm_add, 48 rope, 24 paged_attention, 24 silu_and_mul, 24 store_kv,
1 matmul_q8 (grid 9496). Weight formats: attn_output + ffn_down = f16;
qkv + gate_up = Q8_0; token_embd = f16.

## Verification harness

```
$env:GOLDEN_MODEL="C:/models/tinyllama-15M-stories-Q2_K.gguf"
$env:GOLDEN_TOKENS="29892"; $env:GOLDEN_TOKENS_CHAT="x"
# build target vk_golden_test, then run vk_golden_test.exe
```
Expect `PASS qwen25_plain (match 1/1)` with top5
`29892(10.03) 29889(9.91) 29901(8.66) 278(8.62) 340(8.53)`.
The chat FAIL is expected (different prompt anchor). Bit-exactness = identical
top-5 AND identical logit values.

NANO_DEBUG=1 dumps `gpu top5` — compare configs directly.

Shader build pipeline:
```
& "C:/VulkanSDK/1.4.357.0/Bin/glslangValidator.exe" -V --target-env vulkan1.2 -o build_vk/spv/<name>.spv shaders/vulkan/<name>.comp
python tools/gen_spv_header.py build_vk/spv include/nanovllm/spv.h
cd build_vk; cmake --build . --config Release --target nanovllm_vulkan
```
CMake doesn't reliably re-fire for NEW shaders — compile manually.

## Bugs found via golden diff (recurring patterns)

- tiled GEMV needs n workgroups (one per output column), not ru(n,16)/16;
  sends 1/16 the columns, rest stay garbage.
- `break` inside a cooperative loop diverges lanes → corrupts coalesced loads
  AND subgroupAdd. Use an if-guard, never break.
- early return before reduction barriers deadlocks the workgroup.
- missing `col*nb_total` in the scale index → all columns identical logits.
- w8 stride 32 not 34 (scale header stripped at upload).

## Next levers (priority order)

1. **K-quant GEMV** (`matmul_qk_gemv.comp`) — qwen2.5-0.5b's q/k/v are Q2_K, so
   this is the MAJORITY of decode weight traffic and still uses the old
   per-column layout. Biggest expected win.
2. **f16 decode matmul** (attn_output, ffn_down, n=896) still on the 16x16 tile
   — 144 dispatches/token. Current gate excludes it (needs ru(n,16)/16>=128).
3. **Dispatch fusion** — 388/token, but per-dispatch cost scales with work
   (tinyllama 0.09ms vs qwen 0.34ms), so only worth it AFTER matmuls are fast.
4. x vector re-read per output column; reference caches it. Minor until
   weights are fast.

Also outstanding (non-perf): `NANO_PIPE_LAYERS` parsed but not wired into the
`if (layer == 0)` guards at `src/vulkan_model.cpp:757/771`.

## User context

- References authorized for study: ThunderKittens (HazyResearch), lolcats,
  HipKittens, LocalAI (mudler).
- User mused "no-matmul and tiling may actually help" — open to algorithmic
  restructure, not just shader tweaks.
- Ponytail mode full. No unrequested abstractions; shortest working diff wins.
- Blocked / abandoned: DEC-2 drafter (never hits on short prompts), Gemma-4-12B,
  Qwen3.5-9B, DEC-1 MTP, KV-1 FP8, ATTN-3, SERV-2, 8GB-MoE, HIP/ROCm.
