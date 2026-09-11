# Nano-vLLM C++/ROCm

A pure C++/HIP rewrite of the Python nano-vLLM repository. No PyTorch, no Triton, no flash-attn.

## Supported GPUs

- RDNA2: RX 6700 XT (`gfx1031`)
- RDNA4: RX 9070 XT (`gfx1201`)

## Build

From a shell where `HIP_PATH` points at ROCm 10.x:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER="$env:HIP_PATH/lib/llvm/bin/clang++.exe"
cmake --build build
```

Linux:

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++
cmake --build build
```

## Run

Full engine port: tokenizer (BPE + GGUF chat templates), safetensors and GGUF loading, Q8_0 quantized inference with fused dequant in-kernel. On Windows the ROCm runtime is delay-loaded via `AddDllDirectory` so the display driver's `System32\amdhip64_7.dll` cannot shadow it.

```powershell
.\build\nanovllm.exe <model_dir|model.gguf> [prompt] [options]
```

### Options

| Switch | Default | Meaning |
|---|---|---|
| `--chat` | off | Wrap `prompt` in the model's chat template (recommended for `-instruct` models; raw prompts get continued as text instead of answered) |
| `--tokens 1,2,3` | – | Use pre-tokenized IDs as the prompt (skips the tokenizer) |
| `--max-tokens N` | 64 | Maximum tokens to generate |
| `--temperature T` | 0.6 | Sampling temperature (greedy argmax is used at low values; set 0.01 for deterministic output) |
| `--max-model-len N` | 4096 | Context length; also caps KV-cache sizing |
| `--bench` | off | Ignore `prompt`; run a synthetic 4-sequence × 64-token prefill + decode benchmark and print PP/decode tok/s |
| `--num-seqs N` | 4 | Number of sequences for `--bench` |
| `--help` | – | Usage line |

### Model formats

- **Directory** with `config.json` + `*.safetensors` + `tokenizer.json` → safetensors (F16/BF16, or Q8_0-safetensors if present).
- **Directory** without safetensors but containing a `*.gguf` → GGUF.
- **Direct `*.gguf` file** → GGUF; config and tokenizer are read from GGUF metadata (`qwen2.*` / `tokenizer.ggml.*` keys), so a bare quantized file works with no companion files.
- Q8_0 weights stay Q8_0 in VRAM (34-byte blocks: fp16 scale + 32×int8); dequantization happens per-element inside the matmul/embedding kernels. Only supported architecture today: Qwen2/Qwen3-style dense (as used by Qwen2.5 & Qwen3).

### Environment

| Var | Effect |
|---|---|
| `HIP_PATH` | ROCm runtime location (Windows: falls back to the path baked in at build time) |
| `NANO_DEBUG=1` | Extra stderr: tokenizer merges, BPE steps, per-sample decisions, VRAM estimate |

### Tests

```powershell
.\build\nanovllm_test.exe        # kernel smoke tests
$env:GOLDEN_MODEL="G:\path\to\Qwen2.5-7B-Instruct"; .\build\nanovllm_golden.exe  # golden-token regressions
```

`nanovllm_golden.exe` checks generated token IDs against reference Qwen2.5 outputs; it exits non-zero on mismatch. `scripts/autofix.ps1` is a local build+test loop.
