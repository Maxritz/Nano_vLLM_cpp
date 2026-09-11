# Model family support

Status of this engine vs llama.cpp, which today runs ~90+ architecture families.
"Supported" here means: loads, runs, and passes output checks on this codebase.

## Supported now

| Family | Formats | Dtypes | Status |
|---|---|---|---|
| **Qwen2 / Qwen2.5** dense (0.5B–72B: base, -instruct, -coder, math) | dir (config.json + safetensors + tokenizer.json) and/or `.gguf` | F16, BF16, **Q8_0** | ✅ verified (golden tests on 7B safetensors; coder-3b Q8_0 GGUF end-to-end) |
| **Qwen3** dense (0.6B–32B) | same | F16, BF16, Q8_0 | 🟢 wired, not yet validated: per-head `q_norm`/`k_norm` load & apply (`model.cpp:304,450`), `qwen3.*` GGUF keys work because config is read as `${arch}.*` |
| **QwQ** (Qwen2-MoE-preview arch — the 32B **dense-preview** variants that ship as `qwen2` arch) | GGUF | F16/Q8_0 | 🟡 only if the GGUF's `general.architecture` is `qwen2`; the MoE release is not |

Requirements for a model to fall in the rows above:

- `general.architecture` = `qwen2` or `qwen3` (HF dir: `Qwen2ForCausalLM`/`Qwen3ForCausalLM` in config.json)
- `head_dim <= 256` (attention kernel limit), dims multiple of 32 for Q8_0 row slicing
- byte-level BPE tokenizer (`tokenizer.ggml.tokens`/`.merges` or HF `tokenizer.json`)
- dense MLP (SwiGLU), GQA or MHA, full rotary (partial-rotary models load but rope dims would be wrong)
- weights F16/BF16/Q8_0; Q8_0 stays in VRAM as Q8_0 (fused dequant in-kernel)

## Not supported (and what blocks each family)

| llama.cpp family group | Examples | Blocker |
|---|---|---|
| LLaMA / Mistral / Gemma / Phi / DeepSeek / GLM / Yi / Command-R / gpt-oss … | Llama-3.x, Mistral-Nemo, Gemma-3, Phi-4, DeepSeek-V3/R1, GLM-4.5 | Different arch keys are fine (metadata read is `${arch}.*`), but each needs: tensor-name map entry + tokenizer quirks (Gemma/Phi SPM), **logit soft-capping** (Gemma-2/3), **sliding-window attention** (Gemma-2/3, Mistral long-context, MiniMax), dual BOS, `attn_output_gate`, head-count≠`q/k/v` name sets. No engine features missing for plain LLaMA/Mistral — mostly name/tokenizer plumbing |
| MoE families | Mixtral, Qwen2-MoE, Qwen3-MoE, GPT-OSS, Grok, OLMoE, gpt-oss | expert routing + per-expert weight gather (top-k FFN loop) not implemented |
| MLA / latent attention | DeepSeek-V2/V3, Kimi | KV cache stores latent c_KV, needs different attention + cache kernel |
| Hybrid SSM/state-space | Mamba(2), Jamba, Falcon-H1, Zamba | recurrent state kernels not implemented |
| RWKV | RWKV-5/6 | completely different time-shift/WKV kernels |
| Encoder / embedding-only | BERT, GTE, nomic-bert, jina, GEPA | no bidirectional attention pooling path (these are embedding models anyway) |
| T5 / seq2seq | FLAN-T5 | cross-attention + encoder-decoder scheduler loop absent |
| Multimodal | Qwen2/2.5/3-VL, Gemma-3 vision, Llama-4 vision, Pixtral, SmolVLM | vision tower + image-token splice not present |
| Diffusion / other | SD, stable-audio, BPE-only non-LMs | out of scope |
| Other quant formats | anything GGUF Q2–Q6_K, IQ*, MXFP4 on safetensors | only `Q8_0` has kernels; Q4_0 is the next easiest (same block-scale idea, nibble unpack) |

## Engine-level gaps (any family)

- fp32 compute + fp32 KV cache throughout (no fp16/bf16 activations, no paged-fp8 KV)
- no speculative decoding, no LoRA, no beam/logit-bias plumbing beyond top-p/temp sampling
- context capped at build-time default 4096 via `--max-model-len`; no chunked-prefill
- single GPU, no tensor-parallel

## Realistic path to broader coverage

1. **LLaMA-3 / Mistral dense** — cheapest real win: add name mappings + dual-BOS flag; no new kernels.
2. **Qwen3 validation** (already wired) — run goldens against a Qwen3 GGUF.
3. **Q4_0 kernel** — 72B-Q4 lands in 16GB where Q8 can't.
4. **Sliding window + softcap** — unlocks Gemma-2/3 and long-context Mistral.
5. **MoE gather** — unlocks Mixtral/Qwen3-MoE (big effort, big payoff).
