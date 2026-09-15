# Inference research radar

What each known technique means for Nano_vLLM_cpp (Vulkan iGPU, laptop RAM,
existing public weights only — no training capability). Rule of thumb: if it
needs training or model-shipped weights, it's watch-only; if it's
training-free math or cache policy, it's actionable.

## Speculative / parallel decoding

| Method | Idea | Training? | Fits us? | Action |
|---|---|---|---|---|
| Vanilla spec-decode (Leviathan/Chen) | Small drafter + parallel verify, lossless | No (needs drafter) | Needs a draft model | — |
| EAGLE-1/2/3 | Feature-level drafter heads, dynamic trees | Yes (drafter) | Only where heads ship | DEC-1 (MTP heads) |
| Medusa / Hydra | Extra MLP heads off last hidden state | Yes | Same as above | — |
| **FastMTP** (2509.18362) | Shared-weight MTP head + vocab-compressed drafting, 2.03× | Yes, but pattern is reusable | Vocab-compressed drafting.hadoop works with any head | DEC-1 |
| **TokenSwift** (2502.18890) | Self-draft + n-gram reutilization + dynamic KV + contextual penalty, 3× @100K | Partial (linear heads) / No (n-grams, penalty, dyn-KV) | N-gram table + penalty are pure policy | DEC-2, SAMP-1 |
| **Orthrus** (2605.12825) | Frozen AR + trainable diffusion head, shared KV, consensus = lossless 7.8× | Yes (16% params) | No (per-model training) | Watch |
| Prompt-lookup / REST / NEST | Draft from prompt n-grams / retrieved phrases | No | Trivially fits; tree verify is the only work | DEC-2 |

## Sparse / long-context attention

| Method | Idea | Training? | Fits us? | Action |
|---|---|---|---|---|
| FlashAttention 1/2/4 | IO-aware tiling + online softmax | No | Core math already in `paged_attention.comp` | ATTN-3 (tiling) |
| Sliding window (+StreamingLLM sink) | Fixed local window, O(1) KV | No | Direct kernel bounds | ATTN-1, CTX-1 |
| YaRN / NTK rope scaling | Stretch small-ctx models past trained length | No | Config math | CTX-1 |
| **Squeezed Attention** (2411.09688) | Offline K-means centroids for fixed context, 3–8× KV cut | Offline calibration only | Centroids for RAG doc store | RAG-1 |
| **Star Attention** (2411.17116) | Block-local + global two-phase, 11× mem, ~97–100% acc | No | Pattern fits chunked prefill | Later |
| **BSFA** (2512.07011) | Exact QK, skip ~50% value blocks by calibrated threshold | Calibration set | Needs per-layer thresholds + kernel | Later, measure first |
| Sparse Pattern Sharing (2505.19578) | Compute full attn on head subset, share across similar heads | No | Needs similarity validation + kernel | Later |
| **HGCA** (2507.03153) | Dense-recent on GPU + sparse-salient on CPU, LSE merge | No | Our unified-memory box *is* this shape | CACHE-2 |
| **AttnCache** (2510.25979) | Reuse similar attention maps from DB (prefill-only workloads) | DB build | Approximate; marginal for chat | Later (RAG prefill) |
| NSA / DSA / CSA+HCA | Learned indexer + top-k retrieval | Needs shipped weights | Per-model native only (Qwen3.8 QSA, DS) | Coverage items |
| MLA | Latent-compressed KV | Model-native | Only via MLA models | Coverage/MLA |
| GQA / MQA | Fewer KV heads | Model-native | Already in kernel | Done |
| SSM / GDN / DeltaNet | Fixed-size recurrent state | Model-native | Per-model kernels | Coverage items |
| FAST-Prefill (2602.20515) | FPGA dynamic-sparse prefill | Hardware | N/A | — |
| QuickMerge++ (2508.13204) | Vision token merging + AR prior | Trained prior, vision tower | N/A (no vision tower) | — |

## KV-cache management

| Method | Idea | Training? | Fits us? | Action |
|---|---|---|---|---|
| PagedAttention (2309.06180) | Blocked non-contiguous KV | No | Already mirrored in kernel | Done |
| **RadixAttention** | Prefix tree shares KV across requests | No | Block tables already the interface | CACHE-1 |
| Prefix / on-disk prefix cache (vLLM, DS-V4) | Reuse + persist shared prefixes | No | mmap page cache is half of it | CACHE-1 |
| FP8 / INT4 KV (KIVI, KVQuant) | Quantized cache, in-kernel dequant | No (calibration-lite) | Easy shader work | KV-1 |
| SnapKV / TOVA / H2O / PyramidKV | Importance-vote eviction policies | No | Policy plugins | Later (pick one) |
| Quest / InfLLM / RetrievalAttention | Per-query block retrieval, CPU offload tiers | No | Overlaps slot machinery | Later |
| Chunked prefill | Bound prefill memory by splitting prompts | No | Scheduler work | PF-1 |

## Serving / ecosystem (projects reviewed)

| Project | What it is | Reusable? | Action |
|---|---|---|---|
| FlashInfer (2501.01005) | CUDA serving kernels (block-sparse KV, JIT templates) | No (CUDA-only) | Design notes only |
| Quazmoz/npu-windows | Python + ipex-llm NPU server, chat UI, tool calling, 1–8B @ 2–15 tok/s | No code (Python stack) | Validates SERV-1/AGENT-1 direction; NPU backend = future option |
| mvafin/optimum-intel | Fork of HF optimum-intel (OpenVINO/NNCF) | Upstream only | OpenVINO backend = future option, low priority |

## Merits / demerits (actionable shortlist only)

| Method | Merits | Demerits | Net |
|---|---|---|---|
| Fused route+FFN (shipped) | 1 dispatch vs 2; no extra memory | Optimistic miss costs a re-run; K≤64/E≤256 guard | Keep; miss rate decides wins |
| Radix prefix cache | Multi-turn + multi-request sharing; exact, lossless | Tree + refcount machinery; single-user CLI gains little | Worth it once server does sessions |
| StreamingLLM sink+window | O(1) KV; 5-line policy; exact within window | Quality cliff past window on retrieval tasks | Default-on for long ctx, off for short |
| YaRN rope scaling | Unlocks 4–8× trained length, config-only | Perplexity degrades at extreme stretch; needs eval per model | Ship with per-model scale caps |
| FP8 KV | Halves KV traffic; near-lossless per DS-V3 | Extra dequant ALU; needs kernel per dtype path | Clear win on bandwidth-bound decode |
| SnapKV/TOVA-style eviction | Keeps important tokens, better than raw window | Prefill-time vote pass; policy tuning per model family | Pick one after StreamingLLM measured |
| Prompt-lookup drafting | Zero training, zero extra weights; instant on repetitive text | Acceptance collapses on novel text; verify overhead always paid | Good default; auto-disable on low acceptance |
| N-gram reutilization (TokenSwift) | Same as above + improves as generation grows | Table memory grows with output; needs cap + eviction | Fold into DEC-2 with cap |
| Vocab-compressed drafting (FastMTP/FR-Spec) | Shrinks draft LM-head cost (the decode bottleneck at big vocab) | Needs per-language freq tables; verify stays full-vocab | Worth it at 150K+ vocabs |
| MTP-head verify | Uses weights already shipped in file; lossless | Heads vary in quality; recursive use degrades w/o FastMTP-style tuning | Try on tiny model, measure acceptance |
| HGCA host/device split | Fits contexts bigger than VRAM; exact via LSE merge | PCIe traffic per token; needs salience heuristic for host set | Natural fit for our mmap design |
| Squeezed centroids (RAG) | 3–8× KV cut on fixed doc context, offline cost only | Calibration per doc-store; centroid-compare kernel needed | Do with RAG-1, not before |
| BSFA top-k blocks | ~1.1–1.2×, training-free, >99% acc reported | Per-layer calibration set; gains are thin | Measure only if attention still dominates |
| Chunked prefill | Bounds prefill memory; enables 262K+ prompts on iGPU | More dispatches; needs attention over cache + chunk | Required for long-ctx, no alternative |
| On-disk prefix cache | System prompts / RAG docs prefill once across runs | Invalidation policy; disk SSD wear trivial | Cheap with mmap; do with radix |
| Star Attention (two-phase block-sparse) | 11× memory cut at 97–100% acc, no training | Approximate; built for multi-host comms, not single laptop; custom kernels | Borrow the local-then-global pattern for chunked prefill; skip full port |
| Contextual penalty (sampler) | Kills long-run repetition loops; 5 lines | Slight speedup cost; can over-flatten diverse text | Gate behind flag, default off |
| Vanilla spec-decode | Lossless speedup; well understood | Needs a separate draft model in VRAM; laptop has no room for two models | Skip; self-drafting only |
| EAGLE-1/2/3 | SOTA acceptance via feature uncertainty modeling | Drafter must be trained per model; we can't train | Only where heads ship (see MTP row) |
| Medusa / Hydra | Simple extra heads, parallel drafts | Same training requirement; lower acceptance than EAGLE | Skip unless heads ship |
| Orthrus (dual AR+diffusion) | Lossless 7.8×, shared KV, 16% params tuned | Per-model diffusion-head training on H200s; not available | Watch; adopt if heads ever ship |
| FlashAttention 1/2/4 (full) | Optimal IO tiling; 50–73% FLOP util on A100 | CUDA-only kernels; our iGPU has no HBM/SRAM gap to exploit | Core math already ported; skip the library |
| Sparse Pattern Sharing | Full accuracy, fewer heads computed | Similarity must hold per model; needs validation harness + kernel | Later; needs a similarity study first |
| AttnCache (map reuse) | 1.2–1.6× end-to-end on prefill-only workloads | Approximate; needs similarity DB build + lookup per request | Only if RAG-prefill dominates profile |
| NSA / DSA / CSA+HCA | Learned top-k retrieval, huge ctx at low cost | Indexer weights ship only in DS/Qwen3.8-family files | Implement per-model, never generic |
| MLA | 2% KV of GQA-8 baseline; quality-preserving | New cache format + kernels per model | Per-model coverage item |
| GQA / MQA | Fewer KV heads, exact | Model-native | Done — already in kernel |
| SSM / GDN / DeltaNet | O(1) state, no KV growth | Entirely different kernels per architecture | Per-model coverage items |
| FAST-Prefill (FPGA) | 2.5× TTFT, 4.5× energy on Alveo | Wrong hardware | — |
| QuickMerge++ (vision merge) | Fewer tokens, needs trained prior + vision tower | No tower, no prior | — |
| PagedAttention | Non-contiguous KV, zero waste | — | Done — mirrored in kernel |
| INT4 KV (KIVI/KVQuant) | 4× smaller cache than fp16 | Calibration-lite; quality risk above fp8; shader work | After FP8 KV proves out |
| Quest / InfLLM block retrieval | Attend subset of offloaded blocks per query | Selector kernel + miss policy; complexity creep | Later; overlaps slot machinery |
| NPU backend (OpenVINO/ipex-llm route) | 2–15 tok/s at 5–10W on Core Ultra NPU | Python stack; second backend to maintain; ≤8B models only | Future option; Vulkan covers more models |
| OpenVINO IR backend | Graph opts + NNCF quant for Intel | Heavy dep vs no-deps philosophy; conversion step per model | Future option, low priority |
| SuperBPE tokenizers | 27% less inference compute, +4% MMLU (training-side) | No public model ships it; inference need is just "don't presplit" | One flag if a checkpoint ever appears |

## Standing decisions

- No method requiring per-model training enters the tree (Orthrus, EAGLE heads, Seer, learned indexers) — except where heads ship in the checkpoint (MTP).
- Approximate attention (BSFA, Star, pattern-sharing) only after exact paths are measured hungry.
- Vision/multimodal towers are out of scope until the text engine coverage lands.
