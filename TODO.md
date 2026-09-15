# TODO — Nano_vLLM_cpp (Vulkan-first)

Debug probes (`[PIPE ...]` under `NANO_PIPE`, `gpu top5` under `NANO_DEBUG`) stay in
the tree until the whole backlog is green.

## Now
1. [P0] BOS prepend — `tokenizer.cpp` load_gguf: read `tokenizer.ggml.bos_token_id`,
   honor `add_bos_token` (default TRUE for llama-arch when key absent).
   Verify: `ids[0]==1`, tinyllama top1 flips 8111→22168 (probed via ref), coherent "Paris".
2. [P0] Re-run VERIFY matrix: tinyllama-q2k + moe-q4k GGUF + dense-q2k + `vk_golden_test`.
3. [P1] Wire `chat_template.hpp` (Jinja-lite, tests green) into `--chat` for zephyr-style
   GGUFs (`<|user|>` specials; engine only knows im_start/im_end today).
4. [P1] Re-mirror HIP `model.cpp` after BOS fix lands (agent-written mixed-fuse there is
   unverified; no HIP toolchain on this box).
5. [P2] QUANT-1/2: probe-verify `Q4_0/Q4_1/IQ4_XS` host-upcast table entries vs gguf-py
   `quants.dequantize` (canonical method proven this session).
6. [P2] VRAM: all-F16-upcast files cost ~3.6× vs native qk (Q2_K GPU kernel would close it).

## High
7. VKLAYOUT-1: tiled-layout reader (`vulkan_native` meta + tile index math in matmul_qk/embedding)
8. VKLAYOUT-2: A/B golden `_vk` vs source GGUF, greedy token match
9. ATTN-1: sliding-window bounds — shader+backend wired; model passes `0,0`; needs config
   fields + Gemma GGUF mapping
10. ATTN-2: logit softcap — attention wired; final-logits softcap missing

## Medium
11. ATTN-3 K/V SLM tiling · CTX-1 YaRN+StreamingLLM · TOK-2 wire pre_tokenizer header
(map GGUF `tokenizer.ggml.pre=default`→llama regex) · TOK-3 chat wiring (=#3) ·
HF-2 arch registry · PLUG-1 stable C ABI+manager · SERV-1 OpenAI server (draft in
temp `w3_server.txt`) · KV-1 FP8 cache · PF-1 chunked prefill · PERF-1 telemetry+`--bench`

## Low
12. CACHE-1/2 · PLUG-2 · TOK-1 SPM header (tested, unwired) · QUANT-4 MXFP4 · QUANT-5
AWQ/GPTQ · SERV-2 embeddings+batching · RAG-1 BM25 · MCP-1 · AGENT-1 · REAS-1 · GEN-1
· SAMP-1 contextual rep penalty (`sample_row`) · DEC-1 MTP · DEC-2 prompt-lookup ·
ADAPT-1 LoRA · PROF-1 profiles

## Deferred
13. HIP build/verify pass (Vulkan-first policy).

## Done (verified, this+last session)
mixed-fuse q/k/v+gate/up · Q4_K host dequant canonical (GPU top5 == CPU ref top5) ·
tokenizer ids == sentencepiece golden · Q2_K dequant == gguf-py · MoE e2e · dense Q2_K
"Paris" · P2-1..P2-4 · QUANT-3 · history path-scrub (`9abeae5`).
