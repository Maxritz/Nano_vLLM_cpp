# TODO - Nano_vLLM_cpp (Vulkan-first)

Every backlog item keeps its own line with its own status. Debug probes
([PIPE ...] under NANO_PIPE, gpu top5 under NANO_DEBUG) stay in the tree
until the entire list below is green.

## Session state
- [x] P2-3: dedicated transfer queue w/ ownership barriers, fallback single-family
      (VERIFIED: 'VK: dedicated transfer queue family 2' at runtime)
- [x] P2-4: fused route+FFN single dispatch (optimistic + miss fixup, env-gated
      MOE_NO_FUSED, A/B verified prior session; re-check in VERIFY)
- [ ] VERIFY workers: full build (golden PASS 2x plain+chat; GPU==CPU all tags; tinyllama-1.1b degenerate = model property, SP-faithful proven) + MoE/dense/ATTN/TOK/QUANT golden matrix
      (IN PROGRESS: mixed-fuse fix + Q4_K canonical fix landed; GPU top5 == CPU
      ref top5 on tinyllama-q2k; matrix re-run pending after BOS fix)
- [x] BOS prepend (NEW P0, root cause CONFIRMED): tokenizer.cpp load_gguf reads
      bos_token_id=1 but never prepends; GGUF omits add_bos_token => llama-arch
      default must be TRUE. Probe: ref+BOS flips top1 8111(vector)->22168.
      Fix: store add_bos_/bos_id_, prepend in encode_text. Test: tok_probe ids[0]==1.
- [ ] HIP re-mirror: src/model.cpp agent edits (mixed-fuse) unverified, no toolchain
      here; re-sync after BOS fix lands.

## 8GB-MoE
- [x] 8GB-MoE (1/3): synthesize tiny Q4_K GGUF MoE locally as test artifact
      (DONE: temp moe_tiny_q4k, 8.2MB, loads + prefills)
- [ ] 8GB-MoE (2/3): GGUF MoE naming (blk.N.ffn_*_exps) + lift gguf-only throw
      (loads+prefills now; close on golden token-match in VERIFY)
- [ ] 8GB-MoE (3/3): Q4_K/Q6_K expert host-dequant in moe_place, golden-verified
      (ran; bit-golden vs gguf-py method still open -> fold into VERIFY)

## VKLAYOUT
- [ ] VKLAYOUT-1: tiled-layout reader (vulkan_native meta + tile index math in
      matmul_qk/embedding)
- [ ] VKLAYOUT-2: A/B golden (_vk vs source GGUF, greedy token match)

## ATTN
- [ ] ATTN-1: sliding-window bounds in paged_attention (Gemma SWA + bounded-KV
      decode). State: shader sw_start + backend PC wired; vulkan_model.cpp passes
      0,0 -- needs config fields + GGUF mapping.
- [ ] ATTN-2: logit softcap (tanh) in attention + final logits (Gemma-2/3).
      State: attn softcap in shader; final-logits cap missing.
- [ ] ATTN-3: K/V SLM tiling in paged_attention (prefill throughput, fewer V re-reads)

## CACHE
- [ ] CACHE-1: radix prefix-cache (tree-shared KV blocks across requests)
- [ ] CACHE-2: HGCA-style host salient KV + LSE merge (dense recent on GPU, sparse
      history on host)

## CTX
- [ ] CTX-1: YaRN rope scaling + StreamingLLM eviction (sink + recent window)

## PLUG
- [ ] PLUG-1: stable C ABI + manager (DLL load, switch resolution, builtin fallback)
- [ ] PLUG-2: streaming sink+window policy as first DLL + --kv-policy /
      --ctx-window / --rope-scale switches

## TOK
- [ ] TOK-1: SPM-Unigram Viterbi decode/encode (Gemma/Llama SPM models). State:
      engine encode_unigram proven == sentencepiece golden on tinyllama; poolside
      spm_tokenizer.hpp header also green. Close after VERIFY or keep as fallback.
- [x] TOK-2: honor pre_tokenizer config from tokenizer.json (TikToken-style splits).
      State: poolside pre_tokenizer.hpp green (POSIX->ECMA + Isolated glue); WIRE
      GGUF tokenizer.ggml.pre=default -> llama regex.
- [x] TOK-3: Jinja-lite chat-template renderer (loops/conditionals/roles). State:
      chat_template.hpp tests green; WIRE into --chat incl. zephyr user/assistant
      specials (engine only knows im_start/im_end today).

## HF
- [x] HF-1: read generation_config.json (per-model EOS/sampling defaults)
      (DONE: config.hpp gen_* flags; CLI wins in main_vulkan)
- [ ] HF-2: architectures-registry for loader dispatch (replace scattered probes)

## QUANT
- [ ] QUANT-1: Q4_0/Q4_1 host upcast (Phi-3-mini-q4 + all Q4_0 files fail today).
      State: kUpcastKinds HAS both entries -- probe-verify vs gguf-py dequantize,
      close or fix.
- [ ] QUANT-2: IQ4_XS host upcast (gemma-4-12B file needs it). Same: entry exists,
      verify.
- [x] QUANT-3: Q2_K host upcast (completes Q2_K-family coverage)
      (DONE: gguf-py canonical proof, head 6/6 + block-0 sum exact)
- [ ] QUANT-4: MXFP4 safetensors (GPT-OSS style) load path
- [ ] QUANT-5: AWQ/GPTQ safetensors support (grouped 4-bit, different layout)
- [ ] QUANT-6: Q2_K native GPU kernel (NEW: upcast-to-F16 files cost ~3.6x VRAM vs
      native qk path; kernel closes the gap)

## SERV
- [ ] SERV-1: OpenAI-compatible HTTP server mode (--server stub exists; poolside
      http_server.hpp draft written, unwired)
- [ ] SERV-2: embeddings endpoint + concurrent request batching

## RAG / MCP / AGENT / REASONING / GEN
- [ ] RAG-1: BM25 chunk retrieval + prompt stuffing (no embedding model needed)
- [ ] MCP-1: MCP client (stdio/SSE tools, feed results back to model)
- [ ] AGENT-1: tool-call loop (model tools + MCP exec + feedback, max-iters)
- [ ] REAS-1: thinking-mode handling (parse/strip think tags, budgets, --show-thinking)
- [ ] GEN-1: structured output (JSON-schema/GBNF constrained sampling)

## SAMPLER / KV / PREFILL / DECODE / ADAPTERS / PERF
- [ ] SAMP-1: contextual repetition penalty in sampler (TokenSwift s3.4, windowed).
      State: sample_row has windowed rep_penalty; verify vs spec, keep or extend.
- [ ] KV-1: FP8 KV cache (halves KV, in-shader dequant)
- [ ] PF-1: chunked prefill (bound prefill memory for long prompts)
- [ ] DEC-1: speculative decoding via MTP heads (tiny model already ships mtp weights)
- [ ] DEC-2: prompt-lookup + generated-text n-gram drafting (model-free, TokenSwift-style)
- [ ] ADAPT-1: LoRA adapter attach (Edge0 has the contract: lora_A/B)
- [x] PERF-1: telemetry (TTFT/tok-s/VRAM report + --bench for Vulkan)
- [ ] PROF-1: config profiles unifying switches (like Edge0 prod presets)

## Verified this session (evidence trail)
- tinyllama-q2k: GPU top5 == CPU ref top5 (8111,12126,4087,6492,23583)
- attn_q Q2_K + attn_v Q4_K dequant == gguf-py canonical (sums + head values)
- tokenizer ids == rebuilt-sentencepiece golden (all 13 ids exact)
- mixed-precision fuse: qwen2.5-q2k dense coherent ('Paris...')
- moe-q4k GGUF loads + prefills (exit 0)
- history scrubbed of personal paths (pushed 9abeae5..e65e03d)
