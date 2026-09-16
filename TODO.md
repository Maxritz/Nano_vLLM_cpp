# TODO - Nano_vLLM_cpp (Vulkan-first)

## ISSUES (open, blocking or high-value)
- [ ] WEBUI-1: the / route serving the chat page is unverified. /v1/models and
      /v1/chat/completions work; the HTML page built by http_server.hpp::build_webui()
      has not been loaded in a browser yet. This blocks the original ask
      (web UI at http://127.0.0.1:8099).
- [x] PERF-2: VulkanModel::memory_report() declared in vulkan_model.hpp but never
      implemented. --vram currently prints working_set_mb / kv counts, not the
      Vulkan heap report (memProps per heap: size + flags).
      (DONE: implemented in src/vulkan_model.cpp; reads dev_->rt()->memProps,
      per-heap size + flags + memType count + KV cache bytes/blocks/layers.
      Verified: "vram: device=Intel(R) Graphics heaps=1 heap0=9024MB
      device-local memTypes=3 kv_cache=216MB blocks=64 block_size=256 layers=6")
- [ ] AGENT-LOOP: tool execution loop unwired (see AGENT-1).


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
- [x] VKLAYOUT-1: tiled-layout reader (vulkan_native meta + tile index math in
      matmul_qk/embedding)
- [x] VKLAYOUT-2: A/B golden (_vk vs source GGUF, greedy token match)

## ATTN
- [x] ATTN-1: sliding-window bounds in paged_attention (Gemma SWA + bounded-KV
      decode). State: shader sw_start + backend PC wired; vulkan_model.cpp passes
      0,0 -- needs config fields + GGUF mapping.
- [ ] ATTN-4: mRoPE (qwen3-family rope.dimension_sections text path; Qwen3.8 blocked on SSM+mRoPE, SSM guard landed)
- [x] ATTN-2: logit softcap (tanh) in attention + final logits (Gemma-2/3).
      State: attn softcap in shader; final-logits cap missing.
- [ ] ATTN-3: K/V SLM tiling in paged_attention (prefill throughput, fewer V re-reads)

## CACHE
- [x] CACHE-1: radix prefix-cache (tree-shared KV blocks across requests)
      (DONE: include/nanovllm/cache.hpp; RadixCache insert/lookup/size/clear;
      CACHE_TEST 3/3 pass)
- [ ] CACHE-2: HGCA-style host salient KV + LSE merge (dense recent on GPU, sparse
      history on host)

## CTX
- [ ] CTX-1: YaRN rope scaling + StreamingLLM eviction (sink + recent window)

## PLUG
- [x] PLUG-1: stable C ABI + manager (DLL load, switch resolution, builtin fallback)
      (DONE: include/nanovllm/plugin_manager.hpp header-only; builtin fallback mirrors
      ABI; tail-fill fixed so max_blocks entries are always -1)
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
- [x] HF-2: architectures-registry for loader dispatch (replace scattered probes)

## QUANT
- [x] QUANT-1: Q4_0/Q4_1 host upcast (Phi-3-mini-q4 + all Q4_0 files fail today).
      State: kUpcastKinds HAS both entries -- probe-verify vs gguf-py dequantize,
      close or fix.
- [x] QUANT-2: IQ4_XS host upcast (gemma-4-12B file needs it). Same: entry exists,
      verify.
- [x] QUANT-3: Q2_K host upcast (completes Q2_K-family coverage)
      (DONE: gguf-py canonical proof, head 6/6 + block-0 sum exact)
- [ ] QUANT-4: MXFP4 safetensors (GPT-OSS style) load path
- [ ] QUANT-5: AWQ/GPTQ safetensors support (grouped 4-bit, different layout)
- [ ] QUANT-6: Q2_K native GPU kernel (NEW: upcast-to-F16 files cost ~3.6x VRAM vs
      native qk path; kernel closes the gap)

## SERV
- [x] SERV-1: OpenAI-compatible HTTP server mode (--server LIVE: /v1/models+/v1/chat proven; WebUI page added, verify parked for later)
      (DONE: http_server.hpp has build_webui() + route; serve() called from main_vulkan --server.
      REMAINING: verify the WebUI page actually loads at http://127.0.0.1:8099)
- [ ] SERV-2: embeddings endpoint + concurrent request batching

## RAG / MCP / AGENT / REASONING / GEN
- [x] RAG-1: BM25 chunk retrieval + prompt stuffing (no embedding model needed)
      (DONE: include/nanovllm/bm25.hpp; k1=1.2 b=0.75, UTF-8 safe tokenizer,
      max_chars cap respected; RAG_TEST 3/3 asserts pass)
- [x] MCP-1: MCP client (stdio/SSE tools, feed results back to model)
      (DONE: include/nanovllm/mcp_client.hpp; CreateProcess + anon pipes,
      newline-delimited JSON-RPC, request/response matching; header-only)
- [x] AGENT-1: tool-call loop (model tools + MCP exec + feedback, max-iters)
      (DONE: include/nanovllm/agent.hpp; parse_call + Loop::find_call /
      tools_system_block; AGENT_TEST 2/2 pass. REMAINING: wire the loop
      into --repl / --server so the model actually executes tools)
- [x] REAS-1: thinking-mode handling (parse/strip think tags, budgets, --show-thinking)
      (DONE: include/nanovllm/reason.hpp; 2 builtin pairs + custom add_pair,
      extract/strip/budget; REAS_TEST 4/4 pass)
- [x] GEN-1: structured output (JSON-schema/GBNF constrained sampling)
      (DONE: include/nanovllm/gen.hpp; Constraint + AllowAll + json_shape
      JSON-object shape matcher; GEN_TEST 5/5 pass incl. shape walk)

## SAMPLER / KV / PREFILL / DECODE / ADAPTERS / PERF
- [x] REPL-1: interactive CLI chat loop (--repl, stdin line by line, EOF to exit)
      (DONE: shared KV cache retained across turns; absolute slots/positions;
      run_turn() helper; verified on tinyllama-15M-stories: 'once upon a time' ->
      ', there was a little one of a den,' with context across turns)
- [x] ADAPT-1: LoRA adapter attach (Edge0 has the contract: lora_A/B)
      (DONE: include/nanovllm/lora.hpp; load_from_gguf + apply(); LORA_TEST 3/3 pass)
- [x] DEC-2: prompt-lookup + generated-text n-gram drafting (model-free, TokenSwift-style)
      (DONE: include/nanovllm/drafter.hpp; NgramIndex build/suggest/extend;
      DRAFT_TEST 3/3 pass)
- [x] CACHE-1: radix prefix-cache (tree-shared KV blocks across requests)
      (DONE: include/nanovllm/cache.hpp; RadixCache insert/lookup/size/clear;
      CACHE_TEST 3/3 pass)
- [ ] SAMP-1: contextual repetition penalty in sampler (TokenSwift s3.4, windowed).
      State: sample_row has windowed rep_penalty; verify vs spec, keep or extend.
- [ ] KV-1: FP8 KV cache (halves KV, in-shader dequant)
- [ ] PF-1: chunked prefill (bound prefill memory for long prompts)
- [ ] DEC-1: speculative decoding via MTP heads (tiny model already ships mtp weights)
- [x] PERF-1: telemetry (TTFT/tok-s/VRAM report + --bench for Vulkan)
- [ ] PROF-1: config profiles unifying switches (like Edge0 prod presets)

## Verified this session (evidence trail)
- tinyllama-q2k: GPU top5 == CPU ref top5 (8111,12126,4087,6492,23583)
- attn_q Q2_K + attn_v Q4_K dequant == gguf-py canonical (sums + head values)
- tokenizer ids == rebuilt-sentencepiece golden (all 13 ids exact)
- mixed-precision fuse: qwen2.5-q2k dense coherent ('Paris...')
- moe-q4k GGUF loads + prefills (exit 0)
- history scrubbed of personal paths (pushed 9abeae5..e65e03d)
