# analysis log

## 2026-09-16 — debate-2: chat-template + system-turn wiring
Verdict: MERGE — keep full wiring (template→im_start→raw chain, --system, multi-turn-ready API; already shipped per llama.cpp parity), adopt minimalist's validation-first demand (malformed-template fuzz must fall back to raw prompt, exit 0). Rationale: user-directed full scope + EP1's crash-safety test is the only cheap proof that matters.

## 2026-09-16 — debate-1: tinyllama-1.1b anomaly (model-property vs engine-bug)
Verdict: MERGE — adopt EP1's disambiguation goal but execute via cheaper force-F16-lm_head experiment first (falsifies H-NATIVE directly); llama.cpp only if that fails. Rationale: 6-file correlation (every native-Q4/Q6-in-path file bad, all others coherent) makes H-NATIVE testable in 5 min vs 10+ min external build.
