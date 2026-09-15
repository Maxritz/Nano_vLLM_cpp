"""
requant.py — requantize fp32 weights to a chosen block-32 format.

Two choices, both block-32 (subgroup-aligned per the spec), trading size for
simplicity:

  q8_0  — 34 bytes/32 elements = 8.5 bits/element. Trivial to unpack in a
          shader (one scale, direct int8 read). ~2.1x the size of the
          original Q3_K_M average (~4 bits/element) — that size cost wasn't
          flagged clearly enough when this was first picked as the default;
          it's real and worth choosing against if size matters.

  q4_0  — 18 bytes/32 elements = 4.5 bits/element. Close to the original
          file's average bit width, still block-32/subgroup-aligned, still a
          simple two-nibbles-per-byte unpack (harder than Q8_0's direct read,
          much simpler than K-quant's packed 6-bit sub-scales). Reasonable
          middle ground between size and shader complexity.

Both are vectorized — the original per-block Python loop (Q8_0 only, from
before Q4_0 existed) is preserved in requant_ref.py as `_slow_quantize_q8_0`
and used as a diff-test oracle.

Note: this quantizer is symmetric/linear and self-consistent with this
project's own dequant_q4_0/dequant_q8_0 — it is NOT necessarily bit-identical
to ggml's own quantize_row_q4_0 (which uses a specific rounding-and-clamp
formula). That doesn't matter for round-tripping through this pipeline's own
dequant, which is what's actually verified below; it would matter if you
needed byte-for-byte parity with llama.cpp's own quantizer output.
"""

import numpy as np
from dequant import dequant_q8_0, dequant_q4_0


def quantize_q8_0(x: np.ndarray) -> bytes:
    """fp32 1-D array -> Q8_0 byte blocks, vectorized. Pads the final block
    with zeros if len(x) isn't a multiple of 32."""
    x = x.astype(np.float32)
    n = x.shape[0]
    pad = (-n) % 32
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
    n_blocks = x.shape[0] // 32
    blocks = x.reshape(n_blocks, 32)

    amax = np.max(np.abs(blocks), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(amax > 0, amax / 127.0, 0.0).astype(np.float32)
        inv_d = np.where(d != 0, 1.0 / d, 0.0).astype(np.float32)
    q = np.clip(np.round(blocks * inv_d[:, None]), -127, 127).astype(np.int8)

    d16 = d.astype(np.float16)
    out = np.empty((n_blocks, 34), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:34] = q.view(np.uint8)
    return out.tobytes()


def quantize_q4_0(x: np.ndarray) -> bytes:
    """fp32 1-D array -> Q4_0 byte blocks, vectorized. ~half the size of
    Q8_0 for the same block count. Pads the final block with zeros if
    len(x) isn't a multiple of 32."""
    x = x.astype(np.float32)
    n = x.shape[0]
    pad = (-n) % 32
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
    n_blocks = x.shape[0] // 32
    blocks = x.reshape(n_blocks, 32)

    amax = np.max(np.abs(blocks), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(amax > 0, amax / 8.0, 0.0).astype(np.float32)
        inv_d = np.where(d != 0, 1.0 / d, 0.0).astype(np.float32)
    q = np.clip(np.round(blocks * inv_d[:, None]) + 8, 0, 15).astype(np.uint8)

    low = q[:, 0:16]
    high = q[:, 16:32]
    packed = (low | (high << 4)).astype(np.uint8)

    d16 = d.astype(np.float16)
    out = np.empty((n_blocks, 18), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(n_blocks, 2)
    out[:, 2:18] = packed
    return out.tobytes()


# format name -> (encoder, matching dequant for round-trip checks, ggml type id, bytes/32-elements)
FORMATS = {
    "q8_0": (quantize_q8_0, dequant_q8_0, 8, 34),
    "q4_0": (quantize_q4_0, dequant_q4_0, 2, 18),
}


def roundtrip_error(x: np.ndarray, fmt: str = "q8_0") -> float:
    """Quantize then dequantize in the given format, return max absolute
    error — used by the test suite to confirm encoder and dequant agree."""
    encode, decode, _, _ = FORMATS[fmt]
    n = x.shape[0]
    packed = encode(x)
    n_padded = n + ((-n) % 32)
    y = decode(packed, n_padded)[:n]
    return float(np.max(np.abs(x - y)))
