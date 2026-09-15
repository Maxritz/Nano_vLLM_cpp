"""
requant.py — requantize fp32 weights to block-32 int8.

Worth noting explicitly: the spec calls for block size 32, symmetric linear
int8 quantization. That's not a new format to invent — it's bit-for-bit what
ggml's Q8_0 already is (fp16 scale + 32 signed int8 values per block). So
"requant to block-32" here just means: produce standard Q8_0 blocks. Reusing
dequant_q8_0 from dequant.py to round-trip-verify this module's output.
"""

import struct
import numpy as np
from dequant import dequant_q8_0


def _slow_quantize_q8_0(x: np.ndarray) -> bytes:
    """fp32 1-D array -> Q8_0 byte blocks. Pads the final block with zeros if
    len(x) isn't a multiple of 32 (caller should know if that's acceptable for
    the tensor in question — padding changes the logical shape)."""
    x = x.astype(np.float32)
    n = x.shape[0]
    pad = (-n) % 32
    if pad:
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
    n_blocks = x.shape[0] // 32
    out = bytearray()
    for i in range(n_blocks):
        block = x[i * 32:(i + 1) * 32]
        amax = np.max(np.abs(block))
        d = amax / 127.0 if amax > 0 else 0.0
        inv_d = (1.0 / d) if d != 0 else 0.0
        q = np.clip(np.round(block * inv_d), -127, 127).astype(np.int8)
        out += np.float16(d).tobytes()
        out += q.tobytes()
    return bytes(out)


