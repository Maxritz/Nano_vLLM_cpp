"""
dequant.py — dequantize ggml block formats to fp32 numpy arrays.

Confidence split (see ggml_types.GGML_TYPES for the per-type tag):

  VERIFIED   — F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0. Legacy QK=32 formats,
               simple bit layout, unit-tested against hand-crafted byte vectors
               in test_gguf2vk.py. Trust these.

  BEST-EFFORT — Q4_K, Q6_K. K-quant (QK_K=256) formats with packed 6-bit
               sub-block scales, implemented from the public ggml struct/algorithm
               but NOT cross-checked against a real llama.cpp reference dump.
               Structural parsing (byte offsets, block navigation) is solid;
               the scale-unpacking arithmetic could still be subtly wrong.

  NOT IMPLEMENTED — Q2_K, Q3_K, Q5_K. Bit-packing schemes I don't have confident
               recall of (particularly Q3_K's 12-byte 6-bit scale packing across
               4-value groups with a moving high-bit mask). Raises
               NotImplementedError loudly rather than guessing — a wrong answer
               here is worse than no answer, since it'd corrupt weights silently.

If you need Q2_K/Q3_K/Q5_K support: the right way to close this gap is to dump a
known tensor's expected values from llama.cpp itself (e.g. via a debug print or
gguf-py's own dequant) and diff against this module's output on the same bytes,
then fix until they match exactly — not from-memory guessing.
"""

import struct
import numpy as np

QK_K = 256


def _fp16_to_fp32(raw2bytes: bytes) -> float:
    return float(np.frombuffer(raw2bytes, dtype="<f2")[0])


def _slow_dequant_f32(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data[:n_elements * 4], dtype="<f4").astype(np.float32)


def _slow_dequant_f16(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data[:n_elements * 2], dtype="<f2").astype(np.float32)


def _slow_dequant_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    block_size, type_size = 32, 18
    n_blocks = n_elements // block_size
    out = np.empty(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * type_size:(i + 1) * type_size]
        d = _fp16_to_fp32(block[0:2])
        qs = block[2:18]
        for j in range(16):
            byte = qs[j]
            x0 = (byte & 0x0F) - 8
            x1 = (byte >> 4) - 8
            out[i * 32 + j] = x0 * d
            out[i * 32 + j + 16] = x1 * d
    return out


def _slow_dequant_q4_1(data: bytes, n_elements: int) -> np.ndarray:
    block_size, type_size = 32, 20
    n_blocks = n_elements // block_size
    out = np.empty(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * type_size:(i + 1) * type_size]
        d = _fp16_to_fp32(block[0:2])
        m = _fp16_to_fp32(block[2:4])
        qs = block[4:20]
        for j in range(16):
            byte = qs[j]
            x0 = byte & 0x0F
            x1 = byte >> 4
            out[i * 32 + j] = x0 * d + m
            out[i * 32 + j + 16] = x1 * d + m
    return out


def _slow_dequant_q5_0(data: bytes, n_elements: int) -> np.ndarray:
    block_size, type_size = 32, 22
    n_blocks = n_elements // block_size
    out = np.empty(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * type_size:(i + 1) * type_size]
        d = _fp16_to_fp32(block[0:2])
        qh = struct.unpack("<I", block[2:6])[0]
        qs = block[6:22]
        for j in range(16):
            xh_0 = ((qh >> (j + 0)) << 4) & 0x10
            xh_1 = ((qh >> (j + 12))) & 0x10
            x0 = ((qs[j] & 0x0F) | xh_0) - 16
            x1 = ((qs[j] >> 4) | xh_1) - 16
            out[i * 32 + j] = x0 * d
            out[i * 32 + j + 16] = x1 * d
    return out


def _slow_dequant_q5_1(data: bytes, n_elements: int) -> np.ndarray:
    block_size, type_size = 32, 24
    n_blocks = n_elements // block_size
    out = np.empty(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * type_size:(i + 1) * type_size]
        d = _fp16_to_fp32(block[0:2])
        m = _fp16_to_fp32(block[2:4])
        qh = struct.unpack("<I", block[4:8])[0]
        qs = block[8:24]
        for j in range(16):
            xh_0 = ((qh >> (j + 0)) << 4) & 0x10
            xh_1 = ((qh >> (j + 12))) & 0x10
            x0 = (qs[j] & 0x0F) | xh_0
            x1 = (qs[j] >> 4) | xh_1
            out[i * 32 + j] = x0 * d + m
            out[i * 32 + j + 16] = x1 * d + m
    return out


def _slow_dequant_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    block_size, type_size = 32, 34
    n_blocks = n_elements // block_size
    out = np.empty(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * type_size:(i + 1) * type_size]
        d = _fp16_to_fp32(block[0:2])
        qs = np.frombuffer(block[2:34], dtype=np.int8).astype(np.float32)
        out[i * 32:(i + 1) * 32] = qs * d
    return out


def _get_scale_min_k4(j: int, scales: bytes):
    """Unpack the 6-bit scale/min pair for sub-block j from Q4_K/Q5_K's 12-byte
    packed scales array (8 sub-blocks packed into 12 bytes). Best-effort — see
    module docstring confidence note."""
    if j < 4:
        d = scales[j] & 63
        m = scales[j + 4] & 63
    else:
        d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
        m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return d, m


def _slow_dequant_q4_k(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 144
    n_blocks = n_elements // QK_K
    out = np.empty(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * type_size:(i + 1) * type_size]
        d = _fp16_to_fp32(block[0:2])
        dmin = _fp16_to_fp32(block[2:4])
        scales = block[4:16]
        qs = block[16:144]
        base = i * QK_K
        for j in range(8):
            sc, m = _get_scale_min_k4(j, scales)
            d1 = d * sc
            m1 = dmin * m
            sub = qs[j * 16:(j + 1) * 16]
            for l in range(16):
                byte = sub[l]
                out[base + j * 32 + l] = d1 * (byte & 0x0F) - m1
                out[base + j * 32 + l + 16] = d1 * (byte >> 4) - m1
    return out


def _slow_dequant_q6_k(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 210
    n_blocks = n_elements // QK_K
    out = np.empty(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        block = data[i * type_size:(i + 1) * type_size]
        ql = block[0:128]
        qh = block[128:192]
        scales = struct.unpack("<16b", block[192:208])
        d = _fp16_to_fp32(block[208:210])
        base = i * QK_K
        for half in range(2):
            ql_h = ql[half * 64:(half + 1) * 64]
            qh_h = qh[half * 32:(half + 1) * 32]
            sc_h = scales[half * 8:(half + 1) * 8]
            out_base = base + half * 128
            for l in range(32):
                is_ = l // 16
                q1 = ((ql_h[l] & 0x0F) | (((qh_h[l] >> 0) & 3) << 4)) - 32
                q2 = ((ql_h[l + 32] & 0x0F) | (((qh_h[l] >> 2) & 3) << 4)) - 32
                q3 = ((ql_h[l] >> 4) | (((qh_h[l] >> 4) & 3) << 4)) - 32
                q4 = ((ql_h[l + 32] >> 4) | (((qh_h[l] >> 6) & 3) << 4)) - 32
                out[out_base + l] = d * sc_h[is_ + 0] * q1
                out[out_base + l + 32] = d * sc_h[is_ + 2] * q2
                out[out_base + l + 64] = d * sc_h[is_ + 4] * q3
                out[out_base + l + 96] = d * sc_h[is_ + 6] * q4
    return out


