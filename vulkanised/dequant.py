"""
dequant.py — dequantize ggml block formats to fp32 numpy arrays.

Vectorized with numpy — no per-element Python loops. The original loop-based
implementations (which the hand-crafted unit tests were written against) are
preserved in dequant_ref.py as `_slow_dequant_*` and used as a diff-test oracle
in test_gguf2vk.py: every vectorized function here is checked against its slow
counterpart on random data before being trusted. Q3_K/Q5_K have no slow oracle
(they're new) — those are checked with a scale-packing self-consistency test
instead (see test_gguf2vk.py), which is weaker evidence than an independent
reference but does catch real coding bugs in the bit manipulation.

Confidence split (see ggml_types.GGML_TYPES for the per-type tag):

  VERIFIED    — F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0. Hand-computed test
                vectors, plus diff-tested against the pre-vectorization oracle.

  BEST-EFFORT — Q4_K, Q6_K, Q3_K, Q5_K. K-quant (QK_K=256) formats with packed
                6-bit sub-block scales, implemented from the public ggml
                struct/algorithm layout. Structural parsing is solid; the
                scale-unpacking arithmetic has NOT been cross-checked against
                a real llama.cpp reference dump — treat output as provisional
                until you can diff it against one.

  NOT IMPLEMENTED — Q2_K. Same family as the others but not yet worked through.

To close the remaining confidence gap on Q4_K/Q6_K/Q3_K/Q5_K: dump a known
tensor's expected values from llama.cpp itself (a debug print, or gguf-py's
own dequant if it has one) and diff against this module's output on the same
raw bytes. That's the real verification; nothing here substitutes for it.
"""

import numpy as np

QK_K = 256


def _fp16_col(raw_cols: np.ndarray) -> np.ndarray:
    """raw_cols: (n_blocks, 2) uint8 -> (n_blocks,) float32, via fp16 reinterpret."""
    return raw_cols.copy().view("<f2").astype(np.float32).reshape(-1)


def _u32_col(raw_cols: np.ndarray) -> np.ndarray:
    """raw_cols: (n_blocks, 4) uint8 -> (n_blocks,) uint32."""
    return raw_cols.copy().view("<u4").astype(np.uint32).reshape(-1)


# ---------------------------------------------------------------- legacy (QK=32)

def dequant_f32(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data[:n_elements * 4], dtype="<f4").astype(np.float32)


def dequant_f16(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data[:n_elements * 2], dtype="<f2").astype(np.float32)


def dequant_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 18
    n_blocks = n_elements // 32
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    d = _fp16_col(raw[:, 0:2])
    qs = raw[:, 2:18].astype(np.int16)
    low = (qs & 0x0F) - 8
    high = (qs >> 4) - 8
    out = np.empty((n_blocks, 32), dtype=np.float32)
    out[:, 0:16] = low * d[:, None]
    out[:, 16:32] = high * d[:, None]
    return out.reshape(-1)


def dequant_q4_1(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 20
    n_blocks = n_elements // 32
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    d = _fp16_col(raw[:, 0:2])
    m = _fp16_col(raw[:, 2:4])
    qs = raw[:, 4:20]
    low = (qs & 0x0F).astype(np.float32)
    high = (qs >> 4).astype(np.float32)
    out = np.empty((n_blocks, 32), dtype=np.float32)
    out[:, 0:16] = low * d[:, None] + m[:, None]
    out[:, 16:32] = high * d[:, None] + m[:, None]
    return out.reshape(-1)


def dequant_q5_0(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 22
    n_blocks = n_elements // 32
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    d = _fp16_col(raw[:, 0:2])
    qh = _u32_col(raw[:, 2:6])
    qs = raw[:, 6:22].astype(np.uint32)
    j = np.arange(16, dtype=np.uint32)
    qh_col = qh[:, None]
    xh0 = ((qh_col >> j) << 4) & 0x10
    xh1 = (qh_col >> (j + 12)) & 0x10
    x0 = ((qs & 0x0F) | xh0).astype(np.int32) - 16
    x1 = ((qs >> 4) | xh1).astype(np.int32) - 16
    out = np.empty((n_blocks, 32), dtype=np.float32)
    out[:, 0:16] = x0 * d[:, None]
    out[:, 16:32] = x1 * d[:, None]
    return out.reshape(-1)


def dequant_q5_1(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 24
    n_blocks = n_elements // 32
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    d = _fp16_col(raw[:, 0:2])
    m = _fp16_col(raw[:, 2:4])
    qh = _u32_col(raw[:, 4:8])
    qs = raw[:, 8:24].astype(np.uint32)
    j = np.arange(16, dtype=np.uint32)
    qh_col = qh[:, None]
    xh0 = ((qh_col >> j) << 4) & 0x10
    xh1 = (qh_col >> (j + 12)) & 0x10
    x0 = ((qs & 0x0F) | xh0).astype(np.float32)
    x1 = ((qs >> 4) | xh1).astype(np.float32)
    out = np.empty((n_blocks, 32), dtype=np.float32)
    out[:, 0:16] = x0 * d[:, None] + m[:, None]
    out[:, 16:32] = x1 * d[:, None] + m[:, None]
    return out.reshape(-1)


def dequant_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 34
    n_blocks = n_elements // 32
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    d = _fp16_col(raw[:, 0:2])
    qs = raw[:, 2:34].copy().view(np.int8).astype(np.float32)
    return (qs * d[:, None]).reshape(-1)


# ---------------------------------------------------------------- K-quants (QK_K=256)

def _scale_min_k4_vec(scales: np.ndarray, k: int):
    """Vectorized get_scale_min_k4 — unpacks the 6-bit (scale, min) pair for
    sub-block k from Q4_K/Q5_K's 12-byte packed scales array.
    scales: (n_blocks, 12) uint8. Returns (sc, m) each (n_blocks,) float32."""
    if k < 4:
        sc = (scales[:, k] & 63).astype(np.float32)
        m = (scales[:, k + 4] & 63).astype(np.float32)
    else:
        sc = ((scales[:, k + 4] & 0x0F) | ((scales[:, k - 4] >> 6) << 4)).astype(np.float32)
        m = ((scales[:, k + 4] >> 4) | ((scales[:, k] >> 6) << 4)).astype(np.float32)
    return sc, m


def dequant_q4_k(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 144
    n_blocks = n_elements // QK_K
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    d = _fp16_col(raw[:, 0:2])
    dmin = _fp16_col(raw[:, 2:4])
    scales = raw[:, 4:16]
    qs = raw[:, 16:144]
    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for j in range(8):
        sc, m = _scale_min_k4_vec(scales, j)
        d1 = d * sc
        m1 = dmin * m
        sub = qs[:, j * 16:(j + 1) * 16].astype(np.int32)
        low = sub & 0x0F
        high = sub >> 4
        out[:, j * 32:j * 32 + 16] = d1[:, None] * low - m1[:, None]
        out[:, j * 32 + 16:j * 32 + 32] = d1[:, None] * high - m1[:, None]
    return out.reshape(-1)


def dequant_q6_k(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 210
    n_blocks = n_elements // QK_K
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    ql = raw[:, 0:128]
    qh = raw[:, 128:192]
    scales = raw[:, 192:208].copy().view(np.int8).astype(np.float32)
    d = _fp16_col(raw[:, 208:210])
    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for half in range(2):
        ql_h = ql[:, half * 64:(half + 1) * 64].astype(np.int32)
        qh_h = qh[:, half * 32:(half + 1) * 32].astype(np.int32)
        sc_h = scales[:, half * 8:(half + 1) * 8]
        base = half * 128

        q1 = ((ql_h[:, 0:32] & 0x0F) | (((qh_h >> 0) & 3) << 4)) - 32
        q2 = ((ql_h[:, 32:64] & 0x0F) | (((qh_h >> 2) & 3) << 4)) - 32
        q3 = ((ql_h[:, 0:32] >> 4) | (((qh_h >> 4) & 3) << 4)) - 32
        q4 = ((ql_h[:, 32:64] >> 4) | (((qh_h >> 6) & 3) << 4)) - 32

        sc1 = np.repeat(sc_h[:, 0:2], 16, axis=1)
        sc2 = np.repeat(sc_h[:, 2:4], 16, axis=1)
        sc3 = np.repeat(sc_h[:, 4:6], 16, axis=1)
        sc4 = np.repeat(sc_h[:, 6:8], 16, axis=1)

        out[:, base + 0:base + 32] = d[:, None] * sc1 * q1
        out[:, base + 32:base + 64] = d[:, None] * sc2 * q2
        out[:, base + 64:base + 96] = d[:, None] * sc3 * q3
        out[:, base + 96:base + 128] = d[:, None] * sc4 * q4
    return out.reshape(-1)


_KMASK1 = 0x03030303
_KMASK2 = 0x0f0f0f0f


def unpack_q3k_scales(scale_bytes: np.ndarray) -> np.ndarray:
    """scale_bytes: (n_blocks, 12) uint8 -> (n_blocks, 16) int32 signed scale
    values (already offset by -32, i.e. the actual multiplier index). Split out
    from dequant_q3_k so the packing scheme can be unit-tested against
    pack_q3k_scales independently of the rest of the dequant math."""
    aux0 = _u32_col(scale_bytes[:, 0:4])
    aux1 = _u32_col(scale_bytes[:, 4:8])
    tmp = _u32_col(scale_bytes[:, 8:12])

    a2 = ((aux0 >> 4) & _KMASK2) | (((tmp >> 4) & _KMASK1) << 4)
    a3 = ((aux1 >> 4) & _KMASK2) | (((tmp >> 6) & _KMASK1) << 4)
    a0 = (aux0 & _KMASK2) | (((tmp >> 0) & _KMASK1) << 4)
    a1 = (aux1 & _KMASK2) | (((tmp >> 2) & _KMASK1) << 4)

    packed = np.stack([a0, a1, a2, a3], axis=1).astype(np.uint32)  # (n_blocks, 4)
    scales = packed.copy().view(np.uint8).reshape(-1, 16).astype(np.int32)
    return scales - 32


def pack_q3k_scales(scales16: np.ndarray) -> np.ndarray:
    """Inverse of unpack_q3k_scales, for testing only. scales16: (n_blocks, 16)
    int32 in [-32, 31] -> (n_blocks, 12) uint8 packed bytes. Derived by solving
    the unpack bit equations algebraically, not by literally reversing the
    unpack code — an independent-enough construction to catch real bugs in
    either direction if the two disagree."""
    n_blocks = scales16.shape[0]
    b = (scales16 + 32).astype(np.uint8).reshape(n_blocks, 4, 4)  # (n_blocks,4 groups[a0..a3],4 bytes)
    a0, a1, a2, a3 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    aux0 = ((a2 & 0x0F) << 4) | (a0 & 0x0F)
    aux1 = ((a3 & 0x0F) << 4) | (a1 & 0x0F)
    bits0 = (a0 >> 4) & 3
    bits1 = (a1 >> 4) & 3
    bits2 = (a2 >> 4) & 3
    bits3 = (a3 >> 4) & 3
    tmp = bits0 | (bits1 << 2) | (bits2 << 4) | (bits3 << 6)

    out = np.empty((n_blocks, 12), dtype=np.uint8)
    out[:, 0:4] = aux0
    out[:, 4:8] = aux1
    out[:, 8:12] = tmp
    return out


def dequant_q3_k(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 110
    n_blocks = n_elements // QK_K
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    hmask = raw[:, 0:32].astype(np.int32)
    qs = raw[:, 32:96]
    scale_bytes = raw[:, 96:108]
    d = _fp16_col(raw[:, 108:110])

    scales = unpack_q3k_scales(scale_bytes).astype(np.float32)  # (n_blocks, 16)

    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for half in range(2):
        q_half = qs[:, half * 32:(half + 1) * 32].astype(np.int32)
        base = half * 128
        for j in range(4):
            k = half * 4 + j
            m_bit = 1 << k
            shift = 2 * j
            dl0 = d * scales[:, 2 * k]
            dl1 = d * scales[:, 2 * k + 1]

            q_l = q_half[:, 0:16]
            q_l16 = q_half[:, 16:32]
            hm_l = hmask[:, 0:16]
            hm_l16 = hmask[:, 16:32]

            bits_l = (q_l >> shift) & 3
            bits_l16 = (q_l16 >> shift) & 3

            off_l = np.where((hm_l & m_bit) != 0, 0, 4)
            off_l16 = np.where((hm_l16 & m_bit) != 0, 0, 4)

            out[:, base + j * 32: base + j * 32 + 16] = dl0[:, None] * (bits_l - off_l)
            out[:, base + j * 32 + 16: base + j * 32 + 32] = dl1[:, None] * (bits_l16 - off_l16)
    return out.reshape(-1)


def dequant_q5_k(data: bytes, n_elements: int) -> np.ndarray:
    type_size = 176
    n_blocks = n_elements // QK_K
    raw = np.frombuffer(data[:n_blocks * type_size], dtype=np.uint8).reshape(n_blocks, type_size)
    d = _fp16_col(raw[:, 0:2])
    dmin = _fp16_col(raw[:, 2:4])
    scales = raw[:, 4:16]
    qh = raw[:, 16:48].astype(np.int32)
    qs = raw[:, 48:176]

    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    is_ = 0
    u1, u2 = 1, 2
    for j in range(4):
        sc1, m1_ = _scale_min_k4_vec(scales, is_ + 0)
        sc2, m2_ = _scale_min_k4_vec(scales, is_ + 1)
        d1 = d * sc1
        m1v = dmin * m1_
        d2 = d * sc2
        m2v = dmin * m2_

        ql = qs[:, j * 32:(j + 1) * 32].astype(np.int32)

        hi1 = np.where((qh & u1) != 0, 16, 0)
        hi2 = np.where((qh & u2) != 0, 16, 0)

        val1 = (ql & 0x0F) + hi1
        val2 = (ql >> 4) + hi2

        out[:, j * 64: j * 64 + 32] = d1[:, None] * val1 - m1v[:, None]
        out[:, j * 64 + 32: j * 64 + 64] = d2[:, None] * val2 - m2v[:, None]

        is_ += 2
        u1 <<= 2
        u2 <<= 2
    return out.reshape(-1)


_NOT_IMPLEMENTED = {10: "Q2_K"}

_DISPATCH = {
    0: dequant_f32,
    1: dequant_f16,
    2: dequant_q4_0,
    3: dequant_q4_1,
    6: dequant_q5_0,
    7: dequant_q5_1,
    8: dequant_q8_0,
    11: dequant_q3_k,
    12: dequant_q4_k,
    13: dequant_q5_k,
    14: dequant_q6_k,
}


def dequantize(ggml_type: int, data: bytes, n_elements: int) -> np.ndarray:
    if ggml_type in _NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"{_NOT_IMPLEMENTED[ggml_type]} dequant is not implemented — bit-packing "
            f"not confidently known from memory, refusing to guess. See dequant.py "
            f"module docstring for how to close this gap."
        )
    if ggml_type not in _DISPATCH:
        raise ValueError(f"unsupported ggml type id {ggml_type}")
    return _DISPATCH[ggml_type](data, n_elements)
