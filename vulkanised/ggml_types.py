"""
ggml_types.py — block size / byte size table for ggml tensor dtypes.

Needed to know how many bytes a tensor occupies in the file (block_count * type_size),
since GGUF tensor info only gives shape + type id, not byte length.

Struct layouts taken from ggml's public block_qN_K definitions (ggml-quants.h,
MIT licensed, widely mirrored). Legacy (QK=32) formats are simple and unit-tested
here. K-quant (QK_K=256) formats have packed 6-bit sub-block scale/min schemes that
are easy to get subtly wrong without a reference implementation to diff against —
treat the K-quant dequant math in dequant.py as unverified until checked against
llama.cpp's own dump output on a real tensor.
"""

from collections import namedtuple

TypeInfo = namedtuple("TypeInfo", ["name", "block_size", "type_size", "confidence"])

# confidence: "verified" (unit-tested against hand-crafted vectors),
#             "structural" (byte layout known, dequant math unverified against reference)
GGML_TYPES = {
    0:  TypeInfo("F32",    1,   4, "verified"),
    1:  TypeInfo("F16",    1,   2, "verified"),
    2:  TypeInfo("Q4_0",  32,  18, "verified"),
    3:  TypeInfo("Q4_1",  32,  20, "verified"),
    6:  TypeInfo("Q5_0",  32,  22, "verified"),
    7:  TypeInfo("Q5_1",  32,  24, "verified"),
    8:  TypeInfo("Q8_0",  32,  34, "verified"),
    10: TypeInfo("Q2_K", 256,  84, "not_implemented"),
    11: TypeInfo("Q3_K", 256, 110, "best_effort"),
    12: TypeInfo("Q4_K", 256, 144, "best_effort"),
    13: TypeInfo("Q5_K", 256, 176, "best_effort"),
    14: TypeInfo("Q6_K", 256, 210, "best_effort"),
}


def tensor_element_count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def tensor_byte_size(shape, ggml_type: int) -> int:
    """Total bytes a tensor occupies given its shape and ggml type id."""
    if ggml_type not in GGML_TYPES:
        raise ValueError(f"unknown/unsupported ggml type id {ggml_type}")
    info = GGML_TYPES[ggml_type]
    n_elements = tensor_element_count(shape)
    if n_elements % info.block_size != 0:
        raise ValueError(
            f"tensor element count {n_elements} not divisible by block size "
            f"{info.block_size} for type {info.name} — malformed tensor or wrong type id"
        )
    n_blocks = n_elements // info.block_size
    return n_blocks * info.type_size
