"""
writer.py — write a GGUF-shell container for the converted tensors.

Important: this reuses GGUF's outer binary format (magic/version/KV/tensor-info
table/aligned data blob) purely because it's a well-defined, already-parseable
shell — NOT because the output is standard-GGUF-compatible. Tensor data here is
tile-relayouted (relayout.py) and requantized to block-32 (requant.py), so a
stock llama.cpp/gguf-py reader will load it with the wrong element order and
produce garbage. This is a container of convenience for a Vulkan-native loader
(VAiSt's own), not a drop-in GGUF replacement — the custom
`vulkan_native.*` metadata keys mark it as such so nothing downstream mistakes
it for the real thing.
"""

import struct

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3


def _w_string(buf: bytearray, s: str):
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


# GGUFValueType ids, mirrored from gguf_reader.py
T_UINT32 = 4
T_STRING = 8


def _w_kv_uint32(buf: bytearray, key: str, value: int):
    _w_string(buf, key)
    buf += struct.pack("<I", T_UINT32)
    buf += struct.pack("<I", value)


def _w_kv_string(buf: bytearray, key: str, value: str):
    _w_string(buf, key)
    buf += struct.pack("<I", T_STRING)
    _w_string(buf, value)


def write_gguf(path: str, arch: str, tile_rows: int, tile_cols: int,
               source_note: str, tensors: list, format_name: str = "q8_0",
               tile_block_cols: int = 1):
    """
    tensors: list of dicts, each with:
        name: str
        shape: list[int]   (logical, pre-padding, slowest-varying first)
        ggml_type: int      (matches the chosen requant.FORMATS entry for
                             q8_0/q4_0 modes, or the ORIGINAL tensor's type
                             unchanged for native mode)
        data: bytes         (already tiled — element-tiled+requantized for
                             q8_0/q4_0, block-tiled only for native)
    """
    header = bytearray()
    header += struct.pack("<I", GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensors))

    kv = bytearray()
    n_kv = 0
    _w_kv_string(kv, "general.architecture", arch); n_kv += 1
    _w_kv_string(kv, "vulkan_native.layout", "tiled"); n_kv += 1
    _w_kv_uint32(kv, "vulkan_native.tile_rows", tile_rows); n_kv += 1
    _w_kv_uint32(kv, "vulkan_native.tile_cols", tile_cols); n_kv += 1
    _w_kv_uint32(kv, "vulkan_native.tile_block_cols", tile_block_cols); n_kv += 1
    _w_kv_string(kv, "vulkan_native.source", source_note); n_kv += 1
    _w_kv_string(kv, "vulkan_native.format", format_name); n_kv += 1
    _w_kv_uint32(kv, "general.alignment", 32); n_kv += 1

    header += struct.pack("<Q", n_kv)
    header += kv

    tensor_info = bytearray()
    data_blob = bytearray()
    offset = 0
    for t in tensors:
        _w_string(tensor_info, t["name"])
        tensor_info += struct.pack("<I", len(t["shape"]))
        for d in t["shape"]:
            tensor_info += struct.pack("<Q", d)
        tensor_info += struct.pack("<I", t["ggml_type"])
        tensor_info += struct.pack("<Q", offset)
        data_blob += t["data"]
        offset += len(t["data"])
        # 32-byte align each tensor's start within the blob
        pad = (-offset) % 32
        if pad:
            data_blob += b"\x00" * pad
            offset += pad

    header += tensor_info

    pad = (-len(header)) % 32
    header += b"\x00" * pad

    with open(path, "wb") as f:
        f.write(header)
        f.write(data_blob)
