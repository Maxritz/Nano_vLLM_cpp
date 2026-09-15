"""
gguf_reader.py — GGUF binary format parser, written from scratch.

No dependency on llama.cpp, gguf-py, or any existing GGUF library.
Implements the GGUF v3 spec directly against the byte stream:

    [magic:u32][version:u32][tensor_count:u64][kv_count:u64]
    [kv pairs...]
    [tensor infos...]
    [padding to alignment]
    [tensor data blob]

Reference: GGUF is little-endian throughout.
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum


GGUF_MAGIC = 0x46554747  # ASCII "GGUF" read as LE u32


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# ggml tensor dtypes we care about for now — extend as needed.
class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ4_NL = 20
    # not exhaustive — unknown types are kept as raw int and flagged


_SCALAR_FMT = {
    GGUFValueType.UINT8: ("B", 1),
    GGUFValueType.INT8: ("b", 1),
    GGUFValueType.UINT16: ("H", 2),
    GGUFValueType.INT16: ("h", 2),
    GGUFValueType.UINT32: ("I", 4),
    GGUFValueType.INT32: ("i", 4),
    GGUFValueType.FLOAT32: ("f", 4),
    GGUFValueType.UINT64: ("Q", 8),
    GGUFValueType.INT64: ("q", 8),
    GGUFValueType.FLOAT64: ("d", 8),
}


class ByteCursor:
    """Minimal manual cursor over a bytes buffer — no struct-unpacking library beyond `struct`."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def read(self, n: int) -> bytes:
        out = self.buf[self.pos:self.pos + n]
        if len(out) != n:
            raise EOFError(f"expected {n} bytes at offset {self.pos}, got {len(out)}")
        self.pos += n
        return out

    def read_scalar(self, vtype: GGUFValueType):
        fmt, size = _SCALAR_FMT[vtype]
        return struct.unpack("<" + fmt, self.read(size))[0]

    def read_bool(self) -> bool:
        return self.read(1)[0] != 0

    def read_string(self) -> str:
        length = struct.unpack("<Q", self.read(8))[0]
        raw = self.read(length)
        return raw.decode("utf-8", errors="replace")

    def read_value(self, vtype: GGUFValueType):
        if vtype == GGUFValueType.STRING:
            return self.read_string()
        if vtype == GGUFValueType.BOOL:
            return self.read_bool()
        if vtype == GGUFValueType.ARRAY:
            elem_type = GGUFValueType(struct.unpack("<I", self.read(4))[0])
            length = struct.unpack("<Q", self.read(8))[0]
            return [self.read_value(elem_type) for _ in range(length)]
        if vtype in _SCALAR_FMT:
            return self.read_scalar(vtype)
        raise ValueError(f"unknown GGUF value type {vtype}")


@dataclass
class TensorInfo:
    name: str
    shape: list      # dims, slowest-varying first per GGUF convention (n_dims entries)
    ggml_type: int
    offset: int       # byte offset into tensor data blob


@dataclass
class GGUFFile:
    version: int
    metadata: dict = field(default_factory=dict)
    tensors: list = field(default_factory=list)   # list[TensorInfo]
    tensor_data_start: int = 0
    raw: bytes = b""

    def get(self, key: str, default=None):
        return self.metadata.get(key, default)

    def tensor_bytes(self, info: "TensorInfo") -> bytes:
        """Slice raw tensor bytes for one tensor. Caller must know the type's element/block size
        to know how many bytes to expect; this just gives the offset start — end is resolved by
        the caller against the next tensor's offset or EOF."""
        start = self.tensor_data_start + info.offset
        return self.raw[start:]


def parse_gguf(path: str) -> GGUFFile:
    with open(path, "rb") as f:
        buf = f.read()

    c = ByteCursor(buf)

    magic = struct.unpack("<I", c.read(4))[0]
    if magic != GGUF_MAGIC:
        raise ValueError(f"not a GGUF file (bad magic {magic:#010x})")

    version = struct.unpack("<I", c.read(4))[0]
    tensor_count = struct.unpack("<Q", c.read(8))[0]
    kv_count = struct.unpack("<Q", c.read(8))[0]

    metadata = {}
    for _ in range(kv_count):
        key = c.read_string()
        vtype = GGUFValueType(struct.unpack("<I", c.read(4))[0])
        metadata[key] = c.read_value(vtype)

    tensors = []
    for _ in range(tensor_count):
        name = c.read_string()
        n_dims = struct.unpack("<I", c.read(4))[0]
        shape = [struct.unpack("<Q", c.read(8))[0] for _ in range(n_dims)]
        ggml_type = struct.unpack("<I", c.read(4))[0]
        offset = struct.unpack("<Q", c.read(8))[0]
        tensors.append(TensorInfo(name=name, shape=shape, ggml_type=ggml_type, offset=offset))

    alignment = metadata.get("general.alignment", 32)
    pad = (-c.pos) % alignment
    c.pos += pad
    tensor_data_start = c.pos

    return GGUFFile(
        version=version,
        metadata=metadata,
        tensors=tensors,
        tensor_data_start=tensor_data_start,
        raw=buf,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python gguf_reader.py <model.gguf>")
        sys.exit(1)
    g = parse_gguf(sys.argv[1])
    print(f"GGUF v{g.version}, {len(g.tensors)} tensors, {len(g.metadata)} metadata keys")
    print(f"architecture: {g.get('general.architecture')}")
    for t in g.tensors[:10]:
        print(f"  {t.name:40s} shape={t.shape} type={t.ggml_type} offset={t.offset}")
