"""
native_relayout.py — reorder a 2-D quantized tensor's BLOCKS (not individual
elements) into GPU-tile order, keeping the original quant format's bit width
untouched. This is the "keep quants as original" path.

Why block-level instead of element-level (relayout.py's approach): ggml
quantizes each ROW of a matrix independently, splitting it into fixed-size
blocks (32 elements for legacy formats, 256 for K-quants). A block's bytes
are meaningless split apart — the scale factor(s) at the front of a block
apply to every element in it. So tiling has to move whole blocks, never
slice into one.

Because this never decodes a block's contents — it only ever moves fixed-size
byte chunks around — it works on EVERY known ggml block format, including
Q2_K, which this project's dequant.py explicitly refuses to decode. Native
mode doesn't need to know how to interpret a block, only how big it is.

Padding: padding with all-zero bytes is safe for every format this project's
dequant.py implements — a zero scale (and zero min, for K-quant formats with
one) forces dequantized output to exactly 0 regardless of the quantized
value bits. Confirmed in test_gguf2vk.py for a representative sample of
formats. This gives reasonable (not absolute) confidence it holds for Q2_K
too, since Q2_K follows the same "scale(s) stored as leading fp16 fields,
zero scale nullifies the block" pattern as every other K-quant here — but
it hasn't been checked against Q2_K's specific layout the way the tested
formats have.
"""

import numpy as np


def to_tiled_native(raw: bytes, nrows: int, blocks_per_row: int, type_size: int,
                     tile_rows: int, tile_block_cols: int) -> bytes:
    """raw: nrows*blocks_per_row*type_size bytes, in ggml's native per-row
    block order (all of row 0's blocks, then all of row 1's, ...) -> same
    format, reordered so blocks are grouped into (tile_rows x tile_block_cols)
    tiles, row-major tile order, row-major within each tile. Pads with
    all-zero blocks to the next multiple of tile shape."""
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(nrows, blocks_per_row, type_size)

    pad_rows = (-nrows) % tile_rows
    pad_cols = (-blocks_per_row) % tile_block_cols
    if pad_rows or pad_cols:
        arr = np.pad(arr, ((0, pad_rows), (0, pad_cols), (0, 0)), mode="constant")
    padded_rows, padded_cols, _ = arr.shape

    n_tile_rows = padded_rows // tile_rows
    n_tile_cols = padded_cols // tile_block_cols

    reshaped = arr.reshape(n_tile_rows, tile_rows, n_tile_cols, tile_block_cols, type_size)
    tiled = reshaped.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(tiled).reshape(-1).tobytes()


def from_tiled_native(flat: bytes, nrows: int, blocks_per_row: int, type_size: int,
                       tile_rows: int, tile_block_cols: int) -> bytes:
    """Inverse of to_tiled_native — used by the test suite to round-trip-verify
    losslessness (this path has zero quantization error since nothing is
    decoded or re-encoded, only reordered — a round-trip mismatch here would
    mean a real bug, not rounding noise)."""
    pad_rows = (-nrows) % tile_rows
    pad_cols = (-blocks_per_row) % tile_block_cols
    padded_rows, padded_cols = nrows + pad_rows, blocks_per_row + pad_cols
    n_tile_rows = padded_rows // tile_rows
    n_tile_cols = padded_cols // tile_block_cols

    arr = np.frombuffer(flat, dtype=np.uint8).reshape(
        n_tile_rows, n_tile_cols, tile_rows, tile_block_cols, type_size)
    matrix = arr.transpose(0, 2, 1, 3, 4).reshape(padded_rows, padded_cols, type_size)
    return matrix[:nrows, :blocks_per_row, :].tobytes()
