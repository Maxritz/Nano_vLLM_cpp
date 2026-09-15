"""
relayout.py — reorder a 2-D weight matrix from plain row-major into tiles,
so the Vulkan GEMM shader can read a tile with coalesced, subgroup-aligned
access instead of striding across the full row-major matrix.

Layout convention defined here (pick once, then the shader's read pattern
must match exactly — this is the contract between converter and shader):

  - Matrix split into tiles of shape (tile_rows, tile_cols).
  - Tiles emitted in row-major tile order (left-to-right, top-to-bottom).
  - Within each tile, elements are row-major.
  - Ragged edge tiles (matrix dims not evenly divisible by tile shape) are
    zero-padded — caller must track the padded shape separately from the
    logical shape if it matters downstream.

This is *a* reasonable convention, not *the* one true layout — there's no
single correct tiling without a specific shader to match against. Change it
freely, just keep converter and shader in lockstep.
"""

import numpy as np


def _slow_to_tiled(matrix: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    """(rows, cols) row-major fp32 matrix -> 1-D array in tiled order.
    Pads with zeros to the next multiple of tile shape."""
    rows, cols = matrix.shape
    pad_rows = (-rows) % tile_rows
    pad_cols = (-cols) % tile_cols
    if pad_rows or pad_cols:
        matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode="constant")
    padded_rows, padded_cols = matrix.shape

    n_tile_rows = padded_rows // tile_rows
    n_tile_cols = padded_cols // tile_cols

    out = np.empty(padded_rows * padded_cols, dtype=matrix.dtype)
    pos = 0
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile = matrix[tr * tile_rows:(tr + 1) * tile_rows,
                           tc * tile_cols:(tc + 1) * tile_cols]
            out[pos:pos + tile.size] = tile.reshape(-1)
            pos += tile.size
    return out


def _slow_from_tiled(flat: np.ndarray, rows: int, cols: int, tile_rows: int, tile_cols: int) -> np.ndarray:
    """Inverse of to_tiled — used by the test suite to round-trip-verify the
    relayout is lossless and reversible before trusting it on real weights."""
    pad_rows = (-rows) % tile_rows
    pad_cols = (-cols) % tile_cols
    padded_rows, padded_cols = rows + pad_rows, cols + pad_cols
    n_tile_rows = padded_rows // tile_rows
    n_tile_cols = padded_cols // tile_cols

    matrix = np.empty((padded_rows, padded_cols), dtype=flat.dtype)
    pos = 0
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile = flat[pos:pos + tile_rows * tile_cols].reshape(tile_rows, tile_cols)
            matrix[tr * tile_rows:(tr + 1) * tile_rows,
                   tc * tile_cols:(tc + 1) * tile_cols] = tile
            pos += tile_rows * tile_cols
    return matrix[:rows, :cols]
