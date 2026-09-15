"""
relayout.py — reorder a 2-D weight matrix from plain row-major into tiles,
vectorized via reshape+transpose instead of a Python loop over every tile
(the old version looped n_tile_rows * n_tile_cols times — 100K+ iterations
on a single 5120-wide matrix, the second big bottleneck after requant's
per-block loop). The original loop-based version is preserved in
relayout_ref.py as `_slow_to_tiled`/`_slow_from_tiled` and used as a
diff-test oracle.

Layout convention (unchanged from before — this is the converter/shader
contract, pick once and keep both sides in lockstep):
  - Matrix split into tiles of shape (tile_rows, tile_cols).
  - Tiles emitted in row-major tile order (left-to-right, top-to-bottom).
  - Within each tile, elements are row-major.
  - Ragged edges (matrix dims not evenly divisible by tile shape) are
    zero-padded.
"""

import numpy as np


def to_tiled(matrix: np.ndarray, tile_rows: int, tile_cols: int) -> np.ndarray:
    """(rows, cols) row-major array -> 1-D array in tiled order. Pads with
    zeros to the next multiple of tile shape first."""
    rows, cols = matrix.shape
    pad_rows = (-rows) % tile_rows
    pad_cols = (-cols) % tile_cols
    if pad_rows or pad_cols:
        matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode="constant")
    padded_rows, padded_cols = matrix.shape
    n_tile_rows = padded_rows // tile_rows
    n_tile_cols = padded_cols // tile_cols

    # Splitting the row axis into (n_tile_rows, tile_rows) groups rows
    # r=0..tile_rows-1 under tile-row 0, tile_rows..2*tile_rows-1 under tile-row 1,
    # etc — exactly the same grouping the old loop's matrix[tr*tile_rows:...] did.
    # Same for columns. The transpose then reorders axes so tile index (tr, tc)
    # is outer and in-tile index (r, c) is inner, matching "tiles in row-major
    # tile order, row-major within each tile."
    reshaped = matrix.reshape(n_tile_rows, tile_rows, n_tile_cols, tile_cols)
    tiled = reshaped.transpose(0, 2, 1, 3)
    return np.ascontiguousarray(tiled).reshape(-1)


def from_tiled(flat: np.ndarray, rows: int, cols: int, tile_rows: int, tile_cols: int) -> np.ndarray:
    """Inverse of to_tiled — used by the test suite to round-trip-verify the
    relayout is lossless and reversible before trusting it on real weights."""
    pad_rows = (-rows) % tile_rows
    pad_cols = (-cols) % tile_cols
    padded_rows, padded_cols = rows + pad_rows, cols + pad_cols
    n_tile_rows = padded_rows // tile_rows
    n_tile_cols = padded_cols // tile_cols

    tiled = flat.reshape(n_tile_rows, n_tile_cols, tile_rows, tile_cols)
    matrix = tiled.transpose(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return matrix[:rows, :cols]
