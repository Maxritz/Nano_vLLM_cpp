"""
convert.py — orchestrates the full pipeline:

  GGUF in -> compat check -> per-tensor [dequant -> relayout -> requant]
  OR [native block relayout] -> GGUF-shell out

Usage:
    python3 convert.py <input.gguf> <output.gguf> [options]

Format choice (--format):
    q8_0 (default) — dequant -> tile at element level -> requant to Q8_0.
                      8.5 bits/element, trivial shader unpack, ~2.1x the size
                      of a typical Q3_K_M source file.
    q4_0            — same pipeline, requant to Q4_0 instead. 4.5 bits/element,
                      close to original file size, slightly more shader work.
    native          — NO dequant/requant at all. Reorders each tensor's
                      existing quantized BLOCKS into GPU tiles, keeping the
                      original bit width and quant format exactly as shipped.
                      Zero size penalty, zero added quantization error.
                      Works on every known block format INCLUDING Q2_K, since
                      nothing is decoded — only whole blocks are moved. The
                      tradeoff moves to the shader side: reading a tiled
                      Q3_K/Q4_K/etc. block still means implementing that
                      format's native unpack math in the shader, same
                      complexity dequant.py went through here.

Run the same input multiple times with different output filenames to compare
formats directly.

Tensors using a dequant format not yet implemented (Q2_K, for q8_0/q4_0 modes
only — native mode has no such gap) are reported and skipped, not silently
dropped.
"""

import argparse
import sys
import numpy as np

from gguf_reader import parse_gguf
from compat_check import check_compat, print_report
from ggml_types import tensor_byte_size, tensor_element_count, GGML_TYPES
import dequant
import requant
import relayout
import native_relayout
import writer


def convert_tensor(g, t, tile_rows, tile_cols, fmt):
    """Returns (name, shape, ggml_type, data_bytes) on success, or raises."""
    encode, _, out_ggml_type, _ = requant.FORMATS[fmt]

    n_elements = tensor_element_count(t.shape)
    byte_size = tensor_byte_size(t.shape, t.ggml_type)
    start = g.tensor_data_start + t.offset
    raw = g.raw[start:start + byte_size]

    flat = dequant.dequantize(t.ggml_type, raw, n_elements)

    if len(t.shape) == 2:
        # ggml stores ne[0] as fastest-varying -> matches numpy's last axis
        matrix = flat.reshape(t.shape[::-1])
        tiled_flat = relayout.to_tiled(matrix, tile_rows, tile_cols)
        packed = encode(tiled_flat)
    elif len(t.shape) == 1:
        packed = encode(flat)
    else:
        raise NotImplementedError(f"tensor '{t.name}' has {len(t.shape)} dims — only 1-D/2-D handled")

    return t.name, list(t.shape), out_ggml_type, packed


def convert_tensor_native(g, t, tile_rows, tile_block_cols):
    """Returns (name, shape, ggml_type=t.ggml_type unchanged, data_bytes).
    Never dequantizes — just moves whole blocks. Works on any format with a
    known block_size/type_size in ggml_types.py, including Q2_K."""
    if t.ggml_type not in GGML_TYPES:
        raise ValueError(f"unknown ggml type id {t.ggml_type} — no block size on record")

    info = GGML_TYPES[t.ggml_type]
    byte_size = tensor_byte_size(t.shape, t.ggml_type)
    start = g.tensor_data_start + t.offset
    raw = g.raw[start:start + byte_size]

    if len(t.shape) == 2:
        ncols, nrows = t.shape[0], t.shape[1]  # ne[0]=fastest=cols, ne[1]=rows
        if ncols % info.block_size != 0:
            raise ValueError(
                f"tensor '{t.name}': ncols ({ncols}) not divisible by "
                f"{info.name}'s block size ({info.block_size})"
            )
        blocks_per_row = ncols // info.block_size
        tiled = native_relayout.to_tiled_native(
            raw, nrows, blocks_per_row, info.type_size, tile_rows, tile_block_cols
        )
        return t.name, list(t.shape), t.ggml_type, tiled
    elif len(t.shape) == 1:
        return t.name, list(t.shape), t.ggml_type, raw  # 1-D: pass through unchanged
    else:
        raise NotImplementedError(f"tensor '{t.name}' has {len(t.shape)} dims — only 1-D/2-D handled")


def run(input_path, output_path, tile_rows, tile_cols, fmt="q8_0", tile_block_cols=1, force=False):
    if fmt != "native" and fmt not in requant.FORMATS:
        print(f"Unknown format '{fmt}' — choices are {list(requant.FORMATS.keys()) + ['native']}")
        return 1

    g = parse_gguf(input_path)
    result = check_compat(g)
    print_report(result)
    if not result.ok and not force:
        print("\nRefusing to convert — architecture doesn't fit the spec (see FAIL reasons above).")
        print("Pass --force to convert anyway (e.g. for testing the pipeline on a toy model).")
        return 1

    converted, skipped = [], []
    for t in g.tensors:
        try:
            if fmt == "native":
                name, shape, ggml_type, data = convert_tensor_native(g, t, tile_rows, tile_block_cols)
                label = f"{GGML_TYPES[t.ggml_type].name} -> {GGML_TYPES[t.ggml_type].name} (retiled only)"
            else:
                name, shape, ggml_type, data = convert_tensor(g, t, tile_rows, tile_cols, fmt)
                label = f"{GGML_TYPES[t.ggml_type].name} -> {fmt.upper()}"
            converted.append({"name": name, "shape": shape, "ggml_type": ggml_type, "data": data})
            print(f"  OK      {t.name}  ({label}, {len(data)} bytes)")
        except NotImplementedError as e:
            skipped.append((t.name, str(e)))
            print(f"  SKIPPED {t.name}  — {e}")
        except Exception as e:
            skipped.append((t.name, str(e)))
            print(f"  ERROR   {t.name}  — {e}")

    writer.write_gguf(
        output_path,
        arch=g.get("general.architecture", "unknown"),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile_block_cols=tile_block_cols,
        source_note=input_path,
        tensors=converted,
        format_name=fmt,
    )

    print(f"\n{len(converted)}/{len(g.tensors)} tensors converted, {len(skipped)} skipped.")
    if fmt == "native":
        print("Format: native (original quant formats preserved, blocks retiled only)")
    else:
        _, _, _, out_bytes_per_block = requant.FORMATS[fmt]
        print(f"Format: {fmt} ({out_bytes_per_block} bytes / 32 elements = {out_bytes_per_block*8/32:.1f} bits/element)")
    print(f"Written: {output_path}")
    if skipped:
        print("Skipped tensors mean the output file is INCOMPLETE — not safe to load until closed.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--tile-rows", type=int, default=32)
    ap.add_argument("--tile-cols", type=int, default=8,
                     help="tile width in ELEMENTS — only used by q8_0/q4_0 modes")
    ap.add_argument("--tile-block-cols", type=int, default=1,
                     help="tile width in BLOCKS — only used by native mode")
    ap.add_argument("--format", choices=["q8_0", "q4_0", "native"], default="q8_0",
                     help="q8_0/q4_0: dequant+requant to that format. "
                          "native: keep original quant format, retile blocks only")
    ap.add_argument("--force", action="store_true", help="convert even if compat check fails")
    args = ap.parse_args()
    sys.exit(run(args.input, args.output, args.tile_rows, args.tile_cols,
                 args.format, args.tile_block_cols, args.force))
