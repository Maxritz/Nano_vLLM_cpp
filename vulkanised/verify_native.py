"""
verify_native.py — confirms a --format native converted file is byte-exact
against its source, tensor by tensor. Since native mode never decodes
anything (only moves whole blocks), this is a hard byte-equality check, not
a tolerance comparison — any mismatch is a real bug.

Usage:
    python3 verify_native.py original.gguf converted.gguf

Reads tile_rows/tile_block_cols back from the converted file's own
vulkan_native.* metadata rather than assuming defaults, so this works
regardless of what --tile-rows/--tile-block-cols the conversion actually used.
"""

import sys
from gguf_reader import parse_gguf
from ggml_types import GGML_TYPES, tensor_byte_size
import native_relayout


def verify(original_path: str, converted_path: str) -> bool:
    orig = parse_gguf(original_path)
    conv = parse_gguf(converted_path)

    if conv.get("vulkan_native.layout") != "tiled":
        print(f"'{converted_path}' has no vulkan_native.layout=tiled metadata — "
              f"not a native-mode output file, nothing to verify here.")
        return False
    if conv.get("vulkan_native.format") != "native":
        print(f"'{converted_path}' was converted with format="
              f"'{conv.get('vulkan_native.format')}', not 'native' — this script "
              f"only makes sense for native-mode output (q8_0/q4_0 outputs have "
              f"real quantization error by design, not a byte-exact match to check).")
        return False

    tile_rows = conv.get("vulkan_native.tile_rows")
    tile_block_cols = conv.get("vulkan_native.tile_block_cols")
    print(f"Verifying against tile_rows={tile_rows}, tile_block_cols={tile_block_cols}\n")

    orig_tensors = {t.name: t for t in orig.tensors}
    conv_tensors = {t.name: t for t in conv.tensors}

    missing = set(orig_tensors) - set(conv_tensors)
    extra = set(conv_tensors) - set(orig_tensors)
    if missing:
        print(f"MISMATCH: {len(missing)} tensor(s) in original but missing from converted: "
              f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        print(f"MISMATCH: {len(extra)} tensor(s) in converted but not in original: "
              f"{sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")

    all_ok = not missing and not extra
    checked = 0

    for name, ot in orig_tensors.items():
        if name not in conv_tensors:
            continue
        ct = conv_tensors[name]

        if ot.ggml_type != ct.ggml_type:
            print(f"  MISMATCH {name}: ggml_type changed ({ot.ggml_type} -> {ct.ggml_type}) "
                  f"— native mode should never change the quant format")
            all_ok = False
            continue
        if list(ot.shape) != list(ct.shape):
            print(f"  MISMATCH {name}: shape changed ({ot.shape} -> {ct.shape})")
            all_ok = False
            continue

        o_start = orig.tensor_data_start + ot.offset
        o_size = tensor_byte_size(ot.shape, ot.ggml_type)
        o_raw = orig.raw[o_start:o_start + o_size]

        c_start = conv.tensor_data_start + ct.offset
        # the converted tensor may be padded larger than the original due to
        # tiling — its actual stored size is recomputed from the padded shape
        # below, not assumed equal to the original size.

        if len(ot.shape) == 1:
            c_raw = conv.raw[c_start:c_start + o_size]
            match = c_raw == o_raw
        elif len(ot.shape) == 2:
            info = GGML_TYPES[ot.ggml_type]
            ncols, nrows = ot.shape[0], ot.shape[1]
            blocks_per_row = ncols // info.block_size
            pad_rows = (-nrows) % tile_rows
            pad_cols = (-blocks_per_row) % tile_block_cols
            padded_size = (nrows + pad_rows) * (blocks_per_row + pad_cols) * info.type_size
            c_raw_padded = conv.raw[c_start:c_start + padded_size]
            untiled = native_relayout.from_tiled_native(
                c_raw_padded, nrows, blocks_per_row, info.type_size, tile_rows, tile_block_cols
            )
            match = untiled == o_raw
        else:
            print(f"  SKIP {name}: {len(ot.shape)}-D tensor, verifier only handles 1-D/2-D")
            continue

        checked += 1
        if not match:
            print(f"  MISMATCH {name}: untiled bytes do not match original")
            all_ok = False

    print(f"\n{checked} tensors checked.")
    if all_ok:
        print("PASS — every tensor untiles back to byte-exact original data.")
    else:
        print("FAIL — see mismatches above. Do not trust this converted file.")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python3 verify_native.py <original.gguf> <converted.gguf>")
        sys.exit(1)
    ok = verify(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
