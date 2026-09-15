#!/usr/bin/env python3
"""Regenerates include/nanovllm/spv.h from compiled .spv files."""
import struct, os, sys

def main():
    if len(sys.argv) < 3:
        print("usage: gen_spv_header.py <spv_dir> <output_h>")
        sys.exit(1)
    spv_dir = sys.argv[1]
    out_path = sys.argv[2]
    names = sorted([f[:-4] for f in os.listdir(spv_dir) if f.endswith('.spv')])
    out = '#pragma once\n#include <cstddef>\n/* Auto-generated: embedded SPIR-V for the Vulkan backend. */\n'
    sizes = {}
    for name in names:
        path = os.path.join(spv_dir, name + '.spv')
        with open(path, 'rb') as fh:
            data = fh.read()
        words = list(struct.unpack('<%dI' % (len(data) // 4), data))
        sizes[name] = len(words)
        out += f'static const unsigned int {name}_spv[] = {{\n'
        for i in range(0, len(words), 8):
            chunk = words[i:i+8]
            vals = [f'0x{w:08x}' for w in chunk]
            out += '  ' + ', '.join(vals) + ',\n'
        out += '};\n'
    out += 'static const unsigned int spv_count(const char* n) {\n  if (!n) return 0;\n'
    for name in names:
        out += f'  if (strcmp(n, "{name}")==0) return {sizes[name]};\n'
    out += '  return 0;\n}\n'
    out += 'static const unsigned int* spv_for(const char* n) {\n  if (!n) return 0;\n'
    for name in names:
        out += f'  if (strcmp(n, "{name}")==0) return {name}_spv;\n'
    out += '  return 0;\n}\n'
    with open(out_path, 'w') as f:
        f.write(out)
    print(f'Generated {out_path} with {len(names)} shaders')

if __name__ == '__main__':
    main()
