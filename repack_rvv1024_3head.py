#!/usr/bin/env python3
"""Repack Qwen2.5-Coder-1.5B Q4_0 GGUF attention weights for RVV1024.

Standard library only. Reads GGUF v2/v3 directly; never modifies the input.
Creates one Q, KV, and W_O packed binary per layer plus a JSON manifest.

LAYOUT v1, little-endian, NO file headers or padding:
  Q: 4 tiles (Q heads [0:3], [3:6], [6:9], [9:12]).
     Each tile contains 48 input blocks (32 columns each). For each block:
       for k = 0..31: three 64-byte streams, one per Q head.
       In each stream byte j holds the nibble of that head's output j
       (low 4 bits) and output j+64 (high 4 bits).
       Followed by 384 float32 scales ordered by output row.
     Block 7680 B, group 368640 B, Q file 1474560 B.

  KV: 2 tiles (KV heads 0 and 1). Each tile contains 48 input blocks.
     Each block: 32 * 128 B, where each byte for k,j has K[j] in
     low nibble and V[j] in high nibble. Then 256 float32 scales:
     K[0:128] then V[0:128].
     Block 5120 B, group 245760 B, KV file 491520 B.

  W_O: 4 input-head groups (384 input columns each); each group contains
     6 output-row subtiles of 256 rows. Each subtile has 12 input blocks.
     Each block: 32 * 128 B, low nibble = 128 lower output rows,
     high nibble = 128 upper output rows. Then 256 float32 scales.
     Block 5120 B, subtile 61440 B, group 368640 B, W_O file 1474560 B.

Nibble values are preserved exactly (0..15, with Q4_0 zero point 8).
Only FP16 scales are converted losslessly to FP32. Q/K/V biases are not
repacked; they stay in the original GGUF. The current C++ RVV kernel reads
ordinary Q4_0, so a later kernel patch is REQUIRED to consume these files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import random
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

EMBD = 1536
HEAD = 128
Q_HEADS = 12
KV_HEADS = 2
Q_GROUPS = 4
Q_GROUP_ROWS = 384
Q4_BLOCK = 32
Q4_BYTES = 18
GGML_Q4_0 = 2

Q_PACKED_COL = 3 * 64  # 3 heads, 2x64 nibbles per head
KV_PACKED_COL = 128  # low nibble K, high nibble V
O_PACKED_COL = 128   # low/high 128 rows
Q_BLOCK_BYTES = 32 * Q_PACKED_COL + 384 * 4      # 7680
KV_BLOCK_BYTES = 32 * KV_PACKED_COL + 256 * 4   # 5120
O_BLOCK_BYTES = KV_BLOCK_BYTES                   # 5120
Q_GROUP_BYTES = 48 * Q_BLOCK_BYTES              # 368640
KV_GROUP_BYTES = 48 * KV_BLOCK_BYTES            # 245760
O_SUBTILE_BYTES = 12 * O_BLOCK_BYTES            # 61440
O_GROUP_BYTES = 6 * O_SUBTILE_BYTES             # 368640
Q_FILE_BYTES = 4 * Q_GROUP_BYTES                # 1474560
KV_FILE_BYTES = 2 * KV_GROUP_BYTES              # 491520
O_FILE_BYTES = 4 * O_GROUP_BYTES                # 1474560

META_FIXED_WIDTH = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                    7: 1, 10: 8, 11: 8, 12: 8}
META_FORMAT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
               6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def checked_read(f, n: int) -> bytes:
    if n < 0 or n > 1024 * 1024 * 32:
        raise ValueError(f'bad read size {n}')
    data = f.read(n)
    if len(data) != n:
        raise ValueError('GGUF header truncated')
    return data


def u32(f) -> int:
    return struct.unpack('<I', checked_read(f, 4))[0]


def u64(f) -> int:
    return struct.unpack('<Q', checked_read(f, 8))[0]


def gguf_string(f) -> str:
    size = u64(f)
    if size > (1 << 24):
        raise ValueError(f'GGUF string too large: {size}')
    return checked_read(f, size).decode('utf-8')


def skip_metadata_value(f, t: int) -> None:
    if t in META_FIXED_WIDTH:
        f.seek(META_FIXED_WIDTH[t], 1)
    elif t == 8:
        length = u64(f)
        f.seek(length, 1)
    elif t == 9:
        elt = u32(f)
        n = u64(f)
        if elt in META_FIXED_WIDTH:
            f.seek(n * META_FIXED_WIDTH[elt], 1)
        elif elt == 8:
            for _ in range(n):
                f.seek(u64(f), 1)
        else:
            raise ValueError(f'unsupported GGUF array element type {elt}')
    else:
        raise ValueError(f'unsupported GGUF metadata type {t}')


def metadata_value(f, t: int):
    if t in META_FORMAT:
        return struct.unpack('<' + META_FORMAT[t], checked_read(f, META_FIXED_WIDTH[t]))[0]
    if t == 8:
        return gguf_string(f)
    skip_metadata_value(f, t)
    return None


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    dtype: int
    offset: int  # absolute file offset


class GGUFIndex:
    def __init__(self, filename: Path):
        self.filename = filename
        self.tensors: dict[str, TensorInfo] = {}
        self.meta = {}
        self.file_size = filename.stat().st_size
        with filename.open('rb') as f:
            if checked_read(f, 4) != b'GGUF':
                raise ValueError('input is not a GGUF file')
            self.version = u32(f)
            if self.version not in (2, 3):
                raise ValueError(f'only GGUF v2/v3 supported, got v{self.version}')
            n_tensors = u64(f)
            n_meta = u64(f)
            if n_tensors > 1_000_000 or n_meta > 1_000_000:
                raise ValueError('unreasonable GGUF index counts')
            wanted = {'general.architecture', 'general.alignment',
                      'qwen2.block_count', 'qwen2.embedding_length',
                      'qwen2.attention.head_count', 'qwen2.attention.head_count_kv'}
            for _ in range(n_meta):
                key = gguf_string(f)
                t = u32(f)
                if key in wanted:
                    self.meta[key] = metadata_value(f, t)
                else:
                    skip_metadata_value(f, t)
            alignment = self.meta.get('general.alignment', 32)
            if not isinstance(alignment, int) or alignment < 1 or alignment & (alignment - 1):
                raise ValueError(f'invalid GGUF tensor data alignment: {alignment}')
            self.alignment = alignment
            pending = []
            for _ in range(n_tensors):
                name = gguf_string(f)
                ndims = u32(f)
                if ndims < 1 or ndims > 4:
                    raise ValueError(f'invalid dimensions for {name}: {ndims}')
                dims = tuple(u64(f) for _ in range(ndims))
                dtype = u32(f)
                relative_offset = u64(f)
                if name in self.tensors:
                    raise ValueError(f'duplicate tensor: {name}')
                pending.append((name, dims, dtype, relative_offset))
            self.data_start = (f.tell() + alignment - 1) & -alignment
            if self.data_start > self.file_size:
                raise ValueError('GGUF data section begins after end of file')
            for name, dims, dtype, rel in pending:
                self.tensors[name] = TensorInfo(name, dims, dtype, self.data_start + rel)

    def quant_tensor(self, name: str, dims: tuple[int, int]) -> TensorInfo:
        if name not in self.tensors:
            raise ValueError(f'missing tensor {name}; check model is Qwen2 with standard attention names')
        t = self.tensors[name]
        if t.dtype != GGML_Q4_0 or t.dims != dims:
            raise ValueError(f'{name}: need Q4_0 shape {dims}; found dtype={t.dtype}, shape={t.dims}')
        if dims[0] % Q4_BLOCK:
            raise ValueError(f'{name}: dimension {dims[0]} not divisible by 32')
        size = dims[0] * dims[1] // 32 * Q4_BYTES
        if t.offset + size > self.file_size:
            raise ValueError(f'{name}: tensor exceeds GGUF file boundary')
        return t


class Q4Matrix:
    def __init__(self, buf: mmap.mmap, tensor: TensorInfo):
        self.buf = buf
        self.tensor = tensor
        self.row_stride = (tensor.dims[0] // 32) * 18

    def block_base(self, row: int, block: int) -> int:
        assert 0 <= row < self.tensor.dims[1]
        assert 0 <= block < self.tensor.dims[0] // 32
        return self.tensor.offset + row * self.row_stride + block * 18

    def nibble(self, row: int, col: int) -> int:
        base = self.block_base(row, col // 32)
        value = self.buf[base + 2 + (col % 16)]
        return (value & 15) if col % 32 < 16 else (value >> 4)

    def scale(self, row: int, block: int) -> float:
        return struct.unpack_from('<e', self.buf, self.block_base(row, block))[0]


@dataclass(frozen=True)
class NibblePair:
    low: Q4Matrix
    low_start: int
    high: Q4Matrix
    high_start: int
    width: int


def pack_block(out, pairs: tuple[NibblePair, ...], input_block: int) -> None:
    """Nibble-pair streams followed by FP32 scales in logical row order."""
    lanes = sum(pair.width for pair in pairs)
    total_rows = 2 * lanes
    buf = bytearray(32 * lanes + 4 * total_rows)
    paired_bases = []
    for pair in pairs:
        paired_bases.append((
            [pair.low.block_base(pair.low_start + j, input_block) for j in range(pair.width)],
            [pair.high.block_base(pair.high_start + j, input_block) for j in range(pair.width)],
        ))
    at = 0
    for k in range(32):
        offset = 2 + (k & 15)
        low_half = (k < 16)
        for pair, (lb, hb) in zip(pairs, paired_bases):
            src_low = pair.low.buf
            src_high = pair.high.buf
            for j in range(pair.width):
                lo = src_low[lb[j] + offset]
                hi = src_high[hb[j] + offset]
                if low_half:
                    buf[at] = (lo & 15) | ((hi & 15) << 4)
                else:
                    buf[at] = (lo >> 4) | (hi & 0xF0)
                at += 1
    assert at == 32 * lanes
    # Scales in first-head-low, first-head-high, next-head-low, ... order.
    for pair in pairs:
        for matrix, start in ((pair.low, pair.low_start), (pair.high, pair.high_start)):
            for j in range(pair.width):
                scale = matrix.scale(start + j, input_block)
                struct.pack_into('<f', buf, at, scale)
                at += 4
    assert at == len(buf)
    out.write(buf)


def atomic_write(path: Path, expected_bytes: int, write_fn) -> str:
    temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('wb') as f:
            write_fn(f)
            if f.tell() != expected_bytes:
                raise ValueError(f'packed {path.name} is {f.tell()} bytes, expected {expected_bytes}')
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def make_q(f, q: Q4Matrix) -> None:
    for group in range(Q_GROUPS):
        base = group * Q_GROUP_ROWS
        pairs = tuple(NibblePair(q, base + head * 128,
                                 q, base + head * 128 + 64, 64)
                      for head in range(3))
        for input_block in range(48):
            pack_block(f, pairs, input_block)


def make_kv(f, k: Q4Matrix, v: Q4Matrix) -> None:
    for kvhead in range(KV_HEADS):
        pairs = (NibblePair(k, kvhead * HEAD, v, kvhead * HEAD, HEAD),)
        for input_block in range(48):
            pack_block(f, pairs, input_block)


def make_wo(f, wo: Q4Matrix) -> None:
    for group in range(Q_GROUPS):
        for out_tile in range(6):
            out_base = out_tile * 256
            pairs = (NibblePair(wo, out_base, wo, out_base + 128, 128),)
            for in_block in range(12):
                pack_block(f, pairs, group * 12 + in_block)


def verify_sample_file(path: Path, layout: str, matrices, samples: int,
                       seed: int) -> None:
    """Independently decode packed coordinates and compare with original GGUF."""
    rng = random.Random(seed)
    with path.open('rb') as f:
        if layout == 'q':
            w = matrices['q']
            for _ in range(samples):
                g = rng.randrange(4)
                b = rng.randrange(48)
                k = rng.randrange(32)
                h = rng.randrange(3)
                lane = rng.randrange(128)
                row = g * 384 + h * 128 + lane
                off = g * Q_GROUP_BYTES + b * Q_BLOCK_BYTES
                idx = k * 192 + h * 64 + lane % 64
                f.seek(off + idx)
                byte = f.read(1)[0]
                value = (byte & 15) if lane < 64 else (byte >> 4)
                original = w.nibble(row, b * 32 + k)
                f.seek(off + 6144 + (h * 128 + lane) * 4)
                scale = struct.unpack('<f', f.read(4))[0]
                if value != original or scale != w.scale(row, b):
                    raise AssertionError(f'Q verification failed: group={g}, row={row}, b={b}, k={k}')
        elif layout == 'kv':
            for _ in range(samples):
                g = rng.randrange(2)
                b = rng.randrange(48)
                k = rng.randrange(32)
                lane = rng.randrange(128)
                which = rng.randrange(2)
                w = matrices['k'] if which == 0 else matrices['v']
                row = g * 128 + lane
                off = g * KV_GROUP_BYTES + b * KV_BLOCK_BYTES
                f.seek(off + k * 128 + lane)
                byte = f.read(1)[0]
                value = (byte & 15) if which == 0 else (byte >> 4)
                f.seek(off + 4096 + (which * 128 + lane) * 4)
                scale = struct.unpack('<f', f.read(4))[0]
                if value != w.nibble(row, b * 32 + k) or scale != w.scale(row, b):
                    raise AssertionError(f'KV verification failed: kvhead={g}, kind={which}, row={row}')
        elif layout == 'wo':
            w = matrices['wo']
            for _ in range(samples):
                g = rng.randrange(4)
                tile = rng.randrange(6)
                b = rng.randrange(12)
                k = rng.randrange(32)
                lane = rng.randrange(256)
                row = tile * 256 + lane
                col = g * 384 + b * 32 + k
                off = g * O_GROUP_BYTES + tile * O_SUBTILE_BYTES + b * O_BLOCK_BYTES
                f.seek(off + k * 128 + lane % 128)
                byte = f.read(1)[0]
                value = (byte & 15) if lane < 128 else (byte >> 4)
                f.seek(off + 4096 + lane * 4)
                scale = struct.unpack('<f', f.read(4))[0]
                if value != w.nibble(row, col) or scale != w.scale(row, col // 32):
                    raise AssertionError(f'W_O verification failed: group={g}, row={row}, col={col}')
        else:
            raise AssertionError(layout)


def parse_layers(index: GGUFIndex, selector: str) -> list[int]:
    regex = re.compile(r'^blk\.(\d+)\.attn_q\.weight$')
    found = sorted({int(m.group(1)) for name in index.tensors if (m := regex.match(name))})
    if not found:
        raise ValueError('no blk.<layer>.attn_q.weight found; unsupported GGUF tensor naming')
    if selector == 'all':
        return found
    requested = sorted({int(p) for p in selector.split(',') if p.strip()})
    if not requested:
        raise ValueError('--layers must be all or comma-separated indices')
    missing = sorted(set(requested) - set(found))
    if missing:
        raise ValueError(f'layers missing from model: {missing}; available {found}')
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='original Qwen2.5-Coder-1.5B Q4_0 GGUF')
    parser.add_argument('--out-dir', type=Path, help='output directory (defaults beside GGUF)')
    parser.add_argument('--layers', default='all', help='all (default) or comma-separated indices, e.g. 0,1')
    parser.add_argument('--inspect', action='store_true', help='list relevant tensors and exit without writing')
    parser.add_argument('--verify', type=int, default=128, help='random nibble+FP32-scale checks per file (default 128)')
    args = parser.parse_args()
    if args.verify < 0:
        parser.error('--verify must be >= 0')
    model = args.model.expanduser().resolve()
    if not model.is_file():
        parser.error(f'not found: {model}')
    index = GGUFIndex(model)
    layers = parse_layers(index, args.layers)
    if index.meta.get('general.architecture', 'qwen2') != 'qwen2':
        raise ValueError(f"expected GGUF architecture qwen2, got {index.meta.get('general.architecture')!r}")
    for key, want in [('qwen2.embedding_length', 1536),
                      ('qwen2.attention.head_count', 12),
                      ('qwen2.attention.head_count_kv', 2)]:
        got = index.meta.get(key)
        if got is not None and got != want:
            raise ValueError(f'{key} expected {want}, model has {got}')
    print(f'Model: {model}\nGGUF: v{index.version}, {len(index.tensors)} tensors, {index.alignment}-byte alignment')
    print(f'Layers: {layers[0]}..{layers[-1]} ({len(layers)} selected)')
    names = []
    for layer in layers:
        prefix = f'blk.{layer}.'
        names.append((
            index.quant_tensor(prefix + 'attn_q.weight', (1536, 1536)),
            index.quant_tensor(prefix + 'attn_k.weight', (1536, 256)),
            index.quant_tensor(prefix + 'attn_v.weight', (1536, 256)),
            index.quant_tensor(prefix + 'attn_output.weight', (1536, 1536)),
        ))
    if args.inspect:
        for layer, tensors in zip(layers, names):
            for t in tensors:
                print(f'  layer {layer:2d}: {t.name:36s} Q4_0 {t.dims} offset={t.offset}')
        print('All selected weights are compatible with the v1 repacker.')
        return
    outdir = (args.out_dir or model.parent / (model.stem + '-rvv1024-colsoa')).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        'format': 'rvv1024-qwen2-q4_0-3head-colsoa-v1',
        'input': str(model),
        'input_bytes': index.file_size,
        'gguf_version': index.version,
        'note': 'Raw sidecars, NOT GGUF. C++ kernel must be adapted before loading. No biases.',
        'dimensions': {'hidden': 1536, 'q_heads': 12, 'kv_heads': 2, 'head_dim': 128,
                       'q_heads_per_group': 3, 'input_q4_block': 32},
        'tile_bytes': {
            'q_block': Q_BLOCK_BYTES, 'q_group': Q_GROUP_BYTES,
            'kv_block': KV_BLOCK_BYTES, 'kv_group': KV_GROUP_BYTES,
            'wo_block': O_BLOCK_BYTES, 'wo_output_subtile': O_SUBTILE_BYTES,
            'wo_input_group': O_GROUP_BYTES,
        },
        'files_per_layer': {'q': Q_FILE_BYTES, 'kv': KV_FILE_BYTES, 'wo': O_FILE_BYTES},
        'packing': {
            'q': 'group[4], input_block[48], k[32], head[3], lane[64]: low=Qhead[0..63], high=Qhead[64..127]; FP32 scales Qrows[384]',
            'kv': 'kv_head[2], input_block[48], k[32], lane[128]: low=K, high=V; FP32 scales K[128],V[128]',
            'wo': 'q_group[4], output_subtile[6] (256 rows), input_block[12], k[32], lane[128]: low=output_rows[0..127], high=output_rows[128..255]; FP32 scales rows[256]',
            'raw_nibble_value': '0..15; signed Q4_0 value = nibble - 8',
            'scale_conversion': 'FP16 to FP32 lossless',
        },
        'layers': [],
    }
    with model.open('rb') as mf:
        with mmap.mmap(mf.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for layer, (qt, kt, vt, ot) in zip(layers, names):
                matrices = {'q': Q4Matrix(mm, qt), 'k': Q4Matrix(mm, kt),
                            'v': Q4Matrix(mm, vt), 'wo': Q4Matrix(mm, ot)}
                prefix = f'layer_{layer:03d}'
                paths = {'q': outdir / f'{prefix}.q384.bin',
                         'kv': outdir / f'{prefix}.kv256.bin',
                         'wo': outdir / f'{prefix}.wo384x256.bin'}
                print(f'Layer {layer}: Q(4x384), KV(2x128+128), W_O(4x384 input) ...', flush=True)
                digests = {
                    'q': atomic_write(paths['q'], Q_FILE_BYTES,
                                      lambda f: make_q(f, matrices['q'])),
                    'kv': atomic_write(paths['kv'], KV_FILE_BYTES,
                                       lambda f: make_kv(f, matrices['k'], matrices['v'])),
                    'wo': atomic_write(paths['wo'], O_FILE_BYTES,
                                       lambda f: make_wo(f, matrices['wo'])),
                }
                if args.verify:
                    for i, kind in enumerate(('q', 'kv', 'wo')):
                        verify_sample_file(paths[kind], kind, matrices,
                                           args.verify, seed=20261117 + layer * 13 + i)
                files = {k: {'name': p.name, 'bytes': p.stat().st_size,
                             'sha256': digests[k]} for k, p in paths.items()}
                manifest['layers'].append({'layer': layer, 'files': files,
                    'source_tensors': {'q': qt.name, 'k': kt.name, 'v': vt.name, 'wo': ot.name}})
                print('  OK; sampled verification passed' if args.verify else '  OK', flush=True)
    manifestpath = outdir / 'manifest.json'
    manifestpath.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(layers)} layers to {outdir}')
    print(f'Manifest: {manifestpath}')
    print('Note: the current RVV fused kernel still expects ORIGINAL GGUF Q4_0 weights.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, AssertionError, OSError) as e:
        print(f'error: {e}', file=sys.stderr)
        sys.exit(1)
