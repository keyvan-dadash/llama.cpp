#!/usr/bin/env python3
"""Write ONE complete Qwen2.5-Coder-1.5B GGUF with RVV1024 packed tensors.

Original GGUF tensors and metadata are copied byte-for-byte. Three extra GGML_TYPE_I8
1D tensors are appended PER LAYER (the raw packed data is not a normal I8 matrix):
  blk.{layer}.attn_q.rvv1024       1,474,560 bytes, 4 x 384-output Q SoA
  blk.{layer}.attn_k.rvv1024         491,520 bytes, 2 x (K/V interleaved)
  blk.{layer}.attn_output.rvv1024 1,474,560 bytes, 4 x W_O SoA
The unmodified matrices remain in the GGUF for standard GGML inference.

Run with the *matching llama.cpp loader patch*, which teaches the model loader
that tensors with the rvv1024 suffix are opaque packed byte arrays. This GGUF
is not a standard GGUF repacking mode supported by upstream llama.cpp.

Entirely standalone; no third-party Python packages. Never modifies the source GGUF.
"""
from __future__ import annotations

import argparse
import hashlib
import mmap
import os
import re
import shutil
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




# Reuse the verified layout/packing functions above as an internal namespace.
soa = sys.modules[__name__]

GGML_TYPE_I8 = 24
FORMAT_VERSION = 1
FORMAT_NAME = 'qwen2-q4_0-rvv1024-3head-colsoa-v1'
KIND = {
    'q': ('attn_q.rvv1024', soa.Q_FILE_BYTES, soa.make_q),
    'kv': ('attn_k.rvv1024', soa.KV_FILE_BYTES, soa.make_kv),
    'wo': ('attn_output.rvv1024', soa.O_FILE_BYTES, soa.make_wo),
}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def pack_u64(value: int) -> bytes:
    return struct.pack('<Q', value)


def str_gguf(s: str) -> bytes:
    raw = s.encode('utf-8')
    return pack_u64(len(raw)) + raw


def metadata_string(key: str, value: str) -> bytes:
    return str_gguf(key) + struct.pack('<I', 8) + str_gguf(value)


def metadata_u32(key: str, value: int) -> bytes:
    return str_gguf(key) + struct.pack('<II', 4, value)


def tensor_desc(name: str, nbytes: int, offset: int) -> bytes:
    # GGUF tensor: name, n_dims, dims[], ggml_type, relative offset.
    return str_gguf(name) + struct.pack('<IQIQ', 1, nbytes, GGML_TYPE_I8, offset)


@dataclass(frozen=True)
class SourceGGUF:
    version: int
    tensors_count: int
    kv_count: int
    metadata: bytes
    tensor_table: bytes
    data_offset: int


def source_sections(path: Path) -> SourceGGUF:
    with path.open('rb') as file:
        if file.read(4) != b'GGUF':
            raise ValueError('not a GGUF file')
        version, n_tensors, n_kv = struct.unpack('<IQQ', soa.checked_read(file, 20))
        if version not in (2, 3) or n_tensors > 1_000_000 or n_kv > 1_000_000:
            raise ValueError('unsupported GGUF version or malformed metadata counts')
        meta_start = file.tell()
        for _ in range(n_kv):
            soa.gguf_string(file)
            soa.skip_metadata_value(file, soa.u32(file))
        meta_end = file.tell()
        for _ in range(n_tensors):
            soa.gguf_string(file)
            ndims = soa.u32(file)
            if not 1 <= ndims <= 4:
                raise ValueError('invalid dimension count')
            soa.checked_read(file, 8 * ndims + 4 + 8)
        table_end = file.tell()
        file.seek(meta_start)
        metadata = soa.checked_read(file, meta_end - meta_start)
        tensor_table = soa.checked_read(file, table_end - meta_end)
    return SourceGGUF(version, n_tensors, n_kv, metadata, tensor_table, 0)


def validate_source(index: soa.GGUFIndex) -> tuple[list[int], list[dict[str, soa.TensorInfo]]]:
    if index.version not in (2, 3):
        raise ValueError('requires GGUF v2/v3')
    if index.meta.get('general.architecture') != 'qwen2':
        raise ValueError('GGUF must have general.architecture=qwen2')
    for key, want in [
        ('qwen2.embedding_length', 1536),
        ('qwen2.attention.head_count', 12),
        ('qwen2.attention.head_count_kv', 2),
    ]:
        if index.meta.get(key) != want:
            raise ValueError(f'{key}: expected {want}, got {index.meta.get(key)!r}')
    layers = soa.parse_layers(index, 'all')
    n_layer_meta = index.meta.get('qwen2.block_count')
    if n_layer_meta != len(layers) or layers != list(range(n_layer_meta)):
        raise ValueError('all model layers must be present in this single GGUF')
    if n_layer_meta != 28:
        raise ValueError(f'expected Qwen2.5-Coder-1.5B (28 layers), got {n_layer_meta}')
    if any(name.endswith('.rvv1024') for name in index.tensors):
        raise ValueError('input already contains rvv1024 tensors; use the original GGUF')
    sources = []
    for layer in layers:
        pre = f'blk.{layer}.'
        sources.append({
            'q':  index.quant_tensor(pre + 'attn_q.weight', (1536, 1536)),
            'k':  index.quant_tensor(pre + 'attn_k.weight', (1536, 256)),
            'v':  index.quant_tensor(pre + 'attn_v.weight', (1536, 256)),
            'wo': index.quant_tensor(pre + 'attn_output.weight', (1536, 1536)),
        })
    return layers, sources


def build_append_info(layers: list[int], source_data_length: int, alignment: int):
    items = []
    offset = align_up(source_data_length, alignment)
    for layer in layers:
        for kind in ('q', 'kv', 'wo'):
            suffix, size, _ = KIND[kind]
            name = f'blk.{layer}.{suffix}'
            items.append((layer, kind, name, offset, size))
            offset = align_up(offset + size, alignment)
    return items


def verify_one(mm_source: mmap.mmap, mm_output: mmap.mmap,
               src_info: dict[str, soa.TensorInfo],
               output_index: soa.GGUFIndex, layer: int, samples: int) -> None:
    import random
    rng = random.Random(20261117 + layer * 31)
    m = {key: soa.Q4Matrix(mm_source, val) for key, val in src_info.items()}
    for kind in ('q', 'kv', 'wo'):
        name = f'blk.{layer}.{KIND[kind][0]}'
        p = output_index.tensors[name].offset
        for _ in range(samples):
            if kind == 'q':
                group = rng.randrange(4)
                b = rng.randrange(48)
                k = rng.randrange(32)
                head = rng.randrange(3)
                lane = rng.randrange(128)
                row = group * 384 + head * 128 + lane
                at = p + group * soa.Q_GROUP_BYTES + b * soa.Q_BLOCK_BYTES
                packed = mm_output[at + k * 192 + head * 64 + lane % 64]
                got = packed & 15 if lane < 64 else packed >> 4
                want = m['q'].nibble(row, b * 32 + k)
                got_scale = struct.unpack_from('<f', mm_output, at + 6144 + (head * 128 + lane)*4)[0]
                want_scale = m['q'].scale(row, b)
            elif kind == 'kv':
                group = rng.randrange(2)
                b = rng.randrange(48)
                k = rng.randrange(32)
                lane = rng.randrange(128)
                which = rng.randrange(2)
                row = group * 128 + lane
                at = p + group * soa.KV_GROUP_BYTES + b * soa.KV_BLOCK_BYTES
                packed = mm_output[at + k * 128 + lane]
                got = packed & 15 if which == 0 else packed >> 4
                name_k = 'k' if which == 0 else 'v'
                want = m[name_k].nibble(row, b * 32 + k)
                got_scale = struct.unpack_from('<f', mm_output, at + 4096 + (which*128 + lane)*4)[0]
                want_scale = m[name_k].scale(row, b)
            else:
                group = rng.randrange(4)
                tile = rng.randrange(6)
                b = rng.randrange(12)
                k = rng.randrange(32)
                lane = rng.randrange(256)
                row = tile * 256 + lane
                input_col = group * 384 + b * 32 + k
                at = p + group * soa.O_GROUP_BYTES + tile * soa.O_SUBTILE_BYTES + b * soa.O_BLOCK_BYTES
                packed = mm_output[at + k * 128 + lane % 128]
                got = packed & 15 if lane < 128 else packed >> 4
                want = m['wo'].nibble(row, input_col)
                got_scale = struct.unpack_from('<f', mm_output, at + 4096 + lane*4)[0]
                want_scale = m['wo'].scale(row, input_col // 32)
            if got != want or got_scale != want_scale:
                raise ValueError(f'layer {layer} {kind}: packed nibble/FP32 scale mismatch')


def repack(model: Path, output: Path, sample_checks: int, force: bool) -> None:
    if model == output:
        raise ValueError('input and output paths must differ')
    if output.exists() and not force:
        raise ValueError(f'{output} exists; use --force to overwrite')
    index = soa.GGUFIndex(model)
    layers, all_sources = validate_source(index)
    source = source_sections(model)
    if source.tensors_count != len(index.tensors):
        raise ValueError('source GGUF tensor index changed while reading')
    original_data_length = model.stat().st_size - index.data_start
    if original_data_length <= 0:
        raise ValueError('source GGUF contains no tensor data')
    # Regular GGUF files pad the last tensor to alignment. Never infer offsets
    # from tensors read in memory: retain the source's original byte offsets.
    additions = build_append_info(layers, original_data_length, index.alignment)
    append_table = b''.join(tensor_desc(name, nbytes, offset)
                            for _, _, name, offset, nbytes in additions)
    append_meta = (metadata_string('rvv1024.layout', FORMAT_NAME)
                   + metadata_u32('rvv1024.layout_version', FORMAT_VERSION))
    tmp = output.with_name(output.name + '.tmp')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with model.open('rb') as src, tmp.open('wb') as dst:
            dst.write(b'GGUF')
            dst.write(struct.pack('<IQQ', source.version, source.tensors_count + len(additions),
                                  source.kv_count + 2))
            dst.write(source.metadata)
            dst.write(append_meta)
            dst.write(source.tensor_table)
            dst.write(append_table)
            dst.write(b'\x00' * ((-dst.tell()) % index.alignment))
            dst_data_start = dst.tell()
            src.seek(index.data_start)
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
            # A malformed source file with trailing bytes is rejected by the
            # independent GGUF reader during final output verification.
            with model.open('rb') as inf, mmap.mmap(inf.fileno(), 0, access=mmap.ACCESS_READ) as source_mm:
                for layer, kind, name, offset, nbytes in additions:
                    desired = dst_data_start + offset
                    if dst.tell() < desired:
                        dst.write(b'\x00' * (desired - dst.tell()))
                    if dst.tell() != desired:
                        raise ValueError(f'bad output offset for {name}: {dst.tell()} vs {desired}')
                    info = all_sources[layer]
                    matrices = {k: soa.Q4Matrix(source_mm, tensor) for k, tensor in info.items()}
                    print(f'Packing layer {layer:02d} / 27: {kind.upper():2s} ({nbytes:,} bytes)', flush=True)
                    before = dst.tell()
                    if kind == 'q':
                        soa.make_q(dst, matrices['q'])
                    elif kind == 'kv':
                        soa.make_kv(dst, matrices['k'], matrices['v'])
                    else:
                        soa.make_wo(dst, matrices['wo'])
                    if dst.tell() - before != nbytes:
                        raise ValueError(f'{name}: output size mismatch')
                    dst.write(b'\x00' * ((-dst.tell()) % index.alignment))
        # Parse the result exactly like a normal GGUF and validate *every*
        # appended tensor's index, data type, shape and relative offset.
        outindex = soa.GGUFIndex(tmp)
        expected_count = source.tensors_count + len(additions)
        if len(outindex.tensors) != expected_count:
            raise ValueError(f'new GGUF has {len(outindex.tensors)} tensors, expected {expected_count}')
        for layer, kind, name, offset, nbytes in additions:
            ti = outindex.tensors[name]
            if ti.dtype != GGML_TYPE_I8 or ti.dims != (nbytes,) or ti.offset != outindex.data_start + offset:
                raise ValueError(f'{name}: tensor descriptor does not match RVV packed ABI')
        # Confirm original tensors and tensor payloads are identical across files.
        with model.open('rb') as orig, tmp.open('rb') as final:
            orig.seek(index.data_start)
            final.seek(outindex.data_start)
            left = original_data_length
            while left:
                n = min(left, 8*1024*1024)
                if orig.read(n) != final.read(n):
                    raise ValueError('an original GGUF tensor byte was modified')
                left -= n
        if sample_checks:
            with model.open('rb') as orig, tmp.open('rb') as final:
                with mmap.mmap(orig.fileno(), 0, access=mmap.ACCESS_READ) as source_mm, \
                     mmap.mmap(final.fileno(), 0, access=mmap.ACCESS_READ) as final_mm:
                    for layer in layers:
                        verify_one(source_mm, final_mm, all_sources[layer], outindex, layer, sample_checks)
            print(f'PASS: {sample_checks} random value+scale checks per tensor per layer')
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f'Wrote complete GGUF: {output} ({output.stat().st_size:,} bytes)')
    print(f'Original tensors preserved: {source.tensors_count}; added packed tensors: {len(additions)}')
    print('Load this GGUF with the companion llama.cpp packed-tensor loader patch.')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, type=Path, help='original Qwen2.5-Coder-1.5B GGUF')
    ap.add_argument('--output', required=True, type=Path, help='new complete GGUF path')
    ap.add_argument('--verify', type=int, default=256, metavar='N', help='sample checks per packed tensor/layer')
    ap.add_argument('--force', action='store_true', help='replace existing output GGUF')
    args = ap.parse_args()
    if args.verify < 0:
        ap.error('--verify must be nonnegative')
    try:
        repack(args.model.expanduser().resolve(), args.output.expanduser().resolve(), args.verify, args.force)
    except (OSError, AssertionError, ValueError, struct.error) as exc:
        ap.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    main()
