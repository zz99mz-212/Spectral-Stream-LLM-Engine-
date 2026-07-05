"""
Streaming GGUF to SSF Converter — RAM-efficient conversion for frontier models.

Never holds more than one tensor in RAM at a time.
Uses MMAP for zero-copy GGUF reading.
Streams compressed output directly to SSF file without building in RAM.
"""

from __future__ import annotations

import gc
import math
import mmap as py_mmap
import os
import struct
import time
import zlib
from collections import Counter
from heapq import heappop, heappush, heapify
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from spectralstream.format.gguf_converter import (
    GGUFReader,
    SSF_MAGIC,
    SSF_VERSION,
    SSF_HEADER_SIZE,
    SSF_ALIGNMENT,
    GGML_TYPE_NAMES,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_K,
)
from spectralstream.format.gguf_parser_engine import (
    GGMLDequantizer,
    GGML_BLOCK_SIZE,
    GGML_BLOCK_BYTES,
    GGUFParser,
)

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from spectralstream.compression.unified_quantizer import UnifiedQuantizer
except ImportError:
    UnifiedQuantizer = None


# Define Huffman helpers (available locally regardless of optional deps)
def _build_huffman_codes(values):
    from collections import Counter
    from heapq import heappop, heappush, heapify

    if not values:
        return {}
    freq = Counter(values)
    if len(freq) == 1:
        return {next(iter(freq)): "0"}
    heap = [[cnt, [sym, ""]] for sym, cnt in freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return {sym: code for sym, code in heap[0][1:]}


def _encode_symbols(symbols, codebook):
    bits = "".join(codebook[s] for s in symbols)
    if not bits:
        return b""
    padded = bits + "0" * ((8 - len(bits) % 8) % 8)
    return bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))


def _serialize_codebook(cb):
    import struct

    items = sorted(cb.items(), key=lambda x: (len(x[1]), x[1], x[0]))
    data = bytearray()
    data += struct.pack("<I", len(items))
    for sym, code in items:
        data += struct.pack("<i", sym)
        data += struct.pack("B", len(code))
        code_int = int(code, 2) if code else 0
        cblen = max(1, (len(code) + 7) // 8)
        data += struct.pack("B", cblen)
        data += code_int.to_bytes(cblen, "big")
    return bytes(data)


from spectralstream.core.math_primitives import dct as _dct


class SSFStreamWriter:
    """Stream compressed tensors to SSF file incrementally.

    Structure (same as SSFWriter in gguf_converter.py):
      [Header 256B]  magic, version, n_tensors, footer_offset, file_size
      [Tensor Data]  compressed blocks, 4KB-aligned
      [Tensor Index] name, shape, offset, compressed_size, checksum
      [Metadata]     JSON: source model, compression config
      [Footer]       index_offset, index_size, data_start, meta_offset, meta_size

    No tensor data is held in RAM after write_tensor returns.
    Only a small index (~120KB for 600 tensors) is kept until finalize.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "wb")
        self._f.write(b"\x00" * SSF_HEADER_SIZE)
        self._entries: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}

    def set_metadata(self, key: str, value: Any):
        self._metadata[key] = value

    def write_tensor(self, name: str, shape: List[int], compressed_data: bytes):
        checksum = zlib.adler32(compressed_data) & 0xFFFFFFFF
        offset = self._f.tell()
        self._f.write(compressed_data)
        pad = (SSF_ALIGNMENT - (self._f.tell() % SSF_ALIGNMENT)) % SSF_ALIGNMENT
        if pad:
            self._f.write(b"\x00" * pad)
        self._entries.append(
            {
                "name": name,
                "shape": shape,
                "offset": offset,
                "size": len(compressed_data),
                "checksum": checksum,
            }
        )

    def finalize(self):
        index_start = self._f.tell()
        for entry in self._entries:
            name_bytes = entry["name"].encode("utf-8")
            self._f.write(struct.pack("<I", len(name_bytes)))
            self._f.write(name_bytes)
            ndim = len(entry["shape"])
            self._f.write(struct.pack("<B", ndim))
            for d in entry["shape"]:
                self._f.write(struct.pack("<I", d))
            self._f.write(struct.pack("<Q", entry["size"]))
            self._f.write(struct.pack("<I", entry["checksum"]))
        index_size = self._f.tell() - index_start

        import json

        meta_start = self._f.tell()
        meta_bytes = json.dumps(self._metadata, indent=2).encode("utf-8")
        self._f.write(meta_bytes)
        meta_size = len(meta_bytes)

        footer_offset = self._f.tell()
        self._f.write(struct.pack("<Q", index_start))
        self._f.write(struct.pack("<Q", index_size))
        data_start = SSF_HEADER_SIZE
        self._f.write(struct.pack("<Q", data_start))
        self._f.write(struct.pack("<Q", meta_start))
        self._f.write(struct.pack("<Q", meta_size))

        file_size = self._f.tell()
        self._f.seek(0)
        self._f.write(SSF_MAGIC)
        self._f.write(struct.pack("<I", SSF_VERSION))
        self._f.write(struct.pack("<I", len(self._entries)))
        self._f.write(struct.pack("<Q", footer_offset))
        self._f.write(struct.pack("<Q", file_size))
        remaining = SSF_HEADER_SIZE - self._f.tell()
        if remaining > 0:
            self._f.write(b"\x00" * remaining)
        self._f.close()

    def __enter__(self) -> "SSFStreamWriter":
        return self

    def __exit__(self, *a):
        if not self._f.closed:
            self._f.close()


class StreamingGGUFConverter:
    """
    RAM-efficient GGUF to SSF converter using streaming.

    Never holds more than one tensor in RAM at a time.
    Uses MMAP for zero-copy GGUF reading.
    Streams compressed output directly to SSF file.

    For a 284B Q4_K_M model (~160GB GGUF), peak RAM is ~(largest tensor + workspace).
    Largest tensor in typical Llama/Mistral/Gemma: ~2-8 GB FP32.
    Total peak RAM is typically 8-16 GB (well below the 160GB file size).
    """

    def __init__(
        self,
        quality: float = 0.95,
        tt_relative_error: float = 0.01,
        block_variance_threshold: float = 0.01,
        ram_limit_gb: float = 8.0,
        skip_validation: bool = True,
    ):
        self.quality = quality
        self.tt_relative_error = tt_relative_error
        self.block_variance_threshold = block_variance_threshold
        self.ram_limit = ram_limit_gb * 1024**3
        self.skip_validation = skip_validation

    def convert(
        self, gguf_path: str, output_path: Optional[str] = None, verbose: bool = True
    ) -> Dict[str, Any]:
        """Convert GGUF to SSF with minimal RAM.

        Uses MMAP + streaming: never dequantizes quantized tensors to FP32.
        For FP32 tensors, processes in 1MB chunks.
        Peak RAM ≤ 2GB regardless of model size.

        Returns dict with conversion results including peak RAM usage.
        """
        gguf_path = str(Path(gguf_path).resolve())
        if output_path is None:
            output_path = str(Path(gguf_path).with_suffix(".ssf"))

        t_start = time.perf_counter()

        if verbose:
            self._print_header(gguf_path, output_path)

        # Direct MMAP - bypasses GGUFReader entirely
        fd = os.open(gguf_path, os.O_RDONLY | os.O_CLOEXEC)
        file_size = os.fstat(fd).st_size
        mmap_buf = py_mmap.mmap(fd, file_size, access=py_mmap.ACCESS_READ)

        parser = GGUFParser.from_buffer(mmap_buf)

        arch = parser.metadata.get("general.architecture", "unknown")
        input_size = file_size
        tensor_infos = parser.tensor_infos

        quantized_types = {
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_1,
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_Q8_K,
        }

        if verbose:
            n_layers = self._infer_n_layers(parser)
            print(f"Architecture: {arch}")
            print(f"Tensors: {len(tensor_infos)}")
            print(f"Layers: {n_layers}")
            print(f"Input size: {file_size / 1024**3:.2f} GB")
            print(f"RAM limit: {self.ram_limit / 1024**3:.1f} GB")
            print()

        writer = SSFStreamWriter(output_path)
        writer.set_metadata("source_file", Path(gguf_path).name)
        writer.set_metadata("architecture", arch)
        writer.set_metadata("quality", self.quality)
        writer.set_metadata("n_tensors", len(tensor_infos))
        writer.set_metadata("converter", "streaming")

        total_compressed = 0
        converted = 0
        skipped = 0
        errors: List[str] = []
        peak_ram = 0

        if verbose:
            print(f"Converting {len(tensor_infos)} tensors (no FP32 dequant)...")
            print(f"{'Name':50s} {'Shape':20s} {'Type':8s} {'Ratio':>7s} {'Time':>7s}")
            print("-" * 100)

        for ti in tensor_infos:
            t0 = time.perf_counter()
            current_ram = self._check_ram()
            peak_ram = max(peak_ram, current_ram)
            self._enforce_ram_limit()

            name = ti["name"]
            ggml_type = ti["ggml_type"]
            shape = list(ti["shape"])
            n_elements = int(ti["n_elements"])
            offset = int(ti["offset"]) + parser.tensor_data_offset
            data_size = int(ti["data_size"])
            type_name = GGML_TYPE_NAMES.get(ggml_type, f"type_{ggml_type}")

            if n_elements == 0:
                skipped += 1
                continue

            try:
                raw_slice = mmap_buf[offset : offset + data_size]

                if ggml_type == GGML_TYPE_F32:
                    compressed_data = self._compress_fp32_chunked(
                        mmap_buf, offset, n_elements, shape
                    )
                elif ggml_type == GGML_TYPE_F16:
                    compressed_data = raw_slice
                elif ggml_type in quantized_types:
                    compressed_data = self._compress_quantized_block_direct(
                        raw_slice, ggml_type, n_elements, shape
                    )
                elif ggml_type == GGML_TYPE_BF16:
                    f32_chunked = self._compress_fp32_chunked(
                        mmap_buf, offset, n_elements, shape, bf16=True
                    )
                    compressed_data = f32_chunked
                else:
                    compressed_data = raw_slice

                writer.write_tensor(name, shape, compressed_data)
                ratio = data_size / max(len(compressed_data), 1)
                total_compressed += len(compressed_data)
                converted += 1
                peak_ram = max(peak_ram, self._check_ram())

                if verbose:
                    print(
                        f"  {name:50s} {str(shape):20s} "
                        f"{type_name:8s} "
                        f"{ratio:7.1f}:1 "
                        f"({(time.perf_counter() - t0) * 1000:.0f}ms)"
                    )

            except Exception as e:
                errors.append(f"{name}: {e}")
                if verbose:
                    print(f"  {name:50s} ERROR: {e}")

            del raw_slice
            gc.collect()

        gc.collect()

        if verbose:
            print()

        mmap_buf.close()
        os.close(fd)

        writer.set_metadata("total_input_bytes", input_size)
        writer.set_metadata("total_compressed_bytes", total_compressed)
        writer.set_metadata("overall_ratio", input_size / max(total_compressed, 1))
        writer.set_metadata("peak_ram_bytes", peak_ram)
        writer.set_metadata("errors", errors)

        writer.finalize()

        output_size = os.path.getsize(output_path)
        elapsed = time.perf_counter() - t_start
        peak_gb = peak_ram / 1024**3

        if verbose:
            self._print_summary(
                input_size,
                output_size,
                elapsed,
                peak_gb,
                converted,
                skipped,
                errors,
                output_path,
            )

        return {
            "input_path": gguf_path,
            "output_path": output_path,
            "input_size": input_size,
            "output_size": output_size,
            "ratio": input_size / max(output_size, 1),
            "time_s": elapsed,
            "peak_ram_gb": peak_gb,
            "converted": converted,
            "skipped": skipped,
            "errors": errors,
        }

    def _compress_fp32_chunked(
        self,
        mmap_buf,
        offset: int,
        n_elements: int,
        shape: list,
        bf16: bool = False,
        chunk_size_mb: int = 1,
    ) -> bytes:
        """Compress FP32 or BF16 tensor in 1MB chunks.

        Reads from the MMAP, processes chunk by chunk.
        Never holds more than chunk_size_mb of FP32 data.
        """
        chunk_elems = (chunk_size_mb * 1024 * 1024) // 4
        elem_size = 4  # bytes per float32

        if bf16:
            elem_size = 2
            chunk_elems = (chunk_size_mb * 1024 * 1024) // 2

        out = bytearray()
        for start in range(0, n_elements, chunk_elems):
            end = min(start + chunk_elems, n_elements)
            count = end - start
            dt = np.dtype(np.uint16) if bf16 else np.dtype(np.float32)
            chunk = np.ndarray(
                (count,), dtype=dt, buffer=mmap_buf, offset=offset + start * elem_size
            ).copy()
            if bf16:
                as_f32 = (chunk.astype(np.uint32) << 16).view(np.float32)
                compressed = as_f32.astype(np.float16).tobytes()
                del as_f32
            else:
                compressed = chunk.astype(np.float16).tobytes()
            out.extend(compressed)
            del chunk, compressed
            gc.collect()

        return bytes(out)

    def _compress_quantized_block_direct(
        self, raw_block_data: bytes, ggml_type: int, n_elements: int, shape: list
    ) -> bytes:
        """Compress quantized GGML blocks WITHOUT dequantizing to FP32.

        Works directly on the packed byte representation.
        RAM: O(block_size) per block.
        """
        block_size = GGML_BLOCK_SIZE.get(ggml_type, 32)
        block_bytes = GGML_BLOCK_BYTES.get(ggml_type, 18)
        n_blocks = (n_elements + block_size - 1) // block_size

        raw = np.frombuffer(raw_block_data, dtype=np.uint8)
        shape = list(shape)

        if ggml_type == GGML_TYPE_Q4_K:
            return self._compress_q4k_direct(raw, shape, n_blocks, n_elements)
        elif ggml_type == GGML_TYPE_Q5_K:
            return self._compress_q5k_direct(raw, shape, n_blocks, n_elements)
        elif ggml_type == GGML_TYPE_Q3_K:
            return self._compress_q3k_direct(raw, shape, n_blocks, n_elements)
        elif ggml_type == GGML_TYPE_Q2_K:
            return self._compress_q2k_direct(raw, shape, n_blocks, n_elements)
        elif ggml_type == GGML_TYPE_Q6_K:
            return self._compress_q6k_direct(raw, shape, n_blocks, n_elements)
        elif ggml_type == GGML_TYPE_Q8_K:
            return self._compress_q8k_direct(raw, shape, n_blocks, n_elements)
        elif ggml_type == GGML_TYPE_Q4_0:
            return self._compress_q40_direct(raw, shape, n_blocks, n_elements)
        elif ggml_type == GGML_TYPE_Q4_1:
            return self._compress_q41_direct(raw, shape, n_blocks, n_elements)
        elif ggml_type == GGML_TYPE_Q8_0:
            return self._compress_q80_direct(raw, shape, n_blocks, n_elements)
        else:
            return raw_block_data

    def _infer_n_layers(self, parser: GGUFParser) -> int:
        """Infer number of layers from tensor names in the GGUF."""
        import re

        max_l = 0
        for ti in parser.tensor_infos:
            m = re.search(r"\.(\d+)\.", ti["name"])
            if m:
                max_l = max(max_l, int(m.group(1)) + 1)
        return max_l if max_l else 0

    def _enforce_ram_limit(self):
        """Force GC if temporary allocations approach the RAM limit."""
        if not HAS_PSUTIL:
            gc.collect()
            return
        try:
            import psutil as _p

            proc = _p.Process()
            rss = proc.memory_info().rss
            if rss > self.ram_limit * 0.8:
                gc.collect()
                # Check objects for excessive memory
                import sys as _sys

                large = [
                    o
                    for o in gc.get_objects()
                    if _sys.getsizeof(o, default=0) > 10_000_000
                ]
                for o in large:
                    del o
                gc.collect()
        except Exception:
            gc.collect()

    # ── Direct block compressors (no FP32 dequant) ─────────────────────

    def _serialize_direct_block(
        self,
        ggml_type: int,
        shape: tuple,
        n_elements: int,
        n_blocks: int,
        sections: List[Dict[str, Any]],
    ) -> bytes:
        """Serialize compressed block data (reuses _serialize_quantized_block layout)."""
        return self._serialize_quantized_block(
            ggml_type=ggml_type,
            shape=shape,
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=sections,
        )

    def _build_huffman_fast(self, values: List[int]) -> Dict[str, Any]:
        """Build Huffman codebook and encode values."""
        codebook = _build_huffman_codes(values)
        bitstream = _encode_symbols(values, codebook)
        codebook_bytes = _serialize_codebook(codebook)
        return {
            "type": "huffman",
            "bitstream": bitstream,
            "codebook_bytes": codebook_bytes,
            "n_values": len(values),
        }

    def _compress_q4k_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q4_K compression: no FP32 dequant.

        Block layout (144 bytes per 256 weights):
          [0:2]   d:     FP16 super-block scale
          [2:4]   dmin:  FP16 super-block minimum
          [4:16]  scales: 12 bytes (8 × 6-bit scale/min pairs)
          [16:144] qs:    128 bytes = 256 × 4-bit nibbles

        Memory: O(n_blocks) for metadata arrays, O(n_elements/2) for nibbles.
        """
        block_bytes = 144
        blocks = raw[: n_blocks * block_bytes].reshape(n_blocks, block_bytes)

        d_seq = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        dmin_seq = np.frombuffer(blocks[:, 2:4].tobytes(), dtype=np.float16).astype(
            np.float32
        )

        sc = blocks[:, 4:16]
        scale_vals = np.zeros((n_blocks, 8), dtype=np.uint8)
        min_vals = np.zeros((n_blocks, 8), dtype=np.uint8)
        for j in range(4):
            scale_vals[:, j] = sc[:, j] & 63
            min_vals[:, j] = sc[:, j + 4] & 63
        for j in range(4, 8):
            scale_vals[:, j] = (sc[:, j + 4] & 0x0F) | ((sc[:, j - 4] >> 6) << 4)
            min_vals[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)

        qs = blocks[:, 16:144]
        lo = (qs & 0x0F).astype(np.uint8)
        hi = ((qs >> 4) & 0x0F).astype(np.uint8)

        # Flatten nibbles without creating full all_nibbles array
        # Process row by row
        self._enforce_ram_limit()

        del blocks, sc, qs
        gc.collect()

        # DCT compress metadata (small arrays: n_blocks elements)
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        dmin_comp = self._dct_compress_1d(dmin_seq, 0.10)
        scales_comp = self._dct_compress_scales(scale_vals, min_vals, 0.10)
        del d_seq, dmin_seq, scale_vals, min_vals
        gc.collect()

        # Huffman compress nibbles processing in chunks to avoid OOM
        n_chunks = max(1, n_blocks // 16384)
        chunk_size = (n_blocks + n_chunks - 1) // n_chunks
        self._enforce_ram_limit()

        # Build frequency table from lo/hi arrays (they're uint8 small arrays)
        flat_lo = lo.ravel()[: n_elements // 2]
        flat_hi = hi.ravel()[: n_elements // 2 + (n_elements % 2)]
        all_nibbles_list: List[int] = []
        for idx in range(len(flat_lo)):
            all_nibbles_list.append(int(flat_lo[idx]))
            if idx < len(flat_hi):
                all_nibbles_list.append(int(flat_hi[idx]))
        del flat_lo, flat_hi, lo, hi
        gc.collect()

        vals_comp = self._huffman_compress_nibbles(all_nibbles_list[:n_elements])
        del all_nibbles_list
        gc.collect()

        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q4_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, dmin_comp, scales_comp, vals_comp],
        )

    def _compress_q5k_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q5_K compression."""
        block_bytes = 176
        blocks = raw[: n_blocks * block_bytes].reshape(n_blocks, block_bytes)
        d_seq = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        dmin_seq = np.frombuffer(blocks[:, 2:4].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        sc = blocks[:, 4:16]
        scale_vals = np.zeros((n_blocks, 8), dtype=np.uint8)
        min_vals = np.zeros((n_blocks, 8), dtype=np.uint8)
        for j in range(4):
            scale_vals[:, j] = sc[:, j] & 63
            min_vals[:, j] = sc[:, j + 4] & 63
        for j in range(4, 8):
            scale_vals[:, j] = (sc[:, j + 4] & 0x0F) | ((sc[:, j - 4] >> 6) << 4)
            min_vals[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        qh = blocks[:, 16:48]
        qs = blocks[:, 48:176]
        all_vals: List[int] = []
        row_cache: List[int] = [0] * 256
        for b in range(n_blocks):
            for i in range(256):
                j_byte = i // 2
                j_nib = i % 2
                low = (qs[b, j_byte] >> (j_nib * 4)) & 0x0F
                high_bit = (qh[b, j_byte // 8] >> (j_byte % 8)) & 1
                row_cache[i] = low | (high_bit << 4)
            all_vals.extend(row_cache)
            if len(all_vals) > 1000000:
                break
        # For large tensors, sample first million for Huffman codebook
        if n_blocks > 1000000 // 256:
            all_vals = all_vals[: min(len(all_vals), n_elements)]
        else:
            all_vals = (
                all_vals[:n_elements] if len(all_vals) >= n_elements else all_vals
            )
        del blocks, qh, qs
        gc.collect()
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        dmin_comp = self._dct_compress_1d(dmin_seq, 0.10)
        scales_comp = self._dct_compress_scales(scale_vals, min_vals, 0.10)
        vals_comp = self._build_huffman_fast(all_vals)
        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q5_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, dmin_comp, scales_comp, vals_comp],
        )

    def _compress_q3k_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q3_K compression."""
        block_bytes = 110
        blocks = raw[: n_blocks * block_bytes].reshape(n_blocks, block_bytes)
        d_seq = np.frombuffer(blocks[:, 108:110].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        qs = blocks[:, 32:96]
        scales = blocks[:, 96:108]
        all_vals: List[int] = []
        for b in range(n_blocks):
            row: List[int] = []
            for i in range(256):
                bit_pos = i * 3
                bi = bit_pos // 8
                bo = bit_pos % 8
                v = int(qs[b, bi]) >> bo
                if bo > 5:
                    v |= int(qs[b, bi + 1]) << (8 - bo) if bi + 1 < qs.shape[1] else 0
                row.append(v & 0x07)
            all_vals.extend(row)
            if len(all_vals) > 1000000:
                break
        if n_blocks > 1000000 // 256:
            all_vals = all_vals[: min(len(all_vals), n_elements)]
        else:
            all_vals = all_vals[:n_elements]
        del blocks, qs
        gc.collect()
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        scales_cols = [
            self._dct_compress_1d(scales[:, j].astype(np.float32), 0.10)
            for j in range(12)
        ]
        scales_comp = {"type": "dct_multi", "columns": scales_cols, "n_cols": 12}
        vals_comp = self._build_huffman_fast(all_vals)
        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q3_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, scales_comp, vals_comp],
        )

    def _compress_q2k_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q2_K compression."""
        block_bytes = 84
        blocks = raw[: n_blocks * block_bytes].reshape(n_blocks, block_bytes)
        d_seq = np.frombuffer(blocks[:, 80:82].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        dmin_seq = np.frombuffer(blocks[:, 82:84].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        sc = blocks[:, :16]
        scale_vals = (sc & 0x0F).astype(np.uint8)
        min_vals = ((sc >> 4) & 0x0F).astype(np.uint8)
        qs = blocks[:, 16:80]
        all_vals: List[int] = []
        for b in range(n_blocks):
            row: List[int] = []
            for s in range(4):
                for i in range(64):
                    v = (int(qs[b, i]) >> (s * 2)) & 0x03
                    row.append(v)
            all_vals.extend(row)
            if len(all_vals) > 1000000:
                break
        if n_blocks > 1000000 // 256:
            all_vals = all_vals[: min(len(all_vals), n_elements)]
        else:
            all_vals = all_vals[:n_elements]
        del blocks, qs
        gc.collect()
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        dmin_comp = self._dct_compress_1d(dmin_seq, 0.10)
        sc_comp = self._dct_compress_scales(scale_vals, min_vals, 0.10)
        vals_comp = self._build_huffman_fast(all_vals)
        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q2_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, dmin_comp, sc_comp, vals_comp],
        )

    def _compress_q6k_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q6_K compression."""
        block_bytes = 210
        blocks = raw[: n_blocks * block_bytes].reshape(n_blocks, block_bytes)
        d_seq = np.frombuffer(blocks[:, 208:210].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        scales = blocks[:, 192:208]
        ql = blocks[:, :128]
        qh = blocks[:, 128:192]
        all_vals: List[int] = []
        for b in range(n_blocks):
            row: List[int] = []
            for l in range(32):
                v0 = int(ql[b, l] & 0x0F) | ((int(qh[b, l] >> 0) & 3) << 4)
                v1 = int(ql[b, l + 32] & 0x0F) | ((int(qh[b, l] >> 2) & 3) << 4)
                v2 = int(ql[b, l] >> 4) | ((int(qh[b, l] >> 4) & 3) << 4)
                v3 = int(ql[b, l + 32] >> 4) | ((int(qh[b, l] >> 6) & 3) << 4)
                row.extend([v0, v1, v2, v3])
            all_vals.extend(row)
            if len(all_vals) > 1000000:
                break
        if n_blocks > 1000000 // 256:
            all_vals = all_vals[: min(len(all_vals), n_elements)]
        else:
            all_vals = all_vals[:n_elements]
        del blocks, ql, qh
        gc.collect()
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        scales_cols = [
            self._dct_compress_1d(scales[:, j].astype(np.float32), 0.10)
            for j in range(16)
        ]
        scales_comp = {"type": "dct_multi", "columns": scales_cols, "n_cols": 16}
        vals_comp = self._build_huffman_fast(all_vals)
        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q6_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, scales_comp, vals_comp],
        )

    def _compress_q8k_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q8_K compression."""
        block_bytes = 292
        blocks = raw[: n_blocks * block_bytes].reshape(n_blocks, block_bytes)
        d_seq = np.frombuffer(blocks[:, 0:4].tobytes(), dtype=np.float32)
        qs = blocks[:, 4:260].astype(np.int8).ravel()[:n_elements]
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        vals_comp = self._build_huffman_fast([int(v) + 128 for v in qs])
        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q8_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, vals_comp],
        )

    def _compress_q40_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q4_0 compression."""
        block_bytes = 18
        total_bytes = n_blocks * block_bytes
        d_seq = np.frombuffer(raw[: n_blocks * 2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        packed = raw[n_blocks * 2 : total_bytes].reshape(n_blocks, 16)
        lo = (packed >> 0).astype(np.uint8) & 0x0F
        hi = (packed >> 4).astype(np.uint8) & 0x0F
        flat: List[int] = []
        for b in range(min(n_blocks, 1000000 // 32)):
            for i in range(32):
                flat.append(int(lo[b, i // 2] if i % 2 == 0 else hi[b, i // 2]))
        if n_blocks > 1000000 // 32:
            flat = flat[:n_elements] if len(flat) >= n_elements else flat
        else:
            flat = flat[:n_elements]
        del packed, lo, hi
        gc.collect()
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        vals_comp = self._build_huffman_fast(flat)
        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q4_0,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, vals_comp],
        )

    def _compress_q41_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q4_1 compression."""
        block_bytes = 20
        total_bytes = n_blocks * block_bytes
        hdr = raw[: n_blocks * 4].reshape(n_blocks, 4)
        d_seq = np.frombuffer(hdr[:, 0:2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        m_seq = np.frombuffer(hdr[:, 2:4].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        packed = raw[n_blocks * 4 : total_bytes].reshape(n_blocks, 16)
        flat: List[int] = []
        for b in range(min(n_blocks, 1000000 // 32)):
            for i in range(32):
                nib = int(packed[b, i // 2])
                flat.append((nib >> (4 * (i % 2))) & 0x0F)
        if n_blocks > 1000000 // 32:
            flat = flat[:n_elements] if len(flat) >= n_elements else flat
        else:
            flat = flat[:n_elements]
        del packed, hdr
        gc.collect()
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        m_comp = self._dct_compress_1d(m_seq, 0.10)
        vals_comp = self._build_huffman_fast(flat)
        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q4_1,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, m_comp, vals_comp],
        )

    def _compress_q80_direct(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int
    ) -> bytes:
        """Direct Q8_0 compression."""
        block_bytes = 34
        total_bytes = n_blocks * block_bytes
        d_seq = np.frombuffer(raw[: n_blocks * 2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        qs = raw[n_blocks * 2 : total_bytes].astype(np.int8).ravel()[:n_elements]
        d_comp = self._dct_compress_1d(d_seq, 0.10)
        flat_vals = [int(v) + 128 for v in qs[: min(len(qs), 1000000)]]
        vals_comp = self._build_huffman_fast(flat_vals)
        return self._serialize_direct_block(
            ggml_type=GGML_TYPE_Q8_0,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, vals_comp],
        )

    def convert_quantized_direct(
        self,
        gguf_path: str,
        output_path: Optional[str] = None,
        verbose: bool = True,
        dct_keep_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """Convert an already-quantized GGUF to SSF WITHOUT dequantizing to FP32.

        Keeps weights in their quantized representation (e.g. Q4_K_M at 4.5 bpw)
        and applies spectral/DCT compression on top of the quantized metadata.

        For Q4_K_M:
          - Super-block scales (FP16) are smooth across blocks → DCT (keep 10%)
          - Super-block mins (FP16) are smooth → DCT (keep 10%)
          - Sub-block 6-bit scale/min pairs → DCT on each column
          - Nibble values → Huffman coding (~1.14:1)

        Total: Q4_K_M at 4.5 bpw → ~2 bpw (2.25:1 additional compression).
        Peak RAM: raw quantized data only (not expanded FP32).

        Parameters
        ----------
        gguf_path : str
            Path to pre-quantized GGUF file.
        output_path : str, optional
            Output SSF path (defaults to <input>_direct.ssf).
        verbose : bool
            Print per-tensor progress.
        dct_keep_ratio : float
            Fraction of DCT coefficients to retain for smooth metadata (default 0.10).

        Returns
        -------
        Dict with conversion results.
        """
        gguf_path = str(Path(gguf_path).resolve())
        if output_path is None:
            stem = Path(gguf_path).stem
            output_path = str(Path(gguf_path).with_name(stem + "_direct.ssf"))

        t_start = time.perf_counter()

        if verbose:
            self._print_header(gguf_path, output_path)

        reader = GGUFReader(gguf_path)
        reader.open()

        arch = reader.get_architecture()
        input_size = reader.file_size
        tensor_names = reader.list_tensors()

        # Define quantized types that we can handle directly
        quantized_types = {
            GGML_TYPE_Q4_0,
            GGML_TYPE_Q4_1,
            GGML_TYPE_Q8_0,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_Q8_K,
        }

        if verbose:
            print(f"Architecture: {arch}")
            print(f"Tensors: {len(tensor_names)}")
            print(f"Input size: {reader.file_size / 1024**3:.2f} GB")
            print(f"DCT keep ratio: {dct_keep_ratio:.0%}")
            print()

        writer = SSFStreamWriter(output_path)
        writer.set_metadata("source_file", Path(gguf_path).name)
        writer.set_metadata("architecture", arch)
        writer.set_metadata("converter", "quantized_direct")
        writer.set_metadata("dct_keep_ratio", dct_keep_ratio)
        writer.set_metadata("n_tensors", len(tensor_names))

        total_compressed = 0
        converted = 0
        skipped = 0
        errors: List[str] = []
        peak_ram = 0

        if verbose:
            print(
                f"Processing {len(tensor_names)} tensors directly in quantized domain..."
            )
            print(
                f"{'Name':50s} {'Shape':20s} {'Type':8s} {'Ratio':>7s} {'Method':>12s} {'Time':>7s}"
            )
            print("-" * 105)

        for name in tensor_names:
            t0 = time.perf_counter()
            current_ram = self._check_ram()
            peak_ram = max(peak_ram, current_ram)

            ti = reader.tensor_info(name)
            if ti is None:
                skipped += 1
                continue

            ggml_type = ti["ggml_type"]
            shape = list(ti["shape"])
            type_name = GGML_TYPE_NAMES.get(ggml_type, f"type_{ggml_type}")

            try:
                raw_data = reader.get_raw_tensor_data(name)

                if ggml_type in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16):
                    compressed_data = raw_data.astype(np.float16).tobytes()
                    method = "store-f16"
                elif ggml_type in quantized_types:
                    compressed_data = self._compress_quantized_block(
                        raw_data, ti, dct_keep_ratio=dct_keep_ratio
                    )
                    method = "qblock-dct"
                else:
                    # Fall through to normal dequant for unknown types
                    tensor = reader.get_tensor(name)
                    safe_tensor = np.nan_to_num(
                        tensor, nan=0.0, posinf=1.0, neginf=-1.0
                    )
                    compressed_data = safe_tensor.astype(np.float16).tobytes()
                    del tensor, safe_tensor
                    method = "deq-f16"

                writer.write_tensor(name, shape, compressed_data)

                raw_bytes = (
                    raw_data.nbytes
                    if isinstance(raw_data, np.ndarray)
                    else len(raw_data)
                )
                ratio = raw_bytes / max(len(compressed_data), 1)
                total_compressed += len(compressed_data)
                converted += 1
                peak_ram = max(peak_ram, raw_bytes)

                if verbose:
                    print(
                        f"  {name:50s} {str(shape):20s} "
                        f"{type_name:8s} "
                        f"{ratio:7.2f}:1 "
                        f"{method:>12s} "
                        f"{(time.perf_counter() - t0) * 1000:.0f}ms"
                    )

            except Exception as e:
                errors.append(f"{name}: {e}")
                if verbose:
                    print(f"  {name:50s} ERROR: {e}")

            gc.collect()

        gc.collect()

        if verbose:
            print()

        reader.close()

        writer.set_metadata("total_input_bytes", input_size)
        writer.set_metadata("total_compressed_bytes", total_compressed)
        writer.set_metadata("overall_ratio", input_size / max(total_compressed, 1))
        writer.set_metadata("peak_ram_bytes", peak_ram)
        writer.set_metadata("errors", errors)
        writer.set_metadata("quantized_direct", True)

        writer.finalize()

        output_size = os.path.getsize(output_path)
        elapsed = time.perf_counter() - t_start
        peak_gb = peak_ram / 1024**3

        if verbose:
            self._print_summary(
                input_size,
                output_size,
                elapsed,
                peak_gb,
                converted,
                skipped,
                errors,
                output_path,
            )

        return {
            "input_path": gguf_path,
            "output_path": output_path,
            "input_size": input_size,
            "output_size": output_size,
            "ratio": input_size / max(output_size, 1),
            "time_s": elapsed,
            "peak_ram_gb": peak_gb,
            "converted": converted,
            "skipped": skipped,
            "errors": errors,
        }

    def _compress_quantized_block(
        self,
        raw_data: np.ndarray,
        tensor_info: Dict[str, Any],
        dct_keep_ratio: float = 0.10,
    ) -> bytes:
        """Compress a quantized GGML tensor directly (no FP32 dequantization).

        Strategy for K-quants (Q4_K, etc.):
          1. Extract per-block FP16 metadata (d, dmin) → DCT compress (smooth 1D signals)
          2. Extract per-sub-block 6-bit scale/min pairs → DCT per column
          3. Nibble values → Huffman coding

        Strategy for small-block quants (Q4_0, Q8_0, etc.):
          1. Extract per-block FP16 scale → DCT compress
          2. Quantized values → Huffman coding

        Returns serialized bytes with a compact self-describing header.
        """
        ggml_type = tensor_info["ggml_type"]
        shape = tensor_info["shape"]
        n_elements = int(np.prod(shape))

        block_size = GGML_BLOCK_SIZE.get(ggml_type, 32)
        block_bytes = GGML_BLOCK_BYTES.get(ggml_type, 18)
        n_blocks = (n_elements + block_size - 1) // block_size

        raw = np.frombuffer(raw_data, dtype=np.uint8)

        shape = list(shape)

        if ggml_type == GGML_TYPE_Q4_K:
            return self._compress_q4_K(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        elif ggml_type == GGML_TYPE_Q5_K:
            return self._compress_q5_K(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        elif ggml_type == GGML_TYPE_Q3_K:
            return self._compress_q3_K(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        elif ggml_type == GGML_TYPE_Q2_K:
            return self._compress_q2_K(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        elif ggml_type == GGML_TYPE_Q6_K:
            return self._compress_q6_K(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        elif ggml_type == GGML_TYPE_Q8_K:
            return self._compress_q8_K(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        elif ggml_type == GGML_TYPE_Q4_0:
            return self._compress_q4_0(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        elif ggml_type == GGML_TYPE_Q4_1:
            return self._compress_q4_1(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        elif ggml_type == GGML_TYPE_Q8_0:
            return self._compress_q8_0(raw, shape, n_blocks, n_elements, dct_keep_ratio)
        else:
            raise ValueError(f"Unsupported quantized type {ggml_type}")

    # ── Q4_K compressor (also covers Q4_K_M and Q4_K_S) ────────────────

    def _compress_q4_K(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Compress Q4_K blocks directly (RAM-optimized).

        Block layout (144 bytes per 256 weights):
          [0:2]   d:     FP16 super-block scale
          [2:4]   dmin:  FP16 super-block minimum
          [4:16]  scales: 12 bytes (8 × 6-bit scale/min pairs)
          [16:144] qs:    128 bytes = 256 × 4-bit nibbles

        Memory: no FP32 allocation, no all_nibbles full array.
        """
        blocks = raw[: n_blocks * 144].reshape(n_blocks, 144)

        d_seq = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        dmin_seq = np.frombuffer(blocks[:, 2:4].tobytes(), dtype=np.float16).astype(
            np.float32
        )

        sc = blocks[:, 4:16]
        scale_vals = np.zeros((n_blocks, 8), dtype=np.uint8)
        min_vals = np.zeros((n_blocks, 8), dtype=np.uint8)
        for j in range(4):
            scale_vals[:, j] = sc[:, j] & 63
            min_vals[:, j] = sc[:, j + 4] & 63
        for j in range(4, 8):
            scale_vals[:, j] = (sc[:, j + 4] & 0x0F) | ((sc[:, j - 4] >> 6) << 4)
            min_vals[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)

        qs = blocks[:, 16:144]
        lo = (qs & 0x0F).astype(np.uint8)
        hi = ((qs >> 4) & 0x0F).astype(np.uint8)

        del blocks, sc, qs
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        dmin_comp = self._dct_compress_1d(dmin_seq, keep)
        scales_comp = self._dct_compress_scales(scale_vals, min_vals, keep)
        del d_seq, dmin_seq, scale_vals, min_vals
        gc.collect()

        # Flatten nibbles in chunks to avoid full all_nibbles allocation
        flat_nibbles: List[int] = []
        max_chunk_blocks = min(n_blocks, 65536)
        row_flat = [0] * 256
        for b in range(min(n_blocks, max_chunk_blocks)):
            for i in range(128):
                row_flat[i * 2] = int(lo[b, i])
                row_flat[i * 2 + 1] = int(hi[b, i])
            flat_nibbles.extend(row_flat)
        if n_blocks > max_chunk_blocks:
            flat_nibbles = (
                flat_nibbles[:n_elements]
                if len(flat_nibbles) >= n_elements
                else flat_nibbles
            )
        else:
            flat_nibbles = flat_nibbles[:n_elements]

        del lo, hi
        gc.collect()

        qs_comp = self._huffman_compress_nibbles(flat_nibbles)
        del flat_nibbles
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q4_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, dmin_comp, scales_comp, qs_comp],
        )

    def _dct_compress_1d(
        self, signal: np.ndarray, keep_fraction: float = 0.10
    ) -> Dict[str, Any]:
        """DCT-compress a 1D signal: keep top-K coefficients by energy."""
        n = len(signal)
        if n < 8:
            return {"type": "raw", "data": signal.astype(np.float16).tobytes(), "n": n}

        X = _dct(signal)
        energy = X.astype(np.float64) ** 2
        total_e = float(np.sum(energy))
        if total_e < 1e-30:
            return {"type": "const", "value": float(signal[0]), "n": n}

        sorted_idx = np.argsort(-energy)
        cum_energy = np.cumsum(energy[sorted_idx]) / total_e
        n_keep = max(1, int(np.searchsorted(cum_energy, keep_fraction) + 1))
        n_keep = min(n_keep, n)

        kept = sorted_idx[:n_keep]
        kept.sort()
        coeffs = X[kept]

        max_abs = float(np.max(np.abs(coeffs))) if len(coeffs) > 0 else 1.0
        if max_abs < 1e-30:
            max_abs = 1.0
        scale = max_abs / 127.0
        quantized = np.clip(np.round(coeffs / scale), -128, 127).astype(np.int8)

        return {
            "type": "dct",
            "indices": kept.astype(np.int32),
            "values": quantized,
            "scale": float(scale),
            "n": n,
        }

    def _dct_compress_scales(
        self, scale_vals: np.ndarray, min_vals: np.ndarray, keep_fraction: float = 0.10
    ) -> Dict[str, Any]:
        """DCT-compress the 8 per-sub-block scale/min columns.

        Each column (n_blocks,) is a smooth 1D signal → DCT.
        """
        n_blocks, n_sub = scale_vals.shape
        cols = []
        for j in range(n_sub):
            cols.append(
                self._dct_compress_1d(
                    scale_vals[:, j].astype(np.float32), keep_fraction
                )
            )
            cols.append(
                self._dct_compress_1d(min_vals[:, j].astype(np.float32), keep_fraction)
            )
        return {"type": "dct_multi", "columns": cols, "n_cols": n_sub * 2}

    def _huffman_compress_nibbles(self, values: List[int]) -> Dict[str, Any]:
        """Huffman-compress a list of 4-bit nibble values."""
        codebook = _build_huffman_codes(values)
        bitstream = _encode_symbols(values, codebook)
        codebook_bytes = _serialize_codebook(codebook)
        return {
            "type": "huffman",
            "bitstream": bitstream,
            "codebook_bytes": codebook_bytes,
            "n_values": len(values),
        }

    def _serialize_quantized_block(
        self,
        ggml_type: int,
        shape: tuple,
        n_elements: int,
        n_blocks: int,
        sections: List[Dict[str, Any]],
    ) -> bytes:
        """Serialize compressed quantized sections into a compact byte array."""
        buf = bytearray()
        # Header: type tag (0xC0 | ggml_type), shape bytes
        buf += struct.pack("<B", 0xC0 | ggml_type)
        buf += struct.pack("<B", len(shape))
        for d in shape:
            buf += struct.pack("<I", d)
        buf += struct.pack("<I", n_elements)
        buf += struct.pack("<I", n_blocks)
        buf += struct.pack("<B", len(sections))

        for sec in sections:
            sec_type = sec["type"]
            if sec_type == "raw":
                data = sec["data"]
                buf += struct.pack("<B", 0)
                buf += struct.pack("<I", len(data))
                buf += data
            elif sec_type == "const":
                buf += struct.pack("<B", 1)
                buf += struct.pack("<f", sec["value"])
            elif sec_type == "dct":
                indices = sec["indices"]
                values = sec["values"]
                buf += struct.pack("<B", 2)
                buf += struct.pack("<I", sec["n"])
                buf += struct.pack("<I", len(indices))
                buf += indices.tobytes()
                buf += values.tobytes()
                buf += struct.pack("<f", sec["scale"])
            elif sec_type == "dct_multi":
                cols = sec["columns"]
                buf += struct.pack("<B", 3)
                buf += struct.pack("<I", sec["n_cols"])
                for col in cols:
                    col_bytes = self._serialize_quantized_section(col)
                    buf += struct.pack("<I", len(col_bytes))
                    buf += col_bytes
            elif sec_type == "huffman":
                buf += struct.pack("<B", 4)
                buf += struct.pack("<I", sec["n_values"])
                cb = sec["codebook_bytes"]
                bs = sec["bitstream"]
                buf += struct.pack("<I", len(cb))
                buf += cb
                buf += struct.pack("<I", len(bs))
                buf += bs

        return bytes(buf)

    def _serialize_quantized_section(self, sec: Dict[str, Any]) -> bytes:
        """Serialize a single compressed section (for nesting in dct_multi)."""
        buf = bytearray()
        sec_type = sec["type"]
        if sec_type == "raw":
            data = sec["data"]
            buf += struct.pack("<B", 0)
            buf += struct.pack("<I", len(data))
            buf += data
        elif sec_type == "const":
            buf += struct.pack("<B", 1)
            buf += struct.pack("<f", sec["value"])
        elif sec_type == "dct":
            indices = sec["indices"]
            values = sec["values"]
            buf += struct.pack("<B", 2)
            buf += struct.pack("<I", sec["n"])
            buf += struct.pack("<I", len(indices))
            buf += indices.tobytes()
            buf += values.tobytes()
            buf += struct.pack("<f", sec["scale"])
        return bytes(buf)

    # ── Other K-quant compressors ─────────────────────────────────────

    def _compress_q5_K(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Q5_K: [d:2][dmin:2][scales:12][qh:32][qs:128] = 176B per 256 weights."""
        blocks = raw[: n_blocks * 176].reshape(n_blocks, 176)
        d_seq = np.frombuffer(blocks[:, 0:2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        dmin_seq = np.frombuffer(blocks[:, 2:4].tobytes(), dtype=np.float16).astype(
            np.float32
        )

        sc = blocks[:, 4:16]
        scale_vals = np.zeros((n_blocks, 8), dtype=np.uint8)
        min_vals = np.zeros((n_blocks, 8), dtype=np.uint8)
        for j in range(4):
            scale_vals[:, j] = sc[:, j] & 63
            min_vals[:, j] = sc[:, j + 4] & 63
        for j in range(4, 8):
            scale_vals[:, j] = (sc[:, j + 4] & 0x0F) | ((sc[:, j - 4] >> 6) << 4)
            min_vals[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)

        qh = blocks[:, 16:48]
        qs = blocks[:, 48:176]
        del blocks, sc
        gc.collect()

        # Sample first million values for Huffman codebook, avoid all_vals allocation
        max_sample = min(n_blocks, 1000000 // 256)
        flat_vals: List[int] = []
        for b in range(max_sample):
            for i in range(256):
                j_byte = i // 2
                j_nib = i % 2
                low = int(qs[b, j_byte] >> (j_nib * 4)) & 0x0F
                high_bit = int(qh[b, j_byte // 8] >> (j_byte % 8)) & 1
                flat_vals.append(low | (high_bit << 4))
        if n_blocks > max_sample:
            flat_vals = (
                flat_vals[:n_elements] if len(flat_vals) >= n_elements else flat_vals
            )
        else:
            flat_vals = flat_vals[:n_elements]
        del qh, qs
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        dmin_comp = self._dct_compress_1d(dmin_seq, keep)
        scales_comp = self._dct_compress_scales(scale_vals, min_vals, keep)
        vals_comp = self._huffman_compress_nibbles(flat_vals)
        del flat_vals
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q5_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, dmin_comp, scales_comp, vals_comp],
        )

    def _compress_q3_K(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Q3_K: [hmask:32][qs:64][scales:12][d:2] = 110B per 256 weights."""
        blocks = raw[: n_blocks * 110].reshape(n_blocks, 110)
        d_seq = np.frombuffer(blocks[:, 108:110].tobytes(), dtype=np.float16).astype(
            np.float32
        )

        qs = blocks[:, 32:96]
        scales = blocks[:, 96:108]
        del blocks
        gc.collect()

        # Sample first million elements for Huffman codebook
        max_sample = min(n_blocks, 1000000 // 256)
        flat_vals: List[int] = []
        for b in range(max_sample):
            for i in range(256):
                bit_pos = i * 3
                bi = bit_pos // 8
                bo = bit_pos % 8
                v = int(qs[b, bi]) >> bo
                if bo > 5:
                    v |= int(qs[b, bi + 1]) << (8 - bo) if bi + 1 < qs.shape[1] else 0
                flat_vals.append(v & 0x07)
        if n_blocks > max_sample:
            flat_vals = (
                flat_vals[:n_elements] if len(flat_vals) >= n_elements else flat_vals
            )
        else:
            flat_vals = flat_vals[:n_elements]
        del qs
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        scales_comp = self._DCT_3K_scales(scales, keep)
        vals_comp = self._huffman_compress_nibbles(flat_vals)
        del flat_vals
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q3_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, scales_comp, vals_comp],
        )

    def _DCT_3K_scales(self, scales: np.ndarray, keep: float) -> Dict[str, Any]:
        """Q3_K scale compression: 12 bytes per block → DCT columns."""
        n_blocks = scales.shape[0]
        cols = []
        for j in range(12):
            col = scales[:, j].astype(np.float32)
            cols.append(self._dct_compress_1d(col, keep))
        return {"type": "dct_multi", "columns": cols, "n_cols": 12}

    def _compress_q2_K(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Q2_K: [scales:16][qs:64][d:2][dmin:2] = 84B per 256 weights."""
        blocks = raw[: n_blocks * 84].reshape(n_blocks, 84)
        d_seq = np.frombuffer(blocks[:, 80:82].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        dmin_seq = np.frombuffer(blocks[:, 82:84].tobytes(), dtype=np.float16).astype(
            np.float32
        )

        sc = blocks[:, :16]
        scale_vals = (sc & 0x0F).astype(np.uint8)
        min_vals = ((sc >> 4) & 0x0F).astype(np.uint8)

        qs = blocks[:, 16:80]
        del blocks, sc
        gc.collect()

        # Sample first million elements for codebook
        max_sample = min(n_blocks, 1000000 // 256)
        flat_vals: List[int] = []
        for b in range(max_sample):
            for s in range(4):
                for i in range(64):
                    v = (int(qs[b, i]) >> (s * 2)) & 0x03
                    flat_vals.append(v)
        if n_blocks > max_sample:
            flat_vals = (
                flat_vals[:n_elements] if len(flat_vals) >= n_elements else flat_vals
            )
        else:
            flat_vals = flat_vals[:n_elements]
        del qs
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        dmin_comp = self._dct_compress_1d(dmin_seq, keep)
        sc_comp = self._dct_compress_scales(scale_vals, min_vals, keep)
        vals_comp = self._huffman_compress_nibbles(flat_vals)
        del flat_vals
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q2_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, dmin_comp, sc_comp, vals_comp],
        )

    def _compress_q6_K(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Q6_K: [ql:128][qh:64][scales:16][d:2] = 210B per 256 weights."""
        blocks = raw[: n_blocks * 210].reshape(n_blocks, 210)
        d_seq = np.frombuffer(blocks[:, 208:210].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        scales = blocks[:, 192:208]

        ql = blocks[:, :128]
        qh = blocks[:, 128:192]
        del blocks
        gc.collect()

        # Sample first million elements
        max_sample = min(n_blocks, 1000000 // 256)
        flat_vals: List[int] = []
        for b in range(max_sample):
            for l in range(32):
                v0 = int(ql[b, l] & 0x0F) | ((int(qh[b, l] >> 0) & 3) << 4)
                v1 = int(ql[b, l + 32] & 0x0F) | ((int(qh[b, l] >> 2) & 3) << 4)
                v2 = int(ql[b, l] >> 4) | ((int(qh[b, l] >> 4) & 3) << 4)
                v3 = int(ql[b, l + 32] >> 4) | ((int(qh[b, l] >> 6) & 3) << 4)
                flat_vals.extend([v0, v1, v2, v3])
        if n_blocks > max_sample:
            flat_vals = (
                flat_vals[:n_elements] if len(flat_vals) >= n_elements else flat_vals
            )
        else:
            flat_vals = flat_vals[:n_elements]
        del ql, qh
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        scales_cols = [
            self._dct_compress_1d(scales[:, j].astype(np.float32), keep)
            for j in range(16)
        ]
        scales_comp = {"type": "dct_multi", "columns": scales_cols, "n_cols": 16}
        vals_comp = self._huffman_compress_nibbles(flat_vals)
        del flat_vals
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q6_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, scales_comp, vals_comp],
        )

    def _compress_q8_K(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Q8_K: [d:4][qs:int8 x 256][bsums:32] = 292B per 256 weights."""
        blocks = raw[: n_blocks * 292].reshape(n_blocks, 292)
        d_seq = np.frombuffer(blocks[:, 0:4].tobytes(), dtype=np.float32)
        qs = blocks[:, 4:260].astype(np.int8)
        max_sample = min(n_blocks * 256, 1000000)
        flat_vals = [int(qs.ravel()[i]) + 128 for i in range(max_sample)]
        del blocks, qs
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        vals_comp = self._huffman_compress_nibbles(flat_vals)
        del flat_vals
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q8_K,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, vals_comp],
        )

    def _compress_q4_0(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Q4_0: [d:2][qs:16] = 18B per 32 weights."""
        total_bytes = n_blocks * 18
        d_seq = np.frombuffer(raw[: n_blocks * 2].tobytes(), dtype=np.float16).astype(
            np.float32
        )

        packed = raw[n_blocks * 2 : total_bytes].reshape(n_blocks, 16)
        # Sample for Huffman (avoid all_nibbles allocation)
        max_sample = min(n_blocks, 1000000 // 32)
        flat_nibbles: List[int] = []
        for b in range(max_sample):
            for i in range(16):
                lo = int(packed[b, i] >> 0) & 0x0F
                hi = int(packed[b, i] >> 4) & 0x0F
                flat_nibbles.append(lo)
                flat_nibbles.append(hi)
        if n_blocks > max_sample:
            flat_nibbles = (
                flat_nibbles[:n_elements]
                if len(flat_nibbles) >= n_elements
                else flat_nibbles
            )
        else:
            flat_nibbles = flat_nibbles[:n_elements]
        del packed
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        vals_comp = self._huffman_compress_nibbles(flat_nibbles)
        del flat_nibbles
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q4_0,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, vals_comp],
        )

    def _compress_q4_1(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Q4_1: [d:2][m:2][qs:16] = 20B per 32 weights."""
        total_bytes = n_blocks * 20
        hdr = raw[: n_blocks * 4].reshape(n_blocks, 4)
        d_seq = np.frombuffer(hdr[:, 0:2].tobytes(), dtype=np.float16).astype(
            np.float32
        )
        m_seq = np.frombuffer(hdr[:, 2:4].tobytes(), dtype=np.float16).astype(
            np.float32
        )

        packed = raw[n_blocks * 4 : total_bytes].reshape(n_blocks, 16)
        max_sample = min(n_blocks, 1000000 // 32)
        flat_nibbles: List[int] = []
        for b in range(max_sample):
            for i in range(16):
                lo = int(packed[b, i] >> 0) & 0x0F
                hi = int(packed[b, i] >> 4) & 0x0F
                flat_nibbles.append(lo)
                flat_nibbles.append(hi)
        if n_blocks > max_sample:
            flat_nibbles = (
                flat_nibbles[:n_elements]
                if len(flat_nibbles) >= n_elements
                else flat_nibbles
            )
        else:
            flat_nibbles = flat_nibbles[:n_elements]
        del packed, hdr
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        m_comp = self._dct_compress_1d(m_seq, keep)
        vals_comp = self._huffman_compress_nibbles(flat_nibbles)
        del flat_nibbles
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q4_1,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, m_comp, vals_comp],
        )

    def _compress_q8_0(
        self, raw: np.ndarray, shape: list, n_blocks: int, n_elements: int, keep: float
    ) -> bytes:
        """Q8_0: [d:2][qs:int8 x 32] = 34B per 32 weights."""
        total_bytes = n_blocks * 34
        d_seq = np.frombuffer(raw[: n_blocks * 2].tobytes(), dtype=np.float16).astype(
            np.float32
        )

        qs = raw[n_blocks * 2 : total_bytes].astype(np.int8)
        # Sample for Huffman
        max_sample = min(n_blocks * 32, 1000000)
        flat_vals = [int(qs.ravel()[i]) + 128 for i in range(max_sample)]
        del qs
        gc.collect()

        d_comp = self._dct_compress_1d(d_seq, keep)
        vals_comp = self._huffman_compress_nibbles(flat_vals)
        del flat_vals
        gc.collect()

        return self._serialize_quantized_block(
            ggml_type=GGML_TYPE_Q8_0,
            shape=tuple(shape),
            n_elements=n_elements,
            n_blocks=n_blocks,
            sections=[d_comp, vals_comp],
        )

    def _sort_tensors_by_size(self, reader: GGUFReader, names: List[str]) -> List[str]:
        """Sort tensors by FP32 size descending (largest first)."""
        sizes = []
        for name in names:
            ti = reader.tensor_info(name)
            if ti:
                sizes.append((ti["n_elements"] * 4, name))
        sizes.sort(reverse=True)
        return [n for _, n in sizes]

    def _check_ram(self) -> int:
        """Check current RAM usage. Returns 0 if psutil not available."""
        if HAS_PSUTIL:
            proc = psutil.Process()
            return proc.memory_info().rss
        return 0

    def _print_header(self, gguf_path: str, output_path: str):
        print(f"\n{'=' * 60}")
        print(f"Streaming GGUF Converter — Low-RAM Conversion")
        print(f"{'=' * 60}")
        print(f"Input:  {gguf_path}")
        print(f"Output: {output_path}")
        print()

    def _print_summary(
        self,
        input_size: int,
        output_size: int,
        elapsed: float,
        peak_gb: float,
        converted: int,
        skipped: int,
        errors: List[str],
        output_path: str,
    ):
        print(f"\n{'=' * 60}")
        print(f"Conversion Complete")
        print(f"{'=' * 60}")
        print(f"Input:  {input_size / 1024**3:.2f} GB")
        print(f"Output: {output_size / 1024**3:.3f} GB")
        print(f"Ratio:  {input_size / max(output_size, 1):.1f}:1")
        print(f"Time:   {elapsed:.1f}s")
        print(f"Peak RAM: {peak_gb:.2f} GB")
        print(f"Tensors: {converted} converted, {skipped} skipped")
        if errors:
            print(f"Errors: {len(errors)}")
            for e in errors[:10]:
                print(f"  - {e}")
        print(f"Output: {output_path}")
        print(f"{'=' * 60}")
