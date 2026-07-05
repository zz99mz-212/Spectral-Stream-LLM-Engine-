"""
GGUF Converter — Reads GGUF models, dequantizes all GGML types, applies
the unified quantizer pipeline, and writes SSF output with full validation.

Features:
  - All GGML quantization types (Q4_0 through Q8_K, IQ types, TQ2_0, BF16)
  - MMAP zero-copy access for large models
  - Parallel tensor conversion
  - Per-tensor compression stats and quality metrics (SNR, PSNR, max error)
  - JSON + human-readable compression reports
  - Batch conversion of multiple GGUF files
"""

from __future__ import annotations

import json
import math
import mmap as py_mmap
import os
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.format.gguf_parser_engine import (
    GGUFParser,
    GGMLDequantizer,
    GGML_BLOCK_SIZE,
    GGML_BLOCK_BYTES,
    GGML_TYPE_NAMES,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q8_1,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_K,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_S,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_TQ2_0,
)

from spectralstream.format.core import _format_size as format_size

try:
    from spectralstream.compression.unified_quantizer import UnifiedQuantizer
except ImportError:
    UnifiedQuantizer = None

from spectralstream.format.ssf_format_pipeline import (
    SSFFormatSpec,
    SSFTensorIndexEntry,
    SSFFooter,
    SSF_MAGIC,
    SSF_VERSION,
    SSF_HEADER_SIZE,
    SSF_FOOTER_SIZE,
    SSF_ALIGNMENT,
    CompressionType,
    COMPRESSION_NAMES,
    _sha256,
    _align_up,
)


# ═══════════════════════════════════════════════════════════════════════════
# Per-tensor compression stats
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TensorStats:
    name: str
    shape: List[int]
    ggml_type_name: str
    fp32_bytes: int
    gguf_bytes: int
    compressed_bytes: int
    ratio_vs_fp32: float
    ratio_vs_gguf: float
    mse: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    max_error: float = 0.0
    snr_db: float = 0.0
    convert_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ConversionReport:
    input_path: str
    output_path: str
    model_name: str
    architecture: str
    n_layers: int
    total_tensors: int
    converted_tensors: int
    skipped_tensors: int
    input_size_bytes: int
    output_size_bytes: int
    overall_ratio: float
    avg_snr_db: float
    avg_psnr_db: float
    avg_mse: float
    max_error_global: float
    total_time_s: float
    tensor_stats: List[TensorStats]
    errors: List[str]


# ═══════════════════════════════════════════════════════════════════════════
# GGUF Reader — zero-copy MMAP access
# ═══════════════════════════════════════════════════════════════════════════


class GGUFReader:
    """Read GGUF files using MMAP for zero-copy tensor access.

    Dequantizes all GGML types to FP32 on demand.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._parser: Optional[GGUFParser] = None
        self._mmap_obj: Optional[py_mmap.mmap] = None
        self._fd: Optional[int] = None
        self._tensor_index: Dict[str, dict] = {}
        self._metadata: Dict[str, Any] = {}
        self._n_layers: int = 0
        self._layer_index: Dict[int, List[str]] = {}

    def open(self) -> "GGUFReader":
        self._parser = GGUFParser(str(self.path))
        self._parser.parse()
        self._metadata = dict(self._parser.metadata)

        self._fd = os.open(str(self.path), os.O_RDONLY | os.O_CLOEXEC)
        file_size = os.fstat(self._fd).st_size
        self._mmap_obj = py_mmap.mmap(self._fd, file_size, access=py_mmap.ACCESS_READ)

        for ti in self._parser.tensor_infos:
            self._tensor_index[ti["name"]] = ti
            m = re.search(r"\.(\d+)\.", ti["name"])
            if m:
                li = int(m.group(1))
                self._layer_index.setdefault(li, []).append(ti["name"])
                self._n_layers = max(self._n_layers, li + 1)

        if not self._n_layers:
            self._n_layers = 32

        return self

    def close(self):
        if self._mmap_obj is not None:
            self._mmap_obj.close()
            self._mmap_obj = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def get_tensor(self, name: str) -> np.ndarray:
        ti = self._tensor_index.get(name)
        if ti is None:
            raise KeyError(f"Tensor {name!r} not found")

        ggml_type = ti["ggml_type"]
        offset = int(ti["offset"]) + self._parser.tensor_data_offset
        shape = ti["shape"]
        n_elements = int(ti["n_elements"])

        if ggml_type == GGML_TYPE_F32:
            return np.ndarray(
                (n_elements,), dtype=np.float32, buffer=self._mmap_obj, offset=offset
            ).reshape(shape)

        if ggml_type == GGML_TYPE_F16:
            return (
                np.ndarray(
                    (n_elements,),
                    dtype=np.float16,
                    buffer=self._mmap_obj,
                    offset=offset,
                )
                .astype(np.float32)
                .reshape(shape)
            )

        data_size = int(ti["data_size"])
        raw = np.frombuffer(
            self._mmap_obj, dtype=np.uint8, offset=offset, count=data_size
        )
        result = GGMLDequantizer.dequantize_fast(raw, ggml_type)
        return result[:n_elements].reshape(shape)

    def get_tensor_view(self, name: str) -> np.ndarray:
        """Zero-copy view into mmap for F32/F16 tensors. No RAM allocated."""
        ti = self._tensor_index.get(name)
        if ti is None:
            raise KeyError(f"Tensor {name!r} not found")

        ggml_type = ti["ggml_type"]
        offset = int(ti["offset"]) + self._parser.tensor_data_offset
        shape = ti["shape"]
        n_elements = int(ti["n_elements"])

        if ggml_type == GGML_TYPE_F32:
            return np.ndarray(
                (n_elements,), dtype=np.float32, buffer=self._mmap_obj, offset=offset
            ).reshape(shape)

        if ggml_type == GGML_TYPE_F16:
            return np.ndarray(
                (n_elements,), dtype=np.float16, buffer=self._mmap_obj, offset=offset
            ).reshape(shape)

        raise TypeError(
            f"Tensor {name!r} is {GGML_TYPE_NAMES.get(ggml_type, ggml_type)}, "
            f"not F32/F16. Use get_tensor() for quantized types."
        )

    def list_tensors(self) -> List[str]:
        return list(self._tensor_index.keys())

    def tensor_info(self, name: str) -> Optional[dict]:
        return self._tensor_index.get(name)

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def n_tensors(self) -> int:
        return len(self._tensor_index)

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def file_size(self) -> int:
        return os.path.getsize(self.path)

    def get_architecture(self) -> str:
        for key in ("general.architecture", "architecture"):
            if key in self._metadata:
                return str(self._metadata[key])
        return "unknown"

    def get_raw_tensor_data(self, name: str) -> np.ndarray:
        """Read raw tensor bytes WITHOUT dequantization.

        For F32/F16 tensors, returns a zero-copy view of the mmap.
        For quantized types (Q4_K, etc.), returns raw uint8 bytes
        in the on-disk GGML block format.

        This is the key method for quantized-domain conversion:
        it avoids the FP32 dequantization expansion entirely.
        """
        ti = self._tensor_index.get(name)
        if ti is None:
            raise KeyError(f"Tensor {name!r} not found")

        ggml_type = ti["ggml_type"]
        offset = int(ti["offset"]) + self._parser.tensor_data_offset
        shape = ti["shape"]
        n_elements = int(ti["n_elements"])

        if ggml_type == GGML_TYPE_F32:
            return np.ndarray(
                (n_elements,), dtype=np.float32, buffer=self._mmap_obj, offset=offset
            ).reshape(shape)

        if ggml_type == GGML_TYPE_F16:
            return np.ndarray(
                (n_elements,), dtype=np.float16, buffer=self._mmap_obj, offset=offset
            ).reshape(shape)

        data_size = int(ti["data_size"])
        return np.frombuffer(
            self._mmap_obj, dtype=np.uint8, offset=offset, count=data_size
        )

    def __enter__(self) -> "GGUFReader":
        return self.open()

    def __exit__(self, *a):
        self.close()


# ═══════════════════════════════════════════════════════════════════════════
# SSF Writer — SpectralStream Format output
# ═══════════════════════════════════════════════════════════════════════════


class SSFWriter:
    """Write compressed tensors in SSF (SpectralStream Format) using canonical SSFFormatSpec.

    Layout (from ssf_format_pipeline):
      [Header 256B]  magic, version, min_compat, flags, n_tensors, sizes, metadata ptr, checksum
      [Tensor Data]  compressed blocks, 4KB-aligned for MMAP
      [Index Size]   uint64
      [Tensor Index] name, shape, dtype, comp_type, flags, quality, offset, size, checksum
      [Meta Size]    uint64
      [Metadata]     JSON
      [Footer 64B]   index_offset, index_size, file_checksum, format_version
    """

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self._entries: List[SSFTensorIndexEntry] = []
        self._data_blocks: List[bytes] = []
        self._metadata: Dict[str, Any] = {}
        self._total_original = 0
        self._total_compressed = 0

    def set_metadata(self, key: str, value: Any):
        self._metadata[key] = value

    def write_tensor(
        self,
        name: str,
        shape: List[int],
        compressed_data: bytes,
        checksum: Optional[str] = None,
        original_size: Optional[int] = None,
    ):
        if original_size is None:
            original_size = int(np.prod(shape)) * 4
        # Prepend compression type byte for SSFReader compatibility
        data_with_type = struct.pack("<B", CompressionType.RAW) + compressed_data
        self._total_original += original_size
        self._total_compressed += len(data_with_type)
        entry = SSFTensorIndexEntry(
            name=name,
            shape=tuple(shape),
            dtype=np.dtype("float32"),
            compression_type=CompressionType.RAW,
            flags=0,
            n_quality_levels=0,
            data_offset=0,
            compressed_size=len(data_with_type),
            original_size=original_size,
            checksum=_sha256(data_with_type),
        )
        self._entries.append(entry)
        self._data_blocks.append(data_with_type)

    def finalize(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._entries.sort(key=lambda e: e.name)
        data_offset = SSF_HEADER_SIZE

        data_blocks_aligned = bytearray()
        for i, blk in enumerate(self._data_blocks):
            self._entries[i].data_offset = data_offset
            padded = _align_up(len(blk), SSF_ALIGNMENT)
            data_blocks_aligned += blk.ljust(padded, b"\x00")
            data_offset += padded

        idx_data = SSFFormatSpec.encode_tensor_index(self._entries)
        idx_size = len(idx_data)
        idx_offset = data_offset

        meta_json = json.dumps(self._metadata, indent=2).encode("utf-8")
        meta_size = len(meta_json)
        meta_offset = idx_offset + 8 + idx_size + 8

        all_data = bytearray()
        all_data += SSFFormatSpec.encode_header(
            len(self._entries),
            self._total_original,
            self._total_compressed,
            meta_offset,
            meta_size,
            0,
        )
        all_data += data_blocks_aligned
        all_data += struct.pack("<Q", idx_size)
        all_data += idx_data
        all_data += struct.pack("<Q", meta_size)
        all_data += meta_json

        payload_bytes = bytes(all_data[SSF_HEADER_SIZE:])
        file_checksum = _sha256(payload_bytes)
        footer = SSFFormatSpec.encode_footer(idx_offset, idx_size, file_checksum, 0)
        all_data += footer

        self.output_path.write_bytes(bytes(all_data))


# ═══════════════════════════════════════════════════════════════════════════
# GGUF Converter — Main conversion engine
# ═══════════════════════════════════════════════════════════════════════════


class GGUFConverter:
    """Convert GGUF models to SSF format using the unified quantizer pipeline.

    Features:
      - All GGML quantization types dequantized to FP32
      - Parallel tensor conversion
      - Per-tensor quality validation (SNR, PSNR, max error)
      - JSON + human-readable compression reports
    """

    def __init__(
        self,
        quality: float = 0.95,
        tt_relative_error: float = 0.01,
        block_variance_threshold: float = 0.01,
        max_workers: int = 4,
        target_ratio: Optional[float] = None,
        min_quality_snr: float = 20.0,
    ):
        self.quality = quality
        self.tt_relative_error = tt_relative_error
        self.block_variance_threshold = block_variance_threshold
        self.max_workers = max_workers
        self.target_ratio = target_ratio
        self.min_quality_snr = min_quality_snr

    def convert(
        self, gguf_path: str, output_path: Optional[str] = None, verbose: bool = True
    ) -> ConversionReport:
        """Convert a single GGUF file to SSF format.

        Parameters
        ----------
        gguf_path : str
            Path to input GGUF file.
        output_path : str, optional
            Path for output SSF file. Defaults to <input>.ssf.
        verbose : bool
            Print progress and per-tensor stats.

        Returns
        -------
        ConversionReport with full compression statistics.
        """
        gguf_path = str(Path(gguf_path).resolve())
        if output_path is None:
            output_path = str(Path(gguf_path).with_suffix(".ssf"))

        t_start = time.perf_counter()

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"GGUF Converter — SpectralStream Format")
            print(f"{'=' * 60}")
            print(f"Input:  {gguf_path}")
            print(f"Output: {output_path}")
            print()

        # Open GGUF reader
        reader = GGUFReader(gguf_path)
        reader.open()

        arch = reader.get_architecture()
        input_size = reader.file_size
        tensor_names = reader.list_tensors()

        if verbose:
            print(f"Architecture: {arch}")
            print(f"Tensors: {len(tensor_names)}")
            print(f"Layers: {reader.n_layers}")
            print(f"Input size: {format_size(input_size)}")
            print()

        # Prepare quantizer and writer
        quantizer = UnifiedQuantizer(
            quality=self.quality,
            tt_relative_error=self.tt_relative_error,
            block_variance_threshold=self.block_variance_threshold,
        )
        writer = SSFWriter(output_path)
        writer.set_metadata("source_file", Path(gguf_path).name)
        writer.set_metadata("architecture", arch)
        writer.set_metadata("quality", self.quality)
        writer.set_metadata("tt_relative_error", self.tt_relative_error)
        writer.set_metadata("n_layers", reader.n_layers)
        writer.set_metadata("n_tensors", len(tensor_names))

        # Convert tensors
        all_stats: List[TensorStats] = []
        converted = 0
        skipped = 0
        total_compressed = 0
        total_snr = 0.0
        total_psnr = 0.0
        total_mse = 0.0
        global_max_error = 0.0
        snr_count = 0
        errors: List[str] = []

        def convert_one(name: str) -> Optional[TensorStats]:
            nonlocal converted, skipped, total_compressed
            nonlocal total_snr, total_psnr, total_mse, global_max_error, snr_count

            t0 = time.perf_counter()
            stats = TensorStats(
                name=name,
                shape=[],
                ggml_type_name="",
                fp32_bytes=0,
                gguf_bytes=0,
                compressed_bytes=0,
                ratio_vs_fp32=1.0,
                ratio_vs_gguf=1.0,
            )

            try:
                ti = reader.tensor_info(name)
                if ti is None:
                    stats.error = "tensor info not found"
                    return stats

                tensor = reader.get_tensor(name)
                stats.shape = list(tensor.shape)
                stats.ggml_type_name = GGML_TYPE_NAMES.get(
                    ti["ggml_type"], f"type_{ti['ggml_type']}"
                )
                stats.fp32_bytes = tensor.nbytes
                stats.gguf_bytes = int(ti["data_size"])

                # Skip very small tensors (store raw)
                if tensor.ndim < 2 or tensor.size < 64:
                    compressed_data = tensor.astype(np.float32).tobytes()
                    writer.write_tensor(name, list(tensor.shape), compressed_data)
                    stats.compressed_bytes = len(compressed_data)
                    stats.ratio_vs_fp32 = stats.fp32_bytes / max(
                        stats.compressed_bytes, 1
                    )
                    stats.ratio_vs_gguf = stats.gguf_bytes / max(
                        stats.compressed_bytes, 1
                    )
                    stats.convert_time_ms = (time.perf_counter() - t0) * 1000
                    converted += 1
                    return stats

                # Cap tensor to valid range to avoid NaN in SVD/DCT
                safe_tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)

                # Fast path: store very large tensors as FP16 (good quality, 2x FP32 size)
                if safe_tensor.size > 5000000:
                    compressed_data = safe_tensor.astype(np.float16).tobytes()
                    writer.write_tensor(name, list(safe_tensor.shape), compressed_data)
                    stats.compressed_bytes = len(compressed_data)
                    stats.ratio_vs_fp32 = stats.fp32_bytes / max(
                        stats.compressed_bytes, 1
                    )
                    stats.ratio_vs_gguf = stats.gguf_bytes / max(
                        stats.compressed_bytes, 1
                    )
                    stats.convert_time_ms = (time.perf_counter() - t0) * 1000
                    converted += 1
                    total_compressed += stats.compressed_bytes
                    if verbose:
                        print(
                            f"  {name:50s} {str(safe_tensor.shape):20s} "
                            f"{stats.ggml_type_name:8s} "
                            f"{stats.ratio_vs_fp32:7.1f}:1 "
                            f"SNR=    FP16 "
                            f"({stats.convert_time_ms:.0f}ms)"
                        )
                    return stats

                # Compress through unified pipeline
                compressed = quantizer.compress(safe_tensor, layer_name=name)

                # Validate: decompress and compute quality metrics
                decompressed = quantizer.decompress(compressed)
                metrics = quantizer.compute_quality_metrics(safe_tensor, decompressed)
                stats.mse = metrics["mse"]
                stats.psnr = metrics["psnr"]
                stats.ssim = metrics["ssim"]
                stats.max_error = metrics["max_abs_error"]

                # SNR (safe computation)
                signal_power = float(np.mean(np.abs(safe_tensor.astype(np.float64))))
                noise_rmse = math.sqrt(max(metrics["mse"], 1e-30))
                if signal_power > 1e-30 and noise_rmse > 1e-30:
                    stats.snr_db = 20.0 * math.log10(signal_power / noise_rmse)
                else:
                    stats.snr_db = 0.0

                # Serialize compressed data (reuse already-compressed dict)
                compressed_data = quantizer.serialize_compressed(compressed)
                writer.write_tensor(name, list(tensor.shape), compressed_data)
                stats.compressed_bytes = len(compressed_data)

                stats.ratio_vs_fp32 = stats.fp32_bytes / max(stats.compressed_bytes, 1)
                stats.ratio_vs_gguf = stats.gguf_bytes / max(stats.compressed_bytes, 1)

                stats.convert_time_ms = (time.perf_counter() - t0) * 1000

                # Accumulate (with NaN/inf guards)
                converted += 1
                total_compressed += stats.compressed_bytes
                if not (math.isnan(stats.snr_db) or math.isinf(stats.snr_db)):
                    total_snr += max(stats.snr_db, -100.0)
                if not (math.isnan(stats.psnr) or math.isinf(stats.psnr)):
                    total_psnr += max(stats.psnr, 0.0)
                mse_safe = (
                    stats.mse
                    if not (math.isnan(stats.mse) or math.isinf(stats.mse))
                    else 0.0
                )
                total_mse += min(mse_safe, 1e10)
                max_err_safe = (
                    stats.max_error
                    if not (math.isnan(stats.max_error) or math.isinf(stats.max_error))
                    else 0.0
                )
                global_max_error = max(global_max_error, min(max_err_safe, 1e10))
                if stats.snr_db > 0 and not (
                    math.isnan(stats.snr_db) or math.isinf(stats.snr_db)
                ):
                    snr_count += 1

                if verbose:
                    snr_str = f"{stats.snr_db:.1f}dB" if stats.snr_db > 0 else "N/A"
                    print(
                        f"  {name:50s} {str(tensor.shape):20s} "
                        f"{stats.ggml_type_name:8s} "
                        f"{stats.ratio_vs_fp32:7.1f}:1 "
                        f"SNR={snr_str:>8s} "
                        f"({stats.convert_time_ms:.0f}ms)"
                    )

            except Exception as e:
                stats.error = str(e)
                errors.append(f"{name}: {e}")
                if verbose:
                    print(f"  {name:50s} ERROR: {e}")

            return stats

        # Parallel conversion
        if verbose and len(tensor_names) > 1:
            print(
                f"Converting {len(tensor_names)} tensors (workers={self.max_workers})..."
            )
            print(
                f"{'Name':50s} {'Shape':20s} {'Type':8s} {'Ratio':>7s} {'Quality':>10s} {'Time':>7s}"
            )
            print("-" * 110)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(convert_one, name): name for name in tensor_names}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_stats.append(result)

        # Sort stats by name
        all_stats.sort(key=lambda s: s.name)

        # Finalize SSF
        writer.set_metadata("total_input_bytes", input_size)
        writer.set_metadata("total_compressed_bytes", total_compressed)
        writer.set_metadata("overall_ratio", input_size / max(total_compressed, 1))

        reader.close()
        writer.finalize()

        output_size = os.path.getsize(output_path)
        elapsed = time.perf_counter() - t_start

        avg_snr = total_snr / max(snr_count, 1)
        avg_psnr = total_psnr / max(converted, 1)
        avg_mse = total_mse / max(converted, 1)

        report = ConversionReport(
            input_path=gguf_path,
            output_path=output_path,
            model_name=Path(gguf_path).stem,
            architecture=arch,
            n_layers=reader.n_layers,
            total_tensors=len(tensor_names),
            converted_tensors=converted,
            skipped_tensors=skipped,
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            overall_ratio=input_size / max(output_size, 1),
            avg_snr_db=avg_snr,
            avg_psnr_db=avg_psnr,
            avg_mse=avg_mse,
            max_error_global=global_max_error,
            total_time_s=elapsed,
            tensor_stats=all_stats,
            errors=errors,
        )

        if verbose:
            self._print_report(report)

        return report

    def convert_batch(
        self,
        gguf_paths: List[str],
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> List[ConversionReport]:
        """Convert multiple GGUF files to SSF format.

        Parameters
        ----------
        gguf_paths : List[str]
            List of GGUF file paths.
        output_dir : str, optional
            Directory for output files. Same as input if None.
        verbose : bool
            Print progress.

        Returns
        -------
        List of ConversionReport objects.
        """
        reports: List[ConversionReport] = []

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Batch Conversion: {len(gguf_paths)} files")
            print(f"{'=' * 60}\n")

        for i, path in enumerate(gguf_paths, 1):
            if verbose:
                print(f"\n[{i}/{len(gguf_paths)}] Converting: {Path(path).name}")

            out_path = None
            if output_dir:
                out_dir = Path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = str(out_dir / Path(path).with_suffix(".ssf").name)

            try:
                report = self.convert(path, output_path=out_path, verbose=verbose)
                reports.append(report)
            except Exception as e:
                if verbose:
                    print(f"  FAILED: {e}")

        if verbose and len(reports) > 1:
            self._print_batch_summary(reports)

        return reports

    def _print_report(self, report: ConversionReport):
        print(f"\n{'=' * 60}")
        print(f"Compression Report: {report.model_name}")
        print(f"{'=' * 60}")
        print(f"Architecture:       {report.architecture}")
        print(f"Layers:             {report.n_layers}")
        print(
            f"Tensors:            {report.converted_tensors}/{report.total_tensors} converted"
        )
        print(f"Input:              {format_size(report.input_size_bytes)}")
        print(f"Output:             {format_size(report.output_size_bytes)}")
        print(f"Overall Ratio:      {report.overall_ratio:.1f}:1")
        print(f"Avg SNR:            {report.avg_snr_db:.1f} dB")
        print(f"Avg PSNR:           {report.avg_psnr_db:.1f} dB")
        print(f"Avg MSE:            {report.avg_mse:.6e}")
        print(f"Max Error:          {report.max_error_global:.6f}")
        print(f"Time:               {report.total_time_s:.1f}s")

        if report.errors:
            print(f"\nErrors ({len(report.errors)}):")
            for err in report.errors[:10]:
                print(f"  - {err}")

        # Per-type summary
        type_stats: Dict[str, Dict[str, float]] = {}
        for ts in report.tensor_stats:
            if ts.error:
                continue
            tn = ts.ggml_type_name
            if tn not in type_stats:
                type_stats[tn] = {"count": 0, "total_fp32": 0, "total_compressed": 0}
            type_stats[tn]["count"] += 1
            type_stats[tn]["total_fp32"] += ts.fp32_bytes
            type_stats[tn]["total_compressed"] += ts.compressed_bytes

        if type_stats:
            print(f"\nPer-Type Summary:")
            print(
                f"  {'Type':10s} {'Count':>6s} {'FP32':>10s} {'Compressed':>12s} {'Ratio':>8s}"
            )
            print(f"  {'-' * 50}")
            for tn, st in sorted(type_stats.items()):
                ratio = st["total_fp32"] / max(st["total_compressed"], 1)
                print(
                    f"  {tn:10s} {int(st['count']):6d} "
                    f"{format_size(int(st['total_fp32'])):>10s} "
                    f"{format_size(int(st['total_compressed'])):>12s} "
                    f"{ratio:7.1f}:1"
                )

        # Top 5 most compressed
        valid_stats = [
            ts for ts in report.tensor_stats if not ts.error and ts.ratio_vs_fp32 > 1
        ]
        if valid_stats:
            top = sorted(valid_stats, key=lambda s: s.ratio_vs_fp32, reverse=True)[:5]
            print(f"\nTop 5 Compression Ratios:")
            for ts in top:
                print(f"  {ts.name:50s} {ts.ratio_vs_fp32:7.1f}:1")

        print(f"\nOutput: {report.output_path}")
        print(f"{'=' * 60}")

    def _print_batch_summary(self, reports: List[ConversionReport]):
        print(f"\n{'=' * 60}")
        print(f"Batch Conversion Summary")
        print(f"{'=' * 60}")
        print(f"Files:         {len(reports)}")
        total_in = sum(r.input_size_bytes for r in reports)
        total_out = sum(r.output_size_bytes for r in reports)
        total_time = sum(r.total_time_s for r in reports)
        avg_ratio = total_in / max(total_out, 1)
        print(f"Total Input:   {format_size(total_in)}")
        print(f"Total Output:  {format_size(total_out)}")
        print(f"Average Ratio: {avg_ratio:.1f}:1")
        print(f"Total Time:    {total_time:.1f}s")
        print()

        for r in reports:
            status = "OK" if not r.errors else f"ERRORS({len(r.errors)})"
            print(
                f"  {r.model_name:40s} "
                f"{r.overall_ratio:7.1f}:1 "
                f"SNR={r.avg_snr_db:5.1f}dB "
                f"{status}"
            )
        print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
# Compression Report I/O
# ═══════════════════════════════════════════════════════════════════════════


def save_report_json(report: ConversionReport, path: str):
    """Save conversion report as JSON."""
    data = {
        "input_path": report.input_path,
        "output_path": report.output_path,
        "model_name": report.model_name,
        "architecture": report.architecture,
        "n_layers": report.n_layers,
        "total_tensors": report.total_tensors,
        "converted_tensors": report.converted_tensors,
        "skipped_tensors": report.skipped_tensors,
        "input_size_bytes": report.input_size_bytes,
        "output_size_bytes": report.output_size_bytes,
        "overall_ratio": report.overall_ratio,
        "avg_snr_db": report.avg_snr_db,
        "avg_psnr_db": report.avg_psnr_db,
        "avg_mse": report.avg_mse,
        "max_error_global": report.max_error_global,
        "total_time_s": report.total_time_s,
        "errors": report.errors,
        "tensor_stats": [
            {
                "name": ts.name,
                "shape": ts.shape,
                "ggml_type": ts.ggml_type_name,
                "fp32_bytes": ts.fp32_bytes,
                "gguf_bytes": ts.gguf_bytes,
                "compressed_bytes": ts.compressed_bytes,
                "ratio_vs_fp32": ts.ratio_vs_fp32,
                "ratio_vs_gguf": ts.ratio_vs_gguf,
                "mse": ts.mse,
                "psnr": ts.psnr,
                "ssim": ts.ssim,
                "max_error": ts.max_error,
                "snr_db": ts.snr_db,
                "convert_time_ms": ts.convert_time_ms,
                "error": ts.error,
            }
            for ts in report.tensor_stats
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_report_json(path: str) -> dict:
    """Load a conversion report from JSON."""
    with open(path, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Quick validation utility
# ═══════════════════════════════════════════════════════════════════════════


def validate_gguf(path: str, verbose: bool = True) -> dict:
    """Validate a GGUF file: parse, dequantize a sample, report stats."""
    results: Dict[str, Any] = {"path": path, "valid": False, "errors": []}

    if verbose:
        print(f"\nValidating: {path}\n" + "-" * 60)

    try:
        reader = GGUFReader(path)
        reader.open()

        results["architecture"] = reader.get_architecture()
        results["tensor_count"] = reader.n_tensors
        results["n_layers"] = reader.n_layers
        results["file_size_mb"] = reader.file_size / 1024**2

        if verbose:
            print(f"  Architecture: {results['architecture']}")
            print(f"  Tensors: {results['tensor_count']}")
            print(f"  Layers: {results['n_layers']}")
            print(f"  Size: {results['file_size_mb']:.1f} MB")

        # Sample dequantization
        tensor_names = reader.list_tensors()
        sample_name = None
        for name in tensor_names:
            ti = reader.tensor_info(name)
            if ti and ti["ggml_type"] not in (
                GGML_TYPE_F32,
                GGML_TYPE_F16,
                GGML_TYPE_BF16,
            ):
                sample_name = name
                break

        if sample_name:
            tensor = reader.get_tensor(sample_name)
            results["sample_tensor"] = sample_name
            results["sample_shape"] = list(tensor.shape)
            results["sample_stats"] = {
                "min": float(tensor.min()),
                "max": float(tensor.max()),
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
            }
            if verbose:
                ti = reader.tensor_info(sample_name)
                print(
                    f"\n  Sample: {sample_name} ({GGML_TYPE_NAMES.get(ti['ggml_type'], '?')})"
                )
                print(f"    Shape: {tensor.shape}")
                print(f"    Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
                print(f"    Mean:  {tensor.mean():.6f}, Std: {tensor.std():.6f}")

        reader.close()
        results["valid"] = True

    except Exception as e:
        results["errors"].append(str(e))

    if verbose:
        print("-" * 60)
        print(f"  VALID: {results['valid']}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    import sys

    args = sys.argv[1:]

    if not args:
        print("GGUF Converter — SpectralStream Format")
        print()
        print("Usage:")
        print(
            "  python -m spectralstream.gguf_converter convert model.gguf [-o output.ssf] [--quality 0.95]"
        )
        print("  python -m spectralstream.gguf_converter validate model.gguf")
        print(
            "  python -m spectralstream.gguf_converter batch dir/ [-o out_dir/] [--quality 0.95]"
        )
        print()
        print("Options:")
        print("  --quality FLOAT    Quality (0.0-1.0, default 0.95)")
        print("  --tt-error FLOAT   TT-SVD relative error (default 0.01)")
        print("  --workers INT      Parallel workers (default 4)")
        print("  -o PATH            Output path or directory")
        print("  --json PATH        Save report as JSON")
        return

    command = args[0]

    quality = 0.95
    tt_error = 0.01
    workers = 4
    output = None
    json_path = None

    i = 1
    while i < len(args):
        if args[i] == "--quality" and i + 1 < len(args):
            quality = float(args[i + 1])
            i += 2
        elif args[i] == "--tt-error" and i + 1 < len(args):
            tt_error = float(args[i + 1])
            i += 2
        elif args[i] == "--workers" and i + 1 < len(args):
            workers = int(args[i + 1])
            i += 2
        elif args[i] == "-o" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        elif args[i] == "--json" and i + 1 < len(args):
            json_path = args[i + 1]
            i += 2
        else:
            i += 1

    converter = GGUFConverter(
        quality=quality,
        tt_relative_error=tt_error,
        max_workers=workers,
    )

    if command == "convert":
        if len(args) < 2:
            print("Error: specify GGUF file path")
            return
        gguf_path = args[1]
        report = converter.convert(gguf_path, output_path=output, verbose=True)
        if json_path:
            save_report_json(report, json_path)
            print(f"\nReport saved to: {json_path}")

    elif command == "validate":
        if len(args) < 2:
            print("Error: specify GGUF file path")
            return
        validate_gguf(args[1], verbose=True)

    elif command == "batch":
        if len(args) < 2:
            print("Error: specify directory or list of GGUF files")
            return
        input_path = Path(args[1])
        if input_path.is_dir():
            gguf_files = sorted(input_path.glob("*.gguf"))
        else:
            gguf_files = [input_path]
        reports = converter.convert_batch(
            [str(f) for f in gguf_files],
            output_dir=output,
            verbose=True,
        )
        if json_path:
            all_data = []
            for r in reports:
                all_data.append(
                    {
                        "model": r.model_name,
                        "ratio": r.overall_ratio,
                        "snr_db": r.avg_snr_db,
                        "input_size": r.input_size_bytes,
                        "output_size": r.output_size_bytes,
                    }
                )
            with open(json_path, "w") as f:
                json.dump(all_data, f, indent=2)
            print(f"\nBatch report saved to: {json_path}")

    else:
        print(f"Unknown command: {command}")
        print("Use 'convert', 'validate', or 'batch'")


if __name__ == "__main__":
    main()
