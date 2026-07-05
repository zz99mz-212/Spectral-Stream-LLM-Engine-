"""
Chunked Compressor — compress large tensors in fixed-size chunks.
O(chunk_size) RAM per tensor instead of loading the full tensor.
"""

from __future__ import annotations

import gc
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ChunkedCompressor:
    """Compress large tensors in chunks — O(chunk_size) RAM per tensor.

    For a 10GB tensor, processes in 64MB chunks:
    - RAM usage: ~64MB for current chunk + ~100MB for compressor state
    - Total: ~200MB instead of 10GB+ decompress buffer

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
        Reference to the compression engine (for method selection + compression)
    chunk_size_mb : int
        Size of each chunk in megabytes (default: 64)
    """

    _CHUNK_HEADER_FMT = "<II"  # chunk_index, chunk_nbytes
    _CHUNK_HEADER_SIZE = struct.calcsize(_CHUNK_HEADER_FMT)

    def __init__(self, engine: Any, chunk_size_mb: int = 64) -> None:
        self._engine = engine
        self._chunk_size_mb: int = chunk_size_mb
        self._metadata: Dict[str, Any] = {}

    def compress_chunked(
        self,
        name: str,
        tensor_view: np.ndarray,
        target_ratio: float,
        max_error: float,
    ) -> Tuple[bytes, dict, float, float]:
        """Compress a tensor in chunks and merge results.

        Parameters
        ----------
        name : str
            Tensor name (for profiling and method selection)
        tensor_view : np.ndarray
            Memory-mapped or regular numpy array
        target_ratio : float
            Desired compression ratio
        max_error : float
            Maximum acceptable relative error

        Returns
        -------
        compressed_data : bytes
            Merged compressed data with chunk headers
        metadata : dict
            Compression metadata including per-chunk info
        overall_ratio : float
            Achieved compression ratio
        overall_error : float
            Achieved relative error
        """
        flat = tensor_view.ravel()
        n = flat.size
        elem_size = flat.dtype.itemsize
        chunk_elems = max(1, (self._chunk_size_mb * 1024 * 1024) // elem_size)

        chunks: List[bytes] = []
        per_chunk_ratios: List[float] = []
        per_chunk_errors: List[float] = []
        chunk_index = 0

        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            chunk = np.asarray(flat[start:end], dtype=np.float32).reshape(1, -1)

            profile = self._engine.profiler.profile_tensor(
                chunk, name=f"{name}_chunk_{chunk_index}"
            )
            error_budget = max_error / max(target_ratio, 1.0)
            methods = self._engine._select_methods(profile, error_budget, target_ratio)
            data, meta, ratio_val, error_val = (
                self._engine.compress_tensor_with_validation(
                    chunk, profile, methods, error_budget
                )
            )

            header = struct.pack(self._CHUNK_HEADER_FMT, chunk_index, len(data))
            chunks.append(header + data)
            per_chunk_ratios.append(ratio_val)
            per_chunk_errors.append(error_val)

            del chunk, profile, data, meta
            if (chunk_index + 1) % 5 == 0:
                gc.collect()

            chunk_index += 1

        del flat
        gc.collect()

        merged = b"".join(chunks)
        total_ratio = float(tensor_view.nbytes / max(len(merged), 1))
        avg_error = float(np.mean(per_chunk_errors) if per_chunk_errors else max_error)

        self._metadata = {
            "method": "chunked",
            "num_chunks": len(chunks),
            "chunk_size_elems": chunk_elems,
            "original_shape": list(tensor_view.shape),
            "original_dtype": str(tensor_view.dtype),
            "per_chunk_ratios": per_chunk_ratios,
            "per_chunk_errors": per_chunk_errors,
            "original_nbytes": tensor_view.nbytes,
        }

        return merged, dict(self._metadata), total_ratio, avg_error

    def decompress_chunked(
        self, data: bytes, metadata: dict, original_shape: tuple
    ) -> np.ndarray:
        """Reconstruct tensor from chunked compression data.

        Parameters
        ----------
        data : bytes
            Merged compressed data with chunk headers
        metadata : dict
            Compression metadata from compress_chunked
        original_shape : tuple
            Original tensor shape for reconstruction

        Returns
        -------
        np.ndarray
            Reconstructed tensor (float32)
        """
        chunk_elems: int = metadata.get("chunk_size_elems", 0)
        num_chunks: int = metadata.get("num_chunks", 0)
        original_dtype: str = metadata.get("original_dtype", "float32")
        np_dtype: np.dtype = (
            np.dtype(original_dtype) if isinstance(original_dtype, str) else np.float32
        )
        total_elements = int(np.prod(original_shape))
        result = np.empty(total_elements, dtype=np.float32)

        offset = 0
        for i in range(num_chunks):
            if offset + self._CHUNK_HEADER_SIZE > len(data):
                raise ValueError("Corrupt chunked data: truncated header")
            chunk_idx, chunk_nbytes = struct.unpack(
                self._CHUNK_HEADER_FMT,
                data[offset : offset + self._CHUNK_HEADER_SIZE],
            )
            offset += self._CHUNK_HEADER_SIZE
            if offset + chunk_nbytes > len(data):
                raise ValueError("Corrupt chunked data: truncated chunk body")
            chunk_data = data[offset : offset + chunk_nbytes]
            offset += chunk_nbytes

            recon = self._decompress_chunk(chunk_data, chunk_idx)
            start = chunk_idx * chunk_elems
            end = min(start + recon.size, total_elements)
            result[start:end] = recon[: end - start]
            del recon

        return result.reshape(original_shape).astype(np.float32)

    def _decompress_chunk(self, chunk_data: bytes, chunk_index: int) -> np.ndarray:
        """Decompress a single chunk using the engine's decompress method."""
        try:
            recon = self._engine.decompress(chunk_data, {})
            if recon.size > 0:
                return recon
        except Exception:
            pass

        for _mname, inst in self._engine._methods.items():
            if not hasattr(inst, "decompress"):
                continue
            try:
                recon = inst.decompress(chunk_data, {})
                if recon.size > 0:
                    return recon
            except Exception:
                continue

        return np.frombuffer(chunk_data, dtype=np.float16).astype(np.float32)
