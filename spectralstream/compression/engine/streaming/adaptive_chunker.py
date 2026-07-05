from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from .memory_monitor import MemoryMonitor

CHUNK_HEADER_FMT = "<II"
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FMT)
_MIN_CHUNK_BYTES = 16 * 1024 * 1024  # 16 MB floor
_MAX_CHUNK_BYTES = 512 * 1024 * 1024  # 512 MB ceiling


@dataclass
class ChunkResult:
    """Result of compressing one chunk."""

    data: bytes = field(repr=False)
    chunk_index: int = 0
    ratio: float = 0.0
    error: float = 0.0
    method: str = ""
    nbytes_original: int = 0


class AdaptiveChunker:
    """Splits a tensor into memory-bounded chunks, compresses each chunk
    independently, and yields compressed results.

    Designed for tensors that are too large to fit in RAM (e.g., 30 GB
    embedding tables from a 365 GB model on a 64 GB machine).

    Parameters
    ----------
    engine : Any
        Compression engine (provides ``profiler.profile_tensor``,
        ``_select_methods``, ``compress_tensor_with_validation``).
    memory_monitor : MemoryMonitor
        Tracks RSS and signals GC / backpressure.
    chunk_size_mb : int, optional
        Override chunk size (MB). If not provided, uses
        ``memory_monitor.safe_chunk_size_bytes()``.
    dtype_override : np.dtype, optional
        Cast all chunks to this dtype before compression (default float32).
    """

    def __init__(
        self,
        engine: Any,
        memory_monitor: MemoryMonitor,
        chunk_size_mb: Optional[int] = None,
        dtype_override: Optional[np.dtype] = None,
    ) -> None:
        self._engine = engine
        self._monitor = memory_monitor
        self._dtype = dtype_override or np.float32

        if chunk_size_mb is not None:
            self._chunk_bytes: int = chunk_size_mb * 1024 * 1024
        else:
            self._chunk_bytes = memory_monitor.safe_chunk_size_bytes()

        self._chunk_bytes = max(
            _MIN_CHUNK_BYTES, min(self._chunk_bytes, _MAX_CHUNK_BYTES)
        )

        self._metadata: Dict[str, Any] = {}

    @property
    def chunk_bytes(self) -> int:
        return self._chunk_bytes

    @chunk_bytes.setter
    def chunk_bytes(self, value: int) -> None:
        self._chunk_bytes = max(_MIN_CHUNK_BYTES, min(value, _MAX_CHUNK_BYTES))

    def compute_chunk_elems(self, elem_size: int) -> int:
        """Number of elements per chunk for a given element size."""
        return max(1, self._chunk_bytes // elem_size)

    # ── Streaming chunk generator ──────────────────────────────────────

    def stream_chunks(
        self,
        tensor: np.ndarray,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield ``(chunk_index, chunk_array)`` pairs from *tensor*.

        Each chunk is a float32 ``np.ndarray`` no larger than
        ``chunk_bytes``. After yielding, the caller should process the
        chunk and let it be GC'd before the next iteration.
        """
        flat = tensor.ravel()
        n = flat.size
        elem_size = flat.dtype.itemsize
        chunk_elems = self.compute_chunk_elems(elem_size)

        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            chunk = np.asarray(flat[start:end], dtype=self._dtype).reshape(1, -1)
            yield chunk.shape[1], chunk
            del chunk
            self._monitor.maybe_gc()

        del flat

    def stream_array_chunks(
        self,
        arr: np.ndarray,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Same as ``stream_chunks`` but accepts a pre-raveled 1-D array."""
        n = arr.size
        elem_size = arr.dtype.itemsize
        chunk_elems = self.compute_chunk_elems(elem_size)

        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            chunk = np.asarray(arr[start:end], dtype=self._dtype).reshape(1, -1)
            yield chunk.shape[1], chunk
            del chunk
            self._monitor.maybe_gc()

    def stream_mmap_chunks(
        self,
        mmap_path: str,
        dtype_str: str,
        offset: int,
        nbytes: int,
        total_elements: int,
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Stream chunks directly from a memory-mapped file on disk.

        No full-tensor load — only ``chunk_bytes`` of RAM per iteration.
        Handles BF16 via on-the-fly conversion.
        """
        elem_size = (
            2
            if dtype_str == "BF16"
            else np.dtype(
                np.float32
                if dtype_str == "BF16"
                else (np.float32 if dtype_str == "F32" else np.float16)
            ).itemsize
        )

        mmap_dtype = np.uint16 if dtype_str == "BF16" else np.float32
        chunk_elems = self.compute_chunk_elems(elem_size)

        mm = None
        mmap_obj = None
        try:
            mm = np.memmap(
                mmap_path,
                dtype=mmap_dtype,
                mode="r",
                offset=offset,
                shape=(total_elements,),
            )
            mmap_obj = getattr(mm, "_mmap", None)
            for start in range(0, total_elements, chunk_elems):
                self._monitor.maybe_gc()

                end = min(start + chunk_elems, total_elements)
                raw = np.array(mm[start:end], dtype=np.float32)

                if dtype_str == "BF16":
                    raw = (raw.astype(np.uint32) << 16).view(np.float32)

                yield end - start, raw.reshape(1, -1)
                del raw
        finally:
            if mmap_obj is not None:
                try:
                    mmap_obj.close()
                except Exception:
                    pass
            if mm is not None:
                del mm
            self._monitor.maybe_gc(force=True)

    # ── Compress-chunked entry point ───────────────────────────────────

    def compress_chunked(
        self,
        name: str,
        tensor: np.ndarray,
        target_ratio: float,
        max_error: float,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> ChunkResult:
        """Compress a tensor in chunks, returning merged compressed data.

        Parameters
        ----------
        name : str
            Tensor name (for profiling).
        tensor : np.ndarray
            Full tensor (may be memory-mapped — chunks are copied).
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable per-chunk relative error.
        progress_cb : callable, optional
            ``f(chunk_index, total_chunks)`` called after each chunk.

        Returns
        -------
        ChunkResult
            Merged compressed data with chunk headers, per-chunk stats.
        """
        chunks: List[bytes] = []
        ratios: List[float] = []
        errors: List[float] = []
        orig_nbytes = tensor.nbytes

        total_chunks: int = 0
        for _, _ in self.stream_chunks(tensor):
            total_chunks += 1

        chunk_idx = 0
        for n_elems, chunk_arr in self.stream_chunks(tensor):
            error_budget = max_error / max(target_ratio, 1.0)
            profile = self._engine.profiler.profile_tensor(
                chunk_arr, name=f"{name}_chunk_{chunk_idx}"
            )
            methods = self._engine._select_methods(profile, error_budget, target_ratio)
            data, meta, ratio_val, error_val = (
                self._engine.compress_tensor_with_validation(
                    chunk_arr, profile, methods, error_budget
                )
            )

            header = struct.pack(CHUNK_HEADER_FMT, chunk_idx, len(data))
            chunks.append(header + data)
            ratios.append(ratio_val)
            errors.append(error_val)

            del chunk_arr, profile, data, meta
            self._monitor.maybe_gc()

            if progress_cb:
                progress_cb(chunk_idx + 1, total_chunks)

            chunk_idx += 1

        merged = b"".join(chunks)
        overall_ratio = float(orig_nbytes / max(len(merged), 1))
        avg_error = float(np.mean(errors)) if errors else max_error

        self._metadata = {
            "method": "chunked",
            "num_chunks": len(chunks),
            "chunk_bytes": self._chunk_bytes,
            "original_nbytes": orig_nbytes,
            "per_chunk_ratios": ratios,
            "per_chunk_errors": errors,
        }

        return ChunkResult(
            data=merged,
            chunk_index=0,
            ratio=overall_ratio,
            error=avg_error,
            method="chunked",
            nbytes_original=orig_nbytes,
        )

    def compress_mmap_chunked(
        self,
        name: str,
        mmap_path: str,
        dtype_str: str,
        offset: int,
        nbytes: int,
        shape: Tuple[int, ...],
        target_ratio: float,
        max_error: float,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> ChunkResult:
        """Compress a tensor via mmap streaming — never loads the full tensor.

        Parameters are the same as ``compress_chunked`` but the tensor is
        read directly from a safetensors file via ``MemoryMappedTensorEngine``
        chunking instead of being passed as a ``np.ndarray``.
        """
        elem_size = (
            2
            if dtype_str == "BF16"
            else np.dtype(np.float32 if dtype_str in ("F32",) else np.float16).itemsize
        )
        total_elements = nbytes // elem_size

        chunks: List[bytes] = []
        ratios: List[float] = []
        errors: List[float] = []

        first_pass = 0
        for _, _ in self.stream_mmap_chunks(
            mmap_path, dtype_str, offset, nbytes, total_elements
        ):
            first_pass += 1

        chunk_idx = 0
        for n_elems, chunk_arr in self.stream_mmap_chunks(
            mmap_path, dtype_str, offset, nbytes, total_elements
        ):
            error_budget = max_error / max(target_ratio, 1.0)
            profile = self._engine.profiler.profile_tensor(
                chunk_arr, name=f"{name}_chunk_{chunk_idx}"
            )
            methods = self._engine._select_methods(profile, error_budget, target_ratio)
            data, meta, ratio_val, error_val = (
                self._engine.compress_tensor_with_validation(
                    chunk_arr, profile, methods, error_budget
                )
            )

            header = struct.pack(CHUNK_HEADER_FMT, chunk_idx, len(data))
            chunks.append(header + data)
            ratios.append(ratio_val)
            errors.append(error_val)

            del chunk_arr, profile, data, meta
            self._monitor.maybe_gc()

            if progress_cb:
                progress_cb(chunk_idx + 1, first_pass)

            chunk_idx += 1

        merged = b"".join(chunks)
        overall_ratio = float(nbytes / max(len(merged), 1))
        avg_error = float(np.mean(errors)) if errors else max_error

        self._metadata = {
            "method": "chunked",
            "num_chunks": len(chunks),
            "chunk_bytes": self._chunk_bytes,
            "original_shape": list(shape),
            "original_dtype": dtype_str,
            "per_chunk_ratios": ratios,
            "per_chunk_errors": errors,
        }

        return ChunkResult(
            data=merged,
            chunk_index=0,
            ratio=overall_ratio,
            error=avg_error,
            method="chunked",
            nbytes_original=nbytes,
        )
