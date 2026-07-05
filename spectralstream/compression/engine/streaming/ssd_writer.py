from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from spectralstream.format.core import (
    SSF_FOOTER_SIZE,
    SSF_HEADER_SIZE,
    SSF_PAGE_SIZE,
    TensorDType,
    _align_up,
    _sha256,
)
from spectralstream.format.header import SSFHeader
from spectralstream.format.index import TensorIndex, TensorIndexEntry

SSD_PAGE_SIZE = SSF_PAGE_SIZE  # 4096 bytes — matches Linux page size


@dataclass
class SSDWriteResult:
    """Result of finalizing an SSDWriter session."""

    path: str = ""
    n_tensors: int = 0
    total_original: int = 0
    total_compressed: int = 0
    file_size: int = 0
    ratio: float = 1.0


class SSDWriter:
    """Progressive SSF writer that flushes tensor data page-aligned to
    disk — no in-memory accumulation of compressed tensor blobs beyond
    the current tensor.

    Compared to ``SSFWriter`` (which buffers the index in memory and only
    writes to disk on ``finalize()``), this writer:

    * Writes each compressed tensor block *immediately* to the output file
      (page-aligned for optimal SSD / NVMe performance).
    * Keeps the ``TensorIndex`` in memory (tens of KB even for 10k tensors)
      but **not** the compressed data.
    * Writes index, metadata, and footer at the end (same as SSFWriter).

    Parameters
    ----------
    path : str
        Output ``.ssf`` path.
    metadata : dict, optional
        Top-level metadata to embed in the file.
    flags : int
        SSF header flags.
    page_size : int
        Alignment boundary for tensor data blocks (default 4096).
    """

    def __init__(
        self,
        path: str,
        metadata: Optional[dict] = None,
        flags: int = 0,
        page_size: int = SSD_PAGE_SIZE,
    ) -> None:
        self._path: str = path
        self._meta: dict = metadata or {}
        self._flags: int = flags
        self._page_size: int = page_size
        self._f: Optional[Any] = None
        self._index: TensorIndex = TensorIndex()
        self._total_original: int = 0
        self._total_compressed: int = 0
        self._finalized: bool = False
        self._header_offset: int = 0

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self) -> SSDWriter:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._f = open(self._path, "w+b")
        self._header_offset = self._f.tell()
        self._f.write(b"\x00" * SSF_HEADER_SIZE)
        self._f.flush()
        return self

    def __exit__(self, *args: Any) -> None:
        if not self._finalized:
            self.finalize()

    # ── Writing ─────────────────────────────────────────────────────────

    def write_tensor_block(
        self,
        compressed_data: bytes,
        name: str,
        shape: tuple,
        dtype: np.dtype,
        method_id: int = 350,
        params: Optional[dict] = None,
        quality_metrics: Optional[dict] = None,
        flags: int = 0,
    ) -> int:
        """Write a single compressed tensor block to disk, page-aligned.

        Returns the byte offset in the file where the block was written.
        The block is flushed immediately so it survives crashes up to
        this point.

        Parameters
        ----------
        compressed_data : bytes
            Already-compressed tensor data (raw bytes — do **not** pass
            an uncompressed numpy array).
        name : str
            Tensor name (for the index entry).
        shape : tuple
            Original tensor shape.
        dtype : np.dtype
            Original tensor dtype.
        method_id : int
            Compression method enum value.
        params : dict, optional
            Compression parameters.
        quality_metrics : dict, optional
            Quality metrics (relative error, SNR, etc.).
        flags : int
            Per-tensor flags.

        Returns
        -------
        int
            File offset where the compressed block starts.
        """
        if self._f is None:
            raise RuntimeError("SSDWriter not opened (use with statement)")

        dtype_obj = np.dtype(dtype)
        raw_size = int(np.prod(shape)) * dtype_obj.itemsize
        checksum = _sha256(compressed_data)

        # Page-align the write
        data_offset = _align_up(self._f.tell(), self._page_size)
        pad = data_offset - self._f.tell()
        if pad:
            self._f.write(b"\x00" * pad)

        self._f.write(compressed_data)
        self._f.flush()

        tdtype = TensorDType.from_numpy(dtype_obj)
        qm = dict(quality_metrics or {})
        qm.setdefault("relative_error", 0.0)
        qm.setdefault("compression_ratio", raw_size / max(len(compressed_data), 1))

        entry = TensorIndexEntry(
            name=name,
            shape=shape,
            dtype=tdtype,
            compression_method=method_id,
            compression_params=params or {},
            data_offset=data_offset,
            compressed_size=len(compressed_data),
            original_size=raw_size,
            quality_metrics=qm,
            checksum=checksum,
            flags=flags,
        )
        self._index.add(entry)
        self._total_original += raw_size
        self._total_compressed += len(compressed_data)

        return data_offset

    def write_chunked_tensor_block(
        self,
        chunk_results: List[bytes],
        name: str,
        shape: tuple,
        dtype: np.dtype,
        method_id: int = 350,
        params: Optional[dict] = None,
        quality_metrics: Optional[dict] = None,
    ) -> int:
        """Write a chunked-compressed tensor as one logical block.

        Each chunk in *chunk_results* is a self-contained compressed blob
        (with the chunk header already prepended). They are concatenated
        and written as a single page-aligned block. A single index entry
        is created.
        """
        merged = b"".join(chunk_results)
        return self.write_tensor_block(
            compressed_data=merged,
            name=name,
            shape=shape,
            dtype=dtype,
            method_id=method_id,
            params=params,
            quality_metrics=quality_metrics,
        )

    # ── Finalization ───────────────────────────────────────────────────

    def finalize(self) -> SSDWriteResult:
        """Write index, metadata, and footer; patch header.

        After finalize(), the file is fully valid SSF v3 and ready for
        ``SSFReader``.
        """
        if self._finalized or self._f is None:
            return SSDWriteResult(path=self._path)

        self._f.flush()
        os.fsync(self._f.fileno())

        # --- Pack and write index ---
        idx_data = self._index.pack()
        idx_offset = _align_up(self._f.tell(), self._page_size)
        self._page_align_write(idx_data)

        # --- Pack and write metadata ---
        meta_compressed = self._build_meta()
        meta_offset = _align_up(self._f.tell(), self._page_size)
        self._page_align_write(meta_compressed)

        file_end_before_footer = self._f.tell()

        # --- Patch header ---
        header = SSFHeader(
            n_tensors=len(self._index),
            index_offset=idx_offset,
            index_size=len(idx_data),
            metadata_offset=meta_offset,
            metadata_size=len(meta_compressed),
            tensor_data_offset=SSF_HEADER_SIZE,
            flags=self._flags,
            redundant_header_offset=0,
            footer_offset=file_end_before_footer,
        )
        self._f.seek(0)
        self._f.write(header.pack())
        self._f.flush()

        # --- Write footer checksum ---
        seek_pos = max(file_end_before_footer, SSF_HEADER_SIZE)
        self._f.seek(seek_pos)
        self._f.seek(0)
        file_checksum = _sha256(self._f.read(seek_pos))

        self._f.seek(seek_pos)
        footer = struct.pack(
            f"<32sQQQ{SSF_FOOTER_SIZE - 56}s",
            file_checksum,
            idx_offset,
            meta_offset,
            self._total_original,
            b"\x00" * (SSF_FOOTER_SIZE - 56),
        )
        self._f.write(footer)
        self._f.flush()
        os.fsync(self._f.fileno())

        file_end = self._f.tell()
        self._f.close()
        self._finalized = True

        ratio = self._total_original / max(self._total_compressed, 1)
        return SSDWriteResult(
            path=self._path,
            n_tensors=len(self._index),
            total_original=self._total_original,
            total_compressed=self._total_compressed,
            file_size=file_end,
            ratio=ratio,
        )

    # ── Internal helpers ───────────────────────────────────────────────

    def _page_align_write(self, data: bytes) -> None:
        if self._f is None:
            raise RuntimeError("SSDWriter not opened")
        pad = _align_up(self._f.tell(), self._page_size) - self._f.tell()
        if pad:
            self._f.write(b"\x00" * pad)
        self._f.write(data)
        self._f.flush()

    def _build_meta(self) -> bytes:
        import gzip
        import json

        meta: Dict[str, Any] = dict(self._meta)
        meta.update(
            ssf_version=3,
            n_tensors=len(self._index),
            total_original=self._total_original,
            total_compressed=self._total_compressed,
        )
        meta["tensors"] = [
            {
                "name": e.name,
                "shape": list(e.shape),
                "dtype": e.dtype.name,
                "method_id": e.compression_method,
                "original_size": e.original_size,
                "compressed_size": e.compressed_size,
                "quality": dict(e.quality_metrics),
            }
            for e in self._index
        ]
        return gzip.compress(json.dumps(meta).encode("utf-8"))

    def is_finalized(self) -> bool:
        return self._finalized

    def file_offset(self) -> int:
        """Current write position in the output file."""
        if self._f is None:
            return 0
        return self._f.tell()
