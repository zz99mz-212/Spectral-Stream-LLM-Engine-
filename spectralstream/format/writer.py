from __future__ import annotations

from spectralstream.format._imports import (
    Any,
    BinaryIO,
    Dict,
    List,
    Optional,
    Path,
    gzip,
    json,
    np,
    re,
    struct,
)

from spectralstream.format.core import (
    SSF_HEADER_SIZE,
    SSF_PAGE_SIZE,
    SSF_FOOTER_SIZE,
    TensorDType,
    _align_up,
    _sha256,
)
from spectralstream.format.compression import _compress_via_engine
from spectralstream.format.header import SSFHeader
from spectralstream.format.index import TensorIndex, TensorIndexEntry

_PATH_TRAVERSAL_PATTERN = re.compile(r"\.\.(\\|/)")


def _validate_output_path(path: str) -> None:
    """Validate output path: non-empty string, no path traversal."""
    if not path or not isinstance(path, str):
        raise ValueError("Output path must be a non-empty string")
    if _PATH_TRAVERSAL_PATTERN.search(path):
        raise ValueError(f"Path traversal detected: {path!r}")


class SSFWriter:
    def __init__(
        self,
        path: str,
        metadata: Optional[dict] = None,
        compression_method: int = 350,
        flags: int = 0,
        num_workers: int = 1,
    ):
        _validate_output_path(path)
        self.path = Path(path)
        self._meta = metadata or {}
        self._base_method = compression_method
        self._flags = flags
        self._num_workers = num_workers
        self._f: Optional[BinaryIO] = None
        self._index = TensorIndex()
        self._total_original = 0
        self._total_compressed = 0
        self._finalized = False

    def __enter__(self) -> SSFWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w+b")
        self._f.write(b"\x00" * SSF_HEADER_SIZE)
        return self

    def __exit__(self, *args: Any) -> None:
        if not self._finalized:
            self.finalize()

    def add_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        method: Optional[int] = None,
        params: Optional[dict] = None,
        quality_metrics: Optional[dict] = None,
    ) -> dict:
        if self._f is None:
            raise RuntimeError("SSFWriter not opened (use with statement)")
        method_id = method if method is not None else self._base_method
        t = tensor.copy()
        tdtype = TensorDType.from_numpy(t.dtype)
        if t.dtype != tdtype.to_numpy():
            t = t.astype(tdtype.to_numpy())
        raw = t.tobytes()
        p = params or {}
        compressed, _ = _compress_via_engine(raw, method_id, p)
        checksum = _sha256(compressed)
        data_offset = _align_up(self._f.tell(), SSF_PAGE_SIZE)
        pad = data_offset - self._f.tell()
        if pad:
            self._f.write(b"\x00" * pad)
        self._f.write(compressed)
        qm = (
            dict(quality_metrics)
            if quality_metrics
            else {
                "relative_error": 0.0,
                "snr_db": float("inf"),
                "psnr_db": float("inf"),
                "cosine_similarity": 1.0,
                "compression_ratio": len(raw) / max(len(compressed), 1),
            }
        )
        entry = TensorIndexEntry(
            name=name,
            shape=t.shape,
            dtype=tdtype,
            compression_method=method_id,
            compression_params=p,
            data_offset=data_offset,
            compressed_size=len(compressed),
            original_size=len(raw),
            quality_metrics=qm,
            checksum=checksum,
            flags=self._flags,
        )
        self._index.add(entry)
        self._total_original += len(raw)
        self._total_compressed += len(compressed)
        ratio = len(raw) / max(len(compressed), 1)
        return {
            "name": name,
            "shape": list(t.shape),
            "dtype": tdtype.name,
            "original_size": len(raw),
            "compressed_size": len(compressed),
            "ratio": round(ratio, 1),
            "method_id": method_id,
        }

    def _build_meta(self) -> bytes:
        meta: Dict[str, Any] = dict()
        meta.update(self._meta)
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

    def _page_align_write(self, data: bytes) -> None:
        if self._f is None:
            raise RuntimeError("SSFWriter not opened")
        pad = _align_up(self._f.tell(), SSF_PAGE_SIZE) - self._f.tell()
        if pad:
            self._f.write(b"\x00" * pad)
        self._f.write(data)

    def write_tensor_stream(
        self,
        name: str,
        tensor: np.ndarray,
        method: Optional[int] = None,
        params: Optional[dict] = None,
        quality_metrics: Optional[dict] = None,
    ) -> dict:
        """
        Streaming variant of add_tensor — writes raw bytes to the file handle
        immediately without holding a copy in memory longer than needed.
        Designed for one-tensor-at-a-time compression pipelines.
        """
        if self._f is None:
            raise RuntimeError("SSFWriter not opened (use with statement)")
        method_id = method if method is not None else self._base_method
        tdtype = TensorDType.from_numpy(tensor.dtype)

        raw = tensor.tobytes()
        p = params or {}
        compressed, _ = _compress_via_engine(raw, method_id, p)
        checksum = _sha256(compressed)
        data_offset = _align_up(self._f.tell(), SSF_PAGE_SIZE)
        pad = data_offset - self._f.tell()
        if pad:
            self._f.write(b"\x00" * pad)
        self._f.write(compressed)

        qm = (
            dict(quality_metrics)
            if quality_metrics
            else {
                "relative_error": 0.0,
                "snr_db": float("inf"),
                "psnr_db": float("inf"),
                "cosine_similarity": 1.0,
                "compression_ratio": len(raw) / max(len(compressed), 1),
            }
        )
        entry = TensorIndexEntry(
            name=name,
            shape=tensor.shape,
            dtype=tdtype,
            compression_method=method_id,
            compression_params=p,
            data_offset=data_offset,
            compressed_size=len(compressed),
            original_size=len(raw),
            quality_metrics=qm,
            checksum=checksum,
            flags=self._flags,
        )
        self._index.add(entry)
        self._total_original += len(raw)
        self._total_compressed += len(compressed)

        return {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": tdtype.name,
            "original_size": self._total_original,
            "compressed_size": self._total_compressed,
            "ratio": round(self._total_original / max(self._total_compressed, 1), 1),
            "method_id": method_id,
        }

    def finalize(self) -> dict:
        if self._finalized or self._f is None:
            return {}
        self._f.flush()
        idx_data = self._index.pack()
        idx_offset = _align_up(self._f.tell(), SSF_PAGE_SIZE)
        self._page_align_write(idx_data)
        meta_compressed = self._build_meta()
        meta_offset = _align_up(self._f.tell(), SSF_PAGE_SIZE)
        self._page_align_write(meta_compressed)
        file_end_before_footer = self._f.tell() if self._f else 0
        self._f.seek(0)
        h = SSFHeader(
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
        self._f.write(h.pack())
        file_end_before_footer = max(file_end_before_footer, SSF_HEADER_SIZE)
        self._f.seek(file_end_before_footer)
        self._f.seek(0)
        file_checksum = _sha256(self._f.read(file_end_before_footer))
        footer = struct.pack(
            f"<32sQQQ{SSF_FOOTER_SIZE - 56}s",
            file_checksum,
            idx_offset,
            meta_offset,
            self._total_original,
            b"\x00" * (SSF_FOOTER_SIZE - 56),
        )
        self._f.write(footer)
        file_end = self._f.tell()
        self._f.close()
        self._finalized = True
        r = self._total_original / max(self._total_compressed, 1)
        return {
            "path": str(self.path),
            "file_size": file_end,
            "n_tensors": len(self._index),
            "ratio": round(r, 1),
        }
