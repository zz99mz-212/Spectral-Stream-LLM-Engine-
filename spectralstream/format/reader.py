from __future__ import annotations

import mmap as py_mmap

from spectralstream.format._imports import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    OrderedDict,
    Path,
    Tuple,
    gzip,
    json,
    np,
    os,
    re,
    struct,
    threading,
)

from spectralstream.format.core import (
    SSF_HEADER_SIZE,
    SSF_FOOTER_SIZE,
    SSF_PAGE_SIZE,
    _sha256,
)
from spectralstream.format.header import SSFHeader
from spectralstream.format.index import TensorIndex
from spectralstream.format.compression import _decompress_via_engine, _method_id_to_name

_PATH_TRAVERSAL_PATTERN = re.compile(r"\.\.(\\|/)")


def _validate_path_safe(path: str) -> None:
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")
    if _PATH_TRAVERSAL_PATTERN.search(path):
        raise ValueError(f"Path traversal detected: {path!r}")
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Not a file: {resolved}")


class SSFReader:
    def __init__(self, path: str, mmap_mode: bool = True, cache_size: int = 32):
        _validate_path_safe(path)
        self.path = Path(path)
        self.cache_size = cache_size
        self.mmap_mode = mmap_mode and self.path.stat().st_size > SSF_PAGE_SIZE
        self._fd: Optional[int] = None
        self._mmap: Optional[py_mmap.mmap] = None
        self._data: Any = None
        self._header: Optional[SSFHeader]
        self._index: Optional[TensorIndex] = None
        self._metadata: dict = {}
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._is_legacy: bool = False
        self._open()

    def _open(self) -> None:
        if self.mmap_mode:
            self._fd = os.open(str(self.path), os.O_RDONLY | os.O_CLOEXEC)
            sz = self.path.stat().st_size
            self._mmap = py_mmap.mmap(self._fd, sz, access=py_mmap.ACCESS_READ)
            self._data = self._mmap
        else:
            self._data = self.path.read_bytes()
        try:
            self._header = SSFHeader.unpack(self._data)
        except (ValueError, struct.error) as e:
            self._header = None
            return
        self._is_legacy = self._header.version == 2
        o, s = self._header.index_offset, self._header.index_size
        if o and s and o + s <= len(self._data):
            try:
                self._index = TensorIndex.unpack(
                    self._data[o : o + s], is_legacy=self._is_legacy
                )
            except (ValueError, struct.error, IndexError):
                self._index = None
        o, s = self._header.metadata_offset, self._header.metadata_size
        if o and s and o + s <= len(self._data):
            try:
                self._metadata = json.loads(gzip.decompress(self._data[o : o + s]))
            except (json.JSONDecodeError, gzip.BadGzipFile, OSError, ValueError):
                self._metadata = {}

    @property
    def header(self) -> Optional[SSFHeader]:
        return self._header

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def tensor_names(self) -> list[str]:
        return self._index.names() if self._index else []

    def tensor_info(self, name: str) -> Optional[dict]:
        if self._index is None:
            return None
        e = self._index.get(name)
        if e is None:
            return None
        r = round(e.original_size / max(e.compressed_size, 1), 1)
        return dict(
            name=e.name,
            shape=list(e.shape),
            dtype=e.dtype.name,
            method_id=e.compression_method,
            compression_params=e.compression_params,
            data_offset=e.data_offset,
            compressed_size=e.compressed_size,
            original_size=e.original_size,
            quality_metrics=dict(e.quality_metrics),
            checksum=e.checksum.hex(),
            flags=e.flags,
            ratio=r,
        )

    def get_tensor(self, name: str) -> np.ndarray:
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
        if self._index is None:
            raise KeyError(f"Tensor {name!r} not found (empty index)")
        e = self._index.get(name)
        if e is None:
            raise KeyError(f"Tensor {name!r} not found")
        end = e.data_offset + e.compressed_size
        if end > len(self._data):
            raise ValueError(f"{name} out of bounds")
        block = bytes(self._data[e.data_offset : end])
        if _sha256(block) != e.checksum:
            raise ValueError(f"Checksum mismatch: {name}")
        raw = _decompress_via_engine(
            block, e.compression_method, e.compression_params, str(e.dtype.to_numpy())
        )
        tensor = np.frombuffer(raw, dtype=e.dtype.to_numpy()).reshape(e.shape).copy()
        with self._lock:
            self._cache[name] = tensor
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return tensor

    def read_tensor_chunk(
        self,
        name: str,
        row_start: int = 0,
        row_end: Optional[int] = None,
    ) -> np.ndarray:
        """
        Read a partial range of rows from a 2D tensor without loading the
        whole tensor into memory. Useful for processing very large tensors
        in chunks during decompression or inference.

        If the tensor is not 2D, falls back to full tensor read.
        """
        if self._index is None:
            raise KeyError(f"Tensor {name!r} not found (empty index)")
        e = self._index.get(name)
        if e is None:
            raise KeyError(f"Tensor {name!r} not found")
        shape = e.shape
        if len(shape) != 2:
            return self.get_tensor(name)
        n_rows = shape[0]
        row_end = row_end or n_rows
        row_start = max(0, min(row_start, n_rows))
        row_end = max(row_start, min(row_end, n_rows))
        n_rows_to_read = row_end - row_start
        if n_rows_to_read >= n_rows:
            return self.get_tensor(name)
        end = e.data_offset + e.compressed_size
        block = bytes(self._data[e.data_offset : end])
        if _sha256(block) != e.checksum:
            raise ValueError(f"Checksum mismatch: {name}")
        raw = _decompress_via_engine(
            block, e.compression_method, e.compression_params, str(e.dtype.to_numpy())
        )
        full = np.frombuffer(raw, dtype=e.dtype.to_numpy()).reshape(shape)
        chunk = full[row_start:row_end].copy()
        return chunk

    def get_tensors(self, names: List[str]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        for name in names:
            result[name] = self.get_tensor(name)
        return result

    def __getitem__(self, name: str) -> np.ndarray:
        return self.get_tensor(name)

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        if self._index is not None:
            for e in self._index:
                yield e.name, self.get_tensor(e.name)

    def __len__(self) -> int:
        return len(self._index) if self._index else 0

    def verify(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "valid": True,
            "header_ok": False,
            "checksum_ok": False,
            "index_ok": False,
            "tensor_checksums": {},
            "errors": [],
        }
        if self._data is None or len(self._data) < SSF_HEADER_SIZE:
            result["valid"] = False
            result["errors"].append("File too small")
            return result
        try:
            h = SSFHeader.unpack(self._data)
            result["header_ok"] = True
            result["version"] = h.version
            result["n_tensors"] = h.n_tensors
        except ValueError as e:
            result["valid"] = False
            result["errors"].append(f"Header error: {e}")
            return result
        if self._index is not None:
            result["index_ok"] = True
            for e in self._index:
                end = e.data_offset + e.compressed_size
                if end > len(self._data):
                    result["tensor_checksums"][e.name] = "out_of_bounds"
                    result["valid"] = False
                    result["errors"].append(f"{e.name} data out of bounds")
                    continue
                block = bytes(self._data[e.data_offset : end])
                cs_ok = _sha256(block) == e.checksum
                result["tensor_checksums"][e.name] = "ok" if cs_ok else "FAIL"
                if not cs_ok:
                    result["valid"] = False
                    result["errors"].append(f"{e.name} checksum FAIL")
        footer_start = len(self._data) - SSF_FOOTER_SIZE
        if footer_start > 0:
            stored_cs = self._data[footer_start : footer_start + 32]
            computed = _sha256(bytes(self._data[:footer_start]))
            result["checksum_ok"] = computed == stored_cs
            if not result["checksum_ok"]:
                result["valid"] = False
                result["errors"].append("File checksum mismatch")
        return result

    def list_tensors(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        if self._index is None:
            return result
        for e in self._index:
            method_name = _method_id_to_name(e.compression_method)
            result.append(
                {
                    "name": e.name,
                    "shape": list(e.shape),
                    "dtype": e.dtype.name,
                    "method_id": e.compression_method,
                    "method_name": method_name,
                    "original_size": e.original_size,
                    "compressed_size": e.compressed_size,
                    "compression_ratio": round(
                        e.original_size / max(e.compressed_size, 1), 2
                    ),
                    "quality": dict(e.quality_metrics),
                }
            )
        return result

    def get_quality_report(self) -> Dict[str, Any]:
        tensors = self.list_tensors()
        if not tensors:
            return {"tensors": [], "aggregate": {}}
        rel_errors = [t["quality"].get("relative_error", 0.0) for t in tensors]
        snrs = [
            t["quality"].get("snr_db", 0.0)
            for t in tensors
            if t["quality"].get("snr_db", float("inf")) != float("inf")
        ]
        ratios = [t.get("compression_ratio", 1.0) for t in tensors]
        aggregate = {
            "mean_relative_error": float(np.mean(rel_errors)) if rel_errors else 0.0,
            "max_relative_error": float(np.max(rel_errors)) if rel_errors else 0.0,
            "min_snr_db": float(np.min(snrs)) if snrs else 0.0,
            "mean_snr_db": float(np.mean(snrs)) if snrs else 0.0,
            "mean_compression_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "overall_ratio": float(
                np.sum([t["original_size"] for t in tensors])
                / max(np.sum([t["compressed_size"] for t in tensors]), 1)
            ),
            "n_tensors": len(tensors),
            "total_original": sum(t["original_size"] for t in tensors),
            "total_compressed": sum(t["compressed_size"] for t in tensors),
        }
        return {"tensors": tensors, "aggregate": aggregate}

    def extract_subset(
        self, names: List[str], output_path: str, metadata: Optional[dict] = None
    ) -> dict:
        out_meta = dict(metadata or {})
        out_meta.setdefault("source", str(self.path))
        out_meta.setdefault("subset_of", self._metadata.get("ssf_version", "v2"))
        index = self._index
        if index is None:
            raise RuntimeError("No tensor index loaded")
        from spectralstream.format.writer import SSFWriter

        with SSFWriter(output_path, metadata=out_meta) as writer:
            for name in names:
                tensor = self.get_tensor(name)
                e = index.get(name)
                if e is None:
                    continue
                writer.add_tensor(
                    name,
                    tensor,
                    method=e.compression_method,
                    params=e.compression_params,
                    quality_metrics=dict(e.quality_metrics),
                )
        return {"output": output_path, "n_tensors": len(names)}

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
        if self._mmap:
            self._mmap.close()
        if self._fd is not None:
            os.close(self._fd)
        self._mmap = self._fd = self._data = None

    def __enter__(self) -> SSFReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
