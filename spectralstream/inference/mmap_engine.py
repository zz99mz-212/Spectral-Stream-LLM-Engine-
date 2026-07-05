from __future__ import annotations

import ctypes
import ctypes.util
import mmap as py_mmap
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_libc_path = ctypes.util.find_library("c")
_HAS_LIBC = _libc_path is not None
if _HAS_LIBC:
    try:
        _libc = ctypes.CDLL(_libc_path, use_errno=True)
        _libc.madvise.restype = ctypes.c_int
        _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    except (OSError, AttributeError):
        _HAS_LIBC = False

MADV_WILLNEED = 3
MADV_DONTNEED = 4
MADV_SEQUENTIAL = 1
MADV_RANDOM = 2
MADV_HUGEPAGE = 14


def _align_up(val: int, align: int) -> int:
    return ((val + align - 1) // align) * align


def _page_size() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError):
        return 4096


class MmapEngine:
    def __init__(
        self,
        path: str,
        access: int = py_mmap.ACCESS_READ,
        hint: int = MADV_SEQUENTIAL,
    ) -> None:
        self.path = Path(path)
        self.access = access
        self.hint = hint
        self._mmap: Optional[py_mmap.mmap] = None
        self._fd: Optional[int] = None
        self._file_size: int = 0
        self._lock = threading.Lock()
        self._tensor_index: Dict[str, Dict[str, Any]] = {}
        self._layer_index: Dict[int, List[str]] = {}
        self._n_layers: int = 0
        self._metadata: Dict[str, Any] = {}

    def open(self) -> MmapEngine:
        if self._mmap is not None:
            return self
        with self._lock:
            self._fd = os.open(str(self.path), os.O_RDONLY | os.O_CLOEXEC)
            self._file_size = os.fstat(self._fd).st_size
            self._mmap = py_mmap.mmap(self._fd, self._file_size, access=self.access)
            if _HAS_LIBC and self._file_size > 0:
                try:
                    _libc.madvise(
                        ctypes.c_void_p(
                            id(self._mmap) if hasattr(self._mmap, "nbytes") else 0
                        ),
                        ctypes.c_size_t(self._file_size),
                        ctypes.c_int(self.hint),
                    )
                except Exception:
                    pass
        return self

    def close(self) -> None:
        with self._lock:
            if self._mmap is not None:
                try:
                    self._mmap.close()
                except Exception:
                    pass
                self._mmap = None
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None

    def read_bytes(self, offset: int, count: int) -> bytes:
        if self._mmap is None:
            raise RuntimeError("MmapEngine not opened. Call open() first.")
        with self._lock:
            return bytes(self._mmap[offset : offset + count])

    def read_array(
        self,
        offset: int,
        count: int,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        if self._mmap is None:
            raise RuntimeError("MmapEngine not opened. Call open() first.")
        with self._lock:
            n_bytes = count * np.dtype(dtype).itemsize
            offset = _align_up(offset, 8)
            arr = np.ndarray((count,), dtype=dtype, buffer=self._mmap, offset=offset)
            return arr

    def read_tensor(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        offset: Optional[int] = None,
    ) -> np.ndarray:
        if offset is None:
            info = self._tensor_index.get(name)
            if info is None:
                raise KeyError(f"Tensor {name!r} not found in index")
            offset = info.get("offset", 0)
            if "shape" in info:
                shape = tuple(info["shape"])
            if "dtype" in info:
                dtype = np.dtype(info["dtype"])
        n_elements = int(np.prod(shape))
        return self.read_array(offset, n_elements, dtype).reshape(shape)

    def read_tensor_raw(self, offset: int, nbytes: int) -> np.ndarray:
        return self.read_array(offset, nbytes, np.uint8)

    def add_tensor_index(
        self,
        name: str,
        offset: int,
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        layer: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        info: Dict[str, Any] = {
            "name": name,
            "offset": offset,
            "shape": list(shape),
            "dtype": str(dtype),
            **kwargs,
        }
        self._tensor_index[name] = info
        if layer is not None:
            self._layer_index.setdefault(layer, []).append(name)
            self._n_layers = max(self._n_layers, layer + 1)

    def list_tensors(self) -> List[str]:
        return list(self._tensor_index.keys())

    def get_layer_tensors(self, layer_idx: int) -> List[str]:
        return self._layer_index.get(layer_idx, [])

    @property
    def n_tensors(self) -> int:
        return len(self._tensor_index)

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def file_size(self) -> int:
        return self._file_size

    @property
    def is_open(self) -> bool:
        return self._mmap is not None

    def madvise(self, hint: int, offset: int = 0, length: int = 0) -> bool:
        if not _HAS_LIBC or self._mmap is None:
            return False
        if length == 0:
            length = self._file_size
        try:
            buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._mmap))
            result = _libc.madvise(
                ctypes.c_void_p(buf_addr + offset),
                ctypes.c_size_t(length),
                ctypes.c_int(hint),
            )
            return result == 0
        except Exception:
            return False

    def prefetch(self, names: List[str]) -> None:
        for name in names:
            info = self._tensor_index.get(name)
            if info is None:
                continue
            offset = info.get("offset", 0)
            nbytes = info.get("nbytes", 0)
            if nbytes > 0:
                self.madvise(MADV_WILLNEED, offset, nbytes)

    def __enter__(self) -> MmapEngine:
        return self.open()

    def __exit__(self, *args: Any) -> None:
        self.close()


class MMapTensorStore:
    def __init__(
        self,
        path: Optional[str] = None,
        initial_size: int = 0,
        page_size: Optional[int] = None,
    ) -> None:
        self._page_size = page_size or _page_size()
        self._path = Path(path) if path else None
        self._fd: Optional[int] = None
        self._mmap: Optional[py_mmap.mmap] = None
        self._file_size: int = 0
        self._allocated: int = 0
        self._lock = threading.Lock()
        self._tensor_registry: Dict[str, Dict[str, Any]] = {}
        self._offset: int = 0
        if initial_size > 0:
            self._allocate(initial_size)
        elif path is not None:
            self._open_file(path)

    def _open_file(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        self._fd = os.open(str(p), os.O_RDWR | os.O_CLOEXEC)
        self._file_size = os.fstat(self._fd).st_size
        if self._file_size == 0:
            self._allocate(self._page_size)
        else:
            self._mmap = py_mmap.mmap(self._fd, self._file_size)
            self._offset = self._file_size

    def _allocate(self, size: int) -> None:
        aligned_size = _align_up(size, self._page_size)
        if self._path is not None:
            if self._fd is not None:
                os.close(self._fd)
            self._fd = os.open(
                str(self._path), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644
            )
            os.ftruncate(self._fd, aligned_size)
            if self._mmap is not None:
                self._mmap.close()
            self._mmap = py_mmap.mmap(self._fd, aligned_size)
        else:
            if self._mmap is not None:
                self._mmap.close()
            self._mmap = py_mmap.mmap(-1, aligned_size)
        self._file_size = aligned_size

    def _ensure_space(self, nbytes: int) -> int:
        aligned = _align_up(nbytes, self._page_size)
        if self._offset + aligned > self._file_size:
            new_size = max(
                self._file_size * 2, self._offset + aligned + self._page_size
            )
            self._allocate(new_size)
        offset = self._offset
        self._offset += aligned
        return offset

    def store_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        dtype: Optional[np.dtype] = None,
    ) -> Dict[str, Any]:
        if dtype is None:
            dtype = tensor.dtype
        data = np.asarray(tensor, dtype=dtype)
        nbytes = data.nbytes
        with self._lock:
            offset = self._ensure_space(nbytes)
            self._mmap[offset : offset + nbytes] = data.tobytes()
            info = {
                "name": name,
                "offset": offset,
                "shape": list(data.shape),
                "dtype": str(dtype),
                "nbytes": nbytes,
            }
            self._tensor_registry[name] = info
        return info

    def load_tensor(self, name: str) -> np.ndarray:
        info = self._tensor_registry.get(name)
        if info is None:
            raise KeyError(f"Tensor {name!r} not found in store")
        offset = info["offset"]
        shape = tuple(info["shape"])
        dtype = np.dtype(info["dtype"])
        nbytes = info["nbytes"]
        n_elements = nbytes // np.dtype(dtype).itemsize
        with self._lock:
            arr = np.ndarray(
                (n_elements,),
                dtype=dtype,
                buffer=self._mmap,
                offset=offset,
            )
            return arr.reshape(shape)

    def list_tensors(self) -> List[str]:
        return list(self._tensor_registry.keys())

    def tensor_info(self, name: str) -> Optional[Dict[str, Any]]:
        return self._tensor_registry.get(name)

    @property
    def n_tensors(self) -> int:
        return len(self._tensor_registry)

    @property
    def allocated_bytes(self) -> int:
        return self._offset

    @property
    def file_size(self) -> int:
        return self._file_size

    def sync(self) -> None:
        if self._mmap is not None:
            self._mmap.flush()

    def close(self) -> None:
        with self._lock:
            if self._mmap is not None:
                try:
                    self._mmap.flush()
                    self._mmap.close()
                except Exception:
                    pass
                self._mmap = None
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None

    def __enter__(self) -> MMapTensorStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
