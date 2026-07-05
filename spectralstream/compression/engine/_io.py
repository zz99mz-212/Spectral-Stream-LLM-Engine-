"""Streaming IO: safetensors parsing, SSF writer, checkpoint manager."""

from __future__ import annotations

import gc
import importlib
import threading

from spectralstream.compression._imports import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    json,
    logging,
    np,
    os,
    struct,
    time,
)

try:
    import psutil as _psutil_mod

    HAS_PSUTIL = True
except ImportError:
    _psutil_mod = None  # type: ignore[assignment]
    HAS_PSUTIL = False

from ._constants import SAFETENSORS_HEADER_LEN

logger = logging.getLogger(__name__)


class _SafetensorsIO:
    """Zero-copy safetensors header parsing and tensor reading (SSD streaming)."""

    DTYPE_MAP = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.uint16,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
    }

    def __init__(self, use_mmap: bool = True) -> None:
        self._lock = threading.Lock()
        self._use_mmap = use_mmap

    def scan(self, path: str) -> Dict[str, Tuple[tuple, str, int, int]]:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(SAFETENSORS_HEADER_LEN))[0]
            header = json.loads(f.read(header_len))
        data_start = SAFETENSORS_HEADER_LEN + header_len
        result: Dict[str, Tuple[tuple, str, int, int]] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info.get("dtype", "F32")
            shape = tuple(info.get("shape", []))
            offsets = info.get("data_offsets", [0, 0])
            result[name] = (
                shape,
                dtype,
                data_start + offsets[0],
                offsets[1] - offsets[0],
            )
        return result

    def read(
        self, path: str, shape: tuple, dtype_str: str, offset: int, nbytes: int
    ) -> np.ndarray:
        np_dtype = self.DTYPE_MAP.get(dtype_str, np.float32)
        if self._use_mmap:
            mm = np.memmap(
                path,
                dtype=np_dtype,
                mode="r",
                offset=offset,
                shape=(nbytes // np.dtype(np_dtype).itemsize,),
            )
            tensor = np.array(mm, dtype=np_dtype)
        else:
            with self._lock, open(path, "rb") as f:
                f.seek(offset)
                raw = f.read(nbytes)
            tensor = np.frombuffer(raw, dtype=np_dtype)
        if dtype_str == "BF16":
            tensor = tensor.astype(np.uint32) << 16
            tensor = tensor.view(np.float32)
        if shape:
            tensor = tensor.reshape(shape)
        return tensor.astype(np.float32)

    def load_tensor(
        self, path: str, shape: tuple, dtype_str: str, offset: int, nbytes: int
    ) -> np.ndarray:
        """Zero-copy tensor loading via mmap (read-only).

        Unlike read(), this returns the raw mmap view without an extra np.array() copy.
        For BF16, an unavoidable conversion copy is made.
        Caller must del the returned tensor and gc.collect() after use.
        """
        np_dtype = self.DTYPE_MAP.get(dtype_str, np.float32)
        mm = np.memmap(
            path,
            dtype=np_dtype,
            mode="r",
            offset=offset,
            shape=(nbytes // np.dtype(np_dtype).itemsize,),
        )
        if dtype_str == "BF16":
            # BF16 requires conversion — unavoidable copy
            tensor = mm.astype(np.uint32) << 16
            tensor = tensor.view(np.float32)
            del mm
        else:
            tensor = mm  # zero-copy view into mmap'd file
        if shape:
            tensor = tensor.reshape(shape)
        return tensor

    def _stream_tensors(
        self, path: str
    ) -> Iterator[Tuple[str, np.ndarray, Tuple[int, ...], str, int, int]]:
        """Stream tensors from disk one at a time using mmap.

        Yields: (name, tensor, shape, dtype_str, offset, nbytes)
        Caller MUST delete tensor and gc.collect() after each iteration.
        """
        tensors = self.scan(path)
        for name, (shape, dt, off, nb) in tensors.items():
            tensor = self.load_tensor(path, shape, dt, off, nb)
            yield name, tensor, shape, dt, off, nb

    @staticmethod
    def _check_memory_threshold(min_available_gb: float = 2.0) -> bool:
        """Check if enough RAM is available. Runs GC if below threshold.
        Returns True if safe to proceed, False if below threshold.
        """
        if not HAS_PSUTIL:
            return True
        try:
            avail = _psutil_mod.virtual_memory().available
            avail_gb = avail / (1024**3)
            if avail_gb < min_available_gb:
                gc.collect()
                # Also clear numpy cache by iterating objects
                for obj in gc.get_objects():
                    if isinstance(obj, np.ndarray) and obj.size > 1_000_000:
                        del obj
                gc.collect()
                avail = _psutil_mod.virtual_memory().available
                avail_gb = avail / (1024**3)
                if avail_gb < min_available_gb * 0.5:
                    import time as _time

                    _time.sleep(1)  # backpressure
                    gc.collect()
                    return False
            return True
        except Exception:
            return True

    def get_tensor_info(
        self, path: str, name: str
    ) -> Optional[Tuple[tuple, str, int, int]]:
        tensors = self.scan(path)
        return tensors.get(name)

    def list_tensors(self, path: str) -> List[str]:
        return list(self.scan(path).keys())

    def estimate_model_size(self, path: str) -> int:
        total = 0
        for shape, dt, off, nb in self.scan(path).values():
            total += nb
        return total


class _SSFIOWriter:
    """Streaming SSF v2 writer that writes one tensor at a time."""

    def __init__(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        _ssf_mod = importlib.import_module("spectralstream.format." + "ssf_format")
        SSFWriter = _ssf_mod.SSFWriter

        self._writer = SSFWriter(path, metadata=metadata)
        self._writer.__enter__()

    def __enter__(self) -> _SSFIOWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self._writer.__exit__(*args)

    def add_tensor(self, name: str, tensor: np.ndarray) -> None:
        self._writer.add_tensor(name, tensor)

    def finalize(self) -> dict:
        return self._writer.finalize()

    def _dtype_from_str(self, s: str) -> Any:
        _ssf_mod = importlib.import_module("spectralstream.format." + "ssf_format")
        TensorDType = _ssf_mod.TensorDType

        m = {
            "float32": TensorDType.F32,
            "float16": TensorDType.F16,
            "int8": TensorDType.INT8,
        }
        return m.get(s, TensorDType.F32)

    def writer_finalize(self) -> dict:
        return self._writer.finalize()


class _CheckpointManager:
    """Checkpoint/resume for large model compression."""

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

    def save(
        self,
        completed: int,
        total: int,
        compressed_tensors: List[Dict[str, Any]],
        report_data: Dict[str, Any],
    ) -> None:
        data = {
            "completed": completed,
            "total": total,
            "timestamp": time.time(),
            "compressed_tensors": compressed_tensors,
            "report_data": report_data,
        }
        tmp_path = self.checkpoint_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f, default=str)
        os.replace(tmp_path, self.checkpoint_path)
        logger.info("Checkpoint saved: %d/%d tensors", completed, total)

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)
            logger.info(
                "Checkpoint loaded: %d/%d tensors",
                data.get("completed", 0),
                data.get("total", 0),
            )
            return data
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load checkpoint: %s", e)
            return None

    def clear(self) -> None:
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
