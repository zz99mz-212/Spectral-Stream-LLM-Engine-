"""Streaming IO: safetensors parsing, SSF writer, checkpoint manager.

Dtype handling: tensors are kept in their native storage dtype (BF16 as uint16,
FP8 as uint8, etc.) instead of being expanded to float32.  Conversion to float32
happens just-in-time inside compression methods and is tracked for precision
loss accounting.  After decompression, tensors are converted back to their
native dtype.
"""

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

from spectralstream.core.math_primitives.bfloat16 import (
    bfloat16_to_float32,
    float32_to_bfloat16,
    is_bfloat16,
)
from spectralstream.core.math_primitives.dtype_detection import (
    dtype_is_bf16,
)

from ._constants import SAFETENSORS_HEADER_LEN

logger = logging.getLogger(__name__)


class _SafetensorsIO:
    """Zero-copy safetensors header parsing and tensor reading (SSD streaming).

    BF16 tensors are returned as uint16 arrays — the native on-disk format.
    Callers must use ``_to_float32_for_compress()`` before passing to
    compression methods, and ``_from_float32_after_decompress()`` to get
    the result back in uint16 BF16 format.
    """

    DTYPE_MAP = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.uint16,
        "bfloat16": np.uint16,
        "bf16": np.uint16,
        "F8_E4M3": np.uint8,
        "F8_E5M2": np.uint8,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
    }

    @staticmethod
    def _normalize_dtype(dtype_str: str) -> str:
        if dtype_str.lower() in ("bfloat16", "bf16"):
            return "BF16"
        return dtype_str.upper()

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
        """Read a tensor from safetensors, preserving BF16 as uint16.

        Returns
        -------
        np.ndarray
            For BF16: uint16 array (2 bytes/element — half the memory of float32).
            For other dtypes: native storage dtype.
        """
        dtype_str = self._normalize_dtype(dtype_str)
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
        # BF16: keep as uint16 — DO NOT expand to float32 (saves 50% memory)
        # The precision conversion happens just-in-time in compression methods.
        if shape:
            tensor = tensor.reshape(shape)
        return tensor

    def load_tensor(
        self, path: str, shape: tuple, dtype_str: str, offset: int, nbytes: int
    ) -> np.ndarray:
        """Zero-copy tensor loading via mmap (read-only).

        Unlike read(), this returns the raw mmap view without an extra np.array() copy.
        BF16 tensors are kept in uint16 storage — no float32 expansion.
        Caller must del the returned tensor and gc.collect() after use.
        """
        dtype_str = self._normalize_dtype(dtype_str)
        np_dtype = self.DTYPE_MAP.get(dtype_str, np.float32)
        mm = np.memmap(
            path,
            dtype=np_dtype,
            mode="r",
            offset=offset,
            shape=(nbytes // np.dtype(np_dtype).itemsize,),
        )
        if dtype_str == "BF16":
            # Zero-copy view of BF16 as uint16 (no float32 expansion)
            tensor = mm
        else:
            tensor = mm  # zero-copy view into mmap'd file
        if shape:
            tensor = tensor.reshape(shape)
        return tensor

    @staticmethod
    def _get_native_dtype_str(dtype_str: str) -> str:
        """Convert safetensors dtype to canonical native dtype string."""
        if dtype_str.upper() == "BF16":
            return "bfloat16"
        return dtype_str

    @staticmethod
    def to_float32_for_compress(tensor: np.ndarray, dtype_str: str = "") -> np.ndarray:
        """Convert a tensor to float32 for compression.

        Handles all native dtypes:
        - BF16 (uint16) → float32 via bit manipulation
        - FP8 (uint8) → float32 via astype
        - float16 → float32 via astype
        - float32 → unchanged (zero-copy)
        - integer types → float32 via astype

        This is the entry point that creates the temporary float32 copy —
        caller should free the original tensor to save memory.

        Parameters
        ----------
        tensor : np.ndarray
            Tensor to convert.
        dtype_str : str
            Original dtype string from safetensors header.
            If empty, checks tensor.dtype == np.uint16.

        Returns
        -------
        np.ndarray
            Float32 array for computation.
        """
        is_bf16 = dtype_str.upper() == "BF16" or (
            dtype_str == "" and tensor.dtype == np.uint16
        )
        if is_bf16:
            return bfloat16_to_float32(tensor)
        if tensor.dtype == np.float32:
            return tensor
        return tensor.astype(np.float32)

    @staticmethod
    def from_float32_after_decompress(
        tensor: np.ndarray,
        dtype_str: str = "",
        input_was_bf16: bool = False,
    ) -> np.ndarray:
        """Convert a float32 result back to native dtype.

        Parameters
        ----------
        tensor : np.ndarray
            Float32 result from compression method.
        dtype_str : str
            Original dtype string.
        input_was_bf16 : bool
            Explicit flag (checked first, then dtype_str).

        Returns
        -------
        np.ndarray
            Tensor in original native dtype.
        """
        is_bf16 = input_was_bf16 or (
            dtype_str.upper() == "BF16" if dtype_str else False
        )
        if is_bf16:
            return float32_to_bfloat16(tensor)
        dt_upper = dtype_str.upper() if dtype_str else ""
        if dt_upper == "F16":
            return tensor.astype(np.float16)
        if dt_upper in ("F8_E4M3", "F8_E5M2"):
            return tensor.astype(np.uint8)
        if dt_upper in ("I8", "I4"):
            return tensor.astype(np.int8)
        return tensor

    @staticmethod
    def _compute_precision_conversion_error(
        original_bf16: np.ndarray,
        compressed_float32: np.ndarray,
    ) -> float:
        """Compute the error introduced by BF16→float32→BF16 round-trip.

        This measures the precision loss from converting a BF16 tensor to
        float32 (for computation) and back.  Even with perfect compression,
        there is ~1 ULP of precision loss from the round-to-nearest-even.

        For verification: the error should be ~1e-7 or less.
        """
        reconverted = float32_to_bfloat16(compressed_float32)
        orig_f32 = bfloat16_to_float32(original_bf16)
        recon_f32 = bfloat16_to_float32(reconverted)
        diff = orig_f32.ravel() - recon_f32.ravel()
        return float(np.sqrt(np.mean(diff**2)))

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
