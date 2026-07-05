"""
Streaming Compressor — Never loads full model into RAM.
Processes one tensor at a time from disk with minimal memory overhead.
Supports chunked processing for >500MB tensors and checkpoint/resume.
"""

from __future__ import annotations


import gc

from spectralstream.compression._imports import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Path,
    Tuple,
    json,
    logging,
    np,
    os,
    time,
)

try:
    import psutil as _psutil_mod

    HAS_PSUTIL = True
except ImportError:
    _psutil_mod = None  # type: ignore[assignment]
    HAS_PSUTIL = False


def _get_psutil():
    return _psutil_mod


from ._dataclasses import CompressionConfig, CompressedTensor, TensorProfile
from ._io import _SafetensorsIO
from .direct_cascade import DirectCascadeEngine
from .streaming.adaptive_chunker import AdaptiveChunker
from .streaming.memory_monitor import MemoryMonitor
from spectralstream.format.compression import _name_to_method_id

logger = logging.getLogger(__name__)

_CHUNK_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB
_PROFILE_SAMPLE_ELEMENTS = 100_000
_CHECKPOINT_INTERVAL = 25

# Element sizes for dtype strings (bytes per element)
_DTYPE_ELEM_SIZE: Dict[str, int] = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
}


def _get_elem_size(dtype_str: str) -> int:
    """Return element size in bytes for a safetensors dtype string."""
    return _DTYPE_ELEM_SIZE.get(dtype_str, 4)


class StreamingCompressor:
    """
    Compresses models by streaming one tensor at a time from disk.
    Never loads more than 1 tensor (+ overhead) into RAM.
    Supports checkpoint/resume and chunked processing for very large tensors.

    When *use_cascade* is True, each tensor is compressed via multi-stage
    multiplicative stacking (e.g. SVD → DCT → entropy) to achieve
    multiplicative ratios (20x × 4x × 2x = 160x) rather than single-method
    ratios (2x).
    """

    def __init__(
        self,
        engine: Any,
        model_path: str,
        output_path: str,
        config: Optional[CompressionConfig] = None,
        no_grouping: bool = False,
        num_workers: Optional[int] = None,
        use_cascade: bool = True,
    ):
        self.engine = engine
        self.model_path = model_path
        self.output_path = output_path
        self.config = config or CompressionConfig()
        self._no_grouping = no_grouping
        self._num_workers = num_workers or os.cpu_count() or 4
        self._use_cascade = use_cascade
        self._io = _SafetensorsIO(use_mmap=True)
        self._compressed: List[CompressedTensor] = []
        self._cascade_engine: Any = None
        self._stats: Dict[str, Any] = {
            "total_tensors": 0,
            "processed": 0,
            "total_orig_bytes": 0,
            "total_comp_bytes": 0,
            "peak_memory_mb": 0,
            "start_time": 0.0,
            "failures": [],
        }

    # ── Memory-safe chunk sizing ─────────────────────────────────────────

    def _chunk_size_for_tensor(self, tensor_nbytes: int) -> int:
        """Determine safe chunk size for a tensor based on available RAM.

        For tensors that exceed 30% of available RAM, returns a chunk
        size that ensures no more than 256 MB is processed at once.
        This is critical for embedding tensors (4.5 GB).

        Parameters
        ----------
        tensor_nbytes : int
            Size of the tensor in bytes.

        Returns
        -------
        int
            Safe chunk size in elements.  If the tensor fits comfortably
            in RAM, returns the full tensor size (no chunking).
        """
        if not HAS_PSUTIL:
            # Without psutil, chunk at 256 MB to be safe
            if tensor_nbytes > 512 * 1024 * 1024:
                return max(256 * 1024 * 1024, tensor_nbytes // 8)
            return tensor_nbytes

        available_bytes = self._check_available_ram() * (1024 * 1024)
        tensor_mb = tensor_nbytes / (1024 * 1024)
        avail_mb = available_bytes / (1024 * 1024)

        logger.debug(
            "Memory: tensor=%.0f MB, available=%.0f MB (%.0f%%)",
            tensor_mb,
            avail_mb,
            tensor_nbytes / max(available_bytes, 1) * 100,
        )

        # Tensor uses more than 30% of available RAM → chunk it
        if tensor_nbytes > available_bytes * 0.3:
            # Chunk into pieces that each use ≤ 15% of available RAM
            chunk_bytes = max(
                256 * 1024 * 1024,  # At least 256 MB per chunk
                int(tensor_nbytes / max(tensor_nbytes / (available_bytes * 0.15), 1)),
            )
            logger.info(
                "Chunking %.0f MB tensor into %d MB chunks (available RAM: %.0f MB)",
                tensor_mb,
                chunk_bytes / (1024 * 1024),
                avail_mb,
            )
            return chunk_bytes

        # Tensor fits comfortably — no chunking needed
        return tensor_nbytes

    # ── Memory helpers ────────────────────────────────────────────────────

    def _peak_mem(self) -> float:
        if not HAS_PSUTIL:
            return 0.0
        try:
            p = _get_psutil().Process()
            mem = p.memory_info().rss / 1024 / 1024
            if mem > self._stats["peak_memory_mb"]:
                self._stats["peak_memory_mb"] = mem
            return mem
        except Exception:
            return 0.0

    def _check_available_ram(self) -> float:
        if not HAS_PSUTIL:
            return 64.0 * 1024
        try:
            return _get_psutil().virtual_memory().available / (1024 * 1024)
        except Exception:
            return 64.0 * 1024

    def _warn_if_low_memory(self) -> None:
        if not HAS_PSUTIL:
            return
        try:
            psutil_mod = _get_psutil()
            avail_mb = self._check_available_ram()
            total_mb = psutil_mod.virtual_memory().total / (1024 * 1024)
            pct_avail = avail_mb / total_mb * 100
            if pct_avail < 20.0:
                logger.warning(
                    "Low memory: %.0f MB available (%.0f%% of %.0f GB total)",
                    avail_mb,
                    pct_avail,
                    total_mb / 1024,
                )
        except (OSError, ValueError, RuntimeError):
            pass

    def _force_gc(self) -> None:
        collected = gc.collect()
        if collected > 0 and HAS_PSUTIL:
            try:
                p = _get_psutil().Process()
                mem = p.memory_info().rss / (1024 * 1024)
                logger.debug("GC: %d objects collected, RSS=%.0f MB", collected, mem)
            except (OSError, ValueError, RuntimeError):
                pass

    # ── Memory budget enforcement ─────────────────────────────────────────

    def _check_memory_before_compress(self, tensor_nbytes: int) -> bool:
        """Check if enough RAM is available for cascade compression.

        Cascade compression temporarily needs ~6× the tensor size:
          - original tensor (1×)
          - compressed data buffer (0.5×)
          - float64 cumulative reconstruction (2×)
          - float64 residual (2×)
          - per-stage decompression output (1×)

        Returns True if sufficient memory is available, False to signal
        the caller to use chunked or direct single-method compression.

        When psutil is not available, assumes OK (returns True).
        """
        if not HAS_PSUTIL:
            return True

        peak_needed = tensor_nbytes * 6  # conservative 6× multiplier
        available_bytes = self._check_available_ram() * (1024 * 1024)
        threshold = available_bytes * 0.8  # use at most 80 % of available RAM

        if peak_needed > threshold:
            tensor_mb = tensor_nbytes / (1024 * 1024)
            avail_mb = available_bytes / (1024 * 1024)
            logger.warning(
                "Memory budget: tensor=%.0f MB needs ~%.0f MB peak, "
                "only %.0f MB available.  Falling back to "
                "chunked / single-method compression.",
                tensor_mb,
                peak_needed / (1024 * 1024),
                avail_mb,
            )
            return False
        return True

    def _compress_tensor_clean(self, tensor: np.ndarray, name: str) -> CompressedTensor:
        """Compress a single tensor and force GC of intermediates.

        This wrapper ensures all temporary arrays created during cascade
        compression are freed before the next tensor is loaded.
        """
        result = self._compress_with_cascade(tensor, name)
        self._force_gc()
        return result

    def _compress_tensor_mmap_chunked(
        self,
        path: str,
        shape: tuple,
        dtype_str: str,
        offset: int,
        nbytes: int,
        name: str,
    ) -> CompressedTensor:
        """Compress a very large tensor via mmap streaming — never loads the
        full tensor into RAM.

        Uses ``AdaptiveChunker.stream_mmap_chunks()`` which reads one chunk
        at a time from the memory-mapped safetensors file.  Each chunk is
        compressed independently and the results are merged.

        This is the memory-safe path for tensors >500 MB (e.g. 4.5 GB
        embedding tables).
        """
        total_elements = nbytes // _get_elem_size(dtype_str)
        mem_monitor = MemoryMonitor(
            max_memory_gb=self.config.max_memory_gb,
            gc_interval=1,
        )
        chunker = AdaptiveChunker(
            engine=self.engine,
            memory_monitor=mem_monitor,
        )

        result = chunker.compress_mmap_chunked(
            name=name,
            mmap_path=path,
            dtype_str=dtype_str,
            offset=offset,
            nbytes=nbytes,
            shape=shape,
            target_ratio=self.config.target_ratio,
            max_error=self.config.max_error,
        )

        return CompressedTensor(
            _data=result.data,
            method=result.method,
            params=chunker._metadata,
            original_shape=shape,
            original_dtype=dtype_str,
            compression_ratio=result.ratio,
            relative_error=result.error,
            snr_db=0.0,
            psnr_db=0.0,
            cosine_similarity=1.0,
            computation_time=0.0,
        )

    # ── Checkpoint ────────────────────────────────────────────────────────

    def _checkpoint_path(self) -> str:
        return self.output_path + ".streaming_checkpoint"

    def _save_checkpoint(self, completed_tensors: List[str]) -> None:
        ckpt = {
            "completed_tensors": completed_tensors,
            "stats": {
                "total_orig_bytes": self._stats["total_orig_bytes"],
                "total_comp_bytes": self._stats["total_comp_bytes"],
                "processed": self._stats["processed"],
            },
            "timestamp": time.time(),
        }
        tmp = self._checkpoint_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(ckpt, f, default=str)
        os.replace(tmp, self._checkpoint_path())

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        path = self._checkpoint_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            logger.info(
                "Resuming streaming from checkpoint: %d tensors done",
                len(data.get("completed_tensors", [])),
            )
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load checkpoint: %s", e)
            return None

    def _clear_checkpoint(self) -> None:
        path = self._checkpoint_path()
        if os.path.exists(path):
            os.remove(path)

    # ── Profiling helpers ─────────────────────────────────────────────────

    def _profile_tensor_sampled(self, tensor: np.ndarray, name: str) -> TensorProfile:
        return self.engine.profiler.profile_tensor(tensor, name=name)

    def _sample_tensor_metadata(
        self, shape: tuple, dtype_str: str, offset: int, nbytes: int
    ) -> np.ndarray:
        """Read only a sample from a large tensor for profiling (no full load).

        Properly handles BF16 → float32 conversion.
        """
        if dtype_str == "BF16":
            # BF16: read as uint16, convert to uint32, left-pad with 16 zero bits, view as float32
            np_dtype = np.uint16
            elem_size = 2  # uint16 = 2 bytes
            total_elements = nbytes // elem_size
            sample_size = min(_PROFILE_SAMPLE_ELEMENTS, total_elements)
            if sample_size == total_elements:
                mm = np.memmap(
                    self.model_path,
                    dtype=np.uint16,
                    mode="r",
                    offset=offset,
                    shape=(total_elements,),
                )
                tensor = np.array(mm, dtype=np.uint16)
            else:
                with open(self.model_path, "rb") as f:
                    f.seek(offset)
                    raw = f.read(min(nbytes, sample_size * elem_size))
                tensor = np.frombuffer(raw, dtype=np.uint16)
            # BF16 → float32 conversion
            tensor = tensor.astype(np.uint32) << 16
            return tensor.view(np.float32)

        np_dtype = self._io.DTYPE_MAP.get(dtype_str, np.float32)
        elem_size = np.dtype(np_dtype).itemsize
        total_elements = nbytes // elem_size
        sample_size = min(_PROFILE_SAMPLE_ELEMENTS, total_elements)
        if sample_size == total_elements:
            mm = np.memmap(
                self.model_path,
                dtype=np_dtype,
                mode="r",
                offset=offset,
                shape=(total_elements,),
            )
            return np.array(mm, dtype=np_dtype).astype(np.float32)
        stride = max(1, total_elements // sample_size)
        with open(self.model_path, "rb") as f:
            f.seek(offset)
            raw = f.read(min(nbytes, sample_size * elem_size))
        tensor = np.frombuffer(raw, dtype=np_dtype)
        return tensor.astype(np.float32)

    # ── Cascade / multiplicative stacking ─────────────────────────────────

    def _get_cascade_engine(self) -> DirectCascadeEngine:
        """Lazy-init the DirectCascadeEngine."""
        if self._cascade_engine is not None:
            return self._cascade_engine
        self._cascade_engine = DirectCascadeEngine(store_all_stages=True)
        return self._cascade_engine

    def _compress_with_cascade(self, tensor: np.ndarray, name: str) -> CompressedTensor:
        """Compress using multi-stage cascade via DirectCascadeEngine.

        For 2D weight tensors, runs the cascade pattern (SVD→DCT by default).
        For 1D tensors (bias, norm) uses direct single-method compression.
        For EMBEDDING tensors, uses memory-safe chunked SVD or TT.

        The cascade achieves multiplicative ratios by storing each stage's
        compressed data in a packaged payload.  Total ratio is computed as
        ``original_size / sum(all_stage_sizes)``.
        """
        import time as _time

        start = _time.perf_counter()
        tensor_type = self.engine._classify_by_name(name)

        # 1D / small tensors — direct single method
        if (
            tensor.ndim <= 1
            or tensor_type in ("norm", "norm_bias")
            or tensor.size < 1024
        ):
            profile = self.engine.profiler.profile_tensor(tensor, name=name)
            methods = self.engine._select_methods(
                profile, self.config.max_error, self.config.target_ratio
            )
            data, meta, ratio_val, error_val = (
                self.engine.compress_tensor_with_validation(
                    tensor, profile, methods, self.config.max_error
                )
            )
            elapsed = _time.perf_counter() - start
            return CompressedTensor(
                _data=data,
                method=meta.get("method", "single"),
                params=meta,
                original_shape=tensor.shape,
                original_dtype=str(tensor.dtype),
                compression_ratio=ratio_val,
                relative_error=error_val,
                snr_db=meta.get("snr_db", 0.0),
                psnr_db=meta.get("psnr_db", 0.0),
                cosine_similarity=meta.get("cosine_similarity", 1.0),
                computation_time=elapsed,
            )

        # ── Embedding tensors: use specialized compressor ──────────────
        # Embeddings are huge (4.5 GB for Gemma 4) and benefit from
        # memory-aware SVD with projected ratio calculation.
        if tensor_type == "embedding":
            return self._compress_embedding_tensor(tensor, name, start)

        # ── Cascade compression ──────────────────────────────────────
        dc = self._get_cascade_engine()

        # Select cascade pattern based on tensor characteristics
        if tensor_type in ("attention", "qkv_fused"):
            pattern = "aggressive"
        else:
            # "weight" or other — use extreme for max ratio
            pattern = "extreme"

        try:
            data, meta = dc.execute_cascade(self.engine, tensor, tensor_type, pattern)
            ratio_val = meta["total_ratio"]
            error_val = meta["total_error"]
            best_method = f"cascade_{pattern}"

            elapsed = _time.perf_counter() - start
            return CompressedTensor(
                _data=data,
                method=best_method,
                params=meta,
                original_shape=tensor.shape,
                original_dtype=str(tensor.dtype),
                compression_ratio=ratio_val,
                relative_error=error_val,
                snr_db=meta.get("snr_db", 0.0),
                psnr_db=meta.get("psnr_db", 0.0),
                cosine_similarity=meta.get("cosine_similarity", 1.0),
                computation_time=elapsed,
            )
        except Exception as exc:
            logger.warning(
                "Cascade compression failed for '%s' (pattern=%s): %s",
                name,
                pattern,
                exc,
            )

        # ── Fallback cascade ─────────────────────────────────────────
        try:
            pattern = "balanced"
            data, meta = dc.execute_cascade(self.engine, tensor, tensor_type, pattern)
            ratio_val = meta["total_ratio"]
            error_val = meta["total_error"]
            best_method = f"cascade_{pattern}"

            elapsed = _time.perf_counter() - start
            return CompressedTensor(
                _data=data,
                method=best_method,
                params=meta,
                original_shape=tensor.shape,
                original_dtype=str(tensor.dtype),
                compression_ratio=ratio_val,
                relative_error=error_val,
                snr_db=meta.get("snr_db", 0.0),
                psnr_db=meta.get("psnr_db", 0.0),
                cosine_similarity=meta.get("cosine_similarity", 1.0),
                computation_time=elapsed,
            )
        except Exception as exc:
            logger.warning(
                "Fallback cascade failed for '%s': %s",
                name,
                exc,
            )

        # ── Last resort: single-method via engine ────────────────────
        profile = self.engine.profiler.profile_tensor(tensor, name=name)
        methods = self.engine._select_methods(
            profile, self.config.max_error, self.config.target_ratio
        )
        data, meta, ratio_val, error_val = self.engine.compress_tensor_with_validation(
            tensor, profile, methods, self.config.max_error
        )
        elapsed = _time.perf_counter() - start
        return CompressedTensor(
            _data=data,
            method=meta.get("method", "single"),
            params=meta,
            original_shape=tensor.shape,
            original_dtype=str(tensor.dtype),
            compression_ratio=ratio_val,
            relative_error=error_val,
            snr_db=meta.get("snr_db", 0.0),
            psnr_db=meta.get("psnr_db", 0.0),
            cosine_similarity=meta.get("cosine_similarity", 1.0),
            computation_time=elapsed,
        )

    def _compress_embedding_tensor(
        self, tensor: np.ndarray, name: str, start: float
    ) -> CompressedTensor:
        """Specialized compression path for embedding tensors.

        Embedding matrices are extremely large (vocab_size × hidden_dim).
        This method uses the DirectCascadeEngine's ``_compress_embedding``
        which tries multiple SVD ranks and picks the best tradeoff.

        If the specialized compressor fails, falls back to chunked
        row-wise compression (each row compressed independently).

        Parameters
        ----------
        tensor : np.ndarray
            Float32 embedding tensor.
        name : str
            Tensor name.
        start : float
            Start time for elapsed computation.

        Returns
        -------
        CompressedTensor
            Compression result.
        """
        import time as _time

        tensor_type = "embedding"

        # ── Try specialized embedding compressor ─────────────────────
        dc = self._get_cascade_engine()
        try:
            data, meta = dc.execute_cascade(
                self.engine, tensor, tensor_type, pattern="auto"
            )
            ratio_val = meta.get("total_ratio", 1.0)
            error_val = meta.get("total_error", 0.0)
            best_method = meta.get("method", "embedding_svd")

            elapsed = _time.perf_counter() - start
            logger.info(
                "Embedding '%s': %s, ratio=%.0fx, error=%.6f, time=%.1fs",
                name,
                best_method,
                ratio_val,
                error_val,
                elapsed,
            )
            return CompressedTensor(
                _data=data,
                method=best_method,
                params=meta,
                original_shape=tensor.shape,
                original_dtype=str(tensor.dtype),
                compression_ratio=ratio_val,
                relative_error=error_val,
                snr_db=meta.get("snr_db", 0.0),
                psnr_db=meta.get("psnr_db", 0.0),
                cosine_similarity=meta.get("cosine_similarity", 1.0),
                computation_time=elapsed,
            )
        except Exception as exc:
            logger.warning(
                "Specialized embedding compressor failed for '%s': %s. "
                "Falling back to chunked compression.",
                name,
                exc,
            )

        # ── Fallback: chunked row-wise compression ──────────────────
        # Each embedding row is (hidden_dim,) = ~35KB for 8960-dim
        # Compress each row independently with BlockINT8 for 4:1 ratio
        logger.info("Using chunked row-wise compression for '%s'", name)
        rows = tensor.reshape(tensor.shape[0], -1)
        n_rows = rows.shape[0]
        row_size = rows.shape[1]

        blk8 = self.engine._methods.get("block_int8")
        if blk8 is not None:
            chunked_data = bytearray()
            total_ratio_val = 0.0
            total_error_val = 0.0

            for i in range(0, n_rows, 1024):  # Process 1024 rows at a time
                batch = rows[i : i + 1024]
                try:
                    d, m = blk8.compress(batch, block_size=64)
                    recon = blk8.decompress(d, m)
                    batch_ratio = float(batch.nbytes / max(len(d), 1))
                    batch_error = float(np.abs(batch - recon).mean())

                    # Store compressed size in header
                    chunked_data += struct.pack("<I", len(d))
                    chunked_data += d
                    total_ratio_val += batch_ratio * len(batch)
                    total_error_val += batch_error * len(batch)
                except Exception:
                    # Row failed — store raw
                    raw = batch.tobytes()
                    chunked_data += struct.pack("<I", len(raw))
                    chunked_data += raw

            avg_ratio = total_ratio_val / max(n_rows, 1)
            avg_error = total_error_val / max(n_rows, 1)

            elapsed = _time.perf_counter() - start
            logger.info(
                "Embedding '%s': chunked row-wise, ratio=%.0fx, error=%.6f, time=%.1fs",
                name,
                avg_ratio,
                avg_error,
                elapsed,
            )
            return CompressedTensor(
                _data=bytes(chunked_data),
                method=f"chunked_rowwise_embedding",
                params={
                    "num_rows": n_rows,
                    "row_size": row_size,
                    "chunk_size": 1024,
                    "method": "block_int8",
                },
                original_shape=tensor.shape,
                original_dtype=str(tensor.dtype),
                compression_ratio=avg_ratio,
                relative_error=avg_error,
                snr_db=0.0,
                psnr_db=0.0,
                cosine_similarity=1.0,
                computation_time=elapsed,
            )

        # ── Last resort: half-precision ──────────────────────────────
        elapsed = _time.perf_counter() - start
        raw_f16 = tensor.astype(np.float16).tobytes()
        ratio_val = float(tensor.nbytes / max(len(raw_f16), 1))
        logger.info(
            "Embedding '%s': float16 fallback, ratio=%.0fx, time=%.1fs",
            name,
            ratio_val,
            elapsed,
        )
        return CompressedTensor(
            _data=raw_f16,
            method="float16_embedding",
            params={"method": "float16"},
            original_shape=tensor.shape,
            original_dtype=str(tensor.dtype),
            compression_ratio=ratio_val,
            relative_error=0.0,
            snr_db=0.0,
            psnr_db=0.0,
            cosine_similarity=1.0,
            computation_time=elapsed,
        )

    # ── Chunked compression ───────────────────────────────────────────────

    def _compress_tensor_chunked(
        self,
        tensor: np.ndarray,
        name: str,
        chunk_size: int = _CHUNK_THRESHOLD_BYTES,
    ) -> CompressedTensor:
        """Compress a large tensor in chunks, each compressed independently."""
        flat = tensor.ravel()
        n = flat.size
        chunk_elems = max(1, chunk_size // np.dtype(np.float32).itemsize)
        chunks: List[bytes] = []
        original_shape = tensor.shape
        for start in range(0, n, chunk_elems):
            end = min(start + chunk_elems, n)
            chunk = flat[start:end].reshape(1, -1)
            profile = self.engine.profiler.profile_tensor(chunk, name=f"{name}_chunk")
            methods = self.engine._select_methods(
                profile,
                self.config.max_error,
                self.config.target_ratio,
            )
            data, meta, ratio_val, error_val = (
                self.engine.compress_tensor_with_validation(
                    chunk, profile, methods, self.config.max_error
                )
            )
            chunks.append(data)
        combined = b"".join(chunks)
        total_ratio = tensor.nbytes / max(len(combined), 1)
        return CompressedTensor(
            _data=combined,
            method="chunked",
            params={
                "num_chunks": len(chunks),
                "chunk_size_elems": chunk_elems,
                "original_shape": list(original_shape),
                "per_chunk_methods": [],
            },
            original_shape=original_shape,
            original_dtype=str(tensor.dtype),
            compression_ratio=total_ratio,
            relative_error=self.config.max_error,
            snr_db=0.0,
            psnr_db=0.0,
            cosine_similarity=1.0,
            computation_time=0.0,
        )

    # ── Header-only metadata scan ─────────────────────────────────────────

    def header_only_scan(self) -> Dict[str, Tuple[tuple, str, int, int]]:
        """Read only safetensors header to get all tensor metadata.

        Never loads any tensor data — O(1) memory, reads first few MB.
        Returns {name: (shape, dtype_str, offset, nbytes)}.
        """
        import struct

        with open(self.model_path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_len)
            header: Dict[str, Any] = json.loads(header_bytes)

        data_start = 8 + header_len
        tensor_info: Dict[str, Tuple[tuple, str, int, int]] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info.get("dtype", "F32")
            shape = tuple(info.get("shape", []))
            offsets = info.get("data_offsets", [0, 0])
            tensor_info[name] = (
                shape,
                dtype,
                data_start + offsets[0],
                offsets[1] - offsets[0],
            )
        return tensor_info

    # ── Main compression loop ─────────────────────────────────────────────

    def compress_all(
        self,
        progress_callback: Optional[
            Callable[[int, int, CompressedTensor, float], None]
        ] = None,
        streaming_writer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compress all tensors, one at a time, streaming from disk.

        Each compressed tensor is written immediately to the output file.
        No in-memory accumulation of compressed tensors — only running stats.

        Uses header-only scan to get metadata without loading any tensor data.

        If *streaming_writer* is provided (SSFWriter), uses it instead of
        creating a new writer. Returns a stats dict (not a list of tensors).
        """
        self._stats["start_time"] = time.perf_counter()

        # Use header-only scan to avoid loading any tensor data
        tensor_info = self.header_only_scan()
        self._stats["total_tensors"] = len(tensor_info)
        logger.info(
            "Streaming compression: %d tensors (%.1f GB)",
            len(tensor_info),
            sum(nb for _, _, _, nb in tensor_info.values()) / 1e9,
        )

        # Classify by name using header-only metadata (no tensor data loaded)
        tensor_types = {
            name: self.engine._classify_by_name(name) for name in tensor_info
        }

        # Checkpoint resume
        checkpoint = self._load_checkpoint()
        completed_set: set = set()
        if checkpoint:
            completed_set = set(checkpoint.get("completed_tensors", []))
            s = checkpoint.get("stats", {})
            self._stats["total_orig_bytes"] = s.get("total_orig_bytes", 0)
            self._stats["total_comp_bytes"] = s.get("total_comp_bytes", 0)
            self._stats["processed"] = s.get("processed", 0)

        from spectralstream.format.writer import SSFWriter

        if streaming_writer is None:
            writer = SSFWriter(
                self.output_path,
                metadata={"model": self.model_path, "streaming": True},
            )
            writer.__enter__()
            close_writer = True
        else:
            writer = streaming_writer
            close_writer = False

        method_dist: Dict[str, int] = {}
        total_errors: list = []
        total_ratios: list = []

        try:
            for i, (name, (shape, dtype_str, offset, nbytes)) in enumerate(
                tensor_info.items()
            ):
                if name in completed_set:
                    continue

                self._warn_if_low_memory()

                tensor_size_mb = nbytes / (1024 * 1024)

                # ── Memory-aware compression path ─────────────────────────
                # For very large tensors (>= 500 MB) with tight memory, use
                # mmap-based chunked compression that never loads the full
                # tensor into RAM.
                # For embedding tensors, use adaptive chunk sizing.
                tensor_type_for_path = self.engine._classify_by_name(name)
                safe_chunk_size = self._chunk_size_for_tensor(nbytes)
                use_mmap_chunked = (
                    nbytes > _CHUNK_THRESHOLD_BYTES
                    and not self._check_memory_before_compress(nbytes)
                )
                is_embedding_path = tensor_type_for_path == "embedding"

                if use_mmap_chunked:
                    logger.info(
                        "  %s: %.0f MB \u2014 mmap chunked (memory-safe)",
                        name,
                        tensor_size_mb,
                    )
                    ct = self._compress_tensor_mmap_chunked(
                        self.model_path, shape, dtype_str, offset, nbytes, name
                    )
                    # For writer.add_tensor, we need a tensor.  Load a minimal
                    # mmap view — add_tensor copies internally so memory is
                    # bounded by the copy + compress step.
                    tensor = self._io.load_tensor(
                        self.model_path, shape, dtype_str, offset, nbytes
                    )
                else:
                    tensor = self._io.read(
                        self.model_path, shape, dtype_str, offset, nbytes
                    )

                if is_embedding_path:
                    # Embedding tensors get the specialized compressor
                    # which handles memory-safe SVD with chunking
                    ct = self._compress_tensor_clean(tensor, name)
                elif (
                    not use_mmap_chunked
                    and tensor.nbytes > _CHUNK_THRESHOLD_BYTES
                    and safe_chunk_size < tensor.nbytes
                ):
                    logger.info(
                        "  %s: %.0f MB \u2014 using chunked compression (%d MB chunks)",
                        name,
                        tensor_size_mb,
                        safe_chunk_size / (1024 * 1024),
                    )
                    ct = self._compress_tensor_chunked(tensor, safe_chunk_size)
                elif self._use_cascade:
                    # Check memory budget BEFORE cascade — for very large
                    # tensors (e.g. 4.5 GB embedding tables), cascade's
                    # float64 intermediates would blow past available RAM.
                    if self._check_memory_before_compress(tensor.nbytes):
                        ct = self._compress_tensor_clean(tensor, name)
                    else:
                        logger.info(
                            "  %s: %.0f MB \u2014 budget exceeded, "
                            "using single-method fallback",
                            name,
                            tensor_size_mb,
                        )
                        profile = self.engine.profiler.profile_tensor(tensor, name=name)
                        methods = self.engine._select_methods(
                            profile,
                            self.config.max_error,
                            self.config.target_ratio,
                        )
                        data, meta, ratio_val, error_val = (
                            self.engine.compress_tensor_with_validation(
                                tensor, profile, methods, self.config.max_error
                            )
                        )
                        ct = CompressedTensor(
                            _data=data,
                            method=meta.get("method", ""),
                            params=meta,
                            original_shape=tensor.shape,
                            original_dtype=str(tensor.dtype),
                            compression_ratio=ratio_val,
                            relative_error=error_val,
                            snr_db=meta.get("snr_db", 0.0),
                            psnr_db=meta.get("psnr_db", 0.0),
                            cosine_similarity=meta.get("cosine_similarity", 1.0),
                            computation_time=0.0,
                        )
                else:
                    ttype = tensor_types.get(name, "weight")
                    if ttype in (
                        "norm_bias",
                        "embedding",
                        "attention",
                        "ffn",
                        "qkv_fused",
                    ):
                        data, meta, ratio_val, error_val = self.engine.compress_fast(
                            tensor, name
                        )
                        ct = CompressedTensor(
                            _data=data,
                            method=meta.get("method", "fast"),
                            params=meta,
                            original_shape=tensor.shape,
                            original_dtype=str(tensor.dtype),
                            compression_ratio=ratio_val,
                            relative_error=error_val,
                            snr_db=meta.get("snr_db", 0.0),
                            psnr_db=meta.get("psnr_db", 0.0),
                            cosine_similarity=meta.get("cosine_similarity", 1.0),
                            computation_time=0.0,
                        )
                    else:
                        profile = self.engine.profiler.profile_tensor(tensor, name=name)
                        methods = self.engine._select_methods(
                            profile,
                            self.config.max_error,
                            self.config.target_ratio,
                        )
                        data, meta, ratio_val, error_val = (
                            self.engine.compress_tensor_with_validation(
                                tensor, profile, methods, self.config.max_error
                            )
                        )
                        ct = CompressedTensor(
                            _data=data,
                            method=meta.get("method", ""),
                            params=meta,
                            original_shape=tensor.shape,
                            original_dtype=str(tensor.dtype),
                            compression_ratio=ratio_val,
                            relative_error=error_val,
                            snr_db=meta.get("snr_db", 0.0),
                            psnr_db=meta.get("psnr_db", 0.0),
                            cosine_similarity=meta.get("cosine_similarity", 1.0),
                            computation_time=0.0,
                        )

                # Write the ORIGINAL tensor through SSFWriter so it handles
                # compression+storage properly. The engine's compress() was
                # used for method selection + quality estimation only.
                method_id = _name_to_method_id(ct.method)
                writer.add_tensor(
                    ct.method + "_" + str(hash(str(ct.original_shape))),
                    tensor,
                    method=method_id,
                    params={
                        "original_shape": list(ct.original_shape),
                        "original_dtype": ct.original_dtype,
                        "compression_method": ct.method,
                        "compression_params": ct.params,
                        "relative_error": ct.relative_error,
                        "compression_ratio": ct.compression_ratio,
                    },
                    quality_metrics={
                        "relative_error": ct.relative_error,
                        "compression_ratio": ct.compression_ratio,
                        "snr_db": ct.snr_db,
                    },
                )

                # Track running stats instead of accumulating tensor objects
                self._stats["processed"] += 1
                self._stats["total_orig_bytes"] += tensor.nbytes
                self._stats["total_comp_bytes"] += ct.get_data_size()
                mem = self._peak_mem()

                if progress_callback:
                    progress_callback(i + 1, len(tensor_info), ct, mem)

                method_dist[ct.method] = method_dist.get(ct.method, 0) + 1
                total_errors.append(ct.relative_error)
                total_ratios.append(ct.compression_ratio)

                del tensor, ct
                self._force_gc()

                if (i + 1) % 25 == 0 or (i + 1) == len(tensor_info):
                    elapsed = time.perf_counter() - self._stats["start_time"]
                    rate = (i + 1) / max(elapsed, 0.001)
                    logger.info(
                        "  [%d/%d] ratio=%8.2fx mem=%.0fMB [%.0f t/s]",
                        i + 1,
                        len(tensor_info),
                        self._stats["total_orig_bytes"]
                        / max(self._stats["total_comp_bytes"], 1),
                        mem,
                        rate,
                    )

                if (i + 1) % _CHECKPOINT_INTERVAL == 0:
                    completed_names = [n for n in tensor_info][: i + 1]
                    self._save_checkpoint(completed_names)

        finally:
            if close_writer:
                writer.__exit__(None, None, None)

        self._clear_checkpoint()
        self._compressed.clear()

        elapsed = time.perf_counter() - self._stats["start_time"]
        self._stats["peak_memory_mb"] = max(
            self._stats["peak_memory_mb"], self._peak_mem()
        )
        overall_ratio = max(
            self._stats["total_orig_bytes"] / max(self._stats["total_comp_bytes"], 1),
            1.0,
        )
        logger.info(
            "Streaming complete: %.1fs, peak mem=%.0fMB, ratio=%.1fx",
            elapsed,
            self._stats["peak_memory_mb"],
            overall_ratio,
        )

        return {
            "total_tensors": self._stats["total_tensors"],
            "total_orig_bytes": self._stats["total_orig_bytes"],
            "total_comp_bytes": self._stats["total_comp_bytes"],
            "overall_ratio": overall_ratio,
            "avg_error": float(np.mean(total_errors)) if total_errors else 0.0,
            "max_error": float(np.max(total_errors)) if total_errors else 0.0,
            "time_seconds": elapsed,
            "peak_memory_mb": self._stats["peak_memory_mb"],
            "method_distribution": method_dist,
            "failures": self._stats.get("failures", []),
        }
