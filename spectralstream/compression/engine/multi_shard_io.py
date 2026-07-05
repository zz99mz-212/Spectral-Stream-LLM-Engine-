"""
Multi-Shard Streaming IO for MiMo-V2.5 and similar large models.
Handles:
- Multiple safetensors shard files
- Model index JSON for shard mapping
- Streaming (never load full shard into RAM)
- Expert-parameter routing for MoE
- mmap-based zero-copy tensor loading
"""

import gc
from spectralstream.compression._imports import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Path,
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

from ._dataclasses import CompressedTensor, TensorProfile

logger = logging.getLogger(__name__)


class MultiShardSafetensorsIO:
    """Streaming IO for multi-shard safetensors models (like MiMo-V2.5)."""

    DTYPE_MAP = {
        "F32": np.float32,
        "F16": np.float16,
        "BF16": np.float16,
        "I64": np.int64,
        "I32": np.int32,
        "I16": np.int16,
        "I8": np.int8,
        "U8": np.uint8,
    }

    def __init__(self, model_path: str):
        """
        Initialize with either:
        - Path to a directory containing model.safetensors.index.json
        - Path to a single safetensors file (for single-shard models)
        """
        self.model_path = Path(model_path)
        self.shard_paths: List[Path] = []
        self.index: Dict[str, Tuple[Path, Tuple[int, ...], str, int, int]] = {}
        self.shard_headers: Dict[Path, Dict] = {}

        self._discover_shards()
        self._build_index()

    def _discover_shards(self):
        """Find all safetensors shard files."""
        if self.model_path.is_dir():
            index_file = self.model_path / "model.safetensors.index.json"
            if index_file.exists():
                with open(index_file) as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})

                shard_set = set()
                for tensor_name, shard_rel_path in weight_map.items():
                    shard_path = self.model_path / shard_rel_path
                    if shard_path.exists():
                        shard_set.add(shard_path)
                    else:
                        logger.warning(f"Shard not found: {shard_path}")

                self.shard_paths = sorted(shard_set)
                logger.info(f"Found {len(self.shard_paths)} shards from index")
            else:
                self.shard_paths = sorted(self.model_path.glob("*.safetensors"))
                logger.info(
                    f"Found {len(self.shard_paths)} safetensors files in directory"
                )
        elif self.model_path.suffix == ".safetensors":
            self.shard_paths = [self.model_path]
        else:
            raise ValueError(f"Unsupported model path: {model_path}")

    def _build_index(self):
        """Build unified index of all tensors across all shards."""
        for shard_path in self.shard_paths:
            try:
                with open(shard_path, "rb") as f:
                    header_len = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(header_len))

                data_start = 8 + header_len
                self.shard_headers[shard_path] = header

                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    dtype = info.get("dtype", "F32")
                    shape = tuple(info.get("shape", []))
                    offsets = info.get("data_offsets", [0, 0])
                    self.index[name] = (
                        shard_path,
                        shape,
                        dtype,
                        data_start + offsets[0],
                        offsets[1] - offsets[0],
                    )

                logger.debug(f"  {shard_path.name}: {len(header)} tensors")
            except Exception as e:
                logger.warning(f"Failed to read shard {shard_path}: {e}")

        logger.info(f"Total tensors indexed: {len(self.index)}")

    def list_tensors(self) -> List[str]:
        return list(self.index.keys())

    def get_tensor_info(self, name: str) -> Optional[Dict]:
        if name not in self.index:
            return None
        shard, shape, dtype, offset, nbytes = self.index[name]
        return {
            "name": name,
            "shard": str(shard),
            "shape": shape,
            "dtype": dtype,
            "offset": offset,
            "nbytes": nbytes,
        }

    def load_tensor(
        self, name: str, max_bytes: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Load a single tensor, optionally truncated to max_bytes for sampling."""
        if name not in self.index:
            logger.warning(f"Tensor not found: {name}")
            return None

        shard_path, shape, dtype_str, offset, nbytes = self.index[name]

        if max_bytes and nbytes > max_bytes:
            nbytes = max_bytes

        try:
            np_dtype = self.DTYPE_MAP.get(dtype_str, np.float32)

            with open(shard_path, "rb") as f:
                f.seek(offset)
                raw = f.read(nbytes)

            tensor = np.frombuffer(raw, dtype=np_dtype)
            if shape:
                total_elements = np.prod(shape)
                read_elements = min(len(tensor), total_elements)
                if read_elements < total_elements:
                    new_shape = list(shape)
                    new_shape[0] = (
                        read_elements // np.prod(shape[1:])
                        if np.prod(shape[1:]) > 0
                        else read_elements
                    )
                    if new_shape[0] > 0:
                        tensor = tensor[: int(np.prod(new_shape))].reshape(new_shape)
                else:
                    tensor = tensor.reshape(shape)

            return tensor.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to load tensor {name}: {e}")
            return None

    def load_tensor_streaming(
        self, name: str, max_bytes: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Zero-copy tensor loading via mmap (read-only).

        Unlike load_tensor(), this returns the raw mmap view without
        copying data into RAM. For BF16, an unavoidable conversion copy is made.
        Caller must del the returned tensor and gc.collect() after use.
        """
        if name not in self.index:
            logger.warning(f"Tensor not found: {name}")
            return None

        shard_path, shape, dtype_str, offset, nbytes = self.index[name]
        if max_bytes and nbytes > max_bytes:
            nbytes = max_bytes

        try:
            np_dtype = self.DTYPE_MAP.get(dtype_str, np.float32)
            elem_size = np.dtype(np_dtype).itemsize
            total_elems = nbytes // elem_size

            mm = np.memmap(
                str(shard_path),
                dtype=np_dtype,
                mode="r",
                offset=offset,
                shape=(total_elems,),
            )
            if dtype_str == "BF16":
                tensor = mm.astype(np.uint32) << 16
                tensor = tensor.view(np.float32)
                del mm
            else:
                tensor = mm  # zero-copy view

            if shape:
                total_elements = np.prod(shape)
                read_elements = len(tensor)
                if read_elements < total_elements:
                    new_shape = list(shape)
                    elements_per_row = max(1, int(np.prod(shape[1:])))
                    new_shape[0] = read_elements // elements_per_row
                    if new_shape[0] > 0:
                        tensor = tensor[: int(np.prod(new_shape))].reshape(new_shape)
                else:
                    tensor = tensor.reshape(shape)

            return tensor
        except Exception as e:
            logger.error(f"Failed to streaming-load tensor {name}: {e}")
            return None

    def _stream_tensors(
        self,
    ) -> Iterator[Tuple[str, np.ndarray, Dict]]:
        """Stream tensors one at a time using mmap (zero-copy).

        Yields: (name, tensor_data, info_dict)
        Caller MUST delete tensor and gc.collect() after each iteration.
        Never holds more than one tensor in RAM at a time.
        """
        for name in self.index:
            # Check memory before each tensor
            if HAS_PSUTIL:
                try:
                    avail = _psutil_mod.virtual_memory().available
                    avail_gb = avail / (1024**3)
                    if avail_gb < 2.0:
                        gc.collect()
                        for obj in gc.get_objects():
                            if isinstance(obj, np.ndarray) and obj.size > 1_000_000:
                                del obj
                        gc.collect()
                        if avail_gb < 1.0:
                            import time as _time

                            _time.sleep(1.0)
                            gc.collect()
                except (OSError, ValueError, RuntimeError):
                    pass

            info = self.get_tensor_info(name)
            tensor = self.load_tensor_streaming(name)
            if tensor is not None:
                yield name, tensor, info

    def iterate_tensors(
        self, max_per_shard: Optional[int] = None
    ) -> Iterator[Tuple[str, np.ndarray, Dict]]:
        """Iterate over all tensors, streaming from disk.
        Yields: (name, tensor_data, info_dict)

        NOTE: Uses load_tensor() (full read). For zero-copy mmap streaming,
        use _stream_tensors() instead.
        """
        count = 0
        for name in self.index:
            info = self.get_tensor_info(name)
            tensor = self.load_tensor(name)
            if tensor is not None:
                yield name, tensor, info
                count += 1

    def get_total_size(self) -> int:
        """Get total model size in bytes."""
        return sum(info[4] for info in self.index.values())

    def get_tensors_by_type(self) -> Dict[str, List[str]]:
        """Categorize tensors by their role in the model."""
        categories = {
            "embed": [],
            "attention": [],
            "ffn": [],
            "norm": [],
            "expert": [],
            "moe_gate": [],
            "output": [],
            "other": [],
        }

        for name in self.index:
            nl = name.lower()
            if any(k in nl for k in ["embed", "wte", "tok_emb"]):
                categories["embed"].append(name)
            elif any(k in nl for k in ["attn", "q_proj", "k_proj", "v_proj", "o_proj"]):
                categories["attention"].append(name)
            elif "expert" in nl or "moe" in nl:
                if "gate" in nl or "router" in nl:
                    categories["moe_gate"].append(name)
                else:
                    categories["expert"].append(name)
            elif any(
                k in nl for k in ["ffn", "gate_proj", "up_proj", "down_proj", "mlp"]
            ):
                categories["ffn"].append(name)
            elif any(k in nl for k in ["norm", "ln_", "rms"]):
                categories["norm"].append(name)
            elif any(k in nl for k in ["head", "lm_head", "output"]):
                categories["output"].append(name)
            else:
                categories["other"].append(name)

        return categories


class StreamingCompressionOrchestrator:
    """Orchestrates compression of multi-shard models with streaming.
    Never loads more than 1 tensor into memory at a time."""

    def __init__(self, engine, model_path: str, output_path: str):
        self.engine = engine
        self.model_path = model_path
        self.output_path = output_path
        self.io = MultiShardSafetensorsIO(model_path)
        self.ssf_writer = None

    def compress_streaming(self, config: Any) -> Any:
        """Compress model by streaming one tensor at a time.

        MEMORY: Never holds more than one tensor in RAM.
        Uses mmap-based zero-copy loading. Aggressive GC after each tensor.
        """
        from spectralstream.format.writer import SSFWriter

        total_tensors = len(self.io.index)
        logger.info(
            f"Streaming compression: {total_tensors} tensors, {self.io.get_total_size() / 1e9:.1f} GB"
        )

        t_start = time.perf_counter()
        failures: List[str] = []
        compressed_tensors: List[Any] = []

        with SSFWriter(self.output_path) as writer:
            processed = 0
            total_orig = 0
            total_comp = 0

            # Use _stream_tensors() for zero-copy mmap loading
            for name, tensor, info in self.io._stream_tensors():
                processed += 1

                if processed % 10 == 0:
                    logger.info(f"[{processed}/{total_tensors}] Processing...")

                try:
                    profile = self.engine.profiler.profile_tensor(tensor, name=name)
                    methods = self.engine._select_methods(
                        profile,
                        config.max_error,
                        config.target_ratio,
                    )

                    data, meta, ratio_val, error_val = (
                        self.engine.compress_tensor_with_validation(
                            tensor, profile, methods, config.max_error
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

                    compressed_arr = np.frombuffer(ct.data, dtype=np.uint8)
                    writer.add_tensor(
                        name,
                        compressed_arr,
                        method=350,
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
                    compressed_tensors.append(ct)

                    total_orig += tensor.nbytes
                    total_comp += ct.get_data_size()
                except Exception as e:
                    failures.append(name)
                    logger.error(f"Failed to compress '{name}': {e}")
                finally:
                    # CRITICAL: Free tensor immediately after each iteration
                    del tensor
                    gc.collect()

                if processed % 50 == 0:
                    current_ratio = total_orig / max(total_comp, 1)
                    peak_mem = 0.0
                    if HAS_PSUTIL:
                        try:
                            peak_mem = _psutil_mod.Process().memory_info().rss / (
                                1024 * 1024
                            )
                        except (OSError, ValueError, RuntimeError):
                            pass
                    logger.info(
                        f"  Progress: {processed}/{total_tensors}, ratio so far: {current_ratio:.2f}x "
                        f"mem={peak_mem:.0f}MB"
                    )

        # Build report from collected stats
        compress_end = time.perf_counter()
        method_dist: Dict[str, int] = {}
        errors: List[float] = []
        for ct in compressed_tensors:
            method_dist[ct.method] = method_dist.get(ct.method, 0) + 1
            errors.append(ct.relative_error)
        avg_error = float(np.mean(errors)) if errors else 0.0
        max_error = float(np.max(errors)) if errors else 0.0
        min_error = float(np.min(errors)) if errors else 0.0
        stats = {
            "tensors": compressed_tensors,
            "total_orig_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": total_orig / max(total_comp, 1),
            "average_ratio": total_orig / max(total_comp, 1),
            "avg_error": avg_error,
            "max_error": max_error,
            "min_error": min_error,
            "num_tensors": len(compressed_tensors),
            "method_distribution": method_dist,
            "failures": failures,
            "time_seconds": compress_end - t_start,
            "per_layer_error": {
                ct.method: ct.relative_error for ct in compressed_tensors
            },
        }
        result = self.engine._build_report(stats)
        compressed_tensors.clear()
        gc.collect()
        return result
