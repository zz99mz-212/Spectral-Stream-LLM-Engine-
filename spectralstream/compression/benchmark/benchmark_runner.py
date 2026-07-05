from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine import (
    CompressionConfig,
    CompressionIntelligenceEngine,
)
from spectralstream.compression.engine._io import _SafetensorsIO
from spectralstream.compression.benchmark.loss_calculator import (
    LossCalculator,
    TensorLossMetrics,
)

logger = logging.getLogger(__name__)

TARGET_RATIOS = [50, 100, 500, 1000, 2000, 5000]

SYNTHETIC_SHAPES: Dict[str, List[Tuple[int, ...]]] = {
    "embedding": [(262144, 1536), (128000, 2048)],
    "attention_q": [(1536, 256), (4096, 128)],
    "attention_k": [(1536, 128), (4096, 64)],
    "attention_v": [(1536, 128), (4096, 64)],
    "attention_o": [(256, 1536), (128, 4096)],
    "ffn_gate": [(1536, 6144), (4096, 16384)],
    "ffn_up": [(1536, 6144), (4096, 16384)],
    "ffn_down": [(6144, 1536), (16384, 4096)],
    "output": [(262144, 1536), (128000, 2048)],
    "norm": [(1536,), (4096,)],
}


@dataclass
class BenchmarkTensorResult:
    tensor_name: str
    tensor_type: str
    shape: Tuple[int, ...]
    method: str
    target_ratio: float
    achieved_ratio: float
    metrics: TensorLossMetrics
    compression_time: float
    decompression_time: float
    streaming_peak_memory_mb: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    cascade_pattern: str = ""


@dataclass
class BenchmarkRunResult:
    model_name: str
    target_ratios: List[float]
    tensor_results: Dict[str, List[BenchmarkTensorResult]]
    per_type_results: Dict[str, Dict[float, List[BenchmarkTensorResult]]]
    per_method_results: Dict[str, Dict[float, List[BenchmarkTensorResult]]]
    cascade_results: Dict[str, List[BenchmarkTensorResult]]
    streaming_results: Dict[str, BenchmarkTensorResult]
    run_time: float
    num_tensors: int
    num_failures: int


class BenchmarkRunner:
    def __init__(
        self,
        engine: Optional[CompressionIntelligenceEngine] = None,
        config: Optional[CompressionConfig] = None,
    ):
        self._config = config or CompressionConfig()
        self._engine = engine or CompressionIntelligenceEngine(config=self._config)
        self._loss = LossCalculator()
        self._io = _SafetensorsIOAccessor()

    @property
    def engine(self) -> CompressionIntelligenceEngine:
        return self._engine

    @property
    def loss(self) -> LossCalculator:
        return self._loss

    def benchmark_synthetic(
        self,
        target_ratios: Optional[List[float]] = None,
        tensor_types: Optional[List[str]] = None,
        seed: int = 42,
    ) -> BenchmarkRunResult:
        ratios = target_ratios if target_ratios is not None else TARGET_RATIOS
        rng = np.random.RandomState(seed)
        types_to_test = tensor_types if tensor_types else list(SYNTHETIC_SHAPES.keys())

        all_results: Dict[str, List[BenchmarkTensorResult]] = {}
        per_type: Dict[str, Dict[float, List[BenchmarkTensorResult]]] = {}
        per_method: Dict[str, Dict[float, List[BenchmarkTensorResult]]] = {}

        t0 = time.perf_counter()
        n_total = 0
        n_fail = 0

        for ttype in types_to_test:
            shapes = SYNTHETIC_SHAPES.get(ttype, [(4096, 4096)])
            per_type.setdefault(ttype, {})
            for shape in shapes:
                tensor = rng.randn(*shape).astype(np.float32)
                if ttype == "norm":
                    tensor = np.abs(tensor) * 0.1
                for tr in ratios:
                    try:
                        result = self._test_single_tensor(
                            tensor, ttype, shape, tr, name=f"synthetic_{ttype}"
                        )
                        all_results.setdefault(result.tensor_name, []).append(result)
                        per_type[ttype].setdefault(tr, []).append(result)
                        per_method.setdefault(result.method, {}).setdefault(
                            tr, []
                        ).append(result)
                        n_total += 1
                    except Exception as e:
                        n_fail += 1
                        logger.warning("  synthetic %s @ %dx failed: %s", ttype, tr, e)

        elapsed = time.perf_counter() - t0
        return BenchmarkRunResult(
            model_name="synthetic",
            target_ratios=ratios,
            tensor_results=all_results,
            per_type_results=per_type,
            per_method_results=per_method,
            cascade_results={},
            streaming_results={},
            run_time=elapsed,
            num_tensors=n_total,
            num_failures=n_fail,
        )

    def benchmark_real_model(
        self,
        model_path: str,
        target_ratios: Optional[List[float]] = None,
        max_tensors: Optional[int] = None,
    ) -> BenchmarkRunResult:
        ratios = target_ratios if target_ratios is not None else TARGET_RATIOS

        tensor_info = self._io.scan(model_path)
        items = list(tensor_info.items())
        if max_tensors:
            items = items[:max_tensors]

        all_results: Dict[str, List[BenchmarkTensorResult]] = {}
        per_type: Dict[str, Dict[float, List[BenchmarkTensorResult]]] = {}
        per_method: Dict[str, Dict[float, List[BenchmarkTensorResult]]] = {}
        n_total = 0
        n_fail = 0

        t0 = time.perf_counter()
        for name, (shape, dtype_str, offset, nbytes) in items:
            try:
                tensor = self._io.read(model_path, shape, dtype_str, offset, nbytes)
                ttype = self._classify_by_name(name)
                for tr in ratios:
                    result = self._test_single_tensor(
                        tensor, ttype, shape, tr, name=name
                    )
                    all_results.setdefault(name, []).append(result)
                    per_type.setdefault(ttype, {}).setdefault(tr, []).append(result)
                    per_method.setdefault(result.method, {}).setdefault(tr, []).append(
                        result
                    )
                    n_total += 1
            except Exception as e:
                n_fail += 1
                logger.warning("  %-40s FAILED: %s", name[-40:], e)

        elapsed = time.perf_counter() - t0
        return BenchmarkRunResult(
            model_name=model_path,
            target_ratios=ratios,
            tensor_results=all_results,
            per_type_results=per_type,
            per_method_results=per_method,
            cascade_results={},
            streaming_results={},
            run_time=elapsed,
            num_tensors=n_total,
            num_failures=n_fail,
        )

    def benchmark_single_method(
        self,
        method_name: str,
        tensor: np.ndarray,
        target_ratio: float = 100.0,
        name: str = "",
        params: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkTensorResult:
        engine = self._engine
        profile = engine.profile_tensor(tensor, name=name)
        inst = engine._methods.get(method_name)
        if inst is None:
            raise ValueError(f"Method '{method_name}' not found")

        t1 = time.perf_counter()
        if params:
            data, meta = inst.compress(tensor, **params)
        else:
            data, meta = inst.compress(tensor)
        ct = time.perf_counter() - t1

        t2 = time.perf_counter()
        recon = inst.decompress(data, meta)
        dt = time.perf_counter() - t2

        if recon.shape != tensor.shape:
            recon = recon.reshape(tensor.shape)

        metrics = self._loss.compute_tensor_metrics(
            tensor, recon, tensor.nbytes, len(data)
        )
        ttype = self._classify_by_name(name) if name else "weight"

        return BenchmarkTensorResult(
            tensor_name=name or method_name,
            tensor_type=ttype,
            shape=tensor.shape,
            method=method_name,
            target_ratio=target_ratio,
            achieved_ratio=metrics.compression_ratio,
            metrics=metrics,
            compression_time=ct,
            decompression_time=dt,
            params=params or {},
        )

    def benchmark_cascade(
        self,
        tensor: np.ndarray,
        target_ratio: float = 1200.0,
        name: str = "",
        pattern_name: Optional[str] = None,
        max_error: float = 0.01,
    ) -> BenchmarkTensorResult:
        engine = self._engine
        t1 = time.perf_counter()
        data, meta, ratio, error = engine.compress_cascade(
            tensor, target_ratio, max_error, name, pattern_name
        )
        ct = time.perf_counter() - t1

        t2 = time.perf_counter()
        recon = engine.decompress(data, meta)
        dt = time.perf_counter() - t2

        if recon.shape != tensor.shape:
            recon = recon.reshape(tensor.shape)

        metrics = self._loss.compute_tensor_metrics(
            tensor, recon, tensor.nbytes, len(data)
        )
        ttype = self._classify_by_name(name) if name else "weight"
        pattern = meta.get("pattern_name", meta.get("cascade_pattern", "auto"))

        return BenchmarkTensorResult(
            tensor_name=name or "cascade",
            tensor_type=ttype,
            shape=tensor.shape,
            method=f"cascade:{pattern}",
            target_ratio=target_ratio,
            achieved_ratio=metrics.compression_ratio,
            metrics=metrics,
            compression_time=ct,
            decompression_time=dt,
            cascade_pattern=pattern,
        )

    def benchmark_streaming(
        self,
        tensor: np.ndarray,
        target_ratio: float = 100.0,
        name: str = "",
    ) -> BenchmarkTensorResult:
        t0 = time.perf_counter()
        engine = self._engine
        profile = engine.profile_tensor(tensor, name=name)
        error_budget = 0.01 / max(target_ratio, 1.0)
        methods = engine._select_methods(profile, error_budget, target_ratio)

        data = b""
        meta: Dict[str, Any] = {}
        chunk_size = max(1, tensor.nbytes // 100)
        n_chunks = max(1, tensor.nbytes // chunk_size)
        peak_mem = 0.0

        try:
            import psutil as _psutil_mod

            proc = _psutil_mod.Process()
            peak_mem = proc.memory_info().rss / 1024 / 1024
        except Exception:
            pass

        t1 = time.perf_counter()
        flat = tensor.ravel()
        for i in range(n_chunks):
            chunk = flat[i * chunk_size : min((i + 1) * chunk_size, len(flat))]
            for m in methods:
                inst = m.get("instance")
                if inst is None:
                    continue
                chunk_data, chunk_meta = inst.compress(chunk, **m.get("params", {}))
                data += chunk_data
                meta = chunk_meta
                break

            try:
                import psutil as _psutil_mod

                proc = _psutil_mod.Process()
                current_mem = proc.memory_info().rss / 1024 / 1024
                peak_mem = max(peak_mem, current_mem)
            except Exception:
                pass

        ct = time.perf_counter() - t1

        decompressed_chunks = []
        offset = 0
        t2 = time.perf_counter()
        for m in methods:
            inst = m.get("instance")
            if inst is None:
                continue
            chunk_sizes = [len(data) // n_chunks] * n_chunks
            for cs in chunk_sizes:
                chunk_data = data[offset : offset + cs]
                offset += cs
                recon_chunk = inst.decompress(chunk_data, meta)
                decompressed_chunks.append(recon_chunk)
            break
        recon = np.concatenate(decompressed_chunks)[: len(flat)].reshape(tensor.shape)
        dt = time.perf_counter() - t2

        metrics = self._loss.compute_tensor_metrics(
            tensor, recon, tensor.nbytes, len(data)
        )
        ttype = self._classify_by_name(name) if name else "weight"

        return BenchmarkTensorResult(
            tensor_name=name or "streaming",
            tensor_type=ttype,
            shape=tensor.shape,
            method="streaming",
            target_ratio=target_ratio,
            achieved_ratio=metrics.compression_ratio,
            metrics=metrics,
            compression_time=ct,
            decompression_time=dt,
            streaming_peak_memory_mb=peak_mem,
        )

    def benchmark_multi_ratio(
        self,
        tensor: np.ndarray,
        target_ratios: Optional[List[float]] = None,
        name: str = "",
    ) -> List[BenchmarkTensorResult]:
        ratios = target_ratios if target_ratios is not None else TARGET_RATIOS
        results: List[BenchmarkTensorResult] = []
        for tr in ratios:
            try:
                result = self._test_single_tensor(
                    tensor, self._classify_by_name(name), tensor.shape, tr, name=name
                )
                results.append(result)
            except Exception as e:
                logger.warning("  %-40s @ %dx failed: %s", name[:40], tr, e)
        return results

    def benchmark_all(
        self,
        model_path: str = "",
        target_ratios: Optional[List[float]] = None,
        max_tensors: Optional[int] = None,
        use_synthetic: bool = True,
        seed: int = 42,
    ) -> BenchmarkRunResult:
        if use_synthetic or not model_path:
            return self.benchmark_synthetic(target_ratios, seed=seed)
        return self.benchmark_real_model(model_path, target_ratios, max_tensors)

    def _test_single_tensor(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        shape: Tuple[int, ...],
        target_ratio: float,
        name: str = "",
    ) -> BenchmarkTensorResult:
        engine = self._engine
        profile = engine.profile_tensor(tensor, name=name)
        error_budget = 0.01 / max(target_ratio, 1.0)
        error_budget = max(min(error_budget, 0.05), 0.0001)

        methods = engine._select_methods(profile, error_budget, target_ratio)

        t1 = time.perf_counter()
        data, meta, ratio, error = engine.compress_tensor_with_validation(
            tensor, profile, methods, error_budget
        )
        ct = time.perf_counter() - t1

        t2 = time.perf_counter()
        recon = engine.decompress(data, meta)
        dt = time.perf_counter() - t2

        if recon.shape != tensor.shape:
            recon = recon.reshape(tensor.shape)

        metrics = self._loss.compute_tensor_metrics(
            tensor, recon, tensor.nbytes, len(data)
        )
        method_name = meta.get("method", "unknown")

        return BenchmarkTensorResult(
            tensor_name=name or f"{tensor_type}_{shape}",
            tensor_type=tensor_type,
            shape=shape,
            method=method_name,
            target_ratio=target_ratio,
            achieved_ratio=metrics.compression_ratio,
            metrics=metrics,
            compression_time=ct,
            decompression_time=dt,
        )

    @staticmethod
    def _classify_by_name(name: str) -> str:
        if not name:
            return "weight"
        nl = name.lower()
        if any(k in nl for k in ("embed", "tok_embeddings", "wte")):
            return "embedding"
        if any(k in nl for k in ("attn_q", "q_proj", "wq", "query")):
            return "attention_q"
        if any(k in nl for k in ("attn_k", "k_proj", "wk", "key")):
            return "attention_k"
        if any(k in nl for k in ("attn_v", "v_proj", "wv", "value")):
            return "attention_v"
        if any(k in nl for k in ("attn_o", "o_proj", "wo", "out")):
            return "attention_o"
        if "qkv" in nl:
            return "qkv_fused"
        if any(k in nl for k in ("ffn_gate", "gate_proj", "w1")):
            return "ffn_gate"
        if any(k in nl for k in ("ffn_up", "up_proj", "w3")):
            return "ffn_up"
        if any(k in nl for k in ("ffn_down", "down_proj", "w2")):
            return "ffn_down"
        if any(k in nl for k in ("ffn", "mlp")):
            return "ffn_gate"
        if any(k in nl for k in ("norm", "rms")):
            return "norm"
        if any(k in nl for k in ("output", "lm_head", "head")):
            return "output"
        return "weight"


class _SafetensorsIOAccessor(_SafetensorsIO):
    pass
