"""
Precision-Preserving Compression Engine
========================================
Orchestrator for quantization methods with profiling, error budgeting,
cross-layer optimization, and quality grading.

Uses engine-level built-in methods (_BlockINT8, _HadamardINT8) as
the actual compressors — this module provides the orchestration layer.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TensorProfile:
    name: str = ""
    shape: Tuple[int, ...] = (0,)
    dtype: str = ""
    n_elements: int = 0
    nbytes: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    sparsity: float = 0.0
    effective_rank: float = 0.0
    spectral_entropy: float = 0.0
    recommended_method: str = "block_int8"


@dataclass
class PrecisionResult:
    method: str
    compressed_data: Any
    metadata: dict
    original_shape: Tuple[int, ...]
    original_nbytes: int
    compressed_nbytes: int
    ratio: float
    relative_error: float
    snr_db: float
    psnr_db: float
    cosine_similarity: float
    time_ms: float
    tier: int = 1
    params: dict = field(default_factory=dict)

    @property
    def quality_grade(self) -> str:
        if self.relative_error < 1e-10:
            return "S"
        if self.relative_error < 0.0001:
            return "A"
        if self.relative_error < 0.001:
            return "B"
        if self.relative_error < 0.01:
            return "C"
        if self.relative_error < 0.05:
            return "D"
        return "F"


@dataclass
class PrecisionCompressed:
    data: Any
    method: str
    original_shape: Tuple[int, ...]
    original_nbytes: int
    compressed_nbytes: int
    ratio: float
    relative_error: float
    snr_db: float
    psnr_db: float
    cosine_similarity: float
    metadata: dict
    decompress_fn: Any = field(repr=False, default=None)

    def quality_grade(self) -> str:
        if self.relative_error < 1e-10:
            return "S"
        if self.relative_error < 0.0001:
            return "A"
        if self.relative_error < 0.001:
            return "B"
        if self.relative_error < 0.01:
            return "C"
        if self.relative_error < 0.05:
            return "D"
        return "F"


def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> dict:
    o = orig.astype(np.float64).ravel()
    r = recon.astype(np.float64).ravel()
    noise = o - r
    mse = float(np.mean(noise**2))
    signal_power = float(np.mean(o**2)) + 1e-30
    snr_db = 10.0 * math.log10(signal_power / (mse + 1e-30))
    max_val = float(np.max(np.abs(o)))
    psnr_db = 10.0 * math.log10(max_val**2 / (mse + 1e-30)) if max_val > 0 else snr_db
    rel_error = float(np.linalg.norm(noise) / (np.linalg.norm(o) + 1e-30))
    cos_sim = float(np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-30))
    return {
        "mse": mse,
        "snr_db": snr_db,
        "psnr_db": psnr_db,
        "relative_error": rel_error,
        "cosine_similarity": cos_sim,
    }


class TensorProfiler:
    def profile(self, tensor: np.ndarray, name: str = "") -> TensorProfile:
        tensor = np.asarray(tensor, dtype=np.float32)
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        t = TensorProfile(
            name=name,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            n_elements=n,
            nbytes=tensor.nbytes,
        )
        if n == 0:
            return t
        t.mean = float(np.mean(flat))
        t.std = float(np.std(flat))
        t.min_val = float(np.min(flat))
        t.max_val = float(np.max(flat))
        t.sparsity = float(np.mean(np.abs(flat) < 1e-10))
        if tensor.ndim >= 2 and all(s > 1 for s in tensor.shape[:2]):
            try:
                mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
                s_vals = np.linalg.svd(
                    mat[: min(mat.shape[0], 256), : min(mat.shape[1], 256)],
                    compute_uv=False,
                )
                s_norm = s_vals / (np.sum(s_vals) + 1e-10)
                nonzero = s_norm[s_norm > 1e-10]
                t.effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
            except np.linalg.LinAlgError:
                t.effective_rank = 1.0
        if n >= 4:
            try:
                spectrum = np.abs(np.fft.fft(flat[: min(n, 4096)]))
                power = spectrum / (np.sum(spectrum) + 1e-10)
                t.spectral_entropy = float(
                    -np.sum(power * np.log2(power + 1e-10)) / math.log2(min(n, 4096))
                )
            except Exception:
                pass
        t.recommended_method = "block_int8"
        return t


class ErrorBudgetAllocator:
    SENSITIVITY: Dict[str, float] = {
        "embed": 1.0,
        "tok_embeddings": 1.0,
        "q_proj": 1.0,
        "query": 1.0,
        "k_proj": 0.92,
        "key": 0.92,
        "v_proj": 0.88,
        "value": 0.88,
        "o_proj": 1.0,
        "gate_proj": 0.55,
        "up_proj": 0.60,
        "down_proj": 0.65,
        "norm": 0.50,
        "ln_": 0.50,
        "lm_head": 1.0,
    }

    def _get_sensitivity(self, name: str) -> float:
        name_lower = name.lower()
        for key, val in self.SENSITIVITY.items():
            if key in name_lower:
                return val
        return 0.5

    def allocate(
        self, tensors: Dict[str, np.ndarray], total_error: float = 0.01
    ) -> Dict[str, float]:
        return {name: total_error for name in tensors}


class CrossLayerOptimizer:
    def find_correlated_layers(
        self, tensors: Dict[str, np.ndarray], threshold: float = 0.5
    ) -> List[Tuple[str, str, float]]:
        names = list(tensors.keys())
        correlated = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if tensors[names[i]].shape != tensors[names[j]].shape:
                    continue
                t1 = tensors[names[i]].ravel().astype(np.float64)
                t2 = tensors[names[j]].ravel().astype(np.float64)
                n1, n2 = np.linalg.norm(t1), np.linalg.norm(t2)
                if n1 < 1e-10 or n2 < 1e-10:
                    continue
                cos = float(np.dot(t1, t2) / (n1 * n2))
                if cos >= threshold:
                    correlated.append((names[i], names[j], cos))
        return sorted(correlated, key=lambda x: -x[2])

    def delta_int8_compress(
        self, w1: np.ndarray, w2: np.ndarray, block_size: int = 128
    ) -> PrecisionResult:
        from spectralstream.compression.engine._methods import _BlockINT8

        t0 = time.monotonic()
        w1 = np.asarray(w1, dtype=np.float32)
        w2 = np.asarray(w2, dtype=np.float32)
        delta = w2 - w1
        int8 = _BlockINT8()
        delta_bytes, meta = int8.compress(delta, block_size=block_size)
        compressed_nbytes = w1.nbytes + len(delta_bytes)
        ratio = w2.nbytes / max(compressed_nbytes, 1)
        delta_recon = int8.decompress(delta_bytes, meta)
        w2_recon = w1 + delta_recon.reshape(w2.shape)
        metrics = compute_metrics(w2, w2_recon)
        return PrecisionResult(
            method="delta_int8",
            compressed_data={
                "base": w1,
                "delta_compressed": delta_bytes,
                "delta_metadata": meta,
            },
            metadata={"orig_shape": w2.shape},
            original_shape=w2.shape,
            original_nbytes=w2.nbytes,
            compressed_nbytes=compressed_nbytes,
            ratio=ratio,
            relative_error=metrics["relative_error"],
            snr_db=metrics["snr_db"],
            psnr_db=metrics["psnr_db"],
            cosine_similarity=metrics["cosine_similarity"],
            time_ms=(time.monotonic() - t0) * 1000,
            tier=2,
        )


class PrecisionEngine:
    def __init__(self):
        self.profiler = TensorProfiler()
        self.cross_layer = CrossLayerOptimizer()
        self.budget_allocator = ErrorBudgetAllocator()

    def _get_method(self, name: str):
        from spectralstream.compression.engine._methods import (
            _BlockINT8,
            _HadamardINT8,
        )

        registry = {
            "block_int8": _BlockINT8,
            "block_int8_64": _BlockINT8,
            "block_int8_256": _BlockINT8,
            "hadamard_int8": _HadamardINT8,
        }
        cls = registry.get(name)
        if cls is None:
            return None
        return cls()

    def compress(
        self,
        tensor: np.ndarray,
        target_error: float = 0.01,
        target_ratio: Optional[float] = None,
        name: str = "",
    ) -> PrecisionCompressed:
        tensor = np.asarray(tensor, dtype=np.float32)
        best_result = None
        best_ratio = 0.0

        for method_name, block_size in [
            ("block_int8", 128),
            ("block_int8_64", 64),
            ("block_int8_256", 256),
            ("hadamard_int8", 128),
        ]:
            try:
                method = self._get_method(method_name)
                if method is None:
                    continue
                params = {"block_size": block_size}
                if method_name.startswith("block_int8"):
                    bs = int(method_name.split("_")[-1])
                    params["block_size"] = bs
                data_bytes, meta = method.compress(tensor, **params)
                recon = method.decompress(data_bytes, meta)
                metrics = compute_metrics(tensor, recon)
                compressed_nbytes = len(data_bytes)
                ratio = tensor.nbytes / max(compressed_nbytes, 1)
                rel_err = metrics["relative_error"]
                if rel_err <= target_error and ratio > best_ratio:
                    dt = 0.0
                    result = PrecisionResult(
                        method=method_name,
                        compressed_data=data_bytes,
                        metadata=meta,
                        original_shape=tensor.shape,
                        original_nbytes=tensor.nbytes,
                        compressed_nbytes=compressed_nbytes,
                        ratio=ratio,
                        relative_error=rel_err,
                        snr_db=metrics["snr_db"],
                        psnr_db=metrics["psnr_db"],
                        cosine_similarity=metrics["cosine_similarity"],
                        time_ms=dt,
                        tier=1,
                        params=params,
                    )
                    best_result = result
                    best_ratio = ratio
            except Exception as e:
                logger.debug("Method %s failed: %s", method_name, e)

        if best_result is None:
            return PrecisionCompressed(
                data=tensor,
                method="passthrough",
                original_shape=tensor.shape,
                original_nbytes=tensor.nbytes,
                compressed_nbytes=tensor.nbytes,
                ratio=1.0,
                relative_error=0.0,
                snr_db=999.0,
                psnr_db=999.0,
                cosine_similarity=1.0,
                metadata={},
            )

        def _decompress(data, meta):
            method = self._get_method(
                best_result.method.replace("_64", "").replace("_256", "")
            )
            if method is None:
                return data
            flat = method.decompress(data, meta)
            return flat.reshape(best_result.original_shape)

        return PrecisionCompressed(
            data=best_result.compressed_data,
            method=best_result.method,
            original_shape=best_result.original_shape,
            original_nbytes=best_result.original_nbytes,
            compressed_nbytes=best_result.compressed_nbytes,
            ratio=best_result.ratio,
            relative_error=best_result.relative_error,
            snr_db=best_result.snr_db,
            psnr_db=best_result.psnr_db,
            cosine_similarity=best_result.cosine_similarity,
            metadata=best_result.metadata,
            decompress_fn=_decompress,
        )

    def decompress(self, compressed: PrecisionCompressed) -> np.ndarray:
        if compressed.method == "passthrough":
            return compressed.data
        if compressed.decompress_fn:
            return compressed.decompress_fn(compressed.data, compressed.metadata)
        return compressed.data

    def compress_model(
        self,
        tensors: Dict[str, np.ndarray],
        error_budget: float = 0.01,
    ) -> Dict[str, PrecisionCompressed]:
        allocations = self.budget_allocator.allocate(tensors, error_budget)
        results = {}
        for name, tensor in tensors.items():
            per_tensor_error = allocations.get(name, error_budget)
            results[name] = self.compress(
                tensor, target_error=per_tensor_error, name=name
            )
        return results

    def benchmark(self, tensor: np.ndarray, name: str = "") -> List[PrecisionResult]:
        tensor = np.asarray(tensor, dtype=np.float32)
        results = []
        from spectralstream.compression.engine._methods import _BlockINT8, _HadamardINT8

        block8 = _BlockINT8()
        had8 = _HadamardINT8()
        for bs in [32, 64, 128, 256]:
            try:
                data_bytes, meta = block8.compress(tensor, block_size=bs)
                recon = block8.decompress(data_bytes, meta)
                metrics = compute_metrics(tensor, recon)
                compressed_nbytes = len(data_bytes)
                results.append(
                    PrecisionResult(
                        method="block_int8",
                        compressed_data=data_bytes,
                        metadata=meta,
                        original_shape=tensor.shape,
                        original_nbytes=tensor.nbytes,
                        compressed_nbytes=compressed_nbytes,
                        ratio=tensor.nbytes / max(compressed_nbytes, 1),
                        relative_error=metrics["relative_error"],
                        snr_db=metrics["snr_db"],
                        psnr_db=metrics["psnr_db"],
                        cosine_similarity=metrics["cosine_similarity"],
                        time_ms=0.0,
                        tier=1,
                        params={"block_size": bs},
                    )
                )
            except Exception:
                pass
        for bs in [64, 128, 256]:
            try:
                data_bytes, meta = had8.compress(tensor, block_size=bs)
                recon = had8.decompress(data_bytes, meta)
                metrics = compute_metrics(tensor, recon)
                compressed_nbytes = len(data_bytes)
                results.append(
                    PrecisionResult(
                        method="hadamard_int8",
                        compressed_data=data_bytes,
                        metadata=meta,
                        original_shape=tensor.shape,
                        original_nbytes=tensor.nbytes,
                        compressed_nbytes=compressed_nbytes,
                        ratio=tensor.nbytes / max(compressed_nbytes, 1),
                        relative_error=metrics["relative_error"],
                        snr_db=metrics["snr_db"],
                        psnr_db=metrics["psnr_db"],
                        cosine_similarity=metrics["cosine_similarity"],
                        time_ms=0.0,
                        tier=1,
                        params={"block_size": bs},
                    )
                )
            except Exception:
                pass
        return sorted(
            results, key=lambda r: (-1 if r.relative_error <= 0.01 else 1, -r.ratio)
        )
