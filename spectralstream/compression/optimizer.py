"""
Compression Optimizer — tests each method against real model tensors,
finds optimal parameters for <0.6% loss at maximum ratio.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.methods import METHOD_CLASSES, get_method

logger = logging.getLogger(__name__)


@dataclass
class MethodResult:
    method_name: str
    tensor_name: str
    params: Dict[str, Any]
    compression_ratio: float
    relative_error: float
    snr_db: float
    cosine_similarity: float
    computation_time: float
    compressed_size: int
    original_size: int


@dataclass
class TensorSample:
    name: str
    tensor: np.ndarray
    shape: Tuple[int, ...]
    nbytes: int


class TensorSampler:
    """Sample tensors from a safetensors file for optimization."""

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

    def __init__(
        self, model_path: str, max_samples: int = 50, max_elements: int = 500_000
    ) -> None:
        self.model_path = model_path
        self.max_samples = max_samples
        self.max_elements = max_elements

    def sample_tensors(self) -> List[TensorSample]:
        """Scan model and return representative tensor samples."""
        if not os.path.exists(self.model_path):
            logger.warning("Model path does not exist: %s", self.model_path)
            return self._generate_synthetic_samples()

        try:
            with open(self.model_path, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header_json = f.read(header_len)
            header = json.loads(header_json)
        except Exception as e:
            logger.warning("Failed to read safetensors header: %s", e)
            return self._generate_synthetic_samples()

        data_start = 8 + len(header_json)
        tensor_infos: List[Tuple[str, Tuple[int, ...], str, int, int]] = []

        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info.get("dtype", "F32")
            shape = tuple(info.get("shape", []))
            offsets = info.get("data_offsets", [0, 0])
            tensor_infos.append(
                (name, shape, dtype, data_start + offsets[0], offsets[1] - offsets[0])
            )

        tensor_infos.sort(key=lambda x: x[4], reverse=True)

        samples: List[TensorSample] = []
        for name, shape, dtype_str, offset, nbytes in tensor_infos[: self.max_samples]:
            try:
                np_dtype = self.DTYPE_MAP.get(dtype_str, np.float32)
                with open(self.model_path, "rb") as f:
                    f.seek(offset)
                    raw = f.read(
                        min(nbytes, self.max_elements * np.dtype(np_dtype).itemsize)
                    )
                tensor = np.frombuffer(raw, dtype=np_dtype)
                if shape:
                    try:
                        tensor = tensor.reshape(shape)
                    except ValueError:
                        pass
                tensor = tensor.astype(np.float32)
                samples.append(
                    TensorSample(
                        name=name,
                        tensor=tensor,
                        shape=tensor.shape,
                        nbytes=tensor.nbytes,
                    )
                )
                logger.debug(
                    "Sampled %s: shape=%s, bytes=%d", name, tensor.shape, tensor.nbytes
                )
            except Exception as e:
                logger.debug("Failed to sample %s: %s", name, e)

        if not samples:
            return self._generate_synthetic_samples()

        return samples

    def _generate_synthetic_samples(self) -> List[TensorSample]:
        """Generate synthetic tensors mimicking real model distributions."""
        rng = np.random.RandomState(42)
        samples: List[TensorSample] = []

        configs = [
            ("embedding", (262144, 1536), 0.5),
            ("q_proj.weight", (1536, 256), 1.0),
            ("k_proj.weight", (1536, 256), 1.0),
            ("v_proj.weight", (1536, 256), 1.0),
            ("o_proj.weight", (256, 1536), 1.0),
            ("gate_proj.weight", (1536, 6144), 1.0),
            ("up_proj.weight", (1536, 6144), 1.0),
            ("down_proj.weight", (6144, 1536), 1.0),
            ("norm.weight", (1536,), 0.5),
            ("attention.weight", (1536, 1536), 1.5),
        ]

        for name, shape, std_scale in configs:
            tensor = rng.randn(*shape).astype(np.float32) * std_scale
            samples.append(
                TensorSample(
                    name=name, tensor=tensor, shape=shape, nbytes=tensor.nbytes
                )
            )

        logger.info(
            "Generated %d synthetic samples (model not available)", len(samples)
        )
        return samples


class CompressionOptimizer:
    """
    Tests each compression method against real model tensors.
    Finds optimal parameters for each method to achieve <0.6% loss.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.results: Dict[str, List[MethodResult]] = {}
        self.sampler = TensorSampler(model_path)
        self._best_method_cache: Dict[str, str] = {}

    def optimize_method(
        self,
        method_name: str,
        tensor_sample: TensorSample,
        param_grid: Dict[str, List[Any]],
    ) -> List[MethodResult]:
        """Grid search over parameters for a method-tensor pair."""
        cls = METHOD_CLASSES.get(method_name)
        if cls is None:
            logger.warning("Method '%s' not found in registry", method_name)
            return []

        instance = cls()
        results: List[MethodResult] = []
        tensor = tensor_sample.tensor

        keys = list(param_grid.keys())
        values = list(param_grid.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            try:
                t0 = time.perf_counter()
                compressed_data, meta = instance.compress(tensor, **params)
                recon = instance.decompress(compressed_data, meta).reshape(tensor.shape)
                dt = time.perf_counter() - t0

                noise = (
                    tensor.astype(np.float64).ravel() - recon.astype(np.float64).ravel()
                )
                mse = float(np.mean(noise**2))
                signal_power = (
                    float(np.mean(tensor.astype(np.float64).ravel() ** 2)) + 1e-30
                )
                rel_error = float(
                    np.linalg.norm(noise)
                    / (np.linalg.norm(tensor.astype(np.float64).ravel()) + 1e-30)
                )
                snr = 10.0 * math.log10(signal_power / (mse + 1e-30))
                cos_sim = float(
                    np.dot(
                        tensor.astype(np.float64).ravel(),
                        recon.astype(np.float64).ravel(),
                    )
                    / (
                        np.linalg.norm(tensor.astype(np.float64).ravel())
                        * np.linalg.norm(recon.astype(np.float64).ravel())
                        + 1e-30
                    )
                )
                ratio = tensor.nbytes / max(len(compressed_data), 1)

                results.append(
                    MethodResult(
                        method_name=method_name,
                        tensor_name=tensor_sample.name,
                        params=params,
                        compression_ratio=ratio,
                        relative_error=rel_error,
                        snr_db=snr,
                        cosine_similarity=cos_sim,
                        computation_time=dt,
                        compressed_size=len(compressed_data),
                        original_size=tensor.nbytes,
                    )
                )
            except Exception as e:
                logger.debug(
                    "Method '%s' with params %s failed: %s", method_name, params, e
                )

        results.sort(
            key=lambda r: (r.relative_error <= 0.006, -r.compression_ratio),
            reverse=True,
        )
        return results

    def optimize_all_methods(
        self, tensor_names: Optional[List[str]] = None
    ) -> Dict[str, List[MethodResult]]:
        """Optimize all available methods against sampled tensors."""
        samples = self.sampler.sample_tensors()

        if tensor_names:
            samples = [s for s in samples if s.name in tensor_names]

        param_grids = self._get_param_grids()

        all_results: Dict[str, List[MethodResult]] = {}
        for sample in samples[:10]:  # Limit to 10 representative tensors
            for method_name in list(METHOD_CLASSES.keys())[:50]:  # Test top 50 methods
                if method_name not in param_grids:
                    continue
                results = self.optimize_method(
                    method_name, sample, param_grids[method_name]
                )
                if results:
                    all_results.setdefault(method_name, []).extend(results)

            logger.info("Optimized methods for %s", sample.name)

        # Summarize best results per method
        self.results = all_results
        return all_results

    def find_best_method(self, tensor: np.ndarray, profile: Any) -> str:
        """Find the absolute best method for a given tensor."""
        tensor_sample = TensorSample(
            name=getattr(profile, "name", "unknown"),
            tensor=tensor,
            shape=tensor.shape,
            nbytes=tensor.nbytes,
        )

        best_method = "block_int8"
        best_score = -float("inf")

        for method_name in list(METHOD_CLASSES.keys())[:30]:
            param_grid = self._get_method_params(method_name, profile)
            if not param_grid:
                continue
            results = self.optimize_method(method_name, tensor_sample, param_grid)
            if not results:
                continue

            best = results[0]
            if best.relative_error <= 0.006:
                score = best.compression_ratio * (1.0 - best.relative_error * 10)
                if score > best_score:
                    best_score = score
                    best_method = method_name

        self._best_method_cache[tensor_sample.name] = best_method
        return best_method

    def _get_param_grids(self) -> Dict[str, Dict[str, List[Any]]]:
        """Define parameter grids for each method category."""
        return {
            "block_int8": {"block_size": [32, 64, 128, 256]},
            "block_int4": {"block_size": [16, 32, 64]},
            "hadamard_int8": {"block_size": [64, 128, 256]},
            "hadamard_int4": {"block_size": [32, 64]},
            "sparsity_int4": {"group_size": [16, 32, 64]},
            "mixed_precision": {"block_size": [32, 64, 128]},
            "nf4": {"block_size": [32, 64]},
            "binary_quant": {"block_size": [64, 128]},
            "ternary_quant": {"block_size": [64, 128]},
            "lloyd_max_quant": {"block_size": [32, 64, 128], "n_levels": [8, 16, 32]},
            "e8_lattice": {"block_size": [32, 64]},
            "adaptive_group_quant": {
                "block_size": [32, 64, 128],
                "group_size": [16, 32],
            },
            "outlier_aware_quant": {
                "block_size": [64, 128],
                "outlier_percentile": [95, 99, 99.5],
            },
            "residual_quant": {"block_size": [64, 128], "n_stages": [2, 3]},
            "dynamic_bitwidth": {"block_size": [64, 128]},
            "bqq_binary_quadratic": {"block_size": [64, 128]},
            "gptq_quant": {"block_size": [64, 128]},
            "awq_quant": {"block_size": [64, 128]},
            "block_floating_point": {"block_size": [32, 64, 128]},
            "svd_truncated": {"rank": [16, 32, 64, 128]},
            "tensor_train": {"rank": [8, 16, 32]},
            "tensor_ring": {"rank": [8, 16, 32]},
            "cp_decomposition": {"rank": [8, 16, 32]},
            "tucker_decomposition": {"rank": [8, 16, 32]},
            "butterfly": {"block_size": [32, 64]},
            "monarch": {"block_size": [32, 64]},
            "dct_block": {
                "block_size": [64, 128, 256],
                "keep_ratio": [0.1, 0.2, 0.3, 0.5],
            },
            "dct_2d": {"block_size": [32, 64], "keep_ratio": [0.1, 0.2, 0.3]},
            "fwht": {"block_size": [64, 128], "keep_ratio": [0.2, 0.3, 0.5]},
            "wavelet_haar": {"level": [2, 3, 4], "keep_ratio": [0.1, 0.2, 0.3]},
            "wavelet_daubechies": {"level": [2, 3], "keep_ratio": [0.1, 0.2, 0.3]},
            "fourier": {"keep_ratio": [0.1, 0.2, 0.3]},
            "block_diagonal": {"block_size": [64, 128]},
            "toeplitz": {"block_size": [64, 128]},
            "cascade_2_stage": {"block_size": [64, 128]},
            "cascade_3_stage": {"block_size": [64, 128]},
        }

    def _get_method_params(
        self, method_name: str, profile: Any
    ) -> Dict[str, List[Any]]:
        """Get appropriate param grid for a method given a tensor profile."""
        default_grids = self._get_param_grids()
        if method_name in default_grids:
            return default_grids[method_name]
        return {"block_size": [64, 128, 256]}

    def generate_method_benchmark(self) -> Dict[str, Any]:
        """Generate full benchmark of all methods vs all tensor types."""
        samples = self.sampler.sample_tensors()
        benchmark: Dict[str, Any] = {
            "model_path": self.model_path,
            "num_samples": len(samples),
            "methods": {},
            "best_per_tensor_type": {},
            "overall_ranking": [],
        }

        tensor_types: Dict[str, List[TensorSample]] = {}
        for sample in samples:
            ttype = self._classify_tensor_name(sample.name)
            tensor_types.setdefault(ttype, []).append(sample)

        for method_name in METHOD_CLASSES:
            method_results: List[MethodResult] = []
            for sample in samples[:5]:
                param_grid = self._get_method_params(method_name, None)
                results = self.optimize_method(method_name, sample, param_grid)
                method_results.extend(results)

            if method_results:
                best = method_results[0]
                avg_error = float(np.mean([r.relative_error for r in method_results]))
                avg_ratio = float(
                    np.mean([r.compression_ratio for r in method_results])
                )
                methods_under_06pct = [
                    r for r in method_results if r.relative_error <= 0.006
                ]

                benchmark["methods"][method_name] = {
                    "best_ratio": best.compression_ratio,
                    "best_params": best.params,
                    "best_error": best.relative_error,
                    "avg_error": avg_error,
                    "avg_ratio": avg_ratio,
                    "num_under_threshold": len(methods_under_06pct),
                    "total_trials": len(method_results),
                    "tensor_tested": best.tensor_name,
                }

        # Find best method per tensor type
        for ttype, type_samples in tensor_types.items():
            best_method = None
            best_ratio = 0
            for method_name, mdata in benchmark["methods"].items():
                if mdata["best_error"] <= 0.006 and mdata["best_ratio"] > best_ratio:
                    best_ratio = mdata["best_ratio"]
                    best_method = method_name
            if best_method:
                benchmark["best_per_tensor_type"][ttype] = {
                    "method": best_method,
                    "ratio": best_ratio,
                }

        # Overall ranking: methods that achieve <0.6% error, sorted by ratio
        ranking = sorted(
            [
                (m, d["best_ratio"], d["best_error"])
                for m, d in benchmark["methods"].items()
                if d["best_error"] <= 0.006
            ],
            key=lambda x: -x[1],
        )
        benchmark["overall_ranking"] = [
            {"method": m, "ratio": r, "error": e} for m, r, e in ranking
        ]

        return benchmark

    def save_benchmark(self, path: str) -> None:
        """Save full benchmark results to JSON."""
        benchmark = self.generate_method_benchmark()
        with open(path, "w") as f:
            json.dump(benchmark, f, indent=2, default=str)
        logger.info("Benchmark saved to %s", path)

    @staticmethod
    def _classify_tensor_name(name: str) -> str:
        nl = name.lower()
        if any(k in nl for k in ("embed", "wte")):
            return "embedding"
        if any(k in nl for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn")):
            return "attention"
        if any(k in nl for k in ("gate_proj", "up_proj", "down_proj", "ffn", "mlp")):
            return "ffn"
        if any(k in nl for k in ("norm", "ln_", "rms")):
            return "norm"
        if any(k in nl for k in ("lm_head", "output")):
            return "head"
        return "other"

    @staticmethod
    def compute_optimal_params(
        method_name: str, tensor: np.ndarray, target_error: float = 0.006
    ) -> Dict[str, Any]:
        """Compute optimal parameters for a method on a given tensor."""
        flat = tensor.ravel().astype(np.float64)
        n = len(flat)
        std = float(np.std(flat))
        dynamic_range = float(np.max(np.abs(flat)))

        if method_name in ("block_int8", "hadamard_int8"):
            if n > 1_000_000:
                return {"block_size": 256}
            elif n > 100_000:
                return {"block_size": 128}
            else:
                return {"block_size": 64}

        if method_name in ("block_int4", "hadamard_int4"):
            if target_error <= 0.006:
                return {"block_size": 32}
            return {"block_size": 64}

        if method_name == "sparsity_int4":
            return {"group_size": 32}

        if "svd" in method_name or "tensor" in method_name:
            if std > 1.0:
                return {"rank": 64}
            return {"rank": 32}

        if "dct" in method_name or "wavelet" in method_name or "fourier" in method_name:
            if dynamic_range > 10.0:
                return {"keep_ratio": 0.3, "block_size": 64}
            return {"keep_ratio": 0.2, "block_size": 128}

        return {"block_size": 128}
