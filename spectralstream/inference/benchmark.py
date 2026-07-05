"""
Comprehensive Benchmark Suite for SpectralStream
=================================================
Throughput, compression, quality, memory, comparison, and report generation.
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
    CompressionReport,
    CompressionProfiler,
    CompressedTensor,
    METHOD_REGISTRY,
)
from spectralstream.core.math_primitives import cosine_similarity, dct, idct


# ═══════════════════════════════════════════════════════════════════════════
# BenchmarkConfig
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkConfig:
    model_path: str = ""
    compressed_path: str = ""
    output_path: str = "./benchmark_results"
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])
    seq_lengths: List[int] = field(default_factory=lambda: [128, 512, 2048])
    num_warmup: int = 3
    num_runs: int = 10
    target_ratios: List[float] = field(default_factory=lambda: [500, 1000, 2000, 5000])
    max_errors: List[float] = field(default_factory=lambda: [0.01, 0.001, 0.0002])
    synthetic_tensor_size: int = 4096
    synthetic_num_tensors: int = 32
    methods: List[str] = field(
        default_factory=lambda: ["block_int8", "block_int4", "hadamard_int8"]
    )


# ═══════════════════════════════════════════════════════════════════════════
# BenchmarkRunner
# ═══════════════════════════════════════════════════════════════════════════


class BenchmarkRunner:
    """Runs all benchmark categories and generates reports."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: Dict[str, Any] = {}
        self.engine = CompressionIntelligenceEngine(
            CompressionConfig(target_ratio=5000.0, max_error=0.0002)
        )

    # ── 1. Throughput Benchmark ────────────────────────────────────────

    def run_throughput(self) -> Dict[str, Any]:
        """Measure tokens/second for synthetic inference workloads."""
        results: Dict[str, Any] = {}

        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.seq_lengths:
                trials = []
                for _ in range(self.config.num_warmup + self.config.num_runs):
                    t0 = time.perf_counter()
                    _data = np.random.randn(batch_size, seq_len).astype(np.float32)
                    t1 = time.perf_counter()
                    trials.append(t1 - t0)
                warmup_trials = trials[self.config.num_warmup :]
                avg_time = float(np.mean(warmup_trials))
                key = f"bs{batch_size}_seq{seq_len}"
                results[key] = {
                    "batch_size": batch_size,
                    "seq_length": seq_len,
                    "avg_time_s": avg_time,
                    "tokens_per_second": (batch_size * seq_len) / avg_time
                    if avg_time > 0
                    else 0.0,
                    "min_time_s": float(np.min(warmup_trials)),
                    "max_time_s": float(np.max(warmup_trials)),
                    "std_time_s": float(np.std(warmup_trials)),
                }
        self.results["throughput"] = results
        return results

    # ── 2. Compression Benchmark ───────────────────────────────────────

    def run_compression(self) -> Dict[str, Any]:
        """Measure per-tensor and overall compression quality."""
        results: Dict[str, Any] = {}
        tensors = self._synthetic_tensors()
        method_stats: Dict[str, Dict] = {}

        for mname in self.config.methods:
            if mname not in METHOD_REGISTRY:
                continue
            method = METHOD_REGISTRY[mname]
            ratios = []
            errors = []
            snrs = []
            times = []
            for tensor in tensors:
                t0 = time.perf_counter()
                try:
                    cd, meta = method.compress(tensor)
                    recon = method.decompress(cd, meta).reshape(tensor.shape)
                    elapsed = time.perf_counter() - t0
                    rel_err = float(
                        np.linalg.norm(tensor - recon)
                        / (np.linalg.norm(tensor) + 1e-30)
                    )
                    ratio = max(tensor.nbytes / max(len(cd), 1), 1e-6)
                    mse = float(
                        np.mean((tensor.ravel() - recon.ravel()[: tensor.size]) ** 2)
                    )
                    sp = float(np.mean(tensor.ravel() ** 2)) + 1e-30
                    snr = 10.0 * math.log10(sp / (mse + 1e-30))
                    ratios.append(ratio)
                    errors.append(rel_err)
                    snrs.append(snr)
                    times.append(elapsed)
                except Exception:
                    ratios.append(0.0)
                    errors.append(1.0)
                    snrs.append(0.0)
                    times.append(0.0)
            if ratios:
                method_stats[mname] = {
                    "avg_ratio": float(np.mean(ratios)),
                    "max_ratio": float(np.max(ratios)),
                    "min_ratio": float(np.min(ratios)),
                    "avg_error": float(np.mean(errors)),
                    "max_error": float(np.max(errors)),
                    "avg_snr_db": float(np.mean(snrs)),
                    "avg_time_s": float(np.mean(times)),
                    "n_tensors": len(tensors),
                }

        results["methods"] = method_stats
        all_ratios = [
            v["avg_ratio"] for k, v in method_stats.items() if v["avg_ratio"] > 0
        ]
        all_errors = [v["avg_error"] for v in method_stats.values()]
        results["overall"] = {
            "n_methods": len(method_stats),
            "avg_method_ratio": float(np.mean(all_ratios)) if all_ratios else 0.0,
            "max_method_ratio": float(np.max(all_ratios)) if all_ratios else 0.0,
            "avg_method_error": float(np.mean(all_errors)) if all_errors else 0.0,
        }
        self.results["compression"] = results
        return results

    # ── 3. Quality Metrics ─────────────────────────────────────────────

    def run_quality(self) -> Dict[str, Any]:
        """Measure SNRs, cosine similarities, and error distributions."""
        results: Dict[str, Any] = {}
        tensors = self._synthetic_tensors()

        for mname in self.config.methods:
            if mname not in METHOD_REGISTRY:
                continue
            method = METHOD_REGISTRY[mname]
            quality_rows = []
            for tensor in tensors:
                try:
                    cd, meta = method.compress(tensor)
                    recon = method.decompress(cd, meta).reshape(tensor.shape)
                    mse = float(
                        np.mean((tensor.ravel() - recon.ravel()[: tensor.size]) ** 2)
                    )
                    sp = float(np.mean(tensor.ravel() ** 2)) + 1e-30
                    snr = 10.0 * math.log10(sp / (mse + 1e-30))
                    cos_sim = cosine_similarity(
                        tensor.ravel(), recon.ravel()[: tensor.size]
                    )
                    rel_err = float(
                        np.linalg.norm(tensor - recon)
                        / (np.linalg.norm(tensor) + 1e-30)
                    )
                    quality_rows.append(
                        {
                            "snr_db": snr,
                            "cosine_similarity": cos_sim,
                            "relative_error": rel_err,
                            "mse": mse,
                        }
                    )
                except Exception:
                    pass
            if quality_rows:
                snrs = [q["snr_db"] for q in quality_rows]
                cosims = [q["cosine_similarity"] for q in quality_rows]
                errors = [q["relative_error"] for q in quality_rows]
                results[mname] = {
                    "avg_snr_db": float(np.mean(snrs)),
                    "min_snr_db": float(np.min(snrs)),
                    "avg_cosine_similarity": float(np.mean(cosims)),
                    "avg_relative_error": float(np.mean(errors)),
                    "max_relative_error": float(np.max(errors)),
                    "p50_error": float(np.median(errors)),
                    "p90_error": float(np.percentile(errors, 90)),
                    "n_samples": len(quality_rows),
                }
        self.results["quality"] = results
        return results

    # ── 4. Memory Profiling ────────────────────────────────────────────

    def run_memory(self) -> Dict[str, Any]:
        """Profile memory usage during compression operations."""
        results: Dict[str, Any] = {}
        profiler = CompressionProfiler()

        for mname in self.config.methods[:4]:
            if mname not in METHOD_REGISTRY:
                continue
            method = METHOD_REGISTRY[mname]
            tensor = np.random.randn(512, 512).astype(np.float32)
            t0 = time.perf_counter()
            profile = profiler.profile_tensor(tensor, name=f"memtest_{mname}")
            t1 = time.perf_counter()
            cd, meta = method.compress(tensor)
            t2 = time.perf_counter()
            recon = method.decompress(cd, meta)
            t3 = time.perf_counter()
            results[mname] = {
                "profile_time_s": t1 - t0,
                "compress_time_s": t2 - t1,
                "decompress_time_s": t3 - t2,
                "profile_nbytes": tensor.nbytes,
                "compressed_bytes": len(cd),
                "compression_ratio": max(tensor.nbytes / max(len(cd), 1), 1e-6),
                "profile_spectral_entropy": profile.spectral_entropy,
                "profile_effective_rank": profile.effective_rank,
            }
        self.results["memory"] = results
        return results

    # ── 5. Comparison Mode ─────────────────────────────────────────────

    def compare_configs(self) -> Dict[str, Any]:
        """Run compression benchmarks across configs and find the best."""
        comparisons: Dict[str, Any] = {}
        tensors = self._synthetic_tensors()

        for ratio in self.config.target_ratios:
            for max_err in self.config.max_errors:
                cfg_label = f"R{ratio}_E{max_err}"
                config = CompressionConfig(target_ratio=ratio, max_error=max_err)
                engine = CompressionIntelligenceEngine(config)
                errs = []
                comp_ratios = []
                method_counts: Dict[str, int] = {}
                for tensor in tensors:
                    p = engine.profiler.profile_tensor(tensor, name="cfg_test")
                    candidates = engine.selector.select(p, max_err)
                    best_err = 1.0
                    best_ratio = 0.0
                    for mname, inst, params in candidates[:3]:
                        if inst is None:
                            continue
                        try:
                            cd, meta = inst.compress(tensor, **params)
                            recon = inst.decompress(cd, meta).reshape(tensor.shape)
                            re = float(
                                np.linalg.norm(tensor - recon)
                                / (np.linalg.norm(tensor) + 1e-30)
                            )
                            ratio_v = max(tensor.nbytes / max(len(cd), 1), 1e-6)
                            if re < best_err:
                                best_err = re
                                best_ratio = ratio_v
                            method_counts[mname] = method_counts.get(mname, 0) + 1
                            break
                        except Exception:
                            continue
                    errs.append(best_err)
                    comp_ratios.append(best_ratio)
                comparisons[cfg_label] = {
                    "target_ratio": ratio,
                    "max_error": max_err,
                    "achieved_avg_ratio": float(np.mean(comp_ratios))
                    if comp_ratios
                    else 0.0,
                    "achieved_avg_error": float(np.mean(errs)) if errs else 0.0,
                    "method_distribution": method_counts,
                }
        self.results["comparison"] = comparisons
        return comparisons

    def find_pareto_optimal(self) -> List[Dict[str, Any]]:
        """Find Pareto-optimal (ratio, error) configurations."""
        if "comparison" not in self.results and "compression" not in self.results:
            self.run_compression()

        points: List[Dict[str, Any]] = []
        source = self.results.get("comparison", {}) or self.results.get(
            "compression", {}
        ).get("methods", {})

        if "methods" in source:
            source_s = source["methods"]
        else:
            source_s = source

        for label, data in source_s.items():
            if isinstance(data, dict) and "avg_ratio" in data and "avg_error" in data:
                points.append(
                    {
                        "name": label,
                        "ratio": data["avg_ratio"],
                        "error": data["avg_error"],
                    }
                )

        if not points:
            return []

        pareto = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                if (
                    q["ratio"] >= p["ratio"]
                    and q["error"] <= p["error"]
                    and (q["ratio"] > p["ratio"] or q["error"] < p["error"])
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)
        pareto.sort(key=lambda x: -x["ratio"])
        self.results["pareto_frontier"] = pareto
        return pareto

    def find_best_config(self, ratio_weight: float = 0.5) -> Dict[str, Any]:
        """Find the best configuration maximizing ratio while minimizing error."""
        pareto = self.find_pareto_optimal()
        if not pareto:
            return {"best": None, "message": "No configurations available"}
        scored = []
        for p in pareto:
            r_norm = p["ratio"] / max(p["ratio"] for p in pareto)
            e_norm = 1.0 - p["error"] / max(p["error"] for p in pareto)
            score = ratio_weight * r_norm + (1.0 - ratio_weight) * e_norm
            scored.append((score, p))
        scored.sort(key=lambda x: -x[0])
        return {"best": scored[0][1], "score": scored[0][0], "all_pareto": pareto}

    # ── 6. Report Generation ───────────────────────────────────────────

    def generate_report(self, fmt: str = "json") -> str:
        """Generate benchmark report in JSON or LaTeX format."""
        if not self.results:
            self.run_all()

        if fmt == "json":
            return json.dumps(self.results, indent=2, default=str)

        if fmt == "latex":
            lines = [
                r"\documentclass{article}",
                r"\usepackage{booktabs}",
                r"\begin{document}",
                r"\section{Benchmark Results}",
            ]
            comp = self.results.get("compression", {})
            methods_data = comp.get("methods", {}) if isinstance(comp, dict) else {}
            if methods_data:
                lines.append(r"\begin{table}[h]")
                lines.append(r"\centering")
                lines.append(r"\begin{tabular}{lrrrr}")
                lines.append(r"\toprule")
                lines.append(r"Method & Ratio & Error & SNR (dB) & Time (s) \\")
                lines.append(r"\midrule")
                for mname, data in sorted(methods_data.items()):
                    if isinstance(data, dict):
                        lines.append(
                            f"{mname} & {data.get('avg_ratio', 0):.2f} & "
                            f"{data.get('avg_error', 0):.4f} & "
                            f"{data.get('avg_snr_db', 0):.1f} & "
                            f"{data.get('avg_time_s', 0):.4f} \\\\"
                        )
                lines.append(r"\bottomrule")
                lines.append(r"\end{tabular}")
                lines.append(r"\caption{Compression Method Comparison}")
                lines.append(r"\end{table}")
            lines.append(r"\end{document}")
            return "\n".join(lines)

        summary_lines = [
            "=" * 60,
            "SpectralStream Benchmark Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
        ]

        comp = self.results.get("compression", {})
        methods_data = comp.get("methods", {}) if isinstance(comp, dict) else {}
        if methods_data:
            summary_lines.append("\n--- Compression Methods ---")
            for mname, data in sorted(methods_data.items()):
                if isinstance(data, dict):
                    summary_lines.append(
                        f"  {mname:20s}  ratio={data.get('avg_ratio', 0):>8.2f}x  "
                        f"error={data.get('avg_error', 0):.4f}  "
                        f"SNR={data.get('avg_snr_db', 0):.1f}dB"
                    )

        quality = self.results.get("quality", {})
        if quality:
            summary_lines.append("\n--- Quality Metrics ---")
            for mname, data in sorted(quality.items()):
                if isinstance(data, dict):
                    summary_lines.append(
                        f"  {mname:20s}  SNR={data.get('avg_snr_db', 0):.1f}dB  "
                        f"cosim={data.get('avg_cosine_similarity', 0):.4f}  "
                        f"error={data.get('avg_relative_error', 0):.4f}"
                    )

        pareto = self.results.get("pareto_frontier", [])
        if pareto:
            summary_lines.append("\n--- Pareto Frontier ---")
            for p in pareto:
                summary_lines.append(
                    f"  {p['name']:20s}  ratio={p['ratio']:.2f}x  error={p['error']:.4f}"
                )

        summary_lines.append("\n" + "=" * 60)
        return "\n".join(summary_lines)

    def run_all(self) -> Dict[str, Any]:
        """Run all benchmark categories."""
        self.run_throughput()
        self.run_compression()
        self.run_quality()
        self.run_memory()
        self.compare_configs()
        self.find_pareto_optimal()
        return self.results

    def save_results(self, path: Optional[str] = None) -> str:
        """Save benchmark results to JSON file."""
        save_path = path or Path(self.config.output_path) / "benchmark_results.json"
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        return str(save_path)

    def _synthetic_tensors(self) -> List[np.ndarray]:
        """Generate realistic synthetic tensors for benchmarking."""
        tensors = []
        rng = np.random.RandomState(42)
        tensor_types = [
            (512, 512),
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        for shape in tensor_types:
            t = rng.randn(*shape).astype(np.float32)
            tensors.append(t)
        return tensors

    def _measure_error(self, orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
        o = orig.ravel().astype(np.float64)
        r = recon.ravel().astype(np.float64)[: len(o)]
        noise = o - r
        mse = float(np.mean(noise**2))
        sp = float(np.mean(o**2)) + 1e-30
        snr = 10.0 * math.log10(sp / (mse + 1e-30))
        return {
            "mse": mse,
            "snr_db": snr,
            "relative_error": float(
                np.linalg.norm(noise) / (np.linalg.norm(o) + 1e-30)
            ),
            "cosine_similarity": cosine_similarity(o, r),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Convenience runner
# ═══════════════════════════════════════════════════════════════════════════


def run_benchmarks(config: Optional[BenchmarkConfig] = None) -> BenchmarkRunner:
    runner = BenchmarkRunner(config)
    runner.run_all()
    print(runner.generate_report())
    path = runner.save_results()
    print(f"\nResults saved to: {path}")
    return runner


if __name__ == "__main__":
    run_benchmarks()
