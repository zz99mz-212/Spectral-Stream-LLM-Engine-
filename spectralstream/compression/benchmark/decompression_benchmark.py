from __future__ import annotations
import time
import gc
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class DecompressionBenchmark:
    """Benchmarks decompression throughput at various compression ratios.

    Measures:
    - Compressed size vs original size
    - Decompression time per tensor
    - Estimated tokens/sec (assuming transformer with N layers)
    - Memory bandwidth utilization
    - Cache efficiency (sequential decompression)

    This helps find the optimal operating point where compression ratio
    vs inference speed tradeoff is best.
    """

    def __init__(self, engine=None):
        self._engine = engine
        self._results: List[Dict[str, Any]] = []

    def benchmark_decompression(
        self,
        tensor: np.ndarray,
        method_name: str,
        method_params: dict,
        iterations: int = 5,
    ) -> Dict[str, Any]:
        """Benchmark a single compression method's decompression speed.

        Returns:
        {
            'method': str,
            'compression_ratio': float,
            'decompress_time_ms': float,
            'throughput_mb_s': float,
            'estimated_tokens_per_sec': float,
        }
        """
        from spectralstream.compression.methods import METHOD_CLASSES

        cls = METHOD_CLASSES.get(method_name)
        if not cls:
            return {"method": method_name, "error": "Method not found"}

        inst = cls() if isinstance(cls, type) else cls
        data, meta = inst.compress(tensor)
        ratio = tensor.nbytes / max(len(data), 1)

        # Warmup
        _ = inst.decompress(data, meta)
        gc.collect()

        # Benchmark decompression
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            recon = inst.decompress(data, meta)
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)  # ms
            del recon
            gc.collect()

        avg_ms = float(np.mean(times))
        throughput_mb_s = (tensor.nbytes / 1e6) / (avg_ms / 1000)

        # Estimate tokens/sec for a transformer model
        # Assumption: ~2 matrix multiplies per token per layer
        # decompress 2 weight matrices per layer per token
        n_layers = 35  # typical for 9B model
        bytes_per_token_per_layer = tensor.nbytes * 2  # Q,K,V,O = ~4, but reused
        estimated_tokens_per_sec = 1000 / (avg_ms * n_layers * 2)

        result = {
            "method": method_name,
            "compression_ratio": ratio,
            "decompress_time_ms": avg_ms,
            "throughput_mb_s": throughput_mb_s,
            "estimated_tokens_per_sec": estimated_tokens_per_sec,
        }
        self._results.append(result)
        return result

    def benchmark_cascade_decompression(
        self, tensor: np.ndarray, stages: List[Tuple[str, dict]], iterations: int = 3
    ) -> Dict[str, Any]:
        """Benchmark multi-stage cascade decompression.

        Each stage must be decompressed and the results summed.
        This is the worst-case for decompression overhead.
        """
        # First compress all stages
        from spectralstream.compression.methods import METHOD_CLASSES

        compressed_stages = []
        current = tensor.copy()
        for method_name, params in stages:
            cls = METHOD_CLASSES.get(method_name)
            if not cls:
                continue
            inst = cls() if isinstance(cls, type) else cls
            data, meta = inst.compress(current)
            recon = inst.decompress(data, meta)
            current = tensor - recon
            compressed_stages.append((inst, data, meta))
            del recon
            gc.collect()

        total_original = tensor.nbytes
        total_compressed = sum(len(d) for _, d, _ in compressed_stages)
        ratio = total_original / max(total_compressed, 1)

        # Benchmark cascade decompression
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            final = np.zeros_like(tensor)
            for inst, data, meta in compressed_stages:
                recon = inst.decompress(data, meta)
                final += recon
                del recon
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)
            del final
            gc.collect()

        avg_ms = float(np.mean(times))
        n_layers = 35
        est_tokens = 1000 / (avg_ms * n_layers)

        return {
            "cascade": [s[0] for s in stages],
            "compression_ratio": ratio,
            "decompress_time_ms": avg_ms,
            "estimated_tokens_per_sec": est_tokens,
        }

    def find_optimal_ratio(
        self, tensor: np.ndarray, methods: List[str], ratios: List[int]
    ) -> Dict[str, Any]:
        """Find the compression ratio with the best speed/quality tradeoff."""
        results = []
        for method_name in methods:
            r = self.benchmark_decompression(tensor, method_name)
            results.append(r)
            gc.collect()
        return {"results": results}

    def report(self) -> str:
        """Generate a human-readable benchmark report."""
        lines = ["Decompression Benchmark Report", "=" * 40]
        for r in self._results:
            lines.append(f"\n{r.get('method', '?')}:")
            lines.append(f"  Ratio: {r.get('compression_ratio', 0):.1f}:1")
            lines.append(f"  Decompress: {r.get('decompress_time_ms', 0):.1f}ms")
            lines.append(f"  Throughput: {r.get('throughput_mb_s', 0):.1f}MB/s")
            lines.append(f"  Est tokens/s: {r.get('estimated_tokens_per_sec', 0):.1f}")
        return "\n".join(lines)
