#!/usr/bin/env python3
"""
Compress a model using the full intelligence engine — model-agnostic.

Reads a model in safetensors format, profiles each tensor, selects optimal
compression methods via :class:`CompressionIntelligenceEngine`, and writes
the compressed output to a ``.ssf`` file.  Reports per-tensor and aggregate
compression statistics.

Usage
-----
    python scripts/compress_gemma4.py --model path/to/model.safetensors \
        --output out.ssf --target-ratio 5000 --max-error 0.0002
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import psutil

    def mem() -> float:
        """Return current RSS memory usage of this process in MiB.

        Returns
        -------
        float
            Resident set size in mebibytes, or 0 if ``psutil`` is not
            available.
        """
        return psutil.Process().memory_info().rss / 1024 / 1024
except ImportError:

    def mem() -> float:
        """Return current RSS memory usage of this process in MiB.

        Fallback implementation used when ``psutil`` is not installed.

        Returns
        -------
        float
            Always returns 0.
        """
        return 0


from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
)
from spectralstream.format.writer import SSFWriter
from spectralstream.compression.engine._io import _SafetensorsIO


def main() -> None:
    """Entry point: parse CLI arguments and run the compression pipeline.

    Orchestrates the full compression workflow — tensor scanning, profiling,
    method selection, compression with validation, and SSF writing.  Prints
    progress every 25 tensors and writes a JSON report alongside the output
    file.

    Raises
    ------
    SystemExit
        If the model file does not exist or a fatal error occurs during
        compression.
    """
    parser = argparse.ArgumentParser(description="Compress safetensors model to SSF")
    parser.add_argument("--model", default="models/gemma-4-E2B/model.safetensors")
    parser.add_argument("--output", default="models/gemma-4-E2B/gemma4_compressed.ssf")
    parser.add_argument("--target-ratio", type=float, default=5000.0)
    parser.add_argument("--max-error", type=float, default=0.0002)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    mp = args.model
    op = args.output
    rp = op.replace(".ssf", "_report.json")

    if not os.path.exists(mp):
        logger.error("Model file not found: %s", mp)
        sys.exit(1)

    logger.info(f"init mem={mem():.0f}MB")
    config = CompressionConfig(
        target_ratio=args.target_ratio,
        max_error=args.max_error,
        num_workers=args.workers,
        streaming=True,
    )
    engine = CompressionIntelligenceEngine(config)
    logger.info(f"methods={len(engine._methods)} mem={mem():.0f}MB")

    io = _SafetensorsIO()
    ti = io.scan(mp)
    total = len(ti)
    logger.info(f"tensors={total}")

    n_ok, n_fail = 0, 0
    orig_b, comp_b = 0, 0
    sum_err, max_err = 0.0, 0.0
    mdist = {}
    t0 = time.perf_counter()

    try:
        with SSFWriter(op, metadata={"model": os.path.basename(mp)}) as w:
            for idx, (name, (shape, dt, off, nb)) in enumerate(ti.items()):
                try:
                    tensor = io.read(mp, shape, dt, off, nb)
                    profile = engine.profiler.profile_tensor(tensor, name=name)

                    methods = engine._select_methods(
                        profile, args.max_error, args.target_ratio
                    )

                    ct = engine.compress_tensor_with_validation(
                        tensor, profile, methods, args.max_error
                    )
                    compressed_arr = np.frombuffer(ct.data, dtype=np.uint8)
                    w.add_tensor(
                        name,
                        compressed_arr,
                        method=0,
                        params={
                            "original_shape": list(tensor.shape),
                            "original_dtype": str(tensor.dtype),
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

                    n_ok += 1
                    orig_b += tensor.nbytes
                    comp_b += len(ct.data)
                    sum_err += ct.relative_error
                    max_err = max(max_err, ct.relative_error)
                    mdist[ct.method] = mdist.get(ct.method, 0) + 1

                    del tensor, profile, ct
                    gc.collect()
                except Exception as ex:
                    logger.error(f"FAIL {name}: {ex}")
                    n_fail += 1

                if (idx + 1) % 25 == 0:
                    cr = orig_b / max(comp_b, 1)
                    ae = sum_err / max(n_ok, 1) * 100
                    logger.info(
                        f"  [{idx + 1}/{total}] ratio={cr:.2f}x err={ae:.4f}% mem={mem():.0f}MB top_methods={dict(sorted(mdist.items(), key=lambda x: -x[1])[:3])}"
                    )
    except Exception:
        logger.exception("Fatal error during compression")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    ratio = orig_b / max(comp_b, 1)
    avg_e = sum_err / max(n_ok, 1)

    logger.info(
        f"DONE {elapsed:.0f}s orig={orig_b / 1e9:.2f}GB comp={comp_b / 1e9:.2f}GB ratio={ratio:.2f}x err={avg_e * 100:.4f}% max_err={max_err * 100:.4f}% methods={mdist}"
    )
    for label, met in [
        ("5000:1", ratio >= 5000),
        ("1200:1", ratio >= 1200),
        ("500:1", ratio >= 500),
        ("err<1%", avg_e < 0.01),
        ("err<0.6%", avg_e < 0.006),
        ("max_err<1%", max_err < 0.01),
    ]:
        print(f"  {'OK' if met else '--'} {label}")
    with open(rp, "w") as f:
        json.dump(
            {
                "ratio": ratio,
                "avg_err_pct": avg_e * 100,
                "max_err_pct": max_err * 100,
                "time_s": elapsed,
                "methods": mdist,
                "n_ok": n_ok,
                "n_fail": n_fail,
            },
            f,
        )


if __name__ == "__main__":
    main()
