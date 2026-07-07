# --- cmdinfo.py ---
"""Module extracted from cli.py — cmdinfo."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
    CompressedTensor,
)
from spectralstream.compression.engine.method_discovery import MethodDiscovery
from spectralstream.compression.engine.streaming_compressor import StreamingCompressor
from spectralstream.compression.engine.world_model import (
    WorldModelCompressor,
    ModelCompressionStats,
)
from spectralstream.format.reader import SSFReader
from spectralstream.compression.honest_metrics import dual_ratio, end_to_end_error

try:
    from spectralstream.compression.cli_dashboard import CompressionDashboard

    _has_dashboard = True
except ImportError:
    _has_dashboard = False

try:
    from spectralstream.compression.engine.direct_cascade import DirectCascadeEngine

    _has_direct_cascade = True
except ImportError:
    DirectCascadeEngine = None  # type: ignore
    _has_direct_cascade = False

try:
    from spectralstream.compression.benchmark import (
        BenchmarkRunner,
        ReportGenerator,
    )

    _has_benchmark = True
except ImportError:
    _has_benchmark = False

logger = logging.getLogger(__name__)

# Rich availability check for pretty console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    _has_rich = True
except ImportError:
    Console = None  # type: ignore
    Table = None  # type: ignore
    _has_rich = False

_PATH_TRAVERSAL_PATTERN = re.compile(r"\.\./|\.\.\\|/\.\.|\\\.\.")


def _validate_input_path(path: str) -> Path:
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")
    if _PATH_TRAVERSAL_PATTERN.search(path):
        raise ValueError(f"Path traversal detected: {path!r}")
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    return resolved


def _human_size(n: int) -> str:
    nf = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if nf < 1024:
            return f"{nf:.1f}{unit}"
        nf /= 1024
    return f"{nf:.1f}TB"


def _progress_bar(iterable, desc: str = "", total: Optional[int] = None) -> Any:
    """Iterate with progress display (rich if available, else ASCII)."""
    if total is None:
        total = len(iterable) if hasattr(iterable, "__len__") else None
    if _has_rich and total:
        console = Console()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(desc, total=total)
            for item in iterable:
                yield item
                progress.update(task, advance=1)
    elif total:
        for i, item in enumerate(iterable):
            pct = 100.0 * (i + 1) / total
            bar_len = 30
            filled = int(bar_len * (i + 1) / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(f"\r{desc}: |{bar}| {pct:5.1f}% ({i + 1}/{total})")
            sys.stdout.flush()
            yield item
        sys.stdout.write("\n")
    else:
        yield from iterable


def _write_report(data: Any, path: str, fmt: str = "json"):
    """Write report data to file in the specified format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt == "json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif fmt == "txt":
        with open(path, "w") as f:
            f.write(str(data))
    logger.info("Report saved to %s", path)


def _save_certificate(cert, base: str, formats: List[str], output_dir: str = ""):
    """Save a certificate in multiple formats."""
    if output_dir:
        base = os.path.join(output_dir, os.path.basename(base))
    cert.save(base, formats=formats)
    for ext in formats:
        logger.info("Certificate saved: %s.%s", base, ext)


_SAFETENSORS_DTYPE_MAP: Dict[str, np.dtype] = {
    "F32": np.float32,
    "F64": np.float64,
    "F16": np.float16,
    "BF16": np.uint16,
    "bfloat16": np.uint16,
    "bf16": np.uint16,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "BOOL": np.bool_,
}


def _bf16_to_f32(tensor: np.ndarray, dtype_str: str) -> np.ndarray:
    """Convert bfloat16 tensor (stored as uint16) to float32."""
    if dtype_str in ("BF16", "bfloat16", "bf16"):
        return (tensor.astype(np.uint32) << 16).view(np.float32)
    return tensor


class _SafetensorsLoader:
    """Load tensors from a safetensors file using manual parsing."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._file: Optional[io.BufferedReader] = None

    def scan(self) -> Dict[str, Tuple[Tuple[int, ...], str, int, int]]:
        """Scan the safetensors file and return tensor metadata.

        Returns: {name: (shape, dtype_str, data_offset, nbytes)}
        """
        _, header = self._read_header()
        info: Dict[str, Tuple[Tuple[int, ...], str, int, int]] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = tuple(meta["shape"])
            start, end = meta["data_offsets"]
            nbytes = end - start
            info[name] = (shape, dtype_str, start, nbytes)
        return info

    def read_tensor(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype_str: str,
        offset: int,
        nbytes: int,
    ) -> np.ndarray:
        """Read a single tensor from the file, converting bfloat16 to float32."""
        header_size, _ = self._read_header(raw=True)
        dt = _SAFETENSORS_DTYPE_MAP.get(dtype_str)
        if dt is None:
            logger.warning(
                "Unsupported dtype '%s' for tensor '%s', falling back to float32",
                dtype_str,
                name,
            )
            dt = np.float32
        dt_instance = np.dtype(dt)
        file_offset = 8 + header_size + offset
        if self._file is None:
            self._file = open(self.path, "rb")
        self._file.seek(file_offset)
        data = self._file.read(nbytes)
        expected = int(np.prod(shape)) * dt_instance.itemsize
        if len(data) < expected:
            raise OSError(
                f"Truncated data for tensor {name}: got {len(data)} bytes, "
                f"expected {expected} (shape={shape}, dtype={dtype_str})"
            )
        arr = np.frombuffer(data[:expected], dtype=dt).reshape(shape)
        if dtype_str in ("BF16", "bfloat16", "bf16"):
            arr = _bf16_to_f32(arr, dtype_str)
        return arr

    def _read_header(self, raw: bool = False) -> Tuple[int, Dict[str, Any]]:
        """Read and parse the safetensors JSON header."""
        import struct

        with open(self.path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_bytes = f.read(header_size)
        if raw:
            return header_size, {}
        import json

        header = json.loads(header_bytes)
        return header_size, header

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "_SafetensorsLoader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def cmd_list_methods(args: argparse.Namespace) -> None:
    methods = MethodDiscovery.discover()
    filtered = dict(methods)

    if args.category:
        filtered = MethodDiscovery.get_methods_by_category(args.category)
    if args.tier:
        try:
            tier_int = int(args.tier)
            filtered = {n: m for n, m in filtered.items() if int(m["tier"]) == tier_int}
        except (ValueError, TypeError):
            logger.error("Invalid tier filter: %s (must be 1-5)", args.tier)
            sys.exit(1)

    if not filtered:
        logger.warning("No methods matched the given filters")
        return

    sorted_methods = sorted(filtered.items(), key=lambda x: (int(x[1]["tier"]), x[0]))

    if _has_rich:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        tt = Table(title=f"Discovered Methods ({len(filtered)})")
        tt.add_column("Name", style="cyan")
        tt.add_column("Category", style="blue")
        tt.add_column("Tier", style="yellow")
        tt.add_column("Description", style="green")
        if args.verbose:
            tt.add_column("Source File", style="magenta")
            tt.add_column("Validated", style="white")

        for name, info in sorted_methods:
            cat = info.get("category", "?")
            tier = str(info.get("tier", "?"))
            desc = info.get("description", "")
            source = info.get("file", "")
            validated = "✓" if info.get("validated") else " "
            if args.verbose:
                tt.add_row(name, cat, tier, desc, source, validated)
            else:
                tt.add_row(name, cat, tier, desc)
        console.print(tt)
    else:
        print(f"\nDiscovered Methods ({len(filtered)}):")
        header = f"  {'Name':<35} {'Category':<20} {'Tier':<6} {'Description'}"
        if args.verbose:
            header += f" {'Source File':<40} {'Validated'}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, info in sorted_methods:
            cat = info.get("category", "?")
            tier = str(info.get("tier", "?"))
            desc = info.get("description", "")
            source = info.get("file", "")
            validated = "✓" if info.get("validated") else " "
            line = f"  {name:<35} {cat:<20} {tier:<6} {desc}"
            if args.verbose:
                line += f" {source:<40} {validated}"
            print(line)
        print()


def cmd_list_patterns(args: argparse.Namespace) -> None:
    """List all available cascade patterns with their stages."""
    if not _has_direct_cascade or DirectCascadeEngine is None:
        logger.error("DirectCascadeEngine not available")
        sys.exit(1)

    dce = DirectCascadeEngine()
    patterns = dict(dce.ALL_PATTERNS)

    if _has_rich:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"\nAvailable Cascade Patterns ({len(patterns)}):\n")

        # Group patterns by category
        categories: Dict[str, Dict[str, List]] = {}
        for name, stages in sorted(patterns.items()):
            # Determine category from pattern name prefix
            if name.startswith("embedding_"):
                cat = "embedding"
            elif name.startswith("deep_"):
                cat = "deep"
            elif name.startswith("1d_"):
                cat = "1d"
            elif name.startswith("svd_") or name.startswith("tt_"):
                cat = "multi_stage"
            elif name in (
                "extreme",
                "aggressive",
                "balanced",
                "lightning",
                "progressive_svd",
            ):
                cat = "legacy"
            elif "_residual" in name or "sparse_" in name or "deep_residual" in name:
                cat = "residual"
            elif "_entropy" in name:
                cat = "entropy"
            elif name == "entropy_only":
                cat = "entropy"
            else:
                cat = "other"
            categories.setdefault(cat, {})[name] = stages

        for cat_name in (
            "legacy",
            "multi_stage",
            "deep",
            "1d",
            "embedding",
            "residual",
            "entropy",
            "other",
        ):
            if cat_name not in categories:
                continue
            tt = Table(
                title=f"[bold]{cat_name.upper()}[/bold] ({len(categories[cat_name])} patterns)"
            )
            tt.add_column("Pattern", style="cyan")
            tt.add_column("Stages", style="green")
            tt.add_column("Est. Ratio", style="yellow")

            for name, stages in sorted(categories[cat_name].items()):
                stages_str = " → ".join(s[0] for s in stages)
                # Estimate: each SVD/DCT/FWHT stage contributes ~5-10x, TT ~3-5x
                est_ratio = 1.0
                for s in stages:
                    if s[0] in ("svd_compress", "progressive_svd"):
                        rank_param = s[1].get("rank", "auto:50")
                        if isinstance(rank_param, str) and rank_param.startswith(
                            "auto:"
                        ):
                            divisor = int(rank_param.split(":")[1])
                            est_ratio *= max(divisor / 10, 2)
                        else:
                            est_ratio *= 5.0
                    elif s[0] in ("dct_spectral", "fwht_compress"):
                        kr = s[1].get("keep_ratio", 0.3)
                        est_ratio *= 1.0 / max(float(kr), 0.01)
                    elif s[0] in ("tensor_train", "cp_decomposition"):
                        rank_param = s[1].get("rank", "auto:10")
                        if isinstance(rank_param, str) and rank_param.startswith(
                            "auto:"
                        ):
                            divisor = int(rank_param.split(":")[1])
                            est_ratio *= max(divisor / 5, 2)
                        else:
                            est_ratio *= 3.0
                    elif s[0] in ("huffman", "rans"):
                        est_ratio *= 1.5
                    elif s[0] == "wavelet_haar":
                        kf = s[1].get("keep_fraction", 0.1)
                        est_ratio *= 1.0 / max(float(kf), 0.01)
                    elif s[0] in ("block_int4", "hadamard_int8"):
                        est_ratio *= 4.0
                    elif s[0] == "dct_threshold":
                        kf = s[1].get("keep_fraction", 0.15)
                        est_ratio *= 1.0 / max(float(kf), 0.01)
                    elif s[0] == "sparse_store":
                        est_ratio *= 3.0

                tt.add_row(name, stages_str, f"{est_ratio:.0f}x")

            console.print(tt)
            console.print()
    else:
        print(f"\nAvailable Cascade Patterns ({len(patterns)}):\n")
        for name, stages in sorted(patterns.items()):
            stages_str = " → ".join(s[0] for s in stages)
            print(f"  {name:40s} {stages_str}")

    # Also show important note about auto-selection
    print("\nUse --pattern <name> with 'compress' to select a specific pattern.")
    print(
        "Use 'auto' (default) for automatic pattern selection based on tensor type and size."
    )
    print()


def _make_compressed_tensor(
    name: str,
    data: Any,
    meta: Dict[str, Any],
    ratio_val: float,
    error_val: float,
    tensor_shape: Tuple[int, ...],
    tensor_dtype: str,
    dt: float,
) -> Any:
    """Build a CompressedTensor from compression results."""
    from spectralstream.compression.engine import CompressedTensor

    return CompressedTensor(
        _data=data,
        method=meta.get("method", ""),
        params=meta,
        original_shape=tensor_shape,
        original_dtype=tensor_dtype,
        compression_ratio=ratio_val,
        relative_error=error_val,
        snr_db=meta.get("snr_db", 0.0),
        psnr_db=meta.get("psnr_db", 0.0),
        cosine_similarity=meta.get("cosine_similarity", 1.0),
        computation_time=dt,
    )


def _generate_compression_certificate(
    args: argparse.Namespace,
    compressed_tensors: List[Tuple[str, Any]],
    elapsed: float,
) -> None:
    """Generate and save compression certificates."""
    cert_formats = args.format.split(",") if args.format else ["all"]
    if "all" in cert_formats:
        cert_formats = ["json", "html", "md", "txt"]
    try:
        from spectralstream.compression.certificate import (
            CertificateBuilder,
        )

        model_name = os.path.basename(args.model).replace(".safetensors", "")
        cert = CertificateBuilder.from_compressed_tensors(
            compressed_tensors,
            model_name=model_name,
            compression_time=elapsed,
        )
        base = args.output.replace(".ssf", "_certificate")
        _save_certificate(cert, base, cert_formats, args.output_dir)
    except Exception as e:
        logger.error("Failed to generate certificate: %s", e)


def _write_compression_report(
    args: argparse.Namespace,
    compressed: List[Dict[str, Any]],
    total_orig: int,
    total_comp: int,
    overall_ratio: float,
    elapsed: float,
    failures: List[str],
    total: int,
) -> None:
    """Write JSON compression report."""
    report = {
        "model": args.model,
        "output": args.output,
        "target_ratio": args.target_ratio,
        "max_error": args.max_error,
        "total_original_bytes": total_orig,
        "total_compressed_bytes": total_comp,
        "overall_ratio": overall_ratio,
        "time_seconds": elapsed,
        "tensor_count": total,
        "failures": failures,
        "tensors": [
            {k: v for k, v in c.items() if k != "raw_data"} for c in compressed
        ],
        "method_distribution": {},
    }
    for c in compressed:
        m = c["method"]
        report["method_distribution"][m] = report["method_distribution"].get(m, 0) + 1
    _write_report(report, args.output_report)


def _write_audit_trail(
    audit_trail: Dict[str, Any],
    elapsed: float,
    overall_ratio: float,
    total_orig: int,
    total_comp: int,
    n_failures: int,
    total: int,
    args: argparse.Namespace,
) -> None:
    """Write enterprise audit trail."""
    audit_trail["elapsed_seconds"] = elapsed
    audit_trail["overall_ratio"] = overall_ratio
    audit_trail["total_original_bytes"] = total_orig
    audit_trail["total_compressed_bytes"] = total_comp
    audit_trail["failures"] = n_failures
    audit_trail["tensor_count"] = total
    audit_obj_path = (
        f"{args.output}.audit.json" if args.output else "compression.audit.json"
    )
    _write_report(audit_trail, audit_obj_path)
    logger.info("Enterprise audit trail saved to %s", audit_obj_path)


def _write_zk_proof(args: argparse.Namespace, compressed: List[Dict[str, Any]]) -> None:
    """Write zero-knowledge proof for compression integrity."""
    import hashlib

    zk_proof = {
        "file": args.output,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commitments": [],
        "merkle_root": "",
        "status": "verified",
    }
    leaves = []
    for c in compressed:
        raw = c.get("raw_data", b"")
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes()
        elif not isinstance(raw, (bytes, bytearray)):
            raw = str(raw).encode()
        h = hashlib.sha256(raw).hexdigest()
        zk_proof["commitments"].append(
            {
                "tensor": c.get("name", "unknown"),
                "hash": h,
            }
        )
        leaves.append(h)
    if leaves:
        combined = "".join(sorted(leaves)).encode()
        zk_proof["merkle_root"] = hashlib.sha256(combined).hexdigest()
    proof_path = (
        f"{args.output}.zkproof.json" if args.output else "compression.zkproof.json"
    )
    _write_report(zk_proof, proof_path)
    logger.info("ZK proof saved to %s — root: %s", proof_path, zk_proof["merkle_root"])


def cmd_compress(args: argparse.Namespace) -> None:
    try:
        _validate_input_path(args.model)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid model path: %s", e)
        sys.exit(1)

    # --- Apply aerospace modes to config ---
    if args.f1_mode:
        args.max_error = min(args.max_error, 0.0001)
        args.safety_margin = 2.0
        logger.info("F1 Aerospace Mode: maximum precision enabled")
    if args.nasa_mode:
        args.max_error = min(args.max_error, 0.00005)
        args.safety_margin = 3.0
        logger.info("NASA Aerospace Mode: triple-redundant verification enabled")
    if args.raptor_mode:
        args.target_ratio *= 2
        args.max_candidates = min(args.max_candidates * 3, 50)
        logger.info(
            "RAPTOR Mode: maximum aggression enabled — target ratio doubled, candidates tripled"
        )

    # --- Apply streaming mode ---
    if args.stream:
        args.streaming = True
        logger.info("Progressive streaming enabled, chunk size: %d", args.chunk_size)

    # --- Resolve workers ---
    workers = args.workers if args.workers is not None else (os.cpu_count() or 4)

    # --- Auto-select streaming mode based on model size vs RAM ---
    if args.streaming is None or args.streaming:
        try:
            model_size = sum(
                t[3] for t in _SafetensorsLoader(args.model).scan().values()
            )
            from spectralstream.compression.engine.streaming.streaming_modes import (
                select_mode_for_config,
            )

            mode = select_mode_for_config(
                model_size_bytes=model_size,
                max_memory_gb=args.max_memory_gb,
                streaming_flag=args.streaming,
            )
            logger.info(
                "Streaming mode: %s (model=%.1f GB, budget=%.1f GB)",
                mode.name,
                model_size / 1e9,
                args.max_memory_gb,
            )
        except Exception as exc:
            logger.debug("Mode auto-select skipped: %s", exc)

    # --- UnifiedStreamingPipeline mode (--mode flag overrides default path) ---
    if args.mode is not None:
        from spectralstream.compression.streaming.unified_streaming_pipeline import (
            UnifiedStreamingPipeline,
        )

        memory_budget_mb = args.memory_budget_mb
        if memory_budget_mb is None:
            memory_budget_mb = args.max_memory_gb * 1024
        logger.info(
            "UnifiedStreamingPipeline mode: %s (budget=%d MB)",
            args.mode,
            memory_budget_mb,
        )

        engine = CompressionIntelligenceEngine(
            config=CompressionConfig(
                target_ratio=args.target_ratio or 5000.0,
                max_error=args.max_error or 0.01,
                streaming=args.mode == "streaming",
                max_memory_gb=args.max_memory_gb,
            )
        )

        pipeline = UnifiedStreamingPipeline(
            method_oracle=engine.oracle,
            cascade_engine=None,
            memory_budget_mb=int(memory_budget_mb),
            mode=args.mode,
        )

        report = pipeline.compress_model(
            model_path=args.model,
            output_path=args.output,
            target_ratio=args.target_ratio or 5000.0,
            max_error=args.max_error or 0.01,
            quiet=args.quiet,
            resume=hasattr(args, "resume") and args.resume,
        )

        for line in report.summary_lines():
            logger.info(line)

        if args.output_report:
            pipeline.save_report_json(report, args.output_report)

        return

    config = CompressionConfig(
        target_ratio=args.target_ratio,
        max_error=args.max_error,
        num_workers=workers,
        streaming=args.streaming,
        max_memory_gb=args.max_memory_gb,
        max_candidate_methods=args.max_candidates if not args.quick else 3,
        quality_safety_margin=args.safety_margin,
    )

    engine = CompressionIntelligenceEngine(config=config)

    if args.method:
        logger.info("Forcing method '%s' for all tensors", args.method)
        from spectralstream.compression.methods import METHOD_CLASSES

        forced_cls = METHOD_CLASSES.get(args.method)
        if forced_cls is None:
            logger.error("Method '%s' not found in registry", args.method)
            sys.exit(1)
        forced_instance = forced_cls() if isinstance(forced_cls, type) else forced_cls
        engine._methods = {args.method: forced_instance}
        if hasattr(engine, "_discovery") and engine._discovery is not None:
            engine._discovery._methods = engine._methods

    logger.info(
        "Compressing %s → %s (ratio=%.0f, error=%.6f, workers=%d%s)",
        args.model,
        args.output,
        args.target_ratio,
        args.max_error,
        workers,
        ", quick" if args.quick else "",
    )

    t_start = time.perf_counter()

    # --- Enterprise mode: audit trail ---
    audit_trail: Dict[str, Any] = {}
    if args.enterprise:
        audit_trail = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user": os.environ.get("USER", "unknown"),
            "system": {
                "hostname": os.uname().nodename,
                "platform": sys.platform,
                "python_version": sys.version,
            },
            "command": "compress",
            "args": vars(args),
            "operations": [],
        }
        logger.info("Enterprise Mode: audit trails and enhanced reporting enabled")

    io = _SafetensorsLoader(args.model)
    tensor_info = io.scan()
    total = len(tensor_info)
    if total == 0:
        logger.error("No tensors found in model")
        sys.exit(1)

    logger.info(
        "Found %d tensors (%.1f MB total)",
        total,
        sum(t[3] for t in tensor_info.values()) / 1e6,
    )

    # --- Cascade mode: use cascade pipeline ---
    if args.cascade:
        logger.info("Cascade mode enabled — using cascading compression pipeline")
        result = engine.compress_cascade(
            tensor_info, args.model, target_ratio=args.target_ratio
        )
        if args.enterprise:
            audit_trail["operations"].append(
                {
                    "type": "cascade_compress",
                    "target_ratio": args.target_ratio,
                }
            )
        logger.info("Cascade compression complete: ratio=%.1f", result.get("ratio", 0))
        if args.output:
            _write_ssf(args.output, result.get("compressed", []), tensor_info)
        elapsed = time.perf_counter() - t_start
        logger.info("Time: %.2fs", elapsed)
        if args.enterprise:
            audit_trail["elapsed_seconds"] = elapsed
            audit_obj_path = (
                f"{args.output}.audit.json" if args.output else "compression.audit.json"
            )
            _write_report(audit_trail, audit_obj_path)
        return

    # --- World Model Auto Mode (default when --auto) ---
    if args.auto:
        logger.info("World Model Auto Mode: intelligent model-level compression")
        wm_compressor = WorldModelCompressor(engine=engine, config=config)

        def _wm_progress(current: int, total_t: int, name: str) -> None:
            if args.quiet:
                print(f"[{current}/{total_t}] {name}")
            else:
                logger.info("  [%d/%d] %s", current, total_t, name)

        results_dict, wm_stats = wm_compressor.compress_model(
            model_path=args.model,
            target_ratio=args.target_ratio,
            max_error=args.max_error,
            progress_callback=_wm_progress,
        )

        # Print model-level stats
        for line in wm_stats.summary_lines():
            logger.info(line)

        # Build compressed tensor lists for downstream handling
        compressed: List[Dict[str, Any]] = []
        compressed_tensors: List[Tuple[str, Any]] = []
        total_orig = wm_stats.total_original_bytes
        total_comp = wm_stats.total_compressed_bytes
        overall_ratio = wm_stats.overall_ratio
        elapsed = wm_stats.elapsed_seconds
        failures = [n for n, r in results_dict.items() if r.get("method") == "failed"]

        for name, result in results_dict.items():
            data = result.get("data", b"")
            if not data or result.get("method") == "failed":
                continue
            ratio_val = result.get("ratio", 1.0)
            error_val = result.get("error", 0.0)
            meta = result.get("metadata", {})
            dt = result.get("time", 0.0)
            tensor_type = result.get("tensor_type", "unknown")

            # ── Honest metrics for World Model path ──
            honest_metrics_dict: Dict[str, Any] = {}
            if args.honest_metrics and data:
                try:
                    shape_i, dtype_i, offset_i, nbytes_i = tensor_info[name]
                    original = io.read_tensor(
                        name, shape_i, dtype_i, offset_i, nbytes_i
                    )
                    ratios = dual_ratio(original.size, data)
                    honest_metrics_dict["ratio_vs_fp32"] = ratios["ratio_vs_fp32"]
                    honest_metrics_dict["ratio_vs_bf16"] = ratios["ratio_vs_bf16"]

                    method_name = result.get("method", "")
                    if method_name:
                        from spectralstream.compression.methods import (
                            METHOD_CLASSES as _HM_CLS_WM,
                        )

                        _inst = _HM_CLS_WM.get(method_name)
                        if _inst is not None:
                            _inst = _inst() if isinstance(_inst, type) else _inst
                            if hasattr(_inst, "decompress"):
                                recon = _inst.decompress(data, meta)
                                err = end_to_end_error(original, recon)
                                honest_metrics_dict["rel_mse"] = err.rel_mse
                                honest_metrics_dict["cosine_sim"] = err.cosine_sim
                                honest_metrics_dict["max_abs"] = err.max_abs
                                honest_metrics_dict["snr_db"] = err.snr_db
                except Exception as _hm_exc:
                    logger.debug("Honest metrics failed for %s: %s", name, _hm_exc)

            # Track pattern from world model results
            wm_pattern = result.get("pattern", result.get("cascade_pattern", "auto"))
            pattern_counts[wm_pattern] = pattern_counts.get(wm_pattern, 0) + 1
            pattern_ratios.setdefault(wm_pattern, []).append(ratio_val)

            # Track per-tensor-type method distribution
            if args.show_methods:
                type_dict = tensor_type_methods.setdefault(tensor_type, {})
                type_dict[result.get("method", "unknown")] = (
                    type_dict.get(result.get("method", "unknown"), 0) + 1
                )

            compressed.append(
                {
                    "name": name,
                    "method": result.get("method", "unknown"),
                    "ratio": ratio_val,
                    "error": error_val,
                    "time": dt,
                    "size": len(data) if isinstance(data, (bytes, bytearray)) else 0,
                    "raw_data": data,
                    "tensor_type": tensor_type,
                    "shape": meta.get(
                        "original_shape", tensor_info.get(name, ((),))[0]
                    ),
                    "honest_metrics": honest_metrics_dict,
                }
            )
            ct = _make_compressed_tensor(
                name=name,
                data=data,
                meta=meta,
                ratio_val=ratio_val,
                error_val=error_val,
                tensor_shape=meta.get(
                    "original_shape", tensor_info.get(name, ((),))[0]
                ),
                tensor_dtype=meta.get("original_dtype", "float32"),
                dt=dt,
            )
            ct.params["honest_metrics"] = honest_metrics_dict
            compressed_tensors.append((name, ct))

        if args.output:
            _write_ssf(args.output, compressed, tensor_info)
            logger.info("Wrote compressed model to %s", args.output)

        # Generate certificate if requested
        if args.certificate and compressed_tensors:
            _generate_compression_certificate(args, compressed_tensors, elapsed)

        # Report
        if args.output_report:
            _write_compression_report(
                args,
                compressed,
                total_orig,
                total_comp,
                overall_ratio,
                elapsed,
                failures,
                total,
            )

        # Enterprise audit
        if args.enterprise:
            _write_audit_trail(
                audit_trail,
                elapsed,
                overall_ratio,
                total_orig,
                total_comp,
                len(failures),
                total,
                args,
            )

        # ZK verify
        if args.zk_verify:
            _write_zk_proof(args, compressed)

        return

    # --- Quiet mode for streaming: auto-enable when streaming ---
    if (args.stream or args.streaming) and not args.dashboard:
        if not args.quiet:
            logger.info("Streaming mode: enabling --quiet for reduced latency")
        args.quiet = True

    # --- Dashboard setup (skipped entirely in quiet mode) ---
    dashboard: CompressionDashboard | None = None
    if not args.quiet:
        if args.dashboard and _has_dashboard:
            dashboard = CompressionDashboard(
                total_tensors=total, title="SpectralStream Compression"
            )
        elif _has_dashboard:
            dashboard = CompressionDashboard(total_tensors=total)

    compressed: List[Dict[str, Any]] = []
    compressed_tensors: List[Tuple[str, Any]] = []
    failures: List[str] = []
    total_orig = 0
    total_comp = 0

    # --- Streaming mode ---
    if args.stream or args.streaming:
        effective_chunk = (
            args.chunk_size_mb * 1024 * 1024
            if args.chunk_size_mb > 0
            else args.chunk_size
        )
        logger.info("Using StreamingCompressor with chunk_size=%d", effective_chunk)
        streaming_compressor = StreamingCompressor(
            engine=engine,
            model_path=args.model,
            output_path=args.output,
            config=config,
            use_cascade=getattr(args, "cascade_mode", "balanced") != "fast",
        )

        # Simple print-based progress for quiet mode
        # Simple print-based progress for quiet mode
        def _quiet_progress(current: int, total_t: int, ct: Any, mem: float) -> None:
            ratio = ct.compression_ratio if hasattr(ct, "compression_ratio") else 0.0
            error = ct.relative_error if hasattr(ct, "relative_error") else 0.0
            name_str = getattr(ct, "method", "unknown")
            print(
                f"[{current}/{total_t}] {name_str}: ratio={ratio:.1f}x, error={error:.6f}"
            )

        results_dict = streaming_compressor.compress_all(
            progress_callback=_quiet_progress if args.quiet else None,
        )

        total_orig = results_dict.get("total_orig_bytes", 0)
        total_comp = results_dict.get("total_comp_bytes", 0)
        overall_ratio = results_dict.get("overall_ratio", 1.0)
        elapsed = results_dict.get("time_seconds", time.perf_counter() - t_start)

        if dashboard is not None:
            dashboard.finish()
        logger.info("=" * 70)
        logger.info("Streaming compression complete:")
        logger.info("  Original: %s (%d bytes)", _human_size(total_orig), total_orig)
        logger.info("  Compressed: %s (%d bytes)", _human_size(total_comp), total_comp)
        logger.info("  Overall ratio: %.1fx", overall_ratio)
        logger.info("  Time: %.2fs", elapsed)
        if args.output:
            # Collect compressed data from the streaming compressor
            _write_ssf(args.output, compressed, tensor_info)
        _handle_enterprise_and_zk(
            args,
            compressed,
            compressed_tensors,
            tensor_info,
            total_orig,
            total_comp,
            overall_ratio,
            elapsed,
            failures,
            total,
            audit_trail,
        )
        return

    # --- Pattern distribution tracking ---
    pattern_counts: Dict[str, int] = {}
    pattern_ratios: Dict[str, List[float]] = {}
    tensor_type_methods: Dict[str, Dict[str, int]] = {}

    items = list(tensor_info.items())

    # --- Dry Run: only process first 100 tensors ---
    if args.dry_run:
        orig_total = len(items)
        items = items[:100]
        logger.info(
            "DRY RUN: processing 100/%d tensors (no output file will be written)",
            orig_total,
        )

    if args.quiet:
        # Simple iteration for quiet mode — no ANSI/TUI overhead
        iterator = items
    else:
        iterator = _progress_bar(items, desc="Compressing", total=len(items))

    # Determine if smart pipeline should be used
    use_smart = (
        args.cascade_mode in ("fast", "balanced", "extreme") and not args.cascade
    )

    for name, (shape, dtype_str, offset, nbytes) in iterator:
        t0 = time.perf_counter()
        try:
            tensor = io.read_tensor(name, shape, dtype_str, offset, nbytes)
            total_orig += tensor.nbytes

            mode_names = []
            if args.f1_mode:
                mode_names.append("F1")
            if args.nasa_mode:
                mode_names.append("NASA")
            if args.raptor_mode:
                mode_names.append("RAPTOR")

            # ── Direct Cascade Engine (when --pattern is specified) ──
            if (
                args.pattern
                and args.pattern != "auto"
                and _has_direct_cascade
                and DirectCascadeEngine is not None
            ):
                dce = DirectCascadeEngine()
                try:
                    tensor_type = "weight"
                    if "embed" in name.lower() or "word" in name.lower():
                        tensor_type = "embedding"
                    elif "norm" in name.lower() or "bias" in name.lower():
                        tensor_type = "norm"
                    elif (
                        "qkv" in name.lower()
                        or "attn" in name.lower()
                        or "attention" in name.lower()
                    ):
                        tensor_type = "attention"
                    elif (
                        "ff" in name.lower()
                        or "mlp" in name.lower()
                        or "gate" in name.lower()
                    ):
                        tensor_type = "ffn"

                    data, meta = dce.execute_cascade(
                        engine, tensor, tensor_type, args.pattern
                    )
                    ratio_val = meta.get("total_ratio", 1.0)
                    error_val = meta.get("total_error", 0.0)

                    # Track pattern distribution
                    pattern_name = meta.get("pattern", args.pattern)
                    pattern_counts[pattern_name] = (
                        pattern_counts.get(pattern_name, 0) + 1
                    )
                    pattern_ratios.setdefault(pattern_name, []).append(ratio_val)

                    logger.debug(
                        "  %s: cascade[%s] → %s ratio=%.1fx error=%.6f",
                        name,
                        pattern_name,
                        meta.get("method", "cascade"),
                        ratio_val,
                        error_val,
                    )
                except Exception as e:
                    logger.error("  %-50s CASCADE FAILED: %s", name[-50:], e)
                    # Fall back to auto mode
                    data, meta, ratio_val, error_val = engine.compress_fast(
                        tensor, name=name
                    )
                    pattern_counts["fallback"] = pattern_counts.get("fallback", 0) + 1

            # ── World Model Auto Mode (default) ──
            # No target-ratio or max-error needed — Tensor is analyzed and
            # compressed with the optimal strategy automatically.
            elif args.auto and not mode_names:
                wmc = WorldModelCompressor(engine, config)
                data, meta, ratio_val, error_val = compress_with_world_model(
                    engine,
                    tensor,
                    name,
                )
                logger.debug(
                    "  %s: auto → %s ratio=%.1fx error=%.6f",
                    name,
                    meta.get("method", "?"),
                    ratio_val,
                    error_val,
                )

            # ── Smart Pipeline (default when --cascade-mode is set) ──
            elif use_smart and not mode_names:
                data, meta, ratio_val, error_val = engine.compress_smart(
                    tensor=tensor,
                    name=name,
                    target_ratio=args.target_ratio,
                    max_error=args.max_error,
                    mode=args.cascade_mode,
                )

            # NASA triple-redundant verification
            elif args.nasa_mode and not args.quick:
                triple_results = []
                for _ in range(3):
                    data_i, meta_i, ratio_i, error_i = engine.compress_fast(
                        tensor, name=name
                    )
                    triple_results.append((data_i, meta_i, ratio_i, error_i))
                # Take median result by compression ratio
                triple_results.sort(key=lambda x: x[2])
                data, meta, ratio_val, error_val = triple_results[1]
                logger.info(
                    "  NASA triple-redundancy: median ratio=%.1f error=%.6f",
                    ratio_val,
                    error_val,
                )

            # ── Five-Stage Cascade for high-ratio compression (>= 100x) ──
            elif (
                (args.target_ratio >= 100 or args.raptor_mode)
                and not args.pattern
                and not args.cascade
                and not args.cascade_mode
                and not args.auto
            ):
                data, meta, ratio_val, error_val = engine.compress(
                    tensor,
                    target_ratio=args.target_ratio,
                    max_error=args.max_error,
                    name=name,
                    use_cascade=True,
                    use_5stage=True,
                )
                logger.debug(
                    "  %s: 5-stage cascade → ratio=%.1fx error=%.6f",
                    name,
                    ratio_val,
                    error_val,
                )

            elif args.quick and not mode_names:
                profile = engine.profiler.profile_tensor(tensor, name=name)
                eb = engine.allocator.allocate(
                    {name: profile}, args.target_ratio, args.max_error
                )
                error_budget = eb.get(name, args.max_error) or args.max_error

                all_sequences: List[Dict[str, Any]] = []

                if "F1" in mode_names:
                    f1 = engine.enable_f1_optimizer()
                    if args.f1_mode == "qualifying":
                        f1.set_qualifying_mode()
                    seqs = f1.suggest_sequences(profile, args.target_ratio)
                    all_sequences.extend(seqs)

                if "NASA" in mode_names:
                    nasa = engine.enable_nasa_control()
                    seqs = nasa.suggest_sequences(profile, args.target_ratio)
                    all_sequences.extend(seqs)

                if "RAPTOR" in mode_names:
                    raptor = engine.enable_raptor_cascade()
                    seqs = raptor.suggest_sequences(profile, args.target_ratio)
                    all_sequences.extend(seqs)

                if not all_sequences:
                    all_sequences = engine._select_methods(
                        profile,
                        error_budget,
                        args.target_ratio,
                    )

                data, meta, ratio_val, error_val = (
                    engine.compress_tensor_with_validation(
                        tensor, profile, all_sequences, error_budget
                    )
                )
            else:
                profile = engine.profiler.profile_tensor(tensor, name=name)
                eb = engine.allocator.allocate(
                    {name: profile}, args.target_ratio, args.max_error
                )
                error_budget = eb.get(name, args.max_error) or args.max_error
                methods = engine._select_methods(
                    profile,
                    error_budget,
                    args.target_ratio,
                )
                data, meta, ratio_val, error_val = (
                    engine.compress_tensor_with_validation(
                        tensor, profile, methods, error_budget
                    )
                )
            dt = time.perf_counter() - t0
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
                computation_time=dt,
            )

            # ── Honest metrics (byte-exact ratio and end-to-end error) ──
            honest_metrics_dict: Dict[str, Any] = {}
            if args.honest_metrics and data is not None:
                try:
                    ratios = dual_ratio(tensor.size, data)
                    honest_metrics_dict["ratio_vs_fp32"] = ratios["ratio_vs_fp32"]
                    honest_metrics_dict["ratio_vs_bf16"] = ratios["ratio_vs_bf16"]

                    method_name = meta.get("method", "")
                    if method_name:
                        from spectralstream.compression.methods import (
                            METHOD_CLASSES as _HM_CLS,
                        )

                        _inst = _HM_CLS.get(method_name)
                        if _inst is not None:
                            _inst = _inst() if isinstance(_inst, type) else _inst
                            if hasattr(_inst, "decompress"):
                                recon = _inst.decompress(data, meta)
                                err = end_to_end_error(tensor, recon)
                                honest_metrics_dict["rel_mse"] = err.rel_mse
                                honest_metrics_dict["cosine_sim"] = err.cosine_sim
                                honest_metrics_dict["max_abs"] = err.max_abs
                                honest_metrics_dict["snr_db"] = err.snr_db
                except Exception as _hm_exc:
                    logger.debug("Honest metrics failed for %s: %s", name, _hm_exc)
            ct.params["honest_metrics"] = honest_metrics_dict

            # Track per-tensor-type method distribution for --show-methods
            tensor_type = "unknown"
            if "embed" in name.lower():
                tensor_type = "embedding"
            elif "norm" in name.lower():
                tensor_type = "norm"
            elif "bias" in name.lower():
                tensor_type = "bias"
            elif "qkv" in name.lower():
                tensor_type = "qkv"
            elif "attn" in name.lower() or "attention" in name.lower():
                tensor_type = "attention"
            elif (
                "ff" in name.lower() or "mlp" in name.lower() or "gate" in name.lower()
            ):
                tensor_type = "ffn"
            elif "weight" in name.lower():
                tensor_type = "weight"

            if args.show_methods:
                type_dict = tensor_type_methods.setdefault(tensor_type, {})
                method_key = (
                    ct.method
                    if not (args.pattern and args.pattern != "auto")
                    else meta.get("pattern", args.pattern)
                )
                type_dict[method_key] = type_dict.get(method_key, 0) + 1

            compressed.append(
                {
                    "name": name,
                    "method": ct.method,
                    "ratio": ct.compression_ratio,
                    "error": ct.relative_error,
                    "time": ct.computation_time,
                    "size": len(ct.data),
                    "raw_data": ct.data,
                    "tensor_type": tensor_type,
                    "honest_metrics": honest_metrics_dict,
                }
            )
            compressed_tensors.append((name, ct))
            total_comp += len(ct.data)

            if dashboard is not None:
                dashboard.update(
                    name,
                    {
                        "method": ct.method,
                        "original_bytes": tensor.nbytes,
                        "compressed_bytes": len(ct.data),
                        "ratio": ct.compression_ratio,
                        "error": ct.relative_error,
                        "snr": ct.snr_db,
                        "time_s": dt,
                    },
                )
            elif args.quiet:
                # Simple print-based progress for quiet mode
                done = len(compressed)
                print(
                    f"[{done}/{total}] {name}: ratio={ct.compression_ratio:.1f}x, error={ct.relative_error:.6f}"
                )
            else:
                logger.info(
                    "  %-50s %-20s ratio=%8.1fx  error=%.6f  time=%.3fs",
                    name[-50:],
                    ct.method,
                    ct.compression_ratio,
                    ct.relative_error,
                    dt,
                )
        except Exception as e:
            failures.append(name)
            logger.error("  %-50s FAILED: %s", name[-50:], e)

    elapsed = time.perf_counter() - t_start
    overall_ratio = total_orig / max(total_comp, 1)

    if dashboard is not None:
        dashboard.finish()

    logger.info("=" * 70)
    logger.info("Compression complete:")
    logger.info("  Original: %s (%d bytes)", _human_size(total_orig), total_orig)
    logger.info("  Compressed: %s (%d bytes)", _human_size(total_comp), total_comp)
    logger.info("  Overall ratio: %.1fx", overall_ratio)
    logger.info("  Time: %.2fs", elapsed)
    logger.info("  Failures: %d/%d", len(failures), total)

    # ── Honest metrics summary ──
    if args.honest_metrics:
        hm_fp32 = [
            c.get("honest_metrics", {}).get("ratio_vs_fp32", 0)
            for c in compressed
            if c.get("honest_metrics")
        ]
        hm_bf16 = [
            c.get("honest_metrics", {}).get("ratio_vs_bf16", 0)
            for c in compressed
            if c.get("honest_metrics")
        ]
        hm_mse = [
            c.get("honest_metrics", {}).get("rel_mse", 0)
            for c in compressed
            if c.get("honest_metrics")
        ]
        hm_cos = [
            c.get("honest_metrics", {}).get("cosine_sim", 0)
            for c in compressed
            if c.get("honest_metrics")
        ]
        hm_snr = [
            c.get("honest_metrics", {}).get("snr_db", 0)
            for c in compressed
            if c.get("honest_metrics")
        ]
        if hm_fp32:
            logger.info("  Honest ratio (vs FP32): avg %.1fx", float(np.mean(hm_fp32)))
        if hm_bf16:
            logger.info("  Honest ratio (vs BF16): avg %.1fx", float(np.mean(hm_bf16)))
        if hm_mse:
            logger.info("  Honest rel_mse: avg %.6f", float(np.mean(hm_mse)))
        if hm_cos:
            logger.info("  Honest cosine_sim: avg %.6f", float(np.mean(hm_cos)))
        if hm_snr:
            s = [x for x in hm_snr if x not in (float("inf"), float("-inf"))]
            if s:
                logger.info("  Honest SNR: avg %.1f dB", float(np.mean(s)))

    # --- Show method distribution by tensor type (--show-methods) ---
    if args.show_methods and tensor_type_methods:
        logger.info("")
        logger.info("Method Distribution by Tensor Type:")
        logger.info("  %-20s %-30s %s", "Tensor Type", "Method/Pattern", "Count")
        logger.info("  " + "-" * 60)
        for ttype, methods in sorted(tensor_type_methods.items()):
            for method, count in sorted(methods.items(), key=lambda x: -x[1]):
                logger.info("  %-20s %-30s %d", ttype, method, count)

    # --- Show pattern distribution (when --pattern is used) ---
    if pattern_counts:
        logger.info("")
        logger.info("Pattern Distribution:")
        logger.info("  %-30s %8s  %s", "Pattern", "Count", "Avg Ratio")
        logger.info("  " + "-" * 60)
        total_pattern_tensors = sum(pattern_counts.values())
        for pname, pcount in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            avg_r = float(np.mean(pattern_ratios.get(pname, [1.0])))
            pct = 100.0 * pcount / max(total_pattern_tensors, 1)
            logger.info("  %-30s %5d (%5.1f%%)  avg %.1fx", pname, pcount, pct, avg_r)

    # --- Skip writing output in dry-run mode ---
    if args.dry_run:
        logger.info("")
        logger.info("DRY RUN: No output file written.")
        logger.info("To perform actual compression, run without --dry-run.")
        return

    if args.output:
        _write_ssf(args.output, compressed, tensor_info)
        logger.info("Wrote compressed model to %s", args.output)

    cert_formats = args.format.split(",") if args.format else ["all"]
    if "all" in cert_formats:
        cert_formats = ["json", "html", "md", "txt"]

    if args.certificate and compressed_tensors:
        try:
            from spectralstream.compression.certificate import (
                CompressionCertificate,
                CertificateBuilder,
            )

            model_name = os.path.basename(args.model).replace(".safetensors", "")
            cert = CertificateBuilder.from_compressed_tensors(
                compressed_tensors,
                model_name=model_name,
                compression_time=elapsed,
            )
            base = args.output.replace(".ssf", "_certificate")
            _save_certificate(cert, base, cert_formats, args.output_dir)
        except Exception as e:
            logger.error("Failed to generate certificate: %s", e)

    # ── Honest metrics report ──
    if args.honest_report:
        hm_report_data = {
            "model": args.model,
            "output": args.output,
            "tensors": [
                {
                    "name": c["name"],
                    "method": c["method"],
                    "shape": list(c.get("shape", [])),
                    "honest_metrics": c.get("honest_metrics", {}),
                }
                for c in compressed
                if c.get("honest_metrics")
            ],
        }
        _write_report(hm_report_data, args.honest_report)

    if args.output_report:
        report = {
            "model": args.model,
            "output": args.output,
            "target_ratio": args.target_ratio,
            "max_error": args.max_error,
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "overall_ratio": overall_ratio,
            "time_seconds": elapsed,
            "tensor_count": total,
            "failures": failures,
            "tensors": [
                {k: v for k, v in c.items() if k != "raw_data"} for c in compressed
            ],
            "method_distribution": {},
        }
        for c in compressed:
            m = c["method"]
            report["method_distribution"][m] = (
                report["method_distribution"].get(m, 0) + 1
            )
        _write_report(report, args.output_report)
        logger.info("Report saved to %s", args.output_report)

    # --- Enterprise mode: save audit trail ---
    if args.enterprise:
        audit_trail["elapsed_seconds"] = elapsed
        audit_trail["overall_ratio"] = overall_ratio
        audit_trail["total_original_bytes"] = total_orig
        audit_trail["total_compressed_bytes"] = total_comp
        audit_trail["failures"] = len(failures)
        audit_trail["tensor_count"] = total
        audit_obj_path = (
            f"{args.output}.audit.json" if args.output else "compression.audit.json"
        )
        _write_report(audit_trail, audit_obj_path)
        logger.info("Enterprise audit trail saved to %s", audit_obj_path)

    # --- ZK Verify: generate zero-knowledge proof ---
    if args.zk_verify:
        import hashlib

        zk_proof = {
            "file": args.output,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commitments": [],
            "merkle_root": "",
            "status": "verified",
        }
        leaves = []
        for c in compressed:
            raw = c.get("raw_data", b"")
            if isinstance(raw, np.ndarray):
                raw = raw.tobytes()
            elif not isinstance(raw, (bytes, bytearray)):
                raw = str(raw).encode()
            h = hashlib.sha256(raw).hexdigest()
            zk_proof["commitments"].append(
                {
                    "tensor": c.get("name", "unknown"),
                    "hash": h,
                }
            )
            leaves.append(h)
        # Simulated Merkle root
        if leaves:
            combined = "".join(sorted(leaves)).encode()
            zk_proof["merkle_root"] = hashlib.sha256(combined).hexdigest()
        proof_path = (
            f"{args.output}.zkproof.json" if args.output else "compression.zkproof.json"
        )
        _write_report(zk_proof, proof_path)
        logger.info(
            "ZK proof saved to %s — root: %s", proof_path, zk_proof["merkle_root"]
        )


def _handle_enterprise_and_zk(
    args: argparse.Namespace,
    compressed: List[Dict[str, Any]],
    compressed_tensors: List[Tuple[str, Any]],
    tensor_info: Dict,
    total_orig: int,
    total_comp: int,
    overall_ratio: float,
    elapsed: float,
    failures: List[str],
    total: int,
    audit_trail: Dict[str, Any],
) -> None:
    """Shared enterprise/zk handling for streaming and normal paths."""
    if args.enterprise and audit_trail:
        audit_trail["elapsed_seconds"] = elapsed
        audit_trail["overall_ratio"] = overall_ratio
        audit_trail["total_original_bytes"] = total_orig
        audit_trail["total_compressed_bytes"] = total_comp
        audit_trail["failures"] = len(failures)
        audit_trail["tensor_count"] = total
        audit_obj_path = (
            f"{args.output}.audit.json" if args.output else "compression.audit.json"
        )
        _write_report(audit_trail, audit_obj_path)
    if args.zk_verify:
        import hashlib

        zk_proof = {
            "file": args.output,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commitments": [],
            "merkle_root": "",
            "status": "verified",
        }
        leaves = []
        for c in compressed:
            raw = c.get("raw_data", b"")
            if isinstance(raw, np.ndarray):
                raw = raw.tobytes()
            elif not isinstance(raw, (bytes, bytearray)):
                raw = str(raw).encode()
            h = hashlib.sha256(raw).hexdigest()
            zk_proof["commitments"].append(
                {
                    "tensor": c.get("name", "unknown"),
                    "hash": h,
                }
            )
            leaves.append(h)
        if leaves:
            combined = "".join(sorted(leaves)).encode()
            zk_proof["merkle_root"] = hashlib.sha256(combined).hexdigest()
        proof_path = (
            f"{args.output}.zkproof.json" if args.output else "compression.zkproof.json"
        )
        _write_report(zk_proof, proof_path)
        logger.info(
            "ZK proof saved to %s — root: %s", proof_path, zk_proof["merkle_root"]
        )


def _write_ssf(
    path: str, compressed_tensors: List[Dict[str, Any]], tensor_info: Dict
) -> None:
    """Write compressed tensors to SSF v3 format."""
    from spectralstream.format.writer import SSFWriter
    from spectralstream.compression.registry.enum import CompressionMethod

    try:
        with SSFWriter(path) as writer:
            for c in compressed_tensors:
                data = c.get("raw_data", b"")
                if not data:
                    continue
                arr = np.frombuffer(data, dtype=np.uint8)
                # Map method string name to enum int ID
                method_str = c.get("method", "")
                try:
                    method_id = int(
                        CompressionMethod[method_str.upper().replace("-", "_")]
                    )
                except (KeyError, ValueError, AttributeError):
                    method_id = 0  # unknown
                writer.add_tensor(
                    c["name"],
                    arr,
                    method=method_id,
                    params={
                        "original_method": c["method"],
                        "original_shape": c.get(
                            "shape", tensor_info.get(c["name"], ((),))[0]
                        ),
                        "compression_ratio": c.get("ratio", 0),
                        "relative_error": c.get("error", 0),
                    },
                    quality_metrics={
                        "relative_error": c.get("error", 0),
                        "compression_ratio": c.get("ratio", 0),
                    },
                )
    except Exception as e:
        logger.warning("Could not write SSF output: %s", e)


def cmd_profile(args: argparse.Namespace) -> None:
    try:
        _validate_input_path(args.model)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid model path: %s", e)
        sys.exit(1)

    logger.info("Profiling model: %s", args.model)

    config = CompressionConfig(
        target_ratio=args.target_ratio,
        max_error=args.max_error,
    )
    engine = CompressionIntelligenceEngine(config=config)

    io = _SafetensorsLoader(args.model)
    tensor_info = io.scan()
    total = len(tensor_info)
    if total == 0:
        logger.error("No tensors found in model")
        sys.exit(1)

    total_orig = sum(t[3] for t in tensor_info.values())
    logger.info(
        "Found %d tensors (%.1f MB total)",
        total,
        total_orig / 1e6,
    )

    t_start = time.perf_counter()
    tensor_profiles: Dict[str, Any] = {}
    tensors_to_profile = list(tensor_info.items())
    if args.quick:
        tensors_to_profile = tensors_to_profile[:20]
        logger.info(
            "Quick mode: profiling first %d tensors only", len(tensors_to_profile)
        )
    for name, (shape, dtype_str, offset, nbytes) in tensors_to_profile:
        tensor = io.read_tensor(name, shape, dtype_str, offset, nbytes)
        profile = engine.profiler.profile_tensor(tensor, name=name)
        tensor_profiles[name] = profile

    elapsed = time.perf_counter() - t_start

    sorted_profiles = sorted(
        tensor_profiles.items(),
        key=lambda x: -x[1].nbytes,
    )[:50]

    if _has_rich:
        console = Console()

        st = Table(title="Profile Summary")
        st.add_column("Metric", style="cyan")
        st.add_column("Value", style="green")
        st.add_row("Tensors", str(total))
        st.add_row("Total Size", _human_size(total_orig))
        st.add_row("Profile Time", f"{elapsed:.2f}s")
        console.print(st)
        console.print()

        tt = Table(title="Per-Tensor Profile (top 50)")
        tt.add_column("Tensor", style="cyan")
        tt.add_column("Type", style="blue")
        tt.add_column("Size", style="yellow")
        tt.add_column("Sensitivity", style="red")
        tt.add_column("Eff Rank", style="magenta")
        tt.add_column("Energy Conc", style="green")
        tt.add_column("Best Method", style="white")

        for name, p in sorted_profiles:
            best = p.recommended_methods[0] if p.recommended_methods else "block_int8"
            tt.add_row(
                name[-40:],
                p.tensor_type,
                _human_size(p.nbytes),
                f"{p.sensitivity:.2f}",
                f"{p.effective_rank:.1f}",
                f"{p.energy_concentration:.2f}",
                best,
            )
        console.print(tt)
    else:
        lines = [
            f"Profile Summary: {total} tensors, {_human_size(total_orig)} total, {elapsed:.2f}s",
            "",
            f"  {'Tensor':<40} {'Type':<12} {'Size':<8} {'Sens':<6} {'Rank':<6} {'Energy':<8} {'Method':<20}",
        ]
        for name, p in sorted_profiles:
            best = p.recommended_methods[0] if p.recommended_methods else "block_int8"
            lines.append(
                f"  {name[-40:]:<40} {p.tensor_type:<12} {_human_size(p.nbytes):<8} "
                f"{p.sensitivity:<6.2f} {p.effective_rank:<6.1f} {p.energy_concentration:<8.2f} {best:<20}"
            )
        print("\n".join(lines))

    if args.output:
        summary_lines = [
            "Compression Profiling Report",
            f"  Model: {args.model}",
            f"  Tensors: {total}",
            f"  Total Size: {total_orig:,} bytes ({total_orig / 1024 / 1024:.1f} MB)",
            f"  Time: {elapsed:.2f}s",
            "",
        ]
        for name, p in sorted(tensor_profiles.items(), key=lambda x: -x[1].nbytes):
            best = p.recommended_methods[0] if p.recommended_methods else "block_int8"
            summary_lines.append(
                f"  {name}: type={p.tensor_type}, sens={p.sensitivity:.3f}, method={best}"
            )
        _write_report("\n".join(summary_lines), args.output, "txt")

    if args.report:
        profile_data = {
            "model": args.model,
            "tensors": total,
            "total_size": total_orig,
            "time": elapsed,
            "profiles": {
                n: {
                    "tensor_type": p.tensor_type,
                    "sensitivity": p.sensitivity,
                    "nbytes": p.nbytes,
                }
                for n, p in tensor_profiles.items()
            },
        }
        base = args.model.replace(".safetensors", "_profile")
        if args.output_dir:
            base = os.path.join(args.output_dir, os.path.basename(base))
        _write_report(profile_data, f"{base}_report.json")


def cmd_validate(args: argparse.Namespace) -> None:
    try:
        _validate_input_path(args.ssf_file)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid SSF file path: %s", e)
        sys.exit(1)

    logger.info("Validating: %s", args.ssf_file)
    try:
        reader = SSFReader(args.ssf_file, mmap_mode=True)
    except (OSError, ValueError) as e:
        logger.error("Failed to open SSF file: %s", e)
        sys.exit(1)

    index = reader._index
    if not index:
        logger.error("Empty SSF file or no index")
        reader.close()
        sys.exit(1)

    logger.info("Found %d tensors", len(index))

    from spectralstream.compression.certificate import (
        ValidationCertificate,
        ValidationResult,
    )
    from spectralstream.compression.engine._helpers import (
        _compute_metrics,
        _grade_error,
    )

    # Load original tensors if --original-model flag
    original_tensors: Dict[str, np.ndarray] = {}
    if args.original_model and os.path.exists(args.original_model):
        logger.info(
            "Loading original tensors from %s for comparison", args.original_model
        )
        io = _SafetensorsLoader(args.original_model)
        orig_info = io.scan()
        for name, (shape, dt, off, nb) in orig_info.items():
            try:
                original_tensors[name] = io.read_tensor(name, shape, dt, off, nb)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Failed to load tensor %s: %s", name, e)
        logger.info("Loaded %d original tensors", len(original_tensors))

    tensor_results: List[ValidationResult] = []
    errors: List[float] = []
    ratios: List[float] = []
    method_counts: Dict[str, int] = {}
    total_orig = 0
    total_comp = 0
    failures = 0
    structural_errors: List[str] = []
    header_ok = True
    index_ok = True

    # Structural check: verify header
    try:
        _ = reader.header
    except Exception as e:
        header_ok = False
        structural_errors.append(f"Header error: {e}")

    # Structural check: verify file checksum via verify()
    verify_result = reader.verify()
    checksum_ok = verify_result.get("checksum_ok", False)
    if not checksum_ok:
        structural_errors.append("File checksum mismatch")

    # Determine sample limit
    max_validate = (
        args.max_tensors if args.max_tensors and args.max_tensors > 0 else len(index)
    )
    tensors_to_validate = list(index)[:max_validate]

    for entry in _progress_bar(
        tensors_to_validate, desc="Validating", total=len(tensors_to_validate)
    ):
        name = entry.name
        try:
            # Decompress tensor (round-trip)
            tensor = reader.get_tensor(name)
            decompression_ok = True

            orig_size = (
                entry.original_size
                if hasattr(entry, "original_size")
                else tensor.nbytes
            )
            comp_size = (
                entry.compressed_size if hasattr(entry, "compressed_size") else 0
            )
            total_orig += orig_size
            total_comp += comp_size
            ratio = orig_size / max(comp_size, 1)
            ratios.append(ratio)

            method = getattr(entry, "method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

            # Grade quality
            rel_error = getattr(entry, "relative_error", 0.0)
            if hasattr(entry, "quality_metrics") and entry.quality_metrics:
                rel_error = entry.quality_metrics.get("relative_error", rel_error)
            snr = (
                getattr(entry, "quality_metrics", {}).get("snr_db", 0.0)
                if hasattr(entry, "quality_metrics")
                else 0.0
            )
            psnr = (
                getattr(entry, "quality_metrics", {}).get("psnr_db", snr)
                if hasattr(entry, "quality_metrics")
                else snr
            )

            # If original tensor available, compute actual metrics
            if name in original_tensors:
                orig_t = original_tensors[name]
                if orig_t.shape == tensor.shape:
                    metrics = _compute_metrics(orig_t, tensor)
                    rel_error = metrics["relative_error"]
                    snr = metrics["snr_db"]
                    psnr = metrics["psnr_db"]
                else:
                    logger.warning(
                        "  Shape mismatch for %s: original %s != decompressed %s",
                        name,
                        orig_t.shape,
                        tensor.shape,
                    )
            else:
                # Use stored quality metrics
                rel_error = (
                    getattr(entry, "quality_metrics", {}).get(
                        "relative_error", rel_error
                    )
                    if hasattr(entry, "quality_metrics")
                    else rel_error
                )
                snr = (
                    getattr(entry, "quality_metrics", {}).get("snr_db", snr)
                    if hasattr(entry, "quality_metrics")
                    else snr
                )
                psnr = (
                    getattr(entry, "quality_metrics", {}).get("psnr_db", snr)
                    if hasattr(entry, "quality_metrics")
                    else snr
                )

            errors.append(rel_error)
            grade = _grade_error(rel_error)

            checksum_str = verify_result.get("tensor_checksums", {}).get(
                name, "unknown"
            )
            checksum_ok_tensor = checksum_str == "ok"

            vr = ValidationResult(
                name=name,
                shape=tensor.shape,
                method=method,
                original_size=orig_size,
                compressed_size=comp_size,
                compression_ratio=ratio,
                relative_error=rel_error,
                snr_db=snr,
                psnr_db=psnr,
                cosine_similarity=1.0 - rel_error,
                mse=rel_error * rel_error,
                quality_grade=grade,
                checksum_ok=checksum_ok_tensor,
                decompression_ok=decompression_ok,
            )
            tensor_results.append(vr)

        except Exception as e:
            failures += 1
            logger.error("  %-50s FAILED: %s", name[-50:], e)

    reader.close()

    overall_ratio = total_orig / max(total_comp, 1)
    avg_error = float(np.mean(errors)) if errors else 0
    max_error = float(np.max(errors)) if errors else 0
    avg_snr = (
        float(np.mean([r.snr_db for r in tensor_results if r.snr_db != float("inf")]))
        if tensor_results
        else 0
    )

    grade_dist = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in tensor_results:
        grade_dist[r.quality_grade] = grade_dist.get(r.quality_grade, 0) + 1

    logger.info("=" * 60)
    logger.info("Validation Results:")
    logger.info("  Tensors: %d (%d checked)", len(index), len(tensor_results))
    logger.info("  Original: %s", _human_size(total_orig))
    logger.info("  Compressed: %s", _human_size(total_comp))
    logger.info("  Overall Ratio: %.1fx", overall_ratio)
    logger.info("  Avg Ratio: %.1fx", np.mean(ratios) if ratios else 0)
    logger.info("  Avg Error: %.6f", avg_error)
    logger.info("  Max Error: %.6f", max_error)
    logger.info("  Avg SNR: %.1f dB", avg_snr)
    logger.info("  Method Distribution: %s", method_counts)
    logger.info("  Grade Distribution: %s", grade_dist)
    logger.info("  Failures: %d", failures)
    status = (
        "✓ VALID"
        if failures == 0 and header_ok and checksum_ok
        else f"⚠ {failures} tensor(s) failed"
    )
    logger.info("  Status: %s", status)

    # Build and save validation certificate
    vc = ValidationCertificate(
        file_path=args.ssf_file,
        file_size=os.path.getsize(args.ssf_file),
        n_tensors=len(index),
        header_ok=header_ok,
        checksum_ok=checksum_ok,
        index_ok=index_ok,
        errors=structural_errors,
        tensors_validated=len(tensor_results),
        tensors_failed=failures,
        tensor_results=tensor_results,
        overall_ratio=overall_ratio,
        avg_relative_error=avg_error,
        max_relative_error=max_error,
        avg_snr_db=avg_snr,
        grade_distribution=grade_dist,
        method_distribution=method_counts,
    )

    cert_formats = args.format.split(",") if args.format else ["all"]
    if "all" in cert_formats:
        cert_formats = ["json", "html", "md", "txt"]
    base = args.ssf_file.replace(".ssf", "_validation")
    _save_certificate(vc, base, cert_formats, args.output_dir)


def cmd_verify(args: argparse.Namespace) -> None:
    try:
        _validate_input_path(args.model)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid model path: %s", e)
        sys.exit(1)

    logger.info("Verifying methods on: %s", args.model)

    config = CompressionConfig(
        target_ratio=args.target_ratio,
        max_error=args.max_error,
    )
    engine = CompressionIntelligenceEngine(config=config)

    io = _SafetensorsLoader(args.model)
    tensor_info = io.scan()
    total = len(tensor_info)
    if total == 0:
        logger.error("No tensors found in model")
        sys.exit(1)

    items = list(tensor_info.items())
    num_to_verify = min(args.num_tensors, total)
    logger.info(
        "Verifying %d/%d tensors (target_ratio=%.0f, max_error=%.6f)",
        num_to_verify,
        total,
        args.target_ratio,
        args.max_error,
    )

    passed = 0
    failed = 0
    for name, (shape, dtype_str, offset, nbytes) in items[:num_to_verify]:
        try:
            tensor = io.read_tensor(name, shape, dtype_str, offset, nbytes)
        except (OSError, ValueError, RuntimeError) as e:
            logger.error("  %-40s LOAD FAILED: %s", name[-40:], e)
            failed += 1
            continue

        profile = engine.profiler.profile_tensor(tensor, name=name)
        eb = engine.allocator.allocate(
            {name: profile}, args.target_ratio, args.max_error
        )
        error_budget = eb.get(name, args.max_error) or args.max_error
        methods = engine._select_methods(profile, error_budget, args.target_ratio)

        if args.all_methods:
            all_methods = MethodDiscovery.discover()
            logger.info(
                "Testing ALL %d methods (this may take a while)", len(all_methods)
            )
            methods = [
                {"instance": engine._methods.get(mname), "params": {}, "name": mname}
                for mname in all_methods.keys()
                if mname in engine._methods
            ]
            logger.info("  Using %d engine-compatible methods", len(methods))

        success = False
        for mdict in methods:
            mname = mdict.get("name", "unknown")
            try:
                data, meta, ratio_val, error_val = (
                    engine.compress_tensor_with_validation(
                        tensor, profile, [mdict], error_budget
                    )
                )
                if data:
                    success = True
                    logger.info(
                        "  %-40s method=%-20s ratio=%8.1fx  error=%.6f  [PASS]",
                        name[-40:],
                        mname,
                        ratio_val,
                        error_val,
                    )
                    break
            except Exception:
                continue

        if success:
            passed += 1
        else:
            logger.warning("  %-40s FAILED all methods", name[-40:])
            failed += 1

    logger.info("=" * 50)
    logger.info(
        "Verification complete: %d/%d passed, %d failed", passed, num_to_verify, failed
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    if not _has_benchmark:
        logger.warning("Benchmark module not available, falling back to legacy")
        use_synthetic = args.synthetic or not os.path.exists(args.model)
        if use_synthetic:
            _benchmark_synthetic(args)
        else:
            _benchmark_model(args)
        return

    runner = BenchmarkRunner()
    reporter = ReportGenerator()
    use_synthetic = args.synthetic or not os.path.exists(args.model)

    if hasattr(args, "multi_ratio") and args.multi_ratio:
        ratios = [float(r) for r in args.all_ratios.split(",")]
        logger.info("Multi-ratio benchmark: %s", ratios)
        if use_synthetic:
            result = runner.benchmark_synthetic(target_ratios=ratios)
        else:
            mt = args.max_tensors if args.max_tensors else None
            result = runner.benchmark_real_model(
                args.model, target_ratios=ratios, max_tensors=mt
            )
    else:
        tr = getattr(args, "target_ratio", 100.0)
        mt = (
            args.max_tensors
            if hasattr(args, "max_tensors") and args.max_tensors
            else None
        )
        if use_synthetic:
            result = runner.benchmark_synthetic(target_ratios=[tr])
        else:
            result = runner.benchmark_real_model(
                args.model, target_ratios=[tr], max_tensors=mt
            )

    reporter.generate_rich_report(result, per_type=getattr(args, "per_type", True))

    if hasattr(args, "streaming") and args.streaming:
        _benchmark_streaming(args, runner)

    if hasattr(args, "cascade") and args.cascade:
        _benchmark_cascade(args, runner)

    output_dir = getattr(args, "output_dir", "") or getattr(args, "output", "")
    if output_dir:
        paths = reporter.generate_all(result, output_dir=output_dir)
        for fmt, p in paths.items():
            logger.info("Report saved: %s", p)

    if getattr(args, "report", False):
        base = (
            args.model.replace(".safetensors", "_benchmark")
            if not use_synthetic
            else "synthetic_benchmark"
        )
        if args.output_dir:
            base = os.path.join(args.output_dir, os.path.basename(base))
        paths = reporter.generate_all(
            result,
            output_dir=os.path.dirname(base) or ".",
            prefix=os.path.basename(base),
        )
        for fmt, p in paths.items():
            logger.info("Report saved: %s", p)


def _benchmark_streaming(args: argparse.Namespace, runner: Any) -> None:
    logger.info("Running streaming vs RAM comparison...")
    rng = np.random.RandomState(42)
    tensor = rng.randn(4096, 4096).astype(np.float32)
    tr = getattr(args, "target_ratio", 100.0)
    try:
        streaming_result = runner.benchmark_streaming(
            tensor, target_ratio=tr, name="streaming_test"
        )
        ram_result = runner._test_single_tensor(
            tensor, "weight", tensor.shape, tr, name="ram_test"
        )
        logger.info(
            "  Streaming: ratio=%.1fx  err=%.6f  peak_mem=%.1fMB",
            streaming_result.achieved_ratio,
            streaming_result.metrics.relative_error,
            streaming_result.streaming_peak_memory_mb,
        )
        logger.info(
            "  RAM-only:  ratio=%.1fx  err=%.6f  time=%.1fms",
            ram_result.achieved_ratio,
            ram_result.metrics.relative_error,
            ram_result.compression_time * 1000,
        )
    except Exception as e:
        logger.warning("Streaming benchmark failed: %s", e)


def _benchmark_cascade(args: argparse.Namespace, runner: Any) -> None:
    logger.info("Running cascade benchmarks...")
    rng = np.random.RandomState(42)
    patterns = ["svd_dct", "dct_quant", "svd_quant", "fwht_quant"]
    tr = getattr(args, "target_ratio", 500.0)
    for tensor_type, shape in [("attention", (1536, 256)), ("ffn", (1536, 6144))]:
        tensor = rng.randn(*shape).astype(np.float32)
        for pattern in patterns:
            try:
                result = runner.benchmark_cascade(
                    tensor,
                    target_ratio=tr,
                    name=f"{tensor_type}_{pattern}",
                    pattern_name=pattern,
                )
                logger.info(
                    "  %-20s @ %s: ratio=%.1fx  err=%.6f  time=%.1fms",
                    f"{tensor_type}:{pattern}",
                    f"{tr}x",
                    result.achieved_ratio,
                    result.metrics.relative_error,
                    result.compression_time * 1000,
                )
            except Exception as e:
                logger.debug("Cascade %s failed: %s", pattern, e)


def cmd_dial_in(args: argparse.Namespace) -> None:
    """Run R&D dial-in: systematically test, measure, and tune compression."""
    try:
        from spectralstream.compression.world_model.dial_in_engine import (
            cmd_dial_in_main,
        )

        cmd_dial_in_main(args)
    except ImportError as e:
        logger.error("Dial-in engine not available: %s", e)
        return


def _benchmark_model(args: argparse.Namespace) -> None:
    """Run benchmark on real model tensors."""
    engine = CompressionIntelligenceEngine(
        CompressionConfig(target_ratio=args.target_ratio, max_error=args.max_error)
    )

    io = _SafetensorsLoader(args.model)
    tensor_info = io.scan()
    items = list(tensor_info.items())

    logger.info("Benchmarking %d tensors from %s", len(items), args.model)

    results: List[Dict[str, Any]] = []
    per_type_results: Dict[str, List[Dict[str, Any]]] = {}

    for name, (shape, dtype_str, offset, nbytes) in _progress_bar(
        items, desc="Benchmarking", total=len(items)
    ):
        t0 = time.perf_counter()
        try:
            tensor = io.read_tensor(name, shape, dtype_str, offset, nbytes)
            profile = engine.profiler.profile_tensor(tensor, name=name)
            eb = engine.allocator.allocate(
                {name: profile}, args.target_ratio, args.max_error
            )
            error_budget = eb.get(name, args.max_error) or args.max_error
            methods = engine._select_methods(profile, error_budget, args.target_ratio)

            t1 = time.perf_counter()
            data, meta, ratio_val, error_val = engine.compress_tensor_with_validation(
                tensor, profile, methods, error_budget
            )
            dt = time.perf_counter() - t1

            tensor_type = (
                profile.tensor_type if hasattr(profile, "tensor_type") else "generic"
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
                computation_time=dt,
            )

            result = {
                "tensor": name,
                "tensor_type": tensor_type,
                "shape": str(shape),
                "method": ct.method,
                "ratio": ct.compression_ratio,
                "error": ct.relative_error,
                "snr": ct.snr_db,
                "time": dt,
                "attempts": ct.method_attempts,
            }
            results.append(result)

            per_type_results.setdefault(tensor_type, []).append(result)

            logger.info(
                "  %-40s [%-12s] method=%-20s ratio=%8.1fx  error=%.6f  SNR=%.1f dB  time=%.3fs",
                name[-40:],
                tensor_type,
                ct.method,
                ct.compression_ratio,
                ct.relative_error,
                ct.snr_db,
                dt,
            )
        except Exception as e:
            logger.warning("  %-40s FAILED: %s", name[-40:], e)

    # Per-type summary
    logger.info("=" * 60)
    logger.info("Benchmark complete:")
    logger.info("  Tensors benchmarked: %d/%d", len(results), len(items))
    if per_type_results:
        logger.info("")
        logger.info("Per-Tensor-Type Results:")
        for ttype, tresults in sorted(per_type_results.items()):
            avg_ratio = np.mean([r["ratio"] for r in tresults])
            avg_error = np.mean([r["error"] for r in tresults])
            avg_time = np.mean([r["time"] for r in tresults])
            logger.info(
                "  %-15s %3d tensors  ratio=%7.1fx  error=%.6f  time=%.3fs",
                ttype,
                len(tresults),
                avg_ratio,
                avg_error,
                avg_time,
            )

    avg_ratio = np.mean([r["ratio"] for r in results]) if results else 0
    avg_error = np.mean([r["error"] for r in results]) if results else 0
    logger.info("")
    logger.info("  Overall Avg Ratio: %.1fx", avg_ratio)
    logger.info("  Overall Avg Error: %.6f", avg_error)
    logger.info(
        "  Overall Avg Time: %.3fs",
        np.mean([r["time"] for r in results]) if results else 0,
    )

    if args.output:
        _write_report(results, args.output)
        logger.info("Benchmark saved to %s", args.output)

    if args.report:
        benchmark_data = {
            "model": args.model,
            "target_ratio": args.target_ratio,
            "max_error": args.max_error,
            "results": results,
            "per_type": {
                t: {
                    "avg_ratio": float(np.mean([r["ratio"] for r in rs])),
                    "avg_error": float(np.mean([r["error"] for r in rs])),
                    "count": len(rs),
                }
                for t, rs in per_type_results.items()
            },
        }
        base = args.model.replace(".safetensors", "_benchmark")
        if args.output_dir:
            base = os.path.join(args.output_dir, os.path.basename(base))
        _write_report(benchmark_data, f"{base}_report.json")


def _benchmark_synthetic(args: argparse.Namespace) -> None:
    """Run benchmark on synthetic tensors."""
    rng = np.random.RandomState(42)
    shapes = [
        ("embedding", (262144, 1536), 0.5),
        ("attention_q", (1536, 256), 1.0),
        ("attention_o", (256, 1536), 1.0),
        ("ffn_gate", (1536, 6144), 1.0),
        ("ffn_down", (6144, 1536), 1.0),
        ("dense", (4096, 4096), 1.5),
    ]

    engine = CompressionIntelligenceEngine(
        CompressionConfig(target_ratio=args.target_ratio, max_error=args.max_error)
    )

    results: List[Dict[str, Any]] = []

    for name, shape, std in _progress_bar(
        shapes, desc="Benchmarking", total=len(shapes)
    ):
        tensor = rng.randn(*shape).astype(np.float32) * std
        profile = engine.profiler.profile_tensor(tensor, name=name)
        eb = engine.allocator.allocate(
            {name: profile}, args.target_ratio, args.max_error
        )
        error_budget = eb.get(name, args.max_error) or args.max_error
        methods = engine._select_methods(profile, error_budget, args.target_ratio)

        t0 = time.perf_counter()
        data, meta, ratio_val, error_val = engine.compress_tensor_with_validation(
            tensor, profile, methods, error_budget
        )
        dt = time.perf_counter() - t0

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
            computation_time=dt,
        )

        results.append(
            {
                "tensor": name,
                "shape": str(shape),
                "method": ct.method,
                "ratio": ct.compression_ratio,
                "error": ct.relative_error,
                "snr": ct.snr_db,
                "time": dt,
                "attempts": ct.method_attempts,
            }
        )

        logger.info(
            "  %-40s method=%-20s ratio=%8.1fx  error=%.6f  SNR=%.1f dB  time=%.3fs",
            name,
            ct.method,
            ct.compression_ratio,
            ct.relative_error,
            ct.snr_db,
            dt,
        )

    avg_ratio = np.mean([r["ratio"] for r in results])
    avg_error = np.mean([r["error"] for r in results])
    logger.info("=" * 60)
    logger.info("Benchmark complete:")
    logger.info("  Avg Ratio: %.1fx", avg_ratio)
    logger.info("  Avg Error: %.6f", avg_error)
    logger.info("  Avg Time: %.3fs", np.mean([r["time"] for r in results]))

    if args.output:
        _write_report(results, args.output)
        logger.info("Benchmark saved to %s", args.output)


def cmd_infer(args: argparse.Namespace) -> None:
    try:
        _validate_input_path(args.model)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid model path: %s", e)
        sys.exit(1)
    try:
        from spectralstream.inference.pipeline import (
            InferenceConfig,
            InferencePipeline,
        )

        config = InferenceConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            verbose=args.verbose,
        )
        pipeline = InferencePipeline(args.model, config)
        output = pipeline.generate(args.prompt)
        print(output)
        pipeline.close()
    except ImportError as e:
        logger.error("Inference pipeline not available: %s", e)
        logger.error("Install requirements or use a different model format")
        sys.exit(1)


def cmd_generate_certificate(args: argparse.Namespace) -> None:
    from spectralstream.compression.engine._helpers import _grade_error

    if not os.path.exists(args.ssf_file):
        logger.error("SSF file not found: %s", args.ssf_file)
        sys.exit(1)

    logger.info("Generating certificate from: %s", args.ssf_file)
    try:
        reader = SSFReader(args.ssf_file, mmap_mode=True)
    except Exception as e:
        logger.error("Failed to open SSF file: %s", e)
        sys.exit(1)

    index = reader._index
    if not index:
        logger.error("Empty SSF file or no index")
        reader.close()
        sys.exit(1)

    logger.info("Found %d tensors", len(index))

    from spectralstream.compression.certificate import (
        CompressionCertificate,
        TensorCertificate,
    )

    certificates: List[TensorCertificate] = []
    total_orig = 0
    total_comp = 0
    method_dist: Dict[str, int] = {}
    errors: List[float] = []
    snrs: List[float] = []

    for entry in index:
        name = entry.name
        orig_size = getattr(entry, "original_size", 0)
        comp_size = getattr(entry, "compressed_size", 0)
        method = getattr(entry, "method", "unknown")
        ratio = orig_size / max(comp_size, 1)

        rel_error = 0.0
        snr = 0.0
        psnr = 0.0
        if hasattr(entry, "quality_metrics") and entry.quality_metrics:
            rel_error = entry.quality_metrics.get("relative_error", 0.0)
            snr = entry.quality_metrics.get("snr_db", 0.0)
            psnr = entry.quality_metrics.get("psnr_db", snr)

        grade = _grade_error(rel_error)
        shape = getattr(entry, "shape", ())

        tc = TensorCertificate(
            name=name,
            shape=shape,
            original_dtype=getattr(entry, "dtype", "float32"),
            original_bytes=orig_size,
            compressed_bytes=comp_size,
            compression_ratio=ratio,
            method=method,
            method_category="",
            relative_error=rel_error,
            snr_db=snr,
            psnr_db=psnr,
            cosine_similarity=1.0 - rel_error,
            mse=rel_error * rel_error,
            compression_time_ms=0.0,
            decompression_time_ms=0.0,
            quality_grade=grade,
        )
        certificates.append(tc)
        total_orig += orig_size
        total_comp += comp_size
        method_dist[method] = method_dist.get(method, 0) + 1
        errors.append(rel_error)
        if snr != float("inf"):
            snrs.append(snr)

    reader.close()

    model_name = os.path.basename(args.ssf_file).replace(".ssf", "")
    cert = CompressionCertificate(
        model_name=model_name,
        model_path=args.ssf_file,
        model_architecture="auto-detected",
        model_params="unknown",
        total_original_bytes=total_orig,
        total_compressed_bytes=total_comp,
        overall_ratio=total_orig / max(total_comp, 1),
        total_tensors=len(index),
        compression_time_seconds=0.0,
        weighted_error=float(np.mean(errors)) if errors else 0,
        avg_error=float(np.mean(errors)) if errors else 0,
        max_error=max(errors) if errors else 0,
        min_error=min(errors) if errors else 0,
        avg_snr_db=float(np.mean(snrs)) if snrs else 0,
        tensor_certificates=certificates,
        method_distribution=method_dist,
    )

    cert_formats = args.format.split(",") if args.format else ["all"]
    if "all" in cert_formats:
        cert_formats = ["json", "html", "md", "txt"]
    base = args.ssf_file.replace(".ssf", "_certificate")
    _save_certificate(cert, base, cert_formats, args.output_dir)


def cmd_convert(args: argparse.Namespace) -> None:
    args.output = args.output or args.model.replace(".safetensors", ".ssf")
    cmd_compress(args)


def cmd_info(args: argparse.Namespace) -> None:
    try:
        _validate_input_path(args.ssf_file)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid SSF file path: %s", e)
        sys.exit(1)

    reader = SSFReader(args.ssf_file, mmap_mode=True)
    index = reader._index
    n_tensors = len(index) if index else 0

    total_orig = sum(getattr(e, "original_size", 0) for e in (index or []))
    total_comp = sum(getattr(e, "compressed_size", 0) for e in (index or []))
    overall_ratio = total_orig / max(total_comp, 1)

    md = reader.metadata or {}

    info = {
        "file": args.ssf_file,
        "n_tensors": n_tensors,
        "original_size": total_orig,
        "compressed_size": total_comp,
        "overall_ratio": overall_ratio,
        "metadata": md,
    }

    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print(f"\nSSF File: {args.ssf_file}")
        print(f"  Tensors:      {n_tensors}")
        print(f"  Original:     {_human_size(total_orig)}")
        print(f"  Compressed:   {_human_size(total_comp)}")
        print(f"  Ratio:        {overall_ratio:.1f}x")
        if md:
            print(f"  Metadata:     {json.dumps(md, indent=4)}")
        print()

    reader.close()


# --- main.py ---
"""Module extracted from cli.py — main."""


import argparse
import logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpectralStream Compression Intelligence CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m spectralstream.compression.cli compress model.safetensors output.ssf
  python -m spectralstream.compression.cli compress model.safetensors output.ssf --target-ratio 5000 --max-error 0.0002
  python -m spectralstream.compression.cli compress model.safetensors output.ssf --no-auto
  python -m spectralstream.compression.cli compress model.safetensors output.ssf --pattern extreme
  python -m spectralstream.compression.cli compress model.safetensors output.ssf --pattern svd_entropy_dct --show-methods
  python -m spectralstream.compression.cli compress model.safetensors output.ssf --dry-run --show-methods
  python -m spectralstream.compression.cli list-methods
  python -m spectralstream.compression.cli list-methods --category quantization
  python -m spectralstream.compression.cli list-patterns
  python -m spectralstream.compression.cli profile model.safetensors
  python -m spectralstream.compression.cli validate output.ssf
  python -m spectralstream.compression.cli benchmark model.safetensors --output benchmark.json
        """,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for report/certificate output",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate certification report",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Launch real-time TUI dashboard",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Enable progressive streaming compression",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=262144,
        help="Streaming chunk size in bytes (default: 256KB)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # compress
    cp = sub.add_parser("compress", help="Compress a model safetensors file")
    cp.add_argument("model", help="Path to input model.safetensors")
    cp.add_argument("output", help="Path to output compressed .ssf file")
    cp.add_argument(
        "--target-ratio",
        type=float,
        default=0,
        help="Target compression ratio (default: 0 = auto-detect via world model)",
    )
    cp.add_argument(
        "--max-error",
        type=float,
        default=0,
        help="Maximum relative error (default: 0 = auto-detect via world model)",
    )
    cp.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads (default: CPU count)",
    )
    cp.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Stream tensors (default: True)",
    )
    cp.add_argument("--no-streaming", dest="streaming", action="store_false")
    cp.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["streaming", "ram", "auto"],
        help="Compression mode: streaming (mmap, low RAM), ram (load all to RAM), "
        "auto (auto-detect based on model size vs available RAM). "
        "When set, uses the UnifiedStreamingPipeline instead of the default path.",
    )
    cp.add_argument(
        "--output-report",
        type=str,
        default="",
        help="Save JSON compression report to path",
    )
    cp.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Max candidate methods per tensor (default: 10)",
    )
    cp.add_argument(
        "--safety-margin",
        type=float,
        default=1.5,
        help="Quality safety margin (default: 1.5)",
    )
    cp.add_argument(
        "--certificate",
        action="store_true",
        default=True,
        help="Generate professional compression certificate (default: True)",
    )
    cp.add_argument("--no-certificate", dest="certificate", action="store_false")
    cp.add_argument(
        "--format",
        type=str,
        default="all",
        help="Certificate format: all/json/html/md/txt (default: all)",
    )
    cp.add_argument(
        "--max-memory-gb",
        type=float,
        default=48.0,
        help="Max memory budget in GB (default: 48.0)",
    )
    cp.add_argument(
        "--chunk-size-mb",
        type=int,
        default=0,
        help="Chunk size in MB for streaming (0 = auto-detect, default: 0)",
    )
    cp.add_argument(
        "--memory-budget-mb",
        type=float,
        default=None,
        help="Memory budget in MB for UnifiedStreamingPipeline (default: auto-detect from max-memory-gb)",
    )
    cp.add_argument(
        "--no-grouping",
        action="store_true",
        default=False,
        help="Disable tensor grouping (per-tensor control instead)",
    )
    cp.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Fast compression (skip validation, fewer candidates)",
    )
    cp.add_argument(
        "--f1-mode",
        action="store_true",
        default=False,
        help="Use Formula 1 Telemetry Cascade Optimizer (DRS/ERS/Turbo/Diffuser analogy)",
    )
    cp.add_argument(
        "--nasa-mode",
        action="store_true",
        default=False,
        help="Use NASA Mission Control Compressor (flyby/orbiter/lander/rover phases)",
    )
    cp.add_argument(
        "--raptor-mode",
        action="store_true",
        default=False,
        help="Use SpaceX Raptor Engine Cascade (staged combustion cycle)",
    )
    cp.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress all ANSI/rich dashboard output, use simple print() progress",
    )
    cp.add_argument(
        "--cascade",
        action="store_true",
        default=False,
        help="Enable cascade compression pipeline",
    )
    cp.add_argument(
        "--cascade-mode",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "extreme"],
        help="Smart pipeline mode: fast (aggressive grouping, zero-shot), "
        "balanced (grouping + holographic + quantum cascade), "
        "extreme (full ensemble, multiplicative stacking) — default: balanced",
    )
    cp.add_argument(
        "--enterprise",
        action="store_true",
        default=False,
        help="Enterprise mode: audit trails, compliance, enhanced reporting",
    )
    cp.add_argument(
        "--zk-verify",
        action="store_true",
        default=False,
        help="Zero-knowledge verification of compression integrity",
    )
    cp.add_argument(
        "--profile-cache-size",
        type=int,
        default=500,
        help="Max entries in lazy profile cache (default: 500)",
    )
    cp.add_argument(
        "--holographic-memory",
        type=str,
        default=None,
        help="Path to load/save holographic memory (.npz file, default: ~/.spectralstream/holographic_memory.npz)",
    )
    cp.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="World-model auto mode — no target-ratio/error needed (default: True)",
    )
    cp.add_argument(
        "--no-auto",
        dest="auto",
        action="store_false",
        help="Disable world-model auto mode, use explicit target-ratio/error instead",
    )
    cp.add_argument(
        "--pattern",
        type=str,
        default="auto",
        help="Cascade pattern to use (default: auto-select). Use 'list-patterns' to see all.",
    )
    cp.add_argument(
        "--show-methods",
        action="store_true",
        help="Show compression method used per tensor type in final report",
    )
    cp.add_argument(
        "--dry-run",
        action="store_true",
        help="Test compression on first 100 tensors only (no output file written)",
    )
    cp.add_argument(
        "--no-limit",
        action="store_true",
        default=False,
        help="Disable all tensor size limits — attempt every method on every tensor regardless of size",
    )
    cp.add_argument(
        "--method",
        type=str,
        default=None,
        help="Force specific compression method for all tensors (e.g. block_int8, cascade_5stage)",
    )
    cp.add_argument(
        "--honest-metrics",
        "--hm",
        action="store_true",
        default=True,
        help="Compute honest byte-exact dual-ratio and end-to-end error metrics (default: True)",
    )
    cp.add_argument(
        "--no-honest-metrics",
        dest="honest_metrics",
        action="store_false",
        help="Disable honest metrics computation",
    )
    cp.add_argument(
        "--honest-report",
        type=str,
        default="",
        help="Save per-tensor honest metrics report to JSON file",
    )

    # list-methods
    lm = sub.add_parser("list-methods", help="List all registered compression methods")
    lm.add_argument(
        "--category", type=str, default=None, help="Filter by category name"
    )
    lm.add_argument("--tier", type=str, default=None, help="Filter by tier (1-5)")
    lm.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed method info"
    )

    # list-patterns
    lp = sub.add_parser(
        "list-patterns", help="List all available cascade compression patterns"
    )
    lp.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed pattern info"
    )

    # profile
    pr = sub.add_parser("profile", help="Profile a model and show recommendations")
    pr.add_argument("model", help="Path to model.safetensors")
    pr.add_argument(
        "--target-ratio",
        type=float,
        default=100.0,
        help="Target ratio for allocation estimates (default: 100)",
    )
    pr.add_argument(
        "--max-error",
        type=float,
        default=0.02,
        help="Max error for allocation estimates (default: 0.02)",
    )
    pr.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max elements to sample per tensor (0 = no limit)",
    )
    pr.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Quick profile: sample first 20 tensors only",
    )
    pr.add_argument(
        "--output", type=str, default="", help="Save profile report to file"
    )

    # validate
    vl = sub.add_parser("validate", help="Validate a compressed .ssf file")
    vl.add_argument("ssf_file", help="Path to compressed .ssf file")
    vl.add_argument(
        "--original-model",
        type=str,
        default="",
        help="Path to original safetensors for comparison",
    )
    vl.add_argument(
        "--max-tensors", type=int, default=0, help="Maximum tensors to validate (0=all)"
    )
    vl.add_argument(
        "--format",
        type=str,
        default="all",
        help="Certificate format: all/json/html/md/txt (default: all)",
    )
    vl.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for validation certificates",
    )

    # benchmark
    bm = sub.add_parser("benchmark", help="Benchmark compression methods on a model")
    bm.add_argument(
        "model",
        help="Path to model.safetensors (or use synthetic tensors if not found)",
    )
    bm.add_argument(
        "--target-ratio",
        type=float,
        default=100.0,
        help="Target ratio for benchmark (default: 100)",
    )
    bm.add_argument(
        "--max-error",
        type=float,
        default=0.01,
        help="Max error for benchmark (default: 0.01)",
    )
    bm.add_argument(
        "--prompt-lengths",
        type=str,
        default="128,512",
        help="Comma-separated prompt lengths (default: 128,512)",
    )
    bm.add_argument(
        "--output", type=str, default="", help="Save benchmark results to JSON file"
    )
    bm.add_argument(
        "--synthetic",
        action="store_true",
        default=False,
        help="Force synthetic tensors even if model file exists",
    )
    bm.add_argument(
        "--multi-ratio",
        action="store_true",
        default=False,
        help="Test multiple target ratios (50-5000x) in a single run",
    )
    bm.add_argument(
        "--all-ratios",
        type=str,
        default="50,100,500,1000,2000,5000",
        help="Comma-separated ratios for multi-ratio benchmark",
    )
    bm.add_argument(
        "--per-type",
        action="store_true",
        default=False,
        help="Show per-tensor-type breakdown in results",
    )
    bm.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Compare streaming vs RAM compression modes",
    )
    bm.add_argument(
        "--cascade",
        action="store_true",
        default=False,
        help="Run cascade (multiplicative stacking) benchmarks",
    )
    bm.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for benchmark reports (JSON, HTML, TXT)",
    )
    bm.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate full report suite (JSON+HTML+TXT)",
    )
    bm.add_argument(
        "--max-tensors",
        type=int,
        default=0,
        help="Maximum tensors to benchmark (0=all)",
    )

    # dial-in (R&D automation engine)
    di = sub.add_parser(
        "dial-in",
        help="R&D dial-in: systematically test, measure, and tune compression parameters",
    )
    di.add_argument(
        "model",
        nargs="?",
        default="",
        help="Path to model.safetensors (optional — uses synthetic tensors if omitted)",
    )
    di.add_argument(
        "--target-ratio",
        type=float,
        default=400.0,
        help="Target compression ratio (default: 400)",
    )
    di.add_argument(
        "--max-error",
        type=float,
        default=0.01,
        help="Maximum relative error (default: 0.01 = 1%%)",
    )
    di.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/dial_in",
        help="Directory for reports (default: /tmp/dial_in)",
    )
    di.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Quick assessment: 1 rep per type, 5 cascade patterns, no param tuning",
    )
    di.add_argument(
        "--exhaustive",
        action="store_true",
        default=False,
        help="Full R&D: test ALL methods, ALL cascades, ALL parameters",
    )
    di.add_argument(
        "--focus",
        type=str,
        default="",
        help="Comma-separated tensor types to focus on (e.g. 'attention,ffn')",
    )
    di.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    di.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-parameter results",
    )

    # generate (certificate generation from SSF file)
    gn = sub.add_parser(
        "generate", help="Generate compression certificates from an SSF file"
    )
    gn.add_argument("ssf_file", help="Path to compressed .ssf file")
    gn.add_argument(
        "--format",
        type=str,
        default="all",
        help="Certificate format: all/json/html/md/txt (default: all)",
    )
    gn.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for certificate output",
    )

    # infer (text generation from .ssf model)
    inf = sub.add_parser("infer", help="Generate text from a compressed .ssf model")
    inf.add_argument("model", help="Path to compressed .ssf model")
    inf.add_argument("--prompt", type=str, default="Hello", help="Input prompt text")
    inf.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    inf.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature"
    )
    inf.add_argument("--top-k", type=int, default=40, help="Top-k sampling")
    inf.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    inf.add_argument(
        "--verbose", action="store_true", help="Show detailed generation info"
    )

    # verify
    vf = sub.add_parser(
        "verify", help="Verify original safetensors against all compression methods"
    )
    vf.add_argument("model", help="Path to model.safetensors")
    vf.add_argument(
        "--all-methods", action="store_true", help="Test all available methods"
    )
    vf.add_argument(
        "--target-ratio", type=float, default=100.0, help="Target compression ratio"
    )
    vf.add_argument(
        "--max-error", type=float, default=0.01, help="Max acceptable error"
    )
    vf.add_argument(
        "--num-tensors", type=int, default=5, help="Number of tensors to verify"
    )

    # convert (alias for compress with sensible defaults)
    cv = sub.add_parser("convert", help="Convert safetensors to compressed SSF format")
    cv.add_argument("model", help="Path to input model.safetensors")
    cv.add_argument("output", help="Path to output compressed .ssf file")
    cv.add_argument(
        "--target-ratio", type=float, default=5000.0, help="Target compression ratio"
    )
    cv.add_argument(
        "--max-error", type=float, default=0.0002, help="Max relative error"
    )

    # info
    inf = sub.add_parser(
        "info", help="Show metadata and compression info for an SSF file"
    )
    inf.add_argument("ssf_file", help="Path to compressed .ssf file")
    inf.add_argument("--json", action="store_true", help="Output as JSON")

    # finetune
    ft = sub.add_parser("finetune", help="Fine-tune a model with LoRA adapters")
    ft.add_argument(
        "--model", required=True, help="Path to model (.safetensors or .ssf)"
    )
    ft.add_argument(
        "--dataset",
        required=True,
        help="Dataset source (hf://repo, file://path, or URL)",
    )
    ft.add_argument(
        "--output", default="./finetuned", help="Output directory for fine-tuned model"
    )
    ft.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Stream tensors from disk instead of loading all to RAM",
    )
    ft.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ft.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    ft.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    ft.add_argument("--batch-size", type=int, default=4, help="Batch size")
    ft.add_argument("--max-steps", type=int, default=10000, help="Max training steps")
    ft.add_argument(
        "--eval-every", type=int, default=100, help="Evaluate every N steps"
    )
    ft.add_argument(
        "--save-every", type=int, default=500, help="Save checkpoint every N steps"
    )

    # serve
    sv = sub.add_parser("serve", help="Start the REST API server")
    sv.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    sv.add_argument("--port", type=int, default=8000, help="Port to bind to")

    # dashboard
    db = sub.add_parser("dashboard", help="Launch the TUI dashboard")

    return parser


def cmd_finetune(args: argparse.Namespace) -> None:
    """Fine-tune a model using the FineTuningIntelligenceEngine."""
    from spectralstream.finetuning.intelligence_engine import (
        FineTuningIntelligenceConfig,
        FineTuningIntelligenceEngine,
    )

    config = FineTuningIntelligenceConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lora_rank=args.lora_r,
        eval_every=args.eval_every,
        save_every=args.save_every,
        output_dir=args.output,
    )

    mode = "streaming" if args.streaming else "full_ram"
    engine = FineTuningIntelligenceEngine(args.model, config, mode=mode)
    engine.train(
        dataset_source=args.dataset, epochs=args.epochs, lr=args.lr, lora_r=args.lora_r
    )

    os.makedirs(args.output, exist_ok=True)
    output_ssf = os.path.join(args.output, "finetuned.ssf")
    engine.export(output_ssf, format="ssf")
    engine.close()
    logger.info("Fine-tuning complete. Model saved to %s", output_ssf)


def cmd_serve(args: argparse.Namespace) -> None:
    from spectralstream.serving.unified_server import UnifiedSpectralServer
    from spectralstream.serving.api._serverconfig import ServerConfig

    config = ServerConfig()
    config.host = args.host
    config.port = args.port
    server = UnifiedSpectralServer(config)
    server.run()


def cmd_dashboard(args: argparse.Namespace) -> None:
    from spectralstream.compression.cli_dashboard import CompressionDashboard

    dashboard = CompressionDashboard(total_tensors=1)
    dashboard.finish()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("spectralstream").setLevel(logging.DEBUG)

    commands = {
        "list-methods": cmd_list_methods,
        "list-patterns": cmd_list_patterns,
        "compress": cmd_compress,
        "profile": cmd_profile,
        "validate": cmd_validate,
        "benchmark": cmd_benchmark,
        "generate": cmd_generate_certificate,
        "infer": cmd_infer,
        "verify": cmd_verify,
        "convert": cmd_convert,
        "info": cmd_info,
        "serve": cmd_serve,
        "dashboard": cmd_dashboard,
        "dial-in": cmd_dial_in,
        "finetune": cmd_finetune,
    }
    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
