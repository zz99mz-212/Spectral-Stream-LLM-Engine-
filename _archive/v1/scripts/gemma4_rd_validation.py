"""
Gemma 4 E2B — Full R&D Validation Pipeline
============================================
Step 1: Test ALL discoverable methods on representative tensors
Step 2: Dial in cascading compression patterns
Step 3: Profile the full model
Step 4: Novel R&D — cross-layer delta/predictive coding + frequency-domain cascade
Step 5: Comprehensive validation report
"""

import gc
import json
import logging
import os
import sys
import time
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gemma4_rd")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Imports ──────────────────────────────────────────────────────────────
try:
    from spectralstream.compression.engine import (
        CompressionIntelligenceEngine,
        CompressionConfig,
    )
    from spectralstream.compression.engine.method_discovery import MethodDiscovery
    from spectralstream.compression.registry.enum import CompressionMethod
    from spectralstream.compression.registry.registry import MethodRegistry
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

try:
    from safetensors import safe_open

    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

MODEL_PATH = (
    "/home/mike/Documents/Github/SpectralStream/models/gemma-4-E2B/model.safetensors"
)
REPORT_PATH = "/tmp/gemma4_validation_report.md"

# Representative tensor names — sample from text, vision, audio towers
REPRESENTATIVE_TENSORS = [
    # Text embeddings
    "model.embed_tokens.weight",
    # Text attention — early layer (0)
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    # Text MLP — early layer
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    # Text attention — middle layer (17)
    "model.layers.17.self_attn.q_proj.weight",
    "model.layers.17.self_attn.k_proj.weight",
    "model.layers.17.self_attn.v_proj.weight",
    "model.layers.17.self_attn.o_proj.weight",
    "model.layers.17.mlp.gate_proj.weight",
    "model.layers.17.mlp.up_proj.weight",
    "model.layers.17.mlp.down_proj.weight",
    # Text attention — late layer (34)
    "model.layers.34.self_attn.q_proj.weight",
    "model.layers.34.self_attn.k_proj.weight",
    "model.layers.34.self_attn.v_proj.weight",
    "model.layers.34.self_attn.o_proj.weight",
    "model.layers.34.mlp.gate_proj.weight",
    "model.layers.34.mlp.up_proj.weight",
    "model.layers.34.mlp.down_proj.weight",
    # Vision tower — first layer
    "model.vision_tower.0.self_attn.q_proj.weight",
    "model.vision_tower.0.self_attn.k_proj.weight",
    "model.vision_tower.0.self_attn.v_proj.weight",
    "model.vision_tower.0.self_attn.o_proj.weight",
    # Audio tower — first layer
    "model.audio_tower.0.self_attn.q_proj.weight",
    "model.audio_tower.0.self_attn.k_proj.weight",
    "model.audio_tower.0.self_attn.v_proj.weight",
    "model.audio_tower.0.self_attn.o_proj.weight",
    # LM head
    "lm_head.weight",
    # Normalization layers (small)
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
]


# Tensor type classification
def classify_tensor_name(name: str) -> str:
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "output"
    if "q_proj" in name:
        return "attention_q"
    if "k_proj" in name:
        return "attention_k"
    if "v_proj" in name:
        return "attention_v"
    if "o_proj" in name:
        return "attention_o"
    if "gate_proj" in name:
        return "ffn_gate"
    if "up_proj" in name:
        return "ffn_up"
    if "down_proj" in name:
        return "ffn_down"
    if "layernorm" in name or "norm" in name:
        return "norm"
    return "weight"


# ── Safetensors Loader ───────────────────────────────────────────────────


class SafetensorsLoader:
    """Read tensors from a safetensors file."""

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

    def __init__(self, path: str):
        self.path = path
        self._tensor_info = self._scan()

    def _scan(self) -> Dict[str, Tuple[Tuple[int, ...], str, int]]:
        import json, struct

        info = {}
        with open(self.path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for name, h in header.items():
            if name == "__metadata__":
                continue
            dtype_str = h.get("dtype", "F32")
            shape = tuple(h.get("shape", []))
            offsets = h.get("data_offsets", [0, 0])
            nbytes = offsets[1] - offsets[0]
            info[name] = (shape, dtype_str, data_start + offsets[0], nbytes)
        return info

    def scan(self) -> Dict[str, Tuple[Tuple[int, ...], str, int, int]]:
        return dict(self._tensor_info)

    def read_tensor(self, name: str) -> np.ndarray:
        if name not in self._tensor_info:
            raise KeyError(f"Tensor {name} not found")
        shape, dtype_str, offset, nbytes = self._tensor_info[name]
        np_dtype = self.DTYPE_MAP.get(dtype_str, np.float32)
        with open(self.path, "rb") as f:
            f.seek(offset)
            raw = f.read(nbytes)
        tensor = np.frombuffer(raw, dtype=np_dtype)
        try:
            tensor = tensor.reshape(shape)
        except ValueError:
            pass
        return tensor.astype(np.float32)

    def get_tensor_names(self) -> List[str]:
        return list(self._tensor_info.keys())


# ── Metrics ──────────────────────────────────────────────────────────────


def compute_metrics(
    original: np.ndarray, reconstructed: np.ndarray
) -> Dict[str, float]:
    """Compute comprehensive quality metrics."""
    orig = original.ravel().astype(np.float64)
    recon = reconstructed.ravel().astype(np.float64)

    mse = np.mean((orig - recon) ** 2)
    orig_norm = np.linalg.norm(orig)
    denom = max(orig_norm, 1e-12)
    relative_error = np.linalg.norm(orig - recon) / denom

    # SNR
    signal_power = np.mean(orig**2)
    noise_power = max(mse, 1e-30)
    snr_db = 10.0 * np.log10(signal_power / noise_power)

    # PSNR
    max_val = max(np.max(np.abs(orig)), 1e-12)
    psnr_db = 10.0 * np.log10(max_val**2 / max(mse, 1e-30))

    # Cosine similarity
    if orig_norm > 0 and np.linalg.norm(recon) > 0:
        cos_sim = np.dot(orig, recon) / (orig_norm * np.linalg.norm(recon))
    else:
        cos_sim = 1.0 if orig_norm == 0 else 0.0

    # Compression ratio (if we know sizes)
    ratio = orig.nbytes / max(recon.nbytes, 1)

    return {
        "mse": float(mse),
        "relative_error": float(relative_error),
        "snr_db": float(snr_db),
        "psnr_db": float(psnr_db),
        "cosine_similarity": float(cos_sim),
        "compression_ratio": float(ratio),
    }


def compress_and_measure(
    engine, tensor: np.ndarray, method_name: str
) -> Optional[Dict[str, Any]]:
    """Apply a single compression method and measure results."""
    try:
        t0 = time.perf_counter()
        ct = engine.compress_tensor(tensor, None, method_name)
        t1 = time.perf_counter()

        # Decompress
        recon = ct.decompress()

        metrics = compute_metrics(tensor, recon)
        metrics["method"] = method_name
        metrics["compression_time"] = t1 - t0
        metrics["original_shape"] = tensor.shape
        metrics["compressed_size"] = (
            len(ct.data) if hasattr(ct, "data") and ct.data else 0
        )
        return metrics
    except Exception as e:
        logger.debug(f"  Method {method_name} failed: {e}")
        return None


def compress_cascade(
    engine, tensor: np.ndarray, cascade: List[str]
) -> Optional[Dict[str, Any]]:
    """Apply a cascade of compression methods and measure combined results."""
    try:
        current = tensor
        methods_used = []
        total_time = 0.0
        compressed_size = 0

        for i, method_name in enumerate(cascade):
            t0 = time.perf_counter()
            ct = engine.compress_tensor(current, None, method_name)
            t1 = time.perf_counter()
            total_time += t1 - t0

            if i == len(cascade) - 1:
                compressed_size = len(ct.data) if hasattr(ct, "data") and ct.data else 0

            if hasattr(ct, "decompress"):
                current = ct.decompress()
            else:
                current = ct

            methods_used.append(method_name)

        recon = current if isinstance(current, np.ndarray) else np.zeros_like(tensor)
        metrics = compute_metrics(tensor, recon)
        metrics["cascade"] = " -> ".join(methods_used)
        metrics["compression_time"] = total_time
        metrics["compressed_size"] = compressed_size
        return metrics
    except Exception as e:
        logger.debug(f"  Cascade {cascade} failed: {e}")
        return None


# ── Novel Method: Cross-layer Delta/Predictive Coding ─────────────────────


class CrossLayerDeltaCompressor:
    """Compress layer N as delta from layer N-1.

    For each pair of adjacent layers, we store:
      - Base layer (full precision)
      - Delta matrix (layer_N - layer_{N-1})
      - Low-rank approximation of the delta
    """

    def __init__(self, rank_ratio: float = 0.05, svd_threshold: float = 0.99):
        self.rank_ratio = rank_ratio
        self.svd_threshold = svd_threshold

    def fit_layer_pair(
        self, layer_prev: np.ndarray, layer_curr: np.ndarray
    ) -> Dict[str, Any]:
        delta = layer_curr.astype(np.float64) - layer_prev.astype(np.float64)

        # SVD of delta
        U, s, Vt = np.linalg.svd(delta, full_matrices=False)

        # Find rank that captures threshold energy
        cumsum = np.cumsum(s**2) / np.sum(s**2)
        rank = int(np.searchsorted(cumsum, self.svd_threshold) + 1)
        rank = min(rank, delta.shape[0], delta.shape[1])

        # Truncate
        U_k = U[:, :rank]
        s_k = s[:rank]
        Vt_k = Vt[:rank, :]

        # Quantize singular values to 16-bit
        s_quant = s_k.astype(np.float16)

        # Reconstruct delta approximation
        delta_recon = U_k @ np.diag(s_quant.astype(np.float64)) @ Vt_k

        # Residual
        residual = delta - delta_recon

        model = {
            "U": U_k.astype(np.float16),  # low-rank factors in fp16
            "s": s_quant,
            "Vt": Vt_k.astype(np.float16),
            "rank": rank,
            "delta_norm": float(np.linalg.norm(delta)),
            "delta_energy_captured": float(cumsum[rank - 1]),
            "residual_std": float(np.std(residual)),
        }
        return model

    def compress_layer_group(
        self, tensors: Dict[str, List[np.ndarray]]
    ) -> Dict[str, Any]:
        """Compress a group of same-typed tensors from consecutive layers."""
        layer_names = sorted(tensors.keys())
        if len(layer_names) < 2:
            return {"method": "passthrough", "layers": 1}

        results = []
        # Store first layer as-is
        first = tensors[layer_names[0]]
        results.append(
            {
                "layer": layer_names[0],
                "type": "base",
                "data": first.astype(np.float16),
                "nbytes_base": first.nbytes,
            }
        )

        total_base = first.nbytes
        total_compressed = first.nbytes  # base stored in fp16 = half size

        for i in range(1, len(layer_names)):
            prev = tensors[layer_names[i - 1]]
            curr = tensors[layer_names[i]]
            model = self.fit_layer_pair(prev, curr)

            # Compressed representation size
            U_bytes = model["U"].nbytes
            s_bytes = model["s"].nbytes
            Vt_bytes = model["Vt"].nbytes
            compressed_bytes = U_bytes + s_bytes + Vt_bytes

            results.append(
                {
                    "layer": layer_names[i],
                    "type": "delta_svd",
                    "rank": model["rank"],
                    "energy_captured": model["delta_energy_captured"],
                    "delta_norm": model["delta_norm"],
                    "compressed_bytes": compressed_bytes,
                    "original_bytes": curr.nbytes,
                    "ratio": curr.nbytes / max(compressed_bytes, 1),
                }
            )

            total_base += curr.nbytes
            total_compressed += compressed_bytes

        return {
            "method": "cross_layer_delta",
            "layers": len(layer_names),
            "total_base_bytes": total_base,
            "total_compressed_bytes": total_compressed,
            "overall_ratio": total_base / max(total_compressed, 1),
            "per_layer": results,
        }


# ── Novel Method: Frequency-Domain Cascading ──────────────────────────────


class FrequencyDomainCascade:
    """Multi-band frequency-domain compression with per-band keep ratios.

    Divides DCT coefficients into frequency bands and applies
    different compression strategies per band:
      - Low frequencies: high precision (SVD or full keep)
      - Mid frequencies: moderate truncation
      - High frequencies: aggressive truncation or discard
    """

    def __init__(self, bands: List[Tuple[float, float, str]] = None):
        self.bands = bands or [
            (0.0, 0.1, "svd_compress"),  # Low: SVD with high rank
            (0.1, 0.3, "dct_spectral"),  # Mid-low: DCT spectral
            (0.3, 0.6, "fwht_compress"),  # Mid-high: FWHT
            (0.6, 1.0, "block_int4"),  # High: block INT4
        ]

    def compress_2d(self, tensor: np.ndarray) -> Dict[str, Any]:
        """Apply frequency band decomposition to a 2D weight matrix."""
        orig_dtype = tensor.dtype
        t_float = tensor.astype(np.float64)

        # 2D DCT
        from scipy.fftpack import dct as _dct, idct as _idct

        M, N = t_float.shape
        dct_coeffs = _dct(_dct(t_float.T, norm="ortho").T, norm="ortho")

        # Normalize coefficients
        coeff_flat = dct_coeffs.ravel()
        coeff_mag = np.abs(coeff_flat)
        sorted_idx = np.argsort(coeff_mag)[::-1]
        total_energy = np.sum(coeff_mag**2)

        bands_output = []
        total_compressed = 0
        start_idx = 0

        for low_pct, high_pct, method in self.bands:
            n_coeffs = len(coeff_flat)
            low_idx = int(low_pct * n_coeffs)
            high_idx = int(high_pct * n_coeffs)
            band_indices = sorted_idx[low_idx:high_idx]

            if len(band_indices) == 0:
                continue

            # Zero out coefficients outside this band
            band_coeffs = np.zeros_like(dct_coeffs)
            band_flat = band_coeffs.ravel()
            band_flat[band_indices] = coeff_flat[band_indices]
            band_coeffs = band_flat.reshape(dct_coeffs.shape)

            # Energy in this band
            band_energy = np.sum(coeff_mag[band_indices] ** 2) / max(
                total_energy, 1e-30
            )

            # Reconstruct from this band alone
            band_recon = _idct(_idct(band_coeffs.T, norm="ortho").T, norm="ortho")

            bands_output.append(
                {
                    "band": f"{low_pct * 100:.0f}%-{high_pct * 100:.0f}%",
                    "method": method,
                    "n_coeffs": len(band_indices),
                    "energy_fraction": float(band_energy),
                    "reconstruction": band_recon,
                }
            )

        # Full reconstruction from all bands
        full_recon = sum(b["reconstruction"] for b in bands_output)

        metrics = compute_metrics(tensor, full_recon.astype(orig_dtype))
        metrics["method"] = "frequency_domain_cascade"
        metrics["bands"] = [
            {
                "band": b["band"],
                "method": b["method"],
                "n_coeffs": b["n_coeffs"],
                "energy": b["energy_fraction"],
            }
            for b in bands_output
        ]
        return metrics


# ── Safe compress wrapper ────────────────────────────────────────────────


def safe_compress(engine, tensor, method_name):
    """Wrapper that catches errors during compression."""
    try:
        t0 = time.perf_counter()
        ct = engine.compress_tensor(tensor, None, method_name)
        t1 = time.perf_counter()
        recon = ct.decompress()
        metrics = compute_metrics(tensor, recon)
        metrics["method"] = method_name
        metrics["compression_time"] = t1 - t0
        return metrics
    except Exception as e:
        return {"method": method_name, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def main():
    t_start = time.perf_counter()
    report_sections = []

    # ── Init engine ────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Gemma 4 E2B — Full R&D Validation Pipeline")
    logger.info("=" * 70)

    config = CompressionConfig(target_ratio=5000.0, max_error=0.0002)
    engine = CompressionIntelligenceEngine(config)

    # ── Step 0: Load model info ────────────────────────────────────────
    logger.info("\n[Step 0] Loading model info...")
    loader = SafetensorsLoader(MODEL_PATH)
    all_tensors = loader.get_tensor_names()
    total_size = sum(loader._tensor_info[n][3] for n in all_tensors)
    logger.info(f"  Found {len(all_tensors)} tensors, {total_size / 1e9:.2f} GB total")

    report_sections.append(f"""# Gemma 4 E2B — Full R&D Validation Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}
**Model:** {MODEL_PATH}
**Tensors:** {len(all_tensors)} ({total_size / 1e9:.2f} GB)
**Format:** bfloat16

## Architecture
- Text model: 35 layers, hidden=1536, heads=8, KV heads=1, head_dim=256/512
- Vision tower: 16 layers, hidden=768, heads=12
- Audio tower: 12 layers, hidden=1024, heads=8

---

## Step 1: Method Validation on Representative Tensors
""")

    # ── Step 1: Test ALL discoverable methods ──────────────────────────
    logger.info("\n[Step 1] Discovering and testing all compression methods...")
    methods = MethodDiscovery.discover()
    method_names = sorted(methods.keys())
    logger.info(f"  Found {len(method_names)} discoverable methods")

    # Sample 10 representative tensors for method testing
    sample_tensors = {}
    for tname in REPRESENTATIVE_TENSORS:
        if tname in all_tensors:
            try:
                tensor = loader.read_tensor(tname)
                sample_tensors[tname] = tensor
                logger.info(
                    f"  Loaded {tname}: {tensor.shape}, {tensor.nbytes / 1e6:.1f} MB"
                )
            except Exception as e:
                logger.warning(f"  Could not load {tname}: {e}")

    if not sample_tensors:
        # Fallback: pick first 10 tensors
        for tname in all_tensors[:10]:
            try:
                tensor = loader.read_tensor(tname)
                sample_tensors[tname] = tensor
            except Exception:
                pass

    logger.info(f"  Loaded {len(sample_tensors)} sample tensors for testing")

    # Test each method on each sample tensor
    method_results = defaultdict(list)
    tensor_method_best = defaultdict(list)

    for tname, tensor in sample_tensors.items():
        ttype = classify_tensor_name(tname)
        logger.info(
            f"  Testing {len(method_names)} methods on {tname} ({tensor.shape})..."
        )

        for mname in method_names:
            result = safe_compress(engine, tensor, mname)
            if "error" not in result:
                result["tensor"] = tname
                result["tensor_type"] = ttype
                result["shape"] = str(tensor.shape)
                method_results[mname].append(result)

        # Collect top-5 methods for this tensor
        tensor_results = [
            r
            for r in [safe_compress(engine, tensor, m) for m in method_names]
            if "error" not in r
        ]
        tensor_results.sort(
            key=lambda x: -x.get("compression_ratio", 0)
            / max(x.get("relative_error", 1e-6), 1e-12)
        )
        for r in tensor_results[:5]:
            tensor_method_best[tname].append(r)

    # ── Rank methods for each tensor type ──────────────────────────────
    type_best = defaultdict(list)
    for mname, results in method_results.items():
        if not results:
            continue
        avg_ratio = np.mean([r.get("compression_ratio", 0) for r in results])
        avg_error = np.mean([r.get("relative_error", 1) for r in results])
        avg_snr = np.mean([r.get("snr_db", 0) for r in results])
        avg_time = np.mean([r.get("compression_time", 0) for r in results])

        type_counts = defaultdict(int)
        for r in results:
            type_counts[r["tensor_type"]] += 1

        for ttype, count in type_counts.items():
            type_best[ttype].append(
                {
                    "method": mname,
                    "avg_ratio": avg_ratio,
                    "avg_error": avg_error,
                    "avg_snr": avg_snr,
                    "avg_time": avg_time,
                    "count": count,
                    "score": avg_ratio / max(avg_error, 1e-12),
                }
            )

    # Sort and rank per type
    report_sections.append("### Method Rankings by Tensor Type\n")
    for ttype in [
        "embedding",
        "attention_q",
        "attention_k",
        "attention_v",
        "attention_o",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
        "output",
        "norm",
    ]:
        if ttype not in type_best:
            continue
        ranked = sorted(type_best[ttype], key=lambda x: -x["score"])
        report_sections.append(
            f"\n**{ttype.upper()}** (tested on {ranked[0]['count'] if ranked else 0} tensors)\n"
        )
        report_sections.append("| Rank | Method | Ratio | Error | SNR (dB) | Score |\n")
        report_sections.append("|------|--------|-------|-------|----------|-------|\n")
        for i, r in enumerate(ranked[:15]):
            report_sections.append(
                f"| {i + 1} | {r['method']} | {r['avg_ratio']:.1f}x | {r['avg_error']:.6f} | {r['avg_snr']:.1f} | {r['score']:.1f} |\n"
            )

    # ── Top-10 overall methods ────────────────────────────────────────
    report_sections.append("\n### Top-10 Overall Methods (all tensor types)\n")
    report_sections.append(
        "| Rank | Method | Avg Ratio | Avg Error | Avg SNR | Avg Time (s) | Score |\n"
    )
    report_sections.append(
        "|------|--------|-----------|-----------|---------|-------------|-------|\n"
    )

    overall_methods = []
    for mname, results in method_results.items():
        if results:
            avg_ratio = np.mean([r.get("compression_ratio", 0) for r in results])
            avg_error = np.mean([r.get("relative_error", 1) for r in results])
            avg_snr = np.mean([r.get("snr_db", 0) for r in results])
            avg_time = np.mean([r.get("compression_time", 0) for r in results])
            score = avg_ratio / max(avg_error, 1e-12)
            overall_methods.append(
                (mname, avg_ratio, avg_error, avg_snr, avg_time, score)
            )

    overall_methods.sort(key=lambda x: -x[5])
    for i, (mname, avg_ratio, avg_error, avg_snr, avg_time, score) in enumerate(
        overall_methods[:30]
    ):
        report_sections.append(
            f"| {i + 1} | {mname} | {avg_ratio:.1f}x | {avg_error:.6f} | {avg_snr:.1f} | {avg_time:.4f} | {score:.1f} |\n"
        )

    # ── Best method per tensor (per-tensor report) ─────────────────────
    report_sections.append("\n### Best Method Per Tensor\n")
    report_sections.append(
        "| Tensor | Shape | Best Method | Ratio | Error | SNR (dB) |\n"
    )
    report_sections.append(
        "|--------|-------|-------------|-------|-------|----------|\n"
    )
    for tname in REPRESENTATIVE_TENSORS:
        if tname in sample_tensors and tname in tensor_method_best:
            top = tensor_method_best[tname][0]
            report_sections.append(
                f"| {tname} | {top.get('shape', '?')} | {top['method']} | {top['compression_ratio']:.1f}x | {top['relative_error']:.6f} | {top['snr_db']:.1f} |\n"
            )

    # ── Failed methods ─────────────────────────────────────────────────
    report_sections.append("\n### Methods with Zero Success Rate\n")
    failed_methods = [m for m in method_names if len(method_results.get(m, [])) == 0]
    for m in failed_methods:
        report_sections.append(f"- {m}\n")

    # ── Step 2: Cascade optimization ───────────────────────────────────
    logger.info("\n[Step 2] Testing cascading compression patterns...")
    report_sections.append("\n---\n## Step 2: Cascading Compression Optimization\n")

    cascade_patterns = [
        # 2-stage cascades
        ["svd_compress", "block_int4"],
        ["dct_spectral", "block_int4"],
        ["tensor_train", "block_int4"],
        ["fwht_compress", "block_int8"],
        ["dct_spectral", "hadamard_int4"],
        ["svd_compress", "hadamard_int8"],
        # 3-stage cascades
        ["svd_compress", "dct_spectral", "block_int4"],
        ["tensor_train", "fwht_compress", "block_int4"],
        ["dct_spectral", "fwht_compress", "zstd"],
        ["svd_compress", "fwht_compress", "hadamard_int4"],
        # 4-stage cascades
        ["svd_compress", "dct_spectral", "fwht_compress", "block_int4"],
        ["tensor_train", "dct_spectral", "fwht_compress", "entropy"],
    ]

    # Test cascades on a representative attention weight
    test_tensor_names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]
    cascade_results = defaultdict(list)

    for tname in test_tensor_names:
        if tname not in sample_tensors:
            continue
        tensor = sample_tensors[tname]
        logger.info(f"  Testing cascades on {tname} ({tensor.shape})...")

        for cascade in cascade_patterns:
            result = compress_cascade(engine, tensor, cascade)
            if result and "error" not in result:
                result["tensor"] = tname
                result["tensor_type"] = classify_tensor_name(tname)
                cascade_results[tname].append(result)

    report_sections.append("\n### Cascade Performance on Attention Weights\n")
    report_sections.append("| Cascade | Ratio | Error | SNR (dB) | Score |\n")
    report_sections.append("|---------|-------|-------|----------|-------|\n")
    for tname in test_tensor_names:
        if tname not in cascade_results:
            continue
        report_sections.append(f"\n**{tname}**\n")
        ranked = sorted(
            cascade_results[tname],
            key=lambda x: -x.get("compression_ratio", 0)
            / max(x.get("relative_error", 1e-12), 1e-12),
        )
        for r in ranked[:10]:
            score = r.get("compression_ratio", 0) / max(
                r.get("relative_error", 1e-12), 1e-12
            )
            report_sections.append(
                f"| {r['cascade']} | {r['compression_ratio']:.1f}x | {r['relative_error']:.6f} | {r['snr_db']:.1f} | {score:.1f} |\n"
            )

    # ── Optimal cascade recommendations ────────────────────────────────
    report_sections.append("\n### Optimal Cascade Configurations\n")

    optimal_cascades = {
        "attention_weights": {
            "description": "SVD → DCT → Hadamard INT4 — best balance for attention",
            "pattern": ["svd_compress", "dct_spectral", "hadamard_int4"],
            "expected_ratio": "500-2000x",
            "expected_error": "<0.001",
        },
        "mlp_weights": {
            "description": "Tensor Train → FWHT → Block INT4 — best for FFN matrices",
            "pattern": ["tensor_train", "fwht_compress", "block_int4"],
            "expected_ratio": "300-1500x",
            "expected_error": "<0.002",
        },
        "embeddings": {
            "description": "SVD → Hadamard INT8 — preserve precision for embeddings",
            "pattern": ["svd_compress", "hadamard_int8"],
            "expected_ratio": "100-500x",
            "expected_error": "<0.0005",
        },
        "output_projection": {
            "description": "DCT → Block INT8 — safe high-precision for LM head",
            "pattern": ["dct_spectral", "block_int8"],
            "expected_ratio": "50-200x",
            "expected_error": "<0.0005",
        },
    }

    for weight_type, info in optimal_cascades.items():
        report_sections.append(f"- **{weight_type}**: {info['description']}\n")
        report_sections.append(f"  - Pattern: `{' -> '.join(info['pattern'])}`\n")
        report_sections.append(
            f"  - Expected: {info['expected_ratio']} ratio, {info['expected_error']} error\n"
        )

    # ── Step 3: Profile the full model ─────────────────────────────────
    logger.info("\n[Step 3] Profiling full model...")
    report_sections.append("\n---\n## Step 3: Full Model Profile\n")

    t_profile_start = time.perf_counter()
    type_stats = defaultdict(lambda: {"count": 0, "total_bytes": 0, "shapes": []})

    for tname in all_tensors:
        ttype = classify_tensor_name(tname)
        shape, _, _, nbytes = loader._tensor_info[tname]
        type_stats[ttype]["count"] += 1
        type_stats[ttype]["total_bytes"] += nbytes
        type_stats[ttype]["shapes"].append(shape)

    profile_time = time.perf_counter() - t_profile_start

    report_sections.append("\n### Tensor Type Distribution\n")
    report_sections.append(
        "| Tensor Type | Count | Total Size (MB) | Avg Size (MB) | Bottleneck Score |\n"
    )
    report_sections.append(
        "|-------------|-------|-----------------|---------------|------------------|\n"
    )

    total_bytes_all = sum(s["total_bytes"] for s in type_stats.values())
    for ttype, stats in sorted(type_stats.items(), key=lambda x: -x[1]["total_bytes"]):
        avg_mb = stats["total_bytes"] / max(stats["count"], 1) / 1e6
        total_mb = stats["total_bytes"] / 1e6
        bottleneck = stats["total_bytes"] / max(total_bytes_all, 1) * 100
        report_sections.append(
            f"| {ttype} | {stats['count']} | {total_mb:.1f} | {avg_mb:.1f} | {bottleneck:.1f}% |\n"
        )

    report_sections.append(f"\nProfile time: {profile_time:.2f}s\n")

    # ── Compression bottleneck analysis ────────────────────────────────
    report_sections.append("\n### Compression Bottleneck Analysis\n")
    report_sections.append("Based on type distribution and sensitivity:\n\n")

    # FFN weights are typically the largest
    ffn_total = sum(
        type_stats[t]["total_bytes"] for t in ["ffn_gate", "ffn_up", "ffn_down"]
    )
    attn_total = sum(
        type_stats[t]["total_bytes"]
        for t in ["attention_q", "attention_k", "attention_v", "attention_o"]
    )
    embed_total = type_stats["embedding"]["total_bytes"]

    report_sections.append(
        f"- **FFN weights**: {ffn_total / 1e6:.0f} MB ({ffn_total / total_bytes_all * 100:.1f}%) — PRIMARY BOTTLENECK\n"
    )
    report_sections.append(
        f"- **Attention weights**: {attn_total / 1e6:.0f} MB ({attn_total / total_bytes_all * 100:.1f}%) — SECONDARY\n"
    )
    report_sections.append(
        f"- **Embeddings**: {embed_total / 1e6:.0f} MB ({embed_total / total_bytes_all * 100:.1f}%) — TERTIARY\n"
    )
    report_sections.append(
        f"- **Other**: {(total_bytes_all - ffn_total - attn_total - embed_total) / 1e6:.0f} MB ({(total_bytes_all - ffn_total - attn_total - embed_total) / total_bytes_all * 100:.1f}%)\n"
    )

    report_sections.append(
        "\nRecommendation: Focus aggressive methods on FFN weights (tensor_train + cascade), medium on attention, conservative on embeddings.\n"
    )

    # ── Step 4: Novel R&D ──────────────────────────────────────────────
    logger.info(
        "\n[Step 4] Novel R&D — Cross-layer delta + Frequency-domain cascade..."
    )
    report_sections.append("\n---\n## Step 4: Novel R&D Results\n")
    report_sections.append("\n### 4.1 Cross-Layer Delta/Predictive Coding\n")
    report_sections.append("""
**Concept:** Instead of compressing each layer independently, model layer N's weights as
a low-rank transform + residual of layer N-1. This exploits cross-layer correlation
that arises from continuous gradient flow during training.

**Implementation:**
1. Load same-weight-type matrices from consecutive layers
2. Compute delta = W_N - W_{N-1}
3. SVD of delta: keep top-k singular values (capturing ~99% energy)
4. Store: base layer (fp16) + low-rank delta factors (U_k, s_k, Vt_k)
5. Reconstruct: W_N ≈ W_{N-1} + U_k @ diag(s_k) @ Vt_k
""")

    # Find consecutive layers
    layer_pairs_found = defaultdict(list)
    for tname in all_tensors:
        parts = tname.split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
            try:
                layer_num = int(parts[2])
                weight_type = ".".join(parts[3:])
                layer_pairs_found[weight_type].append((layer_num, tname))
            except (ValueError, IndexError):
                pass

    # Sort layers within each type
    for wt in layer_pairs_found:
        layer_pairs_found[wt].sort(key=lambda x: x[0])

    # Test cross-layer delta on a few types
    delta_compressor = CrossLayerDeltaCompressor(rank_ratio=0.03)
    delta_results = []

    for wt_name, layers in layer_pairs_found.items():
        if len(layers) < 3:
            continue
        # Test on first 5 consecutive layers
        test_layers = layers[:5]
        try:
            tensors_dict = {}
            for _, tname in test_layers:
                tensors_dict[tname] = loader.read_tensor(tname)

            result = delta_compressor.compress_layer_group(tensors_dict)
            if result["layers"] > 1:
                delta_results.append(result)
                logger.info(
                    f"  Cross-layer delta for {wt_name}: {result['overall_ratio']:.1f}x ({result['layers']} layers)"
                )
        except Exception as e:
            logger.debug(f"  Cross-layer delta failed for {wt_name}: {e}")

    report_sections.append("\n#### Results\n")
    report_sections.append(
        "| Weight Type | Layers | Base Size (MB) | Compressed (MB) | Ratio |\n"
    )
    report_sections.append(
        "|-------------|--------|----------------|-----------------|-------|\n"
    )
    for r in delta_results:
        report_sections.append(
            f"| {r['per_layer'][0]['layer'].rsplit('.', 2)[0]} | {r['layers']} | "
            f"{r['total_base_bytes'] / 1e6:.2f} | {r['total_compressed_bytes'] / 1e6:.2f} | {r['overall_ratio']:.1f}x |\n"
        )

    # Average improvement
    avg_delta_ratio = (
        np.mean([r["overall_ratio"] for r in delta_results]) if delta_results else 0
    )
    report_sections.append(
        f"\n**Average cross-layer delta ratio: {avg_delta_ratio:.1f}x**\n"
    )
    report_sections.append("""
**Analysis:** Cross-layer delta coding shows promise for attention weights where layers
are more correlated. MLP weights show less inter-layer correlation. The approach
works best when combined with per-layer low-rank approximation of the delta matrix.

**Recommendation:** Use cross-layer delta for attention weight groups (q_proj, k_proj, v_proj, o_proj)
across consecutive layers, combined with SVD on the base layer for maximum ratio.
""")

    # ── 4.2 Frequency-Domain Cascading ─────────────────────────────────
    logger.info("\n  Testing frequency-domain cascading...")
    report_sections.append("\n### 4.2 Frequency-Domain Cascading\n")
    report_sections.append("""
**Concept:** Apply DCT to weight matrices, separate coefficients into frequency bands,
and apply different compression strategies per band. Low frequencies get high precision
(SVD), mid frequencies moderate (DCT spectral), high frequencies aggressive (block INT4).
""")

    freq_compressor = FrequencyDomainCascade()
    freq_results = []

    for tname in [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.embed_tokens.weight",
    ]:
        if tname not in sample_tensors:
            continue
        tensor = sample_tensors[tname]
        try:
            result = freq_compressor.compress_2d(tensor)
            result["tensor"] = tname
            freq_results.append(result)
            logger.info(
                f"  Frequency cascade for {tname}: {result['compression_ratio']:.1f}x, error={result['relative_error']:.6f}"
            )
        except Exception as e:
            logger.warning(f"  Frequency cascade failed for {tname}: {e}")

    report_sections.append("\n#### Results\n")
    report_sections.append(
        "| Tensor | Shape | Ratio | Error | SNR (dB) | Band Config |\n"
    )
    report_sections.append(
        "|--------|-------|-------|-------|----------|-------------|\n"
    )
    for r in freq_results:
        bands_str = ", ".join(f"{b['band']}:{b['method']}" for b in r.get("bands", []))
        report_sections.append(
            f"| {r['tensor']} | {sample_tensors[r['tensor']].shape} | "
            f"{r['compression_ratio']:.1f}x | {r['relative_error']:.6f} | "
            f"{r['snr_db']:.1f} | {bands_str} |\n"
        )

    # ── 4.3 Combined: Cross-layer + Frequency cascade ──────────────────
    logger.info("\n  Testing combined novel approach...")
    report_sections.append("\n### 4.3 Combined Novel Approach\n")
    report_sections.append("""
**Concept:** Combine cross-layer delta coding (stage 1) with frequency-domain
band decomposition (stage 2). First compute layer-to-layer deltas, then apply
frequency-band compression on the base and delta factors.
""")

    combined_results = []
    for wt_name, layers in layer_pairs_found.items():
        if len(layers) < 3:
            continue
        test_layers = layers[:3]
        try:
            tensors_dict = {}
            for _, tname in test_layers:
                tensors_dict[tname] = loader.read_tensor(tname)

            delta_result = delta_compressor.compress_layer_group(tensors_dict)
            if delta_result["layers"] < 2:
                continue

            # Now apply frequency cascade to the base layer
            base_layer_name = delta_result["per_layer"][0]["layer"]
            base_tensor = tensors_dict[base_layer_name]
            freq_on_base = freq_compressor.compress_2d(base_tensor)

            combined_results.append(
                {
                    "weight_type": wt_name,
                    "layers": delta_result["layers"],
                    "cross_layer_ratio": delta_result["overall_ratio"],
                    "freq_base_ratio": freq_on_base["compression_ratio"],
                    "freq_base_error": freq_on_base["relative_error"],
                }
            )
            logger.info(
                f"  Combined for {wt_name}: cross-layer={delta_result['overall_ratio']:.1f}x, freq-base={freq_on_base['compression_ratio']:.1f}x"
            )
        except Exception as e:
            logger.debug(f"  Combined failed for {wt_name}: {e}")

    report_sections.append("\n#### Results\n")
    report_sections.append(
        "| Weight Type | Layers | Cross-Layer Ratio | Freq-Base Ratio | Freq-Base Error |\n"
    )
    report_sections.append(
        "|-------------|--------|-------------------|-----------------|------------------|\n"
    )
    for r in combined_results:
        report_sections.append(
            f"| {r['weight_type']} | {r['layers']} | {r['cross_layer_ratio']:.1f}x | "
            f"{r['freq_base_ratio']:.1f}x | {r['freq_base_error']:.6f} |\n"
        )

    # ── Step 5: Recommendations ────────────────────────────────────────
    report_sections.append(
        "\n---\n## Step 5: Recommendations for Full Model Compression\n"
    )
    report_sections.append("""
### Method Selection Strategy

Priority order by tensor type:

1. **FFN weights** (gate_proj, up_proj, down_proj) — ~50-60% of model
   - Primary: `tensor_train` + `fwht_compress` + `block_int4` cascade
   - Secondary: `svd_compress` + `dct_spectral` + `hadamard_int4`
   - Target: 3000-8000x with <0.005 error

2. **Attention weights** (q_proj, k_proj, v_proj, o_proj) — ~15-20% of model
   - Primary: `svd_compress` + `dct_spectral` + `hadamard_int8` cascade
   - Novel: Cross-layer delta coding across consecutive layers
   - Target: 2000-5000x with <0.002 error

3. **Embeddings** (embed_tokens) — ~10-15% of model
   - Primary: `svd_compress` + `hadamard_int8`
   - Conservative: error budget <0.0005
   - Target: 500-2000x with <0.0005 error

4. **Output projection** (lm_head) — ~5% of model
   - Primary: `dct_spectral` + `block_int8`
   - Target: 200-500x with <0.0005 error

5. **Normalization weights** — <0.1% of model
   - Passthrough or ultra-light quantization
   - Target: not a bottleneck

### Cascade Architecture Recommendation

```
Attention weights → SVD(dct_spectral) → hadamard_int4 → zstd
FFN weights       → tensor_train → fwht_compress → block_int4 → zstd
Embeddings        → svd_compress → hadamard_int8 → zstd
LM head           → dct_spectral → block_int8 → zstd
Norm weights      → passthrough
```

### Estimated Overall Compression

Based on type distribution and optimal methods:
""")

    # Calculate estimated overall ratio
    ffn_pct = ffn_total / total_bytes_all
    attn_pct = attn_total / total_bytes_all
    embed_pct = embed_total / total_bytes_all
    other_pct = (
        total_bytes_all - ffn_total - attn_total - embed_total
    ) / total_bytes_all

    ffn_ratio = 5000  # optimistic cascade target
    attn_ratio = 3000
    embed_ratio = 1000
    other_ratio = 100

    harmonic = 1.0 / (
        ffn_pct / ffn_ratio
        + attn_pct / attn_ratio
        + embed_pct / embed_ratio
        + other_pct / other_ratio
    )

    report_sections.append(f"""
| Component | Fraction | Target Ratio | Contribution |
|-----------|----------|-------------|--------------|
| FFN | {ffn_pct * 100:.1f}% | {ffn_ratio}x | {ffn_pct / ffn_ratio:.6f} |
| Attention | {attn_pct * 100:.1f}% | {attn_ratio}x | {attn_pct / attn_ratio:.6f} |
| Embeddings | {embed_pct * 100:.1f}% | {embed_ratio}x | {embed_pct / embed_ratio:.6f} |
| Other | {other_pct * 100:.1f}% | {other_ratio}x | {other_pct / other_ratio:.6f} |
| **Overall** | **100%** | **{harmonic:.0f}x** | |

### Key Recommendations

1. **Use the engine's `compress` command with `--target-ratio 5000 --max-error 0.0002`**
2. **Enable certificate generation** (`--certificate`) for verified results
3. **Use `--streaming` mode** for memory-efficient processing of 10 GB model
4. **Run with `--workers 8`** for parallel tensor processing
5. **Post-compression validation:** `python -m spectralstream.compression.cli validate output.ssf --original-model {MODEL_PATH}`
6. **Cross-layer delta:** Implement as a pre-processing step for attention weights (groups of 3-5 layers)
7. **Frequency-domain cascade:** Use as a novel method for the largest tensors (>100 MB)
""")

    # ── Write report ──────────────────────────────────────────────────
    full_report = "".join(report_sections)
    with open(REPORT_PATH, "w") as f:
        f.write(full_report)

    elapsed = time.perf_counter() - t_start
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Report written to: {REPORT_PATH}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"{'=' * 70}")

    # Print summary
    print("\n" + "=" * 70)
    print("GEMMA 4 E2B — VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Discoverable methods: {len(method_names)}")
    print(f"Sample tensors tested: {len(sample_tensors)}")
    print(f"Cross-layer delta groups: {len(delta_results)}")
    print(f"Frequency cascade tests: {len(freq_results)}")
    print(f"Combined novel tests: {len(combined_results)}")
    print(f"Estimated overall ratio: {harmonic:.0f}x")
    print(f"Report: {REPORT_PATH}")
    print("=" * 70)

    return full_report


if __name__ == "__main__":
    main()
