"""
True Method Stacking Engine — combines complementary methods on the ORIGINAL tensor.

Instead of chaining lossy methods on residuals (which fails when the residual
is noise-like after SVD), this engine:

1. Identifies complementary method pairs from the engine's ``_methods`` registry
2. Tests each pair independently on the ORIGINAL tensor (not on residuals)
3. Selects the pair that gives the best ratio/error tradeoff
4. Packages the compressed data from both methods

**Key Insight**: SVD at rank=2 gives 877-1228× on real weights, but the residual
after SVD reconstruction is noise. DCT/FWHT on noise doesn't compress.

**What Works**: Instead of chaining lossy methods on residuals, stack methods
on the ORIGINAL data in different domains:

- SVD (low-rank decomposition) + BlockINT8 (quantization): SVD captures global
  low-rank structure, INT8 captures the detail with block-wise scaling.
- DCT (frequency domain) + BlockINT8 (spatial quantization): Different domains
  extract complementary information.
- SVD + HadamardINT8: SVD captures SVD subspace, Hadamard + INT8 captures the
  Hadamard-domain representation of the original.

**Complementary Pairs**:
- ``svd_compress`` + ``block_int8`` — decomposition + quantization
- ``dct_spectral`` + ``block_int8`` — spectral + quantization
- ``svd_compress`` + ``hadamard_int8`` — decomposition + transform-quant
- ``tensor_train`` + ``fwht_compress`` — tensor network + spectral
- ``svd_compress`` + ``block_int4`` — decomposition + aggressive quantization

Example
-------
>>> from spectralstream.compression.engine import CompressionIntelligenceEngine
>>> from spectralstream.compression.engine.stacking_engine import MethodStackingEngine
>>> import numpy as np
>>> eng = CompressionIntelligenceEngine()
>>> mse = MethodStackingEngine(eng)
>>> # Test on low-rank data (simulates real weights)
>>> U = np.random.randn(256, 20)
>>> V = np.random.randn(20, 512)
>>> t = (U @ V).astype(np.float32)
>>> result = mse.find_best_stacking(t, max_error=0.01)
>>> if result:
>>>     ratio, error, stages, recon = result
>>>     print(f'Best stacking: ratio={ratio:.1f}x, error={error:.6f}')
"""

from __future__ import annotations

import gc as _gc
import json
import logging
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .loss_metrics import TensorLossMetrics

logger = logging.getLogger(__name__)


class MethodStackingEngine:
    """True method stacking — combines methods in complementary ways.

    Unlike residual cascading (which fails when the residual is noise),
    this engine tests each complementary method pair independently on the
    ORIGINAL tensor and selects the pair with the best ratio/error tradeoff.

    Parameters
    ----------
    engine : CompressionIntelligenceEngine
        Engine with ``_methods`` registry containing compression method instances.

    Notes
    -----
    **Complementary Pair Strategy**:

    Each method in a pair operates on the ORIGINAL tensor independently.
    The compressed outputs from both methods are concatenated into a single
    payload.  On decompression, each method reconstructs its own version of
    the tensor and they are combined via averaging or weighted blending.

    This is fundamentally different from residual cascading:
    - Residual cascade: SVD(original) → DCT(original - SVD_recon)
      Problem: residual after SVD is noise, DCT on noise doesn't compress.
    - True stacking: SVD(original) + BlockINT8(original)
      Both methods capture different structure in the SAME signal.

    **Combining Reconstructions**:
    - ``svd_compress`` + ``block_int8``: The SVD captures global low-rank
      structure; BlockINT8 captures local block-level detail.  Reconstruction
      averages both: ``0.5 * svd_recon + 0.5 * block_int8_recon``.
    - ``dct_spectral`` + ``block_int8``: DCT captures frequency-domain
      structure; BlockINT8 captures spatial-domain detail.  Reconstruction
      blends both.
    """

    # ── Complementary Method Pairs ─────────────────────────────────────
    # Each entry is (method1_name, method2_name, blend_weight1)
    # blend_weight1 controls how reconstructions are combined:
    #   recon = w1 * recon1 + (1 - w1) * recon2
    # Default 0.5 = equal blend.
    COMPLEMENTARY_PAIRS: List[Tuple[str, str, float]] = [
        # Decomposition + Quantization: SVD captures low-rank, INT8 captures detail
        ("svd_compress", "block_int8", 0.5),
        # Spectral + Quantization: DCT captures frequency, INT8 captures spatial
        ("dct_spectral", "block_int8", 0.5),
        # Decomposition + Transform-Quant: SVD + Hadamard-INT8
        ("svd_compress", "hadamard_int8", 0.5),
        # Spectral + Quantization (4-bit): aggressive
        ("dct_spectral", "block_int4", 0.5),
        # Tensor network + Spectral: TT + FWHT
        ("tensor_train", "fwht_compress", 0.5),
        # Decomposition + Aggressive Quantization
        ("svd_compress", "block_int4", 0.6),  # Weight SVD more (less error)
        # Decomposition + Spectral
        ("svd_compress", "dct_spectral", 0.6),
        # Dual spectral: DCT + FWHT (different transforms)
        ("dct_spectral", "fwht_compress", 0.5),
        # Tensor train + Quantization
        ("tensor_train", "block_int8", 0.5),
        # SVD + Sparsity INT4
        ("svd_compress", "sparsity_int4", 0.5),
    ]

    # ── Single Methods to Test as Baselines ────────────────────────────
    BASELINE_METHODS: List[str] = [
        "svd_compress",
        "dct_spectral",
        "tensor_train",
        "fwht_compress",
        "block_int8",
        "hadamard_int8",
        "block_int4",
    ]

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        # Filter complementary pairs to only those available in the engine
        self._available_pairs: List[Tuple[str, str, float]] = []
        for m1, m2, w in self.COMPLEMENTARY_PAIRS:
            if m1 in engine._methods and m2 in engine._methods:
                self._available_pairs.append((m1, m2, w))

        self._available_baselines: List[str] = [
            m for m in self.BASELINE_METHODS if m in engine._methods
        ]

        logger.debug(
            "MethodStackingEngine: %d available pairs, %d baselines",
            len(self._available_pairs),
            len(self._available_baselines),
        )

    # ── Public API ──────────────────────────────────────────────────────

    def find_best_stacking(
        self,
        tensor: np.ndarray,
        max_error: float = 0.01,
        min_ratio: float = 1.5,
    ) -> Optional[
        Tuple[float, float, List[Tuple[str, bytes, Dict[str, Any]]], np.ndarray]
    ]:
        """Find the best method or pair of methods for this tensor.

        Tests each baseline method individually, then each complementary pair.
        Returns the best result — either a single method or a stacked pair —
        that satisfies the error budget.

        Parameters
        ----------
        tensor : np.ndarray
            Float32 tensor to compress.
        max_error : float
            Maximum allowed mean absolute error (default 0.01).
        min_ratio : float
            Minimum compression ratio to consider (default 1.5).

        Returns
        -------
        tuple or None
            ``(ratio, error, stages, reconstruction)`` where:
            - ``ratio``: Compression ratio (original_size / compressed_size)
            - ``error``: Mean absolute reconstruction error
            - ``stages``: List of ``(method_name, compressed_data, metadata)`` tuples
            - ``reconstruction``: Reconstructed float32 tensor

            Returns None if no method produces a valid result.
        """
        results: List[
            Tuple[float, float, List[Tuple[str, bytes, Dict[str, Any]]], np.ndarray]
        ] = []

        orig_size = tensor.nbytes
        original = np.ascontiguousarray(tensor, dtype=np.float32)

        # ── Test baseline methods individually ─────────────────────────
        for method_name in self._available_baselines:
            try:
                inst = self.engine._methods[method_name]
                data, meta = inst.compress(original)
                recon = inst.decompress(data, meta)
                if recon.shape != original.shape:
                    recon = recon.reshape(original.shape)

                ratio = float(orig_size / max(len(data), 1))
                error = float(
                    np.abs(
                        original.astype(np.float64) - recon.astype(np.float64)
                    ).mean()
                )

                if ratio >= min_ratio:
                    results.append(
                        (
                            ratio,
                            error,
                            [(method_name, data, meta)],
                            recon.astype(np.float32),
                        )
                    )
            except Exception as exc:
                logger.debug("Baseline '%s' failed: %s", method_name, exc)
                continue

        # ── Test complementary pairs on the ORIGINAL tensor ────────────
        for m1_name, m2_name, w1 in self._available_pairs:
            try:
                inst1 = self.engine._methods[m1_name]
                inst2 = self.engine._methods[m2_name]

                # Method 1 on original
                d1, m1 = inst1.compress(original)
                # Method 2 on original
                d2, m2 = inst2.compress(original)

                # Total compressed size = d1 + d2 (truly additive)
                total_bytes = len(d1) + len(d2)
                total_ratio = float(orig_size / max(total_bytes, 1))

                if total_ratio < min_ratio:
                    continue

                # Decompress both independently
                recon1 = inst1.decompress(d1, m1)
                if recon1.shape != original.shape:
                    recon1 = recon1.reshape(original.shape)

                recon2 = inst2.decompress(d2, m2)
                if recon2.shape != original.shape:
                    recon2 = recon2.reshape(original.shape)

                # Blend reconstructions: w1 * recon1 + w2 * recon2
                w2 = 1.0 - w1
                blended = (
                    w1 * recon1.astype(np.float64) + w2 * recon2.astype(np.float64)
                ).astype(np.float32)

                total_error = float(
                    np.abs(
                        original.astype(np.float64) - blended.astype(np.float64)
                    ).mean()
                )

                results.append(
                    (
                        total_ratio,
                        total_error,
                        [(m1_name, d1, m1), (m2_name, d2, m2)],
                        blended,
                    )
                )

            except Exception as exc:
                logger.debug("Pair (%s, %s) failed: %s", m1_name, m2_name, exc)
                continue

        # ── Select best result ─────────────────────────────────────────
        if not results:
            return None

        # Filter by error budget first
        valid = [
            (r, e, stages, recon) for r, e, stages, recon in results if e <= max_error
        ]
        if valid:
            # Within budget: pick highest ratio
            return max(valid, key=lambda x: x[0])

        # Outside budget: pick the one with lowest error
        return min(results, key=lambda x: x[1])

    # ── Packaged Output ─────────────────────────────────────────────────

    def package_stacked(
        self,
        stages: List[Tuple[str, bytes, Dict[str, Any]]],
        orig_size: int,
        blend_weights: Optional[List[float]] = None,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Package multiple compressed stages into a single payload.

        Format
        ------
        - uint32: number of stages (N)
        - For each stage:
          - uint32: method name length (L)
          - bytes: method name (UTF-8)
          - uint32: JSON metadata length (M)
          - bytes: JSON metadata
          - uint32: compressed data length (D)
          - bytes: compressed data
        - uint32: number of blend weights (0 or N)
        - For each weight:
          - float32: blend weight

        Parameters
        ----------
        stages : list of (method_name, data, metadata)
            Compressed stages to package.
        orig_size : int
            Original tensor size in bytes (for ratio calculation).
        blend_weights : list of float, optional
            Blend weights for reconstruction combination.
            If None, equal weights (1/N) are assumed.

        Returns
        -------
        Tuple[bytes, dict]
            ``(compressed_data, metadata)`` where metadata contains total_ratio,
            n_stages, methods, and blend_weights.
        """
        buf = bytearray()
        buf += struct.pack("<I", len(stages))

        for method_name, comp_data, meta in stages:
            method_bytes = method_name.encode("utf-8")
            meta_clean = {
                k: v
                for k, v in meta.items()
                if isinstance(v, (str, int, float, bool, list, tuple))
            }
            meta_json = json.dumps(meta_clean, default=str).encode("utf-8")

            buf += struct.pack("<I", len(method_bytes))
            buf += method_bytes
            buf += struct.pack("<I", len(meta_json))
            buf += meta_json
            buf += struct.pack("<I", len(comp_data))
            buf += comp_data

        # Blend weights
        if blend_weights and len(blend_weights) == len(stages):
            buf += struct.pack("<I", len(blend_weights))
            for w in blend_weights:
                buf += struct.pack("<f", w)
        else:
            # No custom weights — equal blend assumed on decode
            buf += struct.pack("<I", 0)

        total_compressed = len(buf)
        total_ratio = float(orig_size / max(total_compressed, 1))

        metadata: Dict[str, Any] = {
            "method": "stacked",
            "n_stages": len(stages),
            "methods": [s[0] for s in stages],
            "total_ratio": total_ratio,
            "blend_weights": blend_weights or [],
        }

        return bytes(buf), metadata

    @staticmethod
    def unpack_stacked(
        data: bytes,
    ) -> Tuple[List[Tuple[str, bytes, Dict[str, Any]]], List[float]]:
        """Unpack stages from a payload created by ``package_stacked``.

        Parameters
        ----------
        data : bytes
            Packaged payload.

        Returns
        -------
        Tuple[list, list]
            ``(stages, blend_weights)`` where each stage is
            ``(method_name, compressed_data, metadata)``.
        """
        stages: List[Tuple[str, bytes, Dict[str, Any]]] = []
        pos = 0

        n_stages = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        for _ in range(n_stages):
            name_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            method_name = data[pos : pos + name_len].decode("utf-8")
            pos += name_len

            meta_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            meta = json.loads(data[pos : pos + meta_len].decode("utf-8"))
            pos += meta_len

            comp_len = struct.unpack_from("<I", data, pos)[0]
            pos += 4
            comp_data = data[pos : pos + comp_len]
            pos += comp_len

            stages.append((method_name, comp_data, meta))

        # Blend weights
        n_weights = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        blend_weights: List[float] = []
        if n_weights > 0:
            for _ in range(n_weights):
                w = struct.unpack_from("<f", data, pos)[0]
                pos += 4
                blend_weights.append(w)

        return stages, blend_weights

    # ── Reconstruction ──────────────────────────────────────────────────

    def reconstruct(
        self,
        data: bytes,
        metadata: Dict[str, Any],
        original_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Reconstruct original tensor from stacked compressed data.

        Each stage is decompressed independently, then blended using
        the stored blend weights (or equal weights if none stored).

        Parameters
        ----------
        data : bytes
            Packaged stacked payload.
        metadata : dict
            Metadata from ``package_stacked`` or ``find_best_stacking``.
        original_shape : tuple of int
            Original tensor shape.

        Returns
        -------
        np.ndarray
            Reconstructed float32 tensor.
        """
        stages, blend_weights = self.unpack_stacked(data)

        if not stages:
            return np.zeros(original_shape, dtype=np.float32)

        n_stages = len(stages)

        # Determine blend weights
        if len(blend_weights) == n_stages:
            weights = np.array(blend_weights, dtype=np.float64)
        else:
            # Equal weights
            weights = np.ones(n_stages, dtype=np.float64) / n_stages

        # Decompress each stage
        reconstructions: List[np.ndarray] = []
        for method_name, comp_data, meta in stages:
            inst = self.engine._methods.get(method_name)
            if inst is None:
                logger.warning(
                    "Method '%s' not found during stacking reconstruction",
                    method_name,
                )
                reconstructions.append(np.zeros(original_shape, dtype=np.float32))
                continue

            try:
                meta["original_shape"] = list(original_shape)
                recon = inst.decompress(comp_data, meta)
                if recon.shape != original_shape:
                    recon = recon.reshape(original_shape)
                reconstructions.append(recon.astype(np.float32))
            except Exception as exc:
                logger.warning("Reconstruction of '%s' failed: %s", method_name, exc)
                reconstructions.append(np.zeros(original_shape, dtype=np.float32))

        # Blend
        result = np.zeros(original_shape, dtype=np.float64)
        for i, recon in enumerate(reconstructions):
            result += weights[i] * recon.astype(np.float64)

        return np.ascontiguousarray(result, dtype=np.float32)

    # ── Convenience: Full Compress Pipeline ─────────────────────────────

    def compress(
        self,
        tensor: np.ndarray,
        max_error: float = 0.01,
        min_ratio: float = 1.5,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress a tensor using the best stacking strategy.

        Convenience method that combines ``find_best_stacking`` and
        ``package_stacked`` into a single call.

        Parameters
        ----------
        tensor : np.ndarray
            Float32 tensor to compress.
        max_error : float
            Maximum allowed mean absolute error (default 0.01).
        min_ratio : float
            Minimum compression ratio to consider (default 1.5).

        Returns
        -------
        Tuple[bytes, dict]
            ``(compressed_data, metadata)`` — or passthrough if no stacking
            strategy produces a valid result.
        """
        original = np.ascontiguousarray(tensor, dtype=np.float32)
        orig_size = original.nbytes

        result = self.find_best_stacking(
            original, max_error=max_error, min_ratio=min_ratio
        )

        if result is None:
            # Passthrough fallback
            raw = original.tobytes()
            loss_metrics = TensorLossMetrics.compute(
                original=original,
                reconstructed=original,
                name="stacking_passthrough",
                compressed_size=len(raw),
            )
            return raw, {
                "method": "passthrough",
                "total_ratio": 1.0,
                "total_error": 0.0,
                "loss_metrics": loss_metrics.to_dict(),
                "quality_grade": loss_metrics.quality_grade,
                "is_acceptable": loss_metrics.is_acceptable,
            }

        ratio, error, stages, recon = result

        # Build blend weights from strategy
        n_stages = len(stages)
        if n_stages == 1:
            blend_weights = [1.0]
        else:
            # Check if we have pair weights from the complementary pairs table
            blend_weights = None
            if n_stages == 2:
                m1_name = stages[0][0]
                m2_name = stages[1][0]
                for p1, p2, w1 in self.COMPLEMENTARY_PAIRS:
                    if (p1 == m1_name and p2 == m2_name) or (
                        p1 == m2_name and p2 == m1_name
                    ):
                        if p1 == m1_name:
                            blend_weights = [w1, 1.0 - w1]
                        else:
                            blend_weights = [1.0 - w1, w1]
                        break

        compressed_data, meta = self.package_stacked(
            stages, orig_size, blend_weights=blend_weights
        )

        loss_metrics = TensorLossMetrics.compute(
            original=original,
            reconstructed=recon,
            name="stacking",
            compressed_size=len(compressed_data),
        )

        meta.update(
            {
                "total_error": float(error),
                "total_ratio": float(ratio),
                "mse": loss_metrics.mse,
                "loss_metrics": loss_metrics.to_dict(),
                "quality_grade": loss_metrics.quality_grade,
                "is_acceptable": loss_metrics.is_acceptable,
                "snr_db": loss_metrics.snr_db,
                "psnr_db": loss_metrics.psnr_db,
                "cosine_similarity": loss_metrics.cosine_similarity,
            }
        )

        return compressed_data, meta


# ── Integration helper for DirectCascadeEngine ─────────────────────────


def try_stacking_fallback(
    cascade_self: Any,
    engine: Any,
    tensor: np.ndarray,
    tensor_type: str = "weight",
    max_error: float = 0.01,
) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    """Try method stacking as a fallback when cascade produces low ratio.

    Call this from ``DirectCascadeEngine.execute_cascade()`` when all
    cascade patterns fail or produce ratio < 2.0.

    Parameters
    ----------
    cascade_self : DirectCascadeEngine
        The cascade engine instance (for access to ``_should_compress``, etc.).
    engine : CompressionIntelligenceEngine
        Engine with ``_methods`` registry.
    tensor : np.ndarray
        Float32 tensor to compress.
    tensor_type : str
        Tensor type hint.
    max_error : float
        Maximum allowed error.

    Returns
    -------
    Tuple[bytes, dict] or None
        ``(compressed_data, metadata)`` if stacking succeeds, else None.
    """
    if not cascade_self._should_compress(tensor):
        return None

    try:
        mse = MethodStackingEngine(engine)
        result = mse.compress(tensor, max_error=max_error)
        if result is not None:
            data, meta = result
            ratio = meta.get("total_ratio", 1.0)
            if ratio > 1.5:
                meta["method"] = f"stacked_fallback"
                meta["pattern"] = "stacking_fallback"
                meta["tensor_type"] = tensor_type
                logger.info(
                    "Stacking fallback succeeded: ratio=%.1fx, error=%.6f",
                    ratio,
                    meta.get("total_error", 1.0),
                )
                return data, meta
    except Exception as exc:
        logger.debug("Stacking fallback failed: %s", exc)

    return None
