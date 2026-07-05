from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    HadamardRotator,
    LloydMaxQuantizer,
    dct,
    fwht,
    idct,
    next_power_of_two,
    softmax,
    spectral_entropy,
)


class CrossLayerPredictor:
    """Exploit inter-layer correlations for additional compression.

    In deep networks, adjacent layers often have high correlation.
    This class identifies reference layers and stores deltas
    (differences) instead of full weights, achieving additional
    compression through delta encoding.
    """

    def __init__(self, correlation_threshold: float = 0.5):
        self.correlation_threshold = correlation_threshold
        self._reference_layers: Dict[str, np.ndarray] = {}
        self._delta_encoded: Dict[str, DeltaEncodedLayer] = {}

    def compute_correlation(
        self, layer_a: np.ndarray, layer_b: np.ndarray
    ) -> float:
        """Compute normalized correlation between two layers."""
        a = layer_a.ravel().astype(np.float64)
        b = layer_b.ravel().astype(np.float64)

        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        a_norm = a - np.mean(a)
        b_norm = b - np.mean(b)
        denom = (np.std(a) * np.std(b) + 1e-10)
        corr = float(np.mean(a_norm * b_norm) / denom)
        return corr

    def select_reference_layers(
        self, layers: Dict[str, np.ndarray]
    ) -> List[str]:
        """Select reference layers that minimize total delta size.

        Greedy algorithm: pick the layer with most similar layers
        as reference, then remove correlated pairs.
        """
        names = list(layers.keys())
        n = len(names)

        if n == 0:
            return []

        # Compute correlation matrix
        corr_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.compute_correlation(layers[names[i]], layers[names[j]])
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c

        # Greedy selection
        referenced = set()
        reference_names = []

        for _ in range(n):
            best_idx = -1
            best_score = -1.0
            for i in range(n):
                if i in referenced:
                    continue
                # Score = number of highly correlated unreferenced layers
                score = np.sum(
                    corr_matrix[i, :] > self.correlation_threshold
                )
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx < 0:
                break
            referenced.add(best_idx)
            reference_names.append(names[best_idx])

        return reference_names

    def encode_deltas(
        self, layers: Dict[str, np.ndarray], reference_names: Optional[List[str]] = None
    ) -> Dict[str, DeltaEncodedLayer]:
        """Encode layers as deltas from reference layers.

        For each non-reference layer, finds the most correlated
        reference and stores the delta (difference) instead.
        """
        if reference_names is None:
            reference_names = self.select_reference_layers(layers)

        self._reference_layers = {
            name: layers[name].copy() for name in reference_names if name in layers
        }
        self._delta_encoded.clear()

        for name, weights in layers.items():
            if name in self._reference_layers:
                continue

            best_ref = reference_names[0] if reference_names else None
            best_corr = -1.0
            for ref_name in reference_names:
                if ref_name in layers:
                    c = self.compute_correlation(weights, layers[ref_name])
                    if c > best_corr:
                        best_corr = c
                        best_ref = ref_name

            if best_ref is None or best_corr < self.correlation_threshold:
                # Store as reference if no good match
                self._reference_layers[name] = weights.copy()
                continue

            delta = weights - self._reference_layers[best_ref]
            delta_sparsity = float(np.mean(np.abs(delta) < 1e-10))
            original_size = weights.size * 32
            delta_bits = np.count_nonzero(delta) * 32 + 32  # sparse + offset
            cr = original_size / max(delta_bits, 1)

            self._delta_encoded[name] = DeltaEncodedLayer(
                reference_name=best_ref,
                delta=delta,
                sparsity=delta_sparsity,
                compression_ratio=cr,
            )

        logger.info(
            "Cross-layer: %d references, %d delta-encoded",
            len(self._reference_layers), len(self._delta_encoded),
        )
        return self._delta_encoded

    def decode_layer(
        self, name: str, layers: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """Decode a layer from its delta-encoded representation."""
        if name in self._reference_layers:
            return self._reference_layers[name].copy()

        if name not in self._delta_encoded:
            raise KeyError(f"Layer '{name}' not found in references or deltas")

        delta_info = self._delta_encoded[name]
        ref = self._reference_layers.get(delta_info.reference_name)
        if ref is None:
            raise KeyError(f"Reference layer '{delta_info.reference_name}' not found")

        return ref + delta_info.delta

    def get_stats(self) -> Dict[str, float]:
        """Get compression statistics."""
        total_ref_bits = sum(v.size * 32 for v in self._reference_layers.values())
        total_delta_bits = sum(
            np.count_nonzero(d.delta) * 32 + 32
            for d in self._delta_encoded.values()
        )
        total_original = total_ref_bits + sum(
            d.delta.size * 32 for d in self._delta_encoded.values()
        )
        total_compressed = total_ref_bits + total_delta_bits
        return {
            "n_references": len(self._reference_layers),
            "n_delta_encoded": len(self._delta_encoded),
            "overall_compression_ratio": total_original / max(total_compressed, 1),
            "reference_bits": total_ref_bits,
            "delta_bits": total_delta_bits,
        }
