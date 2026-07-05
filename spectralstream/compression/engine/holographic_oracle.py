"""Holographic Resonance Oracle — associative memory for zero-shot compression method selection.

Every tensor has a unique "resonance signature" — a compact vector of its statistical
and spectral properties. This signature acts as a key in a holographic associative
memory. When compressing a new tensor, we compute its resonance signature → lookup
in associative memory → if found, use cached method directly (zero-shot). If not
found, we do full method testing and store the result.

Phase-coherent matching:
  1. Exact hash match (O(1) fast path)
  2. Cosine similarity in resonance vector space (approximate)
  3. Confidence = similarity × (1 - error_penalty)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._helpers import _sample_flat

BYPASS_HIGH_CONFIDENCE = "bypass_high_confidence"
BYPASS_MEDIUM_CONFIDENCE = "bypass_medium_confidence"
TEST_FULL = "test_full"

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. ResonanceSignature
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ResonanceSignature:
    """Compact fingerprint of a tensor's compression-relevant properties.

    All fields are float by default (for ML-friendly fixed-length vectors)
    except shape_ndim (int) and tensor_type (str).
    """

    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    sparsity_1e3: float = 0.0
    sparsity_1e4: float = 0.0
    spectral_entropy: float = 0.0
    energy_concentration: float = 0.0
    effective_rank_ratio: float = 0.0
    n_elements_log: float = 0.0
    shape_ndim: int = 0
    shape_aspect: float = 0.0
    tensor_type: str = "weight"

    # Internal: origin tracking for debugging
    _tensor_name: str = ""
    _tensor_shape: Tuple[int, ...] = ()

    @staticmethod
    def n_features() -> int:
        return 12  # all float fields except shape_ndim, tensor_type

    def to_vector(self) -> np.ndarray:
        """Convert to fixed-length numpy vector for similarity matching.

        Returns shape (12,) — all numeric features including shape_ndim as float.
        """
        return np.array(
            [
                self.mean,
                self.std,
                self.skewness,
                self.kurtosis,
                self.sparsity_1e3,
                self.sparsity_1e4,
                self.spectral_entropy,
                self.energy_concentration,
                self.effective_rank_ratio,
                self.n_elements_log,
                float(self.shape_ndim),
                self.shape_aspect,
            ],
            dtype=np.float64,
        )

    def signature_tensor(self) -> np.ndarray:
        """Alias for to_vector."""
        return self.to_vector()

    def to_hash(self) -> str:
        """Create a deterministic hash key for exact matching.

        Includes tensor_type and discretised vector so that very similar
        tensors produce different hashes (exact match only).
        """
        vec = self.to_vector()
        rounded = np.round(vec, decimals=4)
        key = self.tensor_type + "|" + ",".join(f"{v:.4f}" for v in rounded)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "sparsity_1e3": self.sparsity_1e3,
            "sparsity_1e4": self.sparsity_1e4,
            "spectral_entropy": self.spectral_entropy,
            "energy_concentration": self.energy_concentration,
            "effective_rank_ratio": self.effective_rank_ratio,
            "n_elements_log": self.n_elements_log,
            "shape_ndim": self.shape_ndim,
            "shape_aspect": self.shape_aspect,
            "tensor_type": self.tensor_type,
        }


# ──────────────────────────────────────────────────────────────────────
# 2. HolographicMemoryStore
# ──────────────────────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    """A single entry in the holographic memory store."""

    signature_hash: str
    signature_vector: np.ndarray
    method_name: str
    params: Dict[str, Any]
    ratio: float
    error: float
    n_success: int = 1
    timestamp: float = 0.0


class HolographicMemoryStore:
    """Holographic associative memory for compression method selection.

    Stores (signature → method_name, params, expected_ratio, expected_error)
    mappings. Supports both exact (hash) and approximate (cosine similarity) matching.

    Persisted to disk as numpy .npz for fast load/save.
    """

    def __init__(self, memory_path: Optional[str] = None):
        self._entries: List[MemoryEntry] = []
        self._hash_index: Dict[str, int] = {}  # hash → index in _entries
        self._tensor_type_index: Dict[str, List[int]] = {}  # tensor_type → indices

        if memory_path is not None and os.path.exists(memory_path):
            try:
                self.load(memory_path)
                logger.info(
                    "Loaded %d entries from %s", len(self._entries), memory_path
                )
            except Exception as exc:
                logger.warning("Failed to load memory from %s: %s", memory_path, exc)

    def store(
        self,
        signature: ResonanceSignature,
        method_name: str,
        params: Dict[str, Any],
        ratio: float,
        error: float,
    ) -> None:
        """Store a successful compression outcome in associative memory.

        If a matching entry exists (same hash), update it with running averages
        to improve confidence over time.
        """
        sig_hash = signature.to_hash()
        sig_vec = signature.to_vector()

        if sig_hash in self._hash_index:
            idx = self._hash_index[sig_hash]
            entry = self._entries[idx]
            n = entry.n_success
            entry.ratio = (entry.ratio * n + ratio) / (n + 1)
            entry.error = (entry.error * n + error) / (n + 1)
            entry.n_success = n + 1
            entry.timestamp = time.time()
            # Keep most recent params
            entry.params = params
            return

        entry = MemoryEntry(
            signature_hash=sig_hash,
            signature_vector=sig_vec,
            method_name=method_name,
            params=params,
            ratio=ratio,
            error=error,
            n_success=1,
            timestamp=time.time(),
        )
        idx = len(self._entries)
        self._entries.append(entry)
        self._hash_index[sig_hash] = idx
        tt = signature.tensor_type
        self._tensor_type_index.setdefault(tt, []).append(idx)

    def recall(
        self,
        signature: ResonanceSignature,
        min_confidence: float = 0.85,
    ) -> Optional[Dict[str, Any]]:
        """Recall optimal method from associative memory.

        Phase-coherent matching:
        1. Check exact hash match first (O(1) fast path)
        2. If no exact match, find nearest neighbors by cosine similarity
           in resonance vector space (constrained to same tensor_type)
        3. If nearest neighbor similarity > threshold, return its method
           with confidence = similarity × (1 - error_penalty)

        Returns dict with keys: method_name, params, ratio, error, confidence, match_type
        or None if no match above threshold.
        """
        # ── Phase 1: Exact hash match ──
        sig_hash = signature.to_hash()
        if sig_hash in self._hash_index:
            idx = self._hash_index[sig_hash]
            entry = self._entries[idx]
            confidence = self._confidence_score(1.0, entry.error, entry.n_success)
            return {
                "method_name": entry.method_name,
                "params": entry.params,
                "ratio": entry.ratio,
                "error": entry.error,
                "confidence": confidence,
                "match_type": "exact",
            }

        # ── Phase 2: Approximate cosine similarity match ──
        candidate_indices = self._tensor_type_index.get(signature.tensor_type, [])
        if not candidate_indices:
            # Fall back to searching all entries
            candidate_indices = list(range(len(self._entries)))

        if not candidate_indices:
            return None

        query_vec = signature.to_vector()
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-30:
            return None

        best_sim = -1.0
        best_idx = -1

        for idx in candidate_indices:
            entry = self._entries[idx]
            entry_norm = np.linalg.norm(entry.signature_vector)
            if entry_norm < 1e-30:
                continue
            sim = float(
                np.dot(query_vec, entry.signature_vector) / (query_norm * entry_norm)
            )
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx < 0 or best_sim < 0.3:
            return None

        entry = self._entries[best_idx]

        # Confidence = cosine_sim × success_factor × error_penalty
        confidence = self._confidence_score(best_sim, entry.error, entry.n_success)

        if confidence < min_confidence:
            return None

        return {
            "method_name": entry.method_name,
            "params": entry.params,
            "ratio": entry.ratio,
            "error": entry.error,
            "confidence": confidence,
            "similarity": best_sim,
            "match_type": "approximate",
        }

    def _confidence_score(
        self, similarity: float, error: float, n_success: int
    ) -> float:
        """Compute confidence from similarity, error, and repetition count."""
        error_penalty = max(0.0, 1.0 - error * 10.0)
        repetition_bonus = min(1.0, n_success / 5.0)
        return similarity * error_penalty * (0.7 + 0.3 * repetition_bonus)

    def similarity_to_entry(
        self, signature: ResonanceSignature, entry_idx: int
    ) -> float:
        """Compute cosine similarity between a signature and a stored entry."""
        if entry_idx < 0 or entry_idx >= len(self._entries):
            return 0.0
        query = signature.to_vector()
        stored = self._entries[entry_idx].signature_vector
        qn = np.linalg.norm(query)
        sn = np.linalg.norm(stored)
        if qn < 1e-30 or sn < 1e-30:
            return 0.0
        return float(np.dot(query, stored) / (qn * sn))

    def n_entries(self) -> int:
        return len(self._entries)

    def n_entries_by_type(self, tensor_type: str) -> int:
        return len(self._tensor_type_index.get(tensor_type, []))

    def save(self, path: str) -> None:
        """Persist memory to disk as numpy .npz."""
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        n = len(self._entries)
        if n == 0:
            logger.warning("No entries to save — skipping")
            return

        vecs = np.zeros((n, ResonanceSignature.n_features()), dtype=np.float64)
        hashes = []
        method_names = []
        ratios = np.zeros(n, dtype=np.float64)
        errors = np.zeros(n, dtype=np.float64)
        n_success_arr = np.zeros(n, dtype=np.int32)
        timestamps = np.zeros(n, dtype=np.float64)
        params_json = []

        for i, entry in enumerate(self._entries):
            vecs[i] = entry.signature_vector
            hashes.append(entry.signature_hash)
            method_names.append(entry.method_name)
            ratios[i] = entry.ratio
            errors[i] = entry.error
            n_success_arr[i] = entry.n_success
            timestamps[i] = entry.timestamp
            params_json.append(json.dumps(entry.params))

        np.savez_compressed(
            path,
            vectors=vecs,
            hashes=hashes,
            method_names=method_names,
            ratios=ratios,
            errors=errors,
            n_success=n_success_arr,
            timestamps=timestamps,
            params_json=params_json,
        )
        logger.info("Saved %d memory entries to %s", n, path)

    def load(self, path: str) -> None:
        """Load persisted memory from numpy .npz."""
        data = np.load(path, allow_pickle=True)

        hashes = (
            data["hashes"].tolist()
            if data["hashes"].ndim > 0
            else [data["hashes"].item()]
        )
        method_names = (
            data["method_names"].tolist()
            if data["method_names"].ndim > 0
            else [data["method_names"].item()]
        )
        params_json = (
            data["params_json"].tolist()
            if data["params_json"].ndim > 0
            else [data["params_json"].item()]
        )

        n = len(hashes)
        self._entries.clear()
        self._hash_index.clear()
        self._tensor_type_index.clear()

        for i in range(n):
            vec = data["vectors"][i]
            params = json.loads(params_json[i]) if params_json[i] else {}

            entry = MemoryEntry(
                signature_hash=str(hashes[i]),
                signature_vector=vec,
                method_name=str(method_names[i]),
                params=params,
                ratio=float(data["ratios"][i]),
                error=float(data["errors"][i]),
                n_success=int(data["n_success"][i]),
                timestamp=float(data["timestamps"][i]),
            )
            idx = len(self._entries)
            self._entries.append(entry)
            self._hash_index[entry.signature_hash] = idx
            # We don't store tensor_type in the saved data — reconstruct from params
            # (approx; for full fidelity, we could add it as a saved field)

    def clear(self) -> None:
        self._entries.clear()
        self._hash_index.clear()
        self._tensor_type_index.clear()

    def get_stats(self) -> Dict[str, Any]:
        if not self._entries:
            return {"n_entries": 0, "n_types": 0}
        return {
            "n_entries": len(self._entries),
            "n_types": len(self._tensor_type_index),
            "avg_success": float(np.mean([e.n_success for e in self._entries])),
            "avg_ratio": float(np.mean([e.ratio for e in self._entries])),
            "avg_error": float(np.mean([e.error for e in self._entries])),
        }


# ──────────────────────────────────────────────────────────────────────
# 3. HolographicOracle
# ──────────────────────────────────────────────────────────────────────


class HolographicOracle:
    """Holographic Method Oracle — learns from every compression.

    Integrates:
    - HolographicMemoryStore for associative recall
    - Existing MethodOracle as fallback when no memory match
    - Performance tracking for continuous learning

    Bypass decisions:
    - Zero-shot (high confidence):   confidence >= 0.90 → skip all testing
    - Approximate (medium):          confidence >= 0.80 → test only top 1
    - Fallback:                      no match → TEST_FULL via MethodOracle
    """

    def __init__(
        self,
        engine: Any,
        method_oracle: Any = None,
        memory_path: Optional[str] = None,
    ):
        self._engine = engine
        self._memory = HolographicMemoryStore(memory_path=memory_path)

        # Lazy-import MethodOracle to avoid circular imports
        self._oracle = method_oracle
        self._oracle_loaded = method_oracle is not None

        # Performance tracking for continuous improvement
        self._n_queries: int = 0
        self._n_hits: int = 0
        self._n_stores: int = 0
        self._n_approximate: int = 0

    @property
    def memory(self) -> HolographicMemoryStore:
        return self._memory

    def _get_oracle(self) -> Any:
        if not self._oracle_loaded:
            from .world_model.method_oracle import MethodOracle

            try:
                self._oracle = MethodOracle(self._engine)
            except Exception as exc:
                logger.warning("Failed to load MethodOracle: %s", exc)
                self._oracle = None
            self._oracle_loaded = True
        return self._oracle

    # ── Signature Computation ──────────────────────────────────────────

    def compute_signature(
        self, tensor: np.ndarray, tensor_type: str = "weight"
    ) -> ResonanceSignature:
        """Compute resonance signature from a tensor.

        Lightweight: uses sampled statistics (~min(10000, n_elements) samples)
        and a lightweight DCT on one representative row/column for spectral features.
        """
        flat = _sample_flat(tensor, max_samples=10000)
        n_elements = tensor.size

        # ── Statistical moments ──
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        if std > 1e-30:
            skewness = float(np.mean(((flat - mean) / std) ** 3))
            kurtosis = float(np.mean(((flat - mean) / std) ** 4)) - 3.0
        else:
            skewness = 0.0
            kurtosis = 0.0

        # ── Sparsity ──
        abs_flat = np.abs(flat)
        sparsity_1e3 = float(np.mean(abs_flat < 0.001))
        sparsity_1e4 = float(np.mean(abs_flat < 0.0001))

        # ── Spectral features (lightweight DCT on a sample) ──
        spectral_entropy = 0.0
        energy_concentration = 0.0

        if flat.size >= 16:
            try:
                # Use a small window of the flattened data for DCT
                dct_input = flat[: min(1024, flat.size)]
                dct_coeffs = self._lightweight_dct(dct_input)
                dct_energy = dct_coeffs**2
                total_energy = float(np.sum(dct_energy))
                if total_energy > 1e-30:
                    energy_dist = dct_energy / (total_energy + 1e-30)
                    spectral_entropy = -float(
                        np.sum(energy_dist * np.log2(energy_dist + 1e-30))
                    )
                    max_entropy = np.log2(len(dct_coeffs))
                    spectral_entropy = (
                        spectral_entropy / max_entropy if max_entropy > 0 else 0.0
                    )
                    n_top = max(1, len(dct_coeffs) // 10)
                    top_energy = float(np.sum(np.sort(dct_energy)[-n_top:]))
                    energy_concentration = top_energy / (total_energy + 1e-30)
            except Exception:
                pass

        # ── Structural features ──
        shape_ndim = tensor.ndim
        shape_aspect = 0.0
        if tensor.ndim >= 2:
            shape_aspect = max(tensor.shape) / max(min(tensor.shape), 1)

        effective_rank_ratio = 0.0
        try:
            if tensor.ndim >= 2 and min(tensor.shape) >= 4:
                sv_sample = tensor[
                    : min(64, tensor.shape[0]), : min(64, tensor.shape[1])
                ]
                s = np.linalg.svd(sv_sample, compute_uv=False)
                s_sum = float(np.sum(s))
                if s_sum > 1e-30:
                    s_norm = s / s_sum
                    eff_rank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-30))))
                    effective_rank_ratio = eff_rank / min(sv_sample.shape)
        except Exception:
            pass

        return ResonanceSignature(
            mean=mean,
            std=std,
            skewness=skewness,
            kurtosis=kurtosis,
            sparsity_1e3=sparsity_1e3,
            sparsity_1e4=sparsity_1e4,
            spectral_entropy=spectral_entropy,
            energy_concentration=energy_concentration,
            effective_rank_ratio=effective_rank_ratio,
            n_elements_log=np.log10(max(n_elements, 1)),
            shape_ndim=shape_ndim,
            shape_aspect=shape_aspect,
            tensor_type=tensor_type,
            _tensor_name="",
            _tensor_shape=tensor.shape,
        )

    @staticmethod
    def _lightweight_dct(x: np.ndarray) -> np.ndarray:
        """Type-II DCT via FFT, same as scipy.fft.dct."""
        n = len(x)
        x2 = np.zeros(2 * n, dtype=np.float64)
        x2[:n] = x
        x2[n:] = x[::-1]
        fft = np.fft.fft(x2)[:n]
        scale = np.sqrt(2.0 / n)
        coeffs = fft.real * scale
        coeffs[0] *= 1.0 / np.sqrt(2.0)
        return coeffs

    # ── Method Selection ───────────────────────────────────────────────

    def select_method(
        self,
        tensor: np.ndarray,
        tensor_type: str = "weight",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
    ) -> Tuple[List[Any], str]:
        """Select compression method using holographic memory.

        Returns (ranked_methods, bypass_decision)
        where bypass_decision is one of:
          BYPASS_HIGH_CONFIDENCE   — zero-shot, use top-1 without testing
          BYPASS_MEDIUM_CONFIDENCE — test only top-1
          TEST_FULL                — full method testing via MethodOracle

        Flow:
        1. Compute tensor's resonance signature
        2. Try holographic recall
        3. If confident match → BYPASS_HIGH_CONFIDENCE with single method
        4. If approximate match → BYPASS_MEDIUM_CONFIDENCE with single method
        5. If no match → TEST_FULL via MethodOracle
        """
        self._n_queries += 1
        signature = self.compute_signature(tensor, tensor_type)

        # Try holographic recall
        recalled = self._memory.recall(signature, min_confidence=0.80)

        if recalled is not None:
            confidence = recalled["confidence"]
            method_name = recalled["method_name"]
            params = recalled.get("params", {})

            # Get method instance from engine
            inst = self._engine._methods.get(method_name)
            if inst is not None:
                from .world_model.method_oracle import RankedMethod

                ranked = [
                    RankedMethod(
                        name=method_name,
                        instance=inst,
                        params=params,
                        expected_ratio=recalled["ratio"],
                        expected_error=recalled["error"],
                        confidence=confidence,
                        vote_score=confidence,
                    )
                ]

                if confidence >= 0.90:
                    self._n_hits += 1
                    if recalled.get("match_type") == "approximate":
                        self._n_approximate += 1
                    return ranked, BYPASS_HIGH_CONFIDENCE
                elif confidence >= 0.80:
                    self._n_approximate += 1
                    return ranked, BYPASS_MEDIUM_CONFIDENCE

        # Fallback: use MethodOracle
        oracle = self._get_oracle()
        if oracle is not None:
            try:
                from ._profiler import CompressionProfiler

                profile = self._engine.profile_tensor(tensor)
                ranked, bypass = oracle.select_with_bypass(
                    profile=profile,
                    tensor_type=tensor_type,
                    target_ratio=target_ratio,
                    max_error=max_error,
                    max_results=10,
                )
                return ranked, bypass
            except Exception as exc:
                logger.debug("MethodOracle fallback failed: %s", exc)

        # Last resort: use _select_methods helper
        from ._helpers import _select_methods

        profile = self._engine.profile_tensor(tensor)
        methods = _select_methods(
            profile, max_error / max(target_ratio, 1.0), target_ratio
        )
        return methods, TEST_FULL

    # ── Recording Success ──────────────────────────────────────────────

    def record_success(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        method_name: str,
        params: Dict[str, Any],
        ratio: float,
        error: float,
    ) -> None:
        """Record successful compression for future recall.

        Computes the resonance signature (or accepts existing one) and
        stores it in associative memory.
        """
        signature = self.compute_signature(tensor, tensor_type)
        self._memory.store(signature, method_name, params, ratio, error)
        self._n_stores += 1

    def record_success_with_signature(
        self,
        signature: ResonanceSignature,
        method_name: str,
        params: Dict[str, Any],
        ratio: float,
        error: float,
    ) -> None:
        """Record success with a pre-computed signature (avoids recomputation)."""
        self._memory.store(signature, method_name, params, ratio, error)
        self._n_stores += 1

    # ── Persistence ────────────────────────────────────────────────────

    def save_memory(self, path: str) -> None:
        self._memory.save(path)

    def load_memory(self, path: str) -> None:
        self._memory.load(path)

    # ── Statistics ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        mem_stats = self._memory.get_stats()
        return {
            "n_queries": self._n_queries,
            "n_hits": self._n_hits,
            "n_approximate": self._n_approximate,
            "n_stores": self._n_stores,
            "hit_rate": self._n_hits / max(self._n_queries, 1),
            **mem_stats,
        }

    def get_memory(self) -> HolographicMemoryStore:
        return self._memory
