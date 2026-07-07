"""Unified Method Selection Oracle — single replacement for ALL 11 competing approaches.

Absorbs:
  1. MethodOracle (engine/world_model/method_oracle.py) — ensemble voting
  2. HolographicOracle (engine/holographic_oracle.py) — associative memory recall
  3. DynamicMethodTester (engine/dynamic_method_tester.py) — tests ALL methods
  4. ModelIntelligence.predict (engine/model_intelligence.py) — digital twin predictions
  5. CompressionStrategySelector (engine/compression_intelligence.py) — heuristic scoring
  6. MethodEvaluator (engine/intelligence.py) — category-affinity scoring
  7. AdaptiveMethodSelector (engine/compression_intelligence.py) — error-threshold cycling
  8. DynamicTensorIntelligence (engine/dynamic_tensor_intelligence.py) — decision tree
  9. UnifiedQuantizationSystem._select_method (engine/unified_quant_system.py) — profile-based
  10. ZeroShotPredictor (engine/dynamic_method_tester.py) — semantic fingerprint
  11. BayesianPerformanceTracker (engine/self_evolving_intelligence.py) — Bayesian posterior

Selection Pipeline (staged, exits early if confident):
  Stage 1 (0-1ms):    Holographic recall — check associative memory for exact match
  Stage 2 (1-10ms):   Zero-shot prediction — semantic fingerprint predicts best method
  Stage 3 (10-50ms):  Ensemble voting — all strategies weighted by past accuracy
  Stage 4 (100-500ms): Quantum superposition — test top 3 methods in parallel
  Stage 5 (R&D only):  Exhaustive — test ALL 80+ methods
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
BYPASS_HIGH_CONFIDENCE = "bypass_high_confidence"
BYPASS_MEDIUM_CONFIDENCE = "bypass_medium_confidence"
TEST_FULL = "test_full"
HOLOGRAPHIC_MEMORY_VERSION = 1


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class MethodSelection:
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    score: float = 0.0
    expected_ratio: float = 1.0
    expected_error: float = 0.01
    bypass_decision: str = TEST_FULL
    stage: str = "none"
    time_ms: float = 0.0


@dataclass
class QuantumSuperpositionTest:
    method_names: List[str] = field(default_factory=list)
    results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_method: str = ""
    time_ms: float = 0.0

    @property
    def n_tested(self) -> int:
        return len(self.results)


# ── Inline TensorFeatures (no external dependency) ────────────────────────────


@dataclass
class _TensorFeatures:
    n_elements: int = 0
    ndim: int = 0
    shape: Tuple[int, ...] = ()
    dtype: str = "float32"
    sparsity: float = 0.0
    mean_abs: float = 0.0
    std: float = 0.0
    mean: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    spectral_entropy: float = 0.0
    dct_concentration: float = 0.0
    energy_concentration: float = 0.0
    effective_rank: float = 0.0
    value_range: float = 0.0
    snr_estimate: float = 20.0
    tensor_type: str = "weight"
    sensitivity: float = 0.5
    compressibility_score: float = 0.0
    outlier_ratio_3sigma: float = 0.01

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.n_elements,
                self.ndim,
                self.sparsity,
                self.mean_abs,
                self.std,
                self.kurtosis,
                self.skewness,
                self.spectral_entropy,
                self.dct_concentration,
                self.effective_rank,
                self.value_range,
                self.snr_estimate,
            ],
            dtype=np.float64,
        )


@dataclass
class _ResonanceSignature:
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
    _tensor_name: str = ""
    _tensor_shape: Tuple[int, ...] = ()

    @staticmethod
    def n_features() -> int:
        return 12

    def to_vector(self) -> np.ndarray:
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

    def to_hash(self) -> str:
        vec = self.to_vector()
        rounded = np.round(vec, decimals=4)
        key = self.tensor_type + "|" + ",".join(f"{v:.4f}" for v in rounded)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ── Holographic Memory Store (self-contained) ─────────────────────────────────


@dataclass
class _MemoryEntry:
    signature_hash: str = ""
    signature_vector: np.ndarray = field(default_factory=lambda: np.zeros(12))
    method_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    ratio: float = 1.0
    error: float = 0.0
    n_success: int = 1
    timestamp: float = 0.0


class _HolographicMemoryStore:
    def __init__(self, memory_path: Optional[str] = None, max_entries: int = 10000):
        self._entries: List[_MemoryEntry] = []
        self._hash_index: Dict[str, int] = {}
        self._tensor_type_index: Dict[str, List[int]] = {}
        self._access_order: List[str] = []
        self._max_entries = max_entries

        if memory_path and os.path.exists(memory_path):
            try:
                self.load(memory_path)
            except Exception as exc:
                logger.debug("Failed to load holographic memory: %s", exc)

    def _evict_lru(self) -> None:
        active = sum(1 for e in self._entries if e is not None)
        while active > self._max_entries and self._access_order:
            oldest_hash = self._access_order.pop(0)
            idx = self._hash_index.get(oldest_hash)
            if idx is not None and self._entries[idx] is not None:
                entry = self._entries[idx]
                ttype_key = entry.signature_hash[:8]
                for ttype, indices in list(self._tensor_type_index.items()):
                    if idx in indices:
                        indices.remove(idx)
                del self._hash_index[oldest_hash]
                self._entries[idx] = None
                active -= 1

    def _touch(self, sig_hash: str) -> None:
        if sig_hash in self._hash_index:
            if sig_hash in self._access_order:
                self._access_order.remove(sig_hash)
            self._access_order.append(sig_hash)

    def store(
        self,
        signature: _ResonanceSignature,
        method_name: str,
        params: Dict[str, Any],
        ratio: float,
        error: float,
    ) -> None:
        sig_hash = signature.to_hash()
        sig_vec = signature.to_vector()
        if sig_hash in self._hash_index:
            idx = self._hash_index[sig_hash]
            entry = self._entries[idx]
            if entry is None:
                return
            n = entry.n_success
            entry.ratio = (entry.ratio * n + ratio) / (n + 1)
            entry.error = (entry.error * n + error) / (n + 1)
            entry.n_success = n + 1
            entry.timestamp = time.time()
            entry.params = params
            self._touch(sig_hash)
            return
        entry = _MemoryEntry(
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
        self._touch(sig_hash)
        self._evict_lru()

    def recall(
        self,
        signature: _ResonanceSignature,
        min_confidence: float = 0.85,
        top_k: int = 1,
    ) -> Optional[Dict[str, Any]]:
        sig_hash = signature.to_hash()
        if sig_hash in self._hash_index:
            idx = self._hash_index[sig_hash]
            entry = self._entries[idx]
            if entry is None:
                return self._recall_approximate(signature, min_confidence, top_k)
            self._touch(sig_hash)
            confidence = self._confidence(1.0, entry.error, entry.n_success)
            return {
                "method_name": entry.method_name,
                "params": entry.params,
                "ratio": entry.ratio,
                "error": entry.error,
                "confidence": confidence,
                "match_type": "exact",
            }
        return self._recall_approximate(signature, min_confidence, top_k)

    def recall_top_k(
        self,
        signature: _ResonanceSignature,
        k: int = 3,
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        sig_hash = signature.to_hash()
        if sig_hash in self._hash_index:
            idx = self._hash_index[sig_hash]
            entry = self._entries[idx]
            if entry is not None:
                self._touch(sig_hash)
                confidence = self._confidence(1.0, entry.error, entry.n_success)
                return [
                    {
                        "method_name": entry.method_name,
                        "params": entry.params,
                        "ratio": entry.ratio,
                        "error": entry.error,
                        "confidence": confidence,
                        "match_type": "exact",
                    }
                ]
        candidate_indices = self._tensor_type_index.get(signature.tensor_type, [])
        if not candidate_indices:
            candidate_indices = [
                i for i in range(len(self._entries)) if self._entries[i] is not None
            ]
        if not candidate_indices:
            return []
        query_vec = signature.to_vector()
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm < 1e-30:
            return []
        scored: List[Tuple[float, int]] = []
        for idx in candidate_indices:
            entry = self._entries[idx]
            if entry is None:
                continue
            entry_vec = entry.signature_vector
            entry_norm = float(np.linalg.norm(entry_vec))
            if entry_norm < 1e-30:
                continue
            cos_sim = float(np.dot(query_vec, entry_vec) / (query_norm * entry_norm))
            l2_dist = float(np.sqrt(np.sum((query_vec - entry_vec) ** 2)))
            combined = cos_sim * 0.7 - l2_dist * 0.3
            scored.append((combined, idx))
        scored.sort(key=lambda x: -x[0])
        results = []
        for score, idx in scored[:k]:
            entry = self._entries[idx]
            if entry is None:
                continue
            confidence = self._confidence(score, entry.error, entry.n_success)
            if confidence < min_confidence:
                continue
            results.append(
                {
                    "method_name": entry.method_name,
                    "params": entry.params,
                    "ratio": entry.ratio,
                    "error": entry.error,
                    "confidence": confidence,
                    "similarity": score,
                    "match_type": "approximate",
                }
            )
            self._touch(entry.signature_hash)
        return results

    def _recall_approximate(
        self,
        signature: _ResonanceSignature,
        min_confidence: float = 0.85,
        top_k: int = 1,
    ) -> Optional[Dict[str, Any]]:
        candidate_indices = self._tensor_type_index.get(signature.tensor_type, [])
        if not candidate_indices:
            candidate_indices = [
                i for i in range(len(self._entries)) if self._entries[i] is not None
            ]
        if not candidate_indices:
            return None
        query_vec = signature.to_vector()
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm < 1e-30:
            return None
        best_sim = -1.0
        best_l2 = float("inf")
        best_idx = -1
        for idx in candidate_indices:
            entry = self._entries[idx]
            if entry is None:
                continue
            entry_vec = entry.signature_vector
            entry_norm = float(np.linalg.norm(entry_vec))
            if entry_norm < 1e-30:
                continue
            cos_sim = float(np.dot(query_vec, entry_vec) / (query_norm * entry_norm))
            l2_dist = float(np.sqrt(np.sum((query_vec - entry_vec) ** 2)))
            combined = cos_sim * 0.7 - l2_dist * 0.3
            if combined > best_sim:
                best_sim = combined
                best_l2 = l2_dist
                best_idx = idx
        if best_idx < 0 or best_sim < -0.5:
            return None
        entry = self._entries[best_idx]
        if entry is None:
            return None
        confidence = self._confidence(best_sim, entry.error, entry.n_success)
        if confidence < min_confidence:
            return None
        self._touch(entry.signature_hash)
        return {
            "method_name": entry.method_name,
            "params": entry.params,
            "ratio": entry.ratio,
            "error": entry.error,
            "confidence": confidence,
            "similarity": best_sim,
            "l2_distance": float(best_l2),
            "match_type": "approximate",
        }

    @staticmethod
    def _confidence(similarity: float, error: float, n_success: int) -> float:
        error_penalty = max(0.0, 1.0 - error * 10.0)
        repetition_bonus = min(1.0, n_success / 5.0)
        return similarity * error_penalty * (0.7 + 0.3 * repetition_bonus)

    def save(self, path: str) -> None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        active = [e for e in self._entries if e is not None]
        n = len(active)
        if n == 0:
            return
        vecs = np.zeros((n, _ResonanceSignature.n_features()), dtype=np.float64)
        hashes = []
        method_names = []
        ratios = np.zeros(n, dtype=np.float64)
        errors = np.zeros(n, dtype=np.float64)
        n_success_arr = np.zeros(n, dtype=np.int32)
        timestamps = np.zeros(n, dtype=np.float64)
        params_json = []
        for i, entry in enumerate(active):
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

    def load(self, path: str) -> None:
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
            entry = _MemoryEntry(
                signature_hash=str(hashes[i]),
                signature_vector=data["vectors"][i],
                method_name=str(method_names[i]),
                params=json.loads(params_json[i]) if params_json[i] else {},
                ratio=float(data["ratios"][i]),
                error=float(data["errors"][i]),
                n_success=int(data["n_success"][i]),
                timestamp=float(data["timestamps"][i]),
            )
            idx = len(self._entries)
            self._entries.append(entry)
            self._hash_index[entry.signature_hash] = idx

    def get_stats(self) -> Dict[str, Any]:
        active = [e for e in self._entries if e is not None]
        if not active:
            return {"n_entries": 0}
        return {
            "n_entries": len(active),
            "n_total_slots": len(self._entries),
            "n_types": len(self._tensor_type_index),
            "max_entries": self._max_entries,
            "avg_success": float(np.mean([e.n_success for e in active])),
            "avg_ratio": float(np.mean([e.ratio for e in active])),
            "avg_error": float(np.mean([e.error for e in active])),
        }


# ── Inline Zero-Shot Predictor ────────────────────────────────────────────────


class _ZeroShotPredictor:
    def __init__(self):
        self._pattern_cache: Dict[str, List[Tuple[str, dict, float]]] = {}

    @staticmethod
    def _classify_by_name(name: str) -> str:
        nl = name.lower()
        if any(k in nl for k in ("q_proj", "wq")):
            return "attention_q"
        if any(k in nl for k in ("k_proj", "wk")):
            return "attention_k"
        if any(k in nl for k in ("v_proj", "wv")):
            return "attention_v"
        if any(k in nl for k in ("o_proj", "wo")):
            return "attention_o"
        if any(k in nl for k in ("gate_proj", "w1")):
            return "ffn_gate"
        if any(k in nl for k in ("up_proj", "w3")):
            return "ffn_up"
        if any(k in nl for k in ("down_proj", "w2")):
            return "ffn_down"
        if any(k in nl for k in ("embed", "wte")):
            return "embedding"
        if any(k in nl for k in ("norm", "rms")):
            return "norm"
        if any(k in nl for k in ("head", "lm_head")):
            return "lm_head"
        return "other"

    @staticmethod
    def semantic_fingerprint(name: str, shape: Tuple[int, ...]) -> str:
        layer_type = _ZeroShotPredictor._classify_by_name(name)
        shape_str = "x".join(str(d) for d in shape)
        return f"{layer_type}_{shape_str}"

    def predict(
        self, name: str, shape: Tuple[int, ...], target_ratio: float
    ) -> List[Tuple[str, dict, float]]:
        fp = self.semantic_fingerprint(name, shape)
        if fp in self._pattern_cache:
            return self._pattern_cache[fp]
        tensor_type = self._classify_by_name(name)
        predictions = self._rule_based(tensor_type, target_ratio)
        return predictions

    def _rule_based(
        self, tensor_type: str, target_ratio: float
    ) -> List[Tuple[str, dict, float]]:
        if tensor_type in ("attention_q", "attention_k", "attention_v", "attention_o"):
            if target_ratio > 200:
                return [("svd_compress", {"rank": 16}, 0.85)]
            return [("svd_compress", {"rank": 32}, 0.9)]
        if tensor_type in ("ffn_gate", "ffn_up", "ffn_down"):
            if target_ratio > 500:
                return [("tensor_train", {"rank": 8}, 0.8)]
            return [("tensor_train", {"rank": 16}, 0.85)]
        if tensor_type == "embedding":
            return [("block_int4", {"block_size": 32}, 0.8)]
        if tensor_type in ("lm_head", "output"):
            return [("block_int4", {"block_size": 32}, 0.7)]
        if target_ratio > 100:
            return [("dct_spectral", {"keep_energy": 0.95}, 0.7)]
        return [("block_int8", {"block_size": 128}, 0.6)]

    def record_result(
        self, name: str, shape: Tuple[int, ...], method_name: str, confidence: float
    ) -> None:
        fp = self.semantic_fingerprint(name, shape)
        if fp not in self._pattern_cache:
            self._pattern_cache[fp] = []
        self._pattern_cache[fp].append((method_name, {}, confidence))
        self._pattern_cache[fp].sort(key=lambda x: -x[2])


# ── Inline Bayesian Performance Tracker ───────────────────────────────────────


@dataclass
class _MethodPerformance:
    method_name: str = ""
    tensor_type: str = ""
    n_trials: int = 0
    n_successes: int = 0
    ratio_mean: float = 3.88
    ratio_variance: float = 1.0
    ratio_n: int = 0
    error_mean: float = 0.01
    error_variance: float = 0.01
    error_n: int = 0
    expected_ratio: float = 3.88
    expected_error: float = 0.01
    confidence: float = 0.0

    def update(self, ratio: float, error: float, success: bool) -> None:
        self.n_trials += 1
        if success:
            self.n_successes += 1
        alpha_r = 1.0 / (1.0 + self.ratio_n)
        self.ratio_mean = (1 - alpha_r) * self.ratio_mean + alpha_r * ratio
        if self.ratio_n > 0:
            self.ratio_variance = (1 - alpha_r) * self.ratio_variance + alpha_r * (
                ratio - self.ratio_mean
            ) ** 2
        self.ratio_n += 1
        alpha_e = 1.0 / (1.0 + self.error_n)
        self.error_mean = (1 - alpha_e) * self.error_mean + alpha_e * error
        if self.error_n > 0:
            self.error_variance = (1 - alpha_e) * self.error_variance + alpha_e * (
                error - self.error_mean
            ) ** 2
        self.error_n += 1
        self.expected_ratio = self.ratio_mean
        self.expected_error = max(self.error_mean, 1e-10)
        self.confidence = min(1.0, self.n_trials / 20.0)

    @property
    def score(self) -> float:
        return (
            self.expected_ratio
            / max(self.expected_error, 1e-10)
            * (0.5 + 0.5 * self.confidence)
        )


class _BayesianTracker:
    def __init__(self):
        self._performances: Dict[str, _MethodPerformance] = {}

    @staticmethod
    def _key(method_name: str, tensor_type: str) -> str:
        return f"{method_name}:{tensor_type}"

    def record(
        self, method_name: str, tensor_type: str, ratio: float, error: float
    ) -> None:
        key = self._key(method_name, tensor_type)
        success = error < 0.01
        if key not in self._performances:
            self._performances[key] = _MethodPerformance(
                method_name=method_name, tensor_type=tensor_type
            )
        self._performances[key].update(ratio, error, success)

    def predict(self, method_name: str, tensor_type: str) -> _MethodPerformance:
        key = self._key(method_name, tensor_type)
        if key in self._performances:
            return self._performances[key]
        similar = [
            p for k, p in self._performances.items() if method_name.split("_")[0] in k
        ]
        if similar:
            avg_ratio = float(np.mean([s.expected_ratio for s in similar]))
            avg_error = float(np.mean([s.expected_error for s in similar]))
            return _MethodPerformance(
                method_name=method_name,
                tensor_type=tensor_type,
                expected_ratio=avg_ratio,
                expected_error=avg_error,
                confidence=0.5,
            )
        return _MethodPerformance(
            method_name=method_name,
            tensor_type=tensor_type,
            expected_ratio=3.88,
            expected_error=0.01,
            confidence=0.1,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UnifiedMethodOracle
# ═══════════════════════════════════════════════════════════════════════════════


class UnifiedMethodOracle:
    """Single unified method selection oracle.

    Selection Pipeline (staged, exits early if confident):
      Stage 1 (0-1ms):    Holographic recall — associative memory for exact match
      Stage 2 (1-10ms):   Zero-shot prediction — semantic fingerprint
      Stage 3 (10-50ms):  Ensemble voting — all strategies weighted by past accuracy
      Stage 4 (100-500ms): Quantum superposition — test top 3 methods in parallel
      Stage 5 (R&D only):  Exhaustive — test ALL 80+ methods
    """

    # Method categories for decision tree
    TYPE_CATEGORY = {
        "attention_q": "decomposition",
        "attention_k": "decomposition",
        "attention_v": "decomposition",
        "attention_o": "spectral",
        "ffn_gate": "structural",
        "ffn_up": "structural",
        "ffn_down": "structural",
        "embedding": "quantization",
        "norm": "quantization",
        "output": "quantization",
    }

    # Default params per method
    DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        "block_int8": {"block_size": 128},
        "block_int4": {"block_size": 32},
        "hadamard_int8": {"block_size": 128},
        "hadamard_int4": {"block_size": 32},
        "svd_compress": {"rank": 32},
        "tensor_train": {"rank": 16},
        "dct_spectral": {"keep_energy": 0.95, "n_bits": 8},
        "dct_2d": {"keep_energy": 0.95},
        "fwht_compress": {"keep_fraction": 0.2},
        "sparsify": {"sparsity": 0.8},
        "block_sparsity": {"sparsity": 0.8},
        "product_quantize": {"bits": 4, "n_subspaces": 8},
        "uniform_quantize": {"bits": 4},
        "hadamard_quant": {"n_bits": 4},
        "rans": {},
        "huffman": {},
    }

    def __init__(
        self,
        method_registry: Optional[Dict[str, Dict[str, Any]]] = None,
        knowledge_graph: Optional[Any] = None,
        holographic_memory: Optional[_HolographicMemoryStore] = None,
        rng_seed: int = 42,
    ):
        self._rng = np.random.RandomState(rng_seed)
        self._method_registry = method_registry or {}
        self._knowledge_graph = knowledge_graph
        self._holographic_memory = holographic_memory or _HolographicMemoryStore()
        self._zero_shot = _ZeroShotPredictor()
        self._bayesian = _BayesianTracker()
        self._engine: Optional[Any] = None
        self._performance_history: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._stage_times: Dict[str, List[float]] = defaultdict(list)
        # Selection cache: tensor_type:target_ratio:max_error -> MethodSelection
        self._selection_cache: Dict[str, MethodSelection] = {}
        self._cache_max_size: int = 256

    def bind_engine(self, engine: Any) -> None:
        self._engine = engine

    # ═════════════════════════════════════════════════════════════════════════
    #  MAIN ENTRY POINT — select_method
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _fast_bypass(tensor: np.ndarray, tensor_type: str) -> Optional[MethodSelection]:
        """Fast bypass for common tensor types — skips all selection stages.

        Proven best-method mappings:
        - norm → block_int4 + huffman
        - bias (1D) → raw passthrough
        - embedding → SVD low-rank + INT4 + huffman

        Returns a MethodSelection or None if no bypass applies.
        """
        if tensor_type == "norm" and tensor.ndim <= 1:
            return MethodSelection(
                name="block_int4",
                params={"block_size": 32},
                confidence=0.95,
                score=0.95,
                expected_ratio=4.0,
                expected_error=0.005,
                bypass_decision=BYPASS_HIGH_CONFIDENCE,
                stage="fast_bypass",
            )
        if tensor_type == "bias":
            return MethodSelection(
                name="passthrough",
                params={},
                confidence=1.0,
                score=1.0,
                expected_ratio=1.0,
                expected_error=0.0,
                bypass_decision=BYPASS_HIGH_CONFIDENCE,
                stage="fast_bypass",
            )
        if tensor_type == "embedding":
            return MethodSelection(
                name="svd_compress",
                params={"rank": min(64, min(tensor.shape) // 4)},
                confidence=0.9,
                score=0.9,
                expected_ratio=100.0,
                expected_error=0.008,
                bypass_decision=BYPASS_HIGH_CONFIDENCE,
                stage="fast_bypass",
            )
        return None

    def _make_cache_key(
        self, tensor_type: str, target_ratio: float, max_error: float
    ) -> str:
        return f"{tensor_type}:{target_ratio}:{max_error}"

    def select_method(
        self,
        tensor: np.ndarray,
        tensor_profile: Optional[Any] = None,
        tensor_type: str = "weight",
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        time_budget_ms: float = 100.0,
        rnd_mode: bool = False,
        name: str = "",
        bypass_threshold: float = 0.9,
    ) -> MethodSelection:
        """Select the best compression method for a tensor.

        Staged pipeline with early exit when bypass_confidence > threshold.
        Results are cached by (tensor_type, target_ratio, max_error).

        Parameters
        ----------
        tensor : np.ndarray
            The tensor to select methods for.
        tensor_profile : optional
            Pre-computed tensor profile.
        tensor_type : str
            Type of tensor (weight, attention_q, etc.).
        target_ratio : float
            Desired compression ratio.
        max_error : float
            Maximum acceptable error.
        time_budget_ms : float
            Time budget for selection in milliseconds.
        rnd_mode : bool
            R&D mode — allow exhaustive testing.
        name : str
            Tensor name for logging and fingerprinting.
        bypass_threshold : float
            Confidence threshold for early exit (default 0.9).

        Returns
        -------
        MethodSelection
            Selected method with params, confidence, and bypass decision.
        """
        t_start = time.perf_counter()
        features = self._extract_features(tensor, tensor_profile, tensor_type)

        # Fast bypass for common tensor types (norm, bias, embedding)
        bypass = self._fast_bypass(tensor, features.tensor_type)
        if bypass is not None:
            bypass.time_ms = (time.perf_counter() - t_start) * 1000
            return bypass

        # Check selection cache
        cache_key = self._make_cache_key(features.tensor_type, target_ratio, max_error)
        if cache_key in self._selection_cache:
            cached = self._selection_cache[cache_key]
            cached.time_ms = (time.perf_counter() - t_start) * 1000
            return cached

        best_selection: Optional[MethodSelection] = None

        # Stage 1: Holographic recall (0-1ms)
        if time_budget_ms >= 0.1:
            t1 = time.perf_counter()
            selection = self._stage1_holographic(tensor, features)
            et1 = (time.perf_counter() - t1) * 1000
            self._stage_times["holographic"].append(et1)
            if selection is not None and selection.confidence >= bypass_threshold:
                selection.time_ms = (time.perf_counter() - t_start) * 1000
                return selection
            best_selection = selection or best_selection

        # Stage 2: Zero-shot prediction + Bayesian posterior (1-10ms)
        if time_budget_ms >= 1.0:
            t2 = time.perf_counter()
            selection = self._stage2_zero_shot_bayesian(features, target_ratio, name)
            et2 = (time.perf_counter() - t2) * 1000
            self._stage_times["zero_shot"].append(et2)
            if selection is not None and selection.confidence >= bypass_threshold:
                selection.time_ms = (time.perf_counter() - t_start) * 1000
                return selection
            best_selection = selection or best_selection

        # Stage 3: Ensemble voting (10-50ms)
        if time_budget_ms >= 10.0:
            t3 = time.perf_counter()
            selection = self._stage3_ensemble_vote(
                tensor, features, target_ratio, max_error, name
            )
            et3 = (time.perf_counter() - t3) * 1000
            self._stage_times["ensemble"].append(et3)
            if selection is not None and selection.confidence >= bypass_threshold:
                selection.time_ms = (time.perf_counter() - t_start) * 1000
                return selection
            best_selection = selection or best_selection

        # Stage 4: Quantum superposition (100-500ms)
        if time_budget_ms >= 100.0:
            t4 = time.perf_counter()
            selection = self._stage4_superposition(
                tensor, features, target_ratio, max_error, name
            )
            et4 = (time.perf_counter() - t4) * 1000
            self._stage_times["superposition"].append(et4)
            if selection is not None:
                selection.time_ms = (time.perf_counter() - t_start) * 1000
                return selection
            best_selection = selection or best_selection

        # Stage 5: Exhaustive (R&D mode only, 1s+)
        if rnd_mode and time_budget_ms >= 1000.0:
            t5 = time.perf_counter()
            selection = self._stage5_exhaustive(tensor, features, target_ratio, name)
            et5 = (time.perf_counter() - t5) * 1000
            self._stage_times["exhaustive"].append(et5)
            if selection is not None:
                selection.time_ms = (time.perf_counter() - t_start) * 1000
                return selection
            best_selection = selection or best_selection

        elapsed = (time.perf_counter() - t_start) * 1000
        if best_selection is not None:
            best_selection.time_ms = elapsed
            # Cache result by tensor type
            if len(self._selection_cache) < self._cache_max_size:
                self._selection_cache[cache_key] = best_selection
            return best_selection
        fallback = MethodSelection(
            name="block_int8",
            params={"block_size": 128},
            confidence=0.5,
            stage="fallback",
            time_ms=elapsed,
            bypass_decision=TEST_FULL,
        )
        if len(self._selection_cache) < self._cache_max_size:
            self._selection_cache[cache_key] = fallback
        return fallback

    # ═════════════════════════════════════════════════════════════════════════
    #  PUBLIC METHODS
    # ═════════════════════════════════════════════════════════════════════════

    def test_in_superposition(
        self,
        tensor: np.ndarray,
        candidates: List[Dict[str, Any]],
        target_ratio: float,
        max_error: float,
    ) -> QuantumSuperpositionTest:
        """Test multiple methods in parallel (quantum superposition simulation)."""
        t0 = time.perf_counter()
        result = QuantumSuperpositionTest()
        result.method_names = [c["name"] for c in candidates]

        for cand in candidates:
            mname = cand["name"]
            inst = cand.get("instance")
            params = cand.get("params", {})
            if inst is None:
                continue
            t_method = time.perf_counter()
            try:
                if hasattr(inst, "compress"):
                    data, meta = inst.compress(tensor, **params)
                else:
                    continue
                if hasattr(inst, "decompress"):
                    recon = inst.decompress(data, meta)
                else:
                    continue
                if recon.shape != tensor.shape:
                    recon = recon.reshape(tensor.shape)
                var_val = float(np.var(tensor))
                mse = float(np.mean((tensor.ravel() - recon.ravel()) ** 2))
                rel_error = mse / var_val if var_val > 1e-30 else float(mse)
                ratio = tensor.nbytes / max(len(data), 1)
                elapsed = (time.perf_counter() - t_method) * 1000
                result.results[mname] = {
                    "ratio": ratio,
                    "error": rel_error,
                    "time_ms": elapsed,
                    "compressed_bytes": len(data),
                }
            except Exception as exc:
                logger.debug("Superposition test '%s' failed: %s", mname, exc)
                continue

        best_score = -1.0
        for mname, res in result.results.items():
            error_penalty = 1.0 / (1.0 + res["error"] * 100)
            ratio_bonus = min(res["ratio"] / max(target_ratio, 1.0), 1.0)
            score = 0.6 * error_penalty + 0.4 * ratio_bonus
            if score > best_score:
                best_score = score
                result.best_method = mname
        result.time_ms = (time.perf_counter() - t0) * 1000
        return result

    def ensemble_vote(
        self,
        tensor_profile: Optional[Any],
        target_ratio: float,
        max_error: float,
        tensor: Optional[np.ndarray] = None,
        tensor_type: str = "weight",
        name: str = "",
    ) -> Dict[str, float]:
        """All strategies vote on methods, weighted by past accuracy."""
        votes: Dict[str, float] = {}
        voter_count: Dict[str, int] = {}

        def _add_votes(voter: Dict[str, float], weight: float = 1.0) -> None:
            for mname, score in voter.items():
                votes[mname] = votes.get(mname, 0.0) + score * weight
                voter_count[mname] = voter_count.get(mname, 0) + 1

        all_methods = self._get_all_methods()

        # 1. Tier baseline (weight: 1.0)
        tier_votes = self._tier_baseline_vote(all_methods)
        _add_votes(tier_votes, 1.0)

        # 2. Category affinity (weight: 1.5)
        cat_votes = self._category_tier_vote(tensor_profile, tensor_type)
        _add_votes(cat_votes, 1.5)

        # 3. Bayesian posterior (weight: 1.5)
        bayes_votes = self._bayesian_vote(tensor_type, all_methods)
        _add_votes(bayes_votes, 1.5)

        # 4. Zero-shot predictor (weight: 1.0)
        if name and tensor is not None:
            zero_votes = self._zero_shot_vote(name, tensor.shape, target_ratio)
            _add_votes(zero_votes, 1.0)

        # 5. Decision tree (weight: 1.0)
        if tensor_profile is not None or tensor is not None:
            tree_votes = self._decision_tree_vote(tensor_profile, tensor)
            _add_votes(tree_votes, 1.0)

        # 6. Holographic memory (weight: 2.0)
        if tensor is not None:
            holo_votes = self._holographic_vote(tensor, tensor_type)
            _add_votes(holo_votes, 2.0)

        # Normalize
        normalized = {}
        for mname in votes:
            normalized[mname] = votes[mname] / max(voter_count.get(mname, 1), 1)
        return normalized

    def recall_holographic(
        self, tensor_signature: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """Associative memory recall from HolographicMemoryStore."""
        try:
            sig = _ResonanceSignature()
            vec = tensor_signature
            if vec.shape[0] >= 12:
                sig.mean = float(vec[0])
                sig.std = float(vec[1])
                sig.skewness = float(vec[2])
                sig.kurtosis = float(vec[3])
                sig.sparsity_1e3 = float(vec[4])
                sig.sparsity_1e4 = float(vec[5])
                sig.spectral_entropy = float(vec[6])
                sig.energy_concentration = float(vec[7])
                sig.effective_rank_ratio = float(vec[8])
                sig.n_elements_log = float(vec[9])
                sig.shape_ndim = int(vec[10])
                sig.shape_aspect = float(vec[11])
            recalled = self._holographic_memory.recall(sig, min_confidence=0.5)
            if recalled is not None:
                return (recalled["method_name"], recalled["confidence"])
            return None
        except Exception as exc:
            logger.debug("Holographic recall failed: %s", exc)
            return None

    def predict_zeroshot(self, fingerprint: np.ndarray) -> Dict[str, float]:
        """Zero-shot prediction placeholder — returns baseline."""
        _ = fingerprint
        return {"block_int8": 0.5}

    def query_bayesian(self, tensor_features: Dict[str, Any]) -> Dict[str, float]:
        """Bayesian posterior for each method given tensor features."""
        results: Dict[str, float] = {}
        try:
            tensor_type = tensor_features.get("tensor_type", "weight")
            all_methods = self._get_all_methods()
            for mname in all_methods:
                perf = self._bayesian.predict(mname, tensor_type)
                results[mname] = float(perf.score)
        except Exception:
            pass
        return results

    # ═════════════════════════════════════════════════════════════════════════
    #  RECORDING
    # ═════════════════════════════════════════════════════════════════════════

    def record_performance(
        self,
        tensor_type: str,
        method_name: str,
        ratio: float,
        error: float,
    ) -> None:
        """Track method performance for confidence-based decisions."""
        if tensor_type not in self._performance_history:
            self._performance_history[tensor_type] = {}
        if method_name not in self._performance_history[tensor_type]:
            self._performance_history[tensor_type][method_name] = {
                "n_tests": 0,
                "avg_error": 0.0,
                "avg_ratio": 0.0,
                "confidence": 0.0,
            }
        h = self._performance_history[tensor_type][method_name]
        n = h["n_tests"]
        h["avg_error"] = (h["avg_error"] * n + error) / (n + 1)
        h["avg_ratio"] = (h["avg_ratio"] * n + ratio) / (n + 1)
        h["n_tests"] = n + 1
        h["confidence"] = min(1.0, (n + 1) / 10.0) * max(
            0.0, 1.0 - h["avg_error"] * 10.0
        )
        self._bayesian.record(method_name, tensor_type, ratio, error)

    def record_compression(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        method_name: str,
        method_params: Dict[str, Any],
        ratio: float,
        error: float,
        name: str = "",
    ) -> None:
        """Record successful compression for holographic memory and Bayesian tracking."""
        sig = self._compute_signature(tensor, tensor_type)
        self._holographic_memory.store(sig, method_name, method_params, ratio, error)
        self._bayesian.record(method_name, tensor_type, ratio, error)
        self.record_performance(tensor_type, method_name, ratio, error)
        self._zero_shot.record_result(name, tensor.shape, method_name, 1.0 - error)

    # ═════════════════════════════════════════════════════════════════════════
    #  STATISTICS
    # ═════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for stage, times in self._stage_times.items():
            if times:
                stats[stage] = {
                    "n": len(times),
                    "mean_ms": float(np.mean(times)),
                    "min_ms": float(np.min(times)),
                    "max_ms": float(np.max(times)),
                    "p50_ms": float(np.median(times)),
                }
            else:
                stats[stage] = {"n": 0}
        stats["holographic_memory"] = self._holographic_memory.get_stats()
        stats["bayesian"] = {"tracked_pairs": len(self._bayesian._performances)}
        return stats

    def clear_cache(self) -> None:
        self._performance_history.clear()
        self._selection_cache.clear()

    # ── Stage 1: Holographic recall ────────────────────────────────────────

    def _stage1_holographic(
        self,
        tensor: np.ndarray,
        features: _TensorFeatures,
    ) -> Optional[MethodSelection]:
        try:
            sig = self._compute_signature(tensor, features.tensor_type)
            recalled = self._holographic_memory.recall(sig, min_confidence=0.5)
            if recalled is not None:
                conf = recalled["confidence"]
                return MethodSelection(
                    name=recalled["method_name"],
                    params=recalled.get("params", {}),
                    confidence=conf,
                    score=conf,
                    expected_ratio=recalled["ratio"],
                    expected_error=recalled["error"],
                    bypass_decision=(
                        BYPASS_HIGH_CONFIDENCE
                        if conf >= 0.9
                        else BYPASS_MEDIUM_CONFIDENCE
                        if conf >= 0.8
                        else TEST_FULL
                    ),
                    stage="holographic",
                )
        except Exception as exc:
            logger.debug("Stage 1 (holographic) failed: %s", exc)
        return None

    # ── Stage 2: Zero-shot + Bayesian ──────────────────────────────────────

    def _stage2_zero_shot_bayesian(
        self,
        features: _TensorFeatures,
        target_ratio: float,
        name: str,
    ) -> Optional[MethodSelection]:
        candidates: Dict[str, float] = {}

        if name:
            preds = self._zero_shot.predict(name, features.shape, target_ratio)
            for mname, _params, conf in preds:
                candidates[mname] = candidates.get(mname, 0.0) + conf * 1.5

        all_methods = self._get_all_method_names()
        for mname in all_methods[:30]:
            perf = self._bayesian.predict(mname, features.tensor_type)
            candidates[mname] = candidates.get(mname, 0.0) + perf.score * 1.2

        history = self._performance_history.get(features.tensor_type, {})
        for mname, h in history.items():
            if h["n_tests"] >= 2:
                candidates[mname] = candidates.get(mname, 0.0) + h["confidence"]

        if not candidates:
            return None
        best_name = max(candidates, key=candidates.get)
        best_score = candidates[best_name]
        confidence = min(1.0, best_score / 3.0)
        return MethodSelection(
            name=best_name,
            confidence=confidence,
            score=best_score,
            bypass_decision=(
                BYPASS_HIGH_CONFIDENCE if confidence >= 0.9 else TEST_FULL
            ),
            stage="zero_shot_bayesian",
        )

    # ── Stage 3: Ensemble voting ───────────────────────────────────────────

    def _stage3_ensemble_vote(
        self,
        tensor: np.ndarray,
        features: _TensorFeatures,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Optional[MethodSelection]:
        profile = _make_mock_profile(features, tensor)
        votes = self.ensemble_vote(
            tensor_profile=profile,
            target_ratio=target_ratio,
            max_error=max_error,
            tensor=tensor,
            tensor_type=features.tensor_type,
            name=name,
        )
        if not votes:
            return None
        best_name = max(votes, key=votes.get)
        best_score = votes[best_name]
        confidence = min(1.0, best_score)
        return MethodSelection(
            name=best_name,
            params=self.DEFAULT_PARAMS.get(best_name, {}),
            confidence=confidence,
            score=best_score,
            bypass_decision=(
                BYPASS_HIGH_CONFIDENCE if confidence >= 0.9 else TEST_FULL
            ),
            stage="ensemble_vote",
        )

    # ── Stage 4: Quantum superposition ─────────────────────────────────────

    def _stage4_superposition(
        self,
        tensor: np.ndarray,
        features: _TensorFeatures,
        target_ratio: float,
        max_error: float,
        name: str,
    ) -> Optional[MethodSelection]:
        profile = _make_mock_profile(features, tensor)
        votes = self.ensemble_vote(
            tensor_profile=profile,
            target_ratio=target_ratio,
            max_error=max_error,
            tensor=tensor,
            tensor_type=features.tensor_type,
            name=name,
        )
        if not votes:
            return None

        top_names = sorted(votes, key=votes.get, reverse=True)[:5]
        method_registry = self._get_all_methods()
        candidates = []
        for mname in top_names:
            inst = self._resolve_instance(mname, method_registry)
            if inst is not None:
                candidates.append(
                    {
                        "name": mname,
                        "instance": inst,
                        "params": self.DEFAULT_PARAMS.get(mname, {}),
                    }
                )

        if not candidates:
            return None

        test_result = self.test_in_superposition(
            tensor, candidates, target_ratio, max_error
        )
        if not test_result.best_method:
            return None

        best_res = test_result.results.get(test_result.best_method, {})
        error_val = best_res.get("error", 0.01)
        error_penalty = 1.0 / (1.0 + error_val * 100)
        confidence = min(1.0, error_penalty * 0.8)

        return MethodSelection(
            name=test_result.best_method,
            params=self.DEFAULT_PARAMS.get(test_result.best_method, {}),
            confidence=confidence,
            score=confidence,
            expected_ratio=best_res.get("ratio", 10.0),
            expected_error=error_val,
            bypass_decision=TEST_FULL,
            stage="superposition",
        )

    # ── Stage 5: Exhaustive ────────────────────────────────────────────────

    def _stage5_exhaustive(
        self,
        tensor: np.ndarray,
        features: _TensorFeatures,
        target_ratio: float,
        name: str,
    ) -> Optional[MethodSelection]:
        all_methods = self._get_all_methods()
        if not all_methods:
            return None

        best_name = ""
        best_score_val = -1.0
        best_ratio = 1.0
        best_error = 1.0

        for mname, minfo in all_methods.items():
            inst = self._resolve_instance(mname, all_methods)
            if (
                inst is None
                or not hasattr(inst, "compress")
                or not hasattr(inst, "decompress")
            ):
                continue
            try:
                data, meta = inst.compress(tensor)
                recon = inst.decompress(data, meta)
                if recon.shape != tensor.shape:
                    recon = recon.reshape(tensor.shape)
                var_val = float(np.var(tensor))
                mse = float(np.mean((tensor.ravel() - recon.ravel()) ** 2))
                rel_error = mse / var_val if var_val > 1e-30 else float(mse)
                ratio = tensor.nbytes / max(len(data), 1)
                score = ratio / max(rel_error, 1e-10)
                if score > best_score_val and ratio > 1.0:
                    best_score_val = score
                    best_name = mname
                    best_ratio = ratio
                    best_error = rel_error
            except Exception:
                continue

        if not best_name:
            return None

        return MethodSelection(
            name=best_name,
            params=self.DEFAULT_PARAMS.get(best_name, {}),
            confidence=min(1.0, best_score_val / 1000.0),
            score=best_score_val,
            expected_ratio=best_ratio,
            expected_error=best_error,
            bypass_decision=TEST_FULL,
            stage="exhaustive",
        )

    # ── Vote helpers ───────────────────────────────────────────────────────

    def _tier_baseline_vote(
        self, all_methods: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        votes: Dict[str, float] = {}
        for mname, minfo in all_methods.items():
            tier = minfo.get("tier", 5)
            try:
                tval = tier.value if hasattr(tier, "value") else int(tier)
            except (ValueError, TypeError):
                tval = 5
            votes[mname] = max(0, 5 - tval) * 0.2
        return votes

    def _category_tier_vote(
        self,
        profile: Optional[Any],
        tensor_type: str,
    ) -> Dict[str, float]:
        votes: Dict[str, float] = {}
        all_methods = self._get_all_methods()
        if not all_methods:
            return votes
        best_cat = self.TYPE_CATEGORY.get(tensor_type, "quantization")
        for mname, minfo in all_methods.items():
            cat = minfo.get("category", "quantization")
            tier = minfo.get("tier", 5)
            try:
                tval = tier.value if hasattr(tier, "value") else int(tier)
            except (ValueError, TypeError):
                tval = 5
            score = 0.5 if cat == best_cat else 0.0
            score += max(0, 5 - tval) * 0.1
            votes[mname] = score
        return votes

    def _bayesian_vote(
        self, tensor_type: str, all_methods: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        votes: Dict[str, float] = {}
        for mname in all_methods:
            perf = self._bayesian.predict(mname, tensor_type)
            votes[mname] = perf.score
        return votes

    def _zero_shot_vote(
        self, name: str, shape: Tuple[int, ...], target_ratio: float
    ) -> Dict[str, float]:
        preds = self._zero_shot.predict(name, shape, target_ratio)
        return {mname: conf for mname, _params, conf in preds}

    def _holographic_vote(
        self, tensor: np.ndarray, tensor_type: str
    ) -> Dict[str, float]:
        try:
            sig = self._compute_signature(tensor, tensor_type)
            recalled = self._holographic_memory.recall(sig, min_confidence=0.3)
            if recalled is not None:
                return {recalled["method_name"]: recalled["confidence"] * 2.0}
        except Exception:
            pass
        return {}

    def _decision_tree_vote(
        self,
        profile: Optional[Any],
        tensor: Optional[np.ndarray],
    ) -> Dict[str, float]:
        votes: Dict[str, float] = {}
        if profile is None and tensor is None:
            return votes
        sparsity = getattr(profile, "sparsity", 0.0) if profile else 0.0
        ndim = getattr(profile, "ndim", tensor.ndim if tensor is not None else 2)
        eff_rank = getattr(profile, "effective_rank", 0.5) if profile else 0.5
        dct_conc = getattr(profile, "dct_concentration", 0.5) if profile else 0.5
        if sparsity > 0.85:
            votes["sparsify"] = 0.9
            votes["block_sparsity"] = 0.8
        elif ndim == 2 and 0 < eff_rank < 0.3:
            votes["svd_compress"] = 0.9
            votes["tensor_train"] = 0.8
        elif dct_conc < 0.25:
            votes["dct_spectral"] = 0.9
        else:
            votes["block_int8"] = 0.7
            votes["dct_spectral"] = 0.6
        return votes

    # ── Helper methods ─────────────────────────────────────────────────────

    def _extract_features(
        self,
        tensor: np.ndarray,
        profile: Optional[Any],
        tensor_type: str,
    ) -> _TensorFeatures:
        if profile is not None:
            return _TensorFeatures(
                n_elements=getattr(profile, "n_elements", tensor.size),
                ndim=tensor.ndim,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                sparsity=getattr(profile, "sparsity", 0.0),
                mean_abs=getattr(profile, "mean_abs", float(np.mean(np.abs(tensor)))),
                std=getattr(profile, "std", float(np.std(tensor))),
                min_val=getattr(profile, "min_val", float(np.min(tensor))),
                max_val=getattr(profile, "max_val", float(np.max(tensor))),
                mean=getattr(profile, "mean", float(np.mean(tensor))),
                kurtosis=getattr(profile, "kurtosis", 0.0),
                skewness=getattr(profile, "skewness", 0.0),
                spectral_entropy=getattr(profile, "spectral_entropy", 0.5),
                dct_concentration=getattr(profile, "dct_concentration", 0.5),
                energy_concentration=getattr(profile, "energy_concentration", 0.5),
                effective_rank=getattr(profile, "effective_rank", 0.5),
                value_range=float(np.max(tensor) - np.min(tensor)),
                snr_estimate=20.0,
                tensor_type=tensor_type,
                sensitivity=getattr(profile, "sensitivity", 0.5),
                compressibility_score=getattr(profile, "compressibility_score", 0.5),
                outlier_ratio_3sigma=getattr(profile, "outlier_ratio_3sigma", 0.01),
            )

        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        if n == 0:
            return _TensorFeatures(tensor_type=tensor_type)

        sparsity = float(np.mean(np.abs(flat) < 1e-10))
        mean_abs = float(np.mean(np.abs(flat)))
        std = float(np.std(flat))

        spectral_entropy = 0.5
        dct_conc = 0.5
        if n >= 16:
            try:
                sample = flat[: min(n, 4096)]
                coeffs = np.fft.fft(sample)
                power = np.abs(coeffs) ** 2
                total_power = float(np.sum(power))
                if total_power > 1e-10:
                    power_dist = power / total_power
                    spectral_entropy = -float(
                        np.sum(power_dist * np.log2(power_dist + 1e-30))
                    )
                    max_ent = np.log2(len(power))
                    spectral_entropy = (
                        spectral_entropy / max_ent if max_ent > 0 else 0.5
                    )
                    sorted_power = np.sort(power)[::-1]
                    cumsum = np.cumsum(sorted_power) / total_power
                    n_top10 = max(1, len(power) // 10)
                    dct_conc = float(np.sum(power[:n_top10]) / total_power)
            except Exception:
                pass

        eff_rank = 0.5
        if tensor.ndim >= 2 and min(tensor.shape) >= 4:
            try:
                s = np.linalg.svd(
                    tensor[: min(64, tensor.shape[0]), : min(64, tensor.shape[1])],
                    compute_uv=False,
                )
                s_norm = s / (np.sum(s) + 1e-10)
                nnz = s_norm[s_norm > 1e-10]
                if len(nnz) > 0:
                    eff_rank = float(np.exp(-np.sum(nnz * np.log(nnz + 1e-30))))
            except Exception:
                pass

        return _TensorFeatures(
            n_elements=n,
            ndim=tensor.ndim,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            sparsity=sparsity,
            mean_abs=mean_abs,
            std=std,
            mean=float(np.mean(flat)),
            min_val=float(np.min(flat)),
            max_val=float(np.max(flat)),
            spectral_entropy=spectral_entropy,
            dct_concentration=dct_conc,
            energy_concentration=dct_conc,
            effective_rank=eff_rank,
            value_range=float(np.max(flat) - np.min(flat)),
            tensor_type=tensor_type,
        )

    def _compute_signature(
        self, tensor: np.ndarray, tensor_type: str
    ) -> _ResonanceSignature:
        flat = tensor.ravel().astype(np.float64)
        n = flat.size
        sample = flat[: min(n, 10000)]
        sg = _ResonanceSignature(tensor_type=tensor_type)
        if len(sample) == 0:
            return sg
        sg.mean = float(np.mean(sample))
        sg.std = float(np.std(sample))
        if sg.std > 1e-30:
            z = (sample - sg.mean) / sg.std
            sg.skewness = float(np.mean(z**3))
            sg.kurtosis = float(np.mean(z**4)) - 3.0
        sg.sparsity_1e3 = float(np.mean(np.abs(sample) < 0.001))
        sg.sparsity_1e4 = float(np.mean(np.abs(sample) < 0.0001))
        sg.n_elements_log = np.log10(max(n, 1))
        sg.shape_ndim = tensor.ndim
        if tensor.ndim >= 2:
            sg.shape_aspect = max(tensor.shape) / max(min(tensor.shape), 1)
        if len(sample) >= 16:
            try:
                dct_input = sample[: min(1024, len(sample))]
                dct_coeffs = self._lightweight_dct(dct_input)
                dct_energy = dct_coeffs**2
                total_energy = float(np.sum(dct_energy))
                if total_energy > 1e-30:
                    dist = dct_energy / (total_energy + 1e-30)
                    sg.spectral_entropy = -float(np.sum(dist * np.log2(dist + 1e-30)))
                    max_ent = np.log2(len(dct_coeffs))
                    sg.spectral_entropy = (
                        sg.spectral_entropy / max_ent if max_ent > 0 else 0.0
                    )
                    n_top = max(1, len(dct_coeffs) // 10)
                    top_energy = float(np.sum(np.sort(dct_energy)[-n_top:]))
                    sg.energy_concentration = top_energy / (total_energy + 1e-30)
            except Exception:
                pass
        if tensor.ndim >= 2 and min(tensor.shape) >= 4:
            try:
                sv_sample = tensor[
                    : min(64, tensor.shape[0]), : min(64, tensor.shape[1])
                ]
                s = np.linalg.svd(sv_sample, compute_uv=False)
                s_sum = float(np.sum(s))
                if s_sum > 1e-30:
                    s_norm = s / s_sum
                    eff_rank = float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-30))))
                    sg.effective_rank_ratio = eff_rank / min(sv_sample.shape)
            except Exception:
                pass
        sg._tensor_shape = tensor.shape
        return sg

    @staticmethod
    def _lightweight_dct(x: np.ndarray) -> np.ndarray:
        n = len(x)
        x2 = np.zeros(2 * n, dtype=np.float64)
        x2[:n] = x
        x2[n:] = x[::-1]
        fft = np.fft.fft(x2)[:n]
        scale = np.sqrt(2.0 / n)
        coeffs = fft.real * scale
        coeffs[0] *= 1.0 / np.sqrt(2.0)
        return coeffs

    def _get_all_methods(self) -> Dict[str, Dict[str, Any]]:
        if self._method_registry:
            return self._method_registry
        if self._engine is not None and hasattr(self._engine, "get_methods"):
            try:
                return self._engine.get_methods()
            except Exception:
                pass
        if self._engine is not None and hasattr(self._engine, "_methods"):
            try:
                methods = {}
                for name in self._engine._methods:
                    methods[name] = {
                        "instance": self._engine._methods[name],
                        "category": getattr(
                            self._engine._methods[name], "category", "quantization"
                        ),
                        "tier": getattr(self._engine._methods[name], "_tier_val", 5),
                    }
                return methods
            except Exception:
                pass
        return {}

    def _get_all_method_names(self) -> List[str]:
        return list(self._get_all_methods().keys())

    @staticmethod
    def _resolve_instance(mname: str, registry: Dict[str, Dict[str, Any]]) -> Any:
        minfo = registry.get(mname, {})
        inst = minfo.get("instance")
        if inst is not None:
            return inst
        cls = minfo.get("class")
        if cls is not None:
            try:
                inst = cls() if isinstance(cls, type) else cls
                minfo["instance"] = inst
                return inst
            except Exception:
                pass
        return None


# ── Utility ───────────────────────────────────────────────────────────────────


def _make_mock_profile(
    features: Optional[Any], tensor: Optional[np.ndarray] = None
) -> Any:
    if features is None and tensor is None:
        return None
    flat = tensor.ravel() if tensor is not None else np.array([0.0])
    n = flat.size if tensor is not None else 1

    class MockProfile:
        pass

    p = MockProfile()
    p.shape = getattr(features, "shape", tensor.shape if tensor is not None else (1,))
    p.n_elements = getattr(features, "n_elements", n)
    p.nbytes = (
        getattr(features, "nbytes", n * 4) if hasattr(features, "nbytes") else n * 4
    )
    p.ndim = getattr(features, "ndim", tensor.ndim if tensor is not None else 1)
    p.dtype = getattr(
        features, "dtype", str(tensor.dtype) if tensor is not None else "float32"
    )
    p.mean = getattr(features, "mean", float(np.mean(flat)))
    p.std = getattr(features, "std", float(np.std(flat)))
    p.min_val = float(np.min(flat))
    p.max_val = float(np.max(flat))
    p.sparsity = getattr(features, "sparsity", 0.0)
    p.effective_rank = getattr(features, "effective_rank", 0.5)
    p.energy_concentration = getattr(features, "energy_concentration", 0.5)
    p.spectral_entropy = getattr(features, "spectral_entropy", 0.5)
    p.sensitivity = getattr(features, "sensitivity", 0.5)
    p.compressibility_score = getattr(features, "compressibility_score", 0.5)
    p.spectral_decay_rate = 0.5
    p.entropy_rate = p.spectral_entropy
    p.nm_sparsity_score = p.sparsity
    p.recommended_method = "block_int8"
    p.recommended_bits = 8
    p.kurtosis = getattr(features, "kurtosis", 0.0)
    p.skewness = getattr(features, "skewness", 0.0)
    p.tensor_type = getattr(features, "tensor_type", "weight")
    p.outlier_ratio_3sigma = getattr(features, "outlier_ratio_3sigma", 0.01)
    return p
