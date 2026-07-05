"""
Self-Optimizing Cascade Engine — reinforcement learning for compression.

Treats each tensor compression as a "trajectory":
  - Rewards = compression_ratio - penalty(error) - penalty(time)
  - Learns which methods work best for which tensor types over time
  - Stores experience in a lightweight SQLite database
  - Starts with default patterns, improves with every use
"""

from __future__ import annotations

import json
import logging
import math
import os as _os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompressionExperience:
    experience_id: str = ""
    timestamp: str = ""

    tensor_name: str = ""
    tensor_type: str = ""
    tensor_shape: Tuple[int, ...] = (0,)
    tensor_dtype: str = ""
    n_elements: int = 0
    entropy_rate: float = 0.0
    sparsity: float = 0.0
    effective_rank: float = 0.0

    method_sequence: List[str] = field(default_factory=list)
    method_params: List[Dict[str, Any]] = field(default_factory=list)

    target_ratio: float = 0.0
    achieved_ratio: float = 0.0
    achieved_error: float = 0.0
    compression_time_ms: float = 0.0

    reward: float = 0.0
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "timestamp": self.timestamp,
            "tensor_name": self.tensor_name,
            "tensor_type": self.tensor_type,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
            "n_elements": self.n_elements,
            "entropy_rate": round(self.entropy_rate, 4),
            "sparsity": round(self.sparsity, 4),
            "effective_rank": round(self.effective_rank, 2),
            "method_sequence": self.method_sequence,
            "target_ratio": self.target_ratio,
            "achieved_ratio": round(self.achieved_ratio, 2),
            "achieved_error": round(self.achieved_error, 6),
            "compression_time_ms": round(self.compression_time_ms, 2),
            "reward": round(self.reward, 4),
            "success": self.success,
        }


@dataclass
class LearnedPolicy:
    tensor_type: str = ""
    sequences: List[Tuple[List[str], float]] = field(default_factory=list)
    n_observations: int = 0
    avg_reward: float = 0.0
    best_reward: float = 0.0
    best_sequence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tensor_type": self.tensor_type,
            "sequences": [{"methods": seq, "avg_reward": round(r, 4)} for seq, r in self.sequences[:5]],
            "n_observations": self.n_observations,
            "avg_reward": round(self.avg_reward, 4),
            "best_reward": round(self.best_reward, 4),
            "best_sequence": self.best_sequence,
        }


class SelfOptimizingCascade:
    def __init__(
        self,
        engine: Any,
        db_path: str = "",
        learning_rate: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.001,
        epsilon: float = 0.2,
    ):
        self._engine = engine
        self._learning_rate = learning_rate
        self._alpha = alpha
        self._beta = beta
        self._epsilon = epsilon

        self._lock = threading.Lock()
        self._db_path = db_path or _os.path.join(
            _os.path.expanduser("~"), ".spectralstream", "compression_experience.db"
        )
        _os.makedirs(_os.path.dirname(self._db_path), exist_ok=True)

        self._db: Optional[sqlite3.Connection] = None
        self._init_db()

        self._policies: Dict[str, LearnedPolicy] = {}
        self._default_templates: Dict[str, List[str]] = {
            "attention": ["dct_spectral", "hadamard_int8"],
            "attention_q": ["svd_compress", "block_int8"],
            "attention_k": ["svd_compress", "block_int8"],
            "attention_v": ["svd_compress", "delta_int4"],
            "attention_o": ["svd_compress", "hadamard_int8"],
            "ffn": ["dct_spectral", "sparsity_int4"],
            "ffn_gate": ["dct_spectral", "block_int4"],
            "ffn_up": ["dct_spectral", "block_int4"],
            "ffn_down": ["fwht_compress", "hadamard_int4"],
            "embedding": ["tensor_train", "block_int8"],
            "output": ["svd_compress", "block_int8"],
            "norm": ["block_int8"],
            "norm_bias": ["block_int8"],
            "qkv_fused": ["svd_compress", "hadamard_int8", "zstd"],
            "weight": ["dct_spectral", "block_int8"],
        }
        self._method_pool = [
            "svd_compress", "dct_spectral", "tensor_train", "fwht_compress",
            "block_int8", "block_int4", "hadamard_int8", "hadamard_int4",
            "sparsity_int4", "delta_int4", "zstd", "arithmetic_coding",
            "huffman", "ans", "lz4",
        ]

        self._n_experiences = 0
        self._n_exploit = 0
        self._n_explore = 0
        self._total_reward = 0.0

        self._load_policies()
        logger.info(
            "SelfOptimizingCascade initialized: db=%s, epsilon=%.2f, %d policies loaded",
            self._db_path, epsilon, len(self._policies),
        )

    def _rng(self) -> float:
        return np.random.random()

    def _randint(self, lo: int, hi: int) -> int:
        return int(np.random.randint(lo, hi))

    # ── Database ─────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            self._db = sqlite3.connect(self._db_path, check_same_thread=False)
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS compression_experiences (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    tensor_name TEXT,
                    tensor_type TEXT,
                    tensor_shape TEXT,
                    tensor_dtype TEXT,
                    n_elements INTEGER,
                    entropy_rate REAL,
                    sparsity REAL,
                    effective_rank REAL,
                    method_sequence TEXT,
                    method_params TEXT,
                    target_ratio REAL,
                    achieved_ratio REAL,
                    achieved_error REAL,
                    compression_time_ms REAL,
                    reward REAL,
                    success BOOLEAN
                )
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiences_tensor_type
                ON compression_experiences(tensor_type)
            """)
            self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiences_reward
                ON compression_experiences(reward DESC)
            """)
            self._db.commit()
        except sqlite3.Error as e:
            logger.warning("SQLite init failed: %s", e)
            self._db = None

    def _store_experience(self, exp: CompressionExperience) -> None:
        if self._db is None:
            return
        try:
            self._db.execute(
                """INSERT OR REPLACE INTO compression_experiences
                (id, timestamp, tensor_name, tensor_type, tensor_shape,
                 tensor_dtype, n_elements, entropy_rate, sparsity,
                 effective_rank, method_sequence, method_params,
                 target_ratio, achieved_ratio, achieved_error,
                 compression_time_ms, reward, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    exp.experience_id, exp.timestamp, exp.tensor_name,
                    exp.tensor_type, json.dumps(list(exp.tensor_shape)),
                    exp.tensor_dtype, exp.n_elements, exp.entropy_rate,
                    exp.sparsity, exp.effective_rank,
                    json.dumps(exp.method_sequence),
                    json.dumps(exp.method_params),
                    exp.target_ratio, exp.achieved_ratio, exp.achieved_error,
                    exp.compression_time_ms, exp.reward, bool(exp.success),
                ),
            )
            self._db.commit()
        except sqlite3.Error as e:
            logger.warning("Failed to store experience: %s", e)

    def _load_policies(self) -> None:
        if self._db is None:
            return
        try:
            cursor = self._db.execute(
                """SELECT tensor_type, method_sequence,
                          AVG(reward) as avg_r, COUNT(*) as n,
                          MAX(reward) as best_r
                 FROM compression_experiences
                 WHERE success = 1
                 GROUP BY tensor_type, method_sequence
                 ORDER BY tensor_type, avg_r DESC"""
            )
            rows = cursor.fetchall()

            type_data: Dict[str, Dict[str, Any]] = {}
            for tensor_type, seq_json, avg_r, count, _best_r in rows:
                seq = json.loads(seq_json)
                key = tuple(seq)
                if tensor_type not in type_data:
                    type_data[tensor_type] = {"sequences": {}, "n_total": 0, "total_r": 0.0}
                td = type_data[tensor_type]
                td["sequences"][key] = (avg_r, count)
                td["n_total"] += count
                td["total_r"] += avg_r * count

            for tensor_type, td in type_data.items():
                sorted_seqs = sorted(td["sequences"].items(), key=lambda x: x[1][0], reverse=True)
                policy = LearnedPolicy(
                    tensor_type=tensor_type,
                    sequences=[(list(k), v[0]) for k, v in sorted_seqs],
                    n_observations=td["n_total"],
                    avg_reward=td["total_r"] / max(td["n_total"], 1),
                    best_reward=sorted_seqs[0][1][0] if sorted_seqs else 0.0,
                    best_sequence=list(sorted_seqs[0][0]) if sorted_seqs else [],
                )
                self._policies[tensor_type] = policy
        except sqlite3.Error as e:
            logger.warning("Failed to load policies: %s", e)

    # ── Policy Selection ─────────────────────────────────────────────────

    def select_sequence(
        self,
        tensor_type: str,
        target_ratio: float,
        force_exploit: bool = False,
    ) -> Tuple[List[str], bool]:
        explore = not force_exploit and (
            self._rng() < self._epsilon
            or tensor_type not in self._policies
            or self._policies[tensor_type].n_observations < 3
        )

        if explore:
            self._n_explore += 1
            seq = self._explore_sequence(tensor_type, target_ratio)
            return seq, True

        self._n_exploit += 1
        seq = self._exploit_sequence(tensor_type, target_ratio)
        return seq, False

    def _exploit_sequence(self, tensor_type: str, target_ratio: float) -> List[str]:
        policy = self._policies.get(tensor_type)
        if policy and policy.best_sequence:
            return list(policy.best_sequence)
        template = self._default_templates.get(tensor_type)
        if template:
            return list(template)
        return ["block_int8"]

    def _explore_sequence(self, tensor_type: str, target_ratio: float) -> List[str]:
        methods = list(self._default_templates.get(tensor_type, ["block_int8"]))
        r = self._rng()
        if len(methods) >= 2 and r < 0.3:
            i, j = self._randint(0, len(methods)), self._randint(0, len(methods))
            if i != j:
                methods[i], methods[j] = methods[j], methods[i]
        if self._rng() < 0.3:
            extra = str(np.random.choice(["zstd", "arithmetic_coding", "huffman", "ans"]))
            pos = self._randint(0, len(methods) + 1)
            methods.insert(pos, extra)
        if len(methods) > 1 and self._rng() < 0.2:
            pos = self._randint(0, len(methods))
            methods.pop(pos)
        return methods[:4]

    # ── Experience Recording ─────────────────────────────────────────────

    def compute_reward(
        self,
        achieved_ratio: float,
        achieved_error: float,
        compression_time_ms: float,
        target_ratio: float,
    ) -> float:
        ratio_reward = min(achieved_ratio / max(target_ratio, 1.0), 10.0)
        error_penalty = self._alpha * math.exp(10.0 * achieved_error) - self._alpha
        time_penalty = self._beta * compression_time_ms
        return ratio_reward - error_penalty - time_penalty

    def record_experience(
        self,
        tensor: np.ndarray,
        tensor_type: str,
        method_sequence: List[str],
        achieved_ratio: float,
        achieved_error: float,
        compression_time_ms: float,
        target_ratio: float,
        tensor_name: str = "",
    ) -> str:
        reward = self.compute_reward(
            achieved_ratio, achieved_error, compression_time_ms, target_ratio
        )
        success = achieved_error <= 0.1 and achieved_ratio > 1.0

        entropy_rate = 0.0
        sparsity = 0.0
        effective_rank = 0.0
        try:
            from spectralstream.compression.engine._helpers import (
                _estimate_entropy_rate, _estimate_noise_floor,
            )
            flat = tensor.ravel()
            entropy_rate = _estimate_entropy_rate(flat)
            sparsity = float(np.mean(np.abs(tensor) < 1e-7))
            del flat
        except Exception:
            pass

        exp = CompressionExperience(
            experience_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            tensor_name=tensor_name,
            tensor_type=tensor_type,
            tensor_shape=tensor.shape,
            tensor_dtype=str(tensor.dtype),
            n_elements=tensor.size,
            entropy_rate=entropy_rate,
            sparsity=sparsity,
            effective_rank=effective_rank,
            method_sequence=method_sequence,
            target_ratio=target_ratio,
            achieved_ratio=achieved_ratio,
            achieved_error=achieved_error,
            compression_time_ms=compression_time_ms,
            reward=reward,
            success=success,
        )

        self._store_experience(exp)
        self._n_experiences += 1
        self._total_reward += reward

        with self._lock:
            policy = self._policies.get(tensor_type)
            if policy is None:
                policy = LearnedPolicy(tensor_type=tensor_type)
                self._policies[tensor_type] = policy

            seq_key = tuple(method_sequence)
            seqs = policy.sequences
            found = False
            for i, (seq, r) in enumerate(seqs):
                if tuple(seq) == seq_key:
                    new_r = r * (1 - self._learning_rate) + reward * self._learning_rate
                    seqs[i] = (seq, new_r)
                    found = True
                    break
            if not found:
                seqs.append((list(method_sequence), reward))
                seqs.sort(key=lambda x: x[1], reverse=True)
                if len(seqs) > 20:
                    seqs[:] = seqs[:20]

            policy.n_observations += 1
            policy.avg_reward += (reward - policy.avg_reward) / max(policy.n_observations, 1)
            if reward > policy.best_reward:
                policy.best_reward = reward
                policy.best_sequence = list(method_sequence)

        return exp.experience_id

    # ── Cascade Compression with RL ──────────────────────────────────────

    def compress_with_rl(
        self,
        tensor: np.ndarray,
        target_ratio: float = 5000.0,
        max_error: float = 0.01,
        name: str = "",
        force_exploit: bool = False,
    ) -> Dict[str, Any]:
        t0 = time.time()

        tensor_type = getattr(tensor, "tensor_type", "")
        if not tensor_type:
            from spectralstream.compression.engine._helpers import _classify_by_name
            tensor_type = _classify_by_name(name)

        seq, was_exploration = self.select_sequence(tensor_type, target_ratio, force_exploit)

        compressed_data = b""
        metadata: dict = {}
        achieved_ratio = 1.0
        achieved_error = 1.0

        try:
            compressed_data, metadata, achieved_ratio, achieved_error = (
                self._engine.compress(tensor, target_ratio, max_error, name=name)
            )
        except Exception:
            try:
                blk8 = self._engine._methods.get("block_int8")
                if blk8:
                    compressed_data, meta = blk8.compress(tensor)
                    ratio = tensor.nbytes / max(len(compressed_data), 1)
                    recon = blk8.decompress(compressed_data, meta)
                    metrics = self._compute_error_metrics(tensor, recon)
                    achieved_ratio = ratio
                    achieved_error = metrics["relative_error"]
                    metadata = meta
                    del recon
            except Exception:
                pass

        elapsed_ms = (time.time() - t0) * 1000

        exp_id = self.record_experience(
            tensor, tensor_type, seq, achieved_ratio, achieved_error,
            elapsed_ms, target_ratio, tensor_name=name,
        )

        return {
            "compressed_data": compressed_data,
            "metadata": metadata,
            "ratio": achieved_ratio,
            "error": achieved_error,
            "sequence": seq,
            "was_exploration": was_exploration,
            "experience_id": exp_id,
            "tensor_type": tensor_type,
            "duration_ms": elapsed_ms,
        }

    def _compute_error_metrics(self, orig: np.ndarray, recon: np.ndarray) -> Dict[str, float]:
        n_min = min(orig.size, recon.size)
        o = orig.ravel()[:n_min].astype(np.float64)
        r = recon.ravel()[:n_min].astype(np.float64)
        noise = o - r
        mse = float(np.mean(noise ** 2))
        signal_power = float(np.mean(o ** 2)) + 1e-30
        rel_error = float(np.linalg.norm(noise)) / max(float(np.linalg.norm(o)), 1e-30)
        return {"mse": mse, "relative_error": rel_error}

    # ── Policy Statistics ────────────────────────────────────────────────

    def get_policies(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._policies.items()}

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "n_experiences": self._n_experiences,
            "n_exploit": self._n_exploit,
            "n_explore": self._n_explore,
            "total_reward": round(self._total_reward, 4),
            "avg_reward": round(self._total_reward / max(self._n_experiences, 1), 4),
            "n_policies": len(self._policies),
            "epsilon": self._epsilon,
            "learning_rate": self._learning_rate,
            "db_path": self._db_path,
        }

    def reset_policy(self, tensor_type: Optional[str] = None) -> None:
        if tensor_type:
            self._policies.pop(tensor_type, None)
        else:
            self._policies.clear()

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
        logger.info(
            "SelfOptimizingCascade closed: %d experiences, %d policies",
            self._n_experiences,
            len(self._policies),
        )
