from __future__ import annotations

import math
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np

from spectralstream.core.math_primitives.transforms import dct, idct, dct_2d, idct_2d


class ZigzagOrder:
    @staticmethod
    def scan_indices(shape: tuple[int, int]) -> np.ndarray:
        rows, cols = shape
        indices = []
        for s in range(rows + cols - 1):
            if s % 2 == 0:
                i = min(s, rows - 1)
                j = s - i
                while i >= 0 and j < cols:
                    indices.append(i * cols + j)
                    i -= 1
                    j += 1
            else:
                j = min(s, cols - 1)
                i = s - j
                while j >= 0 and i < rows:
                    indices.append(i * cols + j)
                    i += 1
                    j -= 1
        return np.array(indices, dtype=np.int32)

    @staticmethod
    def reorder_by_frequency(dct_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shape = dct_matrix.shape
        order = ZigzagOrder.scan_indices(shape)
        flat = dct_matrix.ravel()
        return flat[order].astype(np.float32), order

    @staticmethod
    def assemble_from_frequency(
        freq_values: np.ndarray,
        shape: tuple[int, int],
        n_coefficients: Optional[int] = None,
    ) -> np.ndarray:
        order = ZigzagOrder.scan_indices(shape)
        if n_coefficients is not None:
            n = min(n_coefficients, len(freq_values), len(order))
        else:
            n = len(freq_values)
        dct_flat = np.zeros(np.prod(shape), dtype=np.float64)
        dct_flat[order[:n]] = freq_values[:n]
        return dct_flat.reshape(shape).astype(np.float32)


@dataclass
class DCTWeightPacker:
    name: str
    shape: tuple[int, ...]
    freq_values: np.ndarray
    total_coefficients: int
    mean: float = 0.0
    is_1d: bool = False
    dtype: str = "float32"

    def reconstruct(self, n_coefficients: Optional[int] = None) -> np.ndarray:
        if self.is_1d:
            n = (
                min(n_coefficients, len(self.freq_values))
                if n_coefficients is not None
                else len(self.freq_values)
            )
            dct_coeffs = np.zeros(self.shape[-1], dtype=np.float64)
            dct_coeffs[:n] = self.freq_values[:n]
            signal = idct(dct_coeffs)
            return signal + self.mean
        else:
            dct_matrix = ZigzagOrder.assemble_from_frequency(
                self.freq_values, self.shape, n_coefficients
            )
            signal = idct_2d(dct_matrix)
            return signal + self.mean

    def quality_at(self, n_coefficients: int) -> float:
        n = min(n_coefficients, len(self.freq_values))
        if n <= 0:
            return 0.0
        total_energy = float(np.sum(self.freq_values.astype(np.float64) ** 2))
        if total_energy < 1e-30:
            return 1.0
        cum_energy = float(np.sum(self.freq_values[:n].astype(np.float64) ** 2))
        return cum_energy / total_energy

    @staticmethod
    def from_tensor(name: str, tensor: np.ndarray) -> DCTWeightPacker:
        data = tensor.astype(np.float64)
        shape = data.shape
        is_1d = data.ndim == 1
        if is_1d:
            mean = float(np.mean(data))
            centered = data - mean
            dct_coeffs = dct(centered)
            flat = dct_coeffs.ravel().astype(np.float32)
            return DCTWeightPacker(
                name=name,
                shape=shape,
                freq_values=flat,
                total_coefficients=flat.size,
                mean=mean,
                is_1d=True,
            )
        else:
            mean = float(np.mean(data))
            centered = data - mean
            dct_coeffs = dct_2d(centered)
            freq_values, _ = ZigzagOrder.reorder_by_frequency(dct_coeffs)
            return DCTWeightPacker(
                name=name,
                shape=shape,
                freq_values=freq_values,
                total_coefficients=int(np.prod(shape)),
                mean=mean,
                is_1d=False,
            )


@dataclass
class ProgressiveStage:
    name: str
    coeff_fraction: float
    target_time_s: float
    description: str = ""


STAGES_PROGRESSIVE = [
    ProgressiveStage("stage0_instant", 0.01, 0.01, "1% coeffs — coarse draft, <10ms"),
    ProgressiveStage("stage1_fast", 0.05, 5.0, "5% coeffs — good quality, 1-5s"),
    ProgressiveStage(
        "stage2_full", 1.00, float("inf"), "100% coeffs — full quality, background"
    ),
]


class LayerPriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2


def compute_layer_priority(
    layer_idx: int,
    n_layers: int,
    critical_ratio: float = 0.15,
    high_ratio: float = 0.35,
) -> LayerPriority:
    if n_layers <= 1:
        return LayerPriority.CRITICAL
    if layer_idx == 0 or layer_idx >= n_layers - 1:
        return LayerPriority.CRITICAL
    crit_count = max(1, int(n_layers * critical_ratio))
    high_count = max(1, int(n_layers * high_ratio))
    if layer_idx < crit_count or layer_idx >= n_layers - crit_count:
        return LayerPriority.CRITICAL
    if layer_idx < high_count or layer_idx >= n_layers - high_count:
        return LayerPriority.HIGH
    return LayerPriority.NORMAL


def _layer_from_name(tensor_name: str) -> int:
    m = re.search(r"blk\.(\d+)", tensor_name)
    if m:
        return int(m.group(1))
    if "embed" in tensor_name or "token" in tensor_name:
        return 0
    if "head" in tensor_name or "output" in tensor_name or "norm" in tensor_name:
        return 999999
    return -1


class PredictivePrefetcher:
    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self._token_to_layers: dict[int, dict[int, int]] = {}
        self._layer_freq: dict[int, int] = {}
        self._access_history: list[tuple[int, int]] = []
        self._max_history = 500

    def observe(self, token_id: int, accessed_layers: list[int]):
        if not accessed_layers:
            return
        self._access_history.append((token_id, accessed_layers[0]))
        if len(self._access_history) > self._max_history:
            self._access_history.pop(0)
        for lidx in accessed_layers:
            self._layer_freq[lidx] = self._layer_freq.get(lidx, 0) + 1
            if token_id not in self._token_to_layers:
                self._token_to_layers[token_id] = {}
            self._token_to_layers[token_id][lidx] = (
                self._token_to_layers[token_id].get(lidx, 0) + 1
            )

    def predict_next_layers(
        self, current_token: int, current_layer: int, top_k: int = 3
    ) -> list[int]:
        scores = np.zeros(self.n_layers, dtype=np.float64)
        if current_layer + 1 < self.n_layers:
            scores[current_layer + 1] += 2.0
        if current_layer + 2 < self.n_layers:
            scores[current_layer + 2] += 0.5
        token_layers = self._token_to_layers.get(current_token, {})
        for lidx, count in token_layers.items():
            scores[lidx] += count * 0.5
        for past_token, past_layer in self._access_history[-50:]:
            sim = 1.0 if past_token == current_token else 0.1
            if sim > 0.05:
                scores[past_layer] += sim * 0.3
        scores[0] += 1.0
        scores[self.n_layers - 1] += 1.0
        top = np.argsort(-scores)[:top_k]
        return [int(t) for t in top if scores[t] > 0]


class BackgroundRefiner:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._complete = threading.Event()
        self._complete.set()

    def start(
        self,
        target_stage: int,
        current_stage: int,
        stages: list[ProgressiveStage],
        apply_fn,
    ):
        if self._thread and self._thread.is_alive():
            self._complete.wait()
        self._complete.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(target_stage, current_stage, stages, apply_fn),
            daemon=True,
        )
        self._thread.start()

    def _run(
        self,
        target_stage: int,
        current_stage: int,
        stages: list[ProgressiveStage],
        apply_fn,
    ):
        try:
            for s in range(current_stage + 1, target_stage + 1):
                apply_fn(s, stages[s])
        except Exception:
            pass
        finally:
            self._complete.set()

    def wait(self):
        self._complete.wait()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class ProgressiveModelStore:
    def __init__(self, cache_dir: Optional[str] = None):
        self.weights: dict[str, DCTWeightPacker] = OrderedDict()
        self.config: dict = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._loaded = False

    def list_tensors(self, layer: Optional[int] = None) -> list[str]:
        if layer is not None:
            prefix = f"blk.{layer}."
            return [n for n in self.weights if n.startswith(prefix)]
        return list(self.weights.keys())

    def get_weight(self, name: str) -> DCTWeightPacker:
        return self.weights[name]

    def get_layer_weights(self, layer_idx: int) -> dict[str, DCTWeightPacker]:
        prefix = f"blk.{layer_idx}."
        return {n: w for n, w in self.weights.items() if n.startswith(prefix)}

    def __len__(self) -> int:
        return len(self.weights)

    def __contains__(self, name: str) -> bool:
        return name in self.weights


@dataclass
class ProgressiveQualityReport:
    quality_pct: float
    stage: int
    total_stages: int
    stage_name: str
    coeff_pct: float
    elapsed_s: float

    def __str__(self) -> str:
        return (
            f"Quality: {self.quality_pct:.1f}% | "
            f"Stage {self.stage + 1}/{self.total_stages} "
            f"({self.stage_name}) | "
            f"{self.coeff_pct:.1f}% coeffs | "
            f"{self.elapsed_s:.1f}s"
        )


class ProgressiveWeightLoader:
    def __init__(
        self,
        store: ProgressiveModelStore,
        stages: Optional[list[ProgressiveStage]] = None,
        n_layers: int = 0,
        layer_priority: bool = True,
    ):
        self.store = store
        self.stages = stages or list(STAGES_PROGRESSIVE)
        self.n_layers = n_layers
        self.use_layer_priority = layer_priority

        self._current_stage: int = -1
        self._lock = threading.Lock()
        self._coeff_budgets: dict[str, int] = {}
        self._decompressed: dict[str, np.ndarray] = {}
        self._decompressed_lock = threading.Lock()
        self._tensor_qualities: dict[str, float] = {}

        self._bg_refiner = BackgroundRefiner()
        self._prefetcher = PredictivePrefetcher(n_layers=n_layers)

        self._stage_load_times: list[float] = []
        self._start_time = time.time()

        self.stats = {
            "stage_loads": 0,
            "bg_refinements": 0,
            "predictive_hits": 0,
            "predictive_misses": 0,
        }

    @classmethod
    def from_gguf(
        cls,
        gguf_path: str,
        stages: Optional[list[ProgressiveStage]] = None,
        n_layers: int = 0,
    ) -> ProgressiveWeightLoader:
        store = ProgressiveModelStore()
        model_path = Path(gguf_path)
        if not model_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
        try:
            from gguf import GGUFReader

            reader = GGUFReader(str(gguf_path))
            tensor_list = list(reader.tensors)
            for t in tensor_list:
                data = np.asarray(t.data, dtype=np.float32)
                pw = DCTWeightPacker.from_tensor(t.name, data)
                store.weights[t.name] = pw
        except ImportError:
            raise ImportError("gguf package required for GGUF loading")
        store._loaded = True
        if n_layers == 0:
            n_layers = 32
        return cls(store=store, stages=stages, n_layers=n_layers)

    def advance_to_stage(self, stage: int) -> bool:
        if stage < 0 or stage >= len(self.stages):
            return False
        if stage <= self._current_stage:
            return True
        t0 = time.time()
        for s in range(self._current_stage + 1, stage + 1):
            self._apply_stage(s, self.stages[s])
            self._current_stage = s
        self._stage_load_times.append(time.time() - t0)
        self.stats["stage_loads"] += 1
        return True

    def advance_to_stage_async(self, stage: int):
        if stage <= self._current_stage:
            return
        self._bg_refiner.start(
            stage,
            self._current_stage,
            self.stages,
            lambda s, sd: self._apply_stage_internal(s, sd),
        )

    def _apply_stage_internal(self, stage_idx: int, stage_def: ProgressiveStage):
        with self._lock:
            self._apply_stage(stage_idx, stage_def)
        self.stats["bg_refinements"] += 1

    def _apply_stage(self, stage_idx: int, stage_def: ProgressiveStage):
        with self._lock:
            for tensor_name, pw in self.store.weights.items():
                if pw.is_1d:
                    self._coeff_budgets[tensor_name] = pw.total_coefficients
                else:
                    total = pw.total_coefficients
                    n = max(1, int(total * stage_def.coeff_fraction))
                    self._coeff_budgets[tensor_name] = n
                budget = self._coeff_budgets[tensor_name]
                self._tensor_qualities[tensor_name] = pw.quality_at(budget)
            if self.use_layer_priority and self.n_layers > 0:
                self._apply_layer_priority_boost()
            self._decompressed.clear()

    def _apply_layer_priority_boost(self):
        for tensor_name in list(self._coeff_budgets.keys()):
            lidx = _layer_from_name(tensor_name)
            if lidx < 0:
                continue
            priority = compute_layer_priority(lidx, self.n_layers)
            boost = {
                LayerPriority.CRITICAL: 2.0,
                LayerPriority.HIGH: 1.5,
                LayerPriority.NORMAL: 1.0,
            }.get(priority, 1.0)
            old = self._coeff_budgets[tensor_name]
            total = self.store.weights[tensor_name].total_coefficients
            new_budget = min(total, max(1, int(old * boost)))
            self._coeff_budgets[tensor_name] = new_budget
            self._tensor_qualities[tensor_name] = self.store.weights[
                tensor_name
            ].quality_at(new_budget)

    def get_weight(self, tensor_name: str) -> np.ndarray:
        with self._decompressed_lock:
            if tensor_name in self._decompressed:
                return self._decompressed[tensor_name]
        pw = self.store.weights.get(tensor_name)
        if pw is None:
            raise KeyError(f"Tensor '{tensor_name}' not found")
        n_coeffs = self._coeff_budgets.get(tensor_name, pw.total_coefficients)
        weight = np.ascontiguousarray(pw.reconstruct(n_coefficients=n_coeffs))
        with self._decompressed_lock:
            self._decompressed[tensor_name] = weight
        return weight

    def get_layer_weights(self, layer_idx: int) -> dict[str, np.ndarray]:
        result = {}
        prefix = f"blk.{layer_idx}."
        for name in self.store.weights:
            if name.startswith(prefix):
                result[name] = self.get_weight(name)
        return result

    def observe_access(self, token_id: int, accessed_layers: list[int]):
        self._prefetcher.observe(token_id, accessed_layers)

    def prefetch_predicted(self, current_token: int, current_layer: int):
        predicted = self._prefetcher.predict_next_layers(current_token, current_layer)
        hit = False
        for lidx in predicted:
            prefix = f"blk.{lidx}."
            for name in self.store.weights:
                if name.startswith(prefix):
                    with self._decompressed_lock:
                        if name not in self._decompressed:
                            hit = True
                            break
            if hit:
                break
        if hit:
            self.stats["predictive_hits"] += 1
        else:
            self.stats["predictive_misses"] += 1

    def get_quality(self) -> float:
        if not self._tensor_qualities:
            return 0.0
        qualities, weights = [], []
        for tensor_name, quality in self._tensor_qualities.items():
            lidx = _layer_from_name(tensor_name)
            if self.use_layer_priority and self.n_layers > 0:
                pri = compute_layer_priority(lidx if lidx >= 0 else 0, self.n_layers)
                w = {
                    LayerPriority.CRITICAL: 5.0,
                    LayerPriority.HIGH: 2.0,
                    LayerPriority.NORMAL: 1.0,
                }.get(pri, 1.0)
            else:
                w = 1.0
            qualities.append(quality)
            weights.append(w)
        return float(
            np.average(
                np.array(qualities, dtype=np.float64),
                weights=np.array(weights, dtype=np.float64),
            )
        )

    def get_stage_quality(self, stage: int) -> float:
        if stage < 0 or stage >= len(self.stages):
            return 0.0
        sd = self.stages[stage]
        qualities, weights = [], []
        for tensor_name, pw in self.store.weights.items():
            total = pw.total_coefficients
            n = max(1, int(total * sd.coeff_fraction))
            if self.use_layer_priority and self.n_layers > 0:
                lidx = _layer_from_name(tensor_name)
                pri = compute_layer_priority(lidx if lidx >= 0 else 0, self.n_layers)
                if pri == LayerPriority.CRITICAL:
                    n = min(total, n * 2)
                elif pri == LayerPriority.HIGH:
                    n = min(total, int(n * 1.5))
                w = {
                    LayerPriority.CRITICAL: 5.0,
                    LayerPriority.HIGH: 2.0,
                    LayerPriority.NORMAL: 1.0,
                }.get(pri, 1.0)
            else:
                w = 1.0
            qualities.append(pw.quality_at(n))
            weights.append(w)
        if not qualities:
            return 0.0
        return float(
            np.average(
                np.array(qualities, dtype=np.float64),
                weights=np.array(weights, dtype=np.float64),
            )
        )

    def get_report(self) -> ProgressiveQualityReport:
        quality = self.get_quality() * 100
        stage = max(0, self._current_stage)
        stage_name = self.stages[stage].name if self._current_stage >= 0 else "none"
        total_stages = len(self.stages)
        coeff_pct = 0.0
        if self._coeff_budgets and self.store.weights:
            budgets = np.array(list(self._coeff_budgets.values()), dtype=np.float64)
            totals = np.array(
                [self.store.weights[n].total_coefficients for n in self._coeff_budgets],
                dtype=np.float64,
            )
            coeff_pct = 100.0 * float(np.mean(budgets / np.maximum(totals, 1)))
        elapsed = time.time() - self._start_time
        return ProgressiveQualityReport(
            quality_pct=quality,
            stage=stage,
            total_stages=total_stages,
            stage_name=stage_name,
            coeff_pct=coeff_pct,
            elapsed_s=elapsed,
        )

    def get_status(self) -> str:
        return str(self.get_report())

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Progressive Weight Loader",
            "=" * 60,
            self.get_status(),
            f"Tensors: {len(self.store.weights)}",
            f"Layers: {self.n_layers}",
            f"Layer priority: {'ON' if self.use_layer_priority else 'OFF'}",
            f"Stage loads: {self.stats['stage_loads']}",
            f"BG refinements: {self.stats['bg_refinements']}",
            f"Predictive hits: {self.stats['predictive_hits']}",
            "",
        ]
        lines.append("Stage details:")
        for i, s in enumerate(self.stages):
            active = "<< ACTIVE" if i == self._current_stage else ""
            q = self.get_stage_quality(i) * 100 if self._current_stage >= 0 else 0
            lines.append(
                f"  {i}: {s.name:20s}  {s.coeff_fraction * 100:>5.1f}% coeffs  "
                f"quality ~ {q:.1f}%  target {s.target_time_s:.2f}s  {active}"
            )
        return "\n".join(lines)
