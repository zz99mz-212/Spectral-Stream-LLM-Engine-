"""Tensor World Model — scans ALL tensor metadata before compression.

Builds a unified tensor graph of the entire model with sensitivity tiers,
layer types, and redundancy patterns. Runs parallel profiling across all
tensors using ThreadPoolExecutor.
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .._dataclasses import TensorProfile
from .._profiler import CompressionProfiler
from .._helpers import _classify_by_name

logger = logging.getLogger(__name__)


@dataclass
class TensorGraphNode:
    """Metadata for a single tensor in the model graph."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    n_elements: int
    nbytes: int
    tensor_type: str
    layer_idx: int = -1
    param_type: str = ""
    sensitivity: float = 0.5
    effective_rank: float = 0.0
    spectral_decay_rate: float = 0.0
    entropy: float = 0.0
    compressibility_score: float = 0.0
    profile: Optional[TensorProfile] = None


@dataclass
class SensitivityMap:
    """Per-tensor sensitivity tiers for the entire model."""

    tier_map: Dict[str, int] = field(default_factory=dict)
    high_sensitivity: List[str] = field(default_factory=list)
    medium_sensitivity: List[str] = field(default_factory=list)
    low_sensitivity: List[str] = field(default_factory=list)

    def tier(self, name: str) -> int:
        return self.tier_map.get(name, 2)


@dataclass
class TensorGraph:
    """Unified graph of all tensors in the model."""

    nodes: Dict[str, TensorGraphNode] = field(default_factory=dict)
    by_type: Dict[str, List[str]] = field(default_factory=dict)
    by_layer: Dict[int, List[str]] = field(default_factory=dict)
    total_params: int = 0
    total_bytes: int = 0
    n_tensors: int = 0

    def get(self, name: str) -> Optional[TensorGraphNode]:
        return self.nodes.get(name)

    def tensors_of_type(self, tensor_type: str) -> List[str]:
        return self.by_type.get(tensor_type, [])

    def tensors_in_layer(self, layer: int) -> List[str]:
        return self.by_layer.get(layer, [])


@dataclass
class UnifiedModelProfile:
    """Complete world-model profile of an entire model."""

    graph: TensorGraph = field(default_factory=TensorGraph)
    sensitivity: SensitivityMap = field(default_factory=SensitivityMap)
    redundancy_map: Dict[str, List[str]] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    layer_count: int = 0
    embedding_size: int = 0
    hidden_size: int = 0
    num_heads: int = 0
    estimated_model_size_gb: float = 0.0

    def get_tensor_names(self) -> List[str]:
        return list(self.graph.nodes.keys())


class TensorWorldModel:
    """World-Model Tensor Scanner — scans ALL tensor metadata before compression.

    Builds a complete model-level tensor graph:
    1. Scans all tensor names/shapes/dtypes from a dict or MemoryMappedTensorEngine
    2. Determines sensitivity tiers, layer types, redundancy patterns
    3. Runs parallel profiling across all tensors using ThreadPoolExecutor
    4. Returns UnifiedModelProfile with tensor graph, sensitivity map, redundancy map
    """

    def __init__(
        self,
        profiler: Optional[CompressionProfiler] = None,
        max_workers: int = 4,
    ):
        self._profiler = profiler or CompressionProfiler()
        self._max_workers = max_workers
        self._profile_cache: Dict[str, TensorProfile] = {}

    def scan_from_dict(
        self,
        tensors: Dict[str, np.ndarray],
        max_workers: Optional[int] = None,
    ) -> UnifiedModelProfile:
        """Scan all tensors from a dict of name -> ndarray.

        Builds the initial graph from metadata alone, then profiles
        in parallel using ThreadPoolExecutor.
        """
        nw = max_workers or self._max_workers
        graph = self._build_graph(tensors)
        logger.debug(
            "TensorWorldModel: scanning %d tensors (%s total)",
            graph.n_tensors,
            _fmt_bytes(graph.total_bytes),
        )

        profile = UnifiedModelProfile(graph=graph)
        profile.sensitivity = self._compute_sensitivity(graph)
        profile.redundancy_map = self._find_redundancy(graph)

        self._parallel_profile(tensors, graph, max_workers=nw)

        profile.type_distribution = {t: len(ns) for t, ns in graph.by_type.items()}

        metrics = self._extract_model_metrics(graph)
        profile.layer_count = metrics.get("layer_count", 0)
        profile.embedding_size = metrics.get("embedding_size", 0)
        profile.hidden_size = metrics.get("hidden_size", 0)
        profile.num_heads = metrics.get("num_heads", 0)
        profile.estimated_model_size_gb = graph.total_bytes / (1024**3)

        return profile

    def scan_from_names(
        self,
        tensor_infos: Dict[str, Tuple[tuple, str, int, int]],
    ) -> UnifiedModelProfile:
        """Scan from metadata only (names, shapes, dtypes, sizes).

        Useful for MemoryMappedTensorEngine where we have metadata
        but not actual tensor data yet.
        """
        graph = TensorGraph()
        for name, (shape, dtype_str, offset, nbytes) in tensor_infos.items():
            n_elements = int(np.prod(shape)) if shape else 0
            tensor_type = _classify_by_name(name)
            layer_idx = self._extract_layer_idx(name)
            param_type = self._extract_param_type(name)
            node = TensorGraphNode(
                name=name,
                shape=shape,
                dtype=dtype_str,
                n_elements=n_elements,
                nbytes=nbytes,
                tensor_type=tensor_type,
                layer_idx=layer_idx,
                param_type=param_type,
            )
            graph.nodes[name] = node
            graph.by_type.setdefault(tensor_type, []).append(name)
            graph.by_layer.setdefault(layer_idx, []).append(name)
            graph.total_params += n_elements
            graph.total_bytes += nbytes

        graph.n_tensors = len(graph.nodes)

        profile = UnifiedModelProfile(graph=graph)
        profile.sensitivity = self._compute_sensitivity(graph)
        profile.type_distribution = {t: len(ns) for t, ns in graph.by_type.items()}
        profile.estimated_model_size_gb = graph.total_bytes / (1024**3)
        return profile

    def _build_graph(self, tensors: Dict[str, np.ndarray]) -> TensorGraph:
        """Build TensorGraph from tensor metadata (no profiling yet)."""
        graph = TensorGraph()
        for name, tensor in tensors.items():
            tensor_type = _classify_by_name(name)
            layer_idx = self._extract_layer_idx(name)
            param_type = self._extract_param_type(name)
            n_elements = tensor.size
            nbytes = tensor.nbytes
            node = TensorGraphNode(
                name=name,
                shape=tensor.shape,
                dtype=str(tensor.dtype),
                n_elements=n_elements,
                nbytes=nbytes,
                tensor_type=tensor_type,
                layer_idx=layer_idx,
                param_type=param_type,
                sensitivity=0.5,
            )
            graph.nodes[name] = node
            graph.by_type.setdefault(tensor_type, []).append(name)
            graph.by_layer.setdefault(layer_idx, []).append(name)
            graph.total_params += n_elements
            graph.total_bytes += nbytes
        graph.n_tensors = len(graph.nodes)
        return graph

    def _parallel_profile(
        self,
        tensors: Dict[str, np.ndarray],
        graph: TensorGraph,
        max_workers: int = 4,
    ) -> None:
        """Profile all tensors in parallel, updating their graph nodes."""
        names = list(tensors.keys())
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for name in names:
                future = pool.submit(self._profile_single, tensors[name], name)
                futures[future] = name
            for future in as_completed(futures):
                name = futures[future]
                try:
                    profile = future.result()
                    self._profile_cache[name] = profile
                    node = graph.get(name)
                    if node is not None and profile is not None:
                        node.effective_rank = profile.effective_rank
                        node.spectral_decay_rate = profile.spectral_decay_rate
                        node.entropy = profile.entropy_rate
                        node.compressibility_score = self._compute_compressibility(
                            profile
                        )
                        node.profile = profile
                        node.sensitivity = profile.sensitivity
                except Exception as exc:
                    logger.debug("Parallel profile failed for '%s': %s", name, exc)

    def _profile_single(
        self,
        tensor: np.ndarray,
        name: str,
    ) -> Optional[TensorProfile]:
        """Profile a single tensor. May sample for large tensors."""
        try:
            return self._profiler.profile_tensor(tensor, name)
        except Exception as exc:
            logger.debug("Profile failed for '%s': %s", name, exc)
            return None

    @staticmethod
    def _compute_compressibility(profile: TensorProfile) -> float:
        """Score 0-1 indicating how compressible the tensor is."""
        score = 0.0
        if profile.effective_rank > 0:
            er_ratio = profile.effective_rank / max(
                min(profile.shape) if profile.shape else 1, 1
            )
            score += 0.3 * (1.0 - min(er_ratio, 1.0))
        score += 0.2 * min(profile.energy_concentration, 1.0)
        score += 0.2 * min(profile.nm_sparsity_score, 1.0)
        score += 0.15 * (1.0 - min(profile.entropy_rate / 8.0, 1.0))
        score += 0.15 * (1.0 - min(profile.sensitivity, 1.0))
        return min(score, 1.0)

    @staticmethod
    def _compute_sensitivity(graph: TensorGraph) -> SensitivityMap:
        """Assign sensitivity tiers based on tensor type and name patterns."""
        sm = SensitivityMap()
        for name, node in graph.nodes.items():
            tt = node.tensor_type
            ptype = node.param_type

            if ptype in ("q_weight", "k_weight", "v_weight", "o_weight"):
                tier = 1
            elif tt == "attention_q":
                tier = 1
            elif tt in ("attention_k", "attention_v", "attention_o"):
                tier = 1
            elif tt in ("ffn_gate", "ffn_up", "ffn_down"):
                tier = 2
            elif tt == "embedding":
                tier = 2
            elif tt == "output":
                tier = 2
            elif tt == "norm":
                tier = 3
            elif ptype == "bias":
                tier = 3
            else:
                tier = 2

            sm.tier_map[name] = tier
            if tier == 1:
                sm.high_sensitivity.append(name)
            elif tier == 2:
                sm.medium_sensitivity.append(name)
            else:
                sm.low_sensitivity.append(name)
        return sm

    @staticmethod
    def _find_redundancy(graph: TensorGraph) -> Dict[str, List[str]]:
        """Find tensors with identical shapes and types (potential redundancy)."""
        signature_map: Dict[Tuple, List[str]] = {}
        for name, node in graph.nodes.items():
            sig = (node.shape, node.tensor_type)
            signature_map.setdefault(sig, []).append(name)
        return {sig: names for sig, names in signature_map.items() if len(names) > 1}

    @staticmethod
    def _extract_layer_idx(name: str) -> int:
        """Extract layer index from common naming patterns."""
        import re

        patterns = [
            r"layers\.(\d+)\.",
            r"layer(\d+)\.",
            r"block(\d+)\.",
            r"transformer\.(\d+)\.",
            r"encoder\.(\d+)\.",
            r"decoder\.(\d+)\.",
        ]
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                return int(m.group(1))
        return -1

    @staticmethod
    def _extract_param_type(name: str) -> str:
        """Determine parameter type from name."""
        name_lower = name.lower()
        if "q_proj" in name_lower or "q.weight" in name_lower:
            return "q_weight"
        if "k_proj" in name_lower or "k.weight" in name_lower:
            return "k_weight"
        if "v_proj" in name_lower or "v.weight" in name_lower:
            return "v_weight"
        if "o_proj" in name_lower or "out_proj" in name_lower:
            return "o_weight"
        if "gate_proj" in name_lower or "gate" in name_lower and "weight" in name_lower:
            return "gate_weight"
        if "up_proj" in name_lower or "up" in name_lower and "weight" in name_lower:
            return "up_weight"
        if "down_proj" in name_lower or "down" in name_lower and "weight" in name_lower:
            return "down_weight"
        if "embed" in name_lower or "tok_embed" in name_lower:
            return "embedding"
        if "norm" in name_lower or "ln_" in name_lower:
            return "norm"
        if "bias" in name_lower:
            return "bias"
        if "lm_head" in name_lower or "output" in name_lower:
            return "output"
        return "other"

    @staticmethod
    def _extract_model_metrics(graph: TensorGraph) -> Dict[str, Any]:
        """Extract high-level model architecture metrics from tensor graph."""
        metrics: Dict[str, Any] = {
            "layer_count": 0,
            "embedding_size": 0,
            "hidden_size": 0,
            "num_heads": 0,
        }
        seen_layers: Set[int] = set()
        for node in graph.nodes.values():
            if node.layer_idx >= 0:
                seen_layers.add(node.layer_idx)
            if node.param_type in ("q_weight", "k_weight", "v_weight", "o_weight"):
                if len(node.shape) == 2:
                    metrics["hidden_size"] = max(metrics["hidden_size"], node.shape[1])
        metrics["layer_count"] = len(seen_layers)
        return metrics

    def get_cached_profile(self, name: str) -> Optional[TensorProfile]:
        return self._profile_cache.get(name)

    def clear_cache(self) -> None:
        self._profile_cache.clear()
        gc.collect()


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024**2:
        return f"{n / 1024:.1f} KB"
    elif n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    else:
        return f"{n / 1024**3:.2f} GB"
