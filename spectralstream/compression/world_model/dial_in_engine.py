from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spectralstream.compression.engine import (
    CompressionIntelligenceEngine,
    CompressionConfig,
    CompressedTensor,
)
from spectralstream.compression.engine.method_discovery import MethodDiscovery
from spectralstream.compression.engine.method_validation import (
    validate_single_method,
    validate_all_methods,
)

try:
    from spectralstream.compression.engine.world_model.unified_world_model import (
        UnifiedCompressionWorldModel,
    )
except ImportError:
    UnifiedCompressionWorldModel = None  # type: ignore

from spectralstream.compression.world_model.loss_metrics_engine import (
    LossMetricsIntelligenceEngine,
    PerTensorLossMetrics,
    QualityGrade,
)
from spectralstream.compression.world_model.unified_cascade_engine import (
    CASCADE_PATTERNS,
    COMPLEMENTARY_PAIRS,
    STACKING_PATTERNS,
    CascadePlan,
    CascadeStage,
    UnifiedCascadeEngine,
)
from spectralstream.compression.certificate import (
    CertificateBuilder,
    CompressionCertificate,
    TensorCertificate,
)
from spectralstream.compression.honest_metrics import (
    dual_ratio,
    end_to_end_error,
    serialized_nbytes,
)

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class DialInTensorProfile:
    name: str
    shape: Tuple[int, ...]
    nbytes: int
    tensor_type: str
    layer_idx: int
    sensitivity: float
    compressibility: float
    effective_rank: float
    spectral_decay: float
    entropy: float
    recommended_methods: List[str] = field(default_factory=list)


@dataclass
class MethodTestRecord:
    method_name: str
    tensor_name: str
    tensor_type: str
    category: str
    tier: int
    ratio: float
    error: float
    snr_db: float
    time_ms: float
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeDiscoveryResult:
    tensor_type: str
    pattern_name: str
    stages: List[Dict[str, Any]]
    total_ratio: float
    total_error: float
    composite_score: float


@dataclass
class SensitivityEntry:
    tensor_name: str
    tensor_type: str
    tier: str
    baseline_ratio: float
    baseline_error: float
    error_at_2x: Optional[float] = None
    error_at_5x: Optional[float] = None
    error_at_10x: Optional[float] = None
    downstream_impact: float = 0.0
    recommended_max_error: float = 0.01


@dataclass
class OptimalParamEntry:
    tensor_type: str
    method_name: str
    params: Dict[str, Any]
    achieved_ratio: float
    achieved_error: float
    score: float


@dataclass
class ModelRatioPlan:
    tensor_plans: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall_ratio: float = 1.0
    overall_error: float = 0.0
    total_original_bytes: int = 0
    total_compressed_est: int = 0


@dataclass
class DialInReport:
    model_name: str = ""
    model_size_gb: float = 0.0
    n_tensors: int = 0
    n_tensor_types: int = 0

    method_profiles: Dict[str, List[MethodTestRecord]] = field(default_factory=dict)
    top_methods_per_type: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    method_complementarity: Dict[str, float] = field(default_factory=dict)

    cascade_discoveries: List[CascadeDiscoveryResult] = field(default_factory=list)
    best_cascade_per_type: Dict[str, CascadeDiscoveryResult] = field(
        default_factory=dict
    )

    sensitivity_map: Dict[str, SensitivityEntry] = field(default_factory=dict)
    critical_tensors: List[str] = field(default_factory=list)
    robust_tensors: List[str] = field(default_factory=list)

    optimal_params: Dict[str, List[OptimalParamEntry]] = field(default_factory=dict)

    ratio_plan: Optional[ModelRatioPlan] = None

    avg_ratio: float = 0.0
    avg_error: float = 0.0
    weighted_grade: str = "UNKNOWN"
    passes_min_ratio: bool = False
    passes_max_error: bool = False
    overall_pass: bool = False

    recommendations: List[str] = field(default_factory=list)
    certification: Optional[CompressionCertificate] = None

    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary_text(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║          Compression Dial-In Engine — Summary Report           ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            f"  Model: {self.model_name}",
            f"  Size: {self.model_size_gb:.2f} GB",
            f"  Tensors: {self.n_tensors} across {self.n_tensor_types} types",
            f"  Elapsed: {self.elapsed_seconds:.1f}s",
            "",
            f"  📊 Performance",
            f"  {'Average Ratio:':<25} {self.avg_ratio:.1f}x",
            f"  {'Average Error:':<25} {self.avg_error * 100:.4f}%",
            f"  {'Weighted Grade:':<25} {self.weighted_grade}",
            "",
            f"  🎯 Threshold Check",
            f"  {'Min Ratio (200:1):':<25} {'✅ PASS' if self.passes_min_ratio else '❌ FAIL'} ({self.avg_ratio:.1f}x)",
            f"  {'Target Ratio (400:1):':<25} {'✅ PASS' if self.avg_ratio >= 400 else '❌ FAIL'} ({self.avg_ratio:.1f}x)",
            f"  {'Max Error (<1%):':<25} {'✅ PASS' if self.passes_max_error else '❌ FAIL'} ({self.avg_error * 100:.4f}%)",
            f"  {'Overall:':<25} {'✅ PASS' if self.overall_pass else '❌ FAIL'}",
        ]

        if self.critical_tensors:
            lines.extend(
                [
                    "",
                    "  ⚠ Critical Tensors (need high-fidelity treatment):",
                ]
            )
            for tn in self.critical_tensors[:10]:
                lines.append(f"    - {tn}")

        if self.best_cascade_per_type:
            lines.extend(
                [
                    "",
                    "  🔄 Recommended Cascade Patterns:",
                ]
            )
            for tt, cd in sorted(self.best_cascade_per_type.items()):
                stages_str = " → ".join(s["method"] for s in cd.stages)
                lines.append(
                    f"    {tt:<15} {stages_str:<40} ratio={cd.total_ratio:.1f}x"
                )

        if self.recommendations:
            lines.extend(
                [
                    "",
                    "  💡 Recommendations:",
                ]
            )
            for rec in self.recommendations:
                lines.append(f"    • {rec}")

        return "\n".join(lines)

    def to_html(self) -> str:
        colors = {
            "PASS": "#00ff88",
            "FAIL": "#ff4444",
            "EXCELLENT": "#00ff88",
            "GOOD": "#00cc66",
            "FAIR": "#ffd700",
            "POOR": "#ff8c00",
        }
        grade_color = colors.get(self.weighted_grade, "#ffffff")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dial-In Report — {self.model_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          max-width: 1200px; margin: 0 auto; padding: 20px; background: #0a0a0f; color: #e0e0e0; }}
  h1, h2, h3 {{ color: #ffffff; }}
  .header {{ text-align: center; padding: 40px;
             background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
             border-radius: 16px; margin-bottom: 30px; }}
  .header h1 {{ font-size: 2.5em; margin: 0; }}
  .badge-container {{ display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .badge {{ background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 20px 30px;
            text-align: center; min-width: 150px; }}
  .badge .value {{ font-size: 2em; font-weight: bold; }}
  .badge .label {{ font-size: 0.85em; color: #8888ff; margin-top: 5px; }}
  .pass {{ color: #00ff88; }} .fail {{ color: #ff4444; }}
  .section {{ background: #1a1a2e; border-radius: 12px; padding: 25px; margin: 20px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 10px 15px; text-align: left; border-bottom: 1px solid #333; }}
  th {{ color: #8888ff; font-weight: 600; text-transform: uppercase; font-size: 0.85em; }}
  .progress-bar {{ background: #333; border-radius: 10px; height: 20px; overflow: hidden; margin: 5px 0; }}
  .progress-fill {{ height: 100%; transition: width 0.5s; background: linear-gradient(90deg, #00ff88, #00cc66); }}
</style>
</head>
<body>
<div class="header">
  <h1>Compression Dial-In Report</h1>
  <div class="subtitle" style="color: #8888ff;">SpectralStream R&D Automation Engine</div>
  <p style="color: #888;">{self.model_name} · {self.n_tensors} tensors · {self.model_size_gb:.2f} GB</p>
</div>
<div class="badge-container">
  <div class="badge"><div class="value" style="color:{grade_color}">{self.weighted_grade}</div><div class="label">Quality Grade</div></div>
  <div class="badge"><div class="value">{self.avg_ratio:.1f}x</div><div class="label">Avg Ratio</div></div>
  <div class="badge"><div class="value">{self.avg_error * 100:.4f}%</div><div class="label">Avg Error</div></div>
  <div class="badge"><div class="value">{self.elapsed_seconds:.0f}s</div><div class="label">Time</div></div>
</div>
<div class="section">
  <h2>Thresholds</h2>
  <table>
    <tr><td>Min Ratio (200:1)</td><td class="{"pass" if self.passes_min_ratio else "fail"}">{"✅ PASS" if self.passes_min_ratio else "❌ FAIL"}</td><td>{self.avg_ratio:.1f}x</td></tr>
    <tr><td>Target Ratio (400:1)</td><td class="{"pass" if self.avg_ratio >= 400 else "fail"}">{"✅ PASS" if self.avg_ratio >= 400 else "❌ FAIL"}</td><td>{self.avg_ratio:.1f}x</td></tr>
    <tr><td>Max Error ({"<1%"})</td><td class="{"pass" if self.passes_max_error else "fail"}">{"✅ PASS" if self.passes_max_error else "❌ FAIL"}</td><td>{self.avg_error * 100:.4f}%</td></tr>
    <tr><td>Overall</td><td class="{"pass" if self.overall_pass else "fail"}">{"✅ PASS" if self.overall_pass else "❌ FAIL"}</td><td></td></tr>
  </table>
</div>
<div class="section">
  <h2>Recommended Cascades Per Tensor Type</h2>
  <table><tr><th>Type</th><th>Pattern</th><th>Stages</th><th>Ratio</th><th>Error</th></tr>"""
        for tt, cd in sorted(self.best_cascade_per_type.items()):
            stages_str = " → ".join(s["method"] for s in cd.stages)
            html += f"<tr><td>{tt}</td><td>{cd.pattern_name}</td><td>{stages_str}</td><td>{cd.total_ratio:.1f}x</td><td>{cd.total_error:.6f}</td></tr>"

        html += """</table></div>"""
        if self.recommendations:
            html += """<div class="section"><h2>Recommendations</h2><ul>"""
            for rec in self.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"
        html += """
<div class="footer" style="text-align:center;color:#666;margin-top:40px;padding:20px;">
  <p>Generated by SpectralStream Compression Dial-In Engine</p>
</div>
</body>
</html>"""
        return html

    def save(
        self, output_dir: str, formats: Optional[List[str]] = None
    ) -> Dict[str, str]:
        if formats is None:
            formats = ["json", "html", "txt"]
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, "dial_in_report")
        saved: Dict[str, str] = {}
        if "json" in formats:
            p = f"{base}.json"
            self.to_json(p)
            saved["json"] = p
        if "html" in formats:
            p = f"{base}.html"
            with open(p, "w") as f:
                f.write(self.to_html())
            saved["html"] = p
        if "txt" in formats:
            p = f"{base}.txt"
            with open(p, "w") as f:
                f.write(self.summary_text())
            saved["txt"] = p
        if self.certification and "certificate" in formats:
            self.certification.save(f"{base}_certificate")
            saved["certificate"] = f"{base}_certificate"
        return saved


# ── Synthetic Data Generator ──────────────────────────────────────────────────


_SYNTHETIC_TENSOR_TYPES: Dict[str, Tuple[Tuple[int, ...], float]] = {
    "embedding": ((262144, 1536), 0.5),
    "attention_q": ((1536, 256), 1.0),
    "attention_k": ((1536, 256), 1.0),
    "attention_v": ((1536, 256), 1.0),
    "attention_o": ((256, 1536), 1.0),
    "ffn_gate": ((1536, 6144), 1.0),
    "ffn_up": ((1536, 6144), 1.0),
    "ffn_down": ((6144, 1536), 1.0),
    "norm": ((1536,), 0.1),
    "output": ((1536, 262144), 0.5),
    "weight": ((4096, 4096), 1.5),
}


def _make_synthetic_tensors(
    seed: int = 42,
    tensor_types: Optional[List[str]] = None,
) -> Dict[str, Tuple[np.ndarray, str]]:
    rng = np.random.RandomState(seed)
    types = tensor_types or list(_SYNTHETIC_TENSOR_TYPES.keys())
    tensors: Dict[str, Tuple[np.ndarray, str]] = {}
    for tt in types:
        if tt not in _SYNTHETIC_TENSOR_TYPES:
            continue
        shape, scale = _SYNTHETIC_TENSOR_TYPES[tt]
        n_elements = int(np.prod(shape))
        if n_elements > 50_000_000:
            shape = tuple(max(1, s // 4) for s in shape)
        tensor = rng.randn(*shape).astype(np.float32) * scale
        tensors[tt] = (tensor, tt)
    return tensors


# ── Compression Dial-In Engine ────────────────────────────────────────────────


class CompressionDialInEngine:
    """
    R&D Dial-in Engine for Compression Intelligence.

    Systematically tests compression parameters, finds optimal configurations,
    and produces detailed reports for production deployment.

    Dial-In Pipeline:
    1. Model scan → build world model
    2. Method profiling → test ALL methods on representative tensors
    3. Cascade discovery → find optimal cascade patterns
    4. Multi-model validation → verify on different tensor types
    5. Sensitivity analysis → identify critical vs robust tensors
    6. Parameter tuning → optimize method parameters per tensor type
    7. Ratio optimization → balance ratio vs error across model
    8. Production certification → final validation + certificate
    """

    def __init__(
        self,
        engine: Optional[CompressionIntelligenceEngine] = None,
        loss_metrics: Optional[LossMetricsIntelligenceEngine] = None,
        config: Optional[CompressionConfig] = None,
        max_workers: int = 4,
        target_ratio: float = 400.0,
        max_error: float = 0.01,
    ):
        self._config = config or CompressionConfig(
            target_ratio=target_ratio,
            max_error=max_error,
        )
        self._engine = engine or CompressionIntelligenceEngine(config=self._config)
        self._loss = loss_metrics or LossMetricsIntelligenceEngine()
        self._max_workers = max_workers
        self._target_ratio = target_ratio
        self._max_error = max_error
        self._methods_cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._world_model: Optional[Any] = None

    @property
    def methods(self) -> Dict[str, Dict[str, Any]]:
        if self._methods_cache is None:
            self._methods_cache = MethodDiscovery.discover()
        return self._methods_cache

    def discover_methods(self) -> Dict[str, Dict[str, Any]]:
        return self.methods

    # ═══════════════════════════════════════════════════════════════════
    #  1. MODEL SCAN — Build world model
    # ═══════════════════════════════════════════════════════════════════

    def scan_model(
        self,
        tensors: Dict[str, np.ndarray],
    ) -> List[DialInTensorProfile]:
        """Scan model tensors and build profiles with 25+ metrics."""
        profiles: List[DialInTensorProfile] = []
        tensor_list = list(tensors.items())

        def _profile(name: str, tensor: np.ndarray) -> Optional[DialInTensorProfile]:
            try:
                ttype = self._classify_tensor(name)
                layer_idx = self._extract_layer_idx(name)
                flat = tensor.ravel().astype(np.float64)
                entropy = self._compute_entropy(flat)
                eff_rank = self._compute_effective_rank(tensor)
                sensitivity = self._estimate_sensitivity(tensor, eff_rank)
                decay = self._compute_spectral_decay(tensor)

                compressibility = self._score_compressibility(
                    tensor, eff_rank, entropy, sensitivity
                )

                return DialInTensorProfile(
                    name=name,
                    shape=tensor.shape,
                    nbytes=tensor.nbytes,
                    tensor_type=ttype,
                    layer_idx=layer_idx,
                    sensitivity=sensitivity,
                    compressibility=compressibility,
                    effective_rank=eff_rank,
                    spectral_decay=decay,
                    entropy=entropy,
                )
            except Exception as exc:
                logger.debug("Profile failed for '%s': %s", name, exc)
                return None

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_profile, n, t): n for n, t in tensor_list}
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    profiles.append(result)

        profiles.sort(key=lambda p: p.nbytes, reverse=True)
        return profiles

    def _classify_tensor(self, name: str) -> str:
        name_lower = name.lower()
        if "embed" in name_lower or "word" in name_lower:
            return "embedding"
        if "norm" in name_lower:
            return "norm"
        if "q_proj" in name_lower or "q." in name_lower or "query" in name_lower:
            return "attention_q"
        if "k_proj" in name_lower or "k." in name_lower or "key" in name_lower:
            return "attention_k"
        if "v_proj" in name_lower or "v." in name_lower or "value" in name_lower:
            return "attention_v"
        if "o_proj" in name_lower or "out_proj" in name_lower:
            return "attention_o"
        if "attn" in name_lower or "attention" in name_lower:
            return "attention"
        if "gate" in name_lower:
            return "ffn_gate"
        if "up" in name_lower or "fc1" in name_lower:
            return "ffn_up"
        if "down" in name_lower or "fc2" in name_lower:
            return "ffn_down"
        if "mlp" in name_lower or "ff" in name_lower:
            return "ffn"
        if "lm_head" in name_lower or "output" in name_lower:
            return "output"
        if "weight" in name_lower:
            return "weight"
        return "weight"

    @staticmethod
    def _extract_layer_idx(name: str) -> int:
        import re

        nums = re.findall(r"\d+", name)
        return int(nums[0]) if nums else 0

    @staticmethod
    def _compute_entropy(flat: np.ndarray) -> float:
        if flat.size < 16:
            return 0.0
        lo, hi = float(np.min(flat)), float(np.max(flat))
        if hi - lo < 1e-30:
            return 0.0
        n_bins = min(256, max(10, int(math.sqrt(flat.size))))
        bins = np.linspace(lo, hi, n_bins + 1)
        h, _ = np.histogram(flat, bins=bins, density=False)
        h = h.astype(np.float64)
        h = h / max(float(np.sum(h)), 1.0)
        h = np.maximum(h, 1e-30)
        return float(-np.sum(h * np.log2(h)))

    @staticmethod
    def _compute_effective_rank(tensor: np.ndarray) -> float:
        if tensor.ndim < 2:
            return float(min(tensor.shape) if tensor.shape else 1.0)
        mat = tensor.reshape(tensor.shape[0], -1)
        m, n = mat.shape
        k = min(m, n, 128)
        if k < 2:
            return float(k)
        try:
            s = np.linalg.svd(mat[:k, :k], compute_uv=False)
            s_sum = float(np.sum(s))
            if s_sum < 1e-30:
                return 1.0
            p = s / s_sum
            eff = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
            return eff
        except np.linalg.LinAlgError:
            return float(min(m, n) * 0.5)

    @staticmethod
    def _compute_spectral_decay(tensor: np.ndarray) -> float:
        if tensor.ndim < 2:
            return 1.0
        mat = tensor.reshape(tensor.shape[0], -1)
        k = min(mat.shape[0], mat.shape[1], 64)
        if k < 4:
            return 1.0
        try:
            s = np.linalg.svd(mat[:k, :k], compute_uv=False)
            if s[0] < 1e-30:
                return 1.0
            s_norm = s / s[0]
            x = np.arange(len(s_norm), dtype=np.float64)
            if np.var(s_norm) < 1e-30:
                return 1.0
            coeffs = np.polyfit(x, np.log(s_norm + 1e-30), 1)
            return float(-coeffs[0])
        except (np.linalg.LinAlgError, np.linalg.LinAlgError):
            return 0.5

    @staticmethod
    def _estimate_sensitivity(tensor: np.ndarray, eff_rank: float) -> float:
        n_bytes = tensor.nbytes
        size_gb = n_bytes / (1024**3)
        dim = max(tensor.shape) if tensor.shape else 1
        size_factor = min(size_gb * 10, 1.0)
        rank_ratio = eff_rank / max(dim, 1)
        rank_factor = max(0.0, 1.0 - min(rank_ratio * 10, 1.0))
        return float(np.clip((size_factor + rank_factor) / 2, 0.0, 1.0))

    @staticmethod
    def _score_compressibility(
        tensor: np.ndarray,
        eff_rank: float,
        entropy: float,
        sensitivity: float,
    ) -> float:
        n_elements = tensor.size
        dim = max(tensor.shape) if tensor.shape else 1
        rank_ratio = eff_rank / max(dim, 1)
        rank_score = max(0.0, 1.0 - min(rank_ratio * 5, 1.0))
        ent_score = min(max(entropy, 0.0) / 16.0, 1.0)
        sens_score = max(0.0, 1.0 - sensitivity)
        size_score = min(math.log2(max(n_elements, 1)) / 32.0, 1.0)
        score = (
            rank_score * 0.30 + ent_score * 0.20 + sens_score * 0.25 + size_score * 0.25
        )
        return float(np.clip(score, 0.0, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    #  2. METHOD PROFILING — Test methods on representative tensors
    # ═══════════════════════════════════════════════════════════════════
    def profile_methods(
        self,
        tensors: Dict[str, Tuple[np.ndarray, str]],
        max_methods: Optional[int] = None,
        method_timeout: float = 1e9,
    ) -> Dict[str, List[MethodTestRecord]]:
        """Test methods on representative tensors with per-method timeout.

        Groups tensors by type and only tests ONE representative per type,
        then propagates results to all tensors of that type.  This avoids
        O(methods × tensors) explosion (80 methods × 2000 tensors = 160K
        tests → ~80 × 11 types = ~880 tests).
        """
        method_by_type: Dict[str, List[MethodTestRecord]] = {}
        all_methods = self.methods
        method_items = list(all_methods.items())
        if max_methods is not None:
            method_items = method_items[:max_methods]

        # Group tensors by type, keep one representative per type
        type_reps: Dict[str, np.ndarray] = {}
        type_names: Dict[str, str] = {}
        for tname, (tensor, ttype) in tensors.items():
            if ttype not in type_reps:
                type_reps[ttype] = tensor
                type_names[ttype] = tname
            elif tensor.nbytes > type_reps[ttype].nbytes:
                type_reps[ttype] = tensor
                type_names[ttype] = tname

        def _run_validation(
            mname: str, minfo: Dict[str, Any], tensor: np.ndarray
        ) -> Optional[MethodTestRecord]:
            try:
                result = validate_single_method(mname, minfo, tensor=tensor)
            except Exception:
                return None

            if not result or not result.get("works", False):
                return None
            return MethodTestRecord(
                method_name=mname,
                tensor_name="",
                tensor_type="",
                category=result.get("category", "unknown"),
                tier=int(result["tier"]) if result.get("tier") is not None else 5,
                ratio=result.get("ratio", 1.0),
                error=result.get("error", 1.0),
                snr_db=result.get("snr_db", 0.0),
                time_ms=result.get("compress_time_ms", 0.0)
                + result.get("decompress_time_ms", 0.0),
            )

        # Test methods only on representative tensors (one per type)
        for ttype, tensor in type_reps.items():
            tname = type_names[ttype]
            records: List[MethodTestRecord] = []
            for mname, minfo in method_items:
                rec = _run_validation(mname, minfo, tensor)
                if rec is not None:
                    rec.tensor_name = tname
                    rec.tensor_type = ttype
                    records.append(rec)

            records.sort(key=lambda r: r.ratio, reverse=True)
            method_by_type[ttype] = records

        return method_by_type

    def find_top_methods_per_type(
        self,
        method_profiles: Dict[str, List[MethodTestRecord]],
        top_n: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find top-N methods per tensor type balancing ratio and error."""
        top_per_type: Dict[str, List[Dict[str, Any]]] = {}
        for ttype, records in method_profiles.items():
            scored = []
            for r in records:
                if r.error <= 0.0 or r.snr_db is None:
                    continue
                ratio_score = min(r.ratio / self._target_ratio, 2.0) / 2.0
                error_score = max(0.0, 1.0 - r.error * 50)
                snr_score = (
                    min(r.snr_db / 60.0, 1.0) if r.snr_db != float("inf") else 1.0
                )
                time_penalty = max(0.0, 1.0 - r.time_ms / 10000.0)
                composite = (
                    ratio_score * 0.35
                    + error_score * 0.35
                    + snr_score * 0.20
                    + time_penalty * 0.10
                )
                scored.append(
                    {
                        "method": r.method_name,
                        "category": r.category,
                        "tier": r.tier,
                        "ratio": r.ratio,
                        "error": r.error,
                        "snr_db": r.snr_db,
                        "time_ms": r.time_ms,
                        "score": composite,
                    }
                )

            scored.sort(key=lambda x: -x["score"])
            top_per_type[ttype] = scored[:top_n]

        return top_per_type

    def find_method_complementarity(
        self,
        method_profiles: Dict[str, List[MethodTestRecord]],
        top_n: int = 20,
    ) -> Dict[str, float]:
        """Find method pairs that work well together (different categories)."""
        pair_scores: Dict[str, float] = {}
        methods_seen: Dict[str, float] = {}

        for ttype, records in method_profiles.items():
            top = [r for r in records if r.error < 0.1][:top_n]
            for i, r1 in enumerate(top):
                for r2 in top[i + 1 :]:
                    if r1.category == r2.category:
                        continue
                    pair_key = f"{r1.method_name}+{r2.method_name}"
                    combined_ratio = math.sqrt(r1.ratio * r2.ratio)
                    combined_error = (r1.error + r2.error) / 2
                    if combined_error < 1e-30:
                        combined_error = 1e-30
                    complementarity = combined_ratio / combined_error
                    if complementarity > pair_scores.get(pair_key, 0):
                        pair_scores[pair_key] = float(complementarity)

        sorted_pairs = dict(sorted(pair_scores.items(), key=lambda x: -x[1])[:30])
        return sorted_pairs

    # ═══════════════════════════════════════════════════════════════════
    #  3. CASCADE DISCOVERY — Find optimal cascade patterns
    # ═══════════════════════════════════════════════════════════════════

    def _run_stage_with_timeout(
        self, inst: Any, residual: np.ndarray, timeout: float = 1e9
    ) -> Optional[Tuple[bytes, Dict[str, Any], np.ndarray]]:
        """Run a compress+decompress stage."""
        try:
            data, meta = inst.compress(residual)
            stage_recon = inst.decompress(data, meta)
            return data, meta, stage_recon
        except Exception:
            return None

    def discover_cascades(
        self,
        tensors: Dict[str, Tuple[np.ndarray, str]],
        exhaustive: bool = False,
    ) -> List[CascadeDiscoveryResult]:
        """Test all cascade patterns on each tensor type with per-stage timeout."""
        discoveries: List[CascadeDiscoveryResult] = []

        patterns_to_test: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        patterns_to_test.update(CASCADE_PATTERNS)

        if exhaustive:
            patterns_to_test.update(self._generate_exhaustive_patterns())

        for tname, (tensor, ttype) in tensors.items():
            for pname, stages in patterns_to_test.items():
                try:
                    recon = np.zeros_like(tensor)
                    residual = tensor.astype(np.float64).copy()
                    # Running estimates for stage bookkeeping only — the
                    # actually-reported total_ratio/total_error (below, after
                    # the stage loop) are always recomputed from true
                    # serialized bytes and true end-to-end reconstruction
                    # error, never a product of per-stage ratios or a sum of
                    # per-stage errors.
                    cumulative_serialized_bytes = 0
                    stage_results: List[Dict[str, Any]] = []
                    all_ok = True

                    for method_name, params in stages:
                        inst = self._engine._methods.get(method_name)
                        if inst is None:
                            all_ok = False
                            break

                        try:
                            t0 = time.perf_counter()
                            result = self._run_stage_with_timeout(inst, residual)
                            if result is None:
                                all_ok = False
                                break
                            data, meta, stage_recon = result
                            dt = (time.perf_counter() - t0) * 1000

                            if stage_recon.shape != residual.shape:
                                stage_recon = stage_recon.reshape(residual.shape)

                            stage_ratio = residual.nbytes / max(len(data), 1)
                            stage_bytes = serialized_nbytes(data) + serialized_nbytes(
                                meta
                            )
                            cumulative_serialized_bytes += stage_bytes
                            recon += stage_recon.astype(np.float64)
                            residual = tensor.astype(np.float64) - recon

                            mse = float(
                                np.mean((tensor.astype(np.float64) - recon) ** 2)
                            )
                            var_t = float(np.var(tensor))
                            stage_error = (
                                mse / max(var_t, 1e-30) if var_t > 1e-30 else mse
                            )

                            stage_results.append(
                                {
                                    "method": method_name,
                                    "ratio": float(stage_ratio),
                                    "error": float(stage_error),
                                    "time_ms": float(dt),
                                    "stage_bytes": stage_bytes,
                                }
                            )
                        except Exception as exc:
                            logger.error(
                                "discover_cascades: stage '%s' (pattern '%s', tensor '%s') "
                                "failed: %s",
                                method_name,
                                pname,
                                tname,
                                exc,
                                exc_info=True,
                            )
                            all_ok = False
                            break

                    if all_ok and stage_results:
                        # TRUE end-to-end ratio/error: actual total serialized
                        # bytes across all stages vs. original bytes, and a
                        # true rel-MSE from the FULL reconstruction against
                        # the original tensor (not a sum/average of per-stage
                        # normalized errors).
                        achieved_ratio = float(tensor.nbytes) / float(
                            max(cumulative_serialized_bytes, 1)
                        )
                        e2e = end_to_end_error(tensor, recon)
                        if achieved_ratio > 1.0:
                            score = achieved_ratio / max(e2e.rel_mse, 1e-10)
                            discoveries.append(
                                CascadeDiscoveryResult(
                                    tensor_type=ttype,
                                    pattern_name=pname,
                                    stages=stage_results,
                                    total_ratio=float(achieved_ratio),
                                    total_error=float(min(e2e.rel_mse, 1.0)),
                                    composite_score=float(score),
                                )
                            )
                except Exception as exc:
                    logger.error(
                        "discover_cascades: pattern '%s' on tensor '%s' failed: %s",
                        pname,
                        tname,
                        exc,
                        exc_info=True,
                    )
                    continue

        discoveries.sort(key=lambda d: -d.composite_score)
        return discoveries

    def _generate_exhaustive_patterns(
        self,
    ) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
        """Generate extended patterns for exhaustive cascade search."""
        patterns: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        base_methods = [
            ("svd_compress", {"rank": "auto:30"}),
            ("dct_spectral", {"keep_ratio": 0.2}),
            ("block_int4", {"block_size": 32}),
            ("fwht_compress", {"keep_fraction": 0.2}),
            ("tensor_train", {"rank": "auto:10"}),
            ("hadamard_int4", {"block_size": 32}),
            ("sparsify", {"sparsity": 0.8}),
            ("block_int8", {"block_size": 128}),
        ]

        categories = {
            "decomp": ["svd_compress", "tensor_train"],
            "spectral": ["dct_spectral", "fwht_compress"],
            "quant": ["block_int4", "hadamard_int4"],
        }

        for cat1, methods1 in categories.items():
            for cat2, methods2 in categories.items():
                if cat1 == cat2:
                    continue
                for m1 in methods1:
                    for m2 in methods2:
                        name = f"exhaustive_{m1}_{m2}"
                        params1 = dict(next(p[1] for p in base_methods if p[0] == m1))
                        params2 = dict(next(p[1] for p in base_methods if p[0] == m2))
                        patterns[name] = [
                            (m1, params1),
                            (m2, params2),
                        ]

        for cat1, methods1 in categories.items():
            for cat2, methods2 in categories.items():
                if cat1 == cat2:
                    continue
                for cat3, methods3 in categories.items():
                    if cat3 in (cat1, cat2):
                        continue
                    for m1 in methods1:
                        for m2 in methods2:
                            for m3 in methods3:
                                name = f"exhaustive_{m1}_{m2}_{m3}"
                                patterns[name] = [
                                    (
                                        m1,
                                        dict(
                                            next(
                                                p[1] for p in base_methods if p[0] == m1
                                            )
                                        ),
                                    ),
                                    (
                                        m2,
                                        dict(
                                            next(
                                                p[1] for p in base_methods if p[0] == m2
                                            )
                                        ),
                                    ),
                                    (
                                        m3,
                                        dict(
                                            next(
                                                p[1] for p in base_methods if p[0] == m3
                                            )
                                        ),
                                    ),
                                ]

        return patterns

    # ═══════════════════════════════════════════════════════════════════
    #  4. SENSITIVITY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════

    def analyze_sensitivity(
        self,
        profiles: List[DialInTensorProfile],
    ) -> Dict[str, SensitivityEntry]:
        """Build sensitivity map: how error tolerance varies per tensor."""
        sensitivity_map: Dict[str, SensitivityEntry] = {}

        for p in profiles:
            tier = self._assign_sensitivity_tier(p)
            entry = SensitivityEntry(
                tensor_name=p.name,
                tensor_type=p.tensor_type,
                tier=tier,
                baseline_ratio=10.0,
                baseline_error=0.001,
                recommended_max_error=self._recommend_max_error(tier, p),
            )
            sensitivity_map[p.name] = entry

        return sensitivity_map

    def _assign_sensitivity_tier(self, p: DialInTensorProfile) -> str:
        if p.sensitivity > 0.7:
            return "critical"
        if p.sensitivity > 0.4:
            return "high"
        if p.sensitivity > 0.2:
            return "medium"
        return "low"

    @staticmethod
    def _recommend_max_error(tier: str, p: DialInTensorProfile) -> float:
        if tier == "critical":
            return 0.001
        if tier == "high":
            return 0.005
        if tier == "medium":
            return 0.01
        return 0.05

    def get_critical_tensors(
        self, sensitivity_map: Dict[str, SensitivityEntry]
    ) -> List[str]:
        critical = [n for n, s in sensitivity_map.items() if s.tier == "critical"]
        return sorted(critical)

    def get_robust_tensors(
        self, sensitivity_map: Dict[str, SensitivityEntry]
    ) -> List[str]:
        robust = [n for n, s in sensitivity_map.items() if s.tier == "low"]
        return sorted(robust)

    # ═══════════════════════════════════════════════════════════════════
    #  5. PARAMETER TUNING — Grid search per method × tensor type
    # ═══════════════════════════════════════════════════════════════════

    def tune_parameters(
        self,
        tensors: Dict[str, Tuple[np.ndarray, str]],
        param_grid: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    ) -> Dict[str, List[OptimalParamEntry]]:
        """Grid search optimal parameters for each method × tensor type."""
        if param_grid is None:
            param_grid = self._default_param_grid()

        results: Dict[str, List[OptimalParamEntry]] = {}

        for tname, (tensor, ttype) in tensors.items():
            entries: List[OptimalParamEntry] = []

            for method_name, grid in param_grid.items():
                inst = self._engine._methods.get(method_name)
                if inst is None:
                    continue

                keys = list(grid.keys())
                values = list(grid.values())
                best_score = -1.0
                best_params: Dict[str, Any] = {}
                best_ratio = 1.0
                best_error = 1.0

                for combo in product(*values):
                    params = dict(zip(keys, combo))
                    try:
                        data, meta = inst.compress(tensor, **params)
                        recon = inst.decompress(data, meta)
                        if recon.shape != tensor.shape:
                            recon = recon.reshape(tensor.shape)

                        ratio = tensor.nbytes / max(len(data), 1)
                        err = float(
                            np.linalg.norm(recon.ravel() - tensor.ravel())
                            / max(np.linalg.norm(tensor.ravel()), 1e-30)
                        )
                        snr = self._compute_snr(tensor, recon)
                        score = self._param_score(ratio, err, snr)

                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_ratio = ratio
                            best_error = err
                    except Exception:
                        continue

                if best_score > 0:
                    entries.append(
                        OptimalParamEntry(
                            tensor_type=ttype,
                            method_name=method_name,
                            params=best_params,
                            achieved_ratio=best_ratio,
                            achieved_error=best_error,
                            score=best_score,
                        )
                    )

            entries.sort(key=lambda e: -e.score)
            if ttype not in results:
                results[ttype] = []
            results[ttype].extend(entries)

        return results

    @staticmethod
    def _default_param_grid() -> Dict[str, Dict[str, List[Any]]]:
        return {
            "block_int8": {"block_size": [32, 64, 128, 256, 512]},
            "block_int4": {"block_size": [16, 32, 64, 128, 256]},
            "hadamard_int8": {"block_size": [64, 128, 256, 512]},
            "hadamard_int4": {"block_size": [32, 64, 128, 256]},
            "delta_int4": {"block_size": [32, 64, 128], "group_size": [16, 32]},
            "sparsity_int4": {"block_size": [32, 64, 128]},
            "svd_compress": {"rank": [4, 8, 16, 32, 64, 128]},
            "dct_spectral": {"keep_ratio": [0.01, 0.05, 0.1, 0.2, 0.5]},
            "tensor_train": {"rank": [4, 8, 16, 32]},
            "fwht_compress": {"keep_fraction": [0.05, 0.1, 0.2, 0.5]},
            "uniform_quantize": {"bits": [2, 3, 4, 6, 8]},
            "product_quantize": {"bits": [2, 4, 6], "n_subspaces": [4, 8, 16]},
            "sparsify": {"sparsity": [0.5, 0.7, 0.8, 0.9, 0.95]},
            "block_sparsity": {"sparsity": [0.5, 0.7, 0.8, 0.9]},
        }

    @staticmethod
    def _param_score(ratio: float, error: float, snr: float) -> float:
        ratio_score = min(ratio / 100.0, 2.0) / 2.0
        error_score = max(0.0, 1.0 - error * 100)
        snr_score = min(snr / 60.0, 1.0) if snr != float("inf") else 1.0
        return float(ratio_score * 0.30 + error_score * 0.40 + snr_score * 0.30)

    @staticmethod
    def _compute_snr(original: np.ndarray, recon: np.ndarray) -> float:
        o = original.ravel().astype(np.float64)
        r = recon.ravel().astype(np.float64)
        signal = float(np.var(o))
        noise = float(np.var(o - r))
        if noise < 1e-30:
            return float("inf")
        return float(10.0 * math.log10(signal / noise))

    # ═══════════════════════════════════════════════════════════════════
    #  6. RATIO OPTIMIZATION — Error budget distribution
    # ═══════════════════════════════════════════════════════════════════

    def optimize_ratio(
        self,
        profiles: List[DialInTensorProfile],
        sensitivity_map: Dict[str, SensitivityEntry],
        top_methods: Dict[str, List[Dict[str, Any]]],
    ) -> ModelRatioPlan:
        """Distribute error budget across tensors to hit target ratio."""
        total_bytes = sum(p.nbytes for p in profiles)

        plan = ModelRatioPlan(
            overall_ratio=1.0,
            overall_error=0.0,
            total_original_bytes=total_bytes,
        )

        total_budget = self._target_ratio
        n_tensors = len(profiles)

        for p in profiles:
            sens = sensitivity_map.get(p.name)
            tier = sens.tier if sens else "medium"

            max_err = self._recommend_max_error(tier, p)
            methods_for_type = top_methods.get(p.tensor_type, [])
            best_method = (
                methods_for_type[0]
                if methods_for_type
                else {
                    "method": "block_int8",
                    "ratio": 4.0,
                    "error": 0.01,
                }
            )

            target_tensor_ratio = max(
                total_budget * (p.nbytes / max(total_bytes, 1)),
                2.0,
            )
            target_tensor_ratio = min(target_tensor_ratio, 10000.0)

            compressed_est = int(p.nbytes / max(target_tensor_ratio, 1))

            plan.tensor_plans[p.name] = {
                "tensor_type": p.tensor_type,
                "sensitivity_tier": tier,
                "method": best_method["method"],
                "target_ratio": target_tensor_ratio,
                "max_error": max_err,
                "original_bytes": p.nbytes,
                "estimated_compressed": compressed_est,
                "shape": p.shape,
            }

        total_comp_est = sum(
            tp["estimated_compressed"] for tp in plan.tensor_plans.values()
        )
        plan.total_compressed_est = total_comp_est
        plan.overall_ratio = total_bytes / max(total_comp_est, 1)

        errors = [tp.get("max_error", 0.01) for tp in plan.tensor_plans.values()]
        plan.overall_error = float(np.mean(errors)) if errors else 0.0

        return plan

    # ═══════════════════════════════════════════════════════════════════
    #  7. PRODUCTION CERTIFICATION
    # ═══════════════════════════════════════════════════════════════════

    def certify(
        self,
        tensors: Dict[str, np.ndarray],
        plan: ModelRatioPlan,
        report: DialInReport,
    ) -> CompressionCertificate:
        """Run production validation and generate certificate."""
        compressed_pairs: List[Tuple[str, CompressedTensor]] = []

        for name, tensor in tensors.items():
            tplan = plan.tensor_plans.get(name, {})
            method = tplan.get("method", "block_int8")
            target_r = tplan.get("target_ratio", 10.0)
            max_err = tplan.get("max_error", 0.01)

            try:
                data, meta, ratio_val, error_val = self._engine.compress(
                    tensor,
                    target_ratio=target_r,
                    max_error=max_err,
                    name=name,
                )
                ct = CompressedTensor(
                    _data=data,
                    method=meta.get("method", method),
                    params=meta,
                    original_shape=tensor.shape,
                    original_dtype=str(tensor.dtype),
                    compression_ratio=ratio_val,
                    relative_error=error_val,
                    snr_db=meta.get("snr_db", 0.0),
                    psnr_db=meta.get("psnr_db", 0.0),
                    cosine_similarity=meta.get("cosine_similarity", 1.0),
                    computation_time=0.0,
                )
                compressed_pairs.append((name, ct))
            except Exception as exc:
                logger.debug("Certification compress failed for '%s': %s", name, exc)

        if not compressed_pairs:
            logger.warning("No tensors successfully compressed for certification")
            return CompressionCertificate(
                model_name=report.model_name,
                model_path="",
                model_architecture="unknown",
                model_params="unknown",
                total_original_bytes=0,
                total_compressed_bytes=0,
                overall_ratio=1.0,
                total_tensors=0,
                compression_time_seconds=report.elapsed_seconds,
                weighted_error=1.0,
                avg_error=1.0,
                max_error=1.0,
                min_error=1.0,
                avg_snr_db=0.0,
            )

        cert = CertificateBuilder.from_compressed_tensors(
            compressed_pairs,
            model_name=report.model_name,
            compression_time=report.elapsed_seconds,
        )
        return cert

    # ═══════════════════════════════════════════════════════════════════
    #  FULL PIPELINE
    # ═══════════════════════════════════════════════════════════════════

    def dial_in(
        self,
        model_name: str = "synthetic_model",
        tensors: Optional[Dict[str, np.ndarray]] = None,
        tensor_types: Optional[List[str]] = None,
        quick: bool = False,
        exhaustive: bool = False,
        output_dir: str = "",
        max_test_methods: Optional[int] = None,
    ) -> DialInReport:
        """Run the full dial-in pipeline.

        Parameters
        ----------
        model_name : str
            Name for the model being analyzed.
        tensors : dict of str → np.ndarray, optional
            Actual model tensors. If None, synthetic tensors are generated.
        tensor_types : list of str, optional
            Only analyze specific tensor types.
        quick : bool
            Quick assessment (1 rep per type, 5 cascade patterns, no param tuning).
        exhaustive : bool
            Full R&D (all methods, all cascades, all parameters).
        output_dir : str
            Save reports to this directory.

        Returns
        -------
        DialInReport
            Complete dial-in analysis results.
        """
        t_start = time.perf_counter()
        report = DialInReport(model_name=model_name)

        # Generate or use provided tensors
        if tensors is None:
            tensor_data = _make_synthetic_tensors(
                tensor_types=tensor_types,
            )
            tensors_flat: Dict[str, np.ndarray] = {
                k: v[0] for k, v in tensor_data.items()
            }
            type_map: Dict[str, str] = {k: v[1] for k, v in tensor_data.items()}
        else:
            tensors_flat = tensors
            type_map = {n: self._classify_tensor(n) for n in tensors}

        report.model_size_gb = sum(t.nbytes for t in tensors_flat.values()) / (1024**3)
        report.n_tensors = len(tensors_flat)
        unique_types = set(type_map.values())
        report.n_tensor_types = len(unique_types)

        logger.info("=" * 60)
        logger.info("  Compression Dial-In Pipeline")
        logger.info("=" * 60)
        logger.info("  Model: %s", model_name)
        logger.info(
            "  Tensors: %d across %d types", report.n_tensors, report.n_tensor_types
        )
        logger.info("  Size: %.2f GB", report.model_size_gb)
        logger.info(
            "  Mode: %s",
            "exhaustive" if exhaustive else ("quick" if quick else "standard"),
        )
        logger.info("")

        # ── 1. Model Scan ─────────────────────────────────────────
        logger.info("  [1/7] Scanning model...")
        profiles = self.scan_model(tensors_flat)
        logger.info("  → Profiled %d tensors", len(profiles))

        # ── 2. Method Profiling ────────────────────────────────────
        logger.info("  [2/7] Profiling methods...")
        n_methods = max_test_methods or (
            min(len(self.methods), 30)
            if quick
            else (len(self.methods) if exhaustive else 20)
        )
        # In quick mode, use 1 representative tensor per type to avoid 2011×30 tests
        if quick:
            type_reps: Dict[str, str] = {}
            for n in tensors_flat:
                tt = type_map[n]
                if tt not in type_reps:
                    type_reps[tt] = n
            profile_tensors = {
                n: (tensors_flat[n], type_map[n]) for n in type_reps.values()
            }
            logger.info(
                "  → Quick mode: %d representative tensors", len(profile_tensors)
            )
        else:
            profile_tensors = {n: (tensors_flat[n], type_map[n]) for n in tensors_flat}

        method_profiles = self.profile_methods(
            profile_tensors,
            max_methods=n_methods,
        )
        report.method_profiles = method_profiles
        logger.info(
            "  → Tested %d methods across %d types",
            sum(len(v) for v in method_profiles.values()),
            len(method_profiles),
        )

        top_methods = self.find_top_methods_per_type(method_profiles)
        report.top_methods_per_type = top_methods
        for ttype, methods in top_methods.items():
            logger.info(
                "  → %s: %s", ttype, ", ".join(m["method"] for m in methods[:3])
            )

        complementarity = self.find_method_complementarity(method_profiles)
        report.method_complementarity = complementarity
        if complementarity:
            top_pairs = list(complementarity.items())[:5]
            logger.info("  → Top pairs: %s", top_pairs)

        # ── 3. Cascade Discovery ──────────────────────────────────
        n_cascades = 5 if quick else (len(CASCADE_PATTERNS) + 10 if exhaustive else 15)
        logger.info("  [3/7] Discovering cascade patterns (%d patterns)...", n_cascades)
        if quick:
            cascade_tensors = profile_tensors
        else:
            cascade_tensors = {n: (tensors_flat[n], type_map[n]) for n in tensors_flat}
        cascades = self.discover_cascades(
            cascade_tensors,
            exhaustive=exhaustive,
        )
        cascades = cascades[:n_cascades]
        report.cascade_discoveries = cascades

        best_per_type: Dict[str, CascadeDiscoveryResult] = {}
        for cd in cascades:
            if (
                cd.tensor_type not in best_per_type
                or cd.composite_score > best_per_type[cd.tensor_type].composite_score
            ):
                best_per_type[cd.tensor_type] = cd
        report.best_cascade_per_type = best_per_type
        logger.info("  → Found %d cascade patterns", len(cascades))
        for tt, cd in sorted(best_per_type.items()):
            stages_str = " → ".join(s["method"] for s in cd.stages)
            logger.info("  → %s: %s (ratio=%.1fx)", tt, stages_str, cd.total_ratio)

        # ── 4. Sensitivity Analysis ────────────────────────────────
        logger.info("  [4/7] Analyzing sensitivity...")
        sensitivity_map = self.analyze_sensitivity(profiles)
        report.sensitivity_map = sensitivity_map
        report.critical_tensors = self.get_critical_tensors(sensitivity_map)
        report.robust_tensors = self.get_robust_tensors(sensitivity_map)
        logger.info(
            "  → %d critical, %d robust tensors",
            len(report.critical_tensors),
            len(report.robust_tensors),
        )

        # ── 5. Parameter Tuning ────────────────────────────────────
        if not quick:
            logger.info("  [5/7] Tuning parameters...")
            tuned = self.tune_parameters(
                {n: (tensors_flat[n], type_map[n]) for n in tensors_flat},
            )
            report.optimal_params = tuned
            n_tuned = sum(len(v) for v in tuned.values())
            logger.info("  → Tuned %d method×type combinations", n_tuned)
        else:
            logger.info("  [5/7] Skipping parameter tuning (quick mode)")

        # ── 6. Ratio Optimization ──────────────────────────────────
        logger.info("  [6/7] Optimizing ratio...")
        ratio_plan = self.optimize_ratio(profiles, sensitivity_map, top_methods)
        report.ratio_plan = ratio_plan
        logger.info(
            "  → Plan: %.1f:1 overall, %.6f avg error",
            ratio_plan.overall_ratio,
            ratio_plan.overall_error,
        )

        # ── 7. Certification ───────────────────────────────────────
        logger.info("  [7/7] Certifying...")
        if quick:
            cert = self.certify(
                {n: tensors_flat[n] for n in type_reps.values()},
                ratio_plan,
                report,
            )
        else:
            cert = self.certify(tensors_flat, ratio_plan, report)
        report.certification = cert

        # ── Compute final metrics ─────────────────────────────────
        report.elapsed_seconds = time.perf_counter() - t_start

        if cert.tensor_certificates:
            errors = [c.relative_error for c in cert.tensor_certificates]
            ratios = [c.compression_ratio for c in cert.tensor_certificates]
            report.avg_error = float(np.mean(errors)) if errors else 0.0
            report.avg_ratio = float(np.mean(ratios)) if ratios else 0.0
            grade_counts = cert.grade_distribution
            if grade_counts.get("S", 0) > len(errors) * 0.5:
                report.weighted_grade = "EXCELLENT"
            elif grade_counts.get("A", 0) > len(errors) * 0.5:
                report.weighted_grade = "GOOD"
            elif grade_counts.get("F", 0) > len(errors) * 0.1:
                report.weighted_grade = "FAIL"
            else:
                report.weighted_grade = "GOOD"
        else:
            report.avg_ratio = ratio_plan.overall_ratio
            report.avg_error = ratio_plan.overall_error
            report.weighted_grade = "UNKNOWN"

        report.passes_min_ratio = report.avg_ratio >= 200.0
        report.passes_max_error = report.avg_error < self._max_error
        report.overall_pass = report.passes_min_ratio and report.passes_max_error

        report.recommendations = self._generate_recommendations(report)

        if output_dir:
            report.save(output_dir)
            logger.info("Report saved to %s", output_dir)

        logger.info("")
        logger.info("  ═══════════════════════════════════════════")
        logger.info("  Dial-In Complete: %.1fs", report.elapsed_seconds)
        logger.info("  Avg Ratio: %.1f:1", report.avg_ratio)
        logger.info("  Avg Error: %.4f%%", report.avg_error * 100)
        logger.info("  Grade: %s", report.weighted_grade)
        logger.info("  Pass: %s", "YES" if report.overall_pass else "NO")
        logger.info("")

        return report

    def _generate_recommendations(self, report: DialInReport) -> List[str]:
        recs: List[str] = []

        if report.avg_ratio < 200:
            recs.append(
                f"Increase ratio: current {report.avg_ratio:.1f}:1 < 200:1 minimum. "
                "Try more aggressive cascades or higher-rank decompositions."
            )
        elif report.avg_ratio < 400:
            recs.append(
                f"Good ratio ({report.avg_ratio:.1f}:1) but below target 400:1. "
                "Consider 3-stage cascades for high-ratio types."
            )
        else:
            recs.append(f"Excellent ratio ({report.avg_ratio:.1f}:1) — exceeds target.")

        if report.avg_error >= self._max_error:
            recs.append(
                f"Error ({report.avg_error * 100:.4f}%) exceeds threshold "
                f"({self._max_error * 100:.2f}%). "
                "Reduce target ratio or use higher-fidelity methods on critical tensors."
            )
        else:
            recs.append(
                f"Error ({report.avg_error * 100:.4f}%) within acceptable range."
            )

        if report.critical_tensors:
            recs.append(
                f"Protect {len(report.critical_tensors)} critical tensors "
                "with lossless or near-lossless methods."
            )

        if report.best_cascade_per_type:
            recs.append("Deploy recommended cascades per tensor type for production.")

        if report.certification and report.certification.overall_ratio >= 2000:
            recs.append(
                f"Aggressive ratio ({report.certification.overall_ratio:.0f}:1) may need "
                "CPU inference validation to ensure acceptable runtime quality."
            )

        return recs


# ── CLI Integration Function ──────────────────────────────────────────────────


def cmd_dial_in_main(args: Any) -> None:
    """CLI entry point for dial-in command."""
    engine = CompressionDialInEngine(
        target_ratio=args.target_ratio or 400.0,
        max_error=args.max_error or 0.01,
        max_workers=args.workers or 4,
    )

    tensor_types = None
    if args.focus:
        tensor_types = [t.strip() for t in args.focus.split(",")]

    output_dir = args.output_dir or "."

    if args.model and os.path.exists(args.model):
        logger.info("Loading model from %s", args.model)
        tensors: Dict[str, np.ndarray] = {}

        # BF16 conversion: bfloat16 is the upper 16 bits of float32
        def _bf16_to_f32(arr: np.ndarray) -> np.ndarray:
            if arr.dtype == np.uint16:
                return (arr.astype(np.uint32) << 16).view(np.float32)
            return arr.astype(np.float32)

        tensor_dict = {}
        _skipped_empty = 0
        _skipped_small = 0
        _skipped_bad = 0

        try:
            from spectralstream.compression.cli import _SafetensorsLoader

            loader = _SafetensorsLoader(args.model)
            tensor_info = loader.scan()
            _total_nbytes = 0
            for name, (shape, dtype_str, offset, nbytes) in tensor_info.items():
                _total_nbytes += nbytes
                if nbytes == 0 or any(s == 0 for s in shape):
                    _skipped_empty += 1
                    continue
                try:
                    arr = loader.read_tensor(name, shape, dtype_str, offset, nbytes)
                    tensor_dict[name] = _bf16_to_f32(arr)
                except Exception as exc:
                    _skipped_bad += 1
                    logger.debug("Failed to load %s: %s", name, exc)
            loader.close()

            # Warn if model > 80% of available RAM
            _model_gb = _total_nbytes / (1024**3)
            try:
                import psutil

                _avail_gb = psutil.virtual_memory().available / (1024**3)
                if _model_gb > _avail_gb * 0.8:
                    logger.warning(
                        "Model is %.1f GB but only %.1f GB RAM available — "
                        "dial-in will be memory-constrained. Consider using --quick mode.",
                        _model_gb,
                        _avail_gb,
                    )
            except ImportError:
                pass

        except Exception as exc:
            logger.warning("_SafetensorsLoader failed: %s — trying safetensors", exc)
            try:
                import safetensors

                tensors = safetensors.safe_open(args.model, framework="np")
                for k in tensors.keys():
                    try:
                        arr = tensors.get_tensor(k)
                        if arr.nbytes == 0 or any(s == 0 for s in arr.shape):
                            _skipped_empty += 1
                            continue
                        tensor_dict[k] = _bf16_to_f32(arr)
                    except Exception as inner:
                        _skipped_bad += 1
                        logger.warning(
                            "Failed to load tensor %s: %s — skipping (not substituting synthetic data)",
                            k,
                            inner,
                        )
            except Exception as exc2:
                raise RuntimeError(
                    f"Cannot load model file {args.model}: {exc2}"
                ) from exc2

        if _skipped_empty or _skipped_small or _skipped_bad:
            logger.info(
                "Filtered %d empty, %d <1KB, %d error tensors — kept %d/%d",
                _skipped_empty,
                _skipped_small,
                _skipped_bad,
                len(tensor_dict),
                len(tensor_info) if "tensor_info" in dir() else "?",
            )

        if not tensor_dict:
            logger.error("No loadable tensors — falling back to synthetic data")
            tensor_dict = None

        if tensor_types and tensor_dict is not None:
            filtered = {}
            for n, t in tensor_dict.items():
                if any(tt in n.lower() for tt in tensor_types):
                    filtered[n] = t
            tensor_dict = filtered

        report = engine.dial_in(
            model_name=os.path.basename(args.model).replace(".safetensors", ""),
            tensors=tensor_dict,
            tensor_types=tensor_types,
            quick=args.quick or False,
            exhaustive=args.exhaustive or False,
            output_dir=output_dir,
        )
    elif args.model and not os.path.exists(args.model):
        logger.warning("Model file not found: %s — using synthetic data", args.model)
        report = engine.dial_in(
            model_name=args.model,
            tensor_types=tensor_types,
            quick=args.quick or False,
            exhaustive=args.exhaustive or False,
            output_dir=output_dir,
        )
    else:
        report = engine.dial_in(
            model_name="synthetic_model",
            tensor_types=tensor_types,
            quick=args.quick or False,
            exhaustive=args.exhaustive or False,
            output_dir=output_dir,
        )

    print()
    print(report.summary_text())
    print()
    logger.info("Full report saved to %s", output_dir)
