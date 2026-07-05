"""
Formula 1 Telemetry Cascade Optimizer

Each compression method is an F1 car system:
  DRS  (drag reduction)  = structural methods   (einsort, circulant, monarch)
  ERS  (energy recovery) = decomposition         (SVD, TT, CP, Tucker)
  Turbo                   = quantization          (BlockINT8/4, Hadamard)
  Diffuser                = spectral              (DCT, FWHT, Wavelets)
  Chassis                 = physics/tensor network (MERA, PEPS, MPS)

Modes:
  Qualifying — push ALL methods to maximum aggression for peak ratio
  Race       — balance ratio vs error like tire management
  Pit stop   — when a method degrades quality, "pit" it and switch
  Overtake   — temporary boost on critical layers
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TelemetryPacket:
    timestamp: float = 0.0
    stage_name: str = ""
    method_name: str = ""
    ratio: float = 1.0
    error: float = 0.0
    snr_db: float = 40.0
    cumulative_ratio: float = 1.0
    cumulative_error: float = 0.0
    mode: str = "race"
    pit_stop_count: int = 0
    throttle_pct: float = 100.0


@dataclass
class PitStopRecommendation:
    current_method: str = ""
    replacement_method: str = ""
    reason: str = ""
    confidence: float = 0.0


@dataclass
class QualifyingModeProfile:
    max_aggression: bool = True
    ers_deploy_pct: float = 100.0
    drs_open: bool = True
    turbo_boost: float = 1.5
    diffuser_stall: float = 1.3


@dataclass
class RaceModeProfile:
    tire_conserve_pct: float = 70.0
    ers_harvest_pct: float = 40.0
    drs_activation_pct: float = 60.0
    turbo_wastegate_pct: float = 50.0
    diffuser_angle_pct: float = 75.0


F1_SYSTEM_MAP = {
    "drs_structural": {
        "category": "structural",
        "tier": 2,
        "description": "DRS — drag reduction via matrix structure exploitation",
    },
    "ers_decomposition": {
        "category": "decomposition",
        "tier": 1,
        "description": "ERS — energy recovery via low-rank decomposition",
    },
    "turbo_quantization": {
        "category": "quantization",
        "tier": 5,
        "description": "Turbo — aggressive bit-width reduction",
    },
    "diffuser_spectral": {
        "category": "spectral",
        "tier": 1,
        "description": "Diffuser — spectral energy concentration",
    },
    "chassis_physics": {
        "category": "physics",
        "tier": 3,
        "description": "Chassis — physics-inspired tensor networks",
    },
    "exhaust_entropy": {
        "category": "entropy",
        "tier": 3,
        "description": "Exhaust — lossless entropy coding",
    },
}


class F1CascadeOptimizer:
    """Compression strategy optimizer modelled on F1 car systems.

    Strategies:
      *qualifying* — maximum aggression, peak ratio at all costs
      *race* — balanced like tyre management, conserve quality
      *pit_stop* — hot-swap degrading methods
      *overtake* — temporary boost on critical layers
    """

    QUALIFYING_MAP: Dict[str, str] = {
        "decomposition": "svd_compress",
        "spectral": "dct_spectral",
        "quantization": "block_int4",
        "structural": "einsort",
        "entropy": "rans",
    }

    RACE_MAP: Dict[str, str] = {
        "decomposition": "tensor_train",
        "spectral": "dct_spectral",
        "quantization": "block_int8",
        "structural": "circulant",
        "entropy": "huffman",
    }

    PIT_STOP_ALTERNATIVES: Dict[str, List[str]] = {
        "svd_compress": ["tensor_train", "cp_decomposition", "butterfly"],
        "tensor_train": ["svd_compress", "tt_rank16", "cp_decomposition"],
        "dct_spectral": ["fwht_compress", "dct_2d", "dct_block"],
        "fwht_compress": ["dct_spectral", "wavelet_compress", "dct_2d"],
        "block_int4": ["hadamard_int4", "sparsity_int4", "delta_int4"],
        "block_int8": ["hadamard_int8", "block_int4", "nf4_quant"],
        "einsort": ["circulant", "monarch_structured", "vandermonde"],
        "circulant": ["einsort", "block_sparse", "butterfly_structured"],
        "rans": ["huffman", "arithmetic", "tans"],
        "huffman": ["rans", "arithmetic", "lz77"],
    }

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.telemetry: List[TelemetryPacket] = []
        self.pit_stops: int = 0
        self.overtake_active: bool = False
        self.current_mode: str = "race"
        self.qualifying_profile = QualifyingModeProfile()
        self.race_profile = RaceModeProfile()

    def suggest_sequences(
        self, profile: Any, target_ratio: float = 1200.0
    ) -> List[Dict[str, Any]]:
        """Return ranked method list based on F1 mode and profile."""
        sequences: List[Dict[str, Any]] = []
        n_elements = getattr(profile, "n_elements", 1)
        rank = getattr(profile, "effective_rank", 0.5)
        entropy_rate = getattr(profile, "entropy_rate", 0.5)
        spectral_entropy = getattr(profile, "spectral_entropy", 0.5)

        is_high_rank = rank > 0.3 * min(profile.shape) if profile.shape else False
        is_low_rank = rank < 0.05 * min(profile.shape) if profile.shape else False

        if self.current_mode == "qualifying":
            sequences = self._qualifying_sequences(profile, target_ratio)
        elif self.current_mode == "race":
            sequences = self._race_sequences(profile, target_ratio)
        else:
            sequences = self._race_sequences(profile, target_ratio)

        if is_low_rank:
            pref = "ers_decomposition"
            sequences = [s for s in sequences if s.get("f1_system") != pref] + [
                s for s in sequences if s.get("f1_system") == pref
            ]

        if is_high_rank and target_ratio > 1000:
            pref = "turbo_quantization"
            sequences = [s for s in sequences if s.get("f1_system") != pref] + [
                s for s in sequences if s.get("f1_system") == pref
            ]

        telemetry = TelemetryPacket(
            timestamp=time.time(),
            stage_name="suggest_sequences",
            mode=self.current_mode,
            cumulative_ratio=float(target_ratio),
        )
        self.telemetry.append(telemetry)

        return sequences

    def _qualifying_sequences(
        self, profile: Any, target_ratio: float
    ) -> List[Dict[str, Any]]:
        """Qualifying mode: maximum aggression, push everything.

        Like F1 qualifying — every system at 100% deployment.
        """
        sequences: List[Dict[str, Any]] = []
        qp = self.QUALIFYING_MAP
        boost = self.qualifying_profile.turbo_boost

        # ERS deploy: decomposition at max aggression
        sequences.append(
            {
                "method_name": qp["decomposition"],
                "params": {"rank_frac": 0.008 / boost},
                "f1_system": "ers_decomposition",
                "weight": 5.0 * boost,
                "expected_ratio": 120 * boost,
                "expected_error": 0.008 * boost,
            }
        )

        # Diffuser open: spectral at full
        diff_stall = self.qualifying_profile.diffuser_stall
        sequences.append(
            {
                "method_name": qp["spectral"],
                "params": {"keep_frac": 0.08 / diff_stall},
                "f1_system": "diffuser_spectral",
                "weight": 4.0 * diff_stall,
                "expected_ratio": 10 * diff_stall,
                "expected_error": 0.005 * diff_stall,
            }
        )

        # DRS open: structural methods
        if self.qualifying_profile.drs_open:
            sequences.append(
                {
                    "method_name": qp["structural"],
                    "params": {"block_size": 32},
                    "f1_system": "drs_structural",
                    "weight": 3.0,
                    "expected_ratio": 3.0,
                    "expected_error": 0.003,
                }
            )

        # Turbo: quantization
        sequences.append(
            {
                "method_name": qp["quantization"],
                "params": {"bits": 4},
                "f1_system": "turbo_quantization",
                "weight": 5.0 * boost,
                "expected_ratio": 8.0 * boost,
                "expected_error": 0.015 * boost,
            }
        )

        # Exhaust: entropy coding
        sequences.append(
            {
                "method_name": qp["entropy"],
                "params": {"method": "rans"},
                "f1_system": "exhaust_entropy",
                "weight": 2.0,
                "expected_ratio": 2.0,
                "expected_error": 0.0,
            }
        )

        sequences.sort(key=lambda s: s.get("weight", 1.0), reverse=True)
        return sequences

    def _race_sequences(
        self, profile: Any, target_ratio: float
    ) -> List[Dict[str, Any]]:
        """Race mode: balance ratio vs error like tire management.

        Conservative deployment rates, conserve quality where possible.
        """
        sequences: List[Dict[str, Any]] = []
        rp = self.RACE_MAP
        conserve = self.race_profile.tire_conserve_pct / 100.0

        sequences.append(
            {
                "method_name": rp["decomposition"],
                "params": {"rank_frac": 0.03 / conserve},
                "f1_system": "ers_decomposition",
                "weight": 4.0 * conserve,
                "expected_ratio": 35.0 * conserve,
                "expected_error": 0.003 / conserve,
            }
        )

        sequences.append(
            {
                "method_name": rp["spectral"],
                "params": {"keep_frac": 0.3 / conserve},
                "f1_system": "diffuser_spectral",
                "weight": 3.0 * conserve,
                "expected_ratio": 4.0 * conserve,
                "expected_error": 0.002 / conserve,
            }
        )

        drs_dep = self.race_profile.drs_activation_pct / 100.0
        if drs_dep > 0.3:
            sequences.append(
                {
                    "method_name": rp["structural"],
                    "params": {"block_size": 64},
                    "f1_system": "drs_structural",
                    "weight": 2.0 * drs_dep,
                    "expected_ratio": 2.0 * drs_dep,
                    "expected_error": 0.002,
                }
            )

        turbo = self.race_profile.turbo_wastegate_pct / 100.0
        if target_ratio > 2000:
            sequences.append(
                {
                    "method_name": rp["quantization"],
                    "params": {"bits": 8},
                    "f1_system": "turbo_quantization",
                    "weight": 3.0 * turbo,
                    "expected_ratio": 4.0 * turbo,
                    "expected_error": 0.005,
                }
            )

        sequences.append(
            {
                "method_name": rp["entropy"],
                "params": {"method": "huffman"},
                "f1_system": "exhaust_entropy",
                "weight": 1.5,
                "expected_ratio": 1.8,
                "expected_error": 0.0,
            }
        )

        sequences.sort(key=lambda s: s.get("weight", 1.0), reverse=True)
        return sequences

    def set_qualifying_mode(self) -> None:
        """Switch to qualifying mode — maximum aggression."""
        self.current_mode = "qualifying"

    def set_race_mode(self) -> None:
        """Switch to race mode — balanced."""
        self.current_mode = "race"

    def activate_overtake(self, profile: Any) -> Dict[str, Any]:
        """Overtake button: temporary boost on critical layers.

        Temporarily increases quantization aggression on attention layers.
        """
        self.overtake_active = True
        layer_name = getattr(profile, "name", "")
        is_critical = any(
            k in layer_name.lower()
            for k in ("q_proj", "k_proj", "v_proj", "o_proj", "attn")
        )

        boost_params: Dict[str, Any] = {}
        if is_critical:
            boost_params = {
                "quant_bits": 4,
                "drs_open": True,
                "turbo_boost": 1.8,
                "duration_seconds": 30.0,
            }
        else:
            boost_params = {
                "quant_bits": 6,
                "drs_open": True,
                "turbo_boost": 1.3,
                "duration_seconds": 15.0,
            }

        self.telemetry.append(
            TelemetryPacket(
                timestamp=time.time(),
                stage_name="overtake",
                mode=self.current_mode,
                throttle_pct=boost_params.get("turbo_boost", 1.0) * 100.0,
            )
        )
        return boost_params

    def deactivate_overtake(self) -> None:
        """Deactivate overtake mode."""
        self.overtake_active = False

    def recommend_pit_stop(
        self, current_method: str, current_error: float, error_budget: float
    ) -> PitStopRecommendation:
        """Recommend a pit stop if a method degrades quality.

        When a method's error exceeds budget, 'pit' it and switch.
        """
        alternatives = self.PIT_STOP_ALTERNATIVES.get(current_method, [])
        if current_error <= error_budget or not alternatives:
            return PitStopRecommendation(
                current_method=current_method,
                replacement_method=current_method,
                reason="No pit stop needed",
                confidence=1.0,
            )

        replacement = alternatives[0]
        self.pit_stops += 1
        self.telemetry.append(
            TelemetryPacket(
                timestamp=time.time(),
                stage_name=f"pit_stop_{self.pit_stops}",
                method_name=current_method,
                pit_stop_count=self.pit_stops,
                mode=self.current_mode,
            )
        )
        return PitStopRecommendation(
            current_method=current_method,
            replacement_method=replacement,
            reason=f"Error {current_error:.6f} exceeds budget {error_budget:.6f}",
            confidence=max(0.0, 1.0 - current_error / max(error_budget, 1e-10)),
        )

    def get_telemetry_dashboard(self) -> Dict[str, Any]:
        """Return telemetry dashboard data."""
        if not self.telemetry:
            return {"status": "no telemetry", "packets": []}
        return {
            "mode": self.current_mode,
            "pit_stops": self.pit_stops,
            "overtake_active": self.overtake_active,
            "packets": [
                {
                    "t": p.timestamp,
                    "stage": p.stage_name,
                    "method": p.method_name,
                    "ratio": p.ratio,
                    "error": p.error,
                    "snr": p.snr_db,
                    "mode": p.mode,
                    "pit_stops": p.pit_stop_count,
                }
                for p in self.telemetry[-50:]
            ],
        }

    def get_dashboard_string(self) -> str:
        """ASCII dashboard for CLI display."""
        dash = self.get_telemetry_dashboard()
        lines = [f"F1 Telemetry Dashboard — Mode: {dash['mode'].upper()}"]
        lines.append(
            f"  Pit Stops: {dash['pit_stops']}  Overtake: {'ACTIVE' if dash['overtake_active'] else 'INACTIVE'}"
        )
        lines.append(f"  Recent Telemetry ({len(dash['packets'])} packets):")
        for pkt in dash["packets"][-5:]:
            lines.append(
                f"    [{pkt['stage']}] {pkt.get('method', '')} "
                f"ratio={pkt['ratio']:.1f}x err={pkt['error']:.6f} "
                f"SNR={pkt['snr']:.1f}dB"
            )
        return "\n".join(lines)
