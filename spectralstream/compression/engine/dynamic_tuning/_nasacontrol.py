"""
NASA Mission Control Compressor

Models compression as a space mission — each tensor is a "planet" to explore:
  Tier 1 (Flyby):  fast reconnaissance with decomposition/spectral
  Tier 2 (Orbiter): detailed mapping with structural/physics
  Tier 3 (Lander):  precision sampling with entropy/lossless
  Tier 4 (Rover):   adaptive exploration with hybrid methods

Each phase has go/no-go checks before proceeding. Redundant fallbacks
if quality breaches limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MissionPhase:
    name: str = ""
    tier_level: int = 1
    status: str = "pending"
    go_nogo: bool = False
    ratio_achieved: float = 1.0
    error_achieved: float = 0.0
    method_used: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class MissionReport:
    phases: List[MissionPhase] = field(default_factory=list)
    total_ratio: float = 1.0
    total_error: float = 0.0
    mission_time: float = 0.0
    aborted: bool = False
    abort_reason: str = ""
    total_planets: int = 0
    planets_explored: int = 0


class NASAControlCompressor:
    """Mission-control style compressor with phased exploration.

    Each tensor is a 'planet' explored in phases:
      1. Flyby   — fast recon (decomposition + spectral)
      2. Orbiter — detailed mapping (structural + physics)
      3. Lander  — precision sampling (entropy + lossless)
      4. Rover   — adaptive hybrid methods

    Go/no-go checks at each phase before proceeding.
    Fallback methods if quality breaches limits.
    """

    FLYBY_METHODS: List[str] = [
        "svd_compress",
        "tensor_train",
        "dct_spectral",
        "fwht_compress",
    ]
    ORBITER_METHODS: List[str] = [
        "einsort",
        "circulant",
        "monarch_structured",
        "butterfly_structured",
    ]
    LANDER_METHODS: List[str] = ["rans", "huffman", "arithmetic", "lossless_zstd"]
    ROVER_METHODS: List[str] = [
        "cascade_2_stage",
        "cascade_3_stage",
        "quantize_then_sparsify",
    ]

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.mission_log: Dict[str, MissionReport] = {}
        self._active_mission: Optional[MissionReport] = None

    def suggest_sequences(
        self, profile: Any, target_ratio: float = 1200.0
    ) -> List[Dict[str, Any]]:
        """Return ranked method list following mission phases."""
        sequences: List[Dict[str, Any]] = []
        n_elements = getattr(profile, "n_elements", 1)
        effective_rank = getattr(profile, "effective_rank", 0.5)
        spectral_entropy = getattr(profile, "spectral_entropy", 0.5)
        name = getattr(profile, "name", "")

        planet_name = name if name else f"planet_{hash(str(profile.shape)) % 10000}"
        report = MissionReport()
        report.total_planets = 1
        phase_idx = 0

        # Phase 1: Flyby — fast reconnaissance
        phase_idx += 1
        flyby_seq = self._build_flyby_sequence(profile, target_ratio)
        for seq in flyby_seq:
            seq["mission_phase"] = "flyby"
            seq["weight"] = 5.0
            sequences.append(seq)

        # Phase 2: Orbiter — detailed mapping
        phase_idx += 1
        if target_ratio > 500:
            orbiter_seq = self._build_orbiter_sequence(profile, target_ratio)
            for seq in orbiter_seq:
                seq["mission_phase"] = "orbiter"
                seq["weight"] = 3.0
                sequences.append(seq)

        # Phase 3: Lander — precision sampling
        phase_idx += 1
        if target_ratio > 1000:
            lander_seq = self._build_lander_sequence(profile, target_ratio)
            for seq in lander_seq:
                seq["mission_phase"] = "lander"
                seq["weight"] = 2.0
                sequences.append(seq)

        # Phase 4: Rover — adaptive hybrid
        phase_idx += 1
        if target_ratio > 3000:
            rover_seq = self._build_rover_sequence(profile, target_ratio)
            for seq in rover_seq:
                seq["mission_phase"] = "rover"
                seq["weight"] = 1.5
                sequences.append(seq)

        sequences.sort(key=lambda s: s.get("weight", 1.0), reverse=True)
        report.phases = [
            MissionPhase(name="flyby", tier_level=1),
            MissionPhase(name="orbiter", tier_level=2),
            MissionPhase(name="lander", tier_level=3),
            MissionPhase(name="rover", tier_level=4),
        ]
        self.mission_log[planet_name] = report
        self._active_mission = report

        return sequences

    def _build_flyby_sequence(
        self, profile: Any, target_ratio: float
    ) -> List[Dict[str, Any]]:
        """Tier 1 flyby: fast reconnaissance via decomposition + spectral."""
        sequences: List[Dict[str, Any]] = []
        effective_rank = getattr(profile, "effective_rank", 0.5)

        if effective_rank < 0.1:
            sequences.append(
                {
                    "method_name": "svd_compress",
                    "params": {"rank_frac": 0.02},
                    "expected_ratio": 50.0,
                    "expected_error": 0.005,
                }
            )
        else:
            sequences.append(
                {
                    "method_name": "tensor_train",
                    "params": {"rank_frac": 0.03},
                    "expected_ratio": 35.0,
                    "expected_error": 0.004,
                }
            )

        sequences.append(
            {
                "method_name": "dct_spectral",
                "params": {"keep_frac": 0.25},
                "expected_ratio": 4.0,
                "expected_error": 0.002,
            }
        )

        return sequences

    def _build_orbiter_sequence(
        self, profile: Any, target_ratio: float
    ) -> List[Dict[str, Any]]:
        """Tier 2 orbiter: detailed mapping with structural methods."""
        sequences: List[Dict[str, Any]] = []
        block_diag = getattr(profile, "block_diagonal_score", 0.0)

        if block_diag > 0.5:
            sequences.append(
                {
                    "method_name": "einsort",
                    "params": {"block_size": 64},
                    "expected_ratio": 3.0,
                    "expected_error": 0.003,
                }
            )
        else:
            sequences.append(
                {
                    "method_name": "circulant",
                    "params": {"block_size": 64},
                    "expected_ratio": 2.5,
                    "expected_error": 0.002,
                }
            )

        return sequences

    def _build_lander_sequence(
        self, profile: Any, target_ratio: float
    ) -> List[Dict[str, Any]]:
        """Tier 3 lander: precision sampling with entropy coding."""
        entropy_rate = getattr(profile, "entropy_rate", 0.5)

        if entropy_rate > 0.7:
            return [
                {
                    "method_name": "rans",
                    "params": {"method": "rans"},
                    "expected_ratio": 2.0,
                    "expected_error": 0.0,
                }
            ]
        else:
            return [
                {
                    "method_name": "huffman",
                    "params": {"method": "huffman"},
                    "expected_ratio": 1.8,
                    "expected_error": 0.0,
                }
            ]

    def _build_rover_sequence(
        self, profile: Any, target_ratio: float
    ) -> List[Dict[str, Any]]:
        """Tier 4 rover: adaptive hybrid methods."""
        return [
            {
                "method_name": "cascade_2_stage",
                "params": {},
                "expected_ratio": 5.0,
                "expected_error": 0.01,
            }
        ]

    def go_nogo_check(
        self, phase_name: str, error: float, error_budget: float
    ) -> Tuple[bool, str]:
        """Go/no-go check: proceed only if error is within budget.

        Returns
        -------
        (go: bool, reason: str)
        """
        if error > error_budget * 1.5:
            return (
                False,
                f"ERROR {error:.6f} exceeds budget {error_budget:.6f} — ABORT",
            )
        if error > error_budget:
            return (
                False,
                f"ERROR {error:.6f} exceeds budget {error_budget:.6f} — FALLBACK",
            )
        return (
            True,
            f"GO for {phase_name} (error {error:.6f} within budget {error_budget:.6f})",
        )

    def recommend_fallback(self, failed_method: str, error: float) -> Dict[str, Any]:
        """Recommend fallback method if a mission phase fails."""
        phase_map: Dict[str, List[str]] = {
            "svd_compress": ["tensor_train", "cp_decomposition", "butterfly"],
            "tensor_train": ["svd_compress", "tt_rank16", "tt_rank32"],
            "dct_spectral": ["fwht_compress", "dct_2d", "dct_block"],
            "fwht_compress": ["dct_spectral", "dct_2d", "wavelet_compress"],
            "einsort": ["circulant", "monarch_structured", "vandermonde"],
            "circulant": ["einsort", "butterfly_structured", "block_sparse"],
            "rans": ["huffman", "arithmetic", "lossless_zstd"],
            "huffman": ["rans", "arithmetic", "lossless_zstd"],
        }
        alternatives = phase_map.get(failed_method, ["block_int8", "hadamard_int8"])
        if alternatives:
            alt = alternatives[0]
            return {
                "method_name": alt,
                "reason": f"Fallback from {failed_method} (error {error:.6f})",
                "expected_ratio": 3.0,
                "expected_error": 0.005,
            }
        return {
            "method_name": "block_int8",
            "reason": "Final fallback",
            "expected_ratio": 2.0,
            "expected_error": 0.01,
        }

    def get_mission_report(self, planet_name: str = "") -> Dict[str, Any]:
        """Return the mission report for a given planet (or active mission)."""
        if planet_name and planet_name in self.mission_log:
            report = self.mission_log[planet_name]
        elif self._active_mission:
            report = self._active_mission
        else:
            return {"status": "no active mission"}

        return {
            "phases": [
                {"name": p.name, "tier": p.tier_level, "status": p.status}
                for p in report.phases
            ],
            "total_ratio": report.total_ratio,
            "total_error": report.total_error,
            "planets_explored": report.planets_explored,
            "aborted": report.aborted,
            "abort_reason": report.abort_reason,
        }

    def get_dashboard_string(self) -> str:
        """ASCII dashboard for CLI display."""
        report = self.get_mission_report()
        lines = ["NASA Mission Control — Compression Mission Status"]
        lines.append(f"  Planets Explored: {report.get('planets_explored', 0)}")
        lines.append(f"  Total Ratio: {report.get('total_ratio', 1.0):.1f}x")
        lines.append(f"  Total Error: {report.get('total_error', 0.0):.6f}")
        lines.append(f"  Aborted: {report.get('aborted', False)}")
        if report.get("abort_reason"):
            lines.append(f"  Abort Reason: {report['abort_reason']}")
        lines.append("  Phases:")
        for phase in report.get("phases", []):
            status_mark = "✓" if phase.get("status") == "completed" else "○"
            lines.append(
                f"    {status_mark} {phase.get('name', 'unknown')} (Tier {phase.get('tier', 0)}) [{phase.get('status', 'pending')}]"
            )
        return "\n".join(lines)
