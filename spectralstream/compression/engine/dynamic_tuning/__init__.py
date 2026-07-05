"""
Dynamic Tuning Package — auto-tunes compression method parameters to target ratios.

Exports:
  TunedParams           — tuning result dataclass
  tune_method           — high-level dispatch: tune any method by name
  list_tunable_methods  — enumerate all tunable methods
  QUANTIZATION_TUNERS   — dict: quant method name → tuner function
  PHYSICS_TUNERS        — dict: physics method name → tuner function
  ALL_TUNERS            — combined dispatch table

Individual tuner functions (one per method) also available.
"""

from __future__ import annotations


from .quantization_tuning import (
    ALL_TUNERS,
    PHYSICS_TUNERS,
    QUANTIZATION_TUNERS,
    TunedParams,
    list_tunable_methods,
    tune_method,
)

from .pareto_streaming import (
    ParetoFrontier,
    ParetoPoint,
    ProgressiveStreamingCompressor,
)

from .nas_compression_optimizer import (
    NASCompressionOptimizer,
    StackingPattern,
    PatternScore,
    TensorSignature,
    SynergyMatrix,
    MetaLearningCache,
)

from .f1_cascade_optimizer import (
    F1CascadeOptimizer,
    TelemetryPacket,
    PitStopRecommendation,
    QualifyingModeProfile,
    RaceModeProfile,
)

from ._nasacontrol import (
    NASAControlCompressor,
    MissionPhase,
    MissionReport,
)

from ._raptorcascade import (
    RaptorCascadeEngine,
    RaptorTelemetry,
    PreburnerConfig,
)

from .time_crystal_engine import (
    TimeCrystalCompressionEngine,
    FloquetOperator,
    TimeCrystalCycle,
    CrystalMethodState,
)

from .quantum_field_cascade import (
    QuantumFieldCascadeOptimizer,
    FeynmanVertex,
    MethodPropagator,
    ScatteringAmplitude,
)

from .plasma_confinement import (
    PlasmaConfinementTensorShaper,
    TokamakFieldLine,
    MagneticIsland,
    SafetyFactorProfile,
)

__all__ = [
    "ALL_TUNERS",
    "PHYSICS_TUNERS",
    "QUANTIZATION_TUNERS",
    "TunedParams",
    "list_tunable_methods",
    "tune_method",
    "ParetoFrontier",
    "ParetoPoint",
    "ProgressiveStreamingCompressor",
    "NASCompressionOptimizer",
    "StackingPattern",
    "PatternScore",
    "TensorSignature",
    "SynergyMatrix",
    "MetaLearningCache",
    "F1CascadeOptimizer",
    "TelemetryPacket",
    "PitStopRecommendation",
    "QualifyingModeProfile",
    "RaceModeProfile",
    "NASAControlCompressor",
    "MissionPhase",
    "MissionReport",
    "RaptorCascadeEngine",
    "RaptorTelemetry",
    "PreburnerConfig",
    "TimeCrystalCompressionEngine",
    "FloquetOperator",
    "TimeCrystalCycle",
    "CrystalMethodState",
    "QuantumFieldCascadeOptimizer",
    "FeynmanVertex",
    "MethodPropagator",
    "ScatteringAmplitude",
    "PlasmaConfinementTensorShaper",
    "TokamakFieldLine",
    "MagneticIsland",
    "SafetyFactorProfile",
]
from .quantization_tuning import *  # auto-split re-export
