"""
Quantum Field Theory Cascade Optimizer — Feynman diagram-inspired
method sequence optimization via QFT scattering amplitudes.

Models method selection as a quantum field theory:
  - Methods = particles in the QFT with creation/annihilation operators
  - Coupling constants = inter-method synergy/anti-synergy
  - Feynman diagrams = method interaction vertices
  - Path integrals = integrals over all possible method sequences
"""

from ._quantumfieldcascade import (
    QuantumFieldCascadeOptimizer,
    FeynmanVertex,
    MethodPropagator,
    ScatteringAmplitude,
)

__all__ = [
    "QuantumFieldCascadeOptimizer",
    "FeynmanVertex",
    "MethodPropagator",
    "ScatteringAmplitude",
]
