"""
Plasma Confinement Tensor Shaper — Tokamak-inspired tensor shaping
using poloidal and toroidal magnetic confinement principles.

Based on magnetohydrodynamic (MHD) confinement:
  - Tensor coefficients = plasma in a tokamak
  - Toroidal field → large-scale decomposition (poloidal mode number n=0)
  - Poloidal field → cross-sectional spectral shaping (m=1,2,...)
  - Magnetic islands = high-entropy regions needing compression
  - Safety factor q(r) = compression ratio as a function of radius
"""

from ._plasmaconfinement import (
    PlasmaConfinementTensorShaper,
    TokamakFieldLine,
    MagneticIsland,
    SafetyFactorProfile,
)

__all__ = [
    "PlasmaConfinementTensorShaper",
    "TokamakFieldLine",
    "MagneticIsland",
    "SafetyFactorProfile",
]
