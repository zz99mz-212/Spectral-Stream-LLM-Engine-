"""
Time Crystal Compression Engine — Floquet time-translation symmetry
breaking for perpetually discovering new compression opportunities.

Maps Floquet theory onto method cascades:
  - Time translation symmetry breaking → non-repeating method sequences
  - Floquet eigenstates → tensor subspaces at each cascade cycle
  - Period doubling cascade → Feigenbaum route to optimal compression
"""

from ._timecrystalcompressionengine import (
    TimeCrystalCompressionEngine,
    FloquetOperator,
    TimeCrystalCycle,
    CrystalMethodState,
)

__all__ = [
    "TimeCrystalCompressionEngine",
    "FloquetOperator",
    "TimeCrystalCycle",
    "CrystalMethodState",
]
