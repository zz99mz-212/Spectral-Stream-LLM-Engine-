from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "noise_floor"

__all__ = ["NoiseFloorConfig", "METHOD_NAME"]


@dataclass
class NoiseFloorConfig:
    method: str = "marchenko_pastur"
    energy_threshold: float = 0.95
    damping: float = 1e-4
