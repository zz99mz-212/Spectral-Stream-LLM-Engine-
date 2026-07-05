from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "tucker_decomposition"

__all__ = ["TuckerConfig", "METHOD_NAME"]


@dataclass
class TuckerConfig:
    rank: int = 8
    energy_threshold: float = 0.99
