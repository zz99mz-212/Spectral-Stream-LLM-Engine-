from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "cp_als"

__all__ = ["CPConfig", "METHOD_NAME"]


@dataclass
class CPConfig:
    rank: int = 8
    n_iter: int = 20
    tol: float = 1e-6
