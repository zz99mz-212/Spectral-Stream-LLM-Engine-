from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "nf4_quant"

__all__ = ["NF4Config", "METHOD_NAME"]


@dataclass
class NF4Config:
    block_size: int = 64
    double_quant: bool = True
