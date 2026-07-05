from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "arithmetic_coding"

__all__ = ["ArithmeticCodingConfig", "METHOD_NAME"]


@dataclass
class ArithmeticCodingConfig:
    precision: int = 64
