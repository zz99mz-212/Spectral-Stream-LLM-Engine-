from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from heapq import heappop, heappush, heapify
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

METHOD_NAME = "huffman_coding"

__all__ = ["HuffmanConfig", "_HuffmanNode", "METHOD_NAME"]


@dataclass
class HuffmanConfig:
    max_symbols: int = 256


class _HuffmanNode:
    def __init__(self, symbol: int, freq: int):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq
