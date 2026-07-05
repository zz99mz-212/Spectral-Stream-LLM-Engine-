"""
Common imports for the compression package.

Provides a single point of import for widely used modules across the
compression package. This reduces duplication and ensures consistent
import patterns.

Usage:
    from spectralstream.compression._imports import (
        np, logger, dataclass, field, Path, time, os, json, math, struct,
        Any, Dict, List, Optional, Tuple, Callable, Iterator, Sequence,
    )
"""

from __future__ import annotations


import json
import logging
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np


T = TypeVar("T")


__all__ = [
    "np",
    "logger",
    "dataclass",
    "field",
    "Path",
    "time",
    "os",
    "json",
    "math",
    "struct",
    "Any",
    "Dict",
    "List",
    "Optional",
    "Tuple",
    "Callable",
    "Iterator",
    "Sequence",
]


def _get_logger(name: str) -> logging.Logger:
    """Get a logger for the calling module."""
    return logging.getLogger(name)
