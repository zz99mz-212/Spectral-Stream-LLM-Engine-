from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralstream.core.math_primitives import (
    HadamardRotator,
    LloydMaxQuantizer,
    dct,
    fwht,
    idct,
    next_power_of_two,
    softmax,
    spectral_entropy,
)


@dataclass
class PQConfig:
    """Configuration for Product Quantization."""

    n_subspaces: int = 8
    n_centroids: int = 256
    n_clusters_per_subspace: int = 256
    subspace_dim: int = 4
    codebook_bits: int = 8
    n_bits: int = 8
    n_iter: int = 20
    lloyd_max_iterations: int = 20
