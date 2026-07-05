from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from spectralstream.core.math_primitives import (
    BAND_HIGH,
    BAND_LOW,
    BAND_NORMAL,
    DCTRotator,
    HadamardRotator,
    WaveletTransform,
    apply_spectral_kernel,
    band_limit,
    dct,
    fft,
    fftfreq,
    gibbs_softmax,
    idct,
    ifft,
    next_power_of_two,
    softmax,
    spectral_entropy,
    yukawa_kernel_1d,
)


class VlasovBlock:
    """A single block of tokens for tiled Vlasov attention."""
    indices: np.ndarray
    positions: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    local_phi: np.ndarray
    valid: np.ndarray

    @property
    def size(self) -> int:
        return len(self.indices)
