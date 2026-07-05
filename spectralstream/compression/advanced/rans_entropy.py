"""
rANS Entropy Coding for SpectralStream
========================================
Asymmetric Numeral Systems (rANS) entropy coding for compression
of quantized weight indices and sparse structure.

Implements:
  - RANSEncoder / RANSDecoder — batch rANS with arbitrary precision
  - AdaptiveRANS — online frequency adaptation
  - EntropyAnalyzer — Shannon entropy and distribution detection
"""

from __future__ import annotations


import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RANSConfig:
    """Configuration for rANS entropy coding."""

    precision: int = 32
    batch_size: int = 4096
    max_symbols: int = 256
    adaptive_window: int = 4096
    min_frequency: int = 1


# ═══════════════════════════════════════════════════════════════════════════
# Entropy Analysis
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class EntropyResult:
    """Results of entropy analysis."""

    shannon_entropy: float
    bits_per_symbol: float
    distribution: str
    max_symbol: int
    min_symbol: int
    n_symbols: int
    frequency_table: np.ndarray
    cumulative_freq: np.ndarray
    is_uniform: bool


class EntropyAnalyzer:
    """Analyze symbol distributions for entropy coding.

    Computes Shannon entropy, detects distribution type, and builds
    frequency tables for rANS encoding.
    """

    def __init__(self, max_symbols: int = 256):
        self.max_symbols = max_symbols

    def analyze(self, symbols: np.ndarray) -> EntropyResult:
        """Analyze a sequence of symbols.

        Args:
            symbols: Integer symbol array.

        Returns:
            EntropyResult with analysis results.
        """
        symbols = np.asarray(symbols).ravel().astype(np.int64)
        n_symbols = len(symbols)
        if n_symbols == 0:
            return EntropyResult(
                shannon_entropy=0.0,
                bits_per_symbol=0.0,
                distribution="empty",
                max_symbol=0,
                min_symbol=0,
                n_symbols=0,
                frequency_table=np.zeros(1),
                cumulative_freq=np.zeros(1),
                is_uniform=True,
            )

        min_sym = int(np.min(symbols))
        max_sym = int(np.max(symbols))
        n_unique = max_sym - min_sym + 1
        n_bins = min(max(n_unique, 2), self.max_symbols)

        # Shift symbols to 0-based
        shifted = symbols - min_sym

        # Frequency table
        freq = np.bincount(shifted, minlength=n_bins).astype(np.float64)
        freq = np.maximum(freq, 1.0)  # Laplace smoothing

        # Probability distribution
        probs = freq / np.sum(freq)

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-30))
        bits_per_symbol = float(entropy)

        # Distribution detection
        dist_type = self._detect_distribution(probs)

        # Uniformity check
        uniform_expected = 1.0 / n_bins
        is_uniform = np.all(np.abs(probs - uniform_expected) < 0.05)

        # Cumulative frequencies for rANS
        cum_freq = np.cumsum(freq)
        cum_freq = np.concatenate([[0], cum_freq])

        return EntropyResult(
            shannon_entropy=float(entropy),
            bits_per_symbol=bits_per_symbol,
            distribution=dist_type,
            max_symbol=max_sym,
            min_symbol=min_sym,
            n_symbols=n_bins,
            frequency_table=freq,
            cumulative_freq=cum_freq,
            is_uniform=is_uniform,
        )

    def _detect_distribution(self, probs: np.ndarray) -> str:
        """Detect the type of probability distribution."""
        n = len(probs)
        entropy = -np.sum(probs * np.log2(probs + 1e-30))
        max_entropy = np.log2(n) if n > 1 else 1.0

        if max_entropy < 1e-10:
            return "degenerate"

        normalized_entropy = entropy / max_entropy

        if normalized_entropy > 0.95:
            return "uniform"
        elif normalized_entropy < 0.3:
            return "highly_skewed"
        elif np.max(probs) > 0.5:
            return "peaked"
        else:
            return "moderate"


# ═══════════════════════════════════════════════════════════════════════════
# rANS Encoder / Decoder
# ═══════════════════════════════════════════════════════════════════════════


class RANSEncoder:
    """Batch rANS (range Asymmetric Numeral Systems) encoder.

    Encodes symbols using cumulative frequency tables with
    arbitrary precision arithmetic.

    rANS state update:
        x_{n+1} = floor(x_n / f(s)) * M + cum(s) + (x_n mod f(s))

    where f(s) is the frequency of symbol s, cum(s) is its
    cumulative frequency, and M is the total frequency range.
    """

    def __init__(self, config: Optional[RANSConfig] = None):
        self.config = config or RANSConfig()
        self._precision = self.config.precision
        self._lower_bound = 1 << (self._precision - 1)
        self._upper_bound = (1 << self._precision) - 1

    def encode(
        self,
        symbols: np.ndarray,
        frequencies: np.ndarray,
        cumulative: np.ndarray,
    ) -> np.ndarray:
        """Encode a batch of symbols using rANS.

        Args:
            symbols: Symbol array to encode.
            frequencies: Frequency table (must sum to power of 2).
            cumulative: Cumulative frequency table.

        Returns:
            Encoded state array.
        """
        symbols = np.asarray(symbols, dtype=np.int64).ravel()
        frequencies = np.asarray(frequencies, dtype=np.int64)
        cumulative = np.asarray(cumulative, dtype=np.int64)

        n = len(symbols)
        total = int(cumulative[-1])

        # Ensure total is power of 2
        if total & (total - 1) != 0:
            total = 1 << (total - 1).bit_length()
            frequencies = frequencies.copy()
            frequencies[-1] += total - int(cumulative[-1])

        states = np.full(n, self._lower_bound, dtype=np.int64)
        bitstream: List[np.ndarray] = []

        for i in range(n):
            s = int(symbols[i])
            s = max(0, min(s, len(frequencies) - 1))

            freq_s = max(int(frequencies[s]), 1)
            cum_s = int(cumulative[s])

            # Normalize
            while states[i] < freq_s * (total >> self._precision):
                bitstream.append(states[i] & 1)
                states[i] >>= 1

            # rANS update
            states[i] = (states[i] // freq_s) * total + cum_s + (states[i] % freq_s)

        return states

    def decode_batch(
        self,
        states: np.ndarray,
        frequencies: np.ndarray,
        cumulative: np.ndarray,
        n_symbols: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode a batch of symbols from rANS states.

        Args:
            states: Current rANS states.
            frequencies: Frequency table.
            cumulative: Cumulative frequency table.
            n_symbols: Number of symbols to decode per state.

        Returns:
            (decoded_symbols, new_states)
        """
        n_states = len(states)
        frequencies = np.asarray(frequencies, dtype=np.int64)
        cumulative = np.asarray(cumulative, dtype=np.int64)

        total = int(cumulative[-1])
        if total & (total - 1) != 0:
            total = 1 << (total - 1).bit_length()

        symbols = np.zeros((n_states, n_symbols), dtype=np.int64)
        current_states = states.copy()

        for _ in range(n_symbols):
            # Find symbol
            slot = current_states % total
            s = np.searchsorted(cumulative, slot, side="right") - 1
            s = np.clip(s, 0, len(frequencies) - 1)
            symbols[:, _] = s

            # rANS update (decode)
            freq_s = np.maximum(frequencies[s], 1)
            cum_s = cumulative[s]
            current_states = (current_states // freq_s) * freq_s + slot - cum_s

            # Renormalize
            while np.any(current_states < self._lower_bound):
                low_mask = current_states < self._lower_bound
                current_states = np.where(low_mask, current_states * 2, current_states)

        return symbols, current_states


class RANSDecoder:
    """rANS decoder with state management."""

    def __init__(self, config: Optional[RANSConfig] = None):
        self.config = config or RANSConfig()
        self._precision = self.config.precision
        self._encoder = RANSEncoder(config)

    def decode(
        self,
        states: np.ndarray,
        frequencies: np.ndarray,
        cumulative: np.ndarray,
        n_symbols: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode symbols from rANS states.

        Args:
            states: Encoded states.
            frequencies: Frequency table.
            cumulative: Cumulative frequency table.
            n_symbols: Number of symbols to decode.

        Returns:
            (decoded_symbols, updated_states)
        """
        return self._encoder.decode_batch(states, frequencies, cumulative, n_symbols)


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive rANS
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveRANS:
    """Adaptive rANS with online frequency table updates.

    Maintains a running frequency table that adapts to the input
    distribution, achieving near-optimal compression without
    requiring a pre-computed frequency table.
    """

    def __init__(self, config: Optional[RANSConfig] = None):
        self.config = config or RANSConfig()
        self._n_symbols = self.config.max_symbols
        self._frequencies = np.ones(self._n_symbols, dtype=np.int64)
        self._cumulative = np.cumsum(self._frequencies)
        self._cumulative = np.concatenate([[0], self._cumulative])
        self._total = int(self._cumulative[-1])
        self._window: List[int] = []
        self._window_size = self.config.adaptive_window
        self._encoder = RANSEncoder(config)
        self._decoder = RANSDecoder(config)
        self._analyzer = EntropyAnalyzer(self._n_symbols)

    def _update_frequencies(self, symbol: int) -> None:
        """Update frequency table with new symbol (exponential decay)."""
        symbol = max(0, min(symbol, self._n_symbols - 1))
        self._window.append(symbol)
        if len(self._window) > self._window_size:
            self._window.pop(0)

        # Rebuild frequency table from window
        self._frequencies = np.ones(self._n_symbols, dtype=np.int64)
        for s in self._window:
            self._frequencies[s] += 1

        self._cumulative = np.cumsum(self._frequencies)
        self._cumulative = np.concatenate([[0], self._cumulative])
        self._total = int(self._cumulative[-1])

    def encode_stream(self, symbols: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Encode symbols with online adaptation.

        Args:
            symbols: Stream of symbols to encode.

        Returns:
            (encoded_states, metadata)
        """
        symbols = np.asarray(symbols, dtype=np.int64).ravel()
        n = len(symbols)

        states = np.full(n, 1 << (self._precision - 1), dtype=np.int64)
        self._window.clear()

        for i in range(n):
            s = int(symbols[i])
            s = max(0, min(s, self._n_symbols - 1))

            freq_s = max(int(self._frequencies[s]), 1)
            cum_s = int(self._cumulative[s])

            # Normalize
            while states[i] < freq_s * (self._total >> self._precision):
                states[i] >>= 1

            # rANS update
            states[i] = (
                (states[i] // freq_s) * self._total + cum_s + (states[i] % freq_s)
            )

            self._update_frequencies(s)

        entropy_result = self._analyzer.analyze(symbols)
        metadata = {
            "n_symbols": n,
            "final_entropy": entropy_result.shannon_entropy,
            "bits_per_symbol": entropy_result.bits_per_symbol,
            "distribution": entropy_result.distribution,
        }

        return states, metadata

    def decode_stream(
        self,
        states: np.ndarray,
        n_symbols: int,
    ) -> Tuple[np.ndarray, Dict]:
        """Decode symbols with online adaptation.

        Args:
            states: Encoded states.
            n_symbols: Number of symbols to decode.

        Returns:
            (decoded_symbols, metadata)
        """
        symbols, new_states = self._decoder.decode(
            states, self._frequencies, self._cumulative, n_symbols
        )

        # Flatten since decode returns (n_states, n_symbols_per_state)
        decoded = symbols.ravel()[:n_symbols]

        metadata = {
            "n_decoded": len(decoded),
            "unique_symbols": int(np.unique(decoded).size),
        }

        return decoded, metadata

    def get_compression_ratio(self, symbols: np.ndarray) -> float:
        """Estimate compression ratio for the given symbols."""
        entropy_result = self._analyzer.analyze(symbols)
        original_bits = len(symbols) * 8  # assume 8-bit symbols
        compressed_bits = len(symbols) * entropy_result.bits_per_symbol
        return original_bits / max(compressed_bits, 1)

    def get_frequency_table(self) -> np.ndarray:
        """Get current frequency table."""
        return self._frequencies.copy()

    def reset(self) -> None:
        """Reset adaptive state."""
        self._frequencies = np.ones(self._n_symbols, dtype=np.int64)
        self._cumulative = np.cumsum(self._frequencies)
        self._cumulative = np.concatenate([[0], self._cumulative])
        self._total = int(self._cumulative[-1])
        self._window.clear()


__all__ = [
    "RANSConfig",
    "EntropyResult",
    "EntropyAnalyzer",
    "RANSEncoder",
    "RANSDecoder",
    "AdaptiveRANS",
    "_build_cumulative",
]


def _build_cumulative(
    freq: np.ndarray, precision: int = 12
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build cumulative frequency table for rANS coding.

    Args:
        freq: Frequency counts (1D array).
        precision: Scaling precision in bits.

    Returns:
        (cumulative, scaled_freq, total) tuple.
    """
    total = int(np.sum(freq))
    if total == 0:
        freq = np.ones_like(freq)
        total = int(np.sum(freq))
    scale = (1 << precision) // total
    scaled = np.maximum(freq * scale, 1)
    scaled[0] += (1 << precision) - np.sum(scaled)
    cumulative = np.concatenate([[0], np.cumsum(scaled)[:-1]])
    return cumulative.astype(np.int64), scaled.astype(np.int64), 1 << precision
