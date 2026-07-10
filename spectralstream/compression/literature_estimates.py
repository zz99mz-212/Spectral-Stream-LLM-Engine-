from __future__ import annotations

"""
LITERATURE ESTIMATES, NOT MEASURED HERE

This module contains published compression-ratio values for alternative methods
quoted from the literature. These numbers are NOT measured by SpectralStream and
are provided as a reference context only. The current run's ratio is computed
separately in certificate.py from actual weights.

All external competitor references MUST be labeled with the disclaimer below to
prevent fabrication / misattribution of unmeasured claims.
"""

from typing import List, Tuple

LITERATURE_DISCLAIMER = "literature estimates, not measured here"

# 9 competitor tuples: (name, ratio, description, type)
# These are verbatim literature quotes, NOT measured by SpectralStream.
LITERATURE_ESTIMATES: List[Tuple[str, float, str, str]] = [
    ("FP16 (baseline)", 2.0, "2x storage savings", "lossless"),
    ("INT8 quantization", 4.0, "Standard 8-bit quantization", "lossy"),
    ("INT4 quantization", 8.0, "Standard 4-bit quantization", "lossy"),
    ("NF4 (QLoRA)", 4.0, "Normal float 4, QLoRA standard", "lossy"),
    ("GPTQ 4-bit", 8.0, "Post-training quantization", "lossy"),
    ("AWQ 4-bit", 8.0, "Activation-aware quantization", "lossy"),
    ("GGML Q4_0", 4.5, "llama.cpp Q4_0 quantization", "lossy"),
    ("GGML Q8_0", 2.5, "llama.cpp Q8_0 quantization", "lossy"),
    ("SqueezeLLM", 8.0, "Non-uniform quantization", "lossy"),
]

__all__ = ["LITERATURE_DISCLAIMER", "LITERATURE_ESTIMATES"]
