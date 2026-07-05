"""
Cascade configuration presets.

Defines compression cascade configurations with maximum stage counts
and expected compression ratios for each preset.
"""

from typing import Dict, Any

CASCADE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "lightning": {
        "max_stages": 1,
        "expected_ratio": 5,
        "description": "Single-stage DCT spectral compression for quick results",
    },
    "balanced": {
        "max_stages": 2,
        "expected_ratio": 50,
        "description": "SVD decomposition only — good ratio/quality trade-off",
    },
    "aggressive": {
        "max_stages": 2,
        "expected_ratio": 100,
        "description": "SVD + DCT cascade for higher compression with moderate quality",
    },
    "extreme": {
        "max_stages": 3,
        "expected_ratio": 200,
        "description": "Multi-stage cascade for maximum compression ratio",
    },
}
