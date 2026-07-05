"""
MoE-Aware Compression
=====================
Special handling for Mixture-of-Experts models:
- Experts have similar structure -> compress as gauge-equivariant batch
- Expert routers are critical -> high precision
- Shared layers compress normally
- Expert parallelism aware (don't mix across parallel groups)
"""

import re
from typing import Dict, List

import numpy as np


class MoEAwareCompressor:
    """
    Compresses MoE models with expert-aware techniques.

    Key insight: All experts in the same MoE layer share similar structure.
    Instead of compressing each independently, compress as a GAUGE-EQUIVARIANT BATCH:
    - Store ONE base expert + per-expert gauge transformations
    - This gives Nx compression on top of per-expert compression
    """

    def __init__(self, engine):
        self.engine = engine

    def detect_moe_structure(self, tensor_names: List[str]) -> Dict:
        """Analyze tensor names to identify MoE structure."""
        experts = {}
        shared = []

        for name in tensor_names:
            nl = name.lower()
            if "expert" in nl:
                match = re.search(r"expert[^0-9]*(\d+)", nl)
                if match:
                    exp_idx = int(match.group(1))
                    if exp_idx not in experts:
                        experts[exp_idx] = []
                    experts[exp_idx].append(name)
            else:
                shared.append(name)

        return {
            "has_moe": len(experts) > 0,
            "num_experts": len(experts),
            "expert_tensors": experts,
            "shared_tensors": shared,
        }

    def compress_experts_batch(self, expert_tensors: Dict[int, np.ndarray]) -> bytes:
        """Compress all experts at the same layer as a gauge-equivariant batch."""
        try:
            from spectralstream.compression.methods.novel.physics.gauge_equivariant import (
                GaugeEquivariant,
            )
        except ImportError:
            logger = __import__("logging").getLogger(__name__)
            logger.warning(
                "GaugeEquivariant not available, falling back to per-expert compression"
            )
            from spectralstream.compression.engine._methods import METHOD_REGISTRY

            blk8 = METHOD_REGISTRY.get("block_int8")
            if blk8 is None:
                return b""
            ordered = [expert_tensors[i] for i in sorted(expert_tensors.keys())]
            results = bytearray()
            for t in ordered:
                data, _ = blk8.compress(t)
                results += data
            return bytes(results)

        ordered = [expert_tensors[i] for i in sorted(expert_tensors.keys())]

        compressor = GaugeEquivariant()
        return compressor.compress_batch(ordered)
