"""
Sparsity / Pruning Methods
=============================
Weight pruning and structured sparsity techniques.
"""

from __future__ import annotations

from spectralstream.compression.methods.sparsity.structured_nm import StructuredNM
from spectralstream.compression.methods.sparsity.block_sparsity import BlockSparsityS
from spectralstream.compression.methods.sparsity.unstructured_mag import UnstructuredMag
from spectralstream.compression.methods.sparsity.sparsegpt import SparseGPTS
from spectralstream.compression.methods.sparsity.wanda import WandaS
from spectralstream.compression.methods.sparsity.group_lasso import GroupLasso
from spectralstream.compression.methods.sparsity.channel_prune import ChannelPrune
from spectralstream.compression.methods.sparsity.dynamic_nm import DynamicNM
from spectralstream.compression.methods.sparsity.n_m_tiled import NMTiled
from spectralstream.compression.methods.sparsity.magnitude_prune import MagnitudePrune
from spectralstream.compression.methods.sparsity.sparse_quantize import SparseQuantize

__all__ = [
    "StructuredNM",
    "BlockSparsityS",
    "UnstructuredMag",
    "SparseGPTS",
    "WandaS",
    "GroupLasso",
    "ChannelPrune",
    "DynamicNM",
    "NMTiled",
    "MagnitudePrune",
    "SparseQuantize",
]
