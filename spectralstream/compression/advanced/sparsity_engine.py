from __future__ import annotations

# Auto-split: original sparsity_engine.py split into sparsity_engine/
from ._sparsityconfig import *
from ._pruningpattern import *
from ._pruningsignal import *
from ._pruningresult import *
from ._basepruner import *
from ._magnitudepruner import *
from ._wandapruner import *
from ._sparsegptpruner import *
from ._spectralpruner import *
from ._movementpruner import *
from ._combinedpruner import *
from ._vlasovpruner import *
from ._resonantpruner import *
from ._holographicpruner import *
from ._quantumpruner import *
from ._selforganizingpruner import *
from ._sparsepruner import *
from ._sparseformat import *
from ._dynamicsparseexecutor import *
from ._activationthreshold import *
from ._activationsparsity import *
from ._spectralbandconfig import *
from ._spectralsparsity import *
from ._tiledpattern import *
from ._structuredsparsity import *
from ._layersparsitystate import *
from ._adaptivesparsitymanager import *
from ._hdcsparsitypredictor import *
from ._vlasovsparsity import *
from ._resonantsparsity import *
from ._holographicsparsity import *
from ._quantumsparsity import *
from ._selforganizingsparsity import *
from ._unifiedsparsityengine import *

__all__ = ['_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', 'SparsityConfig', '_dense_from_csr', '_block_mask', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'PruningPattern', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'PruningSignal', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', 'PruningResult', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', 'BasePruner', '_csr_from_dense', '_circular_corr', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'MagnitudePruner', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', 'WandaPruner', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'SparseGPTPruner', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', 'SpectralPruner', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'MovementPruner', '_sparsity_ratio', 'CombinedPruner', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'VlasovPruner', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', 'ResonantPruner', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', 'HolographicPruner', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', 'QuantumPruner', '_csr_from_dense', '_circular_corr', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', 'SelfOrganizingPruner', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'SparsePruner', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'SparseFormat', '_sparsity_ratio', 'DynamicSparseExecutor', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', 'ActivationThreshold', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', 'ActivationSparsity', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', 'SpectralBandConfig', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', 'SpectralSparsity', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'TiledPattern', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', 'StructuredSparsity', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', 'LayerSparsityState', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', 'AdaptiveSparsityManager', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'HDCSparsityPredictor', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', 'VlasovSparsity', '_sparsity_ratio', '_energy_ratio', 'ResonantSparsity', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', 'HolographicSparsity', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', 'QuantumSparsity', '_block_mask', '_dense_from_csr', '_circular_conv', 'SelfOrganizingSparsity', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', '_block_mask', '_dense_from_csr', '_circular_conv', '_sparsity_ratio', '_energy_ratio', '_apply_nm_pattern', '_circular_corr', '_csr_from_dense', '_nm_mask', 'UnifiedSparsityEngine', '_block_mask', '_dense_from_csr', '_circular_conv']
