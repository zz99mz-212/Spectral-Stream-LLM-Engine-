"""Advanced compression sub-package — novel algorithms from v1 archive reintegration.

Modules are imported lazily to avoid circular import issues from the
sparsity_engine sub-package's cross-module references.

Modules:
    hadamard_preconditioner  — FWHT block-diagonal preconditioner
    rans_entropy             — rANS entropy coding
    turboquant_codec         — PolarQuant + QJL (4-bit signal + 1-bit residual)
    hyper_compression_v2     — FrequencyDomain, TT, VQ, TensorRing, Holographic, etc.
    quantum_tensor_net       — Quantum-inspired tensor network inference and compression
    sparsity_engine           — Advanced pruners (Resonant, Vlasov, Holographic, Self-Organizing)
    tt_pq_engine              — Tensor Train + Product Quantization pipeline
    advanced_sparsity         — SparseGPT, structured N:M, Hessian-based pruning
"""

from __future__ import annotations
