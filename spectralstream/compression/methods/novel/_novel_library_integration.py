"""Wrap all novel_compression_library methods with standard (bytes, dict) interface.

All adapters are lazy — no classes instantiated at import time.
get_novel_library_methods() returns adapter references, not instances.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType
from typing import Any, Dict, Tuple

import numpy as np

from ._standalone_integration import _from_bytes, _to_bytes


def _import_novel_module(modname: str) -> ModuleType:
    """Import a novel_compression_library module with CompressionMethod pre-injected."""
    from spectralstream.compression.novel_compression_library._compressionmethod import (
        CompressionMethod,
        _ensure_2d,
        _restore_shape,
        _safe_bytes,
    )

    fullname = f"spectralstream.compression.novel_compression_library._{modname}"
    if fullname in sys.modules:
        return sys.modules[fullname]

    spec = importlib.util.find_spec(fullname)
    if spec is None:
        raise ImportError(f"Cannot find {fullname}")

    mod = importlib.util.module_from_spec(spec)
    mod.CompressionMethod = CompressionMethod
    mod._ensure_2d = _ensure_2d
    mod._restore_shape = _restore_shape
    mod._safe_bytes = _safe_bytes
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


# Map: module_name -> class_name for ALL 70 novel_compression_library modules
NOVEL_MODULE_CLASSES: Dict[str, str] = {
    "_ttsvd": "TTSVD",
    "_ttor": "TTOR",
    "_trsvd": "TRSVD",
    "_cpals": "CPALS",
    "_tuckersvd": "TuckerSVD",
    "_blocktucker": "BlockTucker",
    "_hierarchicaltucker": "HierarchicalTucker",
    "_tensornetwork": "TensorNetwork",
    "_butterflyfactorization": "ButterflyFactorization",
    "_kroneckerproduct": "KroneckerProduct",
    "_blockdiagonal": "BlockDiagonal",
    "_circulantapprox": "CirculantApprox",
    "_toeplitzapprox": "ToeplitzApprox",
    "_hankelapprox": "HankelApprox",
    "_lotr": "LoTR",
    "_lloydmaxquant": "LloydMaxQuant",
    "_adaptivescalarquant": "AdaptiveScalarQuant",
    "_productquantization": "ProductQuantization",
    "_residualvectorquant": "ResidualVectorQuant",
    "_additivecodebookquant": "AdditiveCodebookQuant",
    "_e8latticequant": "E8LatticeQuant",
    "_latticequantanchored": "LatticeQuantAnchored",
    "_mixedprecisionquant": "MixedPrecisionQuant",
    "_hessianawarequant": "HessianAwareQuant",
    "_fisherinfoquant": "FisherInfoQuant",
    "_gptqlayerquant": "GPTQLayerQuant",
    "_awqactivationaware": "AWQActivationAware",
    "_binaryquant": "BinaryQuant",
    "_ternaryquant": "TernaryQuant",
    "_nf4quant": "NF4Quant",
    "_dctspectral": "DCTSpectral",
    "_dct2dblock": "DCT2DBlock",
    "_waveletthreshold": "WaveletThreshold",
    "_fwhtcompress": "FWHTCompress",
    "_randomizedhadamard": "RandomizedHadamard",
    "_winogradtransform": "WinogradTransform",
    "_nttcompress": "NTTCompress",
    "_randomrotationquant": "RandomRotationQuant",
    "_butterflysparsetransform": "ButterflySparseTransform",
    "_sparserandomprojection": "SparseRandomProjection",
    "_structuredsparsity": "StructuredSparsity",
    "_blocksparsity": "BlockSparsity",
    "_unstructuredpruning": "UnstructuredPruning",
    "_sparsegpt": "SparseGPT",
    "_wandapruning": "WandaPruning",
    "_dynamicnmsparsity": "DynamicNMSparsity",
    "_channelpruning": "ChannelPruning",
    "_grouplasso": "GroupLasso",
    "_adaptivesparsityalloc": "AdaptiveSparsityAlloc",
    "_sparsequantizecombined": "SparseQuantizeCombined",
    "_huffmancoding": "HuffmanCoding",
    "_rans": "RANS",
    "_tans": "TANS",
    "_arithmeticcoding": "ArithmeticCoding",
    "_lz77entropy": "LZ77Entropy",
    "_vlasovmeanfield": "VlasovMeanField",
    "_holographicphaseencoding": "HolographicPhaseEncoding",
    "_quantumtensornetwork": "QuantumTensorNetwork",
    "_timecrystalphase": "TimeCrystalPhase",
    "_plasmafielddecomposition": "PlasmaFieldDecomposition",
    "_spectraldensityestimation": "SpectralDensityEstimation",
    "_informationbottleneck": "InformationBottleneck",
    "_ratedistortionoptimal": "RateDistortionOptimal",
    "_fisherraocompression": "FisherRaoCompression",
    "_symplecticweightevolution": "SymplecticWeightEvolution",
    "_landauzenersampling": "LandauZenerSampling",
    "_boltzmannencoding": "BoltzmannEncoding",
    "_maxentropycompression": "MaxEntropyCompression",
    "_crosslayerdelta": "CrossLayerDelta",
    "_hierarchicalclusteredpq": "HierarchicalClusteredPQ",
}


def _get_novel_class(modname: str, clsname: str) -> type:
    """Import and return a class from a novel_compression_library module."""
    mod = _import_novel_module(modname.lstrip("_"))
    cls = getattr(mod, clsname)
    return cls


# All unique method names — suffixed with _nl to avoid collisions with existing METHOD_CLASSES keys
NOVEL_LIBRARY_METHODS: Dict[str, Tuple[str, str, str]] = {
    # Decomposition (unique names)
    "cp_als_nl": ("decomposition", "_cpals", "CPALS"),
    "tucker_svd_nl": ("decomposition", "_tuckersvd", "TuckerSVD"),
    "block_tucker_nl": ("decomposition", "_blocktucker", "BlockTucker"),
    "htucker_nl": ("decomposition", "_hierarchicaltucker", "HierarchicalTucker"),
    "tensor_network_nl": ("decomposition", "_tensornetwork", "TensorNetwork"),
    "butterfly_factor_nl": (
        "decomposition",
        "_butterflyfactorization",
        "ButterflyFactorization",
    ),
    "kronecker_product_nl": ("decomposition", "_kroneckerproduct", "KroneckerProduct"),
    "block_diag_nl": ("decomposition", "_blockdiagonal", "BlockDiagonal"),
    "circulant_approx_nl": ("decomposition", "_circulantapprox", "CirculantApprox"),
    "toeplitz_approx_nl": ("decomposition", "_toeplitzapprox", "ToeplitzApprox"),
    "hankel_approx_nl": ("decomposition", "_hankelapprox", "HankelApprox"),
    "lotr_nl": ("decomposition", "_lotr", "LoTR"),
    "tt_svd_nl": ("decomposition", "_ttsvd", "TTSVD"),
    "tt_orth_nl": ("decomposition", "_ttor", "TTOR"),
    "tr_svd_nl": ("decomposition", "_trsvd", "TRSVD"),
    # Quantization (unique names)
    "lloyd_max_nl": ("quantization", "_lloydmaxquant", "LloydMaxQuant"),
    "adaptive_scalar_nl": (
        "quantization",
        "_adaptivescalarquant",
        "AdaptiveScalarQuant",
    ),
    "rvq_nl": ("quantization", "_residualvectorquant", "ResidualVectorQuant"),
    "additive_codebook_nl": (
        "quantization",
        "_additivecodebookquant",
        "AdditiveCodebookQuant",
    ),
    "hessian_aware_nl": ("quantization", "_hessianawarequant", "HessianAwareQuant"),
    "fisher_info_nl": ("quantization", "_fisherinfoquant", "FisherInfoQuant"),
    "hierarchical_pq_nl": (
        "quantization",
        "_hierarchicalclusteredpq",
        "HierarchicalClusteredPQ",
    ),
    "lattice_anchored_nl": (
        "quantization",
        "_latticequantanchored",
        "LatticeQuantAnchored",
    ),
    "e8_lattice_nl": ("quantization", "_e8latticequant", "E8LatticeQuant"),
    "mixed_precision_nl": (
        "quantization",
        "_mixedprecisionquant",
        "MixedPrecisionQuant",
    ),
    "gptq_layer_quant_nl": ("quantization", "_gptqlayerquant", "GPTQLayerQuant"),
    "awq_activation_nl": ("quantization", "_awqactivationaware", "AWQActivationAware"),
    "binary_quant_nl": ("quantization", "_binaryquant", "BinaryQuant"),
    "ternary_quant_nl": ("quantization", "_ternaryquant", "TernaryQuant"),
    "nf4_nl": ("quantization", "_nf4quant", "NF4Quant"),
    "product_quant_nl": ("quantization", "_productquantization", "ProductQuantization"),
    # Spectral (unique names)
    "ntt_nl": ("spectral", "_nttcompress", "NTTCompress"),
    "dct_spectral_nl": ("spectral", "_dctspectral", "DCTSpectral"),
    "dct_2d_block_nl": ("spectral", "_dct2dblock", "DCT2DBlock"),
    "wavelet_threshold_nl": ("spectral", "_waveletthreshold", "WaveletThreshold"),
    "fwht_nl": ("spectral", "_fwhtcompress", "FWHTCompress"),
    "random_hadamard_nl": ("spectral", "_randomizedhadamard", "RandomizedHadamard"),
    "winograd_nl": ("spectral", "_winogradtransform", "WinogradTransform"),
    "random_rot_quant_nl": ("spectral", "_randomrotationquant", "RandomRotationQuant"),
    "butterfly_sparse_nl": (
        "spectral",
        "_butterflysparsetransform",
        "ButterflySparseTransform",
    ),
    "sparse_random_proj_nl": (
        "spectral",
        "_sparserandomprojection",
        "SparseRandomProjection",
    ),
    # Structural (unique names)
    "structured_sparsity_nl": (
        "structural",
        "_structuredsparsity",
        "StructuredSparsity",
    ),
    "block_sparsity_nl": ("structural", "_blocksparsity", "BlockSparsity"),
    "unstruct_pruning_nl": (
        "structural",
        "_unstructuredpruning",
        "UnstructuredPruning",
    ),
    "sparse_gpt_nl": ("structural", "_sparsegpt", "SparseGPT"),
    "wanda_pruning_nl": ("structural", "_wandapruning", "WandaPruning"),
    "dynamic_nm_nl": ("structural", "_dynamicnmsparsity", "DynamicNMSparsity"),
    "channel_pruning_nl": ("structural", "_channelpruning", "ChannelPruning"),
    "group_lasso_nl": ("structural", "_grouplasso", "GroupLasso"),
    "adaptive_sparsity_nl": (
        "structural",
        "_adaptivesparsityalloc",
        "AdaptiveSparsityAlloc",
    ),
    "sparse_quant_nl": (
        "structural",
        "_sparsequantizecombined",
        "SparseQuantizeCombined",
    ),
    # Entropy (unique names)
    "huffman_nl": ("entropy", "_huffmancoding", "HuffmanCoding"),
    "rans_nl": ("entropy", "_rans", "RANS"),
    "tans_nl": ("entropy", "_tans", "TANS"),
    "arithmetic_nl": ("entropy", "_arithmeticcoding", "ArithmeticCoding"),
    "lz77_entropy_nl": ("entropy", "_lz77entropy", "LZ77Entropy"),
    # Novel / Physics (unique names)
    "vlasov_mf_nl": ("physics", "_vlasovmeanfield", "VlasovMeanField"),
    "holographic_phase_nl": (
        "novel",
        "_holographicphaseencoding",
        "HolographicPhaseEncoding",
    ),
    "qtensor_net_nl": ("physics", "_quantumtensornetwork", "QuantumTensorNetwork"),
    "timecrystal_nl": ("novel", "_timecrystalphase", "TimeCrystalPhase"),
    "plasma_field_nl": (
        "physics",
        "_plasmafielddecomposition",
        "PlasmaFieldDecomposition",
    ),
    "spectral_density_nl": (
        "novel",
        "_spectraldensityestimation",
        "SpectralDensityEstimation",
    ),
    "fisher_rao_nl": ("novel", "_fisherraocompression", "FisherRaoCompression"),
    "symplectic_nl": (
        "novel",
        "_symplecticweightevolution",
        "SymplecticWeightEvolution",
    ),
    # Functional / Information (unique names)
    "info_bottleneck_nl": (
        "functional",
        "_informationbottleneck",
        "InformationBottleneck",
    ),
    "rd_optimal_nl": ("functional", "_ratedistortionoptimal", "RateDistortionOptimal"),
    "landau_zener_nl": ("functional", "_landauzenersampling", "LandauZenerSampling"),
    "boltzmann_nl": ("functional", "_boltzmannencoding", "BoltzmannEncoding"),
    "max_entropy_nl": ("functional", "_maxentropycompression", "MaxEntropyCompression"),
    "cross_layer_delta_nl": ("novel", "_crosslayerdelta", "CrossLayerDelta"),
}


class _NovelLibraryAdapter:
    """Lazy adapter wrapping a novel_compression_library method to (bytes, dict) interface.

    The wrapped class is instantiated only when compress() or decompress() is called.
    """

    def __init__(self, name: str, category: str, modname: str, clsname: str) -> None:
        self.name = name
        self.category = category
        self._modname = modname
        self._clsname = clsname

    def _get_instance(self):
        from spectralstream.compression.novel_compression_library._compressionmethod import (
            CompressionMethod,
        )

        mod = _import_novel_module(self._modname.lstrip("_"))
        cls = getattr(mod, self._clsname)
        return cls()

    def compress(self, tensor: np.ndarray, **kw: Any) -> Tuple[bytes, dict]:
        inst = self._get_instance()
        result, meta = inst.compress(tensor, **kw)
        d = result if isinstance(result, dict) else {"data": result}
        data = _to_bytes(d)
        if "original_shape" not in meta:
            meta["original_shape"] = tensor.shape
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        inst = self._get_instance()
        obj = _from_bytes(data)
        result = inst.decompress(obj, metadata)
        shape = metadata.get("original_shape") or metadata.get("orig_shape")
        if shape is not None:
            result = np.asarray(result).reshape(shape)
        return result.astype(np.float32)


def get_novel_library_methods() -> Dict[str, Tuple[str, Any]]:
    """Return dict of (name -> (category, lazy adapter)) for all novel_library methods.

    No instantiation or testing — memory-safe for registration.
    All adapters are lazily created wrapper references.
    """
    result: Dict[str, Tuple[str, Any]] = {}
    for name, (cat, modname, clsname) in NOVEL_LIBRARY_METHODS.items():
        adapter = _NovelLibraryAdapter(name, cat, modname, clsname)
        result[name] = (cat, adapter)
    return result
