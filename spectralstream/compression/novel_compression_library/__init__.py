from ._compressionmethod import (
    CompressionMethod,
    _ensure_2d,
    _restore_shape,
    _safe_bytes,
)
from ._ttsvd import TTSVD
from ._ttor import TTOR
from ._trsvd import TRSVD
from ._cpals import CPALS
from ._tuckersvd import TuckerSVD
from ._blocktucker import BlockTucker
from ._hierarchicaltucker import HierarchicalTucker
from ._tensornetwork import TensorNetwork
from ._butterflyfactorization import ButterflyFactorization
from ._kroneckerproduct import KroneckerProduct
from ._blockdiagonal import BlockDiagonal
from ._circulantapprox import CirculantApprox
from ._toeplitzapprox import ToeplitzApprox
from ._hankelapprox import HankelApprox
from ._lotr import LoTR
from ._lloydmaxquant import LloydMaxQuant
from ._adaptivescalarquant import AdaptiveScalarQuant
from ._productquantization import ProductQuantization
from ._residualvectorquant import ResidualVectorQuant
from ._additivecodebookquant import AdditiveCodebookQuant
from ._e8latticequant import E8LatticeQuant
from ._latticequantanchored import LatticeQuantAnchored
from ._mixedprecisionquant import MixedPrecisionQuant
from ._hessianawarequant import HessianAwareQuant
from ._fisherinfoquant import FisherInfoQuant
from ._gptqlayerquant import GPTQLayerQuant
from ._awqactivationaware import AWQActivationAware
from ._binaryquant import BinaryQuant
from ._ternaryquant import TernaryQuant
from ._nf4quant import NF4Quant
from ._dctspectral import DCTSpectral
from ._dct2dblock import DCT2DBlock
from ._waveletthreshold import WaveletThreshold
from ._fwhtcompress import FWHTCompress
from ._randomizedhadamard import RandomizedHadamard
from ._winogradtransform import WinogradTransform
from ._nttcompress import NTTCompress
from ._randomrotationquant import RandomRotationQuant
from ._butterflysparsetransform import ButterflySparseTransform
from ._sparserandomprojection import SparseRandomProjection
from ._structuredsparsity import StructuredSparsity
from ._blocksparsity import BlockSparsity
from ._unstructuredpruning import UnstructuredPruning
from ._sparsegpt import SparseGPT
from ._wandapruning import WandaPruning
from ._dynamicnmsparsity import DynamicNMSparsity
from ._channelpruning import ChannelPruning
from ._grouplasso import GroupLasso
from ._adaptivesparsityalloc import AdaptiveSparsityAlloc
from ._sparsequantizecombined import SparseQuantizeCombined
from ._huffmancoding import HuffmanCoding
from ._rans import RANS
from ._tans import TANS
from ._arithmeticcoding import ArithmeticCoding
from ._lz77entropy import LZ77Entropy
from ._vlasovmeanfield import VlasovMeanField
from ._holographicphaseencoding import HolographicPhaseEncoding
from ._quantumtensornetwork import QuantumTensorNetwork
from ._timecrystalphase import TimeCrystalPhase
from ._plasmafielddecomposition import PlasmaFieldDecomposition
from ._spectraldensityestimation import SpectralDensityEstimation
from ._informationbottleneck import InformationBottleneck
from ._ratedistortionoptimal import RateDistortionOptimal
from ._fisherraocompression import FisherRaoCompression
from ._symplecticweightevolution import SymplecticWeightEvolution
from ._landauzenersampling import LandauZenerSampling
from ._boltzmannencoding import BoltzmannEncoding
from ._maxentropycompression import MaxEntropyCompression
from ._crosslayerdelta import CrossLayerDelta
from ._hierarchicalclusteredpq import HierarchicalClusteredPQ
from ._benchmarkresult import BenchmarkResult
from ._compressionmethodbenchmark import CompressionMethodBenchmark

# Category metadata for registry integration
METHOD_METADATA = {
    "tt_svd": "decomposition",
    "tt_orth": "decomposition",
    "tr_svd": "decomposition",
    "cp_als": "decomposition",
    "tucker_svd": "decomposition",
    "block_tucker": "decomposition",
    "htucker": "decomposition",
    "tensor_network": "decomposition",
    "butterfly_factor": "decomposition",
    "kronecker_product": "decomposition",
    "block_diag": "decomposition",
    "circulant_approx": "decomposition",
    "toeplitz_approx": "decomposition",
    "hankel_approx": "decomposition",
    "lotr": "decomposition",
    "lloyd_max": "quantization",
    "adaptive_scalar": "quantization",
    "product_quant": "quantization",
    "rvq": "quantization",
    "additive_codebook": "quantization",
    "e8_lattice": "quantization",
    "lattice_anchored": "quantization",
    "mixed_precision": "quantization",
    "hessian_aware": "quantization",
    "fisher_info": "quantization",
    "gptq_layer": "quantization",
    "awq_activation": "quantization",
    "binary_quant": "quantization",
    "ternary_quant": "quantization",
    "nf4": "quantization",
    "dct_spectral": "spectral",
    "dct_2d_block": "spectral",
    "wavelet_threshold": "spectral",
    "fwht": "spectral",
    "random_hadamard": "spectral",
    "winograd": "spectral",
    "ntt": "spectral",
    "random_rot_quant": "spectral",
    "butterfly_sparse": "spectral",
    "sparse_random_proj": "spectral",
    "structured_2_4": "structural",
    "block_sparsity": "structural",
    "unstruct_pruning": "structural",
    "sparse_gpt": "structural",
    "wanda": "structural",
    "dynamic_nm": "structural",
    "channel_pruning": "structural",
    "group_lasso": "structural",
    "adaptive_sparsity": "structural",
    "sparse_quant": "structural",
    "huffman": "entropy",
    "rans": "entropy",
    "tans": "entropy",
    "arithmetic": "entropy",
    "lz77": "entropy",
    "vlasov_mean_field": "physics",
    "holographic_phase": "novel",
    "qtensor_net": "physics",
    "timecrystal": "novel",
    "plasma_field": "physics",
    "spectral_density": "novel",
    "info_bottleneck": "functional",
    "rd_optimal": "functional",
    "fisher_rao": "novel",
    "symplectic": "novel",
    "landau_zener": "functional",
    "boltzmann": "functional",
    "max_entropy": "functional",
    "cross_layer_delta": "novel",
    "hierarchical_pq": "quantization",
}
