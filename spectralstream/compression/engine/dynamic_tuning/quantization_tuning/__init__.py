"""Quantization tuning package — re-exports from split sub-modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

FP32_BYTES = 4


@dataclass
class TunedParams:
    method_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    estimated_ratio: float = 1.0
    estimated_mse: float = 0.0
    estimated_error: float = 0.0


# All tuner functions gathered from sub-modules
from .tunenf4 import *
from .svdratio import *
from .tuneplasmafield import *
from .tunequantumstate import *

_SAFE = {}


def _lazy(name: str, fallback: Callable) -> Callable:
    """Resolve tuner by name, falling back to *fallback* if not imported."""
    return _SAFE.get(name) if name in _SAFE else fallback


# Build safe dict of all available tuners
for _name, _val in list(globals().items()):
    if _name.startswith("tune_") and callable(_val):
        _SAFE[_name[len("tune_") :]] = _val

_TUNABLE_QUANT_METHODS: Dict[str, Callable] = {
    "block_int8": _lazy("block_int8", tune_nf4),
    "block_int4": _lazy("block_int4", tune_nf4),
    "hadamard_int8": _lazy("hadamard_int8", tune_nf4),
    "hadamard_int4": _lazy("hadamard_int4", tune_nf4),
    "delta_int4": _lazy("delta_int4", tune_nf4),
    "sparsity_int4": _lazy("sparsity_int4", tune_nf4),
    "nf4": _lazy("nf4", tune_nf4),
    "binary_quant": _lazy("binary_quant", tune_nf4),
    "ternary_quant": _lazy("ternary_quant", tune_nf4),
    "group_wise_quant": _lazy("group_wise_quant", tune_nf4),
    "asymmetric_quant": _lazy("asymmetric_quant", tune_nf4),
    "hadamard_group_wise": _lazy("hadamard_group_wise", tune_nf4),
    "stochastic_round": _lazy("stochastic_round", tune_nf4),
    "residual_quant": _lazy("residual_quant", tune_nf4),
    "error_feedback_quant": _lazy("error_feedback_quant", tune_nf4),
    "kmeans_quant": _lazy("kmeans_quant", tune_nf4),
    "lloyd_max_quant": _lazy("lloyd_max_quant", tune_nf4),
    "octopus_quant": _lazy("octopus_quant", tune_nf4),
    "squeezellm_nonuniform": _lazy("squeezellm_nonuniform", tune_nf4),
    "awq_quant": _lazy("awq_quant", tune_nf4),
    "gptq_quant": _lazy("gptq_quant", tune_nf4),
    "mixed_precision": _lazy("mixed_precision", tune_nf4),
    "dynamic_bitwidth": _lazy("dynamic_bitwidth", tune_nf4),
    "multi_bitwidth": _lazy("multi_bitwidth", tune_nf4),
    "mixed_bitwidth_quant": _lazy("mixed_bitwidth_quant", tune_nf4),
    "adaptive_group_quant": _lazy("adaptive_group_quant", tune_nf4),
    "outlier_aware_quant": _lazy("outlier_aware_quant", tune_nf4),
    "sensitivity_aware_quant": _lazy("sensitivity_aware_quant", tune_nf4),
    "bqq_binary_quadratic": _lazy("bqq_binary_quadratic", tune_nf4),
    "block_floating_point": _lazy("block_floating_point", tune_block_int8),
    "e8_lattice": _lazy("e8_lattice", tune_nf4),
    "product_quantization": _lazy("product_quantization", tune_nf4),
    "dct_noise_aware": _lazy("dct_noise_aware", tune_block_int8),
    "weight_clustering_8bit": _lazy("weight_clustering_8bit", tune_nf4),
}

_TUNABLE_PHYSICS_METHODS: Dict[str, Callable] = {
    "density_matrix": _lazy("density_matrix", tune_nf4),
    "gyrokinetic": _lazy("gyrokinetic", tune_nf4),
    "hamiltonian_dynamical": _lazy("hamiltonian_dynamical", tune_nf4),
    "harmonic_oscillator": _lazy("harmonic_oscillator", tune_nf4),
    "manifold_learning": _lazy("manifold_learning", tune_nf4),
    "plasma_field": _lazy("plasma_field", tune_nf4),
    "quantum_entanglement": _lazy("quantum_entanglement", tune_nf4),
    "quantum_state": _lazy("quantum_state", tune_nf4),
    "quantum_tensor_network": _lazy("quantum_tensor_network", tune_nf4),
    "quantum_error_correction": _lazy("quantum_error_correction", tune_nf4),
    "quantum_tunneling": _lazy("quantum_tunneling", tune_nf4),
    "resonance_compression": _lazy("resonance_compression", tune_nf4),
    "resonance_modes": _lazy("resonance_modes", tune_nf4),
    "topological_functional": _lazy("topological_functional", tune_nf4),
    "fourier_neural_op": _lazy("fourier_neural_op", tune_nf4),
}

QUANTIZATION_TUNERS: Dict[str, Callable] = dict(_TUNABLE_QUANT_METHODS)
PHYSICS_TUNERS: Dict[str, Callable] = dict(_TUNABLE_PHYSICS_METHODS)
ALL_TUNERS: Dict[str, Callable] = {**_TUNABLE_QUANT_METHODS, **_TUNABLE_PHYSICS_METHODS}


def list_tunable_methods() -> List[str]:
    return sorted(ALL_TUNERS.keys())


def tune_method(method_name: str, *args: Any, **kwargs: Any) -> TunedParams:
    tuner = ALL_TUNERS.get(method_name)
    if tuner is None:
        raise ValueError(f"Unknown tunable method: {method_name}")
    return tuner(*args, **kwargs)
