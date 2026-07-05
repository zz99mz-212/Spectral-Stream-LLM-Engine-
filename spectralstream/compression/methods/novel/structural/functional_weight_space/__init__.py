from __future__ import annotations

from typing import Callable, List

import numpy as np

from .mlpfit import (
    _make_fourier_class,
    _make_hash_class,
    _make_hierarchical_class,
    _make_mlp_class,
    _make_poly_class,
    _make_rbf_class,
    _make_siren_class,
    _make_sparse_class,
    _make_spline_class,
    _make_symbolic_class,
    _adaptive_select_best,
    _BlockINT8Wrapper,
)


class _FWSAdaptive:
    name = "fws_adaptive"
    category = "functional_weight_space"

    def compress(self, tensor: np.ndarray, **kw):
        data, meta = _adaptive_select_best(tensor)
        return data, meta

    def decompress(self, data: bytes, metadata: dict):
        if metadata.get("_fallback"):
            return _BlockINT8Wrapper.decompress(data, metadata)
        # Re-route to appropriate decompressor based on metadata
        from .mlpfit import (
            _fourier_decompress,
            _hash_decompress,
            _poly_decompress,
            _siren_decompress,
        )

        if "hidden_dim" in metadata:
            return _siren_decompress(data, metadata, np.sin)
        elif "n_feats" in metadata:
            return _fourier_decompress(data, metadata)
        elif "n_grid" in metadata:
            return _hash_decompress(data, metadata)
        elif "degree" in metadata:
            return _poly_decompress(data, metadata)
        return _BlockINT8Wrapper.decompress(data, metadata)


def _build_all_fws_methods() -> List[type]:
    methods: List[type] = []

    # ── SIREN: 5 activations × 4 hidden/layer configs × 2 frequencies = 40 ──
    siren_acts: dict[str, Callable] = {
        "sin": np.sin,
        "relu": lambda x: np.maximum(0, x),
        "tanh": np.tanh,
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -100, 100))),
        "gelu": lambda x: x
        * 0.5
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
    }
    siren_configs = [(16, 2), (32, 2), (32, 3), (64, 2)]
    for act_name, act_fn in siren_acts.items():
        for hidden, layers in siren_configs:
            for freq in [1.0, 2.0]:
                name_str = f"siren_{act_name}_h{hidden}_l{layers}_f{freq}"
                methods.append(
                    _make_siren_class(name_str, act_fn, hidden, layers, freq)
                )

    # ── MLP: 3 activations × 5 architectures = 15 ──
    mlp_acts: dict[str, Callable] = {
        "sin": np.sin,
        "gelu": siren_acts["gelu"],
        "tanh": np.tanh,
    }
    mlp_archs = [[8, 8], [16, 16], [32, 16], [64, 32], [64, 32, 16]]
    for act_name, act_fn in mlp_acts.items():
        for widths in mlp_archs:
            w_str = "x".join(str(w) for w in widths)
            name_str = f"neural_field_{act_name}_{w_str}"
            methods.append(_make_mlp_class(name_str, act_fn, widths))

    # ── Fourier: 4 feature sizes × 3 sigmas = 12 ──
    for n_feats in [32, 64, 128, 256]:
        for sigma in [0.5, 1.0, 2.0]:
            name_str = f"fourier_f{n_feats}_s{sigma}"
            methods.append(_make_fourier_class(name_str, n_feats, sigma))

    # ── Hash: 4 grid sizes = 4 ──
    for n_grid in [16, 32, 64, 128]:
        name_str = f"hash_grid{n_grid}"
        methods.append(_make_hash_class(name_str, n_grid))

    # ── Polynomial: 7 degrees = 7 ──
    for degree in [4, 6, 8, 10, 12, 16, 20]:
        name_str = f"poly_deg{degree}"
        methods.append(_make_poly_class(name_str, degree))

    # ── Spline: 5 knot counts = 5 ──
    for n_knots in [8, 16, 32, 64, 128]:
        name_str = f"spline_k{n_knots}"
        methods.append(_make_spline_class(name_str, n_knots))

    # ── RBF: 4 centers × 3 kernels × 3 center types = 36 ──
    rbf_kernels: dict[str, Callable] = {
        "gauss": lambda d: np.exp(-(d**2)),
        "iq": lambda d: 1.0 / (1.0 + d**2),
        "mq": lambda d: np.sqrt(1.0 + d**2),
    }
    center_types = ["uniform", "random", "seed_42"]
    for n_centers in [8, 16, 32, 64]:
        for kname, kfn in rbf_kernels.items():
            for ctype in center_types:
                name_str = f"rbf_c{n_centers}_{kname}_{ctype}"
                methods.append(_make_rbf_class(name_str, n_centers, kfn, ctype))

    # ── Sparse coding: 2 basis types × 6 configs = 12 ──
    sparse_configs = [(16, 4), (32, 8), (64, 16), (128, 32), (256, 64), (512, 128)]
    for basis_type in ["dct", "fwht"]:
        for n_basis, n_nonzero in sparse_configs:
            name_str = f"sparse_{basis_type}_b{n_basis}_nz{n_nonzero}"
            methods.append(_make_sparse_class(name_str, n_basis, n_nonzero, basis_type))

    # ── Hierarchical: 2 activations × 4 configs = 8 ──
    hier_acts: dict[str, Callable] = {"sin": np.sin, "tanh": np.tanh}
    hier_configs = [(2, 16), (2, 32), (3, 12), (3, 24)]
    for act_name, act_fn in hier_acts.items():
        for n_levels, base_hidden in hier_configs:
            name_str = f"hierarchical_l{n_levels}_h{base_hidden}_{act_name}"
            methods.append(
                _make_hierarchical_class(name_str, n_levels, base_hidden, act_fn)
            )

    # ── Symbolic: 6 function sets = 6 ──
    func_sets = ["trig", "poly", "mixed", "trig_poly", "all", "exp"]
    for fset in func_sets:
        name_str = f"symbolic_{fset}"
        methods.append(_make_symbolic_class(name_str, fset))

    # ── Adaptive selector ──
    methods.append(_FWSAdaptive)

    return methods


ALL_FUNCTIONAL_WEIGHT_SPACE_METHODS: List[type] = _build_all_fws_methods()
