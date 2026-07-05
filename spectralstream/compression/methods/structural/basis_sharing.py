from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BasisSharing:
    name = "basis_sharing"
    category = "structural"

    def __init__(
        self,
        dictionary_size: int = 256,
        n_iter: int = 30,
        sparsity: int = 16,
        block_size: int = 64,
    ):
        self.dictionary_size = dictionary_size
        self.n_iter = n_iter
        self.sparsity = sparsity
        self.block_size = block_size

    def _ksvd(
        self, data: np.ndarray, dict_size: int, n_iter: int, sparsity: int
    ) -> np.ndarray:
        n, d = data.shape
        rng = np.random.RandomState(42)
        n_atoms = min(dict_size, n)
        idx = rng.choice(n, size=n_atoms, replace=False)
        dictionary = data[idx].T.copy()
        dictionary = dictionary / (
            np.linalg.norm(dictionary, axis=0, keepdims=True) + 1e-10
        )
        if n_atoms < dict_size:
            extra = (
                np.random.RandomState(0)
                .randn(d, dict_size - n_atoms)
                .astype(np.float64)
            )
            extra = extra / (np.linalg.norm(extra, axis=0, keepdims=True) + 1e-10)
            dictionary = np.column_stack([dictionary, extra])
        sparse_k = min(sparsity, dict_size)
        for _ in range(n_iter):
            codes = np.zeros((dict_size, n), dtype=np.float64)
            for i in range(n):
                dists = np.linalg.norm(dictionary - data[i : i + 1].T, axis=0)
                top_k = np.argsort(dists)[:sparse_k]
                if sparse_k < dict_size:
                    sub_dict = dictionary[:, top_k]
                    codes[top_k, i] = np.linalg.lstsq(sub_dict, data[i], rcond=None)[0]
                else:
                    codes[:, i] = np.linalg.lstsq(dictionary, data[i], rcond=None)[0]

            for j in range(dict_size):
                used = np.where(np.abs(codes[j]) > 1e-10)[0]
                if len(used) > 0:
                    pred_all = codes[:, used].T @ dictionary.T
                    residual = data[used] - pred_all
                    E_j = residual + np.outer(codes[j, used], dictionary[:, j])
                    U, S, Vt = np.linalg.svd(E_j, full_matrices=False)
                    dictionary[:, j] = Vt[0, :]
                    codes[j, used] = S[0] * U[:, 0]

        return dictionary

    def compress(
        self, tensor: np.ndarray, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dict_size = kwargs.get("dictionary_size", self.dictionary_size)
        n_iter = kwargs.get("n_iter", self.n_iter)
        sparsity = kwargs.get("sparsity", self.sparsity)
        orig_shape = tensor.shape

        mat = tensor.reshape(tensor.shape[0], -1).astype(np.float64)
        n_layers, dim = mat.shape

        dictionary = self._ksvd(mat, dict_size, n_iter, sparsity)

        codes_list = []
        for i in range(n_layers):
            code = np.linalg.lstsq(dictionary, mat[i], rcond=None)[0]
            top_k = np.argsort(np.abs(code))[::-1][:sparsity]
            sparse_code = np.zeros(dict_size, dtype=np.float32)
            sparse_code[top_k] = code[top_k].astype(np.float32)
            codes_list.append(sparse_code)

        data_out = {
            "dictionary": dictionary.astype(np.float32),
            "codes": codes_list,
            "sparsity": sparsity,
        }
        meta = {"orig_shape": orig_shape, "method": self.name}
        return data_out, meta

    def decompress(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        dictionary = data["dictionary"]
        codes = np.array(data["codes"], dtype=np.float64)
        reconstructed = codes @ dictionary.T
        return reconstructed.reshape(metadata["orig_shape"]).astype(np.float32)

    def estimate_ratio(self, tensor: np.ndarray, **kwargs) -> float:
        dict_size = kwargs.get("dictionary_size", self.dictionary_size)
        sparsity = kwargs.get("sparsity", self.sparsity)
        orig = tensor.nbytes
        comp = dict_size * tensor.shape[-1] * 4
        comp += tensor.shape[0] * sparsity * 8
        return comp / max(orig, 1)
