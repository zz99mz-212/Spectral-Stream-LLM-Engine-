"""SVD-based matrix decomposition methods."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .svd_decomposition import svd_truncated


class SVDTruncated:
    """Truncated SVD: W ~ U @ diag(s) @ Vh."""

    name = "svd_truncated"
    category = "decomposition"

    def compress(
        self, tensor: np.ndarray, rank: int = None, energy_threshold: float = 0.90
    ) -> Tuple[bytes, dict]:
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.ndim < 2:
            flat = tensor.ravel().astype(np.float32)
            return flat.astype(np.float16).tobytes(), {
                "original_shape": tensor.shape,
                "shape": tensor.shape,
                "passthrough": True,
            }
        if rank is None:
            rank = max(1, min(64, min(tensor.shape) // 2))
        result, ratio, snr = svd_truncated(tensor, rank, energy_threshold)
        data = result["U"].tobytes() + result["s"].tobytes() + result["Vt"].tobytes()
        meta = dict(
            shape=result["shape"],
            U_shape=list(result["U"].shape),
            s_shape=list(result["s"].shape),
            Vt_shape=list(result["Vt"].shape),
        )
        return data, meta

    def decompress(self, data: bytes, metadata: dict) -> np.ndarray:
        if metadata.get("passthrough"):
            return (
                np.frombuffer(data, dtype=np.float16)
                .copy()
                .reshape(metadata["shape"])
                .astype(np.float32)
            )
        off = 0
        nU = int(np.prod(metadata["U_shape"]))
        U = np.frombuffer(data[off : off + nU * 4], dtype=np.float32).reshape(
            metadata["U_shape"]
        )
        off += nU * 4
        ns = int(np.prod(metadata["s_shape"]))
        s = np.frombuffer(data[off : off + ns * 4], dtype=np.float32)
        off += ns * 4
        nV = int(np.prod(metadata["Vt_shape"]))
        Vt = np.frombuffer(data[off : off + nV * 4], dtype=np.float32).reshape(
            metadata["Vt_shape"]
        )
        recon = (U * s) @ Vt
        return recon.reshape(metadata["shape"]).astype(np.float32)
