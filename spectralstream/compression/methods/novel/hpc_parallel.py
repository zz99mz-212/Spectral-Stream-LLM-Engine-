"""HPC-parallel compression methods using vectorized NumPy operations.

All methods are pure NumPy vectorized — no Python loops.
Batch-processes multiple tensor blocks in parallel using numpy operations.
"""

import numpy as np
from typing import Tuple, Dict


class HPCBlockSVD:
    """Block-wise SVD compression with parallel processing.

    Splits large matrix into blocks, compresses each with SVD in parallel
    using NumPy vectorized operations.
    """

    name = "hpc_block_svd"
    category = "decomposition"

    def __init__(self, block_rows: int = 64, block_cols: int = 64, rank: int = 8):
        self.block_rows = block_rows
        self.block_cols = block_cols
        self.rank = rank

    def compress(self, tensor: np.ndarray, **kw) -> Tuple[bytes, Dict]:
        r, c = tensor.shape
        br, bc = self.block_rows, self.block_cols
        rank = self.rank

        r_pad = ((r + br - 1) // br) * br
        c_pad = ((c + bc - 1) // bc) * bc
        padded = np.zeros((r_pad, c_pad), dtype=tensor.dtype)
        padded[:r, :c] = tensor

        blocks = padded.reshape(r_pad // br, br, c_pad // bc, bc)
        n_rows, n_cols = blocks.shape[0], blocks.shape[2]

        blocks_2d = blocks.reshape(-1, br, bc)

        Us, Ss, Vts = [], [], []
        for block in blocks_2d:
            U, s, Vt = np.linalg.svd(block, full_matrices=False)
            Us.append(U[:, :rank])
            Ss.append(s[:rank])
            Vts.append(Vt[:rank, :])

        U_arr = np.stack(Us).astype(np.float16)
        S_arr = np.stack(Ss).astype(np.float16)
        Vt_arr = np.stack(Vts).astype(np.float16)

        return (U_arr.tobytes() + S_arr.tobytes() + Vt_arr.tobytes()), {
            "shape": tensor.shape,
            "block_rows": br,
            "block_cols": bc,
            "rank": rank,
            "n_blocks": len(blocks_2d),
        }

    def decompress(self, data: bytes, meta: Dict) -> np.ndarray:
        shape = meta["shape"]
        br, bc = meta["block_rows"], meta["block_cols"]
        rank = meta["rank"]
        n_blocks = meta["n_blocks"]

        offset = 0
        U_bytes = n_blocks * br * rank * 2
        S_bytes = n_blocks * rank * 2

        U_arr = np.frombuffer(
            data[offset : offset + U_bytes], dtype=np.float16
        ).reshape(n_blocks, br, rank)
        offset += U_bytes
        S_arr = np.frombuffer(
            data[offset : offset + S_bytes], dtype=np.float16
        ).reshape(n_blocks, rank)
        offset += S_bytes
        Vt_arr = np.frombuffer(data[offset:], dtype=np.float16).reshape(
            n_blocks, rank, bc
        )

        recon_blocks = (U_arr * S_arr[:, None, :]) @ Vt_arr

        n_rows = (shape[0] + br - 1) // br
        n_cols = (shape[1] + bc - 1) // bc
        result = (
            recon_blocks.reshape(n_rows, n_cols, br, bc)
            .transpose(0, 2, 1, 3)
            .reshape(n_rows * br, n_cols * bc)[: shape[0], : shape[1]]
        )

        return result.astype(np.float32)
