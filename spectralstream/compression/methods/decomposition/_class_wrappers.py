"""Backward-compatible wrapper — all classes from submodules."""

from __future__ import annotations

from .butterfly import Butterfly
from .butterfly import Monarch
from .cp import CPDecomposition
from .einsort import EinsortTT
from .einsort import LOTR
from .kronecker import Kronecker
from .kronecker import CURDecomposition
from .matrix_approx import HMatrix
from .matrix_approx import Nystrom
from .matrix_approx import RandomFeature
from .merapeps import ADNTNMERA
from .merapeps import IPEPS2D
from .structured_mat import BlockDiagonal
from .structured_mat import Toeplitz
from .structured_mat import Hankel
from .svd import SVDTruncated
from .tensor_network import TensorNetwork
from .tensor_network import HierarchicalMPS
from .tensor_train import TensorTrain
from .tensor_train import TensorRing
from .tensor_train import TTOrthogonal
from .tensor_train import TTSVD
from .tucker import TuckerDecomposition
from .tucker import BlockTucker
from .tucker import HierarchicalTucker
