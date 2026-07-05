"""
Unstructured Magnitude Pruning
================================
Removes smallest-magnitude weights regardless of structure.
"""

from __future__ import annotations

from spectralstream.compression.methods.structural._class_wrappers import (
    UnstructuredPruning as UnstructuredMag,
)
