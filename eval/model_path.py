"""
Model-path resolution for the eval subsystem.

Resolution order (per D-10 / D-11):
1. ``cli_model`` argument (CLI ``--model`` / ``--compressed``)
2. ``SPECTRALSTREAM_MODEL_PATH`` environment variable
3. ``models/gemma-4-E2B/model.safetensors`` fallback

Reuses the same ``_PATH_TRAVERSAL_PATTERN`` from ``spectralstream/compression/cli.py``
to guard against directory-traversal attacks (T-02-01-01).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Path-traversal regex matching what cli.py uses (line 78).
# Catches ``../``, ``..\\``, ``/..``, ``\..`` in any position.
_PATH_TRAVERSAL_PATTERN = re.compile(r"\.\./|\.\.\\|/\.\.|\\\.\.")

# Fallback path when no explicit model path is given.
_DEFAULT_MODEL_PATH = "models/gemma-4-E2B/model.safetensors"


def resolve_model_path(cli_model: str | None = None) -> str:
    """Resolve and validate a model/safetensors/SSF file path.

    Parameters
    ----------
    cli_model : str or None
        Value supplied via ``--model`` / ``--compressed`` CLI flag, if any.

    Returns
    -------
    str
        Absolute, validated, existing file path.

    Raises
    ------
    ValueError
        If the path contains ``..`` traversal segments.
    FileNotFoundError
        If the resolved path does not point to an existing file.
    """
    raw: str | None = cli_model

    if raw is None or raw == "":
        raw = os.environ.get("SPECTRALSTREAM_MODEL_PATH")

    if raw is None or raw == "":
        raw = _DEFAULT_MODEL_PATH

    if not isinstance(raw, str) or not raw:
        raise ValueError("Model path must be a non-empty string")

    _validate_path_safety(raw)

    resolved = Path(raw).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")

    return str(resolved)


def _validate_path_safety(path: str) -> None:
    """Check for directory-traversal patterns and raise ``ValueError`` if found."""
    if _PATH_TRAVERSAL_PATTERN.search(path):
        raise ValueError(f"Path traversal detected: {path!r}")
