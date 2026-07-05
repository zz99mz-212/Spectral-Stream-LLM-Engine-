from __future__ import annotations

import enum
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_GB = 1024**3


class StreamingMode(enum.IntEnum):
    """Compression memory-strategy modes."""

    FULL_RAM = 1
    """Mode 1: load all weights into RAM, compress in bulk.

    Suitable when ``model_size <= available_ram * 0.5``.
    All tensors are read into numpy arrays before any compression runs;
    yields the highest throughput because the OS page cache is warm for
    the entire model.
    """

    STREAMING = 2
    """Mode 2: stream tensors one-at-a-time from disk.

    Suitable when RAM is constrained (4–16 GB, model >> RAM).
    Tensors are memory-mapped, profiled, compressed, and the result is
    flushed to the output SSF file before the next tensor is touched.
    Peak RSS ≈ max(tensor_size, 2 × chunk_size) + overhead.
    """

    HYBRID = 3
    """Mode 3: profile all tensor *metadata* first, then route each tensor.

    Small tensors (≤ ``ram_budget / 4``) are loaded into RAM and compressed
    in bulk. Large tensors are streamed/chunked individually.
    This avoids repeatedly mmap′ing small tensors while keeping peak
    RSS under control.
    """


def auto_select_mode(
    model_size_bytes: int,
    available_ram_bytes: int,
    mode_hint: Optional[str] = None,
) -> StreamingMode:
    """Automatically select the best streaming mode.

    Parameters
    ----------
    model_size_bytes : int
        Total size of model weights on disk (sum of all tensor nbytes).
    available_ram_bytes : int
        System-wide available RAM (or a user-supplied budget).
    mode_hint : str, optional
        One of ``"full-ram"``, ``"streaming"``, ``"hybrid"``, ``"auto"``.
        ``None`` defaults to ``"auto"``.

    Returns
    -------
    StreamingMode
        The selected mode.

    Logic
    -----
    * ``mode_hint in {"full-ram", "ram"}`` → ``FULL_RAM``
    * ``mode_hint == "streaming"`` → ``STREAMING``
    * ``mode_hint == "hybrid"`` → ``HYBRID``
    * ``mode_hint is None or mode_hint == "auto"``:
        - ``model / ram <= 0.5`` → ``FULL_RAM``
        - ``model / ram <= 2.0`` → ``HYBRID``
        - otherwise → ``STREAMING``
    """
    if mode_hint in ("full-ram", "ram"):
        return StreamingMode.FULL_RAM
    if mode_hint == "streaming":
        return StreamingMode.STREAMING
    if mode_hint == "hybrid":
        return StreamingMode.HYBRID

    ratio = model_size_bytes / max(available_ram_bytes, 1)
    if ratio <= 0.5:
        logger.info("Mode: FULL_RAM (model/ram=%.2f ≤ 0.5)", ratio)
        return StreamingMode.FULL_RAM
    if ratio <= 2.0:
        logger.info("Mode: HYBRID (0.5 < model/ram=%.2f ≤ 2.0)", ratio)
        return StreamingMode.HYBRID
    logger.info("Mode: STREAMING (model/ram=%.2f > 2.0)", ratio)
    return StreamingMode.STREAMING


def select_mode_for_config(
    model_size_bytes: int,
    max_memory_gb: float = 48.0,
    streaming_flag: Optional[bool] = None,
    mode_flag: Optional[str] = None,
) -> StreamingMode:
    """Convenience wrapper for CLI integration.

    Parameters
    ----------
    model_size_bytes : int
        Total weight size on disk.
    max_memory_gb : float
        User-specified max memory budget in GB (``--max-memory-gb``).
    streaming_flag : bool, optional
        ``True`` → force streaming; ``False`` → force full-RAM;
        ``None`` → auto.
    mode_flag : str, optional
        String from ``--streaming-mode {full-ram,streaming,hybrid,auto}``.

    Returns
    -------
    StreamingMode
    """
    budget = int(max_memory_gb * _GB)

    if streaming_flag is False:
        return StreamingMode.FULL_RAM
    if streaming_flag is True:
        return StreamingMode.STREAMING

    return auto_select_mode(model_size_bytes, budget, mode_flag)


class ModeSelector:
    """Caches the model-size / RAM ratio and provides query methods
    for per-tensor routing decisions.

    Parameters
    ----------
    model_size_bytes : int
    available_ram_bytes : int
    mode : StreamingMode, optional
        If not provided, ``auto_select_mode`` is called.
    """

    def __init__(
        self,
        model_size_bytes: int,
        available_ram_bytes: int,
        mode: Optional[StreamingMode] = None,
    ) -> None:
        self._model_bytes: int = model_size_bytes
        self._ram_bytes: int = available_ram_bytes
        self._mode: StreamingMode = (
            mode
            if mode is not None
            else auto_select_mode(model_size_bytes, available_ram_bytes)
        )

    @property
    def mode(self) -> StreamingMode:
        return self._mode

    def should_stream_tensor(self, tensor_nbytes: int, ram_budget: int) -> bool:
        """Return True if this individual tensor should be streamed."""
        if self._mode == StreamingMode.FULL_RAM:
            return False
        if self._mode == StreamingMode.STREAMING:
            return True
        return tensor_nbytes > ram_budget // 4
