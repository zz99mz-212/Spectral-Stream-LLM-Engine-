"""Progressive memory manager — tracks and aggressively releases memory between compression stages.

Enables consumer devices (4-16GB RAM) to compress frontier models by
freeing memory between stages instead of holding all intermediates.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)


class ProgressiveMemoryManager:
    """Tracks and aggressively releases memory between compression stages.

    Implements a phased release strategy:
    - Phase 0: Release input tensor after first stage output is produced
    - Phase 1: Release intermediate stage tensors after next stage starts
    - Phase 2: Release all stage residuals after compression completes
    - Phase 3: Release method instances no longer needed

    Memory budget: configurable per-device profile (4GB, 8GB, 16GB, 64GB)

    Usable as a context manager via __enter__/__exit__.
    """

    def __init__(self, memory_budget_mb: float = 1024.0) -> None:
        self.memory_budget_mb = memory_budget_mb
        self._tracked: Dict[str, Any] = {}
        self._stages: List[str] = []
        self._phases: Dict[int, Set[str]] = {0: set(), 1: set(), 2: set(), 3: set()}
        self._checkpoints: List[str] = []

    def __enter__(self) -> ProgressiveMemoryManager:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release_all()

    def track(self, name: str, obj: Any, phase: int = 0) -> None:
        """Track an object for later release, optionally assigning a release phase."""
        self._tracked[name] = obj
        if phase in self._phases:
            self._phases[phase].add(name)
        logger.debug("Tracking %s (phase=%d, type=%s)", name, phase, type(obj).__name__)

    def release(self, name: str) -> bool:
        """Release a specific tracked object by name. Returns True if released."""
        if name in self._tracked:
            obj = self._tracked.pop(name, None)
            if obj is not None:
                if isinstance(obj, np.ndarray):
                    obj.resize(0, refcheck=False)
                del obj
            for phase_set in self._phases.values():
                phase_set.discard(name)
            gc.collect()
            logger.debug("Released %s", name)
            return True
        return False

    def release_all(self) -> None:
        """Release ALL tracked objects."""
        names = list(self._tracked.keys())
        for name in names:
            self.release(name)
        self._stages.clear()
        gc.collect()
        logger.debug("Released all tracked objects (%d total)", len(names))

    def release_stage(self, stage_idx: int) -> None:
        """Release all objects from a specific cascade stage."""
        if stage_idx < 0 or stage_idx >= len(self._stages):
            return
        stage_name = self._stages[stage_idx]
        released: List[str] = []
        for name in list(self._tracked.keys()):
            if name.startswith(f"stage_{stage_idx}_") or name.startswith(stage_name):
                if self.release(name):
                    released.append(name)
        logger.debug(
            "Released stage %d (%s): %d objects", stage_idx, stage_name, len(released)
        )

    def release_phase(self, phase: int) -> None:
        """Release all objects assigned to a given release phase."""
        if phase not in self._phases:
            return
        names = list(self._phases[phase])
        for name in names:
            self.release(name)
        gc.collect()
        logger.debug("Released phase %d: %d objects", phase, len(names))

    def get_current_memory_mb(self) -> float:
        """Estimate current memory usage from tracked numpy arrays."""
        total = 0.0
        for obj in self._tracked.values():
            if isinstance(obj, np.ndarray):
                total += obj.nbytes / (1024.0 * 1024.0)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, np.ndarray):
                        total += item.nbytes / (1024.0 * 1024.0)
        return total

    def within_budget(self) -> bool:
        """Check if current memory is within budget."""
        return self.get_current_memory_mb() <= self.memory_budget_mb

    def checkpoint(self, name: str) -> None:
        """Create a checkpoint: release everything except current stage objects."""
        self._checkpoints.append(name)
        keep: Set[str] = set()
        current_stage_phase: Optional[int] = None
        for phase, names in self._phases.items():
            for n in names:
                if n.startswith(name):
                    current_stage_phase = phase
                    break
        if current_stage_phase is not None:
            for phase, names in self._phases.items():
                if phase >= current_stage_phase:
                    keep.update(names)
        else:
            return
        to_release = [n for n in self._tracked if n not in keep]
        for n in to_release:
            self.release(n)
        gc.collect()
        logger.debug(
            "Checkpoint %s: kept %d objects, released %d",
            name,
            len(keep),
            len(to_release),
        )

    def add_stage(self, stage_name: str) -> None:
        """Register a compression stage for phased release tracking."""
        self._stages.append(stage_name)
        self._phases[1].add(stage_name)

    def phase_progress(self, phase: int) -> None:
        """Advance through release phases as compression progresses.

        Phase 0 → release input after first stage output
        Phase 1 → release intermediates after next stage starts
        Phase 2 → release all residuals after compression completes
        Phase 3 → release method instances
        """
        for p in range(phase + 1):
            self.release_phase(p)
        logger.debug("Advanced to phase %d", phase)
