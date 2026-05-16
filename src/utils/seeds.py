"""
Global seed setup for reproducibility.

Implements IMPLEMENTATION.md section 7. Must be called at the start of every
entry point: scripts/run_*.py, src/inference/daily.py, src/inference/weekly_refit.py.

PyTorch is optional here (try/except on import) so utilities and tests that
don't touch GPU code don't trigger torch import overhead.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

_LOGGER = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for:
      - Python `random`
      - `PYTHONHASHSEED` environment variable
      - NumPy
      - PyTorch (CPU + CUDA + cuDNN deterministic mode), if installed

    Parameters
    ----------
    seed : int, default=42
        Seed value. Project-wide canonical seed is 42 (see CFG.SEED).

    Notes
    -----
    Deterministic CUDA is slower than non-deterministic but required for
    bit-identical reproducibility across runs. Trade-off accepted for
    scientific validity (research_design.md section 13).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # type: ignore[import-untyped]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        _LOGGER.info(
            "Global seed set to %d (Python+NumPy+PyTorch%s)",
            seed,
            " with CUDA deterministic" if torch.cuda.is_available() else "",
        )
    except ImportError:
        _LOGGER.info("Global seed set to %d (Python+NumPy only; torch not installed)", seed)
