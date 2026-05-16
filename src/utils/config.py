"""
Central configuration constants for the project.

Implements the spec in IMPLEMENTATION.md section 6.

Frozen dataclass to prevent runtime mutation. Any change requires explicit
code edit + git commit, supporting pre-registration audit (research_design.md).

DO NOT modify hyperparameter values after pre-registration lock (end of Tuần 2),
except via bug-fix workflow described in research_design.md section 12.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """All project-wide constants."""

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    TCB_START_DATE: str = "2018-06-04"  # Ngày niêm yết HOSE
    DATA_DIR: str = "data/"
    RAW_DIR: str = "data/raw/"
    PROCESSED_DIR: str = "data/processed/"
    DB_PATH: str = "data/db/predictions.sqlite"

    # File paths (raw)
    PRICE_TCB_PATH: str = "data/raw/price_tcb.parquet"
    PRICE_VNINDEX_PATH: str = "data/raw/price_vnindex.parquet"
    FX_USDVND_PATH: str = "data/raw/fx_usdvnd.parquet"
    MACRO_MONTHLY_PATH: str = "data/raw/macro_monthly.csv"
    MACRO_QUARTERLY_PATH: str = "data/raw/macro_quarterly.csv"
    TCB_FUNDAMENTALS_PATH: str = "data/raw/tcb_fundamentals.csv"

    # -----------------------------------------------------------------
    # Walk-forward (research_design.md section 6)
    # -----------------------------------------------------------------
    HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
    WINDOW_SIZE: int = 1000           # Rolling window, NOT expanding
    TEST_START: str = "2022-07-01"
    REFIT_CADENCE: str = "W-FRI"      # Weekly refit, end of Friday

    # -----------------------------------------------------------------
    # Models — frozen post-Phase 0 (research_design.md section 5)
    # -----------------------------------------------------------------
    # Elastic Net
    EN_ALPHA: float = 0.5             # Mixing param L1/L2
    EN_LAMBDA: float = 0.0            # Filled after Phase 0 5-fold TimeSeriesSplit

    # Random Forest
    RF_N_ESTIMATORS: int = 500
    RF_MAX_DEPTH: int | None = None
    RF_MIN_SAMPLES_LEAF: int = 5
    RF_MAX_FEATURES: str = "sqrt"

    # LSTM
    LSTM_T: int = 20                  # Lookback timesteps
    LSTM_HIDDEN: int = 32
    LSTM_DROPOUT: float = 0.2
    LSTM_LR: float = 1e-3
    LSTM_BATCH: int = 32
    LSTM_PATIENCE: int = 10           # Early stopping
    LSTM_MAX_EPOCHS: int = 100
    LSTM_INNER_VAL_FRAC: float = 0.15

    # -----------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------
    SEED: int = 42

    # -----------------------------------------------------------------
    # Statistical tests (research_design.md section 7)
    # -----------------------------------------------------------------
    BOOTSTRAP_B: int = 2000              # n_bootstrap replications
    BOOTSTRAP_BLOCK_MULT: int = 2        # Mean block size = BOOTSTRAP_BLOCK_MULT * k
    DM_LOSS: str = "0-1"                 # 0-1 loss for DM test
    SIGNIFICANCE_ALPHA: float = 0.05     # After Holm correction

    # -----------------------------------------------------------------
    # Holm correction — Scope A (research_design.md section 7.3)
    # 3 models × 3 baselines = 9 tests per horizon
    # -----------------------------------------------------------------
    N_MODELS: int = 3
    N_BASELINES: int = 3
    N_HOLM_TESTS_PER_HORIZON: int = 9


CFG = Config()
"""Singleton instance. Import as: `from src.utils.config import CFG`."""
