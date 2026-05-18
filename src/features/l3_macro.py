"""
L3 — Macro context features (research_design.md §4.3).

Five features:

    vnindex_return  = log(I_{t-1} / I_{t-2})                          # daily
    usdvnd_change   = log(X_{t-1} / X_{t-2})                          # daily (FX-aligned)
    cpi_yoy_pct     = CPI YoY % from monthly macro CSV                # forward-filled
    sbv_rate_pct    = SBV refinancing rate from monthly macro CSV     # forward-filled
    gdp_yoy_pct     = GDP YoY % from quarterly macro CSV              # forward-filled

Anti-leakage convention
-----------------------
- Daily features (``vnindex_return``, ``usdvnd_change``) use ``shift(1)``
  to ensure the feature at trading day ``t`` depends only on values at
  ``t-1`` and earlier — same pattern as L1 (verified in Session 3).
- Forward-filled features (CPI, SBV, GDP) use :func:`asof_join` keyed on
  ``release_date``: at day ``t`` the value used is from the row with the
  largest ``release_date ≤ t``. The asof_join primitive was verified in
  Session 2 (100% coverage, 24 tests).

Calendar alignment (decisions D3, D4)
-------------------------------------
- **VN-Index** (D4): strict alignment to TCB calendar via ``reindex`` with
  no fill. vnstock VCI returns VN-Index on the same HOSE calendar as TCB,
  so a missing date is a data-quality issue that must surface immediately
  (we raise rather than silently mask).
- **USD/VND FX** (D3): aligned to TCB calendar via ``reindex(method="ffill")``.
  Global FX market has 5-day-max gaps (Easter Monday, etc.) that do not
  match HOSE; on those gap days the USD/VND rate is treated as unchanged
  from the previous trading day (yielding a 0 log-return on the gap day,
  not NaN). This matches the economic semantics that the rate "didn't
  move" while FX was closed.

Schema dependencies
-------------------
Caller must pass DataFrames matching the Session 1 / Session 4 schemas:
- ``price_df`` (TCB price) — DatetimeIndex (HOSE calendar), provides the
  output index.
- ``vnindex_df`` — must contain column ``adj_close``.
- ``fx_df`` — must contain column ``rate``.
- ``macro_monthly_df`` — output of :func:`src.data.loaders.load_macro_monthly`.
- ``macro_quarterly_df`` — output of :func:`src.data.loaders.load_macro_quarterly`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.asof_join import asof_join
from src.utils.logging import get_logger

log = get_logger(__name__)

# Public column names (frozen contract).
L3_FEATURE_COLS: tuple[str, ...] = (
    "vnindex_return",
    "usdvnd_change",
    "cpi_yoy_pct",
    "sbv_rate_pct",
    "gdp_yoy_pct",
)


# ---------------------------------------------------------------------------
# Top-level composer
# ---------------------------------------------------------------------------


def compute_l3_features(
    price_df: pd.DataFrame,
    vnindex_df: pd.DataFrame,
    fx_df: pd.DataFrame,
    macro_monthly_df: pd.DataFrame,
    macro_quarterly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the five L3 macro context features, indexed on TCB's calendar.

    Parameters
    ----------
    price_df : pd.DataFrame
        TCB price panel. Index = HOSE trading calendar; provides the output
        index. Columns content is not inspected; only ``price_df.index``
        is used.
    vnindex_df : pd.DataFrame
        VN-Index panel. Must contain column ``adj_close`` indexed on a
        DatetimeIndex that is a superset of ``price_df.index``.
    fx_df : pd.DataFrame
        USD/VND FX panel. Must contain column ``rate`` indexed on a
        DatetimeIndex (FX calendar, distinct from HOSE).
    macro_monthly_df : pd.DataFrame
        Output of :func:`src.data.loaders.load_macro_monthly`. Must
        contain columns ``release_date``, ``cpi_yoy_pct``,
        ``sbv_refinancing_rate_pct``.
    macro_quarterly_df : pd.DataFrame
        Output of :func:`src.data.loaders.load_macro_quarterly`. Must
        contain columns ``release_date``, ``gdp_yoy_pct``.

    Returns
    -------
    pd.DataFrame
        Indexed on ``price_df.index``, columns =
        :data:`L3_FEATURE_COLS`. Rows in the warmup region or before any
        macro release are NaN in the relevant features.

    Raises
    ------
    ValueError
        If ``vnindex_df`` is missing dates that exist in ``price_df.index``
        (strict alignment per decision D4); if any required column is
        missing from input frames.
    """
    _validate_inputs(price_df, vnindex_df, fx_df, macro_monthly_df, macro_quarterly_df)
    target_index = price_df.index

    # --- Daily features (shift(1) anti-leak) -------------------------------
    vnindex_return = compute_vnindex_return(vnindex_df, target_index)
    usdvnd_change = compute_usdvnd_change(fx_df, target_index)

    # --- Forward-fill features (asof_join on release_date) -----------------
    # The asof_join primitive ignores extra columns (reference_period,
    # release_date_source) — only release_date_col + value_cols are used.
    daily_skeleton = pd.DataFrame(index=target_index)

    cpi_sbv = asof_join(
        daily_skeleton,
        macro_monthly_df,
        value_cols=["cpi_yoy_pct", "sbv_refinancing_rate_pct"],
        release_date_col="release_date",
    )
    gdp = asof_join(
        daily_skeleton,
        macro_quarterly_df,
        value_cols=["gdp_yoy_pct"],
        release_date_col="release_date",
    )

    result = pd.DataFrame(
        {
            "vnindex_return": vnindex_return,
            "usdvnd_change": usdvnd_change,
            "cpi_yoy_pct": cpi_sbv["cpi_yoy_pct"],
            "sbv_rate_pct": cpi_sbv["sbv_refinancing_rate_pct"],
            "gdp_yoy_pct": gdp["gdp_yoy_pct"],
        },
        index=target_index,
    )

    log.info(
        "compute_l3_features: %d rows. Warmup NaN by feature: %s",
        len(result),
        {c: int(result[c].isna().sum()) for c in result.columns},
    )
    return result


# ---------------------------------------------------------------------------
# Individual feature builders (unit-testable)
# ---------------------------------------------------------------------------


def compute_vnindex_return(
    vnindex_df: pd.DataFrame, target_index: pd.DatetimeIndex
) -> pd.Series:
    """
    log(I_{t-1} / I_{t-2}), aligned strictly to ``target_index``.

    Per decision D4: VN-Index and TCB share the HOSE calendar (both
    vnstock VCI). A missing date in ``vnindex_df`` for a date present
    in ``target_index`` is a data integrity issue and raises immediately.
    """
    aligned = vnindex_df["adj_close"].reindex(target_index)
    if aligned.isna().any():
        n_missing = int(aligned.isna().sum())
        missing_dates = aligned.index[aligned.isna()][:5].tolist()
        raise ValueError(
            f"VN-Index is missing {n_missing} dates that exist in the TCB "
            f"calendar. VN-Index and TCB share HOSE calendar — this should "
            f"not happen. First missing: {missing_dates}"
        )
    log_i = np.log(aligned)
    return log_i.shift(1) - log_i.shift(2)


def compute_usdvnd_change(
    fx_df: pd.DataFrame, target_index: pd.DatetimeIndex
) -> pd.Series:
    """
    log(X_{t-1} / X_{t-2}) on TCB calendar, FX gaps forward-filled.

    Per decision D3: USD/VND FX market has 5-day-max gaps (Easter
    Monday etc.) not aligned with HOSE. We reindex with ``method="ffill"``,
    treating the rate as unchanged on FX-closed days. Returns on those
    days will be 0 (not NaN). Dates before FX series starts remain NaN.
    """
    aligned = fx_df["rate"].reindex(target_index, method="ffill")
    log_x = np.log(aligned)
    return log_x.shift(1) - log_x.shift(2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    price_df: pd.DataFrame,
    vnindex_df: pd.DataFrame,
    fx_df: pd.DataFrame,
    macro_monthly_df: pd.DataFrame,
    macro_quarterly_df: pd.DataFrame,
) -> None:
    if not isinstance(price_df.index, pd.DatetimeIndex):
        raise ValueError(
            f"price_df.index must be DatetimeIndex, got "
            f"{type(price_df.index).__name__}"
        )
    if "adj_close" not in vnindex_df.columns:
        raise ValueError(
            f"vnindex_df missing required column 'adj_close'. "
            f"Got: {vnindex_df.columns.tolist()}"
        )
    if "rate" not in fx_df.columns:
        raise ValueError(
            f"fx_df missing required column 'rate'. "
            f"Got: {fx_df.columns.tolist()}"
        )
    for name, df, required in [
        ("macro_monthly_df", macro_monthly_df, ["release_date", "cpi_yoy_pct", "sbv_refinancing_rate_pct"]),
        ("macro_quarterly_df", macro_quarterly_df, ["release_date", "gdp_yoy_pct"]),
    ]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name} missing required column(s): {missing}. "
                f"Got: {df.columns.tolist()}"
            )