"""
L1 — Lag log-return features (research_design.md §4.1).

Four features, computed daily from the adjusted close price:

    r_lag1  = log(P_{t-1} / P_{t-2})
    r_cum5  = log(P_{t-1} / P_{t-6})
    r_cum10 = log(P_{t-1} / P_{t-11})
    r_cum20 = log(P_{t-1} / P_{t-21})

Anti-leakage convention (research_design.md §4.1, §2.5)
-------------------------------------------------------
Every feature at trading day ``t`` uses information only from days ``t-1``
and earlier — the close price ``P_t`` of day ``t`` itself is NOT used.
This matches the practical inference scenario: features are computed
before market close on day ``t`` to issue a directional prediction for
day ``t+k``.

The cumulative-return naming follows the spec: ``r_cum5`` is the return
over the five-trading-day window ending at ``P_{t-1}`` (i.e. from
``P_{t-6}`` to ``P_{t-1}`` inclusive, a span of 5 days).

Warmup behaviour: rows for which insufficient prior history exists return
NaN. The longest-warmup feature is ``r_cum20`` (needs ``P_{t-21}``), so
the first 21 rows of any input are NaN throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

# Public column names (frozen). Downstream pipeline depends on these.
L1_FEATURE_COLS: tuple[str, ...] = ("r_lag1", "r_cum5", "r_cum10", "r_cum20")

# Lag specification: feature_name -> k, where r_cumK = log(P_{t-1} / P_{t-K-1})
# r_lag1 is the same as r_cum1 by this rule (k=1: log(P_{t-1}/P_{t-2})).
_LAG_SPEC: dict[str, int] = {
    "r_lag1":  1,
    "r_cum5":  5,
    "r_cum10": 10,
    "r_cum20": 20,
}


def compute_l1_features(
    price_df: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.DataFrame:
    """
    Compute the four L1 lag log-return features.

    Parameters
    ----------
    price_df : pd.DataFrame
        Daily price panel. Index should be a DatetimeIndex (per Session 1
        parquet convention). Must contain ``price_col``.
    price_col : str, default ``"adj_close"``
        Column to use. Per research_design.md §4.1, must be the
        adjusted close. Session 1 verified ``adj_close == close`` for
        TCB on vnstock, but the semantic-correct column is ``adj_close``.

    Returns
    -------
    pd.DataFrame
        Same index as ``price_df``. Four columns named in
        ``L1_FEATURE_COLS`` order. Rows where insufficient prior history
        exists are NaN (the first 21 rows for any input).

    Raises
    ------
    ValueError
        If ``price_col`` is not in ``price_df.columns``.
        If ``price_df[price_col]`` contains non-positive values (log
        return is undefined).
    """
    _validate(price_df, price_col)

    price = price_df[price_col]
    log_p = np.log(price)

    # log(P_{t-1} / P_{t-1-k}) = log_p.shift(1) - log_p.shift(1+k)
    # Anti-leak: every term uses shift(>=1), so day t itself is never read.
    result = pd.DataFrame(
        {
            name: log_p.shift(1) - log_p.shift(1 + k)
            for name, k in _LAG_SPEC.items()
        },
        index=price_df.index,
    )

    n_nan_r_cum20 = int(result["r_cum20"].isna().sum())
    log.info(
        "compute_l1_features: %d rows, %d warmup NaN in r_cum20 (expected %d)",
        len(result),
        n_nan_r_cum20,
        1 + _LAG_SPEC["r_cum20"],
    )

    return result


def _validate(price_df: pd.DataFrame, price_col: str) -> None:
    if price_col not in price_df.columns:
        raise ValueError(
            f"price_col '{price_col}' not in price_df.columns: "
            f"{price_df.columns.tolist()}"
        )
    price = price_df[price_col]
    # Allow NaN (will propagate to NaN in output), but disallow non-positive
    # actual values since log is undefined there.
    valid = price.dropna()
    if (valid <= 0).any():
        bad = valid[valid <= 0]
        raise ValueError(
            f"price_df['{price_col}'] contains {len(bad)} non-positive "
            "values; log returns are undefined. First few: "
            f"{bad.head().to_dict()}"
        )