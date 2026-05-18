"""
Loaders for manually-acquired CSV macro data.

Scope (Session 4 decision D1)
-----------------------------
- ``load_macro_monthly``  → ``data/raw/macro_monthly.csv``   (CPI + SBV rate)
- ``load_macro_quarterly`` → ``data/raw/macro_quarterly.csv`` (GDP)

L4 (``tcb_fundamentals.csv``) deferred to Session 5 once real TCB IR data is
acquired; loader API will mirror these two patterns.

Schemas (IMPLEMENTATION.md §4.1 + Session 4 decision D7)
--------------------------------------------------------
**macro_monthly.csv**::

    reference_period       str  "YYYY-MM"
    release_date           str  "YYYY-MM-DD"
    release_date_source    str  ∈ {"scraped", "fallback_14d"}   (D7)
    cpi_yoy_pct            float
    sbv_refinancing_rate_pct  float

**macro_quarterly.csv**::

    reference_quarter      str  "YYYY-Qn"
    release_date           str  "YYYY-MM-DD"
    release_date_source    str  ∈ {"scraped", "fallback_30d"}   (D7)
    gdp_yoy_pct            float

The ``release_date_source`` column is an audit trail: it tells a reviewer
whether the publication date in this row was scraped from the actual source
(GSO / SBV announcement) or fell back to the pre-registered conservative
convention from research_design.md §4.3 (CPI: ``period_end + 14 days``;
GDP: ``quarter_end + 30 days``).

Output guarantees
-----------------
The returned DataFrame is ready to pass to :func:`src.data.asof_join.asof_join`:
- ``release_date`` is ``datetime64[ns]`` (not str)
- Rows sorted ascending by ``release_date``
- No duplicate ``release_date``, no NaT
- No NaN in any numeric value column
- ``reference_period`` / ``reference_quarter`` unique
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema declarations
# ---------------------------------------------------------------------------

# Each schema lists the *expected* set of columns in order. Extra OR missing
# columns trigger a hard error: schema is contract, drift = bug.

MACRO_MONTHLY_COLUMNS: tuple[str, ...] = (
    "reference_period",
    "release_date",
    "release_date_source",
    "cpi_yoy_pct",
    "sbv_refinancing_rate_pct",
)
MACRO_MONTHLY_NUMERIC_COLS: tuple[str, ...] = (
    "cpi_yoy_pct",
    "sbv_refinancing_rate_pct",
)
MACRO_MONTHLY_VALID_SOURCES: frozenset[str] = frozenset({"scraped", "fallback_14d"})
_MONTHLY_REF_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

MACRO_QUARTERLY_COLUMNS: tuple[str, ...] = (
    "reference_quarter",
    "release_date",
    "release_date_source",
    "gdp_yoy_pct",
)
MACRO_QUARTERLY_NUMERIC_COLS: tuple[str, ...] = ("gdp_yoy_pct",)
MACRO_QUARTERLY_VALID_SOURCES: frozenset[str] = frozenset({"scraped", "fallback_30d"})
_QUARTERLY_REF_PATTERN = re.compile(r"^\d{4}-Q[1-4]$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_macro_monthly(path: str | Path) -> pd.DataFrame:
    """
    Load and validate ``macro_monthly.csv``.

    See module docstring for schema. Returns a DataFrame sorted by
    ``release_date`` (datetime64[ns]) with all schema invariants enforced.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        On any schema violation: missing or extra columns; unparseable or
        duplicate release_date; bad release_date_source value; bad
        reference_period format or duplicates; NaN in any numeric column.
    """
    return _load(
        path,
        expected_cols=MACRO_MONTHLY_COLUMNS,
        numeric_cols=MACRO_MONTHLY_NUMERIC_COLS,
        valid_sources=MACRO_MONTHLY_VALID_SOURCES,
        ref_period_col="reference_period",
        ref_period_pattern=_MONTHLY_REF_PATTERN,
        ref_period_format_msg="YYYY-MM (e.g. 2024-06)",
    )


def load_macro_quarterly(path: str | Path) -> pd.DataFrame:
    """
    Load and validate ``macro_quarterly.csv``.

    See module docstring for schema. Returns a DataFrame sorted by
    ``release_date`` (datetime64[ns]) with all schema invariants enforced.
    """
    return _load(
        path,
        expected_cols=MACRO_QUARTERLY_COLUMNS,
        numeric_cols=MACRO_QUARTERLY_NUMERIC_COLS,
        valid_sources=MACRO_QUARTERLY_VALID_SOURCES,
        ref_period_col="reference_quarter",
        ref_period_pattern=_QUARTERLY_REF_PATTERN,
        ref_period_format_msg="YYYY-Qn where n ∈ {1,2,3,4} (e.g. 2024-Q2)",
    )


# ---------------------------------------------------------------------------
# Internal generic loader
# ---------------------------------------------------------------------------


def _load(
    path: str | Path,
    expected_cols: tuple[str, ...],
    numeric_cols: tuple[str, ...],
    valid_sources: frozenset[str],
    ref_period_col: str,
    ref_period_pattern: re.Pattern[str],
    ref_period_format_msg: str,
) -> pd.DataFrame:
    """Generic schema-validating CSV loader; shared by monthly and quarterly."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)  # read all as str; coerce numerics below
    if len(df) == 0:
        raise ValueError(f"{path.name}: file is empty (no data rows)")

    _validate_columns(df, expected_cols, path.name)
    _coerce_numerics(df, numeric_cols, path.name)
    _parse_release_date(df, path.name)
    _validate_release_date_source(df, valid_sources, path.name)
    _validate_reference_period(
        df, ref_period_col, ref_period_pattern, ref_period_format_msg, path.name
    )
    _validate_uniqueness(df, ref_period_col, path.name)
    _validate_no_nan_in_numerics(df, numeric_cols, path.name)

    df = df.sort_values("release_date").reset_index(drop=True)

    log.info(
        "Loaded %s: %d rows, release_date range [%s, %s], "
        "release_date_source distribution = %s",
        path.name,
        len(df),
        df["release_date"].min().date(),
        df["release_date"].max().date(),
        df["release_date_source"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------


def _validate_columns(
    df: pd.DataFrame, expected: tuple[str, ...], filename: str
) -> None:
    actual = tuple(df.columns)
    if actual != expected:
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        raise ValueError(
            f"{filename}: column schema mismatch.\n"
            f"  Expected (in order): {list(expected)}\n"
            f"  Got     (in order): {list(actual)}\n"
            f"  Missing: {sorted(missing) or '(none)'}\n"
            f"  Extra:   {sorted(extra) or '(none)'}"
        )


def _coerce_numerics(
    df: pd.DataFrame, numeric_cols: tuple[str, ...], filename: str
) -> None:
    """Coerce string columns to float64. Bad values become NaN; the NaN
    check happens later in ``_validate_no_nan_in_numerics`` with a clearer
    error message naming the column."""
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _parse_release_date(df: pd.DataFrame, filename: str) -> None:
    parsed = pd.to_datetime(df["release_date"], format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        bad_idx = df.index[parsed.isna()].tolist()
        raw_vals = df.loc[bad_idx, "release_date"].tolist()
        raise ValueError(
            f"{filename}: {len(bad_idx)} rows have unparseable release_date "
            f"(expected format YYYY-MM-DD). "
            f"First offenders (row index → raw value): "
            f"{dict(zip(bad_idx[:5], raw_vals[:5]))}"
        )
    # Normalize to datetime64[ns]. pandas 2.x ``pd.to_datetime`` with
    # ``format=...`` returns datetime64[us] which causes downstream
    # ``pd.merge_asof`` (used by asof_join) to require matching dtype on
    # the other join key. Forcing [ns] here keeps the loader output
    # canonical and saves the caller a per-column astype.
    df["release_date"] = parsed.astype("datetime64[ns]")


def _validate_release_date_source(
    df: pd.DataFrame, valid: frozenset[str], filename: str
) -> None:
    bad = set(df["release_date_source"].unique()) - valid
    if bad:
        raise ValueError(
            f"{filename}: invalid release_date_source values: {sorted(bad)}. "
            f"Allowed: {sorted(valid)}"
        )


def _validate_reference_period(
    df: pd.DataFrame,
    col: str,
    pattern: re.Pattern[str],
    format_msg: str,
    filename: str,
) -> None:
    bad_mask = ~df[col].astype(str).str.match(pattern)
    if bad_mask.any():
        bad_idx = df.index[bad_mask].tolist()
        bad_vals = df.loc[bad_idx, col].tolist()
        raise ValueError(
            f"{filename}: {bad_mask.sum()} rows have invalid {col} format "
            f"(expected {format_msg}). "
            f"First offenders: {dict(zip(bad_idx[:5], bad_vals[:5]))}"
        )


def _validate_uniqueness(
    df: pd.DataFrame, ref_period_col: str, filename: str
) -> None:
    if df[ref_period_col].duplicated().any():
        dups = df[ref_period_col][df[ref_period_col].duplicated()].tolist()
        raise ValueError(
            f"{filename}: duplicate {ref_period_col} values: {dups[:5]}"
            f"{'…' if len(dups) > 5 else ''}"
        )
    if df["release_date"].duplicated().any():
        dups = df["release_date"][df["release_date"].duplicated()].tolist()
        raise ValueError(
            f"{filename}: duplicate release_date values: {dups[:5]}"
            f"{'…' if len(dups) > 5 else ''}. "
            "asof_join requires unique release_dates."
        )


def _validate_no_nan_in_numerics(
    df: pd.DataFrame, numeric_cols: tuple[str, ...], filename: str
) -> None:
    for col in numeric_cols:
        if df[col].isna().any():
            n = int(df[col].isna().sum())
            raise ValueError(
                f"{filename}: column '{col}' has {n} NaN/uncoercible values. "
                "Manual scrape must produce complete numeric values; if a "
                "data point is missing, omit the entire row instead of "
                "leaving a NaN."
            )