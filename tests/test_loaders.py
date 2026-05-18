"""
Tests for ``src.data.loaders``.

Strategy
--------
- Schema validation per path: missing/extra columns, bad date format, bad
  release_date_source enum, bad reference_period format, duplicates,
  NaN in numerics, file errors.
- Output guarantee tests: sorted by release_date, dtype correct,
  release_date is datetime64[ns] (ready for asof_join).
- Most paths tested through ``load_macro_monthly`` (shared internal
  ``_load``); ``load_macro_quarterly`` has thinner test set focused on
  schema differences (different columns, different enum, Q-pattern ref).

All tests use ``tmp_path`` fixture; no real ``data/raw/`` files are
touched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loaders import (
    MACRO_MONTHLY_COLUMNS,
    MACRO_QUARTERLY_COLUMNS,
    load_macro_monthly,
    load_macro_quarterly,
)


# ---------------------------------------------------------------------------
# CSV templates
# ---------------------------------------------------------------------------


def _valid_monthly_csv() -> str:
    return (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,2024-02-14,scraped,3.5,4.5\n"
        "2024-02,2024-03-14,scraped,3.6,4.5\n"
        "2024-03,2024-04-14,fallback_14d,3.7,4.0\n"
    )


def _valid_quarterly_csv() -> str:
    return (
        "reference_quarter,release_date,release_date_source,gdp_yoy_pct\n"
        "2024-Q1,2024-04-30,scraped,5.6\n"
        "2024-Q2,2024-07-29,scraped,6.9\n"
        "2024-Q3,2024-10-30,fallback_30d,7.2\n"
    )


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# 1. Happy path
# ===========================================================================


def test_load_macro_monthly_valid(tmp_path):
    path = _write(tmp_path, "macro_monthly.csv", _valid_monthly_csv())
    df = load_macro_monthly(path)

    assert len(df) == 3
    assert tuple(df.columns) == MACRO_MONTHLY_COLUMNS
    assert df["release_date"].dtype == "datetime64[ns]"
    assert df["cpi_yoy_pct"].dtype == "float64"
    assert df["sbv_refinancing_rate_pct"].dtype == "float64"


def test_load_macro_monthly_is_sorted_by_release_date(tmp_path):
    """Even if CSV is out-of-order, loader returns sorted output."""
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-03,2024-04-14,scraped,3.7,4.0\n"  # out of order
        "2024-01,2024-02-14,scraped,3.5,4.5\n"
        "2024-02,2024-03-14,scraped,3.6,4.5\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    df = load_macro_monthly(path)
    assert df["release_date"].is_monotonic_increasing
    assert df["reference_period"].tolist() == ["2024-01", "2024-02", "2024-03"]


def test_load_macro_monthly_releases_dates_unique_and_no_nat(tmp_path):
    """Output guarantees: no NaT and no duplicates in release_date."""
    path = _write(tmp_path, "macro_monthly.csv", _valid_monthly_csv())
    df = load_macro_monthly(path)
    assert not df["release_date"].isna().any()
    assert df["release_date"].is_unique


def test_load_macro_quarterly_valid(tmp_path):
    path = _write(tmp_path, "macro_quarterly.csv", _valid_quarterly_csv())
    df = load_macro_quarterly(path)

    assert len(df) == 3
    assert tuple(df.columns) == MACRO_QUARTERLY_COLUMNS
    assert df["release_date"].dtype == "datetime64[ns]"
    assert df["gdp_yoy_pct"].dtype == "float64"


# ===========================================================================
# 2. Schema column violations
# ===========================================================================


def test_raises_on_missing_column(tmp_path):
    csv = (
        "reference_period,release_date,release_date_source,cpi_yoy_pct\n"  # missing sbv
        "2024-01,2024-02-14,scraped,3.5\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="column schema mismatch"):
        load_macro_monthly(path)


def test_raises_on_extra_column(tmp_path):
    csv = (
        "reference_period,release_date,release_date_source,cpi_yoy_pct,"
        "sbv_refinancing_rate_pct,extra_col\n"
        "2024-01,2024-02-14,scraped,3.5,4.5,XXX\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="column schema mismatch"):
        load_macro_monthly(path)


def test_raises_on_column_order_mismatch(tmp_path):
    """Strict order check — re-ordered columns reject."""
    csv = (
        "release_date,reference_period,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-02-14,2024-01,scraped,3.5,4.5\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="column schema mismatch"):
        load_macro_monthly(path)


# ===========================================================================
# 3. release_date validation
# ===========================================================================


def test_raises_on_bad_release_date_format(tmp_path):
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,Feb 14 2024,scraped,3.5,4.5\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="unparseable release_date"):
        load_macro_monthly(path)


def test_raises_on_duplicate_release_dates(tmp_path):
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,2024-02-14,scraped,3.5,4.5\n"
        "2024-02,2024-02-14,scraped,3.6,4.5\n"  # same release_date
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="duplicate release_date"):
        load_macro_monthly(path)


# ===========================================================================
# 4. release_date_source enum
# ===========================================================================


def test_raises_on_invalid_release_date_source_monthly(tmp_path):
    """fallback_30d is for quarterly, not monthly → reject."""
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,2024-02-14,fallback_30d,3.5,4.5\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="invalid release_date_source"):
        load_macro_monthly(path)


def test_raises_on_invalid_release_date_source_quarterly(tmp_path):
    """fallback_14d is for monthly, not quarterly → reject."""
    csv = (
        "reference_quarter,release_date,release_date_source,gdp_yoy_pct\n"
        "2024-Q1,2024-04-30,fallback_14d,5.6\n"
    )
    path = _write(tmp_path, "macro_quarterly.csv", csv)
    with pytest.raises(ValueError, match="invalid release_date_source"):
        load_macro_quarterly(path)


def test_raises_on_typo_release_date_source(tmp_path):
    """e.g. 'scrapped' → reject."""
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,2024-02-14,scrapped,3.5,4.5\n"  # typo
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="invalid release_date_source"):
        load_macro_monthly(path)


# ===========================================================================
# 5. reference_period / reference_quarter format
# ===========================================================================


def test_raises_on_bad_reference_period_format(tmp_path):
    """'2024/01' not allowed (must be 'YYYY-MM')."""
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024/01,2024-02-14,scraped,3.5,4.5\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="invalid reference_period format"):
        load_macro_monthly(path)


def test_raises_on_bad_reference_quarter_format(tmp_path):
    """'2024-Q5' not allowed (n must be 1-4)."""
    csv = (
        "reference_quarter,release_date,release_date_source,gdp_yoy_pct\n"
        "2024-Q5,2024-04-30,scraped,5.6\n"
    )
    path = _write(tmp_path, "macro_quarterly.csv", csv)
    with pytest.raises(ValueError, match="invalid reference_quarter format"):
        load_macro_quarterly(path)


def test_raises_on_duplicate_reference_period(tmp_path):
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,2024-02-14,scraped,3.5,4.5\n"
        "2024-01,2024-03-14,scraped,3.6,4.5\n"  # same ref_period
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="duplicate reference_period"):
        load_macro_monthly(path)


# ===========================================================================
# 6. NaN in numerics
# ===========================================================================


def test_raises_on_nan_in_cpi(tmp_path):
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,2024-02-14,scraped,,4.5\n"  # empty cpi
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="NaN/uncoercible values"):
        load_macro_monthly(path)


def test_raises_on_non_numeric_in_cpi(tmp_path):
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,2024-02-14,scraped,N/A,4.5\n"  # string in numeric col
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="NaN/uncoercible values"):
        load_macro_monthly(path)


# ===========================================================================
# 7. File-level errors
# ===========================================================================


def test_raises_on_missing_file(tmp_path):
    nonexistent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_macro_monthly(nonexistent)


def test_raises_on_empty_file(tmp_path):
    """Header-only CSV (no data rows) → reject."""
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    with pytest.raises(ValueError, match="empty"):
        load_macro_monthly(path)


# ===========================================================================
# 8. Mixed scraped + fallback (real-world common case)
# ===========================================================================


def test_load_macro_monthly_mixed_sources(tmp_path):
    """Some rows scraped, some fallback — both are valid."""
    csv = (
        "reference_period,release_date,release_date_source,"
        "cpi_yoy_pct,sbv_refinancing_rate_pct\n"
        "2024-01,2024-02-14,fallback_14d,3.5,4.5\n"
        "2024-02,2024-03-13,scraped,3.6,4.5\n"
        "2024-03,2024-04-14,fallback_14d,3.7,4.0\n"
    )
    path = _write(tmp_path, "macro_monthly.csv", csv)
    df = load_macro_monthly(path)
    counts = df["release_date_source"].value_counts().to_dict()
    assert counts == {"fallback_14d": 2, "scraped": 1}