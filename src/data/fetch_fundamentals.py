"""Channel C — TCB Bank Fundamentals (L4) v5.

BREAKTHROUGH: Bypass public API limit bằng cách gọi thẳng private method
`_get_financial_report(get_all=True, limit=100)` của VCIFinance provider.

Lý do: public `balance_sheet()` của vnstock.explorer.vci.financial.Finance
KHÔNG forward `get_all=True` từ constructor xuống `_get_financial_report` —
nên mặc định chỉ trả 4 quarters. Gọi private method trực tiếp với
`get_all=True, limit=100` để unlock full history.

Mapping đã fix theo item_ids thực:
- credit_balance → loans_and_advances_to_customers_net
- npl_ratio_pct → npl (just 'npl', not 'npl_ratio')
- nim_pct → net_interest_margin
- interest_earning_assets: không có trong vnstock balance_sheet, để NaN
"""
from __future__ import annotations
import io
import re
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .schema import TCB_FUNDAMENTALS_SCHEMA

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
QUARTER_COL_PATTERN = re.compile(r"^(\d{4})[-_]?Q(\d)$")


def _now_vn() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(TZ_VN))


# Mapping: (statement_type, [item_ids theo priority])
FIELD_MAPPING: Dict[str, tuple] = {
    "total_assets_vnd_bil": ("balance_sheet", ["total_assets"]),
    "equity_vnd_bil": ("balance_sheet", ["owners_equity", "shareholders_equity"]),
    "credit_balance_vnd_bil": ("balance_sheet", [
        "loans_and_advances_to_customers_net",
        "loans_and_advances_to_customers",
    ]),
    "net_interest_income_vnd_bil": ("income_statement", ["net_interest_income"]),
    "eps_ttm": ("income_statement", ["eps_basic_vnd", "eps_basic", "eps_diluted_vnd"]),
    "npl_ratio_pct": ("ratio", ["npl"]),
    "nim_pct": ("ratio", ["net_interest_margin", "nim"]),
}


def _find_item_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    if "item_id" not in df.columns:
        return None
    iid_str = df["item_id"].astype(str)
    iid_lower = iid_str.str.lower()
    for cand in candidates:
        mask = iid_str == cand
        if mask.any():
            return df.loc[mask].iloc[0]
        mask = iid_lower == cand.lower()
        if mask.any():
            return df.loc[mask].iloc[0]
    return None


def _quarter_col_to_period_end(col: str) -> Optional[pd.Timestamp]:
    m = QUARTER_COL_PATTERN.match(str(col).strip())
    if not m:
        return None
    year = int(m.group(1))
    q = int(m.group(2))
    if not (1 <= q <= 4):
        return None
    return pd.Timestamp(year=year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)


def _extract_quarter_series(row: pd.Series) -> Dict[pd.Timestamp, float]:
    out: Dict[pd.Timestamp, float] = {}
    for col, val in row.items():
        period = _quarter_col_to_period_end(col)
        if period is None:
            continue
        try:
            v = float(val)
            if pd.notna(v):
                out[period] = v
        except (ValueError, TypeError):
            continue
    return out


def _normalize_to_billion(values: Dict[pd.Timestamp, float],
                         field: str) -> Dict[pd.Timestamp, float]:
    if not values:
        return values
    if field.endswith("_pct") or "eps" in field:
        return values
    med = pd.Series(list(values.values())).median()
    if abs(med) > 1e9:
        return {k: v / 1e9 for k, v in values.items()}
    return values


def _fetch_full_history(symbol: str = "TCB") -> Dict[str, pd.DataFrame]:
    """Call private _get_financial_report(get_all=True, limit=100) to unlock full history.

    Fallback: nếu private call fail, dùng public method (chỉ 4 quarters).
    """
    results: Dict[str, pd.DataFrame] = {}

    # Suppress vnai banner spam
    quiet = io.StringIO()

    try:
        from vnstock.explorer.vci.financial import Finance as VCIFinance
        print("  Using vnstock.explorer.vci.financial direct (bypass public 4-quarter limit)")

        with redirect_stdout(quiet):
            v = VCIFinance(symbol=symbol, period="quarter", show_log=False)

        for stmt_type in ["balance_sheet", "ratio", "income_statement"]:
            df = None
            # Try 1: private _get_financial_report with get_all=True + limit
            try:
                with redirect_stdout(quiet):
                    df = v._get_financial_report(
                        stmt_type,
                        period="quarter",
                        lang="en",
                        get_all=True,
                        limit=100,
                        dropna=True,
                    )
                if df is not None and not df.empty:
                    qc = [c for c in df.columns if QUARTER_COL_PATTERN.match(str(c).strip())]
                    print(f"    {stmt_type}: {df.shape[0]} items x {len(qc)} quarters "
                          f"(via _get_financial_report get_all=True)")
                    results[stmt_type] = df
                    continue
            except (TypeError, AttributeError) as e:
                print(f"    {stmt_type}: private call failed ({e}), fallback to public")
            except Exception as e:
                print(f"    {stmt_type}: private call error ({type(e).__name__}: {e}), "
                      f"fallback to public")

            # Try 2: public method (fallback, 4 quarters only)
            try:
                method = getattr(v, stmt_type)
                with redirect_stdout(quiet):
                    df = method(period="quarter", lang="en")
                if df is not None and not df.empty:
                    qc = [c for c in df.columns if QUARTER_COL_PATTERN.match(str(c).strip())]
                    print(f"    {stmt_type}: {df.shape[0]} items x {len(qc)} quarters (public fallback)")
                    results[stmt_type] = df
            except Exception as e:
                print(f"    {stmt_type}: also failed public fallback: {e}")
    except ImportError as e:
        print(f"  Cannot import VCIFinance directly: {e}")
        print(f"  Falling back to vnstock.api.financial.Finance (4-quarter limit)")
        from vnstock.api.financial import Finance
        with redirect_stdout(quiet):
            fin = Finance(symbol=symbol, period="quarter", source="VCI")
        for stmt_type in ["balance_sheet", "ratio", "income_statement"]:
            try:
                with redirect_stdout(quiet):
                    df = getattr(fin, stmt_type)()
                if df is not None and not df.empty:
                    qc = [c for c in df.columns if QUARTER_COL_PATTERN.match(str(c).strip())]
                    print(f"    {stmt_type}: {df.shape[0]} items x {len(qc)} quarters")
                    results[stmt_type] = df
            except Exception as e:
                print(f"    {stmt_type}: FAILED - {e}")

    return results


def fetch_tcb_fundamentals(out_path: Path,
                           start_date: str | None = None,
                           end_date: str | None = None,
                           verbose: bool = True) -> Dict[str, Any]:
    """Fetch TCB quarterly fundamentals.

    v5: bypass vnstock public 4-quarter limit qua private _get_financial_report
    (get_all=True, limit=100).
    """
    print("  Calling vnstock VCI Finance API...")
    statements = _fetch_full_history("TCB")

    field_series: Dict[str, Dict[pd.Timestamp, float]] = {}
    field_meta: Dict[str, str] = {}

    for field, (stmt_type, candidates) in FIELD_MAPPING.items():
        df = statements.get(stmt_type)
        if df is None or df.empty:
            print(f"  [{field}] no {stmt_type} data")
            continue
        row = _find_item_row(df, candidates)
        if row is None:
            print(f"  [{field}] NOT FOUND. Candidates: {candidates}")
            continue
        values = _extract_quarter_series(row)
        item_id = str(row.get("item_id", ""))
        values = _normalize_to_billion(values, field)
        if values:
            field_series[field] = values
            field_meta[field] = f"{stmt_type}/{item_id}"
            print(f"  [{field}] {len(values)} quarters from {stmt_type}/{item_id}")

    all_periods: set = set()
    for series in field_series.values():
        all_periods.update(series.keys())
    if not all_periods:
        raise RuntimeError("No fundamentals data extracted.")

    df = pd.DataFrame(sorted(all_periods), columns=["reference_period"])
    for field in FIELD_MAPPING:
        df[field] = df["reference_period"].map(field_series.get(field, {}))
    df["interest_earning_assets_vnd_bil"] = pd.NA  # Not available in vnstock

    if start_date:
        df = df[df["reference_period"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["reference_period"] <= pd.Timestamp(end_date)]
    df = df.reset_index(drop=True)

    df["release_date"] = df["reference_period"] + pd.Timedelta(days=45)
    df["fetched_at"] = _now_vn()

    df = df[[
        "reference_period", "release_date",
        "total_assets_vnd_bil", "equity_vnd_bil",
        "net_interest_income_vnd_bil", "interest_earning_assets_vnd_bil",
        "npl_ratio_pct", "credit_balance_vnd_bil", "eps_ttm", "nim_pct",
        "fetched_at",
    ]]

    TCB_FUNDAMENTALS_SCHEMA.validate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    null_summary = {
        col: int(df[col].isna().sum()) for col in df.columns
        if col not in ("reference_period", "release_date", "fetched_at")
    }

    return {
        "status": "ok",
        "rows": int(len(df)),
        "date_min": df["reference_period"].min().date().isoformat() if not df.empty else None,
        "date_max": df["reference_period"].max().date().isoformat() if not df.empty else None,
        "null_counts": null_summary,
        "field_sources": field_meta,
        "output": str(out_path),
    }