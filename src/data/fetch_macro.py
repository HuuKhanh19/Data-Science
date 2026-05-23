"""Channel B — Macro scrapers (v4).

Thay đổi từ v3:
- CPI: SCRAPE NSO trực tiếp (nso.gov.vn/en/cpi/) với pagination để lấy full
  history (~96 months). Bỏ hybrid manual+TheGlobalEconomy.
  Lý do: TheGlobalEconomy data mismatch GSO (3.61% vs 4.65% Mar 2026).
  NSO là official source.
- GDP: giữ VBMA CSV endpoint (v3, working).

Anti-leakage: dùng `Date of issue` (DD/MM/YYYY format VN) làm release_date thực
thay vì conservative `reference_period + 14d`.
"""
from __future__ import annotations
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

from .schema import CPI_SCHEMA, GDP_SCHEMA

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 30


def _now_vn() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(TZ_VN))


def _http_get(url: str, retries: int = 3, timeout: int = DEFAULT_TIMEOUT,
              verify_ssl: bool = True) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9,vi;q=0.8"}
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
            if r.status_code == 404:
                raise requests.HTTPError(f"404 Not Found: {url}")
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            last_err = e
            print(f"  [attempt {attempt+1}/{retries}] GET {url} failed: {e}")
    raise RuntimeError(f"All {retries} attempts failed for {url}: {last_err}")


def _http_get_bytes(url: str, retries: int = 3, timeout: int = DEFAULT_TIMEOUT,
                    verify_ssl: bool = True) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last_err = e
            print(f"  [attempt {attempt+1}/{retries}] GET {url} failed: {e}")
    raise RuntimeError(f"All {retries} attempts failed for {url}: {last_err}")


# ============================================================
# CPI - NSO scrape with pagination
# ============================================================

NSO_CPI_BASE = "https://www.nso.gov.vn/en/cpi/"
NSO_PAGE_LIMIT = 25

MONTH_NAMES_FULL = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Block: "Date of issue: DD/MM/YYYY ... Reference period: Month YYYY"
NSO_BLOCK_PATTERN = re.compile(
    r"Date of issue:\s*(\d{1,2})/(\d{1,2})/(\d{4})"
    r"\s*Reference period:\s*([A-Za-z]+)\s*(\d{4})",
    re.IGNORECASE | re.DOTALL,
)

# YoY: "(increased|decreased) by X.XX% compared to (the) same period last year"
NSO_YOY_PATTERN = re.compile(
    r"(increase|decrease)d?\s+by\s+(\d+(?:[\.,]\d+)?)\s*%\s+"
    r"compared\s+to\s+(?:the\s+)?same\s+period\s+last\s+year",
    re.IGNORECASE,
)


def _parse_nso_page(html: str) -> List[Dict[str, Any]]:
    """Extract {reference_period, value_pct, release_date} from one NSO page."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)

    matches = list(NSO_BLOCK_PATTERN.finditer(text))
    if not matches:
        return []

    releases: List[Dict[str, Any]] = []
    for i, m in enumerate(matches):
        try:
            d = int(m.group(1))
            mo = int(m.group(2))
            y = int(m.group(3))
            month_name = m.group(4).lower()
            ref_year = int(m.group(5))
        except (ValueError, IndexError):
            continue

        month_num = MONTH_NAMES_FULL.get(month_name)
        if month_num is None:
            continue

        try:
            # NSO date format: DD/MM/YYYY (Vietnamese)
            release_date = pd.Timestamp(year=y, month=mo, day=d)
        except (ValueError, OverflowError):
            continue

        ref_period_end = (
            pd.Timestamp(year=ref_year, month=month_num, day=1) + pd.offsets.MonthEnd(0)
        )

        body_start = matches[i - 1].end() if i > 0 else 0
        body_end = m.start()
        body = text[body_start:body_end]

        yoy_m = NSO_YOY_PATTERN.search(body)
        if not yoy_m:
            continue

        verb = yoy_m.group(1).lower()
        val_str = yoy_m.group(2).replace(",", ".")
        try:
            val = float(val_str)
            if verb == "decrease":
                val = -val
        except ValueError:
            continue

        releases.append({
            "reference_period": ref_period_end,
            "value_pct": val,
            "release_date": release_date,
        })

    return releases


def fetch_cpi(out_path: Path, start_date: str | None = None,
              end_date: str | None = None,
              debug_dir: Path | None = None,
              verify_ssl: bool = False) -> Dict[str, Any]:
    """Scrape NSO CPI press releases với pagination để lấy full history.

    Pagination: /en/cpi/ (page 1), /en/cpi/page/2/, ...
    Stops khi 404 hoặc page không có release mới.
    """
    print(f"  [cpi] Scraping NSO CPI archive (verify_ssl={verify_ssl})...")

    all_releases: List[Dict[str, Any]] = []
    seen_periods: set = set()
    pages_fetched = 0

    for page in range(1, NSO_PAGE_LIMIT + 1):
        url = NSO_CPI_BASE if page == 1 else f"{NSO_CPI_BASE}page/{page}/"
        try:
            html = _http_get(url, verify_ssl=verify_ssl, retries=2)
        except RuntimeError as e:
            print(f"  [cpi] page {page} fetch failed: {e}")
            break

        pages_fetched += 1

        if debug_dir and page == 1:
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir / "nso_cpi_page1.html").write_text(html, encoding="utf-8")

        page_releases = _parse_nso_page(html)
        new_releases = [r for r in page_releases
                        if r["reference_period"] not in seen_periods]

        if not new_releases:
            print(f"  [cpi] page {page}: 0 new releases - stopping")
            break

        all_releases.extend(new_releases)
        seen_periods.update(r["reference_period"] for r in new_releases)
        oldest = min(r["reference_period"] for r in new_releases).date()
        newest = max(r["reference_period"] for r in new_releases).date()
        print(f"  [cpi] page {page}: +{len(new_releases)} new "
              f"(range {oldest} -> {newest}); total: {len(all_releases)}")

    print(f"  [cpi] Done. Total: {len(all_releases)} releases across {pages_fetched} pages")

    if not all_releases:
        raise RuntimeError(
            "Không extract được CPI từ NSO. Inspect debug HTML."
        )

    df = pd.DataFrame(all_releases).sort_values("reference_period").reset_index(drop=True)
    df = df.drop_duplicates(subset=["reference_period"], keep="last").reset_index(drop=True)

    if start_date:
        df = df[df["reference_period"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["reference_period"] <= pd.Timestamp(end_date)]
    df = df.reset_index(drop=True)

    df["fetched_at"] = _now_vn()

    CPI_SCHEMA.validate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "status": "ok" if len(df) >= 50 else "warning",
        "rows": int(len(df)),
        "date_min": df["reference_period"].min().date().isoformat() if not df.empty else None,
        "date_max": df["reference_period"].max().date().isoformat() if not df.empty else None,
        "pages_fetched": pages_fetched,
        "output": str(out_path),
    }


# ============================================================
# GDP - VBMA CSV endpoint
# ============================================================

VBMA_GDP_CSV_URL = "https://vbma.org.vn/csv/markets/tables/en/gdp_danh_nghia_theo_quy.csv"


def _parse_vbma_quarter_label(label: str) -> pd.Timestamp | None:
    label = str(label).strip()
    m = re.match(r"Q(\d)[\s/-]+(\d{4})", label)
    if m:
        q, year = int(m.group(1)), int(m.group(2))
        if 1 <= q <= 4:
            return pd.Timestamp(year=year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)
    m = re.match(r"(\d{4})[\s_/-]*Q(\d)", label)
    if m:
        year, q = int(m.group(1)), int(m.group(2))
        if 1 <= q <= 4:
            return pd.Timestamp(year=year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)
    m = re.match(r"(\d{1,2})[/-](\d{4})", label)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if month in (3, 6, 9, 12):
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return None


def fetch_gdp(out_path: Path, start_date: str | None = None,
              end_date: str | None = None,
              verify_ssl: bool = False,
              debug_dir: Path | None = None) -> Dict[str, Any]:
    """Fetch GDP từ VBMA CSV endpoint trực tiếp.

    Endpoint: /csv/markets/tables/en/gdp_danh_nghia_theo_quy.csv (Nominal GDP by quarter).
    verify_ssl=False vì certifi không trust VBMA cert chain trên Windows.
    """
    print(f"  [gdp] GET CSV: {VBMA_GDP_CSV_URL}")
    csv_bytes = _http_get_bytes(VBMA_GDP_CSV_URL, verify_ssl=verify_ssl)
    print(f"  [gdp] Received {len(csv_bytes)} bytes")

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "gdp_raw.csv").write_bytes(csv_bytes)
        print(f"  [gdp] Dumped CSV to {debug_dir}/gdp_raw.csv")

    df_raw = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df_raw = pd.read_csv(io.BytesIO(csv_bytes), encoding=encoding)
            print(f"  [gdp] Parsed CSV with encoding={encoding}, shape={df_raw.shape}")
            print(f"  [gdp] Columns: {list(df_raw.columns)[:10]}"
                  + ("..." if len(df_raw.columns) > 10 else ""))
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    if df_raw is None or df_raw.empty:
        raise RuntimeError("Không parse được CSV từ VBMA.")

    rows: List[Tuple[pd.Timestamp, float]] = []

    # Layout A: first col = quarter labels
    first_col = df_raw.columns[0]
    first_col_quarters = [(_parse_vbma_quarter_label(v), v)
                          for v in df_raw[first_col].astype(str).tolist()]
    n_quarter_match = sum(1 for ts, _ in first_col_quarters if ts is not None)

    if n_quarter_match >= 5:
        print(f"  [gdp] Layout A: {n_quarter_match} quarter labels in first col")
        gdp_col = None
        for col in df_raw.columns[1:]:
            col_lower = str(col).lower()
            if "nominal" in col_lower and "gross" in col_lower:
                gdp_col = col
                break
        if gdp_col is None:
            numeric_cols = []
            for col in df_raw.columns[1:]:
                try:
                    vals = pd.to_numeric(df_raw[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        numeric_cols.append((col, vals.median()))
                except Exception:
                    continue
            if numeric_cols:
                gdp_col = max(numeric_cols, key=lambda x: x[1])[0]
                print(f"  [gdp] No explicit GDP col, picked '{gdp_col}' by max median")
        if gdp_col is not None:
            for (ts, _), val in zip(first_col_quarters, df_raw[gdp_col]):
                if ts is None:
                    continue
                try:
                    v = float(str(val).replace(",", ""))
                    rows.append((ts, v))
                except (ValueError, TypeError):
                    continue
    else:
        # Layout B: cols are quarters
        quarter_cols: List[Tuple[str, pd.Timestamp]] = []
        for col in df_raw.columns:
            ts = _parse_vbma_quarter_label(str(col))
            if ts is not None:
                quarter_cols.append((col, ts))
        print(f"  [gdp] Layout B: {len(quarter_cols)} quarter cols")
        if quarter_cols:
            label_col = df_raw.columns[0]
            for _, row in df_raw.iterrows():
                row_label = str(row[label_col]).lower()
                if "nominal" in row_label and "gross domestic product" in row_label:
                    for col, ts in quarter_cols:
                        try:
                            v = float(str(row[col]).replace(",", ""))
                            rows.append((ts, v))
                        except (ValueError, TypeError):
                            continue
                    break

    print(f"  [gdp] Extracted {len(rows)} (quarter, GDP) pairs")

    if not rows:
        raise RuntimeError(
            "Không extract được GDP. Inspect debug CSV."
        )

    all_rows: Dict[pd.Timestamp, float] = {}
    for ts, val in rows:
        all_rows[ts] = val

    df = pd.DataFrame(
        [{"reference_period": k, "nominal_gdp_vnd_bil": v} for k, v in sorted(all_rows.items())]
    )

    if start_date:
        df = df[df["reference_period"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["reference_period"] <= pd.Timestamp(end_date)]
    df = df.reset_index(drop=True)

    df["release_date"] = df["reference_period"] + pd.Timedelta(days=30)
    df["fetched_at"] = _now_vn()

    GDP_SCHEMA.validate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "status": "ok",
        "rows": int(len(df)),
        "date_min": df["reference_period"].min().date().isoformat() if not df.empty else None,
        "date_max": df["reference_period"].max().date().isoformat() if not df.empty else None,
        "output": str(out_path),
    }