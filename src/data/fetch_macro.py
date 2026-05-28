"""Channel B - Macro sources (v6).

Thay đổi từ v5:
- CPI: pivot từ scrape NSO raw text → **IMF Data Portal, chỉ số CPI all-items theo
  tháng** (SDMX, key `VNM.CPI._T.IX.M`, gốc 2024=100), qua thư viện `sdmx1`.
  + Lý do: NSO press-release text không đồng nhất (≈49% tháng thiếu MoM/point-YoY),
    không đảm bảo full coverage để tự động hóa. IMF cho chuỗi SỐ liền mạch
    2017→nay, tái lập được; YoY tự tính (idx_t/idx_{t-12}-1) khớp đúng YoY chính
    thức của GSO (vd 2026-03: 106.83/102.082-1 = 4.65%).
  + Step 1 chỉ collect chỉ số; `cpi_yoy` tính ở Step 2.
  + release_date: nguồn số không có ngày công bố → quy ước = reference_period
    (month-end) + 6 ngày (NSO thực tế ra CPI tháng M vào ~mùng 3-6 tháng M+1 →
    an toàn leakage).
- GDP: giữ nguyên v5 (VBMA TSV, Layout B).
- Đã bỏ toàn bộ code scrape NSO (BeautifulSoup, pagination, _http_get text...).
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

from .schema import CPI_SCHEMA, GDP_SCHEMA

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 30


def _now_vn() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(TZ_VN))


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
# CPI - IMF Data Portal (SDMX), chỉ số all-items theo tháng
# ============================================================

IMF_CPI_DATASET = "CPI"
IMF_CPI_KEY = "VNM.CPI._T.IX.M"  # Vietnam, all-items (_T), index (IX), monthly (M)


def _imf_month_to_end(period: str) -> pd.Timestamp:
    """'2017-M01' -> Timestamp(2017-01-31). IMF dùng định dạng 'YYYY-Mmm'."""
    y, m = period.split("-M")
    return pd.Timestamp(year=int(y), month=int(m), day=1) + pd.offsets.MonthEnd(0)


def fetch_cpi(out_path: Path, start_date: str | None = None,
              end_date: str | None = None,
              prehistory_months: int = 15) -> Dict[str, Any]:
    """Lấy chỉ số CPI Việt Nam (all-items, monthly) từ IMF Data Portal.

    - prehistory_months=15: lùi đủ ≥12 tháng trước phiên đầu panel để `cpi_yoy`
      (cần idx lùi 12 tháng) định nghĩa được từ tháng CPI as-of đầu tiên.
    - start_date=None: lấy từ 2015-01 (toàn bộ vùng cần thiết).
    """
    import sdmx  # thư viện: pip install sdmx1 (import name = 'sdmx')

    # Tính cutoff dưới + startPeriod cho IMF (định dạng 'YYYY-MM')
    if start_date is not None:
        cutoff = pd.Timestamp(start_date) - pd.DateOffset(months=prehistory_months)
    else:
        cutoff = pd.Timestamp("2015-01-01")
    start_period = f"{cutoff.year:04d}-{cutoff.month:02d}"

    print(f"  [cpi] IMF SDMX {IMF_CPI_DATASET}/{IMF_CPI_KEY} từ {start_period} ...")
    imf = sdmx.Client("IMF_DATA")
    msg = imf.data(IMF_CPI_DATASET, key=IMF_CPI_KEY,
                   params={"startPeriod": start_period})
    s = sdmx.to_pandas(msg)
    if isinstance(s, pd.DataFrame):  # phòng version trả DataFrame
        s = s.squeeze("columns")

    periods = s.index.get_level_values("TIME_PERIOD")
    df = pd.DataFrame({
        "reference_period": [_imf_month_to_end(str(p)) for p in periods],
        "cpi_index": pd.to_numeric(pd.Series(s.to_numpy()), errors="coerce").to_numpy(),
    })
    df = (df.dropna(subset=["cpi_index"])
            .drop_duplicates(subset=["reference_period"], keep="last")
            .sort_values("reference_period")
            .reset_index(drop=True))
    print(f"  [cpi] IMF trả {len(df)} tháng "
          f"({df['reference_period'].min().date()} -> {df['reference_period'].max().date()})")

    # Filter [cutoff, end_date]
    df = df[df["reference_period"] >= cutoff]
    if end_date is not None:
        df = df[df["reference_period"] <= pd.Timestamp(end_date)]
    df = df.reset_index(drop=True)

    # release_date: quy ước = month-end + 6 ngày (~mùng 6 tháng sau)
    df["release_date"] = df["reference_period"] + pd.Timedelta(days=6)
    df["fetched_at"] = _now_vn()

    CPI_SCHEMA.validate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "status": "ok" if len(df) >= 50 else "warning",
        "rows": int(len(df)),
        "date_min": df["reference_period"].min().date().isoformat() if not df.empty else None,
        "date_max": df["reference_period"].max().date().isoformat() if not df.empty else None,
        "output": str(out_path),
    }


# ============================================================
# GDP - VBMA TSV endpoint  (giữ nguyên v5)
# ============================================================

VBMA_GDP_CSV_URL = "https://vbma.org.vn/csv/markets/tables/en/gdp_danh_nghia_theo_quy.csv"

QUARTER_LABEL_RE = re.compile(r"Q\s*(\d)\s+(\d{4})", re.IGNORECASE)


def _parse_vbma_quarter_label(label: str) -> pd.Timestamp | None:
    """Parse VBMA quarter label e.g. 'Q1 2015' or 'Q4 2025' -> quarter-end Timestamp."""
    if label is None:
        return None
    label = str(label).strip()
    m = QUARTER_LABEL_RE.match(label)
    if m:
        q, year = int(m.group(1)), int(m.group(2))
        if 1 <= q <= 4:
            return pd.Timestamp(year=year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)
    return None


def _parse_vbma_numeric(val: Any) -> float | None:
    """Parse VBMA value like '"809,613"' -> 809613.0."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().strip('"').strip("'").replace(",", "").replace(" ", "")
    if not s or s.lower() in ("nan", "none", "null", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_gdp(out_path: Path, start_date: str | None = None,
              end_date: str | None = None,
              verify_ssl: bool = False) -> Dict[str, Any]:
    """Fetch GDP từ VBMA TSV endpoint.

    Format thực (verified từ Get-Content):
        \\tQ1 2015\\tQ2 2015\\t...\\tQ1 2026          (header row)
        Nominal Gross Domestic Product\\t"809,613"\\t"970,388"\\t...   (data row 1)
        "Agriculture, Forestry and Fishery "\\t"99,978"\\t...           (data row 2+)

    Layout B: cols = quarter labels, row đầu = total nominal GDP.
    """
    print(f"  [gdp] GET TSV: {VBMA_GDP_CSV_URL}")
    csv_bytes = _http_get_bytes(VBMA_GDP_CSV_URL, verify_ssl=verify_ssl)
    print(f"  [gdp] Received {len(csv_bytes)} bytes")

    bom_hint = ""
    if csv_bytes.startswith(b"\xff\xfe"):
        bom_hint = " [BOM detected: UTF-16 LE]"
    elif csv_bytes.startswith(b"\xfe\xff"):
        bom_hint = " [BOM detected: UTF-16 BE]"
    elif csv_bytes.startswith(b"\xef\xbb\xbf"):
        bom_hint = " [BOM detected: UTF-8]"
    print(f"  [gdp] First 4 bytes: {csv_bytes[:4]!r}{bom_hint}")

    df_raw = None
    last_err = None
    for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df_raw = pd.read_csv(
                io.BytesIO(csv_bytes),
                sep="\t",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                skip_blank_lines=True,
            )
            if any(QUARTER_LABEL_RE.match(str(c).strip()) for c in df_raw.columns):
                print(f"  [gdp] Parsed TSV with encoding={encoding}, shape={df_raw.shape}")
                print(f"  [gdp] First 5 columns: {list(df_raw.columns)[:5]}")
                break
            else:
                print(f"  [gdp] encoding={encoding}: parsed but no quarter cols (garbled) → next")
                df_raw = None
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            print(f"  [gdp] encoding={encoding}: {type(e).__name__} → next")
            continue

    if df_raw is None or df_raw.empty:
        raise RuntimeError(f"Không parse được TSV từ VBMA. Last error: {last_err}")

    quarter_cols: List[Tuple[str, pd.Timestamp]] = []
    for col in df_raw.columns:
        ts = _parse_vbma_quarter_label(str(col))
        if ts is not None:
            quarter_cols.append((col, ts))

    print(f"  [gdp] Found {len(quarter_cols)} quarter columns "
          f"(range {quarter_cols[0][1].date() if quarter_cols else None} -> "
          f"{quarter_cols[-1][1].date() if quarter_cols else None})")

    if len(quarter_cols) < 5:
        raise RuntimeError(
            f"Không đủ quarter columns (found {len(quarter_cols)}). "
            f"Columns: {list(df_raw.columns)[:20]}"
        )

    label_col = df_raw.columns[0]

    target_row = None
    for _, row in df_raw.iterrows():
        row_label = str(row[label_col]).strip().lower().strip('"')
        if "nominal" in row_label and "gross domestic product" in row_label:
            target_row = row
            print(f"  [gdp] Matched label row: '{row[label_col]}'")
            break

    if target_row is None:
        print(f"  [gdp] WARNING: không match 'Nominal Gross Domestic Product'. "
              f"Available labels (first 10): "
              f"{[str(r).strip() for r in df_raw[label_col].head(10).tolist()]}")
        raise RuntimeError("Không tìm thấy row 'Nominal Gross Domestic Product'.")

    rows: List[Tuple[pd.Timestamp, float]] = []
    for col, ts in quarter_cols:
        val = _parse_vbma_numeric(target_row[col])
        if val is not None:
            rows.append((ts, val))

    print(f"  [gdp] Extracted {len(rows)} (quarter, value) pairs")

    if not rows:
        raise RuntimeError("Parsed quarters nhưng tất cả values đều null.")

    df = pd.DataFrame(
        [{"reference_period": k, "nominal_gdp_vnd_bil": v}
         for k, v in sorted(set(rows), key=lambda x: x[0])]
    )
    df = df.drop_duplicates(subset=["reference_period"], keep="last").reset_index(drop=True)

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