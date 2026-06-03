"""Thu thập GDP danh nghĩa theo quý từ VBMA (web scrape TSV).

Khác CPI (API số sạch): đây là cào một file web bẩn — tên .csv nhưng thực ra
TSV, encoding UTF-16 LE. Phải dò encoding + mò đúng hàng "Nominal Gross Domestic
Product" trong layout B (cột = nhãn quý, hàng đầu = tổng GDP danh nghĩa).

  - release_date = quarter_end + 30 ngày (quy ước bảo thủ; VBMA không cung cấp
    ngày công bố).
  - verify_ssl=False mặc định (VBMA lỗi cert).
"""
from __future__ import annotations
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from ._common import _http_get_bytes, _now_vn
from .schema import GDP_SCHEMA

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