"""Thu thập chỉ số CPI Việt Nam (all-items, theo tháng) từ IMF Data Portal (SDMX).

Nguồn: IMF SDMX, dataset CPI, key VNM.CPI._T.IX.M (Việt Nam, all-items _T,
dạng index IX, tần suất tháng M), gốc 2024=100. Gọi qua sdmx1.

Vì sao IMF thay vì scrape NSO: press-release NSO không đồng nhất (~49% tháng
thiếu MoM/point-YoY), không đảm bảo full coverage để tự động hóa. IMF cho chuỗi
SỐ liền mạch 2017→nay, tái lập được; YoY tự tính (idx_t/idx_{t-12}-1) khớp đúng
YoY chính thức của GSO (vd 2026-03: 106.83/102.082-1 = 4.65%).

  - Step 1 chỉ collect chỉ số; cpi_yoy tính ở bước feature.
  - release_date: nguồn số không có ngày công bố → quy ước = reference_period
    (month-end) + 6 ngày (NSO thực tế ra CPI tháng M vào ~mùng 3-6 tháng M+1 →
    an toàn leakage).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ._common import _now_vn
from .schema import CPI_SCHEMA

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