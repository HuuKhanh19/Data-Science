"""Validators chung cho raw data outputs.

- check_monotonic_dates: dates phải strictly increasing
- check_hose_calendar_gap: cảnh báo gap > 12 ngày (WARN), error > 15 ngày
- check_abnormal_returns: cảnh báo |log return| > 15% (dấu hiệu adjustment chưa apply)
- canonicalize_price_unit: assert median close trong expected range (nghìn VND)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    name: str
    n_rows: int
    date_range: Tuple[pd.Timestamp, pd.Timestamp]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        d0, d1 = self.date_range
        d0s = d0.date().isoformat() if pd.notna(d0) else "NaT"
        d1s = d1.date().isoformat() if pd.notna(d1) else "NaT"
        head = f"[{self.name}] rows={self.n_rows} range=({d0s} → {d1s}) "
        head += f"warnings={len(self.warnings)} errors={len(self.errors)}"
        if self.warnings:
            head += "\n  WARN:" + "\n  WARN: ".join([""] + self.warnings[:10])
            if len(self.warnings) > 10:
                head += f"\n  ... ({len(self.warnings) - 10} more)"
        if self.errors:
            head += "\n  ERR:" + "\n  ERR: ".join([""] + self.errors)
        return head


def check_monotonic_dates(df: pd.DataFrame, name: str, date_col: str = "date") -> ValidationReport:
    warnings: List[str] = []
    errors: List[str] = []
    if df.empty:
        return ValidationReport(name, 0, (pd.NaT, pd.NaT), warnings, errors)

    s = df[date_col]
    if not s.is_monotonic_increasing:
        errors.append(f"Column '{date_col}' not strictly increasing")
    n_dup = int(s.duplicated().sum())
    if n_dup > 0:
        errors.append(f"Column '{date_col}' có {n_dup} duplicate dates")

    return ValidationReport(name, len(df), (s.iloc[0], s.iloc[-1]), warnings, errors)


def check_hose_calendar_gap(
    df: pd.DataFrame, name: str,
    date_col: str = "date",
    warn_days: int = 12, error_days: int = 15,
) -> ValidationReport:
    """Gap quá lớn giữa các ngày liên tiếp có thể chỉ ra missing data.

    Lưu ý: Tết âm lịch hợp lệ có thể tạo gap ~7-10 ngày calendar.
    """
    warnings: List[str] = []
    errors: List[str] = []
    if df.empty or len(df) < 2:
        return ValidationReport(name, len(df), (pd.NaT, pd.NaT), warnings, errors)

    s = pd.to_datetime(df[date_col]).sort_values().reset_index(drop=True)
    gaps = (s.diff().dt.days).dropna()
    big = gaps[gaps > warn_days]

    for idx in big.index:
        gap = int(big.loc[idx])
        d_prev = s.iloc[idx - 1].date()
        d_curr = s.iloc[idx].date()
        msg = f"Gap {gap}d: {d_prev} → {d_curr}"
        if gap > error_days:
            errors.append(msg)
        else:
            warnings.append(msg)

    return ValidationReport(name, len(df), (s.iloc[0], s.iloc[-1]), warnings, errors)


def check_abnormal_returns(
    df: pd.DataFrame, name: str,
    price_col: str = "close", date_col: str = "date",
    threshold: float = 0.15,
) -> ValidationReport:
    """Daily |log return| > threshold = signal adjustment chưa apply hoặc data corruption.

    Áp dụng cho adjusted close: nếu source đã adjust splits + stock dividends đúng,
    biến động daily hợp lý không quá ~10-15%. Vượt 15% là dấu hiệu cần inspect.
    Reference: Session cũ verify TCB VCI source → 0/1985 ngày vi phạm 15% mặc dù có
    stock dividend 1:1 năm 2024 → adjusted close đúng.
    """
    warnings: List[str] = []
    errors: List[str] = []
    if df.empty or len(df) < 2:
        return ValidationReport(name, len(df), (pd.NaT, pd.NaT), warnings, errors)

    d = df.sort_values(date_col).reset_index(drop=True)
    p = d[price_col].astype(float)

    if (p <= 0).any():
        n = int((p <= 0).sum())
        errors.append(f"{n} non-positive prices in '{price_col}'")
        return ValidationReport(name, len(df), (d[date_col].iloc[0], d[date_col].iloc[-1]),
                                warnings, errors)

    r = np.log(p / p.shift(1)).dropna()
    abnormal = r[r.abs() > threshold]

    for idx in abnormal.index:
        dt = d[date_col].iloc[idx].date()
        pp = float(p.iloc[idx - 1])
        pc = float(p.iloc[idx])
        ret = float(abnormal.loc[idx])
        warnings.append(
            f"Abnormal log return {ret:+.3f} on {dt}: {pp:.3f} → {pc:.3f}"
        )

    return ValidationReport(name, len(df), (d[date_col].iloc[0], d[date_col].iloc[-1]),
                            warnings, errors)


def canonicalize_price_unit(
    df: pd.DataFrame,
    expected_range: Tuple[float, float] = (5.0, 200.0),
    price_col: str = "close",
) -> pd.DataFrame:
    """Đảm bảo prices ở canonical unit: nghìn VND.

    TCB typical 2018-2026: ~15-50 nghìn VND. Nếu median nằm ngoài expected_range thì raise.
    KHÔNG tự convert — an toàn hơn buộc human inspect.
    """
    if df.empty:
        return df
    med = float(df[price_col].median())
    lo, hi = expected_range
    if not (lo <= med <= hi):
        raise ValueError(
            f"Median {price_col}={med:.2f} ngoài expected range [{lo}, {hi}] "
            f"cho unit 'nghìn VND'. Source có thể trả về VND raw hoặc unit khác. "
            f"Inspect manually trước khi save."
        )
    return df