"""
Validation utilities cho data acquisition.

Áp dụng cho mọi DataFrame OHLC daily (TCB price, VN-Index):
- HOSE calendar gap detection (WARN>12, ERROR>15)
- Unit canonicalization (đảm bảo prices ở nghìn VND)
- Adjusted close monotonicity (catch unexplained jumps)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Locked thresholds từ Session 1 cũ
HOSE_GAP_WARN_DAYS = 12
HOSE_GAP_ERROR_DAYS = 15

# Adjusted close: log return tuyệt đối > ngưỡng này là bất thường
# (sau khi đã trừ stock splits / cổ tức cổ phiếu lớn)
ABNORMAL_RETURN_LOG_THRESHOLD = 0.15  # 15%


@dataclass
class ValidationReport:
    """Tóm tắt validation cho một DataFrame."""
    name: str
    n_rows: int
    date_range: tuple[pd.Timestamp, pd.Timestamp]
    warnings: list[str]
    errors: list[str]

    @property
    def is_ok(self) -> bool:
        return len(self.errors) == 0

    def raise_if_errors(self) -> None:
        if self.errors:
            msg = f"[{self.name}] {len(self.errors)} validation errors:\n" + "\n".join(
                f"  - {e}" for e in self.errors
            )
            raise ValueError(msg)

    def log(self) -> None:
        for w in self.warnings:
            logger.warning(f"[{self.name}] {w}")
        for e in self.errors:
            logger.error(f"[{self.name}] {e}")


def check_calendar_gaps(
    df: pd.DataFrame,
    name: str,
    date_col: str = "date",
    warn_days: int = HOSE_GAP_WARN_DAYS,
    error_days: int = HOSE_GAP_ERROR_DAYS,
) -> ValidationReport:
    """
    Kiểm tra calendar gap giữa các trading days liên tiếp.

    Gap > error_days → có khả năng mất data trong middle (raise).
    Gap > warn_days → cluster holiday bất thường (warn only).

    Lưu ý: gap tính theo calendar days, KHÔNG phải trading days. Holiday
    cluster Tết + cuối tuần có thể ~9-10 ngày là bình thường.
    """
    warnings: list[str] = []
    errors: list[str] = []

    if df.empty:
        errors.append("DataFrame is empty")
        return ValidationReport(name, 0, (pd.NaT, pd.NaT), warnings, errors)

    dates = pd.to_datetime(df[date_col]).sort_values().reset_index(drop=True)
    gaps = dates.diff().dt.days.dropna()

    if (gaps <= 0).any():
        n = int((gaps <= 0).sum())
        errors.append(f"{n} non-positive gaps (duplicate or unsorted dates)")

    big_gaps = gaps[gaps > warn_days]
    for idx, gap in big_gaps.items():
        d1 = dates.iloc[idx - 1].date()
        d2 = dates.iloc[idx].date()
        if gap > error_days:
            errors.append(
                f"Gap {int(gap)} days between {d1} and {d2} exceeds error threshold "
                f"({error_days}) — likely missing data"
            )
        else:
            warnings.append(
                f"Gap {int(gap)} days between {d1} and {d2} (warn threshold {warn_days})"
            )

    return ValidationReport(
        name=name,
        n_rows=len(df),
        date_range=(dates.iloc[0], dates.iloc[-1]),
        warnings=warnings,
        errors=errors,
    )


def check_price_monotonicity(
    df: pd.DataFrame,
    name: str,
    price_col: str = "close",
    threshold: float = ABNORMAL_RETURN_LOG_THRESHOLD,
) -> ValidationReport:
    """
    Kiểm tra log return tuyệt đối hàng ngày không vượt ngưỡng bất thường.

    Áp dụng cho adjusted close: nếu source đã adjust splits + stock dividends
    đúng, biến động daily hợp lý không quá ~10-15%. Vượt 15% là dấu hiệu
    adjustment chưa apply hoặc data corruption.

    Session 1 cũ đã verify TCB VCI source: 0/1985 ngày vi phạm 15% threshold
    mặc dù có stock dividend 1:1 năm 2024 → adjusted close đúng.
    """
    warnings: list[str] = []
    errors: list[str] = []

    if df.empty or len(df) < 2:
        return ValidationReport(name, len(df), (pd.NaT, pd.NaT), warnings, errors)

    df_sorted = df.sort_values("date").reset_index(drop=True)
    prices = df_sorted[price_col].astype(float)

    if (prices <= 0).any():
        n = int((prices <= 0).sum())
        errors.append(f"{n} non-positive prices in column '{price_col}'")
        return ValidationReport(
            name, len(df), (df_sorted["date"].iloc[0], df_sorted["date"].iloc[-1]),
            warnings, errors,
        )

    log_returns = np.log(prices / prices.shift(1)).dropna()
    abnormal = log_returns[log_returns.abs() > threshold]

    for idx in abnormal.index:
        d = df_sorted["date"].iloc[idx].date()
        p_prev = prices.iloc[idx - 1]
        p_curr = prices.iloc[idx]
        ret = abnormal.loc[idx]
        warnings.append(
            f"Abnormal log return {ret:+.3f} on {d}: {p_prev:.3f} → {p_curr:.3f} "
            f"(check for unaccounted corporate action)"
        )

    return ValidationReport(
        name=name,
        n_rows=len(df),
        date_range=(df_sorted["date"].iloc[0], df_sorted["date"].iloc[-1]),
        warnings=warnings,
        errors=errors,
    )


def canonicalize_price_unit(
    df: pd.DataFrame,
    expected_range: tuple[float, float] = (5.0, 200.0),
    price_cols: tuple[str, ...] = ("open", "high", "low", "close"),
) -> pd.DataFrame:
    """
    Đảm bảo prices ở canonical unit: nghìn VND.

    TCB typical price range 2018-2026: ~15-50 nghìn VND. Nếu close median
    nằm ngoài expected_range thì raise — caller phải debug source unit.

    Lưu ý: KHÔNG tự convert (vd. chia 1000 nếu nghi VND raw). An toàn hơn là
    raise và buộc human inspect.
    """
    if df.empty:
        return df

    median_close = float(df["close"].median())
    lo, hi = expected_range

    if not (lo <= median_close <= hi):
        raise ValueError(
            f"Median close {median_close:.2f} ngoài expected range [{lo}, {hi}] "
            f"cho unit 'nghìn VND'. Source có thể đang trả về VND raw hoặc unit khác. "
            f"Cần inspect manually trước khi save."
        )

    return df