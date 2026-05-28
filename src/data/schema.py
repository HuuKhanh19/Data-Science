"""ParquetSchema lock cho 6 raw outputs của Phase 1."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str
    nullable: bool = False
    description: str = ""


@dataclass(frozen=True)
class ParquetSchema:
    name: str
    columns: Tuple[ColumnSpec, ...]
    primary_key: Tuple[str, ...]

    def validate(self, df: pd.DataFrame) -> None:
        expected = {c.name for c in self.columns}
        actual = set(df.columns)
        missing = expected - actual
        if missing:
            raise ValueError(f"[{self.name}] Thiếu cột: {sorted(missing)}")
        for col in self.columns:
            if not col.nullable and col.name in df.columns:
                n_null = int(df[col.name].isna().sum())
                if n_null > 0:
                    raise ValueError(
                        f"[{self.name}] Cột '{col.name}' có {n_null} null nhưng declared non-nullable"
                    )
        if self.primary_key:
            pk = list(self.primary_key)
            n_dup = int(df.duplicated(subset=pk).sum())
            if n_dup > 0:
                raise ValueError(f"[{self.name}] Primary key {pk} có {n_dup} duplicates")


OHLCV_COLUMNS: Tuple[ColumnSpec, ...] = (
    ColumnSpec("date", "datetime64[ns]", nullable=False),
    ColumnSpec("open", "float64", nullable=True),
    ColumnSpec("high", "float64", nullable=True),
    ColumnSpec("low", "float64", nullable=True),
    ColumnSpec("close", "float64", nullable=False),
    ColumnSpec("volume", "int64", nullable=True),
    ColumnSpec("fetched_at", "datetime64[ns, Asia/Ho_Chi_Minh]", nullable=False),
)

TCB_PRICE_SCHEMA = ParquetSchema(name="tcb_price", columns=OHLCV_COLUMNS, primary_key=("date",))
VNINDEX_SCHEMA = ParquetSchema(name="vnindex", columns=OHLCV_COLUMNS, primary_key=("date",))
USDVND_SCHEMA = ParquetSchema(name="usdvnd", columns=OHLCV_COLUMNS, primary_key=("date",))


CPI_SCHEMA = ParquetSchema(
    name="cpi",
    columns=(
        ColumnSpec("reference_period", "datetime64[ns]", nullable=False,
                   description="Month-end của tháng tham chiếu"),
        ColumnSpec("cpi_index", "float64", nullable=False,
                   description="Chỉ số CPI all-items IMF (gốc 2024=100); cpi_yoy=idx_t/idx_{t-12}-1 ở Step 2"),
        ColumnSpec("release_date", "datetime64[ns]", nullable=False,
                   description="Quy ước: reference_period (month-end) + 6 ngày"),
        ColumnSpec("fetched_at", "datetime64[ns, Asia/Ho_Chi_Minh]", nullable=False),
    ),
    primary_key=("reference_period",),
)

GDP_SCHEMA = ParquetSchema(
    name="gdp",
    columns=(
        ColumnSpec("reference_period", "datetime64[ns]", nullable=False),
        ColumnSpec("nominal_gdp_vnd_bil", "float64", nullable=False),
        ColumnSpec("release_date", "datetime64[ns]", nullable=False,
                   description="Conservative: quarter_end + 30 ngày"),
        ColumnSpec("fetched_at", "datetime64[ns, Asia/Ho_Chi_Minh]", nullable=False),
    ),
    primary_key=("reference_period",),
)


# L4 — thêm nim_pct (direct từ ratio nếu có)
TCB_FUNDAMENTALS_SCHEMA = ParquetSchema(
    name="tcb_fundamentals",
    columns=(
        ColumnSpec("reference_period", "datetime64[ns]", nullable=False),
        ColumnSpec("release_date", "datetime64[ns]", nullable=False,
                   description="Conservative: quarter_end + 45 ngày"),
        ColumnSpec("total_assets_vnd_bil", "float64", nullable=True),
        ColumnSpec("equity_vnd_bil", "float64", nullable=True),
        ColumnSpec("net_interest_income_vnd_bil", "float64", nullable=True),
        ColumnSpec("interest_earning_assets_vnd_bil", "float64", nullable=True,
                   description="Fallback để compute NIM nếu nim_pct missing"),
        ColumnSpec("npl_ratio_pct", "float64", nullable=True),
        ColumnSpec("credit_balance_vnd_bil", "float64", nullable=True,
                   description="loans_to_customers (KHÔNG phải interbank loans)"),
        ColumnSpec("eps_ttm", "float64", nullable=True,
                   description="From income_statement.eps_basic_vnd hoặc tương đương"),
        ColumnSpec("nim_pct", "float64", nullable=True,
                   description="NIM direct từ ratio nếu có"),
        ColumnSpec("fetched_at", "datetime64[ns, Asia/Ho_Chi_Minh]", nullable=False),
    ),
    primary_key=("reference_period",),
)


ALL_SCHEMAS = {
    "tcb_price": TCB_PRICE_SCHEMA,
    "vnindex": VNINDEX_SCHEMA,
    "usdvnd": USDVND_SCHEMA,
    "cpi": CPI_SCHEMA,
    "gdp": GDP_SCHEMA,
    "tcb_fundamentals": TCB_FUNDAMENTALS_SCHEMA,
}