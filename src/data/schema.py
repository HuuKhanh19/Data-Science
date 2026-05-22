"""
Schema definitions cho output parquet files.

Lock canonical schema để mọi downstream pipeline (preprocessing, EDA,
feature engineering) đều dựa vào hợp đồng dữ liệu rõ ràng. Validation
chạy ngay sau fetch để catch schema drift sớm.

Canonical price unit: nghìn VND (locked từ Session 1 cũ).
Canonical timezone của fetched_at: Asia/Ho_Chi_Minh (ICT).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class ColumnSpec:
    """Spec cho một column trong parquet schema."""
    name: str
    dtype: str
    nullable: bool = False
    description: str = ""


@dataclass(frozen=True)
class ParquetSchema:
    """Schema cho một parquet file. Validation theo column name + dtype."""
    name: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = ()

    def validate(self, df: pd.DataFrame) -> None:
        """Raise nếu DataFrame không khớp schema."""
        expected_cols = [c.name for c in self.columns]
        actual_cols = list(df.columns)

        # Check column presence + ordering
        missing = set(expected_cols) - set(actual_cols)
        if missing:
            raise ValueError(
                f"[{self.name}] Missing columns: {sorted(missing)}. "
                f"Got: {actual_cols}"
            )
        extra = set(actual_cols) - set(expected_cols)
        if extra:
            raise ValueError(
                f"[{self.name}] Unexpected columns: {sorted(extra)}. "
                f"Expected: {expected_cols}"
            )

        # Check dtypes
        for spec in self.columns:
            actual_dtype = str(df[spec.name].dtype)
            if not _dtype_compatible(actual_dtype, spec.dtype):
                raise ValueError(
                    f"[{self.name}] Column '{spec.name}' dtype mismatch: "
                    f"expected {spec.dtype}, got {actual_dtype}"
                )

            # Null check
            if not spec.nullable and df[spec.name].isna().any():
                n_nulls = df[spec.name].isna().sum()
                raise ValueError(
                    f"[{self.name}] Column '{spec.name}' has {n_nulls} nulls "
                    f"but is declared non-nullable"
                )

        # Check primary key uniqueness
        if self.primary_key:
            if df.duplicated(subset=list(self.primary_key)).any():
                n_dup = df.duplicated(subset=list(self.primary_key)).sum()
                raise ValueError(
                    f"[{self.name}] Primary key {self.primary_key} has "
                    f"{n_dup} duplicate rows"
                )


def _dtype_compatible(actual: str, expected: str) -> bool:
    """
    Lỏng hơn equality để handle pandas 2.x dtype variants.

    Tương đương:
    - datetime64[ns] ↔ datetime64[us] ↔ datetime64[ns, tz]
    - int32 ↔ int64 ↔ Int64
    - float32 ↔ float64
    - str ↔ object (pandas 2.x infers 'str' from CSV cho text columns)
    - int* ↔ float64 (pandas auto-coerce khi có NaN)
    """
    if actual == expected:
        return True
    # Datetime variants (including timezone)
    if expected.startswith("datetime64") and actual.startswith("datetime64"):
        return True
    # String / object equivalence (pandas 2.x với pyarrow backend)
    if expected == "object" and actual in ("str", "object", "string"):
        return True
    # Int variants (int32/int64 acceptable for int64 expected)
    if expected == "int64" and actual in ("int32", "int64", "Int64"):
        return True
    # Float variants (int → float allowed, pandas auto-coerce thường xảy ra)
    if expected == "float64" and actual in (
        "float32", "float64", "int32", "int64", "Int64", "Float64"
    ):
        return True
    # Bool variants
    if expected == "bool" and actual in ("bool", "boolean"):
        return True
    return False


# ============================================================
# Schema definitions cho từng parquet output
# ============================================================

OHLCV_COLUMNS = (
    ColumnSpec("date", "datetime64[ns]", nullable=False, description="Trading date"),
    ColumnSpec("open", "float64", nullable=False, description="Open price (nghìn VND)"),
    ColumnSpec("high", "float64", nullable=False, description="High price (nghìn VND)"),
    ColumnSpec("low", "float64", nullable=False, description="Low price (nghìn VND)"),
    ColumnSpec("close", "float64", nullable=False, description="Adjusted close (nghìn VND)"),
    ColumnSpec("volume", "int64", nullable=True, description="Volume in shares"),
    ColumnSpec("fetched_at", "datetime64[ns, Asia/Ho_Chi_Minh]", nullable=False,
               description="Timestamp khi row được fetch"),
)


TCB_PRICE_SCHEMA = ParquetSchema(
    name="tcb_price",
    columns=OHLCV_COLUMNS,
    primary_key=("date",),
)


VNINDEX_SCHEMA = ParquetSchema(
    name="vnindex",
    columns=OHLCV_COLUMNS,
    primary_key=("date",),
)


USDVND_SCHEMA = ParquetSchema(
    name="usdvnd",
    columns=(
        ColumnSpec("date", "datetime64[ns]", nullable=False),
        ColumnSpec("open", "float64", nullable=True),
        ColumnSpec("high", "float64", nullable=True),
        ColumnSpec("low", "float64", nullable=True),
        ColumnSpec("close", "float64", nullable=False, description="USD/VND exchange rate"),
        ColumnSpec("volume", "int64", nullable=True),
        ColumnSpec("fetched_at", "datetime64[ns, Asia/Ho_Chi_Minh]", nullable=False),
    ),
    primary_key=("date",),
)


# L3 macro: monthly/quarterly, có release_date — anti-leakage critical
MACRO_SCHEMA = ParquetSchema(
    name="macro",
    columns=(
        ColumnSpec("indicator", "object", nullable=False,
                   description="cpi_yoy | sbv_refinancing_rate | gdp_yoy"),
        ColumnSpec("reference_period", "datetime64[ns]", nullable=False,
                   description="Period end date (end of month for CPI, quarter for GDP)"),
        ColumnSpec("release_date", "datetime64[ns]", nullable=False,
                   description="Ngày công bố thực tế. Anti-leakage critical."),
        ColumnSpec("value", "float64", nullable=False, description="Numeric value"),
        ColumnSpec("unit", "object", nullable=False, description="percent_yoy | percent_annual"),
        ColumnSpec("source", "object", nullable=False, description="GSO | SBV | other"),
        ColumnSpec("release_date_inferred", "bool", nullable=False,
                   description="True nếu dùng convention (+14/30/45 ngày) thay vì release_date thật"),
        ColumnSpec("fetched_at", "datetime64[ns, Asia/Ho_Chi_Minh]", nullable=False),
    ),
    primary_key=("indicator", "reference_period"),
)


# L4 TCB fundamentals: quarterly với release_date
TCB_FUNDAMENTALS_SCHEMA = ParquetSchema(
    name="tcb_fundamentals",
    columns=(
        ColumnSpec("reference_period", "datetime64[ns]", nullable=False,
                   description="Quarter end date"),
        ColumnSpec("release_date", "datetime64[ns]", nullable=False),
        ColumnSpec("total_assets", "float64", nullable=False, description="Billion VND"),
        ColumnSpec("pe_ratio", "float64", nullable=True),
        ColumnSpec("npl_ratio", "float64", nullable=True, description="Percent"),
        ColumnSpec("credit_balance", "float64", nullable=False, description="Billion VND, gross loans"),
        ColumnSpec("nim", "float64", nullable=True, description="Percent annual"),
        ColumnSpec("equity", "float64", nullable=False, description="Billion VND"),
        ColumnSpec("release_date_inferred", "bool", nullable=False),
        ColumnSpec("source", "object", nullable=False, description="TCB_IR | vnstock | manual"),
        ColumnSpec("fetched_at", "datetime64[ns, Asia/Ho_Chi_Minh]", nullable=False),
    ),
    primary_key=("reference_period",),
)


ALL_SCHEMAS: dict[str, ParquetSchema] = {
    "tcb_price": TCB_PRICE_SCHEMA,
    "vnindex": VNINDEX_SCHEMA,
    "usdvnd": USDVND_SCHEMA,
    "macro": MACRO_SCHEMA,
    "tcb_fundamentals": TCB_FUNDAMENTALS_SCHEMA,
}