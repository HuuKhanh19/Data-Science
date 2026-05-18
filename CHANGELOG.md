# Changelog

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Semantic versioning: x.y.z (x = major methodology change, y = feature, z = fix).

## [Unreleased]

## [0.5.0] - 2026-05-18

### Added
- `src/data/loaders.py`: schema-validated CSV loaders. `load_macro_monthly` + `load_macro_quarterly` (L4 `load_tcb_fundamentals` defer Session 5). 8 distinct validation paths: column schema, release_date parse, release_date_source enum, reference_period format, uniqueness (ref_period + release_date), numeric NaN, file errors. Output guaranteed sorted by release_date with datetime64[ns] dtype — ready for direct asof_join.
- `src/features/l3_macro.py`: 5 L3 macro features per research_design.md §4.3 — `vnindex_return`, `usdvnd_change`, `cpi_yoy_pct`, `sbv_rate_pct`, `gdp_yoy_pct`. Composer + 2 individual functions; CPI/SBV/GDP composed via asof_join (Session 2 primitive).
- `tests/test_loaders.py`: 20 tests (schema validation paths + happy path + real-world mixed sources).
- `tests/test_l3_macro.py`: 20 tests (per-feature golden + composer integration + Tier 1 anti-leak).
- `docs/session04_l3_macro_loaders.md`: session log per 7-section template.

### Changed
- `IMPLEMENTATION.md §4.1`: added `release_date_source` column to `macro_monthly.csv` and `macro_quarterly.csv` schemas. Enum-validated per frequency (`fallback_14d` for monthly, `fallback_30d` for quarterly). Audit trail for whether publication date was scraped or computed via conservative convention research_design.md §4.3.

### Decisions locked (Session 4)
- **D1**: Loaders cover L3 only this session; L4 (`tcb_fundamentals.csv`) defer Session 5 to gather real TCB IR data context first.
- **D2**: L3 module = top-level composer + 2 individual functions (consistent with L1/L2 pattern).
- **D3**: USD/VND FX alignment via `reindex(method="ffill")` to TCB calendar. FX-closed days → 0 return (semantic "rate unchanged").
- **D4**: VN-Index alignment via strict `reindex` (no fill) — raise on missing date. VN-Index and TCB share HOSE calendar; mismatch = data quality issue.
- **D5**: Anti-leak strategy = 2 global mutation tests (daily features + future macro release injection) + per-feature golden values.
- **D6**: `L3_FEATURE_COLS = ("vnindex_return", "usdvnd_change", "cpi_yoy_pct", "sbv_rate_pct", "gdp_yoy_pct")` — frozen contract.
- **D7**: `release_date_source` audit column added to schema. Tradeoff: +1 column complexity for permanent audit trail.

### Verified
- Tests: **40 passed** (20 loaders + 20 L3). Full suite (Sessions 2+3+4): 110/110.
- Coverage: **100%** trên `src/data/loaders.py` and `src/features/l3_macro.py`.

### Fixed (pandas 2.x compat reinforcement)
- `_parse_release_date` explicit cast to `datetime64[ns]` after `pd.to_datetime(format="%Y-%m-%d")` — pandas 2.x returns `datetime64[us]` by default which breaks downstream `asof_join` dtype match. Same lesson as Session 2 Issue 1; now applied proactively.
- Removed dead defensive dtype check in `_coerce_numerics` (`pd.to_numeric(errors="coerce")` always returns float64; the NaN case is caught by `_validate_no_nan_in_numerics` with a clearer message).

## [0.4.0] - 2026-05-18

### Added
- `src/features/l1_price.py`: 4 lag log-return features (r_lag1, r_cum5, r_cum10, r_cum20) per research_design.md §4.1. Anti-leak by construction: every feature at day t uses only P_{t-1} and earlier via `log_p.shift(1) - log_p.shift(1+k)` pattern.
- `src/features/l2_technical.py`: 6 technical indicators per research_design.md §4.2 — `ma_crossover` (MA_5/MA_20), `momentum_12_3` (J-T skip-3), `bb_position` (Bollinger, ddof=0), `trb_signal` (∈ {-1, 0, 1, NaN}), `rsi14` (Wilder 1978 canonical smoothing), `macd_norm` (EMA fast/slow normalized by price). All anti-leak shifted.
- `tests/test_l1_price.py`: 15 tests (Tier 2 golden + Tier 1 anti-leak).
- `tests/test_l2_technical.py`: 31 tests (per-indicator golden values + edge cases + global anti-leak).
- `docs/session03_l1_l2_features.md`: session log per 7-section template.

### Changed
- `research_design.md §4.2`: Momentum 3-12mo formula clarified from $\log(P_{t-1}/P_{t-252})$ to $\log(P_{t-63}/P_{t-252})$ per decision D7=B (Session 3 prologue). Skip-3 J-T canonical convention matches the "3-12 month" naming and "cắt 63 phiên" wording. Pre-lock fix, no version bump required.

### Decisions locked (Session 3)
- **D1**: 2 separate files `l1_price.py` + `l2_technical.py` per IMPLEMENTATION.md §3 folder structure.
- **D2**: NaN trong warmup, không drop rows ở function level (downstream walk-forward iterator handles).
- **D3**: Default `price_col="adj_close"` (semantic-correct per research_design §4.1).
- **D4**: Strict anti-leak P_{t-1} backward, verified end-to-end.
- **D5**: RSI(14) Wilder (1978) canonical (manual implementation), không dùng `pandas.ewm` default (init differs).
- **D6**: MACD EMA với `adjust=False, min_periods=26`.
- **D7**: Momentum 3-12mo = $\log(P_{t-63}/P_{t-252})$ (skip-3 J-T). Direct empirical proof via `test_momentum_12_3_anti_leak_recent_63_days_dont_affect`.
- **TA-Lib cross-check**: not used. Source of truth = hand-computed golden values from research_design.md citation formulas.

### Verified
- Tests: **46 passed** (15 L1 + 31 L2). Full suite (Sessions 2+3): 70/70.
- Coverage: **100%** trên `src/features/l1_price.py`, **100%** trên `src/features/l2_technical.py`.

## [0.3.0] - 2026-05-18

### Added
- `src/data/asof_join.py`: anti-leakage primitive merge dữ liệu lower-frequency (monthly L3 macro, quarterly L3 GDP, quarterly L4 TCB fundamentals) vào panel daily. Core `asof_join` + thin wrappers `asof_join_quarterly` / `asof_join_monthly` (delegation only, identical behavior). Backed by `pd.merge_asof(direction="backward", allow_exact_matches=True)`.
- `tests/test_asof_join.py`: 24 Tier-1 tests với inline synthetic fixtures. Covers semantics (7), input validation (7), output structure (4), wrapper delegation (2), defensive coverage (4). Critical tests: 100-day-lag anti-leak proof; property test vs manual oracle trên 100 random dates (seed=42).
- `docs/session02_asof_join.md`: session log theo template 7 sections.
- `pytest-cov>=4.0` thêm vào `requirements.txt`.

### Changed
- `IMPLEMENTATION.md §5.1`: contract updated. Function signature thêm `release_date_col: str = "release_date"` param. Unified core `asof_join` API + 2 named wrappers (decision D1). Document drop-on-output behavior (D3) + raise-on-duplicate (D2).

### Decisions locked (Session 2)
- **D1**: Single core `asof_join` + thin wrappers `asof_join_quarterly` / `asof_join_monthly` (no logic duplication).
- **D2**: Duplicate `release_date` values → raise `ValueError` (no silent dedup; ambiguity là data-prep issue).
- **D3**: `release_date` column dropped khỏi output (tránh accidentally consumed làm feature downstream; audit trace preserved qua original `low_freq_df`).
- **D4**: Past the latest release, latest known value forward-filled indefinitely (matches `pd.merge_asof` default; correct cho production inference).
- **D5**: No explicit guard chống misuse `release_date_col` (vd pass `reference_period_end`). Docstring warns; Test 6 demonstrates anti-leak property trên synthetic 100-day lag.

### Verified
- Tests: **24/24 pass** trên Linux (Python 3.12, pandas 2.3) và Windows (Python 3.11.15, conda env `ds`).
- Coverage: **100%** trên `src/data/asof_join.py` (target 70% per IMPLEMENTATION §8).

### Fixed (cross-version pandas compatibility)
- `pd.merge_asof` dtype mismatch: pandas 2.x produces `datetime64[us]` từ `pd.bdate_range` nhưng `datetime64[ns]` từ `pd.to_datetime(list_of_strings)`. Coerce right-side join key về dtype của left trước merge. Preserves daily_df's original index dtype trong output. Lossless cho mọi date thực tế.
- `pd.testing.assert_index_equal` không có kwarg `check_freq` trong một số pandas version; tests chuyển sang `Index.equals()` (lenient về freq metadata).

## [0.2.0] - 2026-05-16

### Added
- `src/data/fetchers.py`: 3 public fetchers (`fetch_tcb_price`, `fetch_vnindex`, `fetch_usdvnd`) với primary→fallback logic. Structural-only DQ validators map về 4 mức của Slide Data Science.
- `scripts/acquire_data.py`: CLI orchestrator. Flags `--start-date`, `--end-date`, `--source`, `--dry-run`, `--skip-existing`. Print summary stats + write `_acquisition_summary.json` audit.
- `notebooks/00_data_source_verification.ipynb`: 9 code cells cho Phase 0 verification. Fetch song song 2 source, Open Q1 algorithmic detection (no hardcode dates), scale-invariant cross-source check (log returns), HOSE calendar gap analysis.
- `docs/session01_data_acquisition.md`: session log đầy đủ theo template 6 sections.
- `docs/`: thư mục mới chứa session logs (append-only, immutable record).

### Changed
- `requirements.txt`: ghi rõ vnstock v4+ cần install chart backend riêng từ private index `vnstocks.com/api/simple` (không có trên PyPI).
- `README.md`: setup chuyển từ `python -m venv` sang `conda create -n ds` (Option B: fresh env + `conda install pytorch pytorch-cuda=12.4`).
- `_fetch_ohlcv_yfinance`: dùng `auto_adjust=True` thay vì `auto_adjust=False` để mọi OHLC cùng adjusted basis, tránh integrity violation.
- `fetch_tcb_price`: thêm rescale `/1000` cho yfinance fallback. Parquet output luôn ở nghìn VND bất kể source.

### Fixed
- OHLC integrity violation tại các ngày sớm 2018-06: do mix raw OHL với adjusted close khi yfinance dùng `auto_adjust=False`.
- `IndexError` trong `summary_stats`: `Series.idxmax()` trên DatetimeIndex Series trả về Timestamp label, không phải integer position. Bỏ `df.index[...]` wrapper.
- Cross-source rel_diff trong notebook 00 cho con số ~99900%: scale mismatch giữa nghìn VND (vnstock) và VND (yfinance). Replace bằng log-return comparison (scale-invariant) + scale factor inference.
- NameError cascade `rel_diff` trong notebook cell 7-8 sau khi update cell 6.

### Resolved (Open Questions from IMPLEMENTATION.md section 11)
- **Q1**: vnstock VCI returns **ADJUSTED close**. Evidence: 0/1985 ngày có `|log return| > 15%` trong toàn bộ lịch sử TCB, kể cả TCB có stock dividend 1:1 trong 2024. Chi tiết: `docs/session01_data_acquisition.md` section 4.3.

### Locked (Phase 0 baselines)
- Canonical price unit: **nghìn VND**.
- Cross-source log-return disagreement threshold (cho future Session 2+ cross-check): **100 bp (1%)**.
- HOSE calendar gap: **WARN > 12, ERROR > 15** calendar days. Max observed = 10 (Tết Nguyên đán, 4 events: 2019, 2022, 2025, 2026).
- Known disagreement dates (off-by-one dividend adjustment timing, KHÔNG flag là bug): 2024-05-21, 2019-03-14, 2019-03-15.

### Data acquired
- `data/raw/price_tcb.parquet` (1,985 rows, 2018-06-04 → 2026-05-15, 64.0 KB)
- `data/raw/price_vnindex.parquet` (2,076 rows, 2018-01-16 → 2026-05-15, 43.7 KB)
- `data/raw/fx_usdvnd.parquet` (2,069 rows, 2018-06-04 → 2026-05-15, 26.4 KB)
- `data/raw/_acquisition_summary.json` (audit log)

## [0.1.0] - 2026-05-16

### Added
- Initial repo bootstrap.
- Folder structure theo `IMPLEMENTATION.md` section 3.
- Utility modules: `config.py` (frozen `CFG` dataclass), `seeds.py` (global seed setup), `logging.py` (project-wide logger format).
- Project documentation: `README.md`, `LICENSE` (MIT), `pyproject.toml`, `requirements.txt`, `.gitignore`.
- Placeholder structure cho `src/{data,features,models,eval,interpret,ablation,storage,app,inference,utils}/`, `tests/`, `notebooks/`, `scripts/`.

### Notes
- Chưa lock pre-registration. Lock dự kiến cuối tuần 2 (28/05/2026) với git tag `v1.0-prereg-locked`.
