# Changelog

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Semantic versioning: x.y.z (x = major methodology change, y = feature, z = fix).

## [Unreleased]

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
