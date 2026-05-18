# Session 02 — As-of join (anti-leakage primitive)

**Ngày**: 18/05/2026
**Phase**: Phase 1 (data layer, IMPLEMENTATION §5.1)
**Pre-conditions met**:
- Session 1 complete (raw price/index/FX parquet files in `data/raw/`)
- `src/utils/logging.py` exists (get_logger)
- IMPLEMENTATION.md §5.1 contract locked
- research_design.md §4.3 / §4.4 specs for L3 / L4

**Deliverables**:
- `src/data/asof_join.py` — implementation (~ 320 LOC including docstrings)
- `tests/test_asof_join.py` — 24 tests, 100% line coverage
- This document
- Patches: `requirements.txt` (add `pytest-cov`), `IMPLEMENTATION.md §5.1`

---

## 1. Mục đích và bối cảnh

### 1.1 Vị trí trong pipeline

`asof_join` là **primitive** được dùng bởi mọi feature layer L3 và L4. Vai trò: merge dữ liệu tần suất thấp (monthly CPI / SBV rate, quarterly GDP, quarterly TCB fundamentals) vào panel daily mà **không gây look-ahead leakage**.

```
data/raw/macro_monthly.csv      ─┐
data/raw/macro_quarterly.csv    ─┼─► [asof_join] ─► features L3 / L4
data/raw/tcb_fundamentals.csv   ─┘                  │
                                                     ▼
                                  features_full.parquet (21 features)
                                                     │
                                                     ▼
                                            walk-forward backtest
```

Mọi feature ở day `t` chỉ được phép dùng thông tin **đã public ≤ t**. Đây là điều kiện cần để claim predictability có hiệu lực khoa học (research_design §2.5, §14.3).

### 1.2 Câu hỏi cần trả lời trong session này

Câu hỏi từ IMPLEMENTATION.md §5.1 và §8 (Tier 1):
- Q1: Hàm có đảm bảo "value tại t = giá trị có `release_date ≤ t` lớn nhất" với mọi t hợp lệ không? → **Yes**, verified bằng 7 semantic tests + 1 property test trên 100 random dates (manual oracle).
- Q2: Hàm có không bao giờ dùng `reference_period` thay vì `release_date` không? → **Yes**, Test 6 (100-day lag) chứng minh trực tiếp.
- Q3: Hàm reject mọi malformed input không silent fail? → **Yes**, 9 validation tests cover các error path.

### 1.3 Scope cố ý KHÔNG bao gồm

- **Real L3 / L4 data acquisition**: thực hiện trong Session 3 (manual GSO/SBV scrape, vnstock cross-check) và Session 4 (TCB IR scrape). Session 2 chỉ build primitive + verify trên synthetic data.
- **Cross-source verification trong fetcher** (đã defer trong Session 1, mục 7.3).
- **L3 / L4 feature engineering**: derive columns (P/E từ price+EPS, growth ratios) thuộc `src/features/l4_fundamentals.py`, không phải `asof_join.py`.

---

## 2. Source code — chi tiết

### 2.1 `src/data/asof_join.py`

#### Triết lý kiến trúc

Ba nguyên tắc thiết kế xếp theo thứ tự ưu tiên:

1. **Anti-leakage first**: API chỉ chấp nhận `release_date_col` làm tên column ngữ nghĩa publication-date. Function không tự suy ra release_date từ `reference_period_*`; caller chịu trách nhiệm pass đúng cột. Test 6 đảm bảo function không leak ngay cả khi raw frame có cả hai cột song song.
2. **Fail loud over silent**: 9 validation paths. Mọi malformed input → raise rõ ràng (TypeError hoặc ValueError với message identifying offending field). Không có code path nào silent-coerce hoặc skip rows.
3. **Single implementation, multiple call sites**: core `asof_join` + 2 thin wrappers `asof_join_quarterly` / `asof_join_monthly`. Monthly và quarterly identical operation; tách function chỉ phục vụ readability ở call site. Decision D1 locked Tuần này.

#### Public API

```python
def asof_join(
    daily_df: pd.DataFrame,
    low_freq_df: pd.DataFrame,
    value_cols: list[str],
    release_date_col: str = "release_date",
) -> pd.DataFrame
```

**Semantics** (IMPLEMENTATION.md §5.1):
- Với mỗi row tại date `t` trong `daily_df`, attach values từ row của `low_freq_df` có `release_date` lớn nhất `≤ t`.
- Forward-fill implicit qua `pd.merge_asof(direction="backward", allow_exact_matches=True)`.
- `release_date_col` bị **drop khỏi output** (decision D3) để tránh accidentally consumed làm feature downstream.

Wrappers `asof_join_quarterly`, `asof_join_monthly` chỉ delegate; signature identical với core.

#### Internal helpers

| Helper | Purpose |
|---|---|
| `_validate_daily(daily_df)` | DatetimeIndex check, unique check, sorted-ascending check |
| `_validate_low_freq(df, vc, rdc)` | release_date_col exists, value_cols exist, release_date coercible to datetime, no NaT, no duplicates (D2: **raise** trên dup, không silent dedup) |
| `_validate_no_column_conflict(...)` | daily_df.columns ∩ {release_date_col, *value_cols} = ∅; index.name không trùng |

#### Quyết định quan trọng

**D1 (locked) — Single core function + 2 wrappers**: rationale ở §1.1 trên. Tests chạy chủ yếu trên core; 2 tests dedicated verify wrapper delegation bằng `assert_frame_equal`.

**D2 (locked) — Raise on duplicate release_dates**: nếu raw data có 2 rows cùng `release_date` (ví dụ TCB consolidated vs standalone báo cáo cùng ngày), đó là **ambiguity** phải resolve ở data prep stage. Function raise `ValueError` với message gồm 5 duplicate dates đầu tiên. Test 12 covers.

**D3 (locked) — Drop `release_date` khỏi output**: function trả `daily_df` columns gốc + `value_cols`. Lý do: `release_date` mang weak time-of-year signal (CPI thường release giữa tháng, GDP cuối tháng đầu quý sau). Nếu accidentally làm feature downstream → leak signal vô tình. Audit trace vẫn preserved qua original `low_freq_df`.

**D4 (locked) — Forward-fill indefinitely past latest release**: at any `t > max(release_date)`, returned value = latest known. Correct behavior cho production inference (today luôn sau latest disclosure). Default behavior của `pd.merge_asof(direction="backward")` đã match.

**D5 (locked) — No explicit anti-misuse guard for "reference_period" in release_date_col**: function không thể tự biết tên column có semantic gì. Docstring có warning + Test 6 chứng minh anti-leak property trên synthetic data với 100-day lag.

#### Cross-version pandas robustness

pandas 2.x cho phép multiple datetime resolutions (`datetime64[ns]`, `datetime64[us]`, `datetime64[ms]`). `pd.merge_asof` yêu cầu **exact dtype match** trên join keys. Constructor khác nhau cho dtype khác nhau:
- `pd.bdate_range(...)` → `datetime64[us]` (pandas ≥ 2.0)
- `pd.to_datetime([list of str])` → `datetime64[ns]`

Code coerce right side về dtype của left side trước merge: `right[release_date_col] = right[release_date_col].astype(left[index_name].dtype)`. Preserves daily_df's original index dtype trong output. Lossless vì mọi date thực tế nằm trong range biểu diễn được của cả 2 resolutions. Issue + fix chi tiết ở §6.

### 2.2 `tests/test_asof_join.py`

24 tests, tổ chức 5 nhóm:

| Nhóm | Tests | Mục đích |
|---|---|---|
| 1. Semantic | 1–7 | Anti-leak proof, forward-fill, multi-column, boundary cases |
| 2. Input validation | 8–14 | Reject malformed inputs loudly |
| 3. Output structure | 15–18 | Index/columns preserved, release_date dropped, property test vs oracle |
| 4. Wrappers | 19–20 | Quarterly/monthly wrappers bit-identical với core |
| 5. Defensive coverage | 21–24 | Path coverage cho remaining error branches |

Fixtures inline (decision Tuần này: conftest.py defer đến khi Session 3+ có concrete reuse).

**Critical test = Test 6** (`test_release_date_lag_100_days_does_not_use_reference_period`):
- Synthetic data: 1 release với `reference_period_end = 2020-03-31`, `release_date = 2020-07-09` (lag 100 days), value = 99.9
- Verify: tại `t ∈ {2020-04-15, 2020-05-15, 2020-07-08}` (sau period end nhưng trước release), returned value = **NaN**.
- Verify: tại `t ∈ {2020-07-09, 2020-07-10}` (on/after release), returned value = 99.9.
- Nếu implementation nhầm dùng `reference_period_end` → test fail trực tiếp.

**Property test = Test 18** (`test_property_random_dates_match_manual_oracle`):
- Random 100 dates từ daily_df (seed=42 matching project canonical CFG.SEED).
- Manual oracle: vòng lặp tính `eligible = sorted_releases[release_date ≤ t]; expected = eligible.iloc[-1]`.
- Assert equal với output của asof_join. Catches edge cases mà specific tests có thể miss.

---

## 3. Data

**Không applicable**. Session 2 không acquire real data; tests dùng synthetic fixtures inline (quarterly_df_2020 với 4 rows). Real L3 / L4 acquisition trong Session 3 (macro) và Session 4 (TCB IR).

---

## 4. Model

Không applicable.

---

## 5. Verification

### 5.1 Test run result

```
================== test session starts ==================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0
plugins: cov-7.1.0
collected 24 items

tests/test_asof_join.py::test_before_first_release_returns_nan PASSED
tests/test_asof_join.py::test_on_release_date_inclusive PASSED
tests/test_asof_join.py::test_just_after_release_date_uses_that_row PASSED
tests/test_asof_join.py::test_between_releases_uses_older_release PASSED
tests/test_asof_join.py::test_after_latest_release_forward_fills PASSED
tests/test_asof_join.py::test_release_date_lag_100_days_does_not_use_reference_period PASSED
tests/test_asof_join.py::test_multiple_value_cols_joined_together PASSED
tests/test_asof_join.py::test_raises_on_missing_value_col PASSED
tests/test_asof_join.py::test_raises_on_missing_release_date_col PASSED
tests/test_asof_join.py::test_raises_on_non_datetimeindex_daily PASSED
tests/test_asof_join.py::test_raises_on_unsorted_daily_index PASSED
tests/test_asof_join.py::test_raises_on_duplicate_release_dates PASSED
tests/test_asof_join.py::test_raises_on_empty_value_cols PASSED
tests/test_asof_join.py::test_raises_on_column_name_conflict PASSED
tests/test_asof_join.py::test_preserves_daily_index_and_other_columns PASSED
tests/test_asof_join.py::test_release_date_dropped_from_output PASSED
tests/test_asof_join.py::test_empty_low_freq_returns_all_nan PASSED
tests/test_asof_join.py::test_property_random_dates_match_manual_oracle PASSED
tests/test_asof_join.py::test_quarterly_wrapper_delegates_to_asof_join PASSED
tests/test_asof_join.py::test_monthly_wrapper_delegates_to_asof_join PASSED
tests/test_asof_join.py::test_handles_unnamed_daily_index PASSED
tests/test_asof_join.py::test_raises_on_duplicate_daily_dates PASSED
tests/test_asof_join.py::test_raises_on_nan_release_date PASSED
tests/test_asof_join.py::test_raises_on_index_name_conflict PASSED

================ tests coverage ================
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
src/data/asof_join.py      62      0   100%
-----------------------------------------------------
TOTAL                      62      0   100%

============= 24 passed in 0.95s =============
```

### 5.2 Locked thresholds

| Item | Value | Reference |
|---|---|---|
| Coverage trên `src/data/asof_join.py` | **100%** (target 70%, IMPLEMENTATION §8) | §5.1 trên |
| Number of tests | **24** (Tier 1) | |
| Number of validation paths | **9 distinct error conditions** | §2.1 internal helpers |
| Property-test sample size | **100 random dates** (seed=42) | Test 18 |

### 5.3 Interpretation

- **24/24 tests pass + 100% coverage** đáp ứng Tier 1 critical anti-leakage spec trong IMPLEMENTATION §8.
- Test 6 chứng minh trực tiếp rằng nếu raw data có cả `reference_period_end` và `release_date` song song (worst-case ambiguity), function vẫn anti-leak.
- Test 18 (property) covers 100 random dates; nếu có edge case unknown, xác suất reproduce ≥ 1% trong 100 samples — chưa thấy fail nào.
- Function **không thể tự bảo vệ** trước trường hợp caller pass nhầm `reference_period_end` qua param `release_date_col`. Đây là D5 acknowledged — docstring + Test 6 alone không đủ; downstream feature pipelines (Session 4) phải explicit pass đúng tên column.

---

## 6. Issues encountered và fixes

### Issue 1 — `MergeError: incompatible merge keys`, dtype `datetime64[us]` vs `datetime64[ns]`

**Symptom**: `test_empty_low_freq_returns_all_nan` fail với `pandas.errors.MergeError: incompatible merge keys [0] dtype('<M8[us]') and dtype('<M8[ns]'), must be the same type`.

**Root cause**: pandas ≥ 2.0 cho phép multiple datetime resolutions tồn tại song song. `pd.bdate_range` mặc định trả về `datetime64[us]`; `pd.to_datetime(list_of_strings)` trả về `datetime64[ns]`. `pd.merge_asof` yêu cầu exact dtype match trên join keys → fail.

Câu hỏi: tại sao chỉ test này fail mà 18 tests khác (cũng có left=us, right=ns) lại pass? Hypothesis: pandas merge_asof có path auto-coerce khi non-empty (qua type promotion) nhưng không khi empty. Không deep-dive — fix là robust regardless.

**Fix**: trong `asof_join`, coerce right side về dtype của left:
```python
target_dtype = left[index_name].dtype
right[release_date_col] = right[release_date_col].astype(target_dtype)
```

Cách này (vs. coerce cả hai về ns) preserves dtype gốc của `daily_df.index` trong output. Lossless cho mọi date thực tế (cả us và ns đều bao trùm range ngày trong project).

**Lesson**: với pandas 2.x, đừng giả định dtype consistency. `merge_asof` (và một số ops khác) strict hơn các operation cũ. Khi build join primitives, explicit dtype-coerce join key.

### Issue 2 — `AssertionError: (None, <BusinessDay>)` trên `assert_series_equal`

**Symptom**: `test_preserves_daily_index_and_other_columns` fail với `(None, <BusinessDay>)` — left side có `freq=None`, right side có `freq=BusinessDay()`.

**Root cause**: `pd.bdate_range` tạo DatetimeIndex với cached `freq='B'`. Sau khi đi qua `reset_index → merge_asof → set_index`, attribute `freq` bị mất. `pd.testing.assert_series_equal` default check `freq` strict → fail dù values và dtype identical.

**Fix**: thay `pd.testing.assert_index_equal(...)` bằng `result.index.equals(daily_df_2020.index)` (lenient với freq attribute). Thay `pd.testing.assert_series_equal(...)` bằng version có `check_freq=False`.

**Lesson**: freq là metadata cached trên DatetimeIndex; bất kỳ pandas op nào tạo Index mới đều có thể drop nó. Test chỉ nên assert thuộc tính mà downstream callers thực sự depend on (values + dtype + index, KHÔNG phải freq).

### Issue 3 — `assert_index_equal` không accept `check_freq` kwarg

**Symptom**: ban đầu tôi dùng `pd.testing.assert_index_equal(..., check_freq=False)` để fix Issue 2 → fail với `TypeError: assert_index_equal() got an unexpected keyword argument 'check_freq'`.

**Root cause**: pandas API asymmetry: `assert_series_equal` có `check_freq` kwarg, nhưng `assert_index_equal` thì không (ít nhất trong pandas version test env). Có thể version khác mới expose.

**Fix**: thay bằng `result.index.equals(daily_df_2020.index)`. Method `Index.equals()` so sánh values + dtype nhưng lenient với freq.

**Lesson**: kiểm tra API surface của test utilities trước khi assume kwarg tồn tại. Khi compare collections, prefer `.equals()` method over assert_*_equal helpers nếu chỉ cần value-level equality.

---

## 7. Open questions / next session

### 7.1 Resolved trong Session 2

| Question | Status | Reference |
|---|---|---|
| D1 — Single asof_join function hay tách quarterly/monthly? | ✅ Single core + 2 wrappers | §2.1 D1 |
| D2 — Behavior on duplicate release_dates? | ✅ Raise ValueError | §2.1 D2 |
| D3 — Keep release_date column trong output? | ✅ Drop | §2.1 D3 |
| pytest-cov tooling | ✅ Added to requirements.txt | patches §7.3 |
| conftest.py? | ✅ Defer; inline fixtures cho Session 2 | §2.2 |

### 7.2 Locked invariants cho downstream

| Item | Value | Reference |
|---|---|---|
| Coverage target met cho `src/data/asof_join.py` | 100% (target 70%) | §5.2 |
| Critical anti-leak property | Holds với 100-day lag synthetic | Test 6 |
| Property test sample size | 100 random dates, seed=42 | Test 18 |
| Output dtype | matches `daily_df.index.dtype` (preserved) | §2.1 cross-version |

### 7.3 Patches required

**`requirements.txt`** — add 1 line:
```
pytest-cov ^4.0
```

**`IMPLEMENTATION.md §5.1`** — replace với nội dung mới:
- Unified core `asof_join` API + 2 wrappers
- Thêm `release_date_col: str = "release_date"` param
- Document drop behavior + raise-on-duplicate
- Reference Session 2 doc

(Snippet exact để paste vào IMPLEMENTATION được gửi inline trong chat thay vì file).

### 7.4 Mở ra cho session sau

- **Real L3 acquisition** (Session 3 dự kiến): manual scrape GSO (CPI YoY monthly, GDP YoY quarterly), SBV (refinancing rate monthly), commit to `data/raw/macro_monthly.csv` + `data/raw/macro_quarterly.csv` per IMPLEMENTATION §4.1 schema. Sẽ test integration: load → asof_join → verify NaN tail at start (TCB 2018-06-04 + macro data range).
- **Real L4 acquisition** (Session 4): manual scrape TCB IR (quarterly reports 2018Q2 trở đi). Verify release_date scraping; nếu missing dùng convention `+45 days` (research_design §4.4).
- **Feature pipeline** (Session 5+): compose `asof_join` calls trong `src/features/l3_macro.py` và `src/features/l4_fundamentals.py`. Sẽ test integration: full 21-feature panel với end_date varying, verify monotonic NaN coverage.
- **Cross-source verification trong fetcher** (defer tiếp): không phải concern của asof_join. Có thể revisit Session 13 production hardening.

### 7.5 Session 3 preview

**Module**: `src/data/loaders.py` (load CSV macro/fundamentals với schema validation) + acquire real L3 data + EDA notebook stationarity tests.

**Inputs cần**:
- GSO website hoặc PDF reports → CPI YoY, GDP YoY
- SBV website → refinancing rate timeline
- (optional) cross-check vnstock macro endpoint

**Tests new**: `tests/test_loaders.py` — schema validation cho 3 CSV files (matches IMPLEMENTATION §4.1 columns/dtypes).