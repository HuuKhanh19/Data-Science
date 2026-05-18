# Session 04 — Loaders + L3 macro features

**Ngày**: 18/05/2026
**Phase**: Phase 1 (data layer) + Phase 3 (feature engineering)
**Pre-conditions met**:
- Session 1 complete (price/index/FX parquet trong `data/raw/`)
- Session 2 complete (asof_join primitive, 100% coverage)
- Session 3 complete (L1 + L2 features, 100% coverage)
- IMPLEMENTATION.md §4.1 schemas locked (will receive D7 amendment, see §7.2 patches)
- research_design.md §4.3 spec (5 L3 features)

**Deliverables**:
- `src/data/loaders.py` — schema-validated CSV loaders (~ 290 LOC)
- `src/features/l3_macro.py` — 5 L3 macro features (~ 220 LOC)
- `tests/test_loaders.py` — 20 tests (schema validation paths)
- `tests/test_l3_macro.py` — 20 tests (golden + anti-leak + composer)
- This document
- Patches: `IMPLEMENTATION.md §4.1` (add `release_date_source` to schemas — D7), `CHANGELOG.md` (v0.5.0), `docs/README.md` (index update)

**Out-of-scope (acknowledged)**:
- L4 `tcb_fundamentals.csv` loader → Session 5 (defer to have real data context).
- Real L3 data acquisition → user task, parallel/async with this session. Code uses synthetic CSV fixtures.

---

## 1. Mục đích và bối cảnh

### 1.1 Vị trí trong pipeline

Session 4 đặt cầu nối giữa **raw CSV macro data** và **daily feature panel** dùng cho training. Layer kiến trúc:

```
data/raw/macro_monthly.csv     ─┐
                                │
                                ▼
                          [load_macro_monthly]
                                │  (schema-validated DataFrame)
                                ▼
                          ┌─────────────────┐
TCB price index           │   asof_join     │ ← Session 2 primitive
VN-Index, USD/VND   ────► │   (anti-leak)   │
                          └─────────────────┘
                                │
                                ▼
                          [compute_l3_features]
                                │  (5 features × daily)
                                ▼
                     ready for L1+L2 join trong pipeline (Session 5+)
```

### 1.2 Câu hỏi cần trả lời

Q1 (Tier 1): Schema validation có catch mọi malformed input không silent-pass? → **Yes**, 20 validation tests cover 8 distinct failure modes (missing/extra cols, bad date format, bad enum, bad ref format, duplicates, NaN, file errors).

Q2 (Tier 1): Anti-leak property có preserved end-to-end qua composer không? → **Yes**, 2 global anti-leak tests verify daily-feature mutation và future-release injection.

Q3 (Tier 2): Mỗi feature có implement đúng spec research_design §4.3 không? → **Yes**, golden values per feature + CPI/SBV/GDP forward-fill semantics verified ở specific dates.

Q4: Calendar alignment có handle correctly cho FX gap days không? → **Yes**, `test_usdvnd_change_handles_fx_gap_via_ffill` verify rằng FX-closed day có log return = 0 (semantic "tỷ giá không đổi").

### 1.3 Scope cố ý KHÔNG bao gồm

- **L4 fundamentals loader**: defer Session 5. Lý do: TCB IR scraping có thể reveal schema nuances (vd EPS dạng nghìn vnd, ratios dạng decimal/percent inconsistency) chỉ thấy được sau khi attempt real scrape. Premature lock schema = risk rework.
- **Real L3 data acquisition**: parallel user task. Code tests dùng synthetic CSV fixtures inline qua `tmp_path` pytest fixture — không chạm `data/raw/`.
- **Integration testing với real data**: sẽ là ad-hoc notebook cell sau khi user commit `macro_monthly.csv` + `macro_quarterly.csv`.
- **Feature pipeline composer** (`src/features/pipeline.py`): defer Session 5 khi có đủ L1+L2+L3+L4.

---

## 2. Source code — chi tiết

### 2.1 `src/data/loaders.py`

#### Triết lý

Bốn nguyên tắc:

1. **Strict schema as contract**: column names, order, types đều match exact. Drift = bug (raise rõ ràng). Không silent coerce hay rename.
2. **Output ready for downstream**: returned DataFrame sorted by `release_date` (datetime64[ns]), no NaT, no duplicates, no NaN in numerics. Caller (`compute_l3_features`) pass thẳng vào `asof_join` không cần preprocessing.
3. **Audit trail via `release_date_source`** (D7): mỗi row tag enum `{"scraped", "fallback_14d"}` (monthly) hoặc `{"scraped", "fallback_30d"}` (quarterly). Reviewer biết được row nào dùng publication date thật, row nào fallback theo conservative convention research_design §4.3.
4. **Generic core + thin wrappers**: `_load` generic function handles cả monthly và quarterly; `load_macro_monthly` / `load_macro_quarterly` chỉ delegate với schema constants khác. Pattern tương tự `asof_join` Session 2.

#### Public API

```python
def load_macro_monthly(path: str | Path) -> pd.DataFrame
def load_macro_quarterly(path: str | Path) -> pd.DataFrame
```

#### Schema enforcement (8 validators tách function)

| Validator | Mô tả | Test ref |
|---|---|---|
| `_validate_columns` | Column names + order match expected exactly | tests 4–6 |
| `_coerce_numerics` | Cast str → float64 via `pd.to_numeric(errors="coerce")` (NaN check downstream) | (no dedicated test) |
| `_parse_release_date` | `YYYY-MM-DD` → datetime64[ns]; raise on bad format | test 7 |
| `_validate_release_date_source` | Enum membership check; allowed values differ per schema | tests 9–11 |
| `_validate_reference_period` | Regex `^\d{4}-(0[1-9]\|1[0-2])$` cho monthly hoặc `^\d{4}-Q[1-4]$` cho quarterly | tests 12–13 |
| `_validate_uniqueness` | No duplicate ref_period AND no duplicate release_date | tests 8, 14 |
| `_validate_no_nan_in_numerics` | Any NaN trong numeric column → raise | tests 15–16 |
| (file-level) | `FileNotFoundError` cho path không tồn tại; empty file rejected | tests 17–18 |

#### Output dtype normalization

pandas 2.x `pd.to_datetime(format=...)` trả `datetime64[us]` mặc định. Loader forced cast về `datetime64[ns]` để match expectation của downstream `asof_join` (đã verify Session 2 issue cùng kiểu). Document trong inline comment.

### 2.2 `src/features/l3_macro.py`

#### Public API

```python
# Top-level composer
def compute_l3_features(
    price_df: pd.DataFrame,           # TCB - provides HOSE calendar
    vnindex_df: pd.DataFrame,          # adj_close column
    fx_df: pd.DataFrame,               # rate column
    macro_monthly_df: pd.DataFrame,    # output of load_macro_monthly
    macro_quarterly_df: pd.DataFrame,  # output of load_macro_quarterly
) -> pd.DataFrame                       # 5 columns × |target_index| rows

# Individual feature builders (unit-testable)
def compute_vnindex_return(vnindex_df, target_index) -> pd.Series
def compute_usdvnd_change(fx_df, target_index) -> pd.Series
```

CPI/SBV/GDP composed inline trong `compute_l3_features` via `asof_join` calls — không tách function vì semantics gói gọn trong asof_join primitive (Session 2 đã verified).

#### Anti-leak per feature

| Feature | Mechanism | Where verified |
|---|---|---|
| `vnindex_return` | `log_p.shift(1) - log_p.shift(2)` | `test_vnindex_return_anti_leak` |
| `usdvnd_change` | Same pattern, on FX-ffilled series | `test_usdvnd_change_anti_leak` |
| `cpi_yoy_pct`, `sbv_rate_pct` | `asof_join(release_date_col="release_date")` | Session 2 (test 6 100-day-lag proof) + Session 4 (`test_anti_leak_future_macro_release_not_visible_before_release_date`) |
| `gdp_yoy_pct` | Same as above | Same |

#### Calendar alignment decisions

| Source | Calendar | Alignment to TCB | Rationale |
|---|---|---|---|
| VN-Index | HOSE (same as TCB via vnstock VCI) | **Strict reindex**, raise on missing | Same source ⇒ same calendar invariant. Mismatch = data quality issue must surface immediately. |
| USD/VND FX | Global FX (5-day max gaps) | **reindex + ffill** | FX-closed day economically = "rate unchanged"; log return = 0 trên day đó, capture next FX move on subsequent day. |
| Macro monthly/quarterly | Release-date semantics | asof_join (NO calendar reindex) | Forward-fill by release_date đúng anti-leak primitive. |

#### Implementation notes

- `asof_join` được gọi với 2 value_cols (cpi, sbv) trong cùng 1 call → single merge_asof operation, không tách 2 calls. Efficient.
- Composer không recompute things đã có ở caller (vd target_index lấy từ `price_df.index` once, reuse). Single pass.
- Logging: composer log dict warmup NaN counts per feature để observability dễ debug.

### 2.3 Tests overview

#### `tests/test_loaders.py` — 20 tests, 5 groups

| Group | Tests | Focus |
|---|---|---|
| 1. Happy path | 4 | Valid CSV → expected DataFrame; sort invariant; dtype check; quarterly variant works |
| 2. Schema violations | 3 | Missing col, extra col, wrong order |
| 3. release_date | 2 | Bad date format, duplicates |
| 4. release_date_source enum | 3 | Cross-frequency (`fallback_30d` in monthly), typo case, quarterly mirror |
| 5. reference_period | 3 | Bad format, bad quarter (Q5), duplicates |
| 6. Numerics | 2 | NaN, non-numeric string |
| 7. File errors | 2 | Missing file, empty file |
| 8. Real-world | 1 | Mixed scraped + fallback rows OK |

#### `tests/test_l3_macro.py` — 20 tests, 5 groups

| Group | Tests | Focus |
|---|---|---|
| 1. `compute_vnindex_return` | 4 | Golden, warmup NaN, strict raise on missing, anti-leak |
| 2. `compute_usdvnd_change` | 4 | Golden, FX-gap ffill, pre-FX-start NaN, anti-leak |
| 3. `compute_l3_features` composer | 6 | Columns/index contract, CPI/SBV/GDP forward-fill at specific dates |
| 4. Anti-leak Tier 1 | 2 | Mutate last daily; inject future macro release |
| 5. Input validation | 5 | All required-column-missing paths |

---

## 3. Data

Không applicable cho code session. Acquisition là parallel user task:

**Targets** (user responsibility, parallel):
- `data/raw/macro_monthly.csv`: ~95 rows từ 2018-06 đến hiện tại. Source: GSO (CPI YoY) + SBV website (refinancing rate).
- `data/raw/macro_quarterly.csv`: ~32 rows từ 2018-Q2 đến hiện tại. Source: GSO (GDP YoY).

**Row-by-row convention**:
- Mỗi row monthly = một CPI release event. `release_date` = ngày GSO public CPI report. `sbv_refinancing_rate_pct` = rate hiệu lực tại release_date đó (nếu SBV decree mid-month, rate mới sẽ visible ở row tháng sau — methodology limitation document trong research_design §15).
- Mỗi row quarterly = một GDP release event. `release_date` = ngày GSO public GDP report.
- `release_date_source`:
  - `"scraped"`: tìm được publication date thực trên GSO/SBV.
  - `"fallback_14d"`: dùng `reference_period_end + 14 days` (monthly).
  - `"fallback_30d"`: dùng `quarter_end + 30 days` (quarterly).

**Integration sanity check** (ad-hoc sau khi user commit CSVs):

```python
from src.data.loaders import load_macro_monthly, load_macro_quarterly
from src.features.l3_macro import compute_l3_features
import pandas as pd

price_df    = pd.read_parquet("data/raw/price_tcb.parquet")
vnindex_df  = pd.read_parquet("data/raw/price_vnindex.parquet")
fx_df       = pd.read_parquet("data/raw/fx_usdvnd.parquet")
macro_m     = load_macro_monthly("data/raw/macro_monthly.csv")
macro_q     = load_macro_quarterly("data/raw/macro_quarterly.csv")

l3 = compute_l3_features(price_df, vnindex_df, fx_df, macro_m, macro_q)
print(l3.tail(20))  # eyeball recent values
print(l3.describe())  # sensible ranges?
print(l3.isna().sum())  # warmup NaN counts expected
```

---

## 4. Model

Không applicable.

---

## 5. Verification

### 5.1 Test run result

```
================== test session starts ==================
collected 40 items

tests/test_loaders.py    ....... 20 PASSED
tests/test_l3_macro.py   ....... 20 PASSED

============== tests coverage ===============
Name                       Stmts   Miss  Cover
-----------------------------------------------
src/data/loaders.py           73      0   100%
src/features/l3_macro.py      41      0   100%
-----------------------------------------------
TOTAL                        114      0   100%

============= 40 passed in 0.93s =============
```

Full suite (Sessions 2 + 3 + 4 combined): **110 passed**, không regression.

### 5.2 Locked invariants

| Item | Value | Reference |
|---|---|---|
| Coverage `src/data/loaders.py` | **100%** | §5.1 |
| Coverage `src/features/l3_macro.py` | **100%** | §5.1 |
| Number of tests new | 40 (20 + 20) | |
| Number of validators trong loaders | **8 distinct** | §2.1 schema enforcement table |
| Anti-leak global | Verified via 2 mutation tests | `test_anti_leak_*` |
| Forward-fill semantic | Verified via 3 specific-date asserts (CPI/SBV/GDP) | `test_*_forward_filled_correctly` |

### 5.3 Empirical evidence cho key decisions

#### D3 (USD/VND FX-gap ffill)

`test_usdvnd_change_handles_fx_gap_via_ffill` setup:
- TCB calendar có 5 ngày liên tiếp; FX có data chỉ tại indices [0, 1, 3] (missing index 2 simulate Easter Monday)
- Expected behavior: tại index 3 (sau FX gap), feature = `log(X[2]/X[1]) = log(ffilled[2]/ffilled[1])`. Vì `ffilled[2]=ffilled[1]=23100`, return = 0.
- Test pass ⇒ ffill semantic implement đúng.

#### D4 (VN-Index strict reindex)

`test_vnindex_return_raises_on_missing_dates` setup:
- TCB calendar 30 days; VN-Index missing day 5
- Expected: raise `ValueError("VN-Index is missing")`
- Test pass ⇒ strict mode active, không silent ffill che data quality issue.

#### D7 (release_date_source audit column)

`test_load_macro_monthly_mixed_sources`:
- CSV có 3 rows với mixed `["fallback_14d", "scraped", "fallback_14d"]`
- Loader pass; output `value_counts() = {"fallback_14d": 2, "scraped": 1}`
- ⇒ audit trail accessible cho reviewer.

`test_raises_on_invalid_release_date_source_monthly`:
- CSV monthly có row với `release_date_source = "fallback_30d"` (quarterly-only enum)
- Loader raise `ValueError("invalid release_date_source")`
- ⇒ cross-frequency confusion caught.

---

## 6. Issues encountered và fixes

### Issue 1 — `pd.to_datetime(format="%Y-%m-%d")` returns `datetime64[us]` (pandas 2.x)

**Symptom**: `test_load_macro_monthly_valid` và `test_load_macro_quarterly_valid` fail với:
```
AssertionError: assert dtype('<M8[us]') == 'datetime64[ns]'
```

**Root cause**: Cùng issue đã gặp Session 2. pandas 2.x đổi default resolution của `pd.to_datetime` từ `[ns]` sang `[us]` trong một số code paths (đặc biệt khi specify `format`). Without explicit cast, downstream `asof_join` (sử dụng `pd.merge_asof`) sẽ require dtype match và raise `MergeError`.

**Fix**: trong `_parse_release_date`, explicit cast `.astype("datetime64[ns]")` sau parse. Now loader output dtype guaranteed cross-pandas-versions.

**Lesson reinforce từ Session 2**: pandas 2.x dtype defaults volatile per code-path. Khi loader/parser produce datetime column meant cho `merge_asof` downstream, ALWAYS explicit cast to canonical `[ns]`. Add to mental checklist.

### Issue 2 — Dead defensive branch trong `_coerce_numerics`

**Symptom**: Coverage report 99% — line 214 unreachable:
```python
if df[col].dtype != "float64":
    raise ValueError(...)
```

**Root cause**: `pd.to_numeric(errors="coerce")` luôn return float64 (NaN cho bad values). Dtype check is dead code. NaN check ở `_validate_no_nan_in_numerics` catch cùng case với clearer error message.

**Fix**: remove dead branch. Coverage → 100%.

**Lesson**: defensive code phải reachable + add value. Dead branches inflate code size, mislead reviewers ("oh this case is handled" — no it isn't, because it can't happen). Coverage report là useful tool catch these.

---

## 7. Open questions / next session

### 7.1 Resolved trong Session 4

| Question | Status | Reference |
|---|---|---|
| D1 — Loaders scope (L3 only vs all 3) | ✅ L3 only; L4 defer Session 5 | §1.3, §2.1 |
| D2 — L3 module structure (composer + helpers) | ✅ Top-level composer + 2 individual functions | §2.2 |
| D3 — FX calendar alignment | ✅ Reindex + ffill (semantic "rate unchanged on FX-closed days") | §2.2, §5.3 |
| D4 — VN-Index calendar alignment | ✅ Strict reindex, raise on missing | §2.2, §5.3 |
| D5 — Anti-leak verification strategy | ✅ 2 global tests + per-feature golden | §2.2, §5.2 |
| D6 — L3 column names | ✅ `L3_FEATURE_COLS` frozen | §2.2 |
| D7 — `release_date_source` audit column | ✅ Added, enum-validated, per-frequency-specific | §2.1, §5.3 |
| Real data acquisition timing | ✅ Parallel/async, code-first | §1.3 (and Session 4 prologue) |

### 7.2 Patches required

**`IMPLEMENTATION.md §4.1`** — update `macro_monthly.csv` và `macro_quarterly.csv` schemas (add `release_date_source` row + update note for D7):

```diff
**`macro_monthly.csv`** (manual, GSO + SBV)
 columns: reference_period (str, "YYYY-MM"),
          release_date (str, "YYYY-MM-DD"),
+         release_date_source (str, enum: "scraped" | "fallback_14d"),  # D7 audit (Session 4)
          cpi_yoy_pct (float),
          sbv_refinancing_rate_pct (float)

**`macro_quarterly.csv`** (manual, GSO)
 columns: reference_quarter (str, "YYYY-Qn"),
          release_date (str, "YYYY-MM-DD"),
+         release_date_source (str, enum: "scraped" | "fallback_30d"),  # D7 audit (Session 4)
          gdp_yoy_pct (float)
```

(`tcb_fundamentals.csv` schema cũng nên add `release_date_source ∈ {"scraped", "fallback_45d"}` khi Session 5 implement — tôi sẽ propose lúc đó.)

**`CHANGELOG.md`** — `v0.5.0` entry (snippet inline trong chat).

**`docs/README.md`** — bump Session 4 to Complete, add Session 5 next placeholder.

### 7.3 Risk register cập nhật

- **L3 manual scrape** (user task, parallel): potential issues
  - CPI YoY có thể chỉ public dạng monthly value (CPI 100=base 2014), cần derive YoY = (CPI_t / CPI_{t-12} - 1) * 100. Document trong CSV `release_date_source` nếu derive vs scrape direct YoY.
  - SBV refinancing rate decrees từ 2018 đến nay có thể tốn time scrape; nhiều tháng cùng rate (rate stable trong period dài).
- **Methodology limitation cần document** (research_design §15): SBV decree mid-month chỉ visible ở next CPI release (because schema combines into single row by CPI release_date). Đã noted trong session04 doc §1.2 và sẽ update research_design §15 với 1 dòng.

### 7.4 Mở ra cho Session 5

**Session 5 scope** (proposal sẽ ở session-prologue):
- `src/data/loaders.py` — add `load_tcb_fundamentals` (mirror existing pattern; new schema includes derived-column computation hooks).
- `src/features/l4_fundamentals.py` — 6 fundamental features per research_design §4.4. Derived columns: `total_assets_growth_yoy`, `pe_ratio`, `credit_growth_yoy`, `equity_assets_ratio` (computed at pipeline time, not stored raw).
- `src/features/pipeline.py` — top-level composer cho L1+L2+L3+L4, produces final 21-feature panel matching IMPLEMENTATION §3.
- Real L4 data acquisition (parallel/post user task): manual scrape TCB IR.

**Pre-session work**: nếu user đã commit `data/raw/macro_monthly.csv` + `macro_quarterly.csv` trước Session 5, có thể chạy ad-hoc real-data sanity check để confirm Session 4 code OK end-to-end.

### 7.5 Session 6+ preview

- Session 6: EDA Phase 0 — stationarity tests (ADF/KPSS), class balance, feature correlation heatmap, Phase 0 Elastic Net λ search via 5-fold TimeSeriesSplit.
- Session 7: **LOCK research_design.md** với git tag `v1.0-prereg-locked` + OSF snapshot. Hard milestone 28/05/2026 (end Tuần 2).
- Session 8+: model implementations, walk-forward engine, statistical tests.