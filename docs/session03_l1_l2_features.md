# Session 03 — L1 + L2 features

**Ngày**: 18/05/2026
**Phase**: Phase 1 → Phase 3 (Feature engineering bắt đầu)
**Pre-conditions met**:
- Session 1 complete (`data/raw/price_tcb.parquet` available, vnstock adj_close verified)
- Session 2 complete (`src/data/asof_join.py` primitive, 100% coverage)
- IMPLEMENTATION.md §3 folder structure expects `src/features/l1_price.py` và `src/features/l2_technical.py`
- research_design.md §4.1 / §4.2 specs

**Deliverables**:
- `src/features/l1_price.py` — 4 lag returns (~ 90 LOC)
- `src/features/l2_technical.py` — 6 technical indicators (~ 300 LOC)
- `tests/test_l1_price.py` — 15 tests (Tier 1 anti-leak + Tier 2 golden values)
- `tests/test_l2_technical.py` — 31 tests (same strategy, per-indicator coverage)
- This document
- Patches: `research_design.md §4.2` (clarify Momentum 3-12mo formula per D7=B), `CHANGELOG.md` (v0.4.0 entry), `docs/README.md` (index row update)

---

## 1. Mục đích và bối cảnh

### 1.1 Vị trí trong pipeline

L1 (lag returns) và L2 (technical indicators) là hai trong bốn lớp features tổ chức theo research_design.md §4. Chúng dùng **chỉ** giá đóng cửa điều chỉnh từ `price_tcb.parquet` (Session 1) — không cần macro hoặc fundamentals, nên có thể implement & verify standalone trước khi acquire L3 / L4.

```
data/raw/price_tcb.parquet
        │
        ├──► compute_l1_features ──► 4 features  (L1)
        │
        └──► compute_l2_features ──► 6 features  (L2)
                                       │
                          (chờ Session 4+: L3, L4 join via asof_join)
                                       │
                                       ▼
                                features_full.parquet (21 features)
```

### 1.2 Câu hỏi cần trả lời

- Q1 (Tier 1): mọi feature tại trading day `t` có dùng **chỉ** P_{t-1} và sớm hơn không? Đây là điều kiện cần cho mọi predictability claim downstream. → **Yes**, verified bằng 2 anti-leak tests global (L1, L2) + 1 anti-leak test riêng cho momentum 12-3.
- Q2 (Tier 2): mỗi formula có implement đúng theo spec research_design.md §4.1 / §4.2 không? → **Yes**, golden hand-computed values cho 10/10 features.
- Q3: warmup NaN pattern có monotonic với window size không? → **Yes**, validated per-feature trong test suite.

### 1.3 Scope cố ý KHÔNG bao gồm

- **L3 / L4 features**: cần real macro + fundamentals data (Session 4-5).
- **Pipeline composer** (`src/features/pipeline.py`): tổng hợp L1+L2+L3+L4 với asof_join + ablation hooks. Defer Session 5+ khi có đủ 4 lớp.
- **TA-Lib cross-validation**: discussed Session 3 prologue, decided **không dùng** — research_design định nghĩa correctness qua citation papers, không qua TA-Lib API. Hand-computed golden values là source of truth.
- **EDA stationarity tests**: defer Phase 0 / Tuần 2 cuối, sau khi đủ features.

---

## 2. Source code — chi tiết

### 2.1 Triết lý chung cho L1 và L2

Ba nguyên tắc xếp theo ưu tiên:

1. **Anti-leakage by construction**: mọi indicator implement với `.shift(1)` hoặc tương đương ở cuối pipeline. Hai cách equivalent (shift-equivariant property của linear/rolling operations) — chọn cách clearer ở mỗi indicator.
2. **Spec → formula → test**: mỗi feature cite trực tiếp công thức từ research_design.md trong docstring; golden test verify bằng hand-computed value với intermediate steps documented.
3. **Hard-coded hyperparameters as module-level constants**: `MA_SHORT=5`, `MA_LONG=20`, `RSI_PERIOD=14`, `MACD_SPAN_FAST=12`, `MACD_SPAN_SLOW=26`, `MOMENTUM_SHORT_SKIP=63`, `MOMENTUM_LONG_LOOKBACK=252`. Mọi hyperparameter từ research_design lock — không kwargs để tránh accidental tuning (chỉ `period` cho RSI và `span_fast/span_slow` cho MACD expose qua kwargs vì có literature variant; defaults match research_design).

### 2.2 `src/features/l1_price.py`

Public API: `compute_l1_features(price_df, price_col="adj_close") -> pd.DataFrame`.

Output columns (`L1_FEATURE_COLS`, frozen): `r_lag1`, `r_cum5`, `r_cum10`, `r_cum20`.

Implementation: `log_p.shift(1) - log_p.shift(1+k)` for k ∈ {1, 5, 10, 11, 20} → corresponds to `log(P_{t-1}/P_{t-1-k})`.

Validations:
- `price_col` exists in `price_df.columns` → raise `ValueError`
- Non-positive values in price → raise `ValueError` (log undefined)
- NaN in price → allowed, propagates to NaN in dependent rows

Warmup: 21 rows NaN (longest = r_cum20 needs `P_{t-21}`).

### 2.3 `src/features/l2_technical.py`

Public API:
- `compute_l2_features(price_df, price_col="adj_close")` — top-level composer
- Individual indicators (unit-testable): `compute_ma_crossover`, `compute_momentum_12_3`, `compute_bb_position`, `compute_trb_signal`, `compute_rsi_wilder`, `compute_macd_normalized`

Output columns (`L2_FEATURE_COLS`): `ma_crossover`, `momentum_12_3`, `bb_position`, `trb_signal`, `rsi14`, `macd_norm`.

#### Indicator-by-indicator notes

| Indicator | Implementation | Convention | Warmup |
|---|---|---|---|
| `ma_crossover` | `price.rolling(5).mean().shift(1) / price.rolling(20).mean().shift(1)` | Brock-Lakonishok-LeBaron (1992) | 20 |
| `momentum_12_3` | `log_p.shift(63) - log_p.shift(252)` | **D7=B locked**, skip-3 J-T | 252 |
| `bb_position` | `(price.shift(1) - rolling_mean) / (2 * rolling_std)`, `ddof=0` | Bollinger (2001) population std | 20 |
| `trb_signal` | `(p_prev > shift(2).rolling(20).max()).astype(float) - (p_prev < shift(2).rolling(20).min()).astype(float)` | Brock-Lakonishok-LeBaron (1992); ∈ {-1, 0, 1, NaN} | 21 |
| `rsi14` | Wilder (1978) manual smoothing + `.shift(1)` | Decision D5: Wilder canonical, **not** pandas `.ewm(alpha=1/14)` default | 15 |
| `macd_norm` | `(price.ewm(12, adjust=False, min_periods=26).mean() - price.ewm(26, adjust=False, min_periods=26).mean()) / price`, then `.shift(1)` | Decision D6: `adjust=False` canonical | 26 |

#### Wilder's RSI implementation note

Pandas `.ewm(alpha=1/14, adjust=False)` differs slightly từ Wilder (1978) original:
- **Pandas**: y_0 = x_0; y_t = α·x_t + (1-α)·y_{t-1}.
- **Wilder**: y_14 = mean(x_1..x_14); y_t = ((14-1)·y_{t-1} + x_t) / 14 với t > 14.

Implementation manual `_wilder_smooth` follow Wilder exactly. Defensive paths cho input ngắn (`n <= period`) hoặc có NaN trong initial window (positions 1..period) → trả all-NaN safely.

#### Trading Range Breakout sign convention

Spec research_design.md §4.2 ghi:
$$ \mathrm{trb}_t = \mathbb{1}[P_{t-1} > \max_{i=t-21}^{t-2}(P_i)] - \mathbb{1}[P_{t-1} < \min_{i=t-21}^{t-2}(P_i)] $$

Window `[t-21, t-2]` chứa exactly 20 days. Implementation dùng `price.shift(2).rolling(20)` thay vì `price.shift(1).rolling(20)` because the spec window excludes `P_{t-1}` itself (so signal compares P_{t-1} against the 20 days strictly before it).

### 2.4 Anti-leak engineering pattern

Mỗi indicator dùng một trong hai patterns equivalent:
- **Pattern A — shift input**: `(price.shift(1)).rolling(N).mean()`. Used trong BB position (P_prev term explicit).
- **Pattern B — shift output**: `price.rolling(N).mean().shift(1)`. Used trong MA crossover, MACD, RSI (rolling/EMA computed on full series, then result shifted).

Both equivalent for linear/order-preserving operations (rolling mean, EMA, rolling max/min). Test suite verifies the anti-leak property at the *output* level (Tier 1 tests), so both patterns are validated end-to-end regardless of which is used internally.

### 2.5 `tests/test_l1_price.py`

15 tests organized:
- Golden values (5): one per feature column, hand-computed.
- Warmup pattern (1): asserts exact NaN count per feature.
- Anti-leak (2): single-row mutation + full-suffix NaN test.
- Output contract (3): columns, index preservation, custom price_col.
- Input validation (4): missing col, non-positive, negative, NaN-propagation behavior.

### 2.6 `tests/test_l2_technical.py`

31 tests organized:
- Per-indicator golden values + edge cases (4-5 tests each × 6 indicators ≈ 25 tests).
- Top-level composer (6 tests): columns, index, warmup composite, custom col, validation errors.
- Anti-leak global (2 tests): single-row mutation, NaN-suffix poisoning.

Notable golden cases:
- `test_rsi_wilder_alternating_pattern_golden`: hand-derives RSI at 2 positions from Wilder's algorithm step by step (50.0 exact, then 1500/28).
- `test_bb_position_golden_index_20`: closed-form ((1.9 / (2√0.19))) on `[10]*19 + [12]` input.
- `test_momentum_12_3_anti_leak_recent_63_days_dont_affect`: mutate `P_{t-62..t}` and verify momentum unchanged → **direct proof rằng D7=B đã được implement** (under D7=A, P_{t-1} would be used and test fails).

---

## 3. Data

Không applicable. Session 3 dùng synthetic inputs only. Real `price_tcb.parquet` integration verified ad-hoc trong Session 4+ khi compose pipeline.

---

## 4. Model

Không applicable.

---

## 5. Verification

### 5.1 Test run result

```
================== test session starts ==================
platform linux -- Python 3.12.3, pytest-9.0.3
collected 46 items

tests/test_l1_price.py        ....... 15 PASSED
tests/test_l2_technical.py    ....... 31 PASSED

============== tests coverage ===============
Name                           Stmts   Miss  Cover
---------------------------------------------------
src/features/l1_price.py          23      0   100%
src/features/l2_technical.py      79      0   100%
---------------------------------------------------
TOTAL                            102      0   100%

============= 46 passed in 1.08s =============
```

Full suite (Session 2 + 3 combined): **70 passed**, no regressions.

### 5.2 Locked invariants

| Item | Value | Reference |
|---|---|---|
| Coverage `src/features/l1_price.py` | **100%** | §5.1 |
| Coverage `src/features/l2_technical.py` | **100%** | §5.1 |
| Number of tests | **46** total (15 + 31) | |
| Anti-leak guarantee | Verified on both modules via mutation + NaN-suffix tests | |

### 5.3 Empirical evidence cho D7=B implementation

`test_momentum_12_3_anti_leak_recent_63_days_dont_affect` setup:
- Input: `[10.0]*100 + [12.0]*160` (260 rows)
- Baseline momentum at t=252: `log(P_189/P_0) = log(12/10)` ≈ 0.1823
- Mutation: `mutated.iloc[197:] = 99.0` (overwrite all from index 197 onwards)
- Re-compute momentum: assert at t=252 vẫn ≈ 0.1823

**Logic**: if implementation used D7=A (`log(P_{t-1}/P_{t-252})`), at t=252 it would use P_251 which is in the mutated range → momentum would change to `log(99/10)` ≈ 2.30. Test failure would catch the wrong implementation directly. Test passing = D7=B confirmed end-to-end.

---

## 6. Issues encountered và fixes

Không có. Session 3 chạy clean ngay lần đầu, một phần do (a) thiết kế phương án implement carefully trước khi code, (b) phương án anti-leak shift đã được practice ở Session 2 (asof_join), (c) Tier 2 golden tests catch any formula bug trước khi commit.

Một self-correction trong implementation đáng note:
- Initial draft của `_wilder_smooth` placed initial mean at `position 0` (averaging `values.iloc[0:period]`); but with `diff()` output, `values.iloc[0]` is NaN. Corrected to average `values.iloc[1:period+1]` (skip the NaN at position 0) and place initial mean at `position period`. Catches Wilder (1978) original specification correctly.

---

## 7. Open questions / next session

### 7.1 Resolved trong Session 3

| Question | Status | Reference |
|---|---|---|
| D1 — l1 + l2 trong 1 file hay tách? | ✅ Tách 2 files | §2.2, §2.3 |
| D2 — Warmup return NaN hay drop? | ✅ Return NaN | §2.2 |
| D3 — Price column convention | ✅ `adj_close` default | §2.2 |
| D4 — Anti-leak strict P_{t-1} | ✅ Verified end-to-end | §5.3 |
| D5 — RSI Wilder vs pandas ewm | ✅ Wilder manual | §2.3 |
| D6 — MACD ewm adjust | ✅ `adjust=False` | §2.3 |
| D7 — Momentum 12-3 formula | ✅ **B: skip-3** | §5.3 |
| TA-Lib cross-check | ✅ Skipped, golden only | §1.3 |

### 7.2 Patches required

**`research_design.md §4.2`** — fix Momentum row (per D7=B clarification, pre-lock):
- Formula: $\log(P_{t-1}/P_{t-252})$ → $\log(P_{t-63}/P_{t-252})$
- "với cắt window 63 phiên đầu" → "skip 63 phiên gần nhất (3 tháng) để tránh short-term reversal contamination"

**`CHANGELOG.md`** — add `v0.4.0` entry (template inline trong chat patches).

**`docs/README.md`** — index row update: Session 3 → Complete; add Session 4 placeholder.

**`IMPLEMENTATION.md`** — no contract change required. §4 (data schemas) và §5.1 (asof_join) untouched. §10 (implementation order) có thể bump status: Session 3 = L1+L2 done.

### 7.3 Mở ra cho session sau

- **Session 4: L3 macro acquisition + L3 features**. Manual scrape GSO (CPI, GDP) + SBV (refinancing rate); commit `data/raw/macro_monthly.csv` + `data/raw/macro_quarterly.csv`. Build `src/data/loaders.py` (schema validation) + `src/features/l3_macro.py` (compose asof_join + L1 + L2 với macro data).
- **Session 5: L4 fundamentals acquisition + L4 features**. Manual scrape TCB IR (quarterly metrics). Build `src/features/l4_fundamentals.py` + `src/features/pipeline.py` (top-level composer cho cả 4 lớp).
- **Session 6: EDA Phase 0** (notebooks/01_phase0_stationarity_class_balance.ipynb). ADF tests cho mọi feature, class balance reporting, hyperparameter Phase 0 search cho Elastic Net λ.

### 7.4 Risk register

- **L3 manual scrape** (Session 4): GSO / SBV web data có thể tricky (PDF, không có structured API). Fallback plan: vnstock có macro endpoint potentially usable as cross-check, but spec đã decide manual primary. If scrape blocks > 1 day, fall back to vnstock macro + add cross-check protocol.
- **TCB IR release_date** (Session 5): spec quy ước fallback `release_date = reference_quarter_end + 45 days` nếu không scrape được actual date. Document mỗi quarter actual vs fallback.