# IMPLEMENTATION.md — Thiết kế triển khai mã nguồn

**Tác giả**: Đặng Hữu Khanh
**Repository**: https://github.com/HuuKhanh19/Data-Science
**Trạng thái**: Working document — không pre-registered, chỉnh sửa tự do trong suốt project
**Liên kết**: Tài liệu này định nghĩa **how it was computed**; phương pháp luận **what was analyzed** được lock trong `research_design.md`.

---

## 1. Mục đích và phạm vi

Tài liệu này quy định:
- Tech stack (Python, libraries, versions)
- Folder structure
- Module responsibilities ở mức cao
- Data schemas (raw files + database)
- Critical interface contracts (anti-leakage)
- Configuration + reproducibility setup
- Development workflow với Claude

**KHÔNG** quy định:
- Function signatures chi tiết (phát triển dần qua các chat sessions)
- Implementation logic chi tiết (qua code + docstrings)
- Optimization techniques (refine khi cần)
- UI styling

**Nguyên tắc edit**: file này có thể sửa free trong suốt project. Khi nào sửa methodology (vd: định nghĩa feature L2, threshold predictability), phải bump `research_design.md` version thay vì sửa silently ở đây.

---

## 2. Tech stack

### 2.1 Ngôn ngữ và phiên bản

- **Python 3.11** (pin trong `pyproject.toml`)
- **Type hints**: bắt buộc cho functions public, optional cho internal helpers

### 2.2 Core libraries (pinned trong `requirements.txt` khi lock Phase 0)

| Library | Version pin (tentative) | Purpose |
|---|---|---|
| numpy | ^2.0 | Numerical core |
| pandas | ^2.2 | DataFrame, time series |
| scipy | ^1.13 | Statistical functions |
| scikit-learn | ^1.5 | Elastic Net, RF, metrics, TimeSeriesSplit |
| torch | ^2.4 | LSTM (CUDA support) |
| statsmodels | ^0.14 | HAC variance, ADF/KPSS tests |
| arch | ^7.0 | Block bootstrap (alternative: custom impl) |
| shap | ^0.46 | SHAP explainers |
| vnstock | latest stable | Vietnamese stock data |
| yfinance | ^0.2 | Fallback price data, USD/VND |
| sqlalchemy | ^2.0 | DB ORM |
| streamlit | ^1.35 | Web app |
| pyarrow | ^17 | Parquet I/O |

### 2.3 Tools

- **Git** + GitHub public repo cho version control + pre-registration
- **OSF** cho immutable pre-registration snapshot
- **GitHub Actions** cho daily inference + weekly refit cron
- **HuggingFace Spaces** cho production hosting Streamlit app
- **pytest** cho testing
- **black + ruff** cho code formatting/linting (optional)

---

## 3. Folder structure

```
data-science-tcb/
├── README.md
├── research_design.md          # LOCKED post-Phase 0
├── IMPLEMENTATION.md           # This file, evolving
├── CHANGELOG.md
├── requirements.txt            # Pinned versions
├── pyproject.toml
├── .github/workflows/
│   ├── daily_inference.yml     # Cron 18:00 ICT daily
│   └── weekly_refit.yml        # Cron Sat 02:00 ICT
│
├── data/
│   ├── raw/                    # Source-of-truth, never edit manually
│   │   ├── price_tcb.parquet
│   │   ├── price_vnindex.parquet
│   │   ├── fx_usdvnd.parquet
│   │   ├── macro_monthly.csv   # CPI, SBV rate
│   │   ├── macro_quarterly.csv # GDP
│   │   └── tcb_fundamentals.csv
│   ├── processed/              # Cache features, regenerable
│   │   └── features_full.parquet
│   └── db/
│       └── predictions.sqlite  # Production predictions DB
│
├── src/
│   ├── data/                   # Data acquisition + I/O
│   │   ├── fetchers.py         # vnstock, yfinance
│   │   ├── loaders.py
│   │   └── asof_join.py        # CRITICAL: anti-leakage join
│   ├── features/
│   │   ├── l1_price.py         # 4 lag returns
│   │   ├── l2_technical.py     # 6 technical indicators
│   │   ├── l3_macro.py         # 5 macro variables
│   │   ├── l4_fundamentals.py  # 6 bank fundamentals
│   │   └── pipeline.py         # Compose layers, ablation support
│   ├── models/
│   │   ├── base.py             # BaseModel abstract class
│   │   ├── elastic_net.py
│   │   ├── random_forest.py
│   │   └── lstm.py             # PyTorch implementation
│   ├── eval/
│   │   ├── walk_forward.py     # Rolling 1000-day iterator
│   │   ├── metrics.py          # Accuracy, BalAcc, MCC
│   │   ├── baselines.py        # Persistence, majority, analytical-50
│   │   ├── dm_test.py          # Diebold-Mariano + Newey-West + Harvey
│   │   ├── bootstrap.py        # Stationary bootstrap (Politis-Romano)
│   │   └── holm.py             # Multiple testing correction
│   ├── interpret/
│   │   ├── shap_analysis.py    # SHAP global + temporal
│   │   └── permutation.py      # Permutation importance (cross-check)
│   ├── ablation/
│   │   └── runner.py           # Forward + LOO + alternative ordering
│   ├── storage/
│   │   ├── db.py               # SQLite schema + ORM
│   │   └── artifacts.py        # Model weights serialization
│   ├── app/
│   │   └── streamlit_app.py    # 3 tabs: Live + Evidence + Interpret
│   ├── inference/
│   │   ├── daily.py            # GitHub Actions entry point (18:00 ICT)
│   │   └── weekly_refit.py     # GitHub Actions entry (Sat 02:00 ICT)
│   └── utils/
│       ├── config.py           # Constants centralized
│       ├── logging.py
│       └── seeds.py            # Global seed setup
│
├── notebooks/                  # EDA + Phase 0 work
│   ├── 01_data_quality_eda.ipynb
│   ├── 02_stationarity_tests.ipynb
│   ├── 03_class_balance.ipynb
│   ├── 04_feature_correlations.ipynb
│   └── 05_phase0_lambda_tuning.ipynb
│
├── scripts/                    # One-shot runners (not part of cron)
│   ├── acquire_data.py
│   ├── run_backtest.py
│   ├── run_ablation.py
│   └── compute_shap.py
│
└── tests/                      # pytest
    ├── test_asof_join.py       # CRITICAL: anti-leakage tests
    ├── test_features.py
    ├── test_walk_forward.py
    ├── test_dm_test.py
    └── test_bootstrap.py
```

**Notes**:
- `data/raw/` là source-of-truth, **không edit thủ công sau khi acquire**. Mọi sửa đổi qua re-acquisition script + new commit.
- `data/processed/` regenerable từ `data/raw/` qua feature pipeline.
- `data/db/predictions.sqlite` là append-only (insert mới, không update lịch sử).

---

## 4. Data schemas

### 4.1 Raw data files

**`price_tcb.parquet`** (vnstock primary, yfinance fallback)
```
columns: date (DatetimeIndex), open, high, low, close, adj_close, volume
dtypes:  date: datetime64[ns], prices: float64, volume: int64
sort:    date ascending, no duplicates, only HOSE trading days
range:   2018-06-04 → present
unit:    prices ở NGHÌN VND (canonical). vnstock VCI native; yfinance fallback
         auto-rescaled (/1000) trong fetch_tcb_price. Volume = số cổ phiếu.
note:    close = adj_close (cả hai cùng adjusted basis sau khi resolve Open Q1).
         Schema giữ 2 columns cho debugging/extension needs trong tương lai.
```

**`price_vnindex.parquet`** (vnstock primary, yfinance `^VNINDEX` fallback)
```
columns: date, close, adj_close
dtypes:  date: datetime64[ns], close/adj_close: float64
unit:    index points (vnstock và yfinance đều native; không cần rescale).
         close = adj_close (index không có corporate-action adjustment ở mức value).
range:   vnstock returns toàn history bất kể start param → ~2018-01-16 → present.
         Inner-join với TCB ở pipeline sau sẽ auto-align về TCB's range.
```

**`fx_usdvnd.parquet`** (yfinance `USDVND=X`; vnstock không có FX)
```
columns: date, rate
dtypes:  date: datetime64[ns], rate: float64
unit:    VND/USD (e.g., 23500 = 23,500 VND per 1 USD).
range:   2018-06-04 → present. Calendar follows global FX market (5-day max gap
         around Easter Monday); KHÔNG đồng nhất với HOSE trading calendar.
```

**`macro_monthly.csv`** (manual, GSO + SBV)
```
columns: reference_period (str, "YYYY-MM"),
         release_date (str, "YYYY-MM-DD"),
         release_date_source (str, enum: "scraped" | "fallback_14d"),
         cpi_yoy_pct (float),
         sbv_refinancing_rate_pct (float)
```

**`macro_quarterly.csv`** (manual, GSO)
```
columns: reference_quarter (str, "YYYY-Qn"),
         release_date (str, "YYYY-MM-DD"),
         release_date_source (str, enum: "scraped" | "fallback_30d"),
         gdp_yoy_pct (float)
```

**`tcb_fundamentals.csv`** (manual, TCB IR)
```
columns: reference_quarter (str, "YYYY-Qn"),
         release_date (str, "YYYY-MM-DD"),
         total_assets_bn_vnd (float),
         eps_ttm (float),
         npl_pct (float),
         total_loans_bn_vnd (float),
         nim_pct (float),
         equity_bn_vnd (float)
```

Derived columns (computed at pipeline time, không lưu raw):
- `total_assets_growth_yoy` = (assets_q − assets_{q-4}) / assets_{q-4}
- `pe_ratio` = price_at_release_date / eps_ttm
- `credit_growth_yoy` = (loans_q − loans_{q-4}) / loans_{q-4}
- `equity_assets_ratio` = equity / total_assets

### 4.2 Database schema (SQLite, `predictions.sqlite`)

```sql
CREATE TABLE predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date  TEXT NOT NULL,          -- "YYYY-MM-DD" phiên dự đoán
    horizon          INTEGER NOT NULL,        -- 1, 5, 10, or 20
    model_name       TEXT NOT NULL,           -- 'elastic_net', 'random_forest', 'lstm'
    prob_up          REAL NOT NULL,           -- P(y=+1) ∈ [0, 1]
    pred_class       INTEGER NOT NULL,        -- -1 or +1
    model_version_hash TEXT NOT NULL,         -- FK to model_versions
    created_at       TEXT NOT NULL,           -- ISO 8601 UTC
    UNIQUE(prediction_date, horizon, model_name, model_version_hash)
);

CREATE TABLE actuals (
    prediction_date  TEXT NOT NULL,
    horizon          INTEGER NOT NULL,
    actual_class     INTEGER NOT NULL,        -- -1 or +1
    populated_at     TEXT NOT NULL,           -- when actual became available
    PRIMARY KEY (prediction_date, horizon)
);

CREATE TABLE model_versions (
    version_hash     TEXT PRIMARY KEY,        -- git commit hash + refit timestamp
    refit_timestamp  TEXT NOT NULL,
    training_window_start TEXT NOT NULL,
    training_window_end   TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    horizon          INTEGER NOT NULL,
    hyperparams_json TEXT NOT NULL,
    artifact_path    TEXT NOT NULL            -- path to pickled weights
);

CREATE TABLE events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    event_type     TEXT NOT NULL,             -- 'refit', 'inference', 'data_fetch', 'error'
    severity       TEXT NOT NULL,             -- 'info', 'warning', 'error'
    message        TEXT NOT NULL,
    metadata_json  TEXT
);

CREATE INDEX idx_predictions_date ON predictions(prediction_date);
CREATE INDEX idx_predictions_horizon ON predictions(horizon);
CREATE INDEX idx_actuals_date ON actuals(prediction_date);
```

---

## 5. Key interface contracts

Đây là 4 interfaces cốt lõi — nếu chúng đúng, phần lớn anti-leakage và walk-forward correctness được đảm bảo.

### 5.1 As-of join (CRITICAL — anti-leakage)

**Vấn đề**: Quarterly L4 và monthly L3 có `release_date` muộn hơn `reference_period`. Naive join theo `reference_period` sẽ leak.

**Contract** (`src/data/asof_join.py`):
```python
def asof_join(
    daily_df: pd.DataFrame,             # DatetimeIndex daily
    low_freq_df: pd.DataFrame,          # cols: release_date_col + value_cols
    value_cols: list[str],
    release_date_col: str = "release_date",
) -> pd.DataFrame:
    """
    As-of join low-frequency data vào daily DataFrame.

    Cho mỗi row tại date t trong daily_df:
        Lấy giá trị từ row của low_freq_df với release_date ≤ t lớn nhất.

    Returns daily_df với value_cols được merge in. release_date_col
    bị DROP khỏi output (decision Session 2 D3 — tránh accidentally
    consumed làm feature downstream).

    Quan trọng:
        - release_date_col PHẢI là publication date, NOT reference_period_end.
        - Duplicate release_dates → raise ValueError (D2, không silent dedup).
        - Rows tại t < earliest release_date → NaN trong value_cols.
        - Past latest release_date → forward-fill latest value indefinitely.
        - Coerce right join key về dtype của left để tương thích pandas 2.x.

    Thin wrappers cho call-site clarity (identical behaviour):
        asof_join_quarterly(daily_df, quarterly_df, value_cols, release_date_col)
        asof_join_monthly(daily_df, monthly_df, value_cols, release_date_col)
    """
```

**Test bắt buộc** (`tests/test_asof_join.py`):
24 tests, 100% line coverage. Critical Tier 1 cases:
- t < earliest release_date → NaN
- t == release_date → row đó (≤ inclusive)
- t giữa hai release_dates → earlier release
- Synthetic 100-day lag chứng minh không dùng `reference_period`
- Property test 100 random dates vs manual oracle

Session log: `docs/session02_asof_join.md`.

### 5.2 Walk-forward iterator

**Contract** (`src/eval/walk_forward.py`):
```python
def rolling_walk_forward(
    features: pd.DataFrame,        # Full feature panel, DatetimeIndex
    labels: pd.DataFrame,           # Labels y_{t,k} cho 4 horizons
    horizon: int,                  # k ∈ {1, 5, 10, 20}
    window_size: int = 1000,
    test_start: str = "2022-07-01",
    refit_cadence: str = "W-FRI",   # Weekly, refit cuối tuần
) -> Iterator[FoldData]:
    """
    Yields FoldData tuples cho mỗi tuần trong test period.

    FoldData:
        - train_X: features in [T_w - window_size - horizon, T_w - horizon]
        - train_y: labels matching train_X (label observable at training time)
        - test_dates: 5 trading days của tuần w
        - test_X: features at test_dates (label unknown at inference time)
        - refit_timestamp: T_w

    Deterministic: cùng inputs → cùng iteration sequence.
    """
```

**Critical**: training data có buffer gap = `horizon` ngày ở cuối (loại trừ rows có label phụ thuộc tương lai chưa quan sát được).

### 5.3 Model API (BaseModel)

**Contract** (`src/models/base.py`):
```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseModel(ABC):
    """Common interface cho 3 models: ElasticNet, RandomForest, LSTM."""

    name: str           # 'elastic_net', 'random_forest', 'lstm'
    horizon: int        # k ∈ {1, 5, 10, 20}
    hyperparams: dict   # Frozen từ Phase 0

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Fit weights/structure. Hyperparams already in self."""

    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Returns P(y=+1) ∈ [0, 1] cho mỗi row."""

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Default: threshold 0.5 trên predict_proba."""
        proba = self.predict_proba(X_test)
        return np.where(proba >= 0.5, 1, -1)

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize trained model to disk."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load trained model from disk."""

    def get_version_hash(self) -> str:
        """Unique identifier: hyperparams + training window + git commit."""
        ...
```

LSTM-specific: vì cần sequence input shape `(batch, T=20, n_features)`, không phải tabular, LSTM `fit` và `predict_proba` sẽ internally chuyển DataFrame → sequence tensor. Interface bên ngoài (DataFrame I/O) giữ uniform với 2 models kia để walk-forward code agnostic.

### 5.4 Feature pipeline (with ablation support)

**Contract** (`src/features/pipeline.py`):
```python
def build_features(
    raw_data: dict[str, pd.DataFrame],
    layers: list[str] = ["L1", "L2", "L3", "L4"],
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Compose features từ raw data theo layers được chỉ định.

    Args:
        raw_data: dict với keys {'price_tcb', 'price_vnindex', 'fx',
                                  'macro_monthly', 'macro_quarterly',
                                  'tcb_fundamentals'}
        layers: subset của ['L1', 'L2', 'L3', 'L4'] cho ablation.
                Default: full = ['L1', 'L2', 'L3', 'L4'] = 21 features.
        end_date: cap features at date (cho walk-forward fold construction).

    Returns:
        DataFrame DatetimeIndex daily, columns = features được chọn.
        Đảm bảo: mọi feature tại date t chỉ dùng info ≤ t (verified bởi as-of join).
    """
```

Ablation usage:
- Forward ablation: `build_features(..., layers=["L1"])`, `["L1", "L2"]`, ..., `["L1", "L2", "L3", "L4"]`
- LOO: `build_features(..., layers=["L2", "L3", "L4"])` (drop L1), etc.
- Alternative ordering không cần thay đổi `build_features`, chỉ thay đổi sequence trong ablation runner

---

## 6. Configuration management

**Central config** (`src/utils/config.py`):
```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Config:
    # Data
    TCB_START_DATE: str = "2018-06-04"
    DATA_DIR: str = "data/"

    # Walk-forward
    HORIZONS: tuple[int, ...] = (1, 5, 10, 20)
    WINDOW_SIZE: int = 1000
    TEST_START: str = "2022-07-01"
    REFIT_CADENCE: str = "W-FRI"

    # Models — frozen post-Phase 0
    EN_ALPHA: float = 0.5
    EN_LAMBDA: float = 0.0  # Filled after Phase 0 tuning
    RF_N_ESTIMATORS: int = 500
    RF_MIN_LEAF: int = 5
    LSTM_HIDDEN: int = 32
    LSTM_T: int = 20
    LSTM_DROPOUT: float = 0.2
    LSTM_LR: float = 1e-3
    LSTM_BATCH: int = 32
    LSTM_PATIENCE: int = 10
    LSTM_MAX_EPOCHS: int = 100
    LSTM_INNER_VAL_FRAC: float = 0.15

    # Reproducibility
    SEED: int = 42

    # Statistical tests
    BOOTSTRAP_B: int = 2000
    BOOTSTRAP_BLOCK_MULT: int = 2  # mean block = 2k

    # Holm correction
    N_MODELS: int = 3
    N_BASELINES: int = 3

CFG = Config()
```

**Lý do `frozen=True`**: prevent accidental mutation during runtime. Bất kỳ thay đổi nào require explicit code edit + commit, hỗ trợ pre-registration audit.

---

## 7. Reproducibility setup

**Entry point seed setup** (`src/utils/seeds.py`):
```python
import os
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42) -> None:
    """Gọi đầu mỗi entry point script."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic CUDA (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Phải gọi từ**: `scripts/run_backtest.py`, `scripts/run_ablation.py`, `src/inference/daily.py`, `src/inference/weekly_refit.py`.

---

## 8. Testing strategy

**Tier 1: critical anti-leakage tests** (must pass before any backtest):
- `test_asof_join.py`: verify no future leak in quarterly/monthly joins
- `test_walk_forward.py`: verify buffer gap = horizon ngày, training data không bao gồm test data

**Tier 2: feature correctness**:
- `test_features.py`: golden test với hand-computed values cho RSI, MACD, BB position trên small toy series

**Tier 3: statistical correctness**:
- `test_dm_test.py`: replicate DM test trên Diebold-Mariano (1995) Table 1 example (sanity check implementation)
- `test_bootstrap.py`: CI coverage simulation — 95% CI should cover true mean ~95% of time on synthetic data

**Coverage target**: 70% line coverage cho `src/eval/` và `src/data/asof_join.py`. Lower target acceptable cho `src/app/`.

---

## 9. Logging conventions

- Use Python `logging` module với format: `[timestamp] [level] [module] message`
- Critical events (refit, inference, errors) cũng log vào `events` table trong SQLite
- Log level mặc định: `INFO`. Set `DEBUG` qua env var `LOG_LEVEL=DEBUG`.

---

## 10. Development workflow với Claude

Mỗi session với Claude khi implement:

1. **Đầu session**: Claude đọc `research_design.md` + `IMPLEMENTATION.md` từ project knowledge để có context.
2. **Focus một module mỗi session**: vd "implement `asof_join.py`", "implement Elastic Net model class", "implement DM test".
3. **Cuối session**: nếu interface contract thay đổi đáng kể, update `IMPLEMENTATION.md` (commit), re-upload lên project knowledge cho future sessions.
4. **Nếu phát hiện methodology issue khi code**: KHÔNG silent fix. Stop, discuss, update `research_design.md` với version bump.

Đề xuất thứ tự implement (bottom-up):

| Session | Module | Output |
|---|---|---|
| 1 | `src/data/fetchers.py` + `acquire_data.py` | Data acquisition working end-to-end |
| 2 | `src/data/asof_join.py` + tests | Anti-leakage join verified |
| 3 | `src/features/l1_price.py`, `l2_technical.py` | L1+L2 features ready |
| 4 | `src/features/l3_macro.py`, `l4_fundamentals.py`, `pipeline.py` | Full 21-feature pipeline |
| 5 | EDA notebooks (Phase 0 stationarity, class balance) | Findings → maybe research_design adjustments |
| 6 | `src/models/base.py` + `elastic_net.py` + `random_forest.py` | 2 simpler models |
| 7 | `src/models/lstm.py` + GPU training verification | LSTM works |
| 8 | `src/eval/walk_forward.py` + `metrics.py` + `baselines.py` | Backtest engine |
| 9 | `src/eval/dm_test.py` + `bootstrap.py` + `holm.py` | Statistical inference pipeline |
| 10 | `scripts/run_backtest.py` end-to-end | **Primary results** |
| 11 | `src/ablation/runner.py` + `scripts/run_ablation.py` | Ablation results |
| 12 | `src/interpret/shap_analysis.py` | Interpretability |
| 13 | `src/storage/db.py` + `src/inference/daily.py` + `weekly_refit.py` | Production code |
| 14 | `src/app/streamlit_app.py` | Web app |
| 15 | `.github/workflows/*.yml` + deploy HF Spaces | Cron + live deployment |

---

## 11. Open implementation questions

Câu hỏi để giải quyết khi vào chi tiết coding:

1. **Adjusted close source**: ✅ **RESOLVED Session 1** (16/05/2026). vnstock VCI returns **ADJUSTED close**.
   - Evidence: 0/1985 ngày có `|log return| > 15%` trong toàn bộ lịch sử TCB (2018-06-04 → 2026-05-15), kể cả khi TCB có stock dividend 1:1 trong 2024. Nếu series unadjusted, ngày ex-rights phải drop ~50%.
   - Locked: canonical price unit là **nghìn VND** (vnstock VCI native). yfinance fallback auto-rescaled trong `fetch_tcb_price`.
   - Locked thresholds cho future Session 2+ cross-check fetcher (chưa implement):
     - Cross-source log-return disagreement: WARN > **100 bp (1%)**. Cơ sở: Q99.9 quan sát = 30 bp, max = 317 bp do off-by-one dividend adjustment timing.
     - HOSE calendar gap: WARN > **12 days**, ERROR > **15 days**. Cơ sở: max observed = 10 days (Tết Nguyên đán, 4 events).
   - Known disagreement dates (off-by-one dividend adjustment giữa vnstock và yfinance, KHÔNG flag là bug): 2024-05-21, 2019-03-14, 2019-03-15.
   - Chi tiết: `docs/session01_data_acquisition.md`.

2. **TCB IR scraping**: có RSS feed hay phải scrape HTML? Format báo cáo (PDF vs Excel)? → verify khi acquire L4 data.
3. **LSTM warm-start qua weekly refit**: có nên init từ tuần trước thay vì from scratch? → trade-off đơn giản code vs sample efficiency. Default: train from scratch mỗi tuần.
4. **SQLite vs filesystem cho model artifacts**: 12 models × 208 refits × 4 horizons = ~10K artifacts. Lưu pickle file system hay BLOB trong SQLite? → benchmark sau.
5. **Streamlit caching strategy**: `@st.cache_data` vs `@st.cache_resource` vs no cache? → sau khi build app.
6. **Inference cron retry policy**: nếu vnstock fail tại 18:00, retry? Wait next day? → default: retry 3 lần, gap 5 phút, then fallback yfinance.

---

## Changelog

- **2026-05-15**: Initial draft.
- **2026-05-16**: Session 1 updates. (1) Section 4.1: thêm unit annotation cho 3 raw parquet schemas (nghìn VND, index points, VND/USD). (2) Section 11 Q1: marked RESOLVED — vnstock VCI returns adjusted close, locked canonical price unit + cross-source threshold + HOSE calendar gap thresholds. Chi tiết: `docs/session01_data_acquisition.md`.
