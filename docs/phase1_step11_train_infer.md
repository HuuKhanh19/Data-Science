# Step 11 (Phase 1) — Walk-forward Train + Inference

Tài liệu này mô tả Step 11 đã làm gì và giải thích mã nguồn. Step 11 là **bước mô hình
hóa đầu tiên thật sự chạm test set**: đọc artifact dữ liệu (`features.parquet`) + siêu
tham số đã khóa (`hparams.json`), refit walk-forward hàng tuần, dự đoán `P(y=+1)` cho
**3 model × 4 horizon** trên toàn vùng test `[1000, 1742)`, xuất một file dự đoán đồng
khung với baseline để Step 12 chạy Diebold–Mariano. Step này **chỉ train + infer + ghi
dự đoán** — không tune (tune là sự kiện một lần ở Step 10), không đánh giá thống kê
(để Step 12).

---

## 1. Mục tiêu, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Refit walk-forward + suy luận; xuất `P(y=+1)` của EN/LGB/LSTM cho mọi `(date,k)` trùng khít baseline |
| **Input** | `data/processed/features.parquet` (Step 7) + `config/hparams.json` (Step 10, đã khóa) |
| **Output** | `data/processed/predictions_model.parquet` (long format proba-only) + `data/processed/_train_infer_log.json` |
| **Lệnh chạy** | `python scripts/train_infer_phase1.py [--horizons …] [--models …] [--gpu N]` (exit 0/1) |
| **Kết quả** (30/05/2026) | 2917 dòng = đúng baseline; 153/152/151/149 tuần refit (k=1/5/10/20); 4/4 check pass |

**Nguyên tắc thiết kế** (nối tiếp Step 8–10):
- *Kiến trúc lai*: logic thuần (`train_infer.py`, DataFrame in → DataFrame ra) + runner
  mỏng (`train_infer_phase1.py`, file-in/file-out, exit 0/1).
- *Tune là quá khứ*: chỉ **đọc** `hparams.json`, không chọn lại tham số nào — giữ tính
  confirmatory/pre-registered.
- *Proba-only*: lưu `P(y=+1) ∈ [0,1]`, **không** lưu sign. Sign suy ở Step 12 bằng ngưỡng
  `0.5` (`≥0.5 → +1`, nhất quán tie→+1). Đủ cho 0-1 loss + Brier + calibration.
- *Đồng khung baseline*: tập `(date,k)` output phải khớp **đúng** `predictions_baseline.parquet`
  → DM so trên cùng điểm. `validate` assert điều này.
- *Năm chốt leakage giữ nguyên*: lag ≤t-1 · as-of release_date · buffer gap `k` ·
  z-score fit-trên-train · features_finite. Step 11 tái dùng splitter Step 8, không nới.

---

## 2. Vòng lặp walk-forward

Lõi là vòng lặp lồng: với mỗi horizon `k`, đi qua `walk_forward_splits(date, k)` (Step 8)
**một lần**, mỗi tuần test refit lại cả 3 model rồi dự đoán. Đi một lần/`k` đảm bảo thứ
tự `date` của 3 model trùng nhau tuyệt đối (không lệch hàng khi ghép cột).

```
cho mỗi k ∈ {1,5,10,20}:
  cho mỗi (train_idx, test_idx) ∈ walk_forward_splits(date, k):   # rolling 1000 + gap k
    EN, LGB : build_window (Step 8) → x train/test đã z-score fit-trên-train (cross-sectional)
    LSTM    : _lstm_window           → chuỗi (T=20) đã z-score fit-trên-train (sequential)
    fit 3 model với hparam ĐÃ KHÓA → predict P(y=+1) cho phiên test của tuần
  gom (date, k, y_true, en_proba, lgb_proba, lstm_proba)
```

Mỗi tuần model bị **vứt đi, train lại từ đầu** trên 1000 phiên gần nhất — đúng "auto-refit
hàng tuần" của đề bài. `y_true` lấy thẳng từ cột nhãn (độc lập model), giữ NaN ở đuôi `k`
phiên (model vẫn dự đoán ở đó cho inference live; Step 12 loại khi tính DM).

---

## 3. Ba model

**Elastic Net logistic** (`_fit_predict_en`). Tuyến tính. `λ` khóa ở Step 10, nhưng
`C = 1/(N·λ)` được **tính lại theo `N` của từng window** (~1000): `λ` là hệ số của
objective chuẩn hóa `1/N` trên loss nên bất biến theo `N`, chỉ `C` của sklearn phải đổi —
đây là lý do Step 10 lưu `λ` chứ không lưu `C`. Giữ `penalty='elasticnet'` (faithful với
hparams; sklearn 1.8 deprecate nhưng vẫn đúng tới 1.10 — warning bị nuốt, không đổi API
để tránh rủi ro tương thích cho study pre-registered).

**LightGBM** (`_fit_predict_lgb`). Cây phi tuyến. Nạp thẳng `n_estimators` + grid params
đã chốt ở Step 10; **không** early-stop lại (không có val ở đây — early stopping là việc
của Step 10). `predict_proba[:,1] = P(+1)` (class 1 ↔ +1).

**LSTM** (`_lstm_window` + `_fit_predict_lstm`). Deep, sequential, dùng GPU. Kiến trúc
frozen §5.4 (không tune). Bốn điểm windowing/huấn luyện đã chốt:
- **STRICT**: chuỗi train không thò ra trước mốc `train_start` → bỏ `T−1=19` mẫu train
  đầu mỗi window (giữ đúng tinh thần "window = 1000 phiên", thống nhất với EN/LGB).
- **gap-as-context**: chuỗi test được phép với tay vào vùng gap `k` làm context — feature
  đã quan sát (lag ≤t-1); gap chỉ chặn *nhãn*, không chặn feature.
- **z-score fit-trên-train**: `μ,σ` fit chỉ trên 1000 phiên train, apply cho mọi row dùng
  trong chuỗi (gồm cả gap-rows của chuỗi test).
- Inner-val = 15% cuối train (time-ordered) cho early stopping (patience 10); seed 42 +
  `cudnn.deterministic` best-effort (chấp nhận residual stochasticity GPU như §5.4). Card
  chọn qua `--gpu` (mặc định 0).

EN/LGB có chốt **single-class guard**: window 1 lớp (cực hiếm) → trả proba hằng, tránh
`fit` ném lỗi.

---

## 4. Giải thích mã nguồn

```
scripts/train_infer_phase1.py    ← runner: đọc input → run() → (full: validate + ghi canonical) → log + exit
└── src/model/train_infer.py     ← logic thuần
    ├── run(df, hparams, horizons, models, gpu)  ← vòng lặp §2, ra long-format DataFrame
    ├── _fit_predict_en / _fit_predict_lgb       ← cross-sectional (dùng build_window Step 8)
    ├── _lstm_window / _fit_predict_lstm         ← sequential (chuỗi T=20, GPU)
    ├── _assert_cuda_usable(gpu)                 ← fail-fast kernel sm_120; chọn + test card
    ├── _quiet()                                 ← nuốt warning benign (deprecation, feature-names)
    └── validate(out, baseline)                  ← §5: 4 check, fail thì raise
```

- **`run`** đi qua splitter một lần/`k`, gọi EN/LGB qua `build_window` và LSTM qua
  `_lstm_window`, ghép thành `CANONICAL = [date, k, y_true, en_proba, lgb_proba, lstm_proba]`.
  Model không yêu cầu → cột proba = NaN (chế độ smoke).
- **`train_infer_phase1.py`** có `--horizons`/`--models` (smoke từng phần) và `--gpu`.
  **FULL** = đủ 4 horizon × 3 model → ghi `predictions_model.parquet` + `validate` + log.
  **Smoke** (thiếu horizon/model) → ghi `_smoke_predictions_model.parquet`, **không** đụng
  file canonical, **không** validate chặt (bảo vệ artifact thật). `_read_json` đọc
  `hparams.json` bền encoding (Step 10 ghi cp1252 trên Windows). Log per `(model,k)`:
  acc, zero_one_loss, weeks, rows.

---

## 5. Bốn check (hợp đồng output, chỉ chạy ở chế độ FULL)

| Check | Khẳng định |
| :--- | :--- |
| `schema` | Đúng 6 cột đúng tên + thứ tự; `k` integer, 3 cột proba float |
| `proba_domain` | 3 cột proba **finite**, không NaN, **∈ [0,1]** |
| `coverage` | Tập `(date,k)` khớp **đúng** `predictions_baseline.parquet`; không trùng `(date,k)` |
| `y_true_tail_nan` | `y_true` NaN **chỉ ở đuôi**, đúng `k` mỗi horizon |

`coverage` là điều kiện sống còn để Step 12 so DM cùng điểm. EN/LGB dùng `build_window`
(giống Step 9), LSTM dùng cùng `test_idx` → cả ba khớp baseline. Lần chạy 30/05/2026: 4/4 pass.

---

## 6. Kết quả thật (acc 0-1, ngưỡng 0.5)

| k | tuần | EN | LGB | LSTM | rào `always_pos` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 153 | 0.5604 | 0.5604 | **0.5617** | 0.560 |
| 5 | 152 | 0.5755 | 0.5632 | 0.5742 | 0.576 |
| 10 | 151 | 0.5961 | 0.5710 | 0.5891 | 0.596 |
| 20 | 149 | 0.6318 | 0.6332 | **0.6390** | 0.632 |

Đọc thẳng: **cả ba model bám sát base rate**, không model nào vượt rào rõ rệt. EN gần như
trùng `always_pos` ở k=5/10/20 (≈ luôn dự đoán +1); LGB lệch xuống nhẹ ở k=5/10; LSTM là
model duy nhất nhỉnh trên rào ở k=1 và k=20, nhưng chênh chỉ cỡ vài phần nghìn — trong
biên nhiễu. Đây đúng bức tranh **predictability yếu** đã thấy ở val log-loss Step 10
(bám ~0.69). **Lưu ý**: acc thô KHÔNG phải phán quyết — phán quyết predictability là DM
test trên *chênh lệch loss* ở Step 12 (acc trùng baseline đã hé lộ edge mỏng, nhưng cần
DM xác nhận có/không ý nghĩa thống kê sau Holm).

---

## 7. Lưu ý & việc còn nợ

**Môi trường (sự thật mang theo).** LSTM cần GPU Blackwell (RTX 5070 Ti, sm_120) → phải
**torch bản cu128** + **numpy ≥ 2.0** (numpy 1.x làm `c10.dll` init fail, WinError 1114).
Thêm: trên Windows phải **import torch TRƯỚC** numpy/lightgbm/sklearn (xung đột thứ tự DLL
OpenMP/MKL) — đã ép preload ở đầu cả module lẫn runner. Chọn card bằng `--gpu`.

**Bất biến.** Tune đã đóng băng — Step 11 và Phase 2 chỉ đọc `hparams.json`, không tune
lại. `predictions_model.parquet` đồng khung `predictions_baseline.parquet`.

**Tiếp theo — Step 12 (đánh giá):** so 3 model với baseline bằng **Diebold–Mariano** trên
0-1 loss (HAC lag `k−1`) + **block bootstrap** (block `2k`, do overlapping labels) +
hiệu chỉnh **Holm** (`M_k=9` trong từng horizon). Đọc `predictions_model.parquet` +
`predictions_baseline.parquet` (cùng `(date,k)`, loại đuôi NaN), kết luận predictability
TCB suy giảm theo `k` đến đâu.