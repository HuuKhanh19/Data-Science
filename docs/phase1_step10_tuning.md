# Step 10 — Hyperparameter tuning (Phase-0)

> Chọn hyperparameter **một lần duy nhất** trên Phase-0 window `[0, 1000)`, khóa vào
> `config/hparams.json`, **không bao giờ chạm test**. Đây là chốt chống p-hacking
> (anti-pattern #7, `research_design §12`). Kiến trúc lai như Step 1–7/9: logic thuần
> `src/model/tuning.py` + runner mỏng `scripts/tune_phase1.py` (file-in/file-out,
> exit 0/1, `_*_log.json`). **CPU-only** — LSTM không tune ở đây nên không đụng GPU/torch.

---

## 1. Vai trò

Tuning là **sự kiện diễn ra đúng một lần trong cả vòng đời project**. Sau Step 10:

- **Step 11** (walk-forward refit hàng tuần): chỉ *học lại weights* trên cửa sổ trượt, **đọc** `hparams.json` chứ không sửa.
- **Phase 2** (tự động hóa): y hệt cơ chế refit của Step 11 trên dữ liệu live, vẫn **đọc cùng `hparams.json` đã khóa**, không tune lại.

Lý do khoa học: nếu re-tune sau khi thấy test (hoặc trên dữ liệu tương lai ở Phase 2), tính confirmatory của pre-registration vỡ — mọi p-value DM ở Step 12 mất ý nghĩa. Muốn tune lại = một **phiên bản nghiên cứu mới** (v1.1) với pre-reg riêng.

---

## 2. Sơ đồ chọn: single time-ordered holdout (KHÔNG k-fold)

Chia Phase-0 thành **một** lát train sớm + **một** lát validation cuối, theo thứ tự thời gian, có **embargo gap `k`** ở ranh giới:

```
[ ---- train [0, 850-k) ---- | gap k | -- val [850, 1000) -- ]
   index 0 .............         đệm       ...... index 999
```

- `val` = 15% cuối Phase-0 = `[850, 1000)` (150 phiên, cố định mọi `k`).
- `train` = `[0, 850-k)` — embargo cắt `k` phiên cuối vì nhãn `y_{t,k}` của chúng phụ thuộc giá trong vùng val.
- Tiêu chí: **min validation `binary_logloss`**, tune **độc lập từng horizon**.

**Vì sao holdout thay vì k-fold** (sửa pre-lock 30/05/2026 so với §5.2/5.3 cũ):

1. **Đóng leakage overlapping-label gọn hơn.** Với nhãn chồng lấn (`k>1`), `k` nhãn cuối train luôn "nhìn lén" sang val. Holdout chỉ có **một** ranh giới → áp embargo `k` một lần là sạch, thay vì 5 ranh giới như k-fold.
2. **Đồng nhất với LSTM.** §5.4 vốn đã dùng "15% cuối training window" làm inner-val → một triết lý validation duy nhất cho cả 3 model.
3. **Gỡ mơ hồ LightGBM.** Trước đây `n_estimators` chọn qua "fold validation *cuối*" — lủng củng. Giờ chỉ một lát val: mỗi tổ hợp grid fit trên train, early-stop và chấm điểm trên cùng lát đó.

Mốc thật: Phase-0 `2019-06-07 → 2023-06-05`; lát val là 150 phiên cuối Phase-0.

---

## 3. Ba model — phạm vi tune khác nhau

| Model | Tune gì | Cố định |
| :--- | :--- | :--- |
| Elastic Net | **chỉ `λ`** — grid `logspace(-4,1,50)`, map `C = 1/(N·λ)` để bám objective pre-reg | `α=0.5`, `saga`, `max_iter=5000` |
| LightGBM | grid **324 tổ hợp** (`num_leaves`×`min_data_in_leaf`×`feature/bagging_fraction`×`λ_l1`×`λ_l2`); `n_estimators` qua **early stopping** (patience 50, trần 2000) | `lr=0.05`, `seed=42`, không `is_unbalance` |
| LSTM | **KHÔNG tune** — ghi config frozen từ `research_design §5.4` | toàn bộ kiến trúc + training hyperparam |

Nhãn `{-1,+1}` → `{0,1}` cho mọi classifier + log-loss. Z-score fit **chỉ trên lát train** (tái dùng `scale_train_test` của Step 8), transform lát val. Tune riêng từng `k` vì nhãn `y_k` khác nhau → `hparams.json` có cấu trúc `{model: {horizon: params}}`.

---

## 4. Kết quả thật (30/05/2026)

`config/hparams.json` (Phase-0 `2019-06-07 → 2023-06-05`, val 150 phiên):

| `k` | `n_train` | EN `λ` | EN val log-loss | LGB `n_estimators` | LGB val log-loss |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 1  | 849 | 0.0569 | 0.6887 | 2  | 0.6843 |
| 5  | 845 | **10.0** ⚠ | 0.6807 | 6  | 0.6721 |
| 10 | 840 | 0.1151 | 0.6845 | 5  | 0.6800 |
| 20 | 830 | 1.5264 | 0.6576 | 12 | 0.6542 |

(`n_train` giảm theo `k` đúng do embargo gap. Bộ LightGBM đầy đủ mỗi `k` lưu trong `hparams.json`.)

Ba điều phải mang sang Step 11–12:

1. **Tín hiệu rất yếu — và đây là kết quả trung thực.** Mọi val log-loss bám sát ngưỡng vô tri ~0.69 (`ln 2 ≈ 0.693`; với base rate 55–59% thì log-loss "đoán theo base rate" cũng cỡ 0.68–0.69). k=1/10 gần chạm trần, k=20 cải thiện rõ nhất nhưng vẫn khiêm tốn. **Đây là số *validation* để chọn hyperparameter, KHÔNG phải phán quyết predictability** — phán quyết nằm ở DM test trên test set (Step 12).

2. **LGB `n_estimators` tí hon (2/6/5/12) là đúng, không phải lỗi.** Early stopping dừng gần như tức thì vì cây hầu như không tìm được gì để học; thêm vòng chỉ làm val log-loss tệ đi. Mô hình thành thật báo "ít thứ để học" — hành vi mong muốn (tránh overfit).

3. **k=5 EN `λ` chạm biên trên (`λ=10`) = max-reg, KHÔNG phải grid hỏng.** Cờ `at_grid_edge` bật đúng. Ở k=5 mô hình tuyến tính muốn regularize tối đa → `C≈1.2e-4` → ép hệ số về ~0 → gần như chỉ còn intercept (đoán base rate). **Quyết định 30/05/2026: chấp nhận `λ=10`, không nới grid** — hàm log-loss đã tiệm cận mức intercept-only, nới `λ>10` chỉ đẩy hệ số sát 0 hơn (đổi ở số thập phân thứ 4), không có cực tiểu nội tại để bracket. Diễn giải khoa học: **không có tín hiệu tuyến tính khai thác được ở k=5**.

---

## 5. Giải thích mã nguồn

```
scripts/tune_phase1.py              runner: đọc features → tune_all → validate → ghi + log
└─ src/model/tuning.py              logic thuần (import từ walk_forward: scale_train_test, …)
     phase0_split(df, k)            single holdout 15% + embargo gap k → (train_idx, val_idx)
     _prepare(...)                  chọn cột → drop NaN nhãn → z-score (fit-train) → encode {0,1}
     tune_elastic_net(...)          quét λ, C=1/(N·λ), min val log-loss; cờ at_grid_edge
     tune_lightgbm(...)             grid 324 × early stopping; n_estimators = best_iteration
     tune_all(df)                   ráp dict {meta, elastic_net, lightgbm, lstm}
     validate(hp, df)               hợp đồng output, fail → raise
     _assert_phase0_finite(df)      fail-fast: chặn ±inf TRƯỚC khi đụng sklearn
```

- **`tuning.py`** thuần, không ghi file. `LGB_GRID` + `EN_LAMBDA_GRID` + `LSTM_CONFIG` là single source of truth của search space (khớp `research_design §5.2–5.4`).
- **`tune_phase1.py`** đọc `features.parquet`, gọi `tune_all` + `validate`, ghi `config/hparams.json` + `config/_tune_log.json` (per-`k`: `λ`, cờ biên, `n_estimators`, val log-loss). Bootstrap `sys.path` để chạy `python scripts/...`. Bắt exception → log lỗi, exit 1 (tiện Phase 2).

---

## 6. Check (hợp đồng chất lượng output)

`validate(hp, df)` chạy 4 check, fail thì raise (runner bắt, exit 1):

| Check | Khẳng định |
| :--- | :--- |
| `structure` | Đủ 3 model, đủ 4 horizon; LSTM config = frozen (`hidden_size=32`) |
| `en_lambda_in_grid` | `λ` mỗi `k` ∈ `[grid_min, grid_max]` |
| `lgb_in_bounds` | Mọi hyperparam LightGBM ∈ grid; `n_estimators ∈ [1, 2000]` |
| `split_no_leakage` | Mỗi `k`: `val_size=150`, gap train↔val đúng `= k`, val không vượt Phase-0 |

Ngoài ra `_assert_phase0_finite` chạy đầu `tune_all`: nếu feature Phase-0 còn `±inf`/`NaN`, raise với **tên feature rõ ràng** thay vì lỗi khó hiểu của sklearn (lưới này sinh ra sau sự cố `pe_ratio=inf`, xem Step 7 check `features_finite`).

---

## 7. Quan hệ với leakage & lưu ý

- Step 10 **chỉ đọc Phase-0** `[0, 1000)`, không chạm test `[1000, 1742)` — chốt chống p-hacking.
- **Z-score fit chỉ trên lát train** trong holdout (giữ đúng kỷ luật Step 8); **embargo gap `k`** đóng overlapping-label tại ranh giới train/val.
- `hparams.json` **khóa vĩnh viễn**: Step 11 và Phase 2 chỉ đọc, không tune lại.
- **Số validation ≠ phán quyết predictability.** Val log-loss ở §4 chỉ để *chọn* hyperparameter; predictability của TCB chỉ kết luận sau DM test (HAC lag `k−1`) + block bootstrap ở Step 12.