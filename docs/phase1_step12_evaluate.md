# Step 12 — Đánh giá thống kê (phán quyết predictability)

> Biến `predictions_model.parquet` + `predictions_baseline.parquet` thành câu trả lời khoa
> học: TCB có predictable ở từng horizon `k` không, suy giảm thế nào theo `k`. Không học gì
> thêm — chỉ đo và kiểm định theo protocol đã pre-register (`research_design.md §6–§8`).
> Kiến trúc lai như mọi step: logic thuần `src/model/evaluate.py` + runner mỏng
> `scripts/evaluate_phase1.py` (file-in/file-out, exit 0/1, `_*_log.json`). Thuần CPU.

---

## 1. Vai trò

Lớp nhãn TCB lệch dương và *tăng* theo `k` nên accuracy thô vô nghĩa (Step 9). Step 12 trả
lời predictability **không bằng accuracy tuyệt đối** mà bằng *chênh lệch loss* model −
baseline qua Diebold-Mariano, kèm CI bootstrap chịu được overlapping labels và hiệu chỉnh
đa kiểm định Holm. Trục phán quyết là **persistence** (`§1.3/§8.1` đã khóa); `always_pos`/
`dyn_majority` là rào tham chiếu, có trong bảng nhưng không lái verdict.

---

## 2. Input → Output

Đọc hai parquet cùng khung `(date, k)`, **merge inner**, **bỏ đuôi `k` phiên** có `y_true`
NaN (vùng inference live), suy sign model từ proba (`≥0.5 → +1`, tie→+1 — khớp baseline).

| Artifact | Nội dung |
| :--- | :--- |
| `reports/results.json` | Toàn bộ số liệu (mọi field cho Step 13 web đọc); NaN/Inf ghi `null` để JS parse được |
| `reports/figures/*.png` | `predictability_vs_k`, `dm_heatmap_k*` (3×3 DM\*), `confusion_{model}_k*`, `calibration_{model}_k*` — 29 hình |
| `data/processed/_evaluate_log.json` | rows, per-horizon n_test + predictable, overall, n_figures |

---

## 3. Bốn khối tính (đều theo `research_design.md`)

**Metrics (§6).** Mỗi (model × `k`): Accuracy (primary), Balanced Accuracy, MCC, confusion,
precision/recall mỗi lớp, calibration 10-bin. Số *mô tả*, chưa phán quyết.

**95% CI — stationary block bootstrap (§7.2).** Politis-Romano 1994, mean block `2k`
(={2,10,20,40}), `B=2000`, percentile `[Q₂.₅, Q₉₇.₅]`. Tính cho cả accuracy lẫn
**Δacc(model − persistence)**; Δacc bootstrap **paired** (cùng index resample cho model &
persistence) để CI không phồng sai. Seed 42.

**Diebold-Mariano (§7.1).** `dₜ = L_baseline − L_model` trên 0-1 loss; variance Newey-West
HAC Bartlett lag `q=k−1` (luôn ≥0); hiệu chỉnh small-sample Harvey 1997; một phía, reference
`t_{N−1}`. **Degenerate guard**: `var(d)=0` (model ≡ baseline, vd EN ≡ always_pos) → `dm=NaN`,
`p_raw=1.0`, cờ `degenerate` — không crash/Inf.

**Holm-Bonferroni (§7.3) + test phụ.** Holm trong **từng horizon**, `M_k = 9` (3 model × 3
baseline), khóa pre-reg; NaN/degenerate coi như `p=1.0` (bảo thủ). Test phụ **analytical-50%**
(z-test acc vs 0.5, HAC lag `q`) báo cáo riêng, **ngoài** family Holm.

**Verdict (§8).** Horizon `k` predictable **iff** tồn tại model với `p_adj(persistence) ≤
0.05` **và** CI Δacc(model − persistence) nằm trọn trên 0. Tổng thể: ≥2/4 → predictable
(`strong` ≥3, `multi` =2, `limited` =1, `no_evidence` =0).

---

## 4. Kết quả thật (31/05/2026)

`overall`: **limited (1/4), `predictable = false`**. Chỉ `k=5` positive.

| `k` | n_test | pct_pos | acc EN / LGB / LSTM | best `p_adj` vs persist | verdict |
| :-: | :-: | :-: | :-- | :-: | :-: |
| 1  | 737 | 0.560 | 0.560 / 0.560 / 0.562 | 1.000 | False |
| 5  | 728 | 0.576 | 0.576 / 0.563 / 0.574 | **0.0345** (EN, LSTM) | **True** |
| 10 | 718 | 0.596 | 0.596 / 0.571 / 0.589 | 0.160 (EN; raw 0.018) | False |
| 20 | 698 | 0.632 | 0.632 / 0.633 / 0.639 | 0.630 | False |

Theo đúng luật đã khóa: *limited evidence*, không đạt ngưỡng 2/4 → **TCB không được claim
predictable**.

### 4.1 Positive ở k=5 là rỗng — caveat bắt buộc

Đây là điểm khoa học cốt lõi, **không được để chìm sau con số verdict**:

1. **EN ở mọi `k` chính là `always_pos`** — `acc = pct_pos`, confusion `tn=0, fn=0`,
   balacc=0.50, mcc=0: nó đoán +1 toàn bộ. Cái "thắng persistence" ở k=5 (`p_adj=0.0345`)
   thực chất là **luật đa số đánh bại persistence yếu** (persistence k=5 chỉ ~48%), **không
   phải model học được tín hiệu**. EN vs always_pos ở k=5 là *degenerate* (`p=1.0`) — zero
   edge trên rào imbalance.
2. **Không model nào vượt `always_pos` ở bất kỳ `k` nào.** Mọi `p_raw` vs always_pos đều
   ≥ 0.15 (phần lớn ≥ 0.5); EN luôn degenerate. Bỏ rào persistence "dễ" ra thì **không còn
   tín hiệu nào** trên rào imbalance "khó".
3. **balacc ≈ 0.50, mcc ≈ 0 ở cả 12 ô** (cao nhất LSTM k=20: mcc=0.11) — độc lập xác nhận
   model không phân biệt được up/down, chỉ bám lớp đa số.

**Đọc trung thực:** kết quả **nhất quán với weak-form EMH** (`§1.4` trường hợp negative).
Blip k=5 là artefact của việc pre-reg chọn persistence (rào yếu) làm trục quyết định, trong
khi always_pos (rào imbalance) mới là bar model trượt ở mọi horizon — đúng nghịch lý "rào
tăng theo k" cảnh báo từ Step 9. Verdict pre-reg **giữ nguyên** (k=5 hợp lệ theo §8.1); phần
discussion của báo cáo phải mang caveat này (nghĩa vụ minh bạch §8).

### 4.2 Thuế multiple-testing (minh hoạ k=10)

EN ở k=10 có raw `p=0.018` vs persistence (sẽ "significant" nếu nhìn trần), nhưng Holm
`M=9` đẩy lên `p_adj=0.160` → không reject. Đây là lý do correction là bắt buộc: 9 test/
horizon, không sửa thì dễ dương tính giả.

---

## 5. Giải thích mã nguồn

```
scripts/evaluate_phase1.py        ← runner: đọc 2 parquet → build → validate → ghi json + figures + log
└── src/model/evaluate.py         ← logic thuần, không I/O
    ├── align(model_df, base_df)  ← merge (date,k), bỏ đuôi NaN, suy sign model
    ├── metrics(y, ŷ)             ← acc/balacc/mcc/confusion/precision/recall
    ├── hac_var_mean(d, q)        ← Newey-West Bartlett lag q (≥0)
    ├── dm_test(L_b, L_m, q)      ← DM + Harvey, một phía; degenerate→NaN+flag
    ├── analytical_50(correct, q) ← z-test acc vs 0.5 (test phụ)
    ├── _sb_indices / boot_ci     ← stationary bootstrap, Δacc paired
    ├── calibration_bins(...)     ← reliability 10-bin
    ├── holm(pmap, M=9)           ← Holm trong horizon
    ├── build_results(df)         ← ráp results.json + verdict §8 + predictability_curve
    └── validate(res)             ← §6 dưới đây, raise nếu fail
```

- `evaluate.py` thuần (không đọc/ghi file). `HORIZONS`, `MODELS` (en/lgb/lstm → cột proba),
  `BASELINES`, `DECISION_BASELINE="persistence"` là hợp đồng. Mọi hằng số kiểm định
  (`B=2000`, `BOOT_SEED=42`, `HOLM_M=9`, `block_len=2k`) khóa cứng theo pre-reg.
- `evaluate_phase1.py` đọc parquet, gọi `align`+`build_results`+`validate`, **ghi
  `results.json` TRƯỚC** rồi mới sinh figures (best-effort: lỗi vẽ chỉ log warning, không
  hỏng artifact khoa học). `_clean` ép numpy/NaN→JSON-safe. Bắt exception → log + exit 1.

---

## 6. Năm check (hợp đồng chất lượng output)

| Check | Khẳng định |
| :--- | :--- |
| `aligned` | `(date,k)` model ≡ baseline sau drop NaN; sign ∈ {−1,+1}; mỗi horizon đủ 3 model |
| `dm_finite_or_flagged` | mọi DM hoặc finite, hoặc NaN **kèm** cờ `degenerate`/`coincident` (không Inf lọt) |
| `holm_monotone` | `p_adj` không giảm theo thứ tự sắp xếp; đúng `M_k = 9` test/horizon |
| `ci_order` | mọi CI `lo ≤ hi`; `delta_acc` nằm trong CI của chính nó |
| `verdict_consistent` | `verdict.predictable` = (cond-1 ∧ cond-2) §8.1; `overall` đúng ngưỡng 2/4 |

Lần chạy 31/05/2026: 5/5 pass, 29 figures, `results.json` parse sạch.

---

## 7. Lưu ý & việc còn nợ

- **Cặp baseline trùng** (`dyn_eq_always_all=true`): giữ đủ 9 dòng DM/horizon, gắn cờ
  `coincident` cho `dyn_majority`+`always_pos` (DM/p trùng khít), `M_k=9` nguyên vẹn — minh
  bạch thay vì âm thầm gộp.
- **cond-3 §8.1** (sensitivity λ×{.5,1,2}, window {750,1000,1250}) **ngoài scope Step 12** —
  cần chạy lại Step 11 với cấu hình khác; ghi rõ ở `meta.sensitivity_note` để không hiểu
  nhầm là đã làm. Chỉ kích hoạt nếu kết quả primary là positive borderline (hiện không phải).
- **Step 12 thuần CPU** — không cần torch nên không dính thứ tự import DLL như Step 11.

**Tiếp theo — Step 13 (web local, deliverable Phase 1):** đọc `predictions_*` + `results.json`
dựng app `localhost`: (a) inference hôm nay (4 dự đoán `k∈{1,5,10,20}` cho phiên mới nhất);
(b) trực quan kết quả khoa học — đường predictability-vs-`k`, DM heatmap, confusion,
calibration — và **hiển thị caveat always_pos** kèm verdict, không claim trading utility.