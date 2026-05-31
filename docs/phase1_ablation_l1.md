# Ablation L1-only — lớp giá có đủ không? (thí nghiệm phụ trợ)

> **Exploratory, NẰM NGOÀI pre-registration.** Train lại đúng 3 model × 4 horizon nhưng
> CHỈ dùng lớp **L1** (4 log-return thuần `r1,r5,r10,r20`), bỏ L2/L3/L4, để đo các lớp
> kỹ thuật + vĩ mô + cơ bản có thêm predictability so với chỉ giá hay không. Features đã
> khóa (anti-pattern §12 #1) nên kết quả này **không vào confirmatory results** — nó là
> robustness/supplementary mà `research_design.md §4` đã lường trước ("4 lớp… để hỗ trợ
> ablation analysis"). Chạy sau khi Step 12 đã xong; không sửa module nào đã done.

---

## 1. Câu hỏi & thiết kế

Study chính dùng đủ 20 feature nhưng kết quả negative (limited, 1/4). Câu hỏi tự nhiên:
*liệu predictability mỏng đó đến từ giá thuần, hay các lớp phái sinh (kỹ thuật/vĩ mô/cơ
bản) mới là thứ đóng góp?* Ablation cô lập bằng cách giữ **mọi thứ khác y hệt** (cùng
walk-forward, cùng z-score, cùng kiến trúc model, cùng baseline, cùng Step 12) và chỉ đổi
tập feature xuống còn L1.

**Re-tune riêng cho L1** (`hparams_L1.json`): hparams gốc tune cho không gian 20 chiều; áp
thẳng `num_leaves`/`n_estimators`/`feature_fraction`/`λ` đó lên 4 feature sẽ lẫn *tác động
bỏ feature* với *lệch hyperparameter*. Mỗi tập feature được tune tối ưu riêng → ablation
sạch. File `config/hparams.json` gốc **giữ nguyên**.

---

## 2. Cơ chế (không sửa module đã done)

`walk_forward.feature_columns(df)` suy cột feature từ `df.columns` (mọi cột trừ `date` + 4
nhãn). Nên chỉ cần **truyền `df` đã rút gọn** còn `date + [r1,r5,r10,r20] + [y_1,y_5,y_10,y_20]`
là cả `tune_all` (Step 10) lẫn `run` (Step 11) tự chạy L1-only — không cần sửa một dòng nào
trong `tuning.py`, `train_infer.py`, `walk_forward.py`, `evaluate.py`. Mọi artifact gắn hậu
tố `_L1`, không đè canonical.

| Artifact | Nội dung |
| :--- | :--- |
| `config/hparams_L1.json` | hparams tune riêng cho L1 |
| `data/processed/predictions_model_L1.parquet` | proba L1, cùng khung `(date,k)` với baseline |
| `reports/results_L1.json` | Step 12 chạy trên preds L1 |
| `reports/figures_L1/ablation_predictability.png` | đường acc L1 vs full + rào |
| `data/processed/_ablation_l1_log.json` | bảng so sánh + `pre_registered:false` |

---

## 3. Kết quả thật (31/05/2026)

Overall **y hệt** nhau: cả hai `limited (1/4)`, `predictable=false`, positive duy nhất ở
k=5. So sánh best-model accuracy + `p_adj` vs persistence:

| k | acc L1 | acc full | Δ(full−L1) | p_adj L1 | p_adj full | verdict L1 / full |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1  | 0.5604 | 0.5617 | +0.0014 | 1.000 | 1.000 | False / False |
| 5  | 0.5755 | 0.5755 | **0.000** | 0.0345 | 0.0345 | **True / True** |
| 10 | 0.5418 | 0.5961 | +0.0543 | 1.000 | 0.160 | False / False |
| 20 | 0.6332 | 0.6390 | +0.0057 | 0.745 | 0.630 | False / False |

Không Δ nào lật được verdict. Ba điểm phải đọc kỹ:

### 3.1 k=5 trùng KHÍT từng chữ số → cái positive duy nhất là rỗng

Ở k=5, EN sụp về **always-+1** trong *cả* L1 lẫn full (`acc = pct_pos = 0.5755`,
confusion `tn=0, fn=0`). Always-+1 **không dùng feature nào** → L1 hay 20 feature đều cho
cùng một predictor → cùng DM vs persistence → cùng `p_adj = 0.0345`. Tức positive duy nhất
của toàn study **độc lập hoàn toàn với feature**: nó là luật đa số đánh bại persistence yếu
(§4.1 doc Step 12), không phải kỹ năng dự báo. Ablation xác nhận điều này một cách định lượng
— mạnh hơn lập luận suông.

### 3.2 Khe hở k=10 (+0.054) KHÔNG phải "L2-L4 có tín hiệu"

Phân rã: full-EN sụp về always-+1 (`acc 0.596 = base rate`), còn L1-EN cố dùng 4 return rồi
**tụt dưới base rate** (`acc 0.542, balacc 0.470, mcc −0.087`). Cả hai đều **không** vượt
persistence (verdict False cả hai). "Full hơn L1 ở k=10" thực chất là *full ≈ rào, L1 dưới
rào* — không phải thêm lớp feature thêm sức dự báo.

### 3.3 Model L1 thường THUA cả always-+1 → fit nhiễu, không có tín hiệu để học

balacc / mcc của 3 model trên L1:

| k | EN | LGB | LSTM |
| :-: | :-: | :-: | :-: |
| 1  | 0.50 / 0 | 0.50 / 0 | 0.487 / −0.05 |
| 5  | 0.50 / 0 | 0.496 / −0.06 | 0.482 / −0.04 |
| 10 | 0.470 / −0.09 | 0.470 / −0.09 | **0.410 / −0.19** |
| 20 | 0.50 / 0 | 0.502 / +0.05 | 0.555 / +0.11 |

balacc cao nhất toàn L1 là LSTM k=20 = 0.555 (mcc 0.11) — vẫn không significant vs
persistence (`p_adj 1.0`, CI Δacc chứa 0). Mọi ô khác ≤ 0.5: cố trích hướng từ giá thuần
cho ra model *dưới* luật đa số tầm thường — triệu chứng kinh điển của "không có tín hiệu,
model fit nhiễu".

---

## 4. Kết luận

Bộ feature kỹ thuật (L2) + vĩ mô (L3) + cơ bản (L4) **không** tạo cải thiện predictability
có ý nghĩa thống kê so với chỉ lớp giá L1, ở bất kỳ horizon nào — overall verdict không đổi.
Quan trọng hơn, **không subset feature nào** (L1 hay full) vượt được rào imbalance, và cái
"thắng" k=5 thì **độc lập với feature** vì do always-+1 lái.

Đây là bằng chứng **củng cố** kết luận negative của study chính (nhất quán weak-form
efficiency cho hướng giá TCB): không chỉ 20 feature thất bại trước rào, mà thêm/bớt lớp
feature cũng không cứu được — và phần "predictable" mỏng ở k=5 không quy được cho bất kỳ
feature nào.

---

## 5. Lưu ý

- **Exploratory, ngoài pre-reg.** Không trộn vào confirmatory results; trích vào mục
  robustness/supplementary của báo cáo cuối. Log đã đánh dấu `pre_registered:false`.
- **Không phải bằng chứng "feature vô dụng tuyệt đối"** — chỉ là *không lớp nào đủ mạnh để
  vượt rào imbalance trên dữ liệu/khung này*. Phát biểu đúng phạm vi: ablation về
  predictability nhị phân, không phải về information content tổng quát.
- Cùng caveat cặp `dyn_majority ≡ always_pos` và DM degenerate như Step 12 (kế thừa nguyên
  `evaluate.py`).

**Tái chạy:** `python scripts/ablation_l1_phase1.py --gpu 1` (FULL + in bảng so sánh).
`--models elastic_net lightgbm` để xem nhanh không cần GPU. Mã: `scripts/ablation_l1_phase1.py`
(tự chứa: tune → train+infer → evaluate trên L1, tái dùng module Step 8/10/11/12).