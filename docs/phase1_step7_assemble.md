# Step 7 (Phase 1) — Lắp ráp, xử lý NA & Xuất Artifact (Assembly)

Tài liệu này mô tả Step 7 đã làm gì và giải thích mã nguồn. Step 7 là **bước cuối của
chuỗi xử lý dữ liệu Phase 1**: ráp hai mảnh interim (`features_raw` Step 5 + `labels`
Step 6) thành **một artifact** `data/processed/features.parquet` — file đầu tiên nằm
trong `data/processed/`, sẵn sàng cho bước mô hình hóa. Step này **chỉ ráp + cắt + kiểm
định**, không thêm/bớt chốt chặn leakage nào.

---

## 1. Mục tiêu, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Merge feature + nhãn theo `date`; cắt vùng warmup đầu chuỗi (tự suy từ NaN); giữ tail-NaN nhãn cho inference live; xuất artifact đã kiểm định |
| **Input** | `data/interim/features_raw.parquet` (Step 5) + `data/interim/labels.parquet` (Step 6) — cùng spine 1994 phiên |
| **Output** | `data/processed/features.parquet` (`date` + 20 feature + 4 nhãn) + `data/processed/_assemble_log.json` |
| **Lệnh chạy** | `python scripts/assemble_phase1.py` (exit 0 nếu OK, 1 nếu lỗi) |
| **Kết quả** (cập nhật 30/05/2026) | 1742 phiên × 25 cột, `2019-06-07 → 2026-05-28`, cắt 252 warmup, 5/5 check pass |

**Nguyên tắc thiết kế** (nối tiếp Step 1–6):
- *Kiến trúc lai*: logic thuần (`assemble.py`, DataFrame in→out) + runner mỏng
  (`assemble_phase1.py`, file-in/file-out, exit 0/1) → Phase 2 tự động hóa được.
- *Single source of truth*: `FEATURES`/`LABELS` import từ `features.py`/`label.py`,
  không hardcode lại danh sách cột.
- *Tự suy mốc cắt*: ranh giới warmup tính trực tiếp từ NaN của dữ liệu, không đọc log
  Step 5 → robust, không phụ thuộc file phụ.

---

## 2. Ba việc

### 2.1 Lắp ráp (merge theo `date`)
Hai file cùng spine HOSE, cùng 1994 dòng, cùng khóa `date` → merge 1:1
(`validate="one_to_one"`). Kiểm chéo `len(features_raw)==len(labels)` và merged không
rớt dòng (date phải khớp tuyệt đối). Ra `date` + 20 feature + 4 nhãn = **25 cột**, sắp
đúng thứ tự canonical bất kể thứ tự cột vào.

### 2.2 Cắt vùng warmup (xử lý NA)
Phân biệt **hai loại NaN khác hẳn nhau**:
- **Leading NaN ở feature (warmup)** — indicator có độ trễ; ràng buộc nặng nhất là
  `momentum_3_12` (cần 252 phiên). Đầu chuỗi feature chưa tính được → dòng **vô dụng cả
  train lẫn inference** (không có vector đặc trưng) → **xóa hẳn**.
- **Tail NaN ở nhãn** — `k` phiên cuối chưa biết tương lai → dòng có **feature hợp lệ,
  chỉ thiếu `y`** → **giữ lại** cho inference live.

Quy tắc cắt: dòng đầu tiên mà **cả 20 feature đều non-NaN**
(`df[FEATURES].notna().all(axis=1).idxmax()`), bỏ mọi dòng trước. Vì Step 5 đảm bảo
`leading_nan_only` (NaN mỗi feature chỉ là prefix, không hố giữa), qua mốc max-prefix là
cả 20 feature non-NaN cùng lúc → khối liền mạch.

Lần chạy: cắt **252 phiên** → `usable_start` = **2019-06-07**. Mốc này do
`momentum_3_12` chi phối (252 phiên), trùng khớp thời điểm L4 YoY thu thập đủ 4 quý —
đúng "giữa 2019" như thiết kế.

### 2.3 Không giảm chiều
**Không** PCA, **không** feature selection — giữ nguyên 20 feature. Đây là chốt chặn
*khoa học*: tập feature đã pre-registered (`research_design.md`); lọc thêm theo dữ liệu
sẽ thành p-hacking, phá tính confirmatory. Step 7 không đụng cột feature nào.

---

## 3. Kết quả thật (29/05/2026)

`data/processed/features.parquet`: **1742 phiên × 25 cột**, `2019-06-07 → 2026-05-28`.

| Hạng mục | Giá trị |
| :--- | :--- |
| rows_raw → rows_final | 1994 → 1742 (cắt **252** warmup) |
| feature_nan_total | **0** (vùng khả dụng sạch hoàn toàn) |
| tail_nan (nhãn) | `y_1`=1, `y_5`=5, `y_10`=10, `y_20`=20 (đúng `k`, chỉ ở đuôi) |
| pct_pos vùng khả dụng | `y_1`=55.6% · `y_5`=56.13% · `y_10`=57.22% · `y_20`=59.18% |

**Lưu ý khoa học về `pct_pos`.** Trên vùng khả dụng, tỷ lệ +1 **tăng dần theo `k`**
(55.6% → 59.18%) — cao hơn cả toàn chuỗi ở Step 6 (54.7→55.98%), vì cắt warmup bỏ năm
đầu 2018–2019 (nhiều phiên giảm), phần còn lại 2019–2026 drift tăng mạnh, rõ nhất ở
horizon dài. Hệ quả cần ghi nhớ cho phần đánh giá: **rào baseline majority-class
(~56–59%) tăng theo `k`, trong khi giả thuyết là predictability *giảm* theo `k`** → model
phải vượt rào cao nhất đúng ở chỗ tín hiệu kỳ vọng yếu nhất (`k=20`). Đây là lý do
baseline persistence + dynamic-majority (`research_design.md` §7.4) là thước đo bắt buộc;
accuracy thô không nói lên predictability.

---

## 4. Quan hệ với leakage

Step 7 **không** thêm/bớt chốt chặn nào — chỉ ráp và cắt (cắt đầu chuỗi không phải nhìn
tương lai). Trạng thái: 2/4 chốt đã đóng (lag `≤t-1` Step 5; as-of `release_date`
Step 4). Hai chốt còn lại — walk-forward buffer gap `k` và Z-score fit-trên-train —
**cố tình không bake vào file**: `features.parquet` giữ feature ở dạng **thô chưa chuẩn
hóa** để mỗi window ở bước mô hình tự fit `μ,σ` riêng, tránh rò thống kê từ tương lai.

---

## 5. Giải thích mã nguồn

```
scripts/assemble_phase1.py     ← runner: đọc 2 interim → assemble → validate → ghi
└── src/data/assemble.py       ← logic thuần
    ├── assemble(feat, labels) ← merge(date) 1:1 + cắt warmup tự suy + sắp cột canonical
    └── validate(out)          ← §6: chạy 5 check, fail thì raise
```

- **`assemble.py`** thuần DataFrame, không I/O. `CANONICAL = ["date", *FEATURES, *LABELS]`
  (25 cột) là hợp đồng thứ tự. `FEATURES`/`LABELS` import từ module nguồn.
- **`assemble_phase1.py`** đọc 2 interim, gọi `assemble` + `validate`, ghi
  `features.parquet` và `_assemble_log.json` (kèm `rows_warmup_cut`, `usable_start`,
  `feature_nan_total`, `tail_nan`, `pct_pos_usable`). Bắt exception → log lỗi, exit 1.

---

## 6. Năm check (hợp đồng chất lượng output)

| Check | Khẳng định |
| :--- | :--- |
| `schema` | Đúng **25 cột** đúng tên + thứ tự (`date` + 20 feature + 4 nhãn) |
| `features_no_nan` | **0 NaN** trong 20 feature trên toàn vùng khả dụng (cũng bắt lỗ NaN giữa chuỗi nếu Step 5 lọt) |
| `features_finite` | Không feature nào có **±inf** — `no_nan` KHÔNG bắt được inf (vd lỗi `eps_basic_vnd=0` → `P/0=inf` từng lọt tới tận sklearn ở Step 10) |
| `label_tail_nan` | Nhãn NaN **chỉ ở đuôi**, đúng `k` mỗi horizon — phân biệt *ngoại lệ hợp lệ* với *lỗi* |
| `date_key` | `date` không trùng + tăng nghiêm ngặt |

Phòng thủ thượng nguồn (đã test): spine lệch độ dài hoặc `date` không khớp dù cùng độ
dài → `assemble` raise **trước khi ghi** (không tạo artifact hỏng). Lần chạy 29/05/2026:
cả 4 pass.

---

## 7. Lưu ý & việc còn nợ

- **`features.parquet` là artifact cuối của chuỗi xử lý dữ liệu Phase 1** — feature ở
  dạng *thô chưa chuẩn hóa*, chưa chia tập. Chuẩn hóa + walk-forward để bước mô hình.
- Món nợ tồn từ Step 1: đồng bộ `data.md` §2 "Kết quả" với số fetch (cpi 109, gdp 37,
  fund 33) trước khi lock dữ liệu v1.0.

**Tiếp theo — bước mô hình hóa (walk-forward + Z-score per-window):** rolling window 1000
phiên refit hàng tuần, ra buffer gap `k` phiên cuối train (chốt leakage 3/4); fit `μ,σ`
chỉ trên train window mỗi tuần rồi mới apply vào test (chốt 4/4). Bốn model độc lập
(`k∈{1,5,10,20}`) dùng chung 20 feature, khác nhãn. Đánh giá: walk-forward + overlapping
labels + block bootstrap + Diebold-Mariano vs 3 baseline.