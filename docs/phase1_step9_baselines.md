# Step 9 — Baselines (đặt rào cho DM test)

> Sinh dự đoán của **ba baseline tầm thường** trên đúng lịch test walk-forward (Step 8),
> để Step 12 lấy làm **rào** so với model qua Diebold-Mariano. Không học, không GPU —
> thuần quy tắc xác định. Kiến trúc lai như Step 1→7: logic thuần `src/model/baselines.py`
> + runner mỏng `scripts/baselines_phase1.py` (file-in/file-out, exit 0/1, `_*_log.json`).

---

## 1. Vai trò

Lớp nhãn TCB lệch dương và *tăng* theo `k` (`pct_pos` 55.6→59.2%), nên một bộ đoán ngu
ngốc "luôn +1" đã đạt ~56–63% accuracy trên test period. Hệ quả: **accuracy thô của model
vô nghĩa** nếu không chứng minh vượt các rào này một cách có ý nghĩa thống kê. Step 9 dựng
sẵn các rào đó dưới dạng chuỗi dự đoán, để DM test (HAC lag `k−1`) so *chênh lệch loss*
model − baseline, không nhìn accuracy tuyệt đối.

Theo hướng 1 đã chốt (`research_design.md §7.4`): family DM gồm **3 chuỗi dự đoán**
(persistence, dynamic-majority, always-+1). "Analytical-50%" là test phụ ở Step 12, không
sinh ở đây.

---

## 2. Ba baseline

| Baseline | Quy tắc $\hat{y}_{t,k}$ | Phụ thuộc |
| :--- | :--- | :--- |
| `persistence` | $\mathrm{sign}(P_t - P_{t-k})$ = hướng `k` phiên vừa qua | cột nhãn |
| `dyn_majority` | lớp đa số trong train window cuối (refit hàng tuần) | splitter Step 8 |
| `always_pos` | luôn $+1$ | — |

**Persistence không cần giá thô.** $\mathrm{sign}(P_t - P_{t-k})$ đúng bằng nhãn đã hiện
thực hóa `k` phiên trước, $y_{t-k,\,k}$ — tức cột nhãn dịch xuống `k` dòng (`y_k.shift(k)`).
Mọi giá liên quan đều quan sát được tại `t` nên không rò rỉ; quy ước tie→+1 được thừa kế
tự động. Tiện vì `features.parquet` không mang cột giá.

**Dynamic-majority** cần splitter Step 8 để lấy train window mỗi tuần: lấy đa số nhãn `y_k`
trong 1000 phiên train (tie đếm bằng nhau → +1), gán hằng cho cả 5 phiên test của tuần đó.
Buffer gap đảm bảo nhãn train luôn quan sát được nên không dính NaN.

---

## 3. Output

`data/processed/predictions_baseline.parquet` — **long format**, một dòng mỗi `(date, k)`:

```
date | k | y_true | persistence | dyn_majority | always_pos
```

Chỉ gồm các phiên nằm trong **test period walk-forward** của từng `k` (không Phase-0). Tập
`(date, k)` khớp *đúng* tập điểm model sẽ dự đoán ở Step 11 — điều kiện để DM so cùng điểm.
`y_true` giữ NaN ở đuôi `k` phiên (model vẫn dự đoán ở đó cho inference live; Step 12 loại
khi tính DM). Ba cột dự đoán không bao giờ NaN.

---

## 4. Kết quả thật (29/05/2026)

`predictions_baseline.parquet`: **2917 dòng**.

| `k` | dòng | tuần | acc `persistence` | acc `dyn_majority` | acc `always_pos` | `dyn_pos_frac` |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1  | 738 | 153 | 0.5346 | 0.5604 | 0.5604 | 1.0 |
| 5  | 733 | 152 | 0.4808 | 0.5755 | 0.5755 | 1.0 |
| 10 | 728 | 151 | 0.5056 | 0.5961 | 0.5961 | 1.0 |
| 20 | 718 | 149 | 0.5458 | 0.6318 | 0.6318 | 1.0 |

Hai phát hiện phải mang sang Step 11–12:

1. **Rào tăng mạnh theo `k`**: `always_pos` = 56.0 → 57.6 → 59.6 → **63.2%**. Cao nhất đúng
   ở `k=20` — chỗ giả thuyết cho tín hiệu yếu nhất. Model phải vượt rào cao nhất ở nơi khó
   nhất. `persistence` ngược lại chỉ ~48–55% (gần như vô tín hiệu, không momentum khai thác
   được) → đây là rào "dễ", `always_pos`/`dyn_majority` là rào "khó".

2. **`dyn_pos_frac = 1.0` ở mọi `k`** → dynamic-majority **trùng khít** always-+1: không
   một cửa sổ train 1000 phiên nào trong giai đoạn test (windows phủ ~2019→2026) có đa số
   `−1`. Hệ quả cho family Holm `§7.3`: hai baseline coincide → Step 12 báo cáo điều này và
   tránh tính trùng p-value (thực chất chỉ còn 2 rào phân biệt: persistence và majority).

---

## 5. Giải thích mã nguồn

```
scripts/baselines_phase1.py        ← runner: đọc features → build → validate → ghi + log
└── src/model/baselines.py         ← logic thuần
    ├── build_baselines(df)        ← §3: ráp long-format trên phiên test walk-forward mỗi k
    ├── _majority(y)               ← đa số lớp, tie → +1
    └── validate(out, df)          ← §6: 4 check, fail thì raise
```

- **`baselines.py`** thuần DataFrame, import `walk_forward_splits` từ Step 8.
  `HORIZONS=(1,5,10,20)`, `PRED_COLS=(persistence, dyn_majority, always_pos)`,
  `CANONICAL=[date, k, y_true, *PRED_COLS]` là hợp đồng cột.
  - `build_baselines`: mỗi `k`, tính `persistence = y_k.shift(k)` trên toàn frame; lặp các
    cặp `(train_idx, test_idx)` từ splitter để điền `dyn_majority` theo tuần; gom mọi
    `test_idx` rồi cắt ra `(date, k, y_true, persistence, dyn_majority, always_pos)`. Vì
    test rows luôn có index `≥ 1000+k > k`, `persistence` sau `shift(k)` không NaN ở đó.
  - `validate`: chạy 4 check (§6).
- **`baselines_phase1.py`** đọc `features.parquet`, gọi `build_baselines` + `validate`, ghi
  parquet và `_baselines_log.json` (rows/weeks/`y_true_nan`/accuracy mỗi baseline bỏ đuôi
  NaN/`dyn_majority_pos_frac`). Có bootstrap `sys.path` để chạy `python scripts/...`. Bắt
  exception → log lỗi, exit 1 (tiện Phase 2).

---

## 6. Bốn check (hợp đồng chất lượng output)

| Check | Khẳng định |
| :--- | :--- |
| `schema` | Đúng 6 cột đúng tên + thứ tự; `k` integer, 3 cột dự đoán float |
| `pred_domain` | 3 cột dự đoán $\in \{-1,+1\}$, không NaN; `always_pos` toàn +1 |
| `coverage` | Tập `(date,k)` khớp **đúng** lịch walk-forward Step 8 (đối chiếu lại splitter), không trùng, `k ∈ {1,5,10,20}` |
| `y_true_tail_nan` | NaN của `y_true` mỗi `k` đúng `= k` và nằm trọn ở đuôi (theo `date`) |

`coverage` chạy lại `walk_forward_splits` để so tập ngày — bắt mọi lệch giữa baseline và
lịch test mà model sẽ dùng, ngay trước khi ghi.

---

## 7. Quan hệ với leakage & lưu ý

- Step 9 **không** thêm/bớt chốt chặn. Persistence dùng `y_k.shift(k)` (mọi giá quan sát
  được tại `t`); dynamic-majority chỉ đọc nhãn *trong* train window do splitter cấp (đã có
  buffer gap). Không baseline nào nhìn tương lai.
- Accuracy trong log là số *thông tin* (chiều cao rào), không phải tín hiệu predictability;
  predictability chỉ kết luận sau DM test + bootstrap CI ở Step 12.
- Trên dữ liệu hiện tại `dyn_majority ≡ always_pos`; nếu Phase 2 data lớn lên và xuất hiện
  cửa sổ train đa số `−1`, hai baseline sẽ tách ra — code không cần đổi.