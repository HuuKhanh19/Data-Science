# Step 13 — Web local (deliverable cuối Phase 1)

> Phục vụ kết quả đã có (`predictions_model.parquet`, `results.json`, `results_L1.json`)
> qua một **API mỏng** + dashboard local. Không train/đánh giá lại — chỉ đọc artifact.
> Ba quyết định kiến trúc đã chốt: (1) **API mỏng tách khỏi data layer** để Phase 2/3 tái
> dùng; (2) web hiển thị **cả study chính lẫn ablation L1**; (3) panel dự đoán kèm
> **track-record + caveat** trung thực (study negative — không trình bày như tín hiệu giao
> dịch). Kiến trúc lai như mọi step: logic đọc thuần `src/web/data.py` + app mỏng
> `src/web/api.py` + runner `scripts/serve_phase1.py`. Thuần CPU, không GPU.

---

## 1. Vai trò, Input, Output

| | Nội dung |
| :--- | :--- |
| **Mục tiêu** | Web local xem dự đoán + kết quả khoa học; cùng giao diện dùng được cho Phase 2 (real-time) không sửa lại |
| **Input** | `data/processed/predictions_model.parquet` (Step 11), `reports/results.json` (Step 12), `reports/results_L1.json` (ablation, tùy chọn) |
| **Output** | Web tại `http://127.0.0.1:8000` (API JSON + dashboard tĩnh) |
| **Lệnh chạy** | `python scripts/serve_phase1.py` |

Phụ thuộc mới (thêm vào env `ds`): `fastapi`, `uvicorn[standard]`.

---

## 2. Kiến trúc 3 lớp

```
scripts/serve_phase1.py     ← runner uvicorn (--host/--port/--reload)
└── src/web/api.py          ← FastAPI: 6 endpoint mỏng + mount static
    └── src/web/data.py     ← LỚP CÁCH LY: đọc artifact, không tính khoa học
        └── static/index.html  ← dashboard (vanilla JS + Chart.js qua CDN)
```

Tách `data.py` (đọc) khỏi `api.py` (vỏ HTTP) là chủ ý: **Phase 2 chỉ thay ruột `data.py`**
(trỏ vào live store của pipeline tự động), endpoint + frontend giữ nguyên. Cần file rỗng
`src/web/__init__.py` để import package.

### Endpoint

| Method · Path | Trả về |
| :--- | :--- |
| `GET /api/health` | `{status:"ok"}` |
| `GET /api/inference/latest` | dự đoán phiên mới nhất (4 horizon × 3 model) + `mode` + `as_of` + `caveat` |
| `GET /api/inference/history?k=&n=` | N phiên đã hiện thực hóa gần nhất của k: dự đoán vs thực tế + `hit_rate` + `base_rate` |
| `GET /api/results` | `results.json` (study chính) |
| `GET /api/results/ablation` | `results_L1.json` (404 gọn nếu chưa chạy ablation) |
| `GET /` | dashboard tĩnh |

Mọi endpoint thiếu artifact → **404 có thông điệp** ("chạy Step N trước"); frontend tự
hiển thị trạng thái thay vì vỡ. Route `/api/*` khai báo trước `mount("/")` nên static
không che API.

---

## 3. Dashboard (6 mục)

1. **Banner verdict** — `overall` (limited 1/4, không predictable) + câu nhấn: *không model
   nào vượt rào always_pos ở bất kỳ horizon nào → nhất quán weak-form efficiency*.
2. **Dự đoán phiên mới nhất** — nhãn mode ("snapshot tĩnh tới {ngày}" ↔ "live"), bảng 4
   horizon × 3 model (UP/DOWN + proba), **ô caveat đỏ**: minh bạch khoa học, không phải
   khuyến nghị giao dịch.
3. **Lịch sử dự đoán vs thực tế** — selector k; mỗi model một dải 20 ô xanh(đúng)/đỏ(sai) +
   hit-rate; tự gắn cờ *"≈ base rate ⇒ không edge"* khi hit-rate bám base_rate.
4. **Predictability theo k** — chart (acc best vs rào always_pos vs 0.5) + bảng verdict
   (acc, p_adj vs persistence, predictable?).
5. **Diebold-Mariano** — selector k; bảng 3 model × 3 baseline `p_adj` (Holm trong horizon),
   tô xanh ô reject, cờ `≡` (baseline trùng) / `deg` (DM degenerate).
6. **Ablation L1 vs full** — chart overlay + bảng so sánh; tự ẩn nếu thiếu `results_L1.json`.

---

## 4. Tính trung thực (study negative)

Web "xem predict" nhưng kết quả là negative, nên thiết kế **chống ngộ nhận** thay vì phô
dự đoán:

- Panel dự đoán **luôn** kèm caveat; không có nút/ngôn ngữ gợi ý mua-bán.
- Track-record phơi bày trực tiếp: EN (≈ always-+1) có hit-rate **đúng bằng base rate** →
  người xem thấy ngay model không có tín hiệu, đúng kết luận study.
- Banner + mục ablation lặp lại thông điệp khoa học (không vượt always_pos; L1 ≈ full).

---

## 5. Seam Phase 1 → Phase 2 (real-time)

Hợp đồng cố định là **endpoint**; chỉ *độ tươi dữ liệu* phía sau `data.py` đổi:

- **Phase 1 (giờ):** predictions = phiên test walk-forward của snapshot tĩnh; "phiên mới
  nhất" = phiên cuối snapshot (`as_of` hiển thị đúng ngày). `MODE="snapshot"`.
- **Phase 2:** pipeline tự động mỗi ngày fetch phiên mới → tính feature (lag ≤ t−1) → nạp
  model refit tuần gần nhất → predict → ghi **live store**. Đổi `MODE="live"` + trỏ
  `latest_inference`/`inference_history` vào live store là web tự thành real-time —
  **không sửa endpoint/UI**.

Khuyến nghị Phase 2: tách **live-predictions** (log append mỗi ngày) khỏi **backtest
walk-forward** (output Step 11 đông cứng) — đừng trộn một file. `results.json` (khoa học)
vẫn từ backtest; chỉ panel dự đoán đọc live.

---

## 6. Kiểm thử

Test qua `fastapi.testclient` (mock artifact đúng schema): 6 endpoint trả đúng; validate
`k∈{1,5,10,20}` (400 nếu sai); thiếu file → 404 gọn; `/` serve `index.html` (200); route
`/api/*` không bị static che; `as_of` nhất quán mọi k khi spine chung. (`httpx` chỉ cần cho
TestClient — chạy thật bằng uvicorn không cần.)

---

## 7. Khép Phase 1

Step 13 là deliverable cuối: **web app local chạy được**, hiển thị study chính + ablation
+ dự đoán có track-record. Chuỗi Phase 1 hoàn tất:

> Step 1 raw → 2 EDA → 3 clean → 4 integrate → 5 features → 6 label → 7 assemble
> (`features.parquet`) → 8 walk-forward → 9 baselines → 10 tuning → 11 train+infer → 12
> evaluate (`results.json`) → **13 web local**. Phụ trợ: ablation L1-only (exploratory).

Kết luận khoa học Phase 1: **predictability hướng giá TCB không đạt ngưỡng** (limited 1/4,
positive duy nhất ở k=5 do luật đa số lái), không lớp feature nào vượt rào imbalance —
nhất quán weak-form efficiency.

**Tiếp theo — Phase 2:** tự động hóa toàn bộ pipeline (fetch → feature → refit tuần →
predict hằng ngày → ghi live store), tái dùng nguyên API/UI Step 13 ở chế độ `live`.