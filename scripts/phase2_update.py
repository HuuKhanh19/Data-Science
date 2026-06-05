"""Phase 2 — cập nhật inference real-time (model ĐÓNG BĂNG, KHÔNG refit).

Chạy lại tầng DATA tới phiên HOSE mới nhất → cập nhật sổ forward
(`data/forward_log.parquet`: ghi dự đoán +k của phiên mới nhất, chấm các dự đoán đã
đủ k phiên) → rebuild web.

TUYỆT ĐỐI KHÔNG chạy split/tune/baselines/train_infer/evaluate — đó là các bước
chọn & đánh giá model đã CHẠM TEST một lần ở Phase 1. Model đã freeze ở `deploy/`;
`predictions_model.parquet` + `reports/results.json` giữ nguyên (past OOS & badge test
bất biến). Bằng chứng tích luỹ mới = sổ forward (track record live, tách riêng).

    python scripts/phase2_update.py [--skip-fetch] [--no-open]

Exit 0 nếu cập nhật xong, 1 nếu hỏng (tiện gắn Windows Task Scheduler — P2.5).
"""
from __future__ import annotations

# Windows: nạp torch TRƯỚC numpy/pandas để tránh lỗi DLL (c10.dll init) khi load LSTM
# cho forward k=5 — giống web_build/web_app. Không có torch thì k=5 tự bỏ qua, vô hại.
try:
    import torch  # noqa: F401
except Exception:
    pass

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPTS = ROOT / "scripts"
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
DEPLOY = ROOT / "deploy"
FWLOG_PATH = ROOT / "data" / "forward_log.parquet"   # cấp gốc data/ -> KHÔNG bị .gitignore (persist được)
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

HORIZONS = (1, 5, 10, 20)
# Phân loại nguồn fetch: thiết yếu (lỗi -> abort) vs chậm (lỗi -> giữ snapshot cũ, chạy tiếp)
ESSENTIAL_SOURCES = {"tcb_price", "vnindex", "usdvnd"}
SLOW_SOURCES = {"cpi", "gdp", "tcb_fundamentals"}

# Tầng DATA (sau fetch). KHÔNG có step model nào ở đây — cố ý.
DATA_STEPS = [
    "clean_phase1.py",
    "integrate_phase1.py",
    "features_phase1.py",
    "label_phase1.py",
    "assemble_phase1.py",
]


# ──────────────────────────── chạy runner con ────────────────────────────
def _run(script: str, *args: str) -> int:
    print(f"\n>>> {script} {' '.join(args)}".rstrip())
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args]).returncode


def _fetch_with_tolerance() -> bool:
    """Chạy fetch; chấp nhận nguồn CHẬM lỗi (giữ snapshot cũ), abort nếu nguồn THIẾT YẾU lỗi.

    `fetch_phase1.py` chỉ ghi đè parquet khi nguồn đó fetch OK; nguồn lỗi -> parquet cũ
    còn nguyên. Nên ta đọc `_fetch_log.json` để phân loại thay vì chỉ nhìn exit code.
    """
    rc = _run("fetch_phase1.py")
    log_path = RAW / "_fetch_log.json"
    if not log_path.exists():
        print("  ✗ thiếu _fetch_log.json — không phân loại được lỗi fetch.")
        return False
    results = json.loads(log_path.read_text(encoding="utf-8")).get("results", {})
    failed = {name for name, r in results.items() if r.get("status") not in ("ok", "warning")}
    ess_fail = failed & ESSENTIAL_SOURCES
    slow_fail = failed & SLOW_SOURCES
    if slow_fail:
        print(f"  ⚠ nguồn chậm lỗi (giữ snapshot cũ, chạy tiếp): {sorted(slow_fail)}")
    if ess_fail:
        print(f"  ✗ nguồn THIẾT YẾU lỗi -> abort: {sorted(ess_fail)}")
        return False
    # rc có thể !=0 chỉ vì nguồn chậm lỗi — vẫn coi là pass nếu thiết yếu OK.
    if rc != 0 and not failed:
        print(f"  ✗ fetch exit={rc} nhưng log không rõ nguồn nào lỗi -> abort.")
        return False
    return True


# ──────────────────────── forward-log (lắp record là hàm THUẦN để test) ────────────────────────
def build_forward_records(features: pd.DataFrame, manifest: dict, price: pd.DataFrame,
                          run_date, predict_fn) -> pd.DataFrame:
    """Lắp các record forward cho phiên mới nhất, MỖI horizon một dòng.

    `predict_fn(features, k, entry, deploy_dir) -> {from_date, proba, pred, ...}`
    (ở runtime là `inference._predict_forward`; test thì truyền hàm giả).
    `from_price` = adj close tại phiên gốc, tra từ `price` (cột date/adj_close).
    Lỗi 1 horizon (vd thiếu torch cho LSTM) -> bỏ qua horizon đó, không chặn cả mẻ.
    """
    feats = features.copy()
    feats["date"] = pd.to_datetime(feats["date"])
    last_date = feats["date"].iloc[-1]
    px = price.copy()
    px["date"] = pd.to_datetime(px["date"])
    p_at = dict(zip(px["date"], px["adj_close"]))
    from_price = p_at.get(last_date.normalize(), float("nan"))

    rows = []
    for k in HORIZONS:
        entry = manifest.get(str(k))
        if entry is None:
            continue
        try:
            fwd = predict_fn(feats, k, entry, DEPLOY)
        except Exception as e:
            print(f"  ⚠ forward k={k} lỗi: {type(e).__name__}: {e}")
            continue
        rows.append({
            "from_date": last_date.normalize(),
            "k": k,
            "run_date": pd.Timestamp(run_date).normalize(),
            "model": entry["model"],
            "feature_set": entry["feature_set"],
            "proba": fwd["proba"],
            "pred": fwd["pred"],
            "from_price": from_price,
        })
    return pd.DataFrame(rows)


def update_forward_log() -> bool:
    from src.data import forward_log as FL
    from src.web import inference as W

    features = pd.read_parquet(PROC / "features.parquet")
    price = (pd.read_parquet(RAW / "tcb_price.parquet")[["date", "close"]]
             .rename(columns={"close": "adj_close"}))
    manifest = json.loads((DEPLOY / "manifest.json").read_text(encoding="utf-8"))

    new = build_forward_records(features, manifest, price,
                                run_date=datetime.now(TZ).date(), predict_fn=W._predict_forward)
    if new.empty:
        print("  ✗ không sinh được record forward nào.")
        return False

    log = pd.read_parquet(FWLOG_PATH) if FWLOG_PATH.exists() else FL.empty_log()
    log = FL.append_predictions(log, new)
    log = FL.resolve(log, price, price_col="adj_close")
    FWLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(FWLOG_PATH, index=False)

    print(f"  → sổ forward: {FWLOG_PATH}  ({len(log)} dòng)")
    print(FL.summary(log).to_string(index=False))
    return True


# ──────────────────────────── main ────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true", help="bỏ qua fetch (dùng raw đang có)")
    ap.add_argument("--no-open", action="store_true", help="không tự mở trình duyệt (server headless)")
    args = ap.parse_args()

    t0 = datetime.now(TZ)
    print(f"=== Phase 2 update @ {t0.isoformat()} (model FROZEN, KHÔNG refit) ===")

    # 1) DATA refresh
    if not args.skip_fetch:
        if not _fetch_with_tolerance():
            return 1
    else:
        print("\n(skip-fetch) dùng data/raw hiện có.")
    for step in DATA_STEPS:
        if _run(step) != 0:
            print(f"  ✗ {step} thất bại -> abort (transform lỗi = bug thật).")
            return 1

    # 2) forward-log (ghi dự đoán mới + chấm cái đã đủ phiên)
    print("\n>>> cập nhật sổ forward")
    if not update_forward_log():
        return 1

    # 3) rebuild web (app_data.json + dashboard tự chứa). KHÔNG đụng model.
    if _run("web_build_phase1.py") != 0:
        return 1
    web_args = () if not args.no_open else ()   # web_app tự xử lý headless; cờ để dành nếu cần
    if _run("web_app_phase1.py", *web_args) != 0:
        return 1

    print(f"\n=== DONE in {(datetime.now(TZ) - t0).total_seconds():.0f}s ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())