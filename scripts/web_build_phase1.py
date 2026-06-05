"""Runner ĐV7 — build payload web local (inference-only).

    python scripts/web_build_phase1.py

Đọc features + tcb_price + predictions_model + results.json + deploy/ → ghi
web/app_data.json (+ _web_build_log.json). exit 0/1. Phase 2: chạy lại sau khi
pipeline cập nhật data → frontend tự đọc app_data.json mới.

Windows: nạp torch TRƯỚC (deploy có thể chứa LSTM cho forward).
"""
from __future__ import annotations

try:
    import torch  # noqa: F401
except Exception:
    pass

import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.web import inference as W  # noqa: E402

PROC = ROOT / "data" / "processed"
FEATURES = PROC / "features.parquet"
PRICE = ROOT / "data" / "raw" / "tcb_price.parquet"
PREDS = PROC / "predictions_model.parquet"
RESULTS = ROOT / "reports" / "results.json"
DEPLOY = ROOT / "deploy"
FWLOG = ROOT / "data" / "forward_log.parquet"
OUT = ROOT / "web" / "app_data.json"
LOG = PROC / "_web_build_log.json"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")
FREEZE_DATE = "2026-05-29"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


def main() -> int:
    log = {"step": "web-build", "ts": datetime.now(TZ).isoformat()}
    try:
        app = W.assemble_app_data(FEATURES, PRICE, PREDS, RESULTS, DEPLOY,
                                  freeze_date=FREEZE_DATE, phase=2,
                                  forward_log_pq=FWLOG)
        W.validate(app)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(_clean(app), ensure_ascii=False, indent=2), encoding="utf-8")

        fwd = {k: (h["forward"] or {}).get("direction", "—")
               for k, h in app["horizons"].items()}
        log.update({"ok": True, "output": str(OUT),
                    "n_price": len(app["price"]),
                    "n_past": {k: len(h["past"]) for k, h in app["horizons"].items()},
                    "forward": fwd,
                    "badges": {k: h["badge"]["tier"] for k, h in app["horizons"].items()}})
        LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"OK — {OUT}  | {len(app['price'])} điểm giá | {app['meta']['summary_label']}")
        for k in ("1", "5", "10", "20"):
            h = app["horizons"][k]
            f = h["forward"] or {}
            print(f"   k={k:<2s} {h['deploy']['model']:<11s}/{h['deploy']['feature_set']:<4s} "
                  f"→ +{k}: {f.get('direction','—')} (p={f.get('proba','—')}) "
                  f"| {h['badge']['tier']}")
        return 0
    except Exception as e:
        log.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc()})
        try:
            LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        print(f"FAIL — {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())