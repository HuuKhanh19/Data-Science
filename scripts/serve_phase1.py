"""Runner Step 13 — chạy web local.

    python scripts/serve_phase1.py                 # http://127.0.0.1:8000
    python scripts/serve_phase1.py --port 9000 --reload

Đọc artifact tĩnh (predictions_model.parquet, results.json, results_L1.json) qua API mỏng
src/web/api.py. Không train/đánh giá lại — chỉ phục vụ kết quả đã có.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload khi sửa code (dev)")
    args = ap.parse_args()
    print(f"[serve] http://{args.host}:{args.port}  (Ctrl+C để dừng)")
    uvicorn.run("src.web.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())