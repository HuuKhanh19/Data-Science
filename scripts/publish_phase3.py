"""Phase 3 — publish dashboard tĩnh lên GitHub Pages (/docs).

Copy `web/tcb_dashboard.html` -> `docs/index.html`, commit & push (CHỈ khi có thay đổi).
GitHub Pages (Settings -> Pages -> Deploy from a branch: `main` /docs) tự build.

Model ĐÓNG BĂNG — đây chỉ là bước XUẤT BẢN file tĩnh đã regenerate ở Phase 2, không
đụng model/dữ liệu. Chạy tay khi muốn, hoặc gắn vào job 4h sáng sau phase2_update.

    python scripts/publish_phase3.py [--no-push]

Exit 0 nếu xong (kể cả khi không có gì để publish), 1 nếu lỗi.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "web" / "tcb_dashboard.html"
DOCS = ROOT / "docs"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          check=check, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-push", action="store_true", help="commit nhưng không push (test)")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"✗ chưa có {SRC} — chạy web_app_phase1.py trước.")
        return 1

    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")     # tránh Jekyll xử lý file
    shutil.copyfile(SRC, DOCS / "index.html")
    print(f"→ docs/index.html cập nhật từ web/{SRC.name}")

    _git("add", "docs/index.html", "docs/.nojekyll")
    # chỉ commit khi có thay đổi staged (tránh commit rỗng ngày không có phiên mới)
    if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("• docs không đổi → bỏ qua commit/push.")
        return 0

    stamp = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    _git("commit", "-m", f"publish dashboard {stamp}")
    print(f"• commit: publish dashboard {stamp}")

    if args.no_push:
        print("(--no-push) bỏ qua push.")
        return 0

    r = _git("push", "origin", "main", check=False)
    if r.returncode != 0:
        print("✗ push lỗi:\n" + (r.stderr or r.stdout))
        return 1
    print("✓ pushed → GitHub Pages sẽ tự build lại (~1 phút).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())