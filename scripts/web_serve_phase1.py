"""Runner ĐV7 — server tĩnh local cho web/ (inference-only, Phase 1).

    python scripts/web_serve_phase1.py                 # bind 0.0.0.0:8000, mở trình duyệt
    python scripts/web_serve_phase1.py --port 8137
    python scripts/web_serve_phase1.py --host 127.0.0.1 # chỉ localhost (chặt hơn)
    python scripts/web_serve_phase1.py --no-open

Phục vụ file tĩnh trong web/ (index.html + app_data.json). KHÔNG inference lúc chạy —
app_data.json build sẵn bằng web_build_phase1.py. fetch() cần HTTP nên KHÔNG mở
index.html bằng file:// trực tiếp.

LƯU Ý mạng: nếu server chạy trên MÁY KHÁC với trình duyệt (vd server Windows, xem trên
Mac), `127.0.0.1` KHÔNG tới được — bind 0.0.0.0 (mặc định) rồi trên máy xem mở
http://<IP-LAN-server>:<port>/ (IP được in ra dưới đây). Hoặc mở trình duyệt ngay trên
máy chạy server.
"""
from __future__ import annotations

import argparse
import http.server
import socket
import socketserver
import webbrowser
from functools import partial
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"


def _lan_ip() -> str:
    """IP LAN ngoài-loopback (để mở từ máy khác). Không gửi gói thật."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")   # luôn lấy app_data.json mới
        super().end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = cho may khac xem; 127.0.0.1 = chi localhost")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if not (WEB / "index.html").exists() or not (WEB / "app_data.json").exists():
        print("[!] thieu web/index.html hoac web/app_data.json - chay: python scripts/web_build_phase1.py")
        return 1

    handler = partial(_Quiet, directory=str(WEB))
    socketserver.TCPServer.allow_reuse_address = True

    # tu nhay cong neu ban (vd node server chiem 8000)
    port = args.port
    httpd = None
    for cand in range(args.port, args.port + 12):
        try:
            httpd = socketserver.TCPServer((args.host, cand), handler)
            port = cand
            break
        except OSError:
            print(f"  cong {cand} ban, thu {cand + 1}...")
    if httpd is None:
        print(f"[!] khong bind duoc cong {args.port}..{args.port + 11}.")
        return 1

    lan = _lan_ip()
    print("  -- TCB web local (Phase 1) --------------------------")
    print(f"   tren MAY NAY        ->  http://localhost:{port}/")
    if args.host == "0.0.0.0" and lan != "127.0.0.1":
        print(f"   tu MAY KHAC (LAN)   ->  http://{lan}:{port}/")
    print("   (Ctrl+C de dung)")
    print("  -----------------------------------------------------")

    if not args.no_open:
        try:
            webbrowser.open(f"http://localhost:{port}/")
        except Exception:
            pass
    with httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  da dung.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())