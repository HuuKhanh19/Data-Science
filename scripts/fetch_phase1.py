"""
CLI runner cho Bước 1 — Data Acquisition (Phase 1, static one-shot).

Chạy fetch 3 channels và lưu parquet vào data/raw/.

Usage:
    python scripts/fetch_phase1.py                          # fetch all channels
    python scripts/fetch_phase1.py --skip-macro             # skip Channel B
    python scripts/fetch_phase1.py --skip-fundamentals      # skip Channel C
    python scripts/fetch_phase1.py --start 2018-06-04 --end 2026-05-21
    python scripts/fetch_phase1.py --output-dir data/raw/

Channel B/C dùng CSV templates trong data/raw/_input/. Nếu file chưa tồn tại
hoặc rỗng → channel đó skip (kèm warning).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Thêm project root vào path để import src.data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    PROJECT_START,
    fetch_tcb_price,
    fetch_usdvnd,
    fetch_vnindex,
    load_macro_csv,
    load_tcb_fundamentals_csv,
    save_parquet,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bước 1 — Data Acquisition (Phase 1 static fetch)"
    )
    parser.add_argument("--start", default=PROJECT_START,
                        help=f"Start date YYYY-MM-DD (default: {PROJECT_START})")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Output directory (default: data/raw)")
    parser.add_argument("--input-dir", default="data/raw/_input",
                        help="Manual input CSV directory (default: data/raw/_input)")
    parser.add_argument("--skip-channel-a", action="store_true",
                        help="Skip vnstock + yfinance fetch")
    parser.add_argument("--skip-macro", action="store_true",
                        help="Skip macro CSV loading")
    parser.add_argument("--skip-fundamentals", action="store_true",
                        help="Skip TCB fundamentals CSV loading")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    logger = logging.getLogger("fetch_phase1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)

    fetch_log: dict[str, dict] = {
        "started_at": datetime.now().isoformat(),
        "start_date": args.start,
        "end_date": args.end or _today(),
        "results": {},
    }

    # ====================================================================
    # Channel A: vnstock + yfinance
    # ====================================================================
    if not args.skip_channel_a:
        for name, fn in [
            ("tcb_price", fetch_tcb_price),
            ("vnindex", fetch_vnindex),
            ("usdvnd", fetch_usdvnd),
        ]:
            try:
                df = fn(start=args.start, end=args.end)
                out = output_dir / f"{name}.parquet"
                save_parquet(df, str(out))
                fetch_log["results"][name] = {
                    "status": "ok",
                    "rows": len(df),
                    "date_min": str(df["date"].min().date()),
                    "date_max": str(df["date"].max().date()),
                    "output": str(out),
                }
            except Exception as e:
                logger.exception(f"[{name}] FAILED")
                fetch_log["results"][name] = {"status": "error", "error": str(e)}
    else:
        logger.info("Skipping Channel A (--skip-channel-a)")

    # ====================================================================
    # Channel B: macro CSV
    # ====================================================================
    if not args.skip_macro:
        macro_csv = input_dir / "macro.csv"
        try:
            df_macro = load_macro_csv(macro_csv)
            out = output_dir / "macro.parquet"
            save_parquet(df_macro, str(out))
            fetch_log["results"]["macro"] = {
                "status": "ok",
                "rows": len(df_macro),
                "indicators": sorted(df_macro["indicator"].unique().tolist()),
                "n_inferred": int(df_macro["release_date_inferred"].sum()),
                "output": str(out),
            }
        except FileNotFoundError as e:
            logger.warning(f"[macro] {e} — skipping")
            fetch_log["results"]["macro"] = {"status": "skipped", "reason": str(e)}
        except Exception as e:
            logger.exception("[macro] FAILED")
            fetch_log["results"]["macro"] = {"status": "error", "error": str(e)}
    else:
        logger.info("Skipping Channel B (--skip-macro)")

    # ====================================================================
    # Channel C: TCB fundamentals CSV
    # ====================================================================
    if not args.skip_fundamentals:
        fund_csv = input_dir / "tcb_fundamentals.csv"
        try:
            df_fund = load_tcb_fundamentals_csv(fund_csv)
            out = output_dir / "tcb_fundamentals.parquet"
            save_parquet(df_fund, str(out))
            fetch_log["results"]["tcb_fundamentals"] = {
                "status": "ok",
                "rows": len(df_fund),
                "n_inferred": int(df_fund["release_date_inferred"].sum()),
                "output": str(out),
            }
        except FileNotFoundError as e:
            logger.warning(f"[tcb_fundamentals] {e} — skipping")
            fetch_log["results"]["tcb_fundamentals"] = {"status": "skipped", "reason": str(e)}
        except Exception as e:
            logger.exception("[tcb_fundamentals] FAILED")
            fetch_log["results"]["tcb_fundamentals"] = {"status": "error", "error": str(e)}
    else:
        logger.info("Skipping Channel C (--skip-fundamentals)")

    # Write fetch log sidecar
    fetch_log["finished_at"] = datetime.now().isoformat()
    log_path = output_dir / "_fetch_log.json"
    if log_path.exists():
        # Append to history
        with open(log_path) as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = [history]
    else:
        history = []
    history.append(fetch_log)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Fetch log appended → {log_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("FETCH SUMMARY")
    logger.info("=" * 60)
    n_ok, n_err, n_skip = 0, 0, 0
    for name, res in fetch_log["results"].items():
        status = res["status"]
        if status == "ok":
            n_ok += 1
            logger.info(f"  [OK]      {name}: {res.get('rows', '?')} rows → {res.get('output')}")
        elif status == "error":
            n_err += 1
            logger.info(f"  [ERROR]   {name}: {res.get('error', '')[:100]}")
        else:
            n_skip += 1
            logger.info(f"  [SKIPPED] {name}")
    logger.info("=" * 60)
    logger.info(f"OK: {n_ok} | ERROR: {n_err} | SKIPPED: {n_skip}")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())