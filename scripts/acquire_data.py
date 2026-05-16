"""
One-shot data acquisition script.

Fetches three raw data products and saves to parquet:
  - data/raw/price_tcb.parquet     (TCB OHLCV, primary vnstock, fallback yfinance)
  - data/raw/price_vnindex.parquet (VN-Index close, primary vnstock, fallback yfinance)
  - data/raw/fx_usdvnd.parquet     (USD/VND rate, yfinance only)

Per Session 1 design:
  - Fetchers do STRUCTURAL DQ only; cross-source agreement and gap thresholds
    are NOT checked here (they belong to notebook 00).
  - This script PRINTS summary statistics for human visual inspection — it
    does not raise on suspicious-but-not-impossible patterns.
  - Output is idempotent: rerunning produces equivalent files (overwriting).

Usage
-----
    python scripts/acquire_data.py
    python scripts/acquire_data.py --end-date 2026-05-15
    python scripts/acquire_data.py --dry-run        # fetch + summarize, no write
    python scripts/acquire_data.py --skip-existing  # skip parquets that exist
    python scripts/acquire_data.py --source yfinance  # force yfinance everywhere

NB: TCB acquisition assumes TCB started trading 2018-06-04 (HOSE listing date,
locked in research_design.md section 2.1 and CFG.TCB_START_DATE).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running as `python scripts/acquire_data.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.fetchers import (  # noqa: E402
    fetch_tcb_price,
    fetch_usdvnd,
    fetch_vnindex,
    summary_stats,
)
from src.utils.config import CFG  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seeds import set_global_seed  # noqa: E402

log = get_logger(__name__)


def _ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _print_summary(stats: dict) -> None:
    """Pretty-print summary stats. Visual inspection — no thresholds here."""
    name = stats.pop("name")
    print(f"\n  === {name} ===")
    for k, v in stats.items():
        # Format floats compactly
        if isinstance(v, float):
            print(f"    {k:30s}: {v:,.4f}")
        elif isinstance(v, int):
            print(f"    {k:30s}: {v:,}")
        else:
            print(f"    {k:30s}: {v}")


def _save_parquet(df: pd.DataFrame, path: str, dry_run: bool) -> None:
    if dry_run:
        log.info("DRY-RUN: would save %d rows to %s", len(df), path)
        return
    _ensure_dir(path)
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    size_kb = Path(path).stat().st_size / 1024
    log.info("Saved %d rows to %s (%.1f KB)", len(df), path, size_kb)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire raw price/index/FX data for TCB project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start-date",
        default=CFG.TCB_START_DATE,
        help="ISO start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="ISO end date. Default: today.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "vnstock", "yfinance"],
        default="auto",
        help="Source policy for stock/index. FX always uses yfinance.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and summarize but do not write parquets.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip targets whose parquet already exists on disk.",
    )
    args = parser.parse_args()

    set_global_seed(CFG.SEED)
    end_iso = args.end_date or pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    print("=" * 70)
    print("Data acquisition")
    print(f"  start_date    : {args.start_date}")
    print(f"  end_date      : {end_iso}")
    print(f"  source policy : {args.source}")
    print(f"  dry_run       : {args.dry_run}")
    print(f"  skip_existing : {args.skip_existing}")
    print("=" * 70)

    targets = [
        ("TCB price", CFG.PRICE_TCB_PATH, fetch_tcb_price, {"source": args.source}),
        ("VN-Index", CFG.PRICE_VNINDEX_PATH, fetch_vnindex, {"source": args.source}),
        ("USD/VND", CFG.FX_USDVND_PATH, fetch_usdvnd, {}),
    ]

    all_stats: dict[str, dict] = {}
    for label, out_path, fetcher, extra_kwargs in targets:
        if args.skip_existing and Path(out_path).exists():
            log.info("Skipping %s (file exists: %s)", label, out_path)
            continue

        log.info("--- Fetching %s ---", label)
        try:
            df = fetcher(start=args.start_date, end=end_iso, **extra_kwargs)
        except Exception as e:
            log.error("FAILED to fetch %s: %s", label, e)
            return 1

        stats = summary_stats(df, name=label)
        all_stats[label] = {k: v for k, v in stats.items() if k != "name"}
        _print_summary(stats)
        _save_parquet(df, out_path, dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)

    # Persist a small audit json next to the parquets
    if not args.dry_run and all_stats:
        audit_path = Path(CFG.RAW_DIR) / "_acquisition_summary.json"
        audit_path.write_text(
            json.dumps(
                {
                    "end_date": end_iso,
                    "source_policy": args.source,
                    "stats": all_stats,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        log.info("Wrote audit summary to %s", audit_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
