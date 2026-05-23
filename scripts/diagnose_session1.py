"""Diagnostic v2 — ASCII-safe cho Windows cp1252 console.

Output gồm 2 phần:
  - Console stdout: ASCII chỉ (safe)
  - Files: data/raw/_debug/grep_results.txt (Vietnamese OK vì write UTF-8)

Mục tiêu cốt yếu:
  Part 1: Tìm source-specific Finance class (VCIFinance) bypass dynamic_method,
          probe params raw để tìm cách lấy >4 quarters.
  Part 2: Grep dump files cho item_ids.

Usage (PowerShell):
    chcp 65001                              # set UTF-8 console (optional)
    python scripts/diagnose_session1.py > diagnose_output.txt 2>&1
"""
from __future__ import annotations
import inspect
import os
import re
import sys
from pathlib import Path

# Force UTF-8 stdout to avoid cp1252 crashes (Python 3.7+)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def safe_print(s: str) -> None:
    """Print fallback to ASCII if console can't handle."""
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", "replace").decode("ascii"))


def main():
    section("VNSTOCK DIAGNOSTIC v2")

    import vnstock
    print(f"vnstock module: {vnstock.__file__}")
    print(f"Python version: {sys.version}")

    # === Part 1A: List vnstock.api submodules ===
    section("Part 1A: Files in vnstock.api.financial")
    try:
        import vnstock.api.financial as fm
        fm_dir = Path(fm.__file__).parent
        print(f"Module dir: {fm_dir}")
        for item in sorted(fm_dir.iterdir()):
            if item.name.startswith("_"):
                continue
            print(f"  {item.name}")
    except Exception as e:
        print(f"FAILED: {e}")

    # === Part 1B: Try import source-specific Finance ===
    section("Part 1B: Source-specific Finance classes")
    source_modules = [
        "vnstock.api.financial.vci",
        "vnstock.api.financial.kbs",
        "vnstock.api.financial.tcbs",
    ]
    for mod_name in source_modules:
        try:
            mod = __import__(mod_name, fromlist=["*"])
            print(f"\n  [{mod_name}] OK")
            print(f"  Members: {[x for x in dir(mod) if not x.startswith('_')]}")
        except ImportError as e:
            print(f"\n  [{mod_name}] NOT FOUND: {e}")

    # === Part 1C: Inspect Finance.__init__ source ===
    section("Part 1C: Source: Finance.__init__")
    try:
        from vnstock.api.financial import Finance
        try:
            src = inspect.getsource(Finance.__init__)
            print(src[:2500])
        except Exception as e:
            print(f"  Cannot get __init__ source: {e}")
    except Exception as e:
        print(f"  Import failed: {e}")

    # === Part 1D: Inspect @dynamic_method ===
    section("Part 1D: Source: dynamic_method decorator")
    try:
        from vnstock.api.financial import dynamic_method
        src = inspect.getsource(dynamic_method)
        print(src[:2500])
    except Exception as e:
        print(f"  FAILED: {e}")

    # === Part 1E: Probe params on balance_sheet ===
    section("Part 1E: Probe params on Finance.balance_sheet")
    try:
        from vnstock.api.financial import Finance
        fin = Finance(symbol="TCB", period="quarter", source="VCI")
        patterns = [
            ("baseline", {}),
            ("period=year", {"period": "year"}),
            ("lang=en", {"lang": "en"}),
            ("dropna=False", {"dropna": False}),
            ("flow=asc", {"flow": "asc"}),
            ("year_range=(2018,2026)", {"year_range": (2018, 2026)}),
            ("from_year=2018", {"from_year": 2018}),
            ("limit=100", {"limit": 100}),
            ("show_all=True", {"show_all": True}),
            ("get_all=True", {"get_all": True}),
            ("get_all=False", {"get_all": False}),
            ("history=True", {"history": True}),
            ("full_history=True", {"full_history": True}),
            ("year=2018", {"year": 2018}),
            ("from_date=2018-01-01", {"from_date": "2018-01-01"}),
        ]
        for desc, kwargs in patterns:
            try:
                df = fin.balance_sheet(**kwargs)
                if df is None:
                    print(f"  {desc:30s} -> None")
                    continue
                qc = [c for c in df.columns
                      if re.match(r"^\d{4}[-_]?Q\d$", str(c).strip())]
                print(f"  {desc:30s} -> shape={df.shape}, quarters={len(qc)}")
            except TypeError as e:
                msg = str(e).split("\n")[0][:90]
                print(f"  {desc:30s} -> TypeError: {msg}")
            except Exception as e:
                msg = str(e).split("\n")[0][:90]
                print(f"  {desc:30s} -> {type(e).__name__}: {msg}")
    except Exception as e:
        print(f"  Probe failed: {e}")

    # === Part 1F: Probe constructor get_all + other params ===
    section("Part 1F: Probe Finance constructor with various combinations")
    try:
        from vnstock.api.financial import Finance
        ctor_patterns = [
            ("source+symbol+period (default get_all=True)",
             {"source": "VCI", "symbol": "TCB", "period": "quarter"}),
            ("+ get_all=False",
             {"source": "VCI", "symbol": "TCB", "period": "quarter", "get_all": False}),
            ("+ show_log=True (might reveal API URL)",
             {"source": "VCI", "symbol": "TCB", "period": "quarter", "show_log": True}),
        ]
        for desc, kwargs in ctor_patterns:
            try:
                f = Finance(**kwargs)
                df = f.balance_sheet()
                qc = [c for c in df.columns
                      if re.match(r"^\d{4}[-_]?Q\d$", str(c).strip())]
                print(f"  {desc}")
                print(f"    -> quarters={len(qc)}, shape={df.shape}")
            except TypeError as e:
                msg = str(e).split("\n")[0][:90]
                print(f"  {desc}")
                print(f"    -> TypeError: {msg}")
            except Exception as e:
                msg = str(e).split("\n")[0][:90]
                print(f"  {desc}")
                print(f"    -> {type(e).__name__}: {msg}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # === Part 1G: Try direct source-specific instantiation ===
    section("Part 1G: Direct source-specific Finance (if VCIFinance exists)")
    try:
        from vnstock.api.financial.vci import Finance as VCIFinance
        print("  Found VCIFinance class")
        print(f"  Init signature: {inspect.signature(VCIFinance.__init__)}")
        try:
            v = VCIFinance(symbol="TCB", period="quarter")
            df = v.balance_sheet()
            qc = [c for c in df.columns
                  if re.match(r"^\d{4}[-_]?Q\d$", str(c).strip())]
            print(f"  Direct call: quarters={len(qc)}, shape={df.shape}")
        except Exception as e:
            print(f"  Direct call failed: {type(e).__name__}: {str(e)[:100]}")
    except ImportError:
        print("  vnstock.api.financial.vci.Finance NOT importable")

    # === Part 2: Grep dumps — write to FILE (not stdout) to avoid encoding ===
    section("Part 2: Grep dump files (results -> data/raw/_debug/grep_results.txt)")

    debug_dir = Path("data/raw/_debug")
    results_file = debug_dir / "grep_results.txt"
    debug_dir.mkdir(parents=True, exist_ok=True)

    grep_specs = [
        ("balance_sheet: credit_balance + interest_earning_assets",
         "fundamentals_balance_sheet_item_ids.txt",
         ["loan", "credit", "advance", "earning", "interest_bearing"]),
        ("ratio: npl_ratio",
         "fundamentals_ratio_item_ids.txt",
         ["npl", "bad", "non_perform", "impair", "provision"]),
        ("income_statement: net_interest_income verify",
         "fundamentals_income_statement_item_ids.txt",
         ["interest", "income"]),
    ]

    with open(results_file, "w", encoding="utf-8") as out:
        for label, fname, patterns in grep_specs:
            out.write(f"\n{'=' * 70}\n")
            out.write(f"  {label}\n")
            out.write(f"  File: {fname}\n")
            out.write(f"  Patterns: {patterns}\n")
            out.write("=" * 70 + "\n")

            fpath = debug_dir / fname
            if not fpath.exists():
                out.write(f"  FILE NOT FOUND\n")
                print(f"  {label}: FILE NOT FOUND ({fpath})")
                continue

            pat = re.compile("|".join(patterns), re.IGNORECASE)
            n_match = 0
            with open(fpath, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if pat.search(line):
                        out.write(f"  L{i:3d}: {line.rstrip()}\n")
                        n_match += 1
            out.write(f"  ({n_match} matches)\n")
            print(f"  {label}: {n_match} matches written to file")

    print(f"\n  Full results -> {results_file}")
    section("DONE")
    print(f"\nGui toi:")
    print(f"  1. Toan bo file diagnose_output.txt")
    print(f"  2. File {results_file}")


if __name__ == "__main__":
    main()