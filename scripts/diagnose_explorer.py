"""Final probe: vnstock.explorer.vci để xem có pagination/full-history method không.

Mục đích: confirm Option B feasibility (probe direct provider class).

Usage:
    python scripts/diagnose_explorer.py > explorer_output.txt 2>&1
"""
from __future__ import annotations
import inspect
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    section("VNSTOCK EXPLORER PROBE")

    import vnstock
    print(f"vnstock path: {vnstock.__file__}")

    # === 1. Find explorer directory ===
    section("1. Explorer directory structure")
    pkg_dir = Path(vnstock.__file__).parent
    explorer_dir = pkg_dir / "explorer"
    print(f"explorer dir: {explorer_dir}")
    print(f"exists: {explorer_dir.exists()}")
    if explorer_dir.exists():
        for item in sorted(explorer_dir.iterdir()):
            if item.is_dir() and not item.name.startswith("_"):
                print(f"  {item.name}/")
                for sub in sorted(item.iterdir()):
                    if not sub.name.startswith("_") and sub.is_file():
                        print(f"    {sub.name}")

    # === 2. Import vci.financial ===
    section("2. Import vnstock.explorer.vci.financial")
    try:
        import vnstock.explorer.vci.financial as vci_fin
        print(f"OK: {vci_fin.__file__}")
        print(f"\nMembers:")
        for name in sorted(dir(vci_fin)):
            if name.startswith("_"):
                continue
            obj = getattr(vci_fin, name)
            print(f"  {name}: {type(obj).__name__}")
    except ImportError as e:
        print(f"NOT FOUND: {e}")
        return

    # === 3. Find Finance/Financial class ===
    section("3. Find VCI Finance provider class")
    candidate_names = ["Finance", "FinancialReport", "Financial", "BalanceSheet"]
    provider_cls = None
    provider_name = None
    for cname in candidate_names:
        cls = getattr(vci_fin, cname, None)
        if cls is not None and inspect.isclass(cls):
            provider_cls = cls
            provider_name = cname
            print(f"Found: {cname}")
            break
    if provider_cls is None:
        # List all classes
        print("No standard name found. All classes:")
        for name in dir(vci_fin):
            if name.startswith("_"):
                continue
            obj = getattr(vci_fin, name)
            if inspect.isclass(obj):
                print(f"  class {name}: {obj}")
        return

    # === 4. Signature + methods ===
    section(f"4. {provider_name} signature & methods")
    try:
        print(f"__init__: {inspect.signature(provider_cls.__init__)}")
    except Exception as e:
        print(f"  __init__: <{e}>")

    print(f"\nMethods:")
    for name in sorted(dir(provider_cls)):
        if name.startswith("_"):
            continue
        attr = getattr(provider_cls, name)
        if not callable(attr):
            continue
        try:
            sig = inspect.signature(attr)
            print(f"  {name}{sig}")
        except (ValueError, TypeError) as e:
            print(f"  {name}: <{e}>")

    # === 5. Source code: balance_sheet (full) ===
    section(f"5. {provider_name}.balance_sheet source (full)")
    try:
        bs_method = getattr(provider_cls, "balance_sheet", None)
        if bs_method:
            src = inspect.getsource(bs_method)
            print(src)
        else:
            print("  No balance_sheet method!")
    except Exception as e:
        print(f"  FAILED: {e}")

    # === 6. Source code of any "fetch" / "_get" helper methods ===
    section(f"6. Helper methods (may reveal API endpoint)")
    helper_keywords = ["fetch", "_get", "_request", "_call", "_api", "_url", "endpoint"]
    found_helpers = []
    for name in dir(provider_cls):
        if any(kw in name.lower() for kw in helper_keywords):
            found_helpers.append(name)
    print(f"  Candidate helpers: {found_helpers}")
    for name in found_helpers[:5]:
        attr = getattr(provider_cls, name)
        if callable(attr):
            try:
                src = inspect.getsource(attr)
                print(f"\n--- {name} ---")
                print(src[:1500])
            except Exception as e:
                print(f"\n--- {name}: cannot get source: {e}")

    # === 7. Try direct call ===
    section(f"7. Direct call to {provider_name}.balance_sheet")
    try:
        provider = provider_cls(symbol="TCB", period="quarter")
        df = provider.balance_sheet()
        import re
        qc = [c for c in df.columns
              if re.match(r"^\d{4}[-_]?Q\d$", str(c).strip())]
        print(f"  shape={df.shape}, quarters={len(qc)}")
        print(f"  quarter cols: {qc}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

    # === 8. Inspect module-level constants (API URLs etc) ===
    section("8. Module constants (potential API URLs)")
    for name in dir(vci_fin):
        if name.startswith("_"):
            continue
        obj = getattr(vci_fin, name)
        if isinstance(obj, (str, dict, list)) and name.isupper():
            print(f"  {name} = {repr(obj)[:200]}")

    section("DONE")


if __name__ == "__main__":
    main()