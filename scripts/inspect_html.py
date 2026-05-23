"""Helper: inspect dumped HTML files để extract context xung quanh known values.

Usage:
    python scripts/inspect_html.py data/raw/_debug/cpi_raw.html 3.61
    python scripts/inspect_html.py data/raw/_debug/gdp_raw.html "Nominal Gross"
    python scripts/inspect_html.py data/raw/_debug/cpi_raw.html --stats
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path


def extract_contexts(html: str, search: str, context: int = 400, max_matches: int = 5) -> list:
    out = []
    for m in re.finditer(re.escape(search), html):
        start = max(0, m.start() - context)
        end = min(len(html), m.end() + context)
        out.append((m.start(), html[start:end]))
        if len(out) >= max_matches:
            break
    return out


def print_stats(html: str) -> None:
    print(f"HTML length: {len(html):,} chars")
    print(f"<script> tags: {html.count('<script')}")
    print(f"<table> tags: {html.count('<table')}")
    print(f"<tbody> tags: {html.count('<tbody')}")
    print(f"<tr> tags: {html.count('<tr')}")
    print(f"'Date.UTC(' occurrences: {html.count('Date.UTC(')}")
    print(f"'Highcharts' occurrences: {html.count('Highcharts')}")
    print(f"'chart.data' occurrences: {html.count('chart.data')}")
    print(f"'series' occurrences: {html.count('series')}")
    print(f"'amCharts' occurrences: {html.count('amCharts')}")
    print(f"'Plotly' occurrences: {html.count('Plotly')}")
    print(f"'Chart.js' occurrences: {html.count('Chart.js')}")
    print(f"'echarts' occurrences: {html.count('echarts')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("search", nargs="?", default=None,
                    help="String to search for (e.g. '3.61')")
    ap.add_argument("--context", type=int, default=400,
                    help="Chars before/after match")
    ap.add_argument("--max", type=int, default=5,
                    help="Max matches to show")
    ap.add_argument("--stats", action="store_true",
                    help="Print structural stats")
    args = ap.parse_args()

    if not args.file.exists():
        raise SystemExit(f"File not found: {args.file}")

    html = args.file.read_text(encoding="utf-8")
    print(f"File: {args.file}")
    print_stats(html)

    if args.stats and args.search is None:
        return

    if args.search is None:
        print("\nTip: pass a string to search, e.g.:")
        print(f"  python scripts/inspect_html.py {args.file} 3.61")
        return

    matches = extract_contexts(html, args.search, args.context, args.max)
    print(f"\n=== Found {len(matches)} occurrences of '{args.search}' (showing max {args.max}) ===")
    for pos, ctx in matches:
        print(f"\n--- At position {pos} ---")
        # Compact whitespace for readability
        compact = re.sub(r"\s+", " ", ctx)
        print(compact[:1000])
        print("---")


if __name__ == "__main__":
    main()