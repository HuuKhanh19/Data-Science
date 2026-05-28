"""Channel B - Macro scrapers (v5).

Thay đổi từ v4:
- CPI: lưu RAW TEXT thay vì parse YoY ở Phase 1.
  + Phase 1 chỉ extract metadata structured (release_date, reference_period, title,
    source_url) + raw body text của press release.
  + Phase 2 preprocessing sẽ extract MoM từ raw_text, rồi compute YoY từ chuỗi MoM.
  + Lý do: separation of concerns — raw layer chỉ collect, không transform.
  + Pagination: dùng `?paged=N` (verified từ HTML inspect: id='loopage_paged',
    class='page-numbers'), KHÔNG dùng `/page/N/` (sai trong v4).
- GDP: VBMA endpoint trả TSV (tab-separated), không phải CSV. Fix sep='\t'.
  Layout B (cols = quarter labels, row đầu = "Nominal Gross Domestic Product"),
  values quoted với thousand-separator dạng `"809,613"`.

Anti-leakage: dùng `Date of issue` (DD/MM/YYYY) làm release_date thực.
"""
from __future__ import annotations
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

from .schema import CPI_SCHEMA, GDP_SCHEMA

TZ_VN = ZoneInfo("Asia/Ho_Chi_Minh")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 30


def _now_vn() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(TZ_VN))


def _http_get(url: str, retries: int = 3, timeout: int = DEFAULT_TIMEOUT,
              verify_ssl: bool = True) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9,vi;q=0.8"}
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
            if r.status_code == 404:
                raise requests.HTTPError(f"404 Not Found: {url}")
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            last_err = e
            print(f"  [attempt {attempt+1}/{retries}] GET {url} failed: {e}")
    raise RuntimeError(f"All {retries} attempts failed for {url}: {last_err}")


def _http_get_bytes(url: str, retries: int = 3, timeout: int = DEFAULT_TIMEOUT,
                    verify_ssl: bool = True) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, verify=verify_ssl)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            last_err = e
            print(f"  [attempt {attempt+1}/{retries}] GET {url} failed: {e}")
    raise RuntimeError(f"All {retries} attempts failed for {url}: {last_err}")


# ============================================================
# CPI - NSO scrape, raw text approach
# ============================================================

NSO_CPI_BASE = "https://www.nso.gov.vn/en/cpi/"
NSO_PAGE_LIMIT = 55  # HTML inspect cho thấy có 51 pages → padding nhẹ

MONTH_NAMES_FULL = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Date of issue: DD/MM/YYYY (Vietnamese format)
DATE_OF_ISSUE_RE = re.compile(r"Date of issue:\s*(\d{1,2})/(\d{1,2})/(\d{4})")
# Reference period: Month YYYY (newer format, page 1-4) — e.g. "April 2026"
REF_PERIOD_TEXT_RE = re.compile(r"Reference period:\s*([A-Za-z]+)\s+(\d{4})")
# Reference period: M/YYYY (older format, page 5+) — e.g. "8/2024"
REF_PERIOD_NUM_RE = re.compile(r"Reference period:\s*(\d{1,2})\s*/\s*(\d{4})")


def _parse_reference_period(text: str) -> tuple[int, int] | None:
    """Try text format first, then numeric M/YYYY. Returns (month_int, year_int) or None.

    NSO English archive uses 2 formats:
    - Newer posts (~Oct 2024 onwards): "Reference period: April 2026"
    - Older posts (~Aug 2024 backwards): "Reference period: 8/2024"
    """
    m = REF_PERIOD_TEXT_RE.search(text)
    if m:
        month_name = m.group(1).lower()
        month_num = MONTH_NAMES_FULL.get(month_name)
        if month_num is not None:
            return month_num, int(m.group(2))
    m = REF_PERIOD_NUM_RE.search(text)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return month, int(m.group(2))
    return None


def _parse_nso_page(html: str) -> List[Dict[str, Any]]:
    """Extract press releases as raw records from one NSO page.

    Returns list of dict với keys: reference_period, release_date, title, raw_text,
    source_url. NO YoY/MoM parsing — Phase 2 sẽ làm.
    """
    soup = BeautifulSoup(html, "lxml")
    sections = soup.find_all("section", class_="item")

    releases: List[Dict[str, Any]] = []
    skipped_no_meta = 0
    skipped_bad_format = 0

    for sec in sections:
        # Title from <h3>
        h3 = sec.find("h3")
        title = h3.get_text(strip=True) if h3 else None

        # All <p> tags: first one(s) = body, last one = metadata span container
        ps = sec.find_all("p")
        if not ps:
            continue

        # Metadata spans (look in any <p> inside section)
        issue_date_span = sec.find("span", class_="archive-issue-date")
        ref_period_span = sec.find("span", class_="archive-reference-period")
        if issue_date_span is None or ref_period_span is None:
            skipped_no_meta += 1
            continue

        date_m = DATE_OF_ISSUE_RE.search(issue_date_span.get_text(strip=True))
        ref_parsed = _parse_reference_period(ref_period_span.get_text(strip=True))
        if not date_m or ref_parsed is None:
            skipped_bad_format += 1
            continue

        try:
            day = int(date_m.group(1))
            month = int(date_m.group(2))
            year = int(date_m.group(3))
            release_date = pd.Timestamp(year=year, month=month, day=day)
        except (ValueError, OverflowError):
            skipped_bad_format += 1
            continue

        month_num, ref_year = ref_parsed
        ref_period_end = (
            pd.Timestamp(year=ref_year, month=month_num, day=1) + pd.offsets.MonthEnd(0)
        )

        # Body text: lấy các <p> KHÔNG chứa span.archive-* (loại metadata block)
        body_paragraphs = []
        for p in ps:
            if p.find("span", class_="archive-issue-date"):
                continue
            txt = p.get_text(separator=" ", strip=True)
            if txt:
                body_paragraphs.append(txt)
        raw_text = "\n".join(body_paragraphs).strip()

        if not raw_text:
            # Fallback: nếu structure khác, dùng cả section text trừ metadata
            full_txt = sec.get_text(separator=" ", strip=True)
            for span in (issue_date_span, ref_period_span):
                full_txt = full_txt.replace(span.get_text(strip=True), "")
            next_span = sec.find("span", class_="archive-next-release")
            if next_span:
                full_txt = full_txt.replace(next_span.get_text(strip=True), "")
            raw_text = re.sub(r"\s+", " ", full_txt).strip()

        if not raw_text:
            # Final guard: nếu sau fallback vẫn empty thì skip
            continue

        # Source URL: gần section nhất, tìm <a href> sibling trước đó hoặc parent <a>
        source_url = None
        prev = sec.find_previous("a", href=True)
        if prev and "/data-and-statistics/" in prev.get("href", ""):
            source_url = prev["href"]

        releases.append({
            "reference_period": ref_period_end,
            "release_date": release_date,
            "title": title,
            "raw_text": raw_text,
            "source_url": source_url,
        })

    if skipped_bad_format > 0 or skipped_no_meta > 0:
        print(f"    [parse] {len(releases)} parsed, "
              f"skipped: {skipped_no_meta} no-meta, "
              f"{skipped_bad_format} bad-format")

    return releases


def fetch_cpi(out_path: Path, start_date: str | None = None,
              end_date: str | None = None,
              verify_ssl: bool = False,
              prehistory_months: int = 13) -> Dict[str, Any]:
    """Scrape NSO CPI press releases.

    Pagination: ?paged=N (WordPress query param, verified từ HTML).
    Phase 1 lưu raw text — Phase 2 sẽ extract MoM/YoY.

    Filter logic:
    - Phase 2 cần MoM của 12 tháng trước start_date để compute YoY đầu tiên.
    - prehistory_months=13 (default): lùi 13 tháng = 12 MoM + 1 buffer.
    - Stop crawling early khi reach data cũ hơn cutoff → tiết kiệm HTTP requests.
    - Pass prehistory_months=0 để disable buffer (chỉ lấy [start_date, end_date]).
    - Pass start_date=None để disable filter (lấy hết history).
    """
    print(f"  [cpi] Scraping NSO CPI archive (verify_ssl={verify_ssl})...")

    # Tính cutoff
    cpi_start_cutoff: pd.Timestamp | None = None
    cpi_end_cutoff: pd.Timestamp | None = None
    if start_date is not None:
        cpi_start_cutoff = pd.Timestamp(start_date) - pd.DateOffset(months=prehistory_months)
    if end_date is not None:
        cpi_end_cutoff = pd.Timestamp(end_date)
    if cpi_start_cutoff is not None:
        print(f"  [cpi] Cutoff: reference_period >= {cpi_start_cutoff.date()} "
              f"(= start_date {start_date} - {prehistory_months} months buffer)")

    all_releases: List[Dict[str, Any]] = []
    seen_periods: set = set()
    pages_fetched = 0
    stopped_by_cutoff = False

    for page in range(1, NSO_PAGE_LIMIT + 1):
        url = NSO_CPI_BASE if page == 1 else f"{NSO_CPI_BASE}?paged={page}"
        try:
            html = _http_get(url, verify_ssl=verify_ssl, retries=2)
        except RuntimeError as e:
            print(f"  [cpi] page {page} fetch failed: {e}")
            break

        pages_fetched += 1

        page_releases = _parse_nso_page(html)

        # Edge case A: page có 0 sections (genuinely empty / 404 fallback page)
        if not page_releases:
            print(f"  [cpi] page {page}: 0 sections found - stopping")
            break

        new_releases = [r for r in page_releases
                        if r["reference_period"] not in seen_periods]

        # Edge case B: page có sections nhưng tất cả đã thấy (server fallback
        # returning same content) → stop, dump HTML để inspect
        if not new_releases:
            print(f"  [cpi] page {page}: {len(page_releases)} sections but all duplicates - stopping")
            break

        all_releases.extend(new_releases)
        seen_periods.update(r["reference_period"] for r in new_releases)
        oldest = min(r["reference_period"] for r in new_releases).date()
        newest = max(r["reference_period"] for r in new_releases).date()
        print(f"  [cpi] page {page}: +{len(new_releases)} new "
              f"(range {oldest} -> {newest}); cumulative: {len(all_releases)}")

        # Early stop: nếu page hiện tại đã reach data cũ hơn cutoff thì
        # các page sau toàn data cũ hơn (NSO listing DESC by date) → stop.
        if cpi_start_cutoff is not None:
            oldest_ts = pd.Timestamp(oldest)
            if oldest_ts < cpi_start_cutoff:
                print(f"  [cpi] Page {page} oldest = {oldest} < cutoff "
                      f"{cpi_start_cutoff.date()} → stopping (early)")
                stopped_by_cutoff = True
                break

    print(f"  [cpi] Done. Total: {len(all_releases)} releases across {pages_fetched} pages")

    if not all_releases:
        raise RuntimeError("Khong extract duoc CPI tu NSO (0 releases parsed).")

    df = pd.DataFrame(all_releases).sort_values("reference_period").reset_index(drop=True)
    df = df.drop_duplicates(subset=["reference_period"], keep="last").reset_index(drop=True)

    # Filter theo [cpi_start_cutoff, cpi_end_cutoff] nếu được set
    if cpi_start_cutoff is not None or cpi_end_cutoff is not None:
        n_before = len(df)
        mask = pd.Series([True] * len(df), index=df.index)
        if cpi_start_cutoff is not None:
            mask &= df["reference_period"] >= cpi_start_cutoff
        if cpi_end_cutoff is not None:
            mask &= df["reference_period"] <= cpi_end_cutoff
        df = df[mask].reset_index(drop=True)
        print(f"  [cpi] Filtered: {n_before} → {len(df)} releases "
              f"(kept reference_period in [{cpi_start_cutoff.date() if cpi_start_cutoff else 'open'}, "
              f"{cpi_end_cutoff.date() if cpi_end_cutoff else 'open'}])")

    df["fetched_at"] = _now_vn()

    CPI_SCHEMA.validate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "status": "ok" if len(df) >= 50 else "warning",
        "rows": int(len(df)),
        "date_min": df["reference_period"].min().date().isoformat() if not df.empty else None,
        "date_max": df["reference_period"].max().date().isoformat() if not df.empty else None,
        "pages_fetched": pages_fetched,
        "output": str(out_path),
    }


# ============================================================
# GDP - VBMA TSV endpoint
# ============================================================

VBMA_GDP_CSV_URL = "https://vbma.org.vn/csv/markets/tables/en/gdp_danh_nghia_theo_quy.csv"

QUARTER_LABEL_RE = re.compile(r"Q\s*(\d)\s+(\d{4})", re.IGNORECASE)


def _parse_vbma_quarter_label(label: str) -> pd.Timestamp | None:
    """Parse VBMA quarter label e.g. 'Q1 2015' or 'Q4 2025' -> quarter-end Timestamp."""
    if label is None:
        return None
    label = str(label).strip()
    m = QUARTER_LABEL_RE.match(label)
    if m:
        q, year = int(m.group(1)), int(m.group(2))
        if 1 <= q <= 4:
            return pd.Timestamp(year=year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)
    return None


def _parse_vbma_numeric(val: Any) -> float | None:
    """Parse VBMA value like '"809,613"' -> 809613.0."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().strip('"').strip("'").replace(",", "").replace(" ", "")
    if not s or s.lower() in ("nan", "none", "null", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_gdp(out_path: Path, start_date: str | None = None,
              end_date: str | None = None,
              verify_ssl: bool = False) -> Dict[str, Any]:
    """Fetch GDP từ VBMA TSV endpoint.

    Format thực (verified từ Get-Content):
        \\tQ1 2015\\tQ2 2015\\t...\\tQ1 2026          (header row)
        Nominal Gross Domestic Product\\t"809,613"\\t"970,388"\\t...   (data row 1)
        "Agriculture, Forestry and Fishery "\\t"99,978"\\t...           (data row 2+)

    Layout B: cols = quarter labels, row đầu = total nominal GDP.
    """
    print(f"  [gdp] GET TSV: {VBMA_GDP_CSV_URL}")
    csv_bytes = _http_get_bytes(VBMA_GDP_CSV_URL, verify_ssl=verify_ssl)
    print(f"  [gdp] Received {len(csv_bytes)} bytes")

    # File là TSV (tab-separated). VBMA encode UTF-16 LE với BOM (\xff\xfe),
    # nên utf-16 phải đứng đầu — utf-8 sẽ raise UnicodeDecodeError, nhưng
    # latin-1 sẽ "thành công" decode → produce garbage (\x00 padding).
    bom_hint = ""
    if csv_bytes.startswith(b"\xff\xfe"):
        bom_hint = " [BOM detected: UTF-16 LE]"
    elif csv_bytes.startswith(b"\xfe\xff"):
        bom_hint = " [BOM detected: UTF-16 BE]"
    elif csv_bytes.startswith(b"\xef\xbb\xbf"):
        bom_hint = " [BOM detected: UTF-8]"
    print(f"  [gdp] First 4 bytes: {csv_bytes[:4]!r}{bom_hint}")

    df_raw = None
    last_err = None
    for encoding in ("utf-16", "utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df_raw = pd.read_csv(
                io.BytesIO(csv_bytes),
                sep="\t",
                encoding=encoding,
                dtype=str,
                keep_default_na=False,
                skip_blank_lines=True,
            )
            # Validate: phải có ít nhất 1 column match "Q<n> YYYY".
            # latin-1 luôn decode được nhưng output garbage → loại bỏ.
            if any(QUARTER_LABEL_RE.match(str(c).strip()) for c in df_raw.columns):
                print(f"  [gdp] Parsed TSV with encoding={encoding}, shape={df_raw.shape}")
                print(f"  [gdp] First 5 columns: {list(df_raw.columns)[:5]}")
                break
            else:
                print(f"  [gdp] encoding={encoding}: parsed but no quarter cols (garbled) → next")
                df_raw = None
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last_err = e
            print(f"  [gdp] encoding={encoding}: {type(e).__name__} → next")
            continue

    if df_raw is None or df_raw.empty:
        raise RuntimeError(f"Không parse được TSV từ VBMA. Last error: {last_err}")

    # Identify quarter columns (cols whose name matches "Q<n> YYYY")
    quarter_cols: List[Tuple[str, pd.Timestamp]] = []
    for col in df_raw.columns:
        ts = _parse_vbma_quarter_label(str(col))
        if ts is not None:
            quarter_cols.append((col, ts))

    print(f"  [gdp] Found {len(quarter_cols)} quarter columns "
          f"(range {quarter_cols[0][1].date() if quarter_cols else None} -> "
          f"{quarter_cols[-1][1].date() if quarter_cols else None})")

    if len(quarter_cols) < 5:
        raise RuntimeError(
            f"Không đủ quarter columns (found {len(quarter_cols)}). "
            f"Columns: {list(df_raw.columns)[:20]}"
        )

    # First column = row labels (e.g. "Nominal Gross Domestic Product", "Agriculture...")
    label_col = df_raw.columns[0]

    # Find the "Nominal Gross Domestic Product" row
    target_row = None
    for _, row in df_raw.iterrows():
        row_label = str(row[label_col]).strip().lower().strip('"')
        if "nominal" in row_label and "gross domestic product" in row_label:
            target_row = row
            print(f"  [gdp] Matched label row: '{row[label_col]}'")
            break

    if target_row is None:
        # Fallback: first non-empty data row
        print(f"  [gdp] WARNING: không match 'Nominal Gross Domestic Product'. "
              f"Available labels (first 10): "
              f"{[str(r).strip() for r in df_raw[label_col].head(10).tolist()]}")
        raise RuntimeError("Không tìm thấy row 'Nominal Gross Domestic Product'.")

    rows: List[Tuple[pd.Timestamp, float]] = []
    for col, ts in quarter_cols:
        val = _parse_vbma_numeric(target_row[col])
        if val is not None:
            rows.append((ts, val))

    print(f"  [gdp] Extracted {len(rows)} (quarter, value) pairs")

    if not rows:
        raise RuntimeError("Parsed quarters nhưng tất cả values đều null.")

    df = pd.DataFrame(
        [{"reference_period": k, "nominal_gdp_vnd_bil": v}
         for k, v in sorted(set(rows), key=lambda x: x[0])]
    )
    df = df.drop_duplicates(subset=["reference_period"], keep="last").reset_index(drop=True)

    if start_date:
        df = df[df["reference_period"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["reference_period"] <= pd.Timestamp(end_date)]
    df = df.reset_index(drop=True)

    df["release_date"] = df["reference_period"] + pd.Timedelta(days=30)
    df["fetched_at"] = _now_vn()

    GDP_SCHEMA.validate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return {
        "status": "ok",
        "rows": int(len(df)),
        "date_min": df["reference_period"].min().date().isoformat() if not df.empty else None,
        "date_max": df["reference_period"].max().date().isoformat() if not df.empty else None,
        "output": str(out_path),
    }