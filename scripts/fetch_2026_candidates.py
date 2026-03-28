"""
fetch_2026_candidates.py  –  v3
--------------------------------
Scrapes 2026 Kerala Assembly candidate data from the ECI affidavit portal.

Discovered endpoint (GET, paginated):
  https://affidavit.eci.gov.in/CandidateCustomFilter
  ?electionType=32-AC-GENERAL-3-60&election=32-AC-GENERAL-3-60+
  &states=S11&phase=&constId=&submitName=100&search=&page={N}

9 pages × ~100 candidates = 866 total contesting candidates.

Usage:
  python scripts/fetch_2026_candidates.py
  python scripts/fetch_2026_candidates.py --pages 9 --save-html
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[1]
CONSTITUENCIES_CSV = BASE_DIR / "data_2021" / "processed" / "constituencies.csv"

ECI_BASE = "https://affidavit.eci.gov.in"
ECI_FILTER_URL = f"{ECI_BASE}/CandidateCustomFilter"

# Discovered parameters
ELECTION_TYPE = "32-AC-GENERAL-3-60"
STATE_CODE = "S11"
STATUS_CONTESTING = "100"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": ECI_BASE + "/",
}


# ---------------------------------------------------------------------------
# Party / Alliance normalisation
# ---------------------------------------------------------------------------

def normalize_party(raw: str) -> str:
    p = raw.strip()
    pu = " ".join(p.upper().split())
    if "COMMUNIST PARTY OF INDIA" in pu and "MARXIST" in pu:
        return "CPI(M)"
    if pu in {"COMMUNIST PARTY OF INDIA", "CPI"}:
        return "CPI"
    if "BHARATIYA JANATA" in pu or pu == "BJP":
        return "BJP"
    if "INDIAN NATIONAL CONGRESS" in pu:
        return "INC"
    if "INDIAN UNION MUSLIM LEAGUE" in pu:
        return "IUML"
    if "KERALA CONGRESS" in pu and any(x in pu for x in ["(M)", " M)"]):
        return "KEC(M)"
    if "KERALA CONGRESS" in pu and any(x in pu for x in ["(J)", " J)"]):
        return "KEC(J)"
    if "KERALA CONGRESS" in pu:
        return "KEC"
    if "JANATA DAL" in pu and "SECULAR" in pu:
        return "JD(S)"
    if "NATIONALIST CONGRESS" in pu:
        return "NCP"
    if "REVOLUTIONARY SOCIALIST" in pu:
        return "RSP"
    if "SOCIAL DEMOCRATIC PARTY" in pu and "INDIA" in pu:
        return "SDPI"
    if "BAHUJAN SAMAJ" in pu:
        return "BSP"
    if "AAM AADMI" in pu:
        return "AAP"
    if "BDJS" in pu or "BHARATH DHARMA JANA SENA" in pu:
        return "BDJS"
    if "WELFARE PARTY" in pu:
        return "WPI"
    if re.search(r"\BINDEPENDENT\b", pu):
        return "IND"
    return p


def get_alliance(party_raw: str) -> str:
    pu = " ".join(party_raw.strip().upper().split())
    if any(x in pu for x in ["COMMUNIST PARTY OF INDIA", "CPI"]):
        return "LDF"
    if "JANATA DAL" in pu and "SECULAR" in pu:
        return "LDF"
    if "KERALA CONGRESS" in pu and any(x in pu for x in ["(M)", " M)"]):
        return "LDF"
    if "NATIONALIST CONGRESS" in pu:
        return "LDF"
    if "REVOLUTIONARY SOCIALIST" in pu:
        return "LDF"
    if "CONGRESS (SECULAR)" in pu or "CONGRESS(SECULAR)" in pu:
        return "LDF"
    if "INDIAN NATIONAL CONGRESS" in pu:
        return "UDF"
    if "INDIAN UNION MUSLIM LEAGUE" in pu:
        return "UDF"
    if "KERALA CONGRESS" in pu and any(x in pu for x in ["(J)", " J)"]):
        return "UDF"
    if "KERALA CONGRESS" in pu:
        return "UDF"
    if "SOCIAL DEMOCRATIC PARTY" in pu and "INDIA" in pu:
        return "UDF"
    if "BHARATIYA JANATA" in pu or "BDJS" in pu or "BHARATH DHARMA JANA SENA" in pu:
        return "NDA"
    return "OTHER"


# ---------------------------------------------------------------------------
# Constituency lookup
# ---------------------------------------------------------------------------

def load_constituency_lookup() -> tuple[dict[int, dict], dict[str, int]]:
    num_lookup: dict[int, dict] = {}
    name_lookup: dict[str, int] = {}
    if not CONSTITUENCIES_CSV.exists():
        return num_lookup, name_lookup
    import pandas as pd
    df = pd.read_csv(CONSTITUENCIES_CSV)
    for _, row in df.iterrows():
        num = int(row["constituency_number"])
        name = str(row["constituency"])
        district = str(row.get("district", ""))
        num_lookup[num] = {"constituency": name, "district": district}
        # Add multiple name variants to lookup
        for variant in [name.upper(), name.upper().replace(" ", ""), name.upper().replace("-", " ")]:
            name_lookup[variant] = num
    return num_lookup, name_lookup


def match_constituency(raw_name: str, name_lookup: dict[str, int]) -> int:
    cleaned = " ".join(raw_name.strip().upper().split())
    if cleaned in name_lookup:
        return name_lookup[cleaned]
    # Try without spaces
    no_space = cleaned.replace(" ", "")
    if no_space in name_lookup:
        return name_lookup[no_space]
    # Substring match
    for known, num in name_lookup.items():
        if len(cleaned) >= 4 and (cleaned in known or known in cleaned):
            return num
    return 0


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def parse_page_html(html: str, num_lookup: dict[int, dict], name_lookup: dict[str, int]) -> list[dict]:
    """Parse a single page of ECI candidate cards."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []

    # The ECI portal uses candidate cards. Try multiple CSS selectors:
    # Pattern 1: Bootstrap col cards (common ECI layout)
    cards = (
        soup.select(".col-md-3.candidate-card")
        or soup.select(".candidate-card")
        or soup.select(".col-md-3 .card")
        or soup.select(".col-lg-3 .card")
        or soup.select(".col-md-4 .card")
        or soup.select(".card-candidate")
        or soup.select(".card")
    )

    for card in cards:
        # Extract all text elements
        h_tags = card.find_all(["h5", "h4", "h3", "strong", "b"])
        p_tags = card.find_all(["p", "span", "small"])

        candidate = ""
        party_raw = ""
        const_raw = ""
        status = ""

        # Candidate name is usually the first heading
        if h_tags:
            candidate = h_tags[0].get_text(" ", strip=True)

        # Rest of content
        all_texts = [t.get_text(" ", strip=True) for t in p_tags]
        for txt in all_texts:
            txt_up = txt.upper()
            if any(kw in txt_up for kw in ["CONTESTING", "ACCEPTED", "REJECTED", "WITHDRAWN"]) and not status:
                status = txt
            elif any(kw in txt_up for kw in [
                "COMMUNIST", "CONGRESS", "JANATA", "BHARATIYA", "MUSLIM LEAGUE",
                "KERALA CONGRESS", "INDEPENDENT", "REVOLUTIONARY", "SOCIALIST",
                "BAHUJAN", "AAM AADMI", "WELFARE PARTY", "DEMOCRATIC",
            ]) and not party_raw:
                party_raw = txt
            elif not const_raw and any(kw in txt_up for kw in ["AC -", "CONSTITUENCY", "LAC"]):
                const_raw = re.sub(r"(?i)constituency\s*[:–\-]*\s*|AC\s*-\s*\d+\s*", "", txt).strip()

        # Alternative: look for data-* attributes
        if not party_raw:
            party_raw = card.get("data-party", "") or card.find(attrs={"data-party": True}) and card.find(attrs={"data-party": True}).get("data-party", "")
        if not const_raw:
            const_raw = card.get("data-constituency", "") or ""

        if not candidate:
            # Fallback: just get all text
            all_text = list(card.stripped_strings)
            if all_text:
                candidate = all_text[0]
            if len(all_text) > 1 and not party_raw:
                party_raw = all_text[1]
            if len(all_text) > 2 and not const_raw:
                const_raw = all_text[2]

        if not candidate:
            continue

        const_num = match_constituency(const_raw, name_lookup)
        const_info = num_lookup.get(const_num, {})

        rows.append({
            "constituency_number": const_num,
            "constituency": const_info.get("constituency", const_raw),
            "district": const_info.get("district", ""),
            "candidate": candidate,
            "party": normalize_party(party_raw) if party_raw else "UNKNOWN",
            "party_full": party_raw,
            "alliance": get_alliance(party_raw) if party_raw else "OTHER",
            "status": status or "CONTESTING",
        })

    # Fallback: table rows
    if not rows:
        for tr in soup.select("table tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            sl = cells[0] if cells[0].isdigit() else ""
            offset = 1 if sl else 0
            candidate = cells[offset] if len(cells) > offset else ""
            party_raw = cells[offset + 1] if len(cells) > offset + 1 else ""
            const_raw = cells[offset + 2] if len(cells) > offset + 2 else ""
            status = cells[offset + 3] if len(cells) > offset + 3 else ""
            if not candidate:
                continue
            const_num = match_constituency(const_raw, name_lookup)
            const_info = num_lookup.get(const_num, {})
            rows.append({
                "constituency_number": const_num,
                "constituency": const_info.get("constituency", const_raw),
                "district": const_info.get("district", ""),
                "candidate": candidate,
                "party": normalize_party(party_raw),
                "party_full": party_raw,
                "alliance": get_alliance(party_raw),
                "status": status,
            })
    return rows


def detect_total_pages(html: str) -> int:
    """Try to detect total page count from pagination HTML."""
    soup = BeautifulSoup(html, "html.parser")
    # Look for pagination links
    pagination = soup.select(".pagination li a, .page-link, nav[aria-label='pagination'] a")
    page_nums = []
    for a in pagination:
        txt = a.get_text(strip=True)
        if txt.isdigit():
            page_nums.append(int(txt))
    if page_nums:
        return max(page_nums)

    # Look for "Showing X to Y of Z" text
    text = soup.get_text(" ")
    m = re.search(r"of\s+([\d,]+)\s+(?:results|entries|candidates)", text, re.IGNORECASE)
    if m:
        total = int(m.group(1).replace(",", ""))
        # Guess page size from visible candidates
        cards = soup.select(".card") or soup.select("table tbody tr")
        per_page = max(len(cards), 100)
        return max(1, -(-total // per_page))  # ceil division

    return 9  # default from browser observation


def fetch_page(session: requests.Session, page_num: int) -> str:
    """Fetch a single page of Kerala 2026 candidates."""
    params = {
        "electionType": ELECTION_TYPE,
        "election": ELECTION_TYPE + "+",
        "states": STATE_CODE,
        "phase": "",
        "constId": "",
        "submitName": STATUS_CONTESTING,
        "search": "",
        "page": str(page_num),
    }
    resp = session.get(ECI_FILTER_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def init_session() -> tuple[requests.Session, str]:
    """Initialize session with cookies from the portal homepage."""
    session = requests.Session()
    try:
        resp = session.get(ECI_BASE, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")
        meta = soup.find("meta", {"name": "csrf-token"})
        csrf = meta.get("content", "") if meta else ""
        inp = soup.find("input", {"name": "_token"})
        if not csrf and inp:
            csrf = inp.get("value", "")
        if not csrf:
            csrf = requests.utils.unquote(session.cookies.get("XSRF-TOKEN", ""))
        return session, csrf
    except Exception:
        return session, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BASE_DIR / "data_2026" / "processed" / "kerala_2026_candidates.csv"))
    parser.add_argument("--pages", type=int, default=0, help="Number of pages to fetch (0=auto-detect)")
    parser.add_argument("--save-html", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    num_lookup, name_lookup = load_constituency_lookup()
    print(f"Loaded {len(num_lookup)} constituency definitions.")

    session, csrf = init_session()
    print(f"Session initialized | CSRF={'found' if csrf else 'not found'}")

    # Fetch page 1 to detect total pages
    print("\nFetching page 1...")
    html1 = fetch_page(session, 1)

    if args.save_html:
        html_dir = out_path.parent
        (html_dir / "raw_page_1.html").write_text(html1, encoding="utf-8")
        print(f"  Saved raw HTML to {html_dir / 'raw_page_1.html'}")

    total_pages = args.pages or detect_total_pages(html1)
    print(f"  Detected total pages: {total_pages}")

    rows = parse_page_html(html1, num_lookup, name_lookup)
    print(f"  Page 1: parsed {len(rows)} candidates")

    # Fetch remaining pages
    for page in range(2, total_pages + 1):
        print(f"Fetching page {page}/{total_pages}...")
        try:
            html = fetch_page(session, page)
            if args.save_html:
                (out_path.parent / f"raw_page_{page}.html").write_text(html, encoding="utf-8")
            page_rows = parse_page_html(html, num_lookup, name_lookup)
            print(f"  Page {page}: parsed {len(page_rows)} candidates")
            rows.extend(page_rows)
        except Exception as exc:
            print(f"  [warn] Page {page} failed: {exc}", file=sys.stderr)
        time.sleep(0.4)

    if not rows:
        print("\n❌ No candidates parsed. Saving HTML for debugging...")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        (out_path.parent / "raw_page_1.html").write_text(html1, encoding="utf-8")
        print(f"  Saved: {out_path.parent / 'raw_page_1.html'}")
        print("  Inspect this file to understand the HTML structure, then fix parse_page_html()")
        return 1

    df = pd.DataFrame(rows).drop_duplicates(subset=["constituency_number", "candidate"])
    df = df.sort_values(["constituency_number", "candidate"])
    df.to_csv(out_path, index=False)

    print(f"\n✅ Saved {len(df)} unique candidates to {out_path}")
    print(f"   Constituencies: {df['constituency_number'].nunique()}/140")
    print(f"   Alliance counts:\n{df['alliance'].value_counts().to_string()}")
    print(f"\nSample:\n{df.head(10).to_string(index=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
