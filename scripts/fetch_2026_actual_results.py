"""
fetch_2026_actual_results.py

Builds the 2026 Kerala Assembly actual results dataset under data_2026_actual/.

Primary source: ECI constituency-wise result pages:
  https://results.eci.gov.in/ResultAcGenMay2026/ConstituencywiseS11{ac}.htm

Stores the top 3 candidates by total votes for each of the 140 constituencies.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = BASE_DIR / "data_2026_actual"
PROCESSED_ROOT = OUTPUT_ROOT / "processed"
DETAIL_ROOT = PROCESSED_ROOT / "constituency_results" / "2026"
RAW_ROOT = OUTPUT_ROOT / "raw" / "eci_constituencywise"

ECI_BASE = "https://results.eci.gov.in/ResultAcGenMay2026"
ECI_CONSTITUENCY_URL = ECI_BASE + "/ConstituencywiseS11{ac}.htm"
TOP_N = 3
REQUEST_DELAY_SEC = 6.0
MAX_RETRIES = 2
RETRY_BACKOFF_SEC = 25.0

PARTY_NORMALIZATION = {
    "CPIM": "CPI(M)",
    "CPI(M)": "CPI(M)",
    "CPI[M]": "CPI(M)",
    "KEC(J)": "KEC(J)",
    "KC(J)": "KEC(J)",
    "CMPKSC": "CMPKSC",
    "RMPOI": "RMPOI",
    "RJD(K)": "RJD",
    "RJD": "RJD",
}

ECI_PARTY_TO_CODE = {
    "INDIAN NATIONAL CONGRESS": "INC",
    "COMMUNIST PARTY OF INDIA (MARXIST)": "CPI(M)",
    "COMMUNIST PARTY OF INDIA": "CPI",
    "INDIAN UNION MUSLIM LEAGUE": "IUML",
    "BHARATIYA JANATA PARTY": "BJP",
    "KERALA CONGRESS": "KEC",
    "KERALA CONGRESS (JACOB)": "KEC(J)",
    "KERALA CONGRESS(JACOB)": "KEC(J)",
    "REVOLUTIONARY SOCIALIST PARTY": "RSP",
    "REVOLUTIONARY MARXIST PARTY OF INDIA": "RMPOI",
    "RASHTRIYA JANATA DAL": "RJD",
    "COMMUNIST MARXIST PARTY KERALA STATE COMMITTEE": "CMPKSC",
    "INDEPENDENT": "IND",
    "NONE OF THE ABOVE": "NOTA",
}


def normalize_party(party: str) -> str:
    p = re.sub(r"\s+", " ", (party or "").strip().upper())
    if p in PARTY_NORMALIZATION:
        return PARTY_NORMALIZATION[p]
    if p in ECI_PARTY_TO_CODE:
        return ECI_PARTY_TO_CODE[p]
    mapped = ECI_PARTY_TO_CODE.get(re.sub(r"[^A-Z0-9() ]+", " ", p))
    if mapped:
        return mapped
    return p


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "unknown"


def alliance_for_party(party: str) -> str:
    p = normalize_party(party)
    if p in {"INC", "IUML", "KEC", "KEC(J)", "RSP", "CMPKSC", "RMPOI"}:
        return "UDF"
    if p in {"CPI(M)", "CPI", "RJD"}:
        return "LDF"
    if p == "BJP":
        return "NDA"
    if p == "IND":
        return "OTHER"
    return "OTHER"


def parse_int(value: str) -> int:
    return int(str(value).replace(",", "").strip())


def parse_float(value: str) -> float:
    return float(str(value).replace(",", "").strip())


def load_constituency_metadata(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "data_2021" / "processed" / "constituencies.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing constituency metadata: {path}")
    meta = pd.read_csv(path)
    meta = meta[meta["year"] == 2021].copy()
    return meta.drop(columns=["year"])


class ECI2026Fetcher:
    def __init__(self, base_dir: Path, cache_html: bool = True) -> None:
        self.base_dir = base_dir
        self.cache_html = cache_html

    def constituency_url(self, ac: int) -> str:
        return ECI_CONSTITUENCY_URL.format(ac=ac)

    def _request_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"{ECI_BASE}/",
        }

    def fetch_constituency_html(self, ac: int) -> str:
        cache_path = RAW_ROOT / f"{ac:03d}.html"
        if self.cache_html and cache_path.exists():
            cached = cache_path.read_text(encoding="utf-8")
            if "Access Denied" not in cached and "Assembly Constituency" in cached:
                return cached

        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            with requests.Session() as session:
                session.headers.update(self._request_headers())
                response = session.get(self.constituency_url(ac), timeout=60)
            if response.status_code == 403 or "Access Denied" in response.text:
                last_error = f"ECI blocked request for AC {ac} (attempt {attempt})"
                time.sleep(RETRY_BACKOFF_SEC)
                continue
            response.raise_for_status()
            if "Assembly Constituency" not in response.text:
                last_error = f"Unexpected ECI response for AC {ac} (attempt {attempt})"
                time.sleep(RETRY_BACKOFF_SEC)
                continue

            if self.cache_html:
                RAW_ROOT.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(response.text, encoding="utf-8")
            return response.text

        raise RuntimeError(last_error or f"Failed to fetch AC {ac}")

    def parse_constituency_html(self, ac: int, html: str) -> tuple[str, list[dict[str, object]]]:
        soup = BeautifulSoup(html, "html.parser")
        heading = soup.find(["h1", "h2", "h3"])
        heading_text = heading.get_text(" ", strip=True) if heading else ""
        match = re.search(r"Assembly Constituency\s+\d+\s*-\s*(.+?)\s*\(", heading_text, re.I)
        constituency_name = match.group(1).strip() if match else ""

        records: list[dict[str, object]] = []
        for tr in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 7:
                continue
            if cells[0] in {"S.N.", "Serial No.", "1"} and cells[1] in {"Candidate", "CANDIDATE"}:
                continue
            if cells[1].upper() in {"CANDIDATE", "TOTAL"}:
                continue
            if not cells[0].isdigit():
                continue
            candidate = cells[1].strip()
            party = normalize_party(cells[2])
            if party == "NOTA":
                continue
            records.append(
                {
                    "serial_number": int(cells[0]),
                    "candidate": candidate,
                    "party": party,
                    "evm_votes": parse_int(cells[3]),
                    "postal_votes": parse_int(cells[4]),
                    "votes": parse_int(cells[5]),
                    "vote_share": parse_float(cells[6]),
                }
            )

        if not records:
            raise RuntimeError(f"No candidate rows parsed for AC {ac}")
        records.sort(key=lambda row: row["votes"], reverse=True)
        return constituency_name, records

    def fetch_top3_for_constituency(self, ac: int) -> list[dict[str, object]]:
        html = self.fetch_constituency_html(ac)
        constituency_name, records = self.parse_constituency_html(ac, html)
        top = records[:TOP_N]
        source_url = self.constituency_url(ac)
        rows: list[dict[str, object]] = []
        for rank, row in enumerate(top, start=1):
            rows.append(
                {
                    "constituency_number": ac,
                    "constituency": constituency_name,
                    "rank": rank,
                    "candidate": row["candidate"],
                    "party": row["party"],
                    "evm_votes": row["evm_votes"],
                    "postal_votes": row["postal_votes"],
                    "votes": row["votes"],
                    "vote_share": row["vote_share"],
                    "is_winner": rank == 1,
                    "source_url": source_url,
                }
            )
        return rows


def build_results_df(base_dir: Path, fetcher: ECI2026Fetcher) -> pd.DataFrame:
    meta = load_constituency_metadata(base_dir)
    all_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    pending = list(range(1, 141))
    for pass_no in range(1, 3):
        if pass_no > 1:
            pending = sorted({item["constituency_number"] for item in failures})
            failures = []
            if not pending:
                break
            time.sleep(30)

        for ac in pending:
            try:
                rows = fetcher.fetch_top3_for_constituency(ac)
                all_rows = [row for row in all_rows if row["constituency_number"] != ac]
                all_rows.extend(rows)
            except Exception as exc:
                failures.append({"constituency_number": ac, "error": str(exc)})
            time.sleep(REQUEST_DELAY_SEC)

    if failures:
        fail_path = OUTPUT_ROOT / "fetch_failures.json"
        fail_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    fetched = {row["constituency_number"] for row in all_rows}
    if len(fetched) != 140:
        missing = sorted(set(range(1, 141)) - fetched)
        raise RuntimeError(
            f"Expected 140 constituencies, got {len(fetched)}. Missing AC: {missing}. "
            f"See {OUTPUT_ROOT / 'fetch_failures.json'}"
        )

    df = pd.DataFrame(all_rows)
    df = df.merge(meta, on="constituency_number", how="left", validate="many_to_one")
    if df["district"].isna().any():
        missing = sorted(df.loc[df["district"].isna(), "constituency_number"].unique().tolist())
        raise RuntimeError(f"Missing constituency metadata for AC numbers: {missing}")

    # Prefer canonical constituency names from metadata.
    df["constituency"] = df["constituency_y"].fillna(df["constituency_x"])
    df = df.drop(columns=["constituency_x", "constituency_y"])
    df["year"] = 2026
    df["alliance"] = df["party"].map(alliance_for_party)
    df = df[
        [
            "year",
            "district",
            "district_id",
            "constituency_id",
            "constituency_number",
            "constituency",
            "rank",
            "candidate",
            "party",
            "alliance",
            "evm_votes",
            "postal_votes",
            "votes",
            "vote_share",
            "is_winner",
            "source_url",
        ]
    ].sort_values(["constituency_number", "rank"])
    return df


def write_outputs(df: pd.DataFrame) -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)

    top3_path = PROCESSED_ROOT / "kerala_assembly_top3_results_2026.csv"
    winners_path = PROCESSED_ROOT / "kerala_assembly_winners_2026.csv"
    df.to_csv(top3_path, index=False)

    winners = df[df["is_winner"]].copy()
    winners.to_csv(winners_path, index=False)

    for (constituency_number, constituency), frame in df.groupby(
        ["constituency_number", "constituency"], sort=True
    ):
        file_path = DETAIL_ROOT / f"{int(constituency_number):03d}_{slugify(constituency)}.csv"
        frame.sort_values("rank").to_csv(file_path, index=False)

    summary = (
        winners.groupby(["party", "alliance"], as_index=False)
        .agg(seats_won=("constituency_number", "count"))
        .sort_values(["seats_won", "party"], ascending=[False, True])
    )
    summary_path = PROCESSED_ROOT / "state_summary.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "year": 2026,
        "source": ECI_BASE,
        "constituencies": 140,
        "top_n": TOP_N,
        "outputs": {
            "top3_results": str(top3_path.relative_to(BASE_DIR)),
            "winners": str(winners_path.relative_to(BASE_DIR)),
            "state_summary": str(summary_path.relative_to(BASE_DIR)),
            "detail_dir": str(DETAIL_ROOT.relative_to(BASE_DIR)),
            "raw_html_dir": str(RAW_ROOT.relative_to(BASE_DIR)),
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run(base_dir: Path, cache_html: bool = True) -> pd.DataFrame:
    fetcher = ECI2026Fetcher(base_dir=base_dir, cache_html=cache_html)
    df = build_results_df(base_dir, fetcher)
    write_outputs(df)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch 2026 Kerala Assembly top-3 actual results by constituency from ECI."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Repository root containing data_2021 metadata.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not cache downloaded ECI HTML pages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run(args.base_dir, cache_html=not args.no_cache)
    constituencies = df["constituency_number"].nunique()
    print(f"Wrote top {TOP_N} results for {constituencies} constituencies to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
