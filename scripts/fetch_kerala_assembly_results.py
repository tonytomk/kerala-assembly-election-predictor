from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


DISTRICTS = {
    1: "Kasaragod",
    2: "Kannur",
    3: "Wayanad",
    4: "Kozhikode",
    5: "Malappuram",
    6: "Palakkad",
    7: "Thrissur",
    8: "Ernakulam",
    9: "Idukki",
    10: "Kottayam",
    11: "Alappuzha",
    12: "Pathanamthitta",
    13: "Kollam",
    14: "Thiruvananthapuram",
}

CEO_BASE = "https://www.ceo.kerala.gov.in"
ECI_2021_BASE = "https://results.eci.gov.in/Result2021"
YEARS = (2011, 2016, 2021)
LETTER_ORDER = [chr(code) for code in range(ord("A"), ord("U") + 1)]


@dataclass
class Constituency:
    year: int
    district_id: int
    district: str
    constituency_id: int
    constituency_number: int
    constituency: str
    source_url: str


def normalize_name(value: str) -> str:
    value = re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()
    value = re.sub(r"^(ADV|DR|PROF|PROFESSOR|SMT|SRI|SHRI|KU|MRS|MS|MR)\s+", "", value)
    return re.sub(r"\s+", " ", value)


def token_sort_key(value: str) -> str:
    return " ".join(sorted(normalize_name(value).split()))


def normalize_party_name(party: str | None) -> str:
    p = (party or "").strip().upper()
    if p == "MLKSC":
        return "IUML"
    # Streamline CPI(M) variants
    if "CPI" in p and ("(M)" in p or "[M]" in p or " [M] " in p):
        return "CPI(M)"
    # Streamline KEC(M) variants
    if p in {"KC(M)", "KC[M]", "KEC(M)"} or (("KC" in p or "KEC" in p) and ("(M)" in p or "[M]" in p)):
        return "KEC(M)"
    # Streamline Congress (Secular)
    if "CONGRESS(SECULAR)" in p or "CONGRESS (SECULAR)" in p:
        return "C(S)"
    return p


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "unknown"


RELATIONSHIP_MARKERS = {"S/o", "D/o", "W/o", "S/O", "D/O", "W/O", "S/O.", "D/O.", "W/O.", "S/O,", "D/O,", "W/O,"}
CATEGORY_MARKERS = {"GENERAL", "SC", "ST", "OBC"}
SEX_MARKERS = {"MALE", "FEMALE", "THIRD"}
NUMERIC_TOKEN_RE = re.compile(r"^\d[\d,]*(?:\.\d+)?$")
SYMBOL_TOKENS = {
    "LADDER", "LOTUS", "HAND", "SICKLE", "HAMMER", "STAR", "POT", "PINEAPPLE", "FOOTBALL",
    "BATTERY", "TORCH", "ELEPHANT", "AUTO", "RICKSHAW", "DISH", "ANTENNA", "GLASS", "TUMBLER",
    "WALKING", "STICK", "CUP", "SAUCER", "CAKE", "FLUTE", "ICE", "CREAM", "PLATE", "STAND",
    "BOAT", "MAN", "SAIL", "FROCK", "WINDOW", "HELMET", "HELICOPTER", "GAS", "CYLINDER",
    "COAT", "SCISSORS", "TELEVISION", "VIOLIN", "GINGER", "CCTV", "CAMERA", "RING", "NECKLACE",
    "CAULIFLOWER", "BEAD", "FUNNEL", "CRANE", "WHISTLE", "CLOCK", "TWO", "LEAVES", "ALMIRAH",
    "REMOTE", "BICYCLE", "PUMP", "BALLOON", "COCONUT", "FARM", "TRACTOR", "CHALATA", "KISAN",
    "EARS", "CORN", "LADY", "FARMER", "CARRYING", "PADDY", "HEAD", "ARROW", "DIAMOND",
    "REFRIGERATOR", "CAN", "BUCKET", "TV"
}


def clean_symbol_token(token: str) -> str:
    return re.sub(r"[^A-Z]", "", token.upper())


def is_symbol_token(token: str) -> bool:
    cleaned = clean_symbol_token(token)
    return bool(cleaned) and cleaned in SYMBOL_TOKENS


def is_probable_party_token(token: str) -> bool:
    if token in SEX_MARKERS or token in CATEGORY_MARKERS or token in RELATIONSHIP_MARKERS:
        return False
    if NUMERIC_TOKEN_RE.fullmatch(token):
        return False
    cleaned = re.sub(r"[^A-Za-z()]", "", token)
    if len(cleaned) < 2:
        return False
    lower_count = sum(1 for char in token if char.islower())
    return lower_count <= 1 and not is_symbol_token(token)


def parse_int_token(token: str) -> int:
    return int(token.replace(",", "").strip())


def parse_float_token(token: str) -> float:
    cleaned = token.replace(",", "").replace("%", "").strip()
    return float(cleaned)


class KeralaElectionFetcher:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.raw_root = output_root / "raw"
        self.processed_root = output_root / "processed"
        self.detail_root = self.processed_root / "constituency_results"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0 Safari/537.36"
                )
            }
        )

    def run(self, years: Iterable[int]) -> None:
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.detail_root.mkdir(parents=True, exist_ok=True)

        all_results: list[pd.DataFrame] = []
        constituencies: list[dict[str, object]] = []

        for year in years:
            if year == 2011:
                result_df, constituency_df = self.fetch_2011()
            elif year == 2016:
                result_df, constituency_df = self.fetch_2016()
            elif year == 2021:
                result_df, constituency_df = self.fetch_2021()
            else:
                raise ValueError(f"Unsupported year: {year}")

            all_results.append(result_df)
            constituencies.extend(constituency_df.to_dict(orient="records"))
            self.write_constituency_files(result_df, year)

        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values(["year", "constituency_number", "votes"], ascending=[True, True, False])
        combined.to_csv(self.processed_root / "kerala_assembly_candidate_results.csv", index=False)

        constituency_df = pd.DataFrame(constituencies).drop_duplicates(
            subset=["year", "constituency_id"]
        ).sort_values(["year", "constituency_number"])
        constituency_df.to_csv(self.processed_root / "constituencies.csv", index=False)

        summary_df = self.build_state_summary(combined)
        summary_df.to_csv(self.processed_root / "state_summary.csv", index=False)

        manifest = {
            "years": list(years),
            "outputs": {
                "candidate_results": str(self.processed_root / "kerala_assembly_candidate_results.csv"),
                "constituencies": str(self.processed_root / "constituencies.csv"),
                "state_summary": str(self.processed_root / "state_summary.csv"),
                "detail_dir": str(self.detail_root),
            },
        }
        (self.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def fetch_json(self, url: str, **kwargs) -> dict:
        response = self.session.get(url, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()

    def fetch_text(self, url: str, **kwargs) -> str:
        response = self.session.get(url, timeout=60, **kwargs)
        response.raise_for_status()
        return response.text

    def download(self, url: str, destination: Path) -> Path:
        if destination.exists():
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        response = self.session.get(url, timeout=120)
        response.raise_for_status()
        destination.write_bytes(response.content)
        return destination

    def get_constituencies_from_show_lac(
        self,
        year: int,
        show_lac_url_template: str,
        extra_source_template: str,
    ) -> list[Constituency]:
        constituencies: list[Constituency] = []
        for district_id, district_name in DISTRICTS.items():
            payload = self.fetch_json(show_lac_url_template.format(district_id=district_id))
            select_html = payload["selectHtml"]
            soup = BeautifulSoup(select_html, "html.parser")
            for option in soup.select("option[value]"):
                value = option.get("value", "").strip()
                label = option.get_text(" ", strip=True)
                if not value:
                    continue
                number_match = re.match(r"(\d+)\s*:\s*(.+)", label)
                if not number_match:
                    continue
                constituency_number = int(number_match.group(1))
                constituency_name = number_match.group(2).strip()
                constituency_id = int(value)
                constituencies.append(
                    Constituency(
                        year=year,
                        district_id=district_id,
                        district=district_name,
                        constituency_id=constituency_id,
                        constituency_number=constituency_number,
                        constituency=constituency_name,
                        source_url=extra_source_template.format(constituency_number=constituency_number),
                    )
                )
        return constituencies

    def fetch_2011(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        constituencies = self.get_constituencies_from_show_lac(
            year=2011,
            show_lac_url_template=f"{CEO_BASE}/candidates/show_lac/?id={{district_id}}",
            extra_source_template=f"{CEO_BASE}/pdf/form20/{{constituency_number:03d}}.pdf",
        )
        nominations = self.parse_2011_nominations()
        records: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []
        pdf_root = self.raw_root / "pdfs" / "2011"
        for item in constituencies:
            html_rows = self.parse_2011_candidate_rows(self.fetch_text(f"{CEO_BASE}/candidates/loadhtml/{item.constituency_id}"))
            source_url = f"{CEO_BASE}/candidates/loadhtml/{item.constituency_id}"
            merged_rows: list[dict[str, object]] | None = None
            if html_rows and all("votes" in row for row in html_rows):
                merged_rows = html_rows
            else:
                try:
                    pdf_path = self.download(
                        f"{CEO_BASE}/pdf/form20/{item.constituency_number:03d}.pdf",
                        pdf_root / f"{item.constituency_number:03d}.pdf",
                    )
                    totals_by_name = self.parse_form20_totals(pdf_path)
                    try:
                        merged_rows = self.merge_candidate_rows_with_totals(
                            html_rows, totals_by_name, include_nota=False
                        )
                    except RuntimeError:
                        nomination_rows = nominations.get(item.constituency_number, [])
                        try:
                            merged_rows = self.merge_candidate_rows_with_totals(
                                nomination_rows, totals_by_name, include_nota=False
                            )
                        except RuntimeError:
                            merged_rows = self.merge_candidate_rows_with_unknown_parties(
                                html_rows + nomination_rows,
                                totals_by_name,
                                include_nota=False,
                            )
                        source_url = f"{CEO_BASE}/pdf/form20/{item.constituency_number:03d}.pdf"
                except Exception as e:
                    skipped.append(
                        {
                            "constituency_number": item.constituency_number,
                            "constituency": item.constituency,
                            "error": str(e),
                        }
                    )
                    continue

            if not merged_rows:
                skipped.append(
                    {
                        "constituency_number": item.constituency_number,
                        "constituency": item.constituency,
                        "error": "merged_rows empty",
                    }
                )
                continue

            valid_votes_total = sum(row["votes"] for row in merged_rows)
            if valid_votes_total <= 0:
                skipped.append(
                    {
                        "constituency_number": item.constituency_number,
                        "constituency": item.constituency,
                        "error": "valid_votes_total <= 0",
                    }
                )
                continue
            max_votes = max(row["votes"] for row in merged_rows)
            for row in merged_rows:
                records.append(
                    {
                        "year": 2011,
                        "district": item.district,
                        "constituency_id": item.constituency_id,
                        "constituency_number": item.constituency_number,
                        "constituency": item.constituency,
                        "candidate": row["candidate"],
                        "party": normalize_party_name(row["party"]),
                        "votes": row["votes"],
                        "vote_share": round((row["votes"] / valid_votes_total) * 100, 4),
                        "is_winner": row["votes"] == max_votes,
                        "source_url": source_url,
                    }
                )

        if skipped:
            skipped_path = self.output_root / "2011_skipped_constituencies.json"
            skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")

        return pd.DataFrame(records), self.constituencies_to_df(constituencies)

    def fetch_2016(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        cached_constituencies_path = self.processed_root / "constituencies.csv"
        if cached_constituencies_path.exists():
            cached = pd.read_csv(cached_constituencies_path)
            cached_2016 = cached[cached["year"] == 2016].copy()
            if not cached_2016.empty:
                constituencies = [
                    Constituency(
                        year=int(r["year"]),
                        district_id=int(r["district_id"]),
                        district=str(r["district"]),
                        constituency_id=int(r["constituency_id"]),
                        constituency_number=int(r["constituency_number"]),
                        constituency=str(r["constituency"]),
                        source_url=str(r.get("source_url", "")),
                    )
                    for _, r in cached_2016.iterrows()
                ]
            else:
                constituencies = self.get_constituencies_from_show_lac(
                    year=2016,
                    show_lac_url_template=f"{CEO_BASE}/expenditureGE2016/show_lac/?id={{district_id}}",
                    extra_source_template=f"{CEO_BASE}/ceokerala/pdf/BOOTH_WISE_RESULTS/GE2016/{{constituency_number:03d}}.pdf",
                )
        else:
            constituencies = self.get_constituencies_from_show_lac(
                year=2016,
                show_lac_url_template=f"{CEO_BASE}/expenditureGE2016/show_lac/?id={{district_id}}",
                extra_source_template=f"{CEO_BASE}/ceokerala/pdf/BOOTH_WISE_RESULTS/GE2016/{{constituency_number:03d}}.pdf",
            )

        records: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []
        pdf_root = self.raw_root / "pdfs" / "2016"

        # Persist constituency metadata early so a later Ctrl+C doesn't
        # force re-downloading constituency lists.
        if not cached_constituencies_path.exists():
            early_constituencies_df = self.constituencies_to_df(constituencies)
            self.processed_root.mkdir(parents=True, exist_ok=True)
            early_constituencies_df.to_csv(cached_constituencies_path, index=False)

        for item in constituencies:
            candidates_html = self.fetch_text(f"{CEO_BASE}/expenditureGE2016/loadhtml/{item.constituency_id}")
            candidate_rows = self.parse_2016_candidate_rows(candidates_html)
            totals_by_name: dict[str, int] | None = None

            try:
                pdf_path = self.download(
                    f"{CEO_BASE}/ceokerala/pdf/BOOTH_WISE_RESULTS/GE2016/{item.constituency_number:03d}.pdf",
                    pdf_root / f"{item.constituency_number:03d}.pdf",
                )
                totals_by_name = self.parse_form20_totals(pdf_path)
                merged_rows = self.merge_candidate_rows_with_totals(candidate_rows, totals_by_name, include_nota=True)
            except RuntimeError as e:
                if totals_by_name is None:
                    skipped.append(
                        {
                            "constituency_number": item.constituency_number,
                            "constituency": item.constituency,
                            "error": f"parse_form20_totals failed: {e}",
                        }
                    )
                    continue
                try:
                    merged_rows = self.merge_candidate_rows_with_unknown_parties(candidate_rows, totals_by_name, include_nota=True)
                except Exception as e:
                    skipped.append(
                        {
                            "constituency_number": item.constituency_number,
                            "constituency": item.constituency,
                            "error": f"merge/party-match failed: {e}",
                        }
                    )
                    continue
            except Exception as e:
                skipped.append(
                    {
                        "constituency_number": item.constituency_number,
                        "constituency": item.constituency,
                        "error": str(e),
                    }
                )
                continue

            if not merged_rows:
                skipped.append(
                    {
                        "constituency_number": item.constituency_number,
                        "constituency": item.constituency,
                        "error": "merged_rows empty",
                    }
                )
                continue

            valid_votes_total = sum(row["votes"] for row in merged_rows)
            if valid_votes_total <= 0:
                skipped.append(
                    {
                        "constituency_number": item.constituency_number,
                        "constituency": item.constituency,
                        "error": "valid_votes_total <= 0",
                    }
                )
                continue
            max_votes = max(row["votes"] for row in merged_rows)
            for row in merged_rows:
                records.append(
                    {
                        "year": 2016,
                        "district": item.district,
                        "constituency_id": item.constituency_id,
                        "constituency_number": item.constituency_number,
                        "constituency": item.constituency,
                        "candidate": row["candidate"],
                        "party": normalize_party_name(row["party"]),
                        "votes": row["votes"],
                        "vote_share": round((row["votes"] / valid_votes_total) * 100, 4),
                        "is_winner": row["votes"] == max_votes,
                        "source_url": f"{CEO_BASE}/ceokerala/pdf/BOOTH_WISE_RESULTS/GE2016/{item.constituency_number:03d}.pdf",
                    }
                )

        if skipped:
            skipped_path = self.output_root / "2016_skipped_constituencies.json"
            skipped_path.write_text(json.dumps(skipped, indent=2), encoding="utf-8")

        return pd.DataFrame(records), self.constituencies_to_df(constituencies)

    def fetch_2021(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Building constituency metadata for 2021 requires a CEO API call.
        # If we already generated `constituencies.csv` previously, reuse it to
        # keep re-runs offline/fast.
        cached_constituencies_path = self.processed_root / "constituencies.csv"
        if cached_constituencies_path.exists():
            cached = pd.read_csv(cached_constituencies_path)
            cached_2021 = cached[cached["year"] == 2021].copy()
            if not cached_2021.empty:
                constituencies_df = cached_2021
            else:
                constituencies_df = self.constituencies_to_df(self.get_2021_constituencies())
        else:
            constituencies_df = self.constituencies_to_df(self.get_2021_constituencies())
        report_path = self.download(
            f"{CEO_BASE}/pdf/GE-2021/statistical_report.pdf",
            self.raw_root / "reports" / "2021" / "statistical_report.pdf",
        )
        result_df = self.parse_2021_statistical_report(report_path)
        result_df = result_df.merge(
            constituencies_df[["constituency_number", "district", "constituency_id"]],
            on="constituency_number",
            how="left",
        )
        result_df["year"] = 2021
        result_df["source_url"] = f"{CEO_BASE}/pdf/GE-2021/statistical_report.pdf"
        result_df = result_df[
            [
                "year",
                "district",
                "constituency_id",
                "constituency_number",
                "constituency",
                "candidate",
                "party",
                "votes",
                "vote_share",
                "is_winner",
                "source_url",
            ]
        ]
        return result_df, constituencies_df

    def parse_2016_candidate_rows(self, html: str) -> list[dict[str, str]]:
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.select("tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            rows.append({"candidate": cells[1], "party": cells[2]})
        if not rows:
            raise RuntimeError("No 2016 candidate rows found")
        return rows

    def parse_form20_totals(self, pdf_path: Path) -> dict[str, int]:
        reader = PdfReader(str(pdf_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[-2:])
        if not text.strip():
            raise RuntimeError(f"No extractable text in {pdf_path}")
        map_match = re.search(r"([A-U]\s*-\s*.+?)\s+Place", text, flags=re.DOTALL)
        if not map_match:
            raise RuntimeError(f"Could not parse candidate map from {pdf_path}")
        mapping_text = map_match.group(1).replace("\n", " ")
        candidate_pairs = re.findall(r"([A-U])\s*-\s*([^,]+)", mapping_text)
        candidate_map = {letter: name.strip() for letter, name in candidate_pairs}

        totals_match = re.search(
            r"Total\s+Votes\s+Polled\s+((?:\d+\s+){24,30})",
            text,
            flags=re.DOTALL,
        )
        if not totals_match:
            raise RuntimeError(f"Could not parse total-vote line from {pdf_path}")
        totals = [int(token) for token in re.findall(r"\d+", totals_match.group(1))]
        candidate_totals = {letter: totals[index] for index, letter in enumerate(LETTER_ORDER)}
        return {
            candidate_map.get(letter, letter): votes
            for letter, votes in candidate_totals.items()
            if letter in candidate_map
        }

    def parse_2011_candidate_rows(self, html: str) -> list[dict[str, object]]:
        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict[str, object]] = []
        for tr in soup.select("tbody tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if len(cells) >= 5:
                rows.append({
                    "candidate": cells[0],
                    "party": cells[1],
                    "votes": int(cells[4].replace(',', '')),
                })
        return rows

    def parse_2011_nominations(self) -> dict[int, list[dict[str, str]]]:
        pdf_path = self.download(
            f"{CEO_BASE}/pdf/generalelection2011/nominations/NOMINATION_STMT-04.pdf",
            self.raw_root / "pdfs" / "2011" / "NOMINATION_STMT-04.pdf",
        )
        reader = PdfReader(str(pdf_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        nominations: dict[int, list[dict[str, str]]] = {}
        current_lac: int | None = None
        for line in lines:
            lac_match = re.match(r"^(\d+)\s+(.+?)No & Name of LAC\s*:$", line)
            if lac_match:
                current_lac = int(lac_match.group(1))
                nominations.setdefault(current_lac, [])
                continue

            if current_lac is None:
                continue
            if line.startswith("Sl No.") or line.startswith("Page ") or "Election Department" in line:
                continue
            if "No & Name of District" in line:
                continue

            candidate_match = re.match(
                r"^(.*?)\s+[MF]\s+\d+\s+\d{4}-\d{2}-\d{2}\s*([A-Z][A-Z\[\]\-\.\/]*)\d+$",
                line,
            )
            if not candidate_match:
                continue

            candidate = candidate_match.group(1).strip(" .")
            party = candidate_match.group(2).strip()
            nominations[current_lac].append({"candidate": candidate, "party": party})

        return nominations

    def merge_candidate_rows_with_totals(
        self,
        candidate_rows: list[dict[str, str]],
        totals_by_name: dict[str, int],
        include_nota: bool = False,
    ) -> list[dict[str, object]]:
        exact_lookup = {normalize_name(name): (normalize_name(name), votes) for name, votes in totals_by_name.items()}
        token_lookup = {token_sort_key(name): (normalize_name(name), votes) for name, votes in totals_by_name.items()}
        merged: list[dict[str, object]] = []
        used_keys: set[str] = set()

        for row in candidate_rows:
            normalized = normalize_name(row["candidate"])
            token_key = token_sort_key(row["candidate"])
            match = exact_lookup.get(normalized) or token_lookup.get(token_key)
            if match is None:
                continue
            matched_key, votes = match
            if matched_key in used_keys:
                continue
            used_keys.add(matched_key)
            merged.append(
                {
                    "candidate": row["candidate"],
                    "party": row["party"],
                    "votes": votes,
                }
            )

        unmatched_totals = {normalize_name(name) for name in totals_by_name if normalize_name(name) not in used_keys and normalize_name(name) != "NOTA"}
        if unmatched_totals:
            raise RuntimeError(f"Unmatched vote totals for candidates: {sorted(unmatched_totals)}")

        if include_nota:
            for name, votes in totals_by_name.items():
                if normalize_name(name) == "NOTA":
                    merged.append({"candidate": "NOTA", "party": "NOTA", "votes": votes})
                    break

        return merged

    def merge_candidate_rows_with_unknown_parties(
        self,
        candidate_rows: list[dict[str, str]],
        totals_by_name: dict[str, int],
        include_nota: bool = False,
    ) -> list[dict[str, object]]:
        exact_party_lookup: dict[str, tuple[str, str]] = {}
        token_party_lookup: dict[str, tuple[str, str]] = {}
        for row in candidate_rows:
            candidate_name = row.get("candidate", "").strip()
            party_name = row.get("party", "").strip() or "UNKNOWN"
            if not candidate_name:
                continue
            exact_party_lookup.setdefault(normalize_name(candidate_name), (candidate_name, party_name))
            token_party_lookup.setdefault(token_sort_key(candidate_name), (candidate_name, party_name))

        merged: list[dict[str, object]] = []
        for name, votes in totals_by_name.items():
            normalized = normalize_name(name)
            if normalized == "NOTA" and not include_nota:
                continue
            match = exact_party_lookup.get(normalized) or token_party_lookup.get(token_sort_key(name))
            if match is None:
                candidate_name, party_name = name, "UNKNOWN"
            else:
                candidate_name, party_name = match
            if normalize_name(name) == "NOTA":
                candidate_name, party_name = "NOTA", "NOTA"
            merged.append({"candidate": candidate_name, "party": party_name, "votes": votes})

        return merged

    def parse_2021_statistical_report(self, pdf_path: Path) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        skipped_rows: list[dict[str, object]] = []
        valid_votes_by_constituency: dict[int, int] = {}
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = pdf.pages[162:]
            current_constituency_number: int | None = None
            current_constituency_name: str | None = None
            current_row_tokens: list[str] | None = None
            pending_next_tokens: list[str] = []

            for page in pages:
                for line in self.extract_pdf_lines(page):
                    tokens = line["tokens"]
                    min_x0 = line["min_x0"]
                    if not tokens or tokens == ["0"]:
                        continue
                    if tokens[0] in {"Election", "VALID", "VOTES", "CANDIDATE", "POLLED", "Constituency", "Page", "ST1006"}:
                        continue
                    if len(tokens) >= 5 and tokens[1] == "." and "TOTAL" in tokens and "ELECTORS" in tokens:
                        if current_row_tokens:
                            try:
                                records.append(
                                    self.parse_2021_candidate_record(
                                        current_row_tokens,
                                        current_constituency_number,
                                        current_constituency_name,
                                    )
                                )
                            except Exception:
                                skipped_rows.append(
                                    {
                                        "constituency_number": current_constituency_number,
                                        "constituency": current_constituency_name,
                                        "tokens": current_row_tokens,
                                    }
                                )
                            current_row_tokens = None
                        pending_next_tokens = []
                        current_constituency_number = int(tokens[0])
                        total_idx = tokens.index("TOTAL")
                        current_constituency_name = " ".join(tokens[2:total_idx]).title()
                        continue
                    if tokens[:3] == ["TURN", "OUT", "TOTAL:"]:
                        if current_row_tokens:
                            try:
                                records.append(
                                    self.parse_2021_candidate_record(
                                        current_row_tokens,
                                        current_constituency_number,
                                        current_constituency_name,
                                    )
                                )
                            except Exception:
                                skipped_rows.append(
                                    {
                                        "constituency_number": current_constituency_number,
                                        "constituency": current_constituency_name,
                                        "tokens": current_row_tokens,
                                    }
                                )
                            current_row_tokens = None
                        # Capture valid votes from the turnout summary line.
                        # This is later used to reconstruct missing NOTA votes/share.
                        if current_constituency_number is not None:
                            numeric_tokens = [
                                token for token in tokens if NUMERIC_TOKEN_RE.fullmatch(token)
                            ]
                            int_tokens = [t for t in numeric_tokens if "." not in t]
                            if int_tokens:
                                valid_votes_by_constituency[current_constituency_number] = parse_int_token(
                                    int_tokens[-1]
                                )
                        pending_next_tokens = []
                        continue
                    if self.is_2021_candidate_row_start(tokens):
                        if current_row_tokens:
                            try:
                                records.append(
                                    self.parse_2021_candidate_record(
                                        current_row_tokens,
                                        current_constituency_number,
                                        current_constituency_name,
                                    )
                                )
                            except Exception:
                                skipped_rows.append(
                                    {
                                        "constituency_number": current_constituency_number,
                                        "constituency": current_constituency_name,
                                        "tokens": current_row_tokens,
                                    }
                                )
                        current_row_tokens = tokens + pending_next_tokens
                        pending_next_tokens = []
                        continue
                    if current_row_tokens and self.is_2021_candidate_row_continuation(tokens, min_x0):
                        if tokens[0].isdigit() and len(tokens) > 1 and tokens[1] in CATEGORY_MARKERS:
                            numeric_count = sum(1 for token in current_row_tokens if NUMERIC_TOKEN_RE.fullmatch(token))
                            if numeric_count >= 4:
                                pending_next_tokens.extend(tokens)
                                continue
                        current_row_tokens.extend(tokens)
                        continue

            if current_row_tokens:
                try:
                    records.append(
                        self.parse_2021_candidate_record(
                            current_row_tokens,
                            current_constituency_number,
                            current_constituency_name,
                        )
                    )
                except Exception:
                    skipped_rows.append(
                        {
                            "constituency_number": current_constituency_number,
                            "constituency": current_constituency_name,
                            "tokens": current_row_tokens,
                        }
                    )

        df = pd.DataFrame(records)
        df = df[
            ~df["candidate"].astype(str).str.contains(
                r"Disclaimer|Postal Votes|R\.P\. Act|Statutory|Returning Officers",
                regex=True,
                na=False,
            )
        ].copy()

        # Reconstruct missing NOTA rows (pdf extraction sometimes parses
        # the NOTA votes into a candidate row but loses the candidate/party
        # label, resulting in `candidate` being NaN for exactly one row per
        # constituency).
        for constituency_number, valid_votes in valid_votes_by_constituency.items():
            subset = df[df["constituency_number"] == constituency_number]
            if subset.empty:
                continue
            if (subset["candidate"].astype(str) == "NOTA").any():
                continue

            candidate_is_missing = subset["candidate"].isna() | (
                subset["candidate"].astype(str).str.strip() == ""
            )
            missing_rows = subset[candidate_is_missing]

            if not missing_rows.empty:
                nota_votes = int(missing_rows.iloc[0]["votes"])
                nota_vote_share = round((nota_votes / valid_votes) * 100, 2)
                df.loc[missing_rows.index[0], "candidate"] = "NOTA"
                df.loc[missing_rows.index[0], "party"] = "NOTA"
                df.loc[missing_rows.index[0], "vote_share"] = nota_vote_share
                continue

            # Fallback: if NOTA votes weren't parsed at all, use the remainder.
            votes_sum = int(subset["votes"].sum())
            nota_votes = valid_votes - votes_sum
            if nota_votes <= 0:
                continue

            nota_vote_share = round((nota_votes / valid_votes) * 100, 2)
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "serial_number": 0,
                                "constituency_number": constituency_number,
                                "constituency": subset["constituency"].iloc[0],
                                "candidate": "NOTA",
                                "party": "NOTA",
                                "votes": nota_votes,
                                "vote_share": nota_vote_share,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        # Drop skipped rows that are resolved by NOTA reconstruction.
        resolved_constituencies = set(
            df.loc[df["candidate"].astype(str) == "NOTA", "constituency_number"].tolist()
        )
        skipped_rows = [
            row
            for row in skipped_rows
            if row.get("constituency_number") not in resolved_constituencies
        ]

        skipped_path = self.output_root / "2021_skipped_rows.json"
        skipped_path.write_text(json.dumps(skipped_rows, indent=2), encoding="utf-8")
        df["is_winner"] = df.groupby("constituency_number")["votes"].transform("max") == df["votes"]
        return df

    def extract_pdf_lines(self, page) -> list[dict[str, object]]:
        lines: dict[float, list[dict[str, object]]] = {}
        for word in page.extract_words(use_text_flow=True):
            key = round(word["top"], 1)
            lines.setdefault(key, []).append(word)
        ordered_lines: list[dict[str, object]] = []
        for key in sorted(lines):
            words = sorted(lines[key], key=lambda item: item["x0"])
            tokens = [word["text"].strip() for word in words if word["text"].strip()]
            if not tokens:
                continue
            ordered_lines.append(
                {
                    "tokens": tokens,
                    "min_x0": min(word["x0"] for word in words),
                }
            )
        return ordered_lines

    def is_2021_candidate_row_start(self, tokens: list[str]) -> bool:
        if not tokens or not tokens[0].isdigit():
            return False
        if len(tokens) > 1 and tokens[1] == ".":
            return False
        if len(tokens) > 1 and tokens[1] in CATEGORY_MARKERS:
            return False
        return True

    def is_2021_candidate_row_continuation(self, tokens: list[str], min_x0: float) -> bool:
        if not tokens:
            return False
        if tokens[0] in {"TURN", "Constituency", "VALID", "VOTES", "CANDIDATE", "Page", "ST1006", "Election"}:
            return False
        if tokens == ["0"]:
            return False
        if tokens[0].isdigit() and len(tokens) > 1 and tokens[1] in CATEGORY_MARKERS:
            return True
        if any(token in SEX_MARKERS for token in tokens):
            return True
        # Some candidate-row numeric fields (NOTA votes/share) are extracted on the "right"
        # of the page with commas, making `min_x0` heuristics fail. If the line contains numeric
        # tokens, treat it as a continuation of the current candidate row.
        if any(NUMERIC_TOKEN_RE.fullmatch(token) for token in tokens):
            return True
        if min_x0 < 220 and not (tokens[0].isdigit() and len(tokens) > 1 and tokens[1] == "."):
            return True
        return False

    def parse_2021_candidate_record(
        self,
        tokens: list[str],
        constituency_number: int | None,
        constituency_name: str | None,
    ) -> dict[str, object]:
        if constituency_number is None or constituency_name is None:
            raise RuntimeError(f"Encountered candidate row before constituency header: {tokens}")
        numeric_positions = [idx for idx, token in enumerate(tokens) if NUMERIC_TOKEN_RE.fullmatch(token)]
        if len(numeric_positions) < 4:
            raise RuntimeError(f"Unexpected 2021 candidate row: {tokens}")
        general_idx, postal_idx, total_idx, pct_idx = numeric_positions[-4:]
        votes = parse_int_token(tokens[total_idx])
        vote_share = parse_float_token(tokens[pct_idx])
        serial_number = int(tokens[0])

        if len(tokens) > 1 and tokens[1] == "NOTA":
            return {
                "serial_number": serial_number,
                "constituency_number": constituency_number,
                "constituency": constituency_name,
                "candidate": "NOTA",
                "party": "NOTA",
                "votes": votes,
                "vote_share": vote_share,
            }

        sex_idx = next((idx for idx, token in enumerate(tokens) if token in SEX_MARKERS), None)
        first_numeric_idx = next((idx for idx in numeric_positions if idx > 0), numeric_positions[0])
        candidate_head = [
            token
            for token in tokens[1:first_numeric_idx]
            if token not in SEX_MARKERS and token not in CATEGORY_MARKERS and not is_symbol_token(token)
        ]
        candidate_tail = [
            token
            for token in tokens[pct_idx + 1:]
            if not NUMERIC_TOKEN_RE.fullmatch(token)
            and token not in SEX_MARKERS
            and token not in CATEGORY_MARKERS
            and not is_probable_party_token(token)
            and not is_symbol_token(token)
            and len(token.strip('.')) > 1
        ]
        candidate = " ".join(candidate_head + candidate_tail).strip()

        party = "UNKNOWN"
        party_segments: list[list[str]] = []
        if sex_idx is not None and sex_idx + 1 < general_idx:
            party_segments.append(tokens[sex_idx + 1:general_idx])
        if first_numeric_idx < general_idx:
            party_segments.append(tokens[first_numeric_idx:general_idx])
        if pct_idx + 1 < len(tokens):
            party_segments.append(tokens[pct_idx + 1:])
        for segment in party_segments:
            party_candidates = [token for token in segment if is_probable_party_token(token)]
            if party_candidates:
                party = party_candidates[0]
                break
        if party == "UNKNOWN" and any(marker in tokens for marker in RELATIONSHIP_MARKERS):
            party = "IND"
        if party == "UNKNOWN" and serial_number > 0 and votes < 2000:
            party = "IND"

        return {
            "serial_number": serial_number,
            "constituency_number": constituency_number,
            "constituency": constituency_name,
            "candidate": candidate,
            "party": normalize_party_name(party),
            "votes": votes,
            "vote_share": vote_share,
        }

    def constituencies_to_df(self, constituencies: list[Constituency]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "year": item.year,
                    "district_id": item.district_id,
                    "district": item.district,
                    "constituency_id": item.constituency_id,
                    "constituency_number": item.constituency_number,
                    "constituency": item.constituency,
                    "source_url": item.source_url,
                }
                for item in constituencies
            ]
        )

    def get_2021_constituencies(self) -> list[Constituency]:
        constituencies = self.get_constituencies_from_show_lac(
            year=2021,
            show_lac_url_template=f"{CEO_BASE}/expenditurege2021/show_lac/?id={{district_id}}",
            extra_source_template=f"{CEO_BASE}/ceokerala/pdf/BOOTH_WISE_RESULTS/GE2021/{{constituency_number:03d}}.pdf",
        )
        return constituencies

    def build_state_summary(self, candidate_results: pd.DataFrame) -> pd.DataFrame:
        winners = candidate_results[candidate_results["is_winner"]].copy()
        winners["party"] = winners["party"].fillna("UNKNOWN")
        seats_by_party = (
            winners.groupby(["year", "party"], as_index=False)
            .agg(seats_won=("candidate", "count"), votes=("votes", "sum"))
            .sort_values(["year", "seats_won", "votes"], ascending=[True, False, False])
        )
        turnout_summary = (
            candidate_results.groupby("year", as_index=False)
            .agg(
                constituencies=("constituency_id", "nunique"),
                candidates=("candidate", "count"),
                total_valid_votes=("votes", "sum"),
            )
        )
        return seats_by_party.merge(turnout_summary, on="year", how="left")

    def write_constituency_files(self, result_df: pd.DataFrame, year: int) -> None:
        year_root = self.detail_root / str(year)
        year_root.mkdir(parents=True, exist_ok=True)
        for (constituency_number, constituency), frame in result_df.groupby(
            ["constituency_number", "constituency"], sort=True
        ):
            file_path = year_root / f"{int(constituency_number):03d}_{slugify(constituency)}.csv"
            frame.sort_values("votes", ascending=False).to_csv(file_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Kerala assembly election data for 2011, 2016, and 2021.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(YEARS),
        help="Election years to fetch. Defaults to 2011 2016 2021.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for raw and processed files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid_years = sorted(set(args.years) - set(YEARS))
    if invalid_years:
        raise SystemExit(f"Unsupported year(s): {', '.join(map(str, invalid_years))}")
    fetcher = KeralaElectionFetcher(output_root=args.output_dir)
    fetcher.run(args.years)


if __name__ == "__main__":
    main()

