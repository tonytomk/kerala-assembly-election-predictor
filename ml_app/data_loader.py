from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    base_dir: Path

    def candidate_results_path(self, year: int) -> Path:
        return self.base_dir / f"data_{year}" / "processed" / "kerala_assembly_candidate_results.csv"

    def constituencies_path(self) -> Path:
        return self.base_dir / "data_2021" / "processed" / "constituencies.csv"

    def candidates_2026_path(self) -> Path:
        return self.base_dir / "data_2026" / "processed" / "kerala_2026_candidates.csv"

    def by_election_candidate_results_path(self) -> Path:
        return (
            self.base_dir
            / "data_byelections_2021_2026"
            / "processed"
            / "kerala_assembly_byelection_candidate_results.csv"
        )


def load_candidate_results_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure consistent dtypes.
    df["year"] = df["year"].astype(int)
    df["constituency_number"] = df["constituency_number"].astype(int)
    df["party"] = df["party"].astype(str)
    df["candidate"] = df["candidate"].astype(str)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)
    df["vote_share"] = pd.to_numeric(df["vote_share"], errors="coerce").fillna(0.0)
    return df


def load_all_elections(
    base_dir: Path,
    years: Iterable[int] = (2011, 2016, 2021),
) -> pd.DataFrame:
    paths = DataPaths(base_dir=base_dir)
    dfs: list[pd.DataFrame] = []
    for y in years:
        dfs.append(load_candidate_results_csv(paths.candidate_results_path(y)))

    # Add by-election observations as additional rows (treated as the election year
    # they occurred in).
    by_path = paths.by_election_candidate_results_path()
    by_df = load_candidate_results_csv(by_path)
    dfs.append(by_df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def load_2026_candidates(base_dir: Path) -> pd.DataFrame:
    """
    Load the official 2026 Kerala Assembly candidate list.

    Returns a DataFrame with columns:
        district, constituency_number, constituency, alliance, party, candidate_name
    """
    path = DataPaths(base_dir=base_dir).candidates_2026_path()
    df = pd.read_csv(path)

    # Normalise column names so they match the rest of the codebase.
    df = df.rename(columns={"constituency_no": "constituency_number"})
    df["constituency_number"] = df["constituency_number"].astype(int)
    df["alliance"] = df["alliance"].astype(str).str.strip()
    df["party"] = df["party"].astype(str).str.strip()
    df["candidate_name"] = df["candidate_name"].astype(str).str.strip()
    return df

