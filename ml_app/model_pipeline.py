from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from ml_app.data_loader import load_all_elections, load_2026_candidates
from ml_app.predictor import predict_top3_parties_2026


@dataclass(frozen=True)
class ModelConfig:
    target_year: int = 2026
    by_election_years: tuple[int, ...] = (2022, 2023, 2024, 2025)
    lambda_recency: float = 0.55


def default_cache_dir(base_dir: Path) -> Path:
    return base_dir / "ml_app_cache"


def _enrich_with_2026_candidates(pred_df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Join official 2026 candidate data onto the prediction DataFrame.

    The predictor collapses parties into alliance-level labels (LDF/UDF/NDA or
    specific party names for minor parties).  We match each predicted row to the
    official 2026 candidate list using the alliance column from that CSV.

    New columns added:
        candidate_name  – real candidate name (or "Unknown")
        actual_party    – party abbreviation from the official list
        actual_alliance – alliance tag from the official list (LDF/UDF/NDA/AAP/…)
    """
    try:
        cands = load_2026_candidates(base_dir)
    except FileNotFoundError:
        pred_df["candidate_name"] = "Unknown"
        pred_df["actual_party"] = pred_df["party"]
        pred_df["actual_alliance"] = pred_df["party"]
        return pred_df

    # Build a lookup: (constituency_number, alliance) -> {candidate_name, party}
    # The predictor's `party` column is already an alliance tag (LDF/UDF/NDA) for
    # the three main blocs, so we join on that.
    cand_lookup = cands[["constituency_number", "alliance", "party", "candidate_name"]].copy()
    cand_lookup = cand_lookup.rename(columns={
        "party": "actual_party",
        "alliance": "actual_alliance",
        "candidate_name": "candidate_name",
    })

    # The pred_df `party` column holds the predicted alliance key.
    # Map it to the same alliance labels used in the 2026 CSV.
    # The CSV uses: LDF, UDF, NDA, AAP (and occasionally independent labels).
    merged = pred_df.merge(
        cand_lookup,
        left_on=["constituency_number", "party"],
        right_on=["constituency_number", "actual_alliance"],
        how="left",
    )

    # Fill missing values for constituencies where no candidate was listed
    # (should be rare – only if a minor independent party appears in predictions).
    merged["candidate_name"] = merged["candidate_name"].fillna("Unknown")
    merged["actual_party"] = merged["actual_party"].fillna(merged["party"])
    merged["actual_alliance"] = merged["actual_alliance"].fillna(merged["party"])

    return merged


def train_and_predict_all_constituencies(
    base_dir: Path,
    cache_dir: Path,
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """
    Builds an enriched prediction table:
      constituency_number, rank, party, predicted_vote_share,
      candidate_name, actual_party, actual_alliance
    """
    if config is None:
        config = ModelConfig()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"pred_top3_{config.target_year}.pkl"

    if cache_path.exists():
        loaded = joblib.load(cache_path)
        # Accept cached result only if it has the enriched columns.
        if (
            isinstance(loaded, pd.DataFrame)
            and not loaded.empty
            and "candidate_name" in loaded.columns
        ):
            return loaded
        # Otherwise regenerate (old cache without candidate data).
        cache_path.unlink(missing_ok=True)

    all_df = load_all_elections(base_dir=base_dir, years=(2011, 2016, 2021))
    pred_df = predict_top3_parties_2026(
        all_elections_df=all_df,
        target_year=config.target_year,
        by_election_years=config.by_election_years,
        lambda_recency=config.lambda_recency,
    )

    # Enrich with official 2026 candidate data.
    pred_df = _enrich_with_2026_candidates(pred_df, base_dir)

    joblib.dump(pred_df, cache_path)

    csv_path = cache_dir / f"kerala_{config.target_year}_top3_predictions.csv"
    try:
        pred_df.to_csv(csv_path, index=False)
    except Exception:
        pass

    return pred_df


def load_constituencies(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "data_2021" / "processed" / "constituencies.csv"
    return pd.read_csv(path)
