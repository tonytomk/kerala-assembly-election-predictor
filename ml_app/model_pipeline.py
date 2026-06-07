from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

from ml_app.alliance_mapping import format_alliance_tag_for_year
from ml_app.constituency_overrides import get_override_rows
from ml_app.data_loader import load_all_elections, load_2026_candidates
from ml_app.predictor import _compute_party_vote_shares, predict_top3_parties_2026, _ensemble_predict


@dataclass(frozen=True)
class ModelConfig:
    target_year: int = 2026
    by_election_years: tuple[int, ...] = (2022, 2023, 2024, 2025)
    lambda_recency: float = 0.55
    local_body_weight: float = 0.50
    local_body_front_weight: float = 0.75


def default_cache_dir(base_dir: Path) -> Path:
    return base_dir / "ml_app_cache"


def _load_local_body_prior(base_dir: Path) -> pd.DataFrame:
    """
    Convert the 2025 local-body alignment file into a constituency-level prior.

    The output has columns:
        constituency_number, party, local_body_vote_share
    """
    path = base_dir / "data_localbody_2025" / "local_body_to_assembly_alignment_2025.csv"
    if not path.exists():
        return pd.DataFrame(columns=["constituency_number", "party", "local_body_vote_share"])

    raw = pd.read_csv(path)
    required = {"assembly_constituency_no", "udf_wins", "ldf_wins", "nda_wins", "oth_wins"}
    if not required.issubset(raw.columns):
        return pd.DataFrame(columns=["constituency_number", "party", "local_body_vote_share"])

    agg = (
        raw.groupby(["assembly_constituency_no", "assembly_constituency"], as_index=False)
        .agg(
            udf_wins=("udf_wins", "sum"),
            ldf_wins=("ldf_wins", "sum"),
            nda_wins=("nda_wins", "sum"),
            oth_wins=("oth_wins", "sum"),
        )
        .rename(columns={"assembly_constituency_no": "constituency_number"})
    )

    rows = []
    for _, row in agg.iterrows():
        udf_wins = float(row["udf_wins"])
        ldf_wins = float(row["ldf_wins"])
        nda_wins = float(row["nda_wins"])
        oth_wins = float(row["oth_wins"])
        if oth_wins > 0:
            dominant = max(
                [("UDF", udf_wins), ("LDF", ldf_wins), ("NDA", nda_wins)],
                key=lambda item: item[1],
            )[0]
            if dominant == "UDF":
                udf_wins += oth_wins
            elif dominant == "LDF":
                ldf_wins += oth_wins
            else:
                nda_wins += oth_wins
            oth_wins = 0.0

        total = float(udf_wins + ldf_wins + nda_wins + oth_wins)
        if total <= 0:
            continue
        for party, value in (("UDF", udf_wins), ("LDF", ldf_wins), ("NDA", nda_wins), ("OTHER", oth_wins)):
            rows.append(
                {
                    "constituency_number": int(row["constituency_number"]),
                    "party": party,
                    "local_body_vote_share": (value / total) * 100.0,
                }
            )

    prior = pd.DataFrame(rows)

    dominant_rows = []
    if not prior.empty:
        for constituency_number, sub in prior.groupby("constituency_number"):
            pivot = sub.pivot_table(
                index="constituency_number",
                columns="party",
                values="local_body_vote_share",
                aggfunc="sum",
                fill_value=0.0,
            )
            if pivot.empty:
                continue
            row = pivot.iloc[0]
            top_party = row[["UDF", "LDF", "NDA"]].idxmax()
            top_share = float(row[top_party])
            second_share = float(row[["UDF", "LDF", "NDA"]].drop(top_party).max())
            dominant_rows.append(
                {
                    "constituency_number": int(constituency_number),
                    "local_body_front": top_party,
                    "local_body_front_margin": max(0.0, top_share - second_share),
                }
            )
        if dominant_rows:
            prior = prior.merge(pd.DataFrame(dominant_rows), on="constituency_number", how="left")

    summary_md_path = base_dir / "data_localbody_2025" / "constituency_local_body_summary.md"
    if summary_md_path.exists():
        summary_rows = []
        current_constituency_number = None
        for raw_line in summary_md_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("## "):
                try:
                    current_constituency_number = int(line.split(".", 1)[0].replace("##", "").strip())
                except ValueError:
                    current_constituency_number = None
                continue
            if current_constituency_number is None or not line.startswith("|"):
                continue
            if line.startswith("| ---") or line.startswith("| Local Body"):
                continue

            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 8:
                continue

            try:
                ruling_front = cells[2].upper()
                udf_share = float(cells[4])
                ldf_share = float(cells[5])
                nda_share = float(cells[6])
                oth_share = float(cells[7])

                # For panchayat-heavy data, the OTH column usually behaves like
                # support for the ruling front rather than a separate bloc.
                if ruling_front == "UDF":
                    udf_share += oth_share
                    oth_share = 0.0
                elif ruling_front == "LDF":
                    ldf_share += oth_share
                    oth_share = 0.0
                elif ruling_front == "NDA":
                    nda_share += oth_share
                    oth_share = 0.0

                summary_rows.extend(
                    [
                        {"constituency_number": current_constituency_number, "party": "UDF", "local_body_vote_share": udf_share},
                        {"constituency_number": current_constituency_number, "party": "LDF", "local_body_vote_share": ldf_share},
                        {"constituency_number": current_constituency_number, "party": "NDA", "local_body_vote_share": nda_share},
                        {"constituency_number": current_constituency_number, "party": "OTHER", "local_body_vote_share": oth_share},
                    ]
                )
            except ValueError:
                continue

        summary_prior = pd.DataFrame(summary_rows)
        if not summary_prior.empty:
            summary_prior = (
                summary_prior.groupby(["constituency_number", "party"], as_index=False)
                .agg(local_body_vote_share=("local_body_vote_share", "sum"))
            )
            totals = summary_prior.groupby("constituency_number")["local_body_vote_share"].transform("sum")
            summary_prior["local_body_vote_share"] = summary_prior["local_body_vote_share"].where(
                totals <= 0,
                (summary_prior["local_body_vote_share"] / totals) * 100.0,
            )

            if prior.empty:
                prior = summary_prior
            else:
                merged = prior.merge(
                    summary_prior,
                    on=["constituency_number", "party"],
                    how="outer",
                    suffixes=("_align", "_md"),
                )
                merged["local_body_vote_share"] = merged["local_body_vote_share_align"].fillna(merged["local_body_vote_share_md"])
                both = merged["local_body_vote_share_align"].notna() & merged["local_body_vote_share_md"].notna()
                merged.loc[both, "local_body_vote_share"] = (
                    0.6 * merged.loc[both, "local_body_vote_share_align"]
                    + 0.4 * merged.loc[both, "local_body_vote_share_md"]
                )
                prior = merged[["constituency_number", "party", "local_body_vote_share"]].copy()

    # Use the statewide 2025 local-body summary as a fallback for constituencies
    # that do not have explicit alignment rows in the constituency-level file.
    summary_path = base_dir / "data_localbody_2025" / "local_body_front_distribution_2025.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if {"udf", "ldf", "nda", "others"}.issubset(summary.columns):
            front_totals = summary[["udf", "ldf", "nda", "others"]].sum(numeric_only=True)
            grand_total = float(front_totals.sum())
            if grand_total > 0:
                statewide_share = {
                    "UDF": float(front_totals["udf"] / grand_total) * 100.0,
                    "LDF": float(front_totals["ldf"] / grand_total) * 100.0,
                    "NDA": float(front_totals["nda"] / grand_total) * 100.0,
                    "OTHER": float(front_totals["others"] / grand_total) * 100.0,
                }

                present = set(prior["constituency_number"].astype(int).unique().tolist())
                missing = sorted(set(range(1, 141)) - present)
                if missing:
                    fallback_rows = []
                    for constituency_number in missing:
                        for party, share in statewide_share.items():
                            fallback_rows.append(
                                {
                                    "constituency_number": int(constituency_number),
                                    "party": party,
                                    "local_body_vote_share": share,
                                }
                            )
                    prior = pd.concat([prior, pd.DataFrame(fallback_rows)], ignore_index=True)

    return prior


def _enrich_with_2026_candidates(pred_df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Join official 2026 candidate data onto the prediction DataFrame.

    The predictor collapses parties into alliance-level labels (LDF/UDF/NDA or
    specific party names for minor parties).  We match each predicted row to the
    official 2026 candidate list using the alliance column from that CSV.

    New columns added:
        candidate_name  - real candidate name (or "Unknown")
        actual_party    - party abbreviation from the official list
        actual_alliance - alliance tag from the official list (LDF/UDF/NDA/AAP/...) 
    """
    try:
        cands = load_2026_candidates(base_dir)
    except FileNotFoundError:
        pred_df["candidate_name"] = "Unknown"
        pred_df["actual_party"] = pred_df["party"]
        pred_df["actual_alliance"] = pred_df["party"]
        return pred_df

    cand_lookup = cands[["constituency_number", "alliance", "party", "candidate_name"]].copy()
    cand_lookup = cand_lookup.rename(
        columns={
            "party": "actual_party",
            "alliance": "actual_alliance",
            "candidate_name": "candidate_name",
        }
    )

    merged = pred_df.merge(
        cand_lookup,
        left_on=["constituency_number", "party"],
        right_on=["constituency_number", "actual_alliance"],
        how="left",
    )

    merged["candidate_name"] = merged["candidate_name"].fillna("Unknown")
    merged["actual_party"] = merged["actual_party"].fillna(merged["party"])
    merged["actual_alliance"] = merged["actual_alliance"].fillna(merged["party"])

    for override in get_override_rows("candidate_enrichment_overrides"):
        c_num = int(override.get("constituency_number", -1))
        predicted_party = str(override.get("predicted_party", "OTHER")).strip()
        mask = (merged["constituency_number"] == c_num) & (merged["party"] == predicted_party)
        if mask.any():
            for col, value in override.items():
                if col in {"constituency_number", "predicted_party"}:
                    continue
                merged.loc[mask, col] = value

    return merged



def _normalize_person_name(name: object) -> str:
    """
    Normalize candidate names so minor formatting differences still match.

    This strips common honorifics and punctuation, which lets us match cases
    like `ADV. P. AISHA POTTY` and `P. Aisha Potty`.
    """
    text = str(name or "").upper().strip()
    if not text or text == "NAN":
        return ""
    text = re.sub(r"\b(?:ADV|DR|SMT|SRI|SHRI|KUM|MRS|MR|PROF)\.?\s*", " ", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def _apply_candidate_continuity_boost(pred_df: pd.DataFrame, all_df: pd.DataFrame, switched_alliance_boost: float = 3.0) -> pd.DataFrame:
    """
    Give a boost to 2026 candidates who were previous constituency winners.

    If the candidate is now running under a different alliance than the one they
    won with previously, we give a stronger bump because they bring personal
    vote strength plus the new alliance vote.
    """
    if pred_df.empty or all_df.empty or "candidate_name" not in pred_df.columns:
        return pred_df

    winner_df = all_df.copy()
    if "is_winner" not in winner_df.columns or "candidate" not in winner_df.columns:
        return pred_df

    winner_df = winner_df[winner_df["is_winner"].astype(bool)].copy()
    if winner_df.empty:
        return pred_df

    winner_df["winner_norm"] = winner_df["candidate"].map(_normalize_person_name)
    winner_df = winner_df[winner_df["winner_norm"] != ""]
    if winner_df.empty:
        return pred_df

    winner_df["historical_alliance"] = winner_df.apply(
        lambda row: format_alliance_tag_for_year(row["party"], int(row["year"])),
        axis=1,
    )
    winner_df["historical_alliance"] = winner_df["historical_alliance"].replace({"": "OTHER"})
    winner_df = winner_df.sort_values(["constituency_number", "year"]).drop_duplicates(
        subset=["constituency_number", "winner_norm"],
        keep="last",
    )
    if winner_df.empty:
        return pred_df

    winner_lookup = winner_df.set_index(["constituency_number", "winner_norm"])["historical_alliance"].to_dict()
    if not winner_lookup:
        return pred_df

    boosted = pred_df.copy()
    boosted["candidate_norm"] = boosted["candidate_name"].map(_normalize_person_name)

    for idx, row in boosted.iterrows():
        key = (int(row["constituency_number"]), row["candidate_norm"])
        historical_alliance = winner_lookup.get(key)
        if not historical_alliance:
            continue
        current_alliance = str(row.get("actual_alliance", row.get("party", ""))).strip().upper()
        if not current_alliance or current_alliance == "UNKNOWN":
            continue
        if current_alliance != historical_alliance:
            boosted.at[idx, "predicted_vote_share"] = float(row["predicted_vote_share"]) + float(switched_alliance_boost)

    return boosted.drop(columns=["candidate_norm"], errors="ignore")

    return boosted.drop(columns=["candidate_norm"], errors="ignore")

def _load_constituency_name_lookup(base_dir: Path) -> dict[int, str]:
    """
    Load the constituency number -> name mapping used by the dashboard.

    The dashboard keeps its source-of-truth names in `dashboard/src/data`, so we
    prefer that file and fall back to the constituency table if it is missing.
    """
    lookup: dict[int, str] = {}

    mapping_path = base_dir.parent / "dashboard" / "src" / "data" / "constituency_number_name_mapping.json"
    if mapping_path.exists():
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            raw_lookup = payload.get("constituency_number_to_name", {})
            lookup.update({int(number): str(name) for number, name in raw_lookup.items()})
        except Exception:
            lookup = {}

    if lookup:
        return lookup

    try:
        const_df = load_constituencies(base_dir)
    except Exception:
        return lookup

    if {"constituency_number", "constituency"}.issubset(const_df.columns):
        for _, row in const_df.dropna(subset=["constituency_number", "constituency"]).drop_duplicates(
            subset=["constituency_number"]
        ).iterrows():
            lookup[int(row["constituency_number"])] = str(row["constituency"]).strip()

    return lookup


def _write_dashboard_predictions_json(pred_df: pd.DataFrame, base_dir: Path) -> None:
    """
    Write the dashboard-friendly nested predictions JSON file.
    """
    dashboard_data_dir = base_dir.parent / "dashboard" / "src" / "data"
    dashboard_data_dir.mkdir(parents=True, exist_ok=True)

    name_lookup = _load_constituency_name_lookup(base_dir)
    records: list[dict[str, object]] = []

    for constituency_number, group in pred_df.sort_values(["constituency_number", "rank"]).groupby("constituency_number"):
        first_row = group.iloc[0]
        predictions = []
        for _, row in group.sort_values("rank").iterrows():
            predictions.append(
                {
                    "party": str(row["party"]),
                    "candidate_name": str(row.get("candidate_name", "")),
                    "predicted_vote_share": float(row["predicted_vote_share"]),
                    "rank": int(row["rank"]),
                }
            )

        records.append(
            {
                "constituency_number": int(constituency_number),
                "constituency_name": name_lookup.get(int(constituency_number), f"Seat {int(constituency_number)}"),
                "target_year": int(first_row["target_year"]),
                "actual_alliance": str(first_row.get("actual_alliance", "")),
                "actual_party": str(first_row.get("actual_party", "")),
                "candidate_name": str(first_row.get("candidate_name", "")),
                "predictions": predictions,
            }
        )

    output_path = dashboard_data_dir / "predictions.json"
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_dashboard_history_json(base_dir: Path) -> None:
    """
    Write the dashboard-friendly historical vote-share data for the selected seat.
    """
    dashboard_data_dir = base_dir.parent / "dashboard" / "src" / "data"
    dashboard_data_dir.mkdir(parents=True, exist_ok=True)

    name_lookup = _load_constituency_name_lookup(base_dir)
    district_lookup: dict[int, str] = {}
    try:
        const_df = load_constituencies(base_dir)
        if {"constituency_number", "district"}.issubset(const_df.columns):
            for _, row in const_df.dropna(subset=["constituency_number", "district"]).drop_duplicates(
                subset=["constituency_number"]
            ).iterrows():
                district_lookup[int(row["constituency_number"])] = str(row["district"]).strip()
    except Exception:
        const_df = pd.DataFrame()

    all_df = load_all_elections(base_dir=base_dir, years=(2011, 2016, 2021))
    party_df = _compute_party_vote_shares(all_df)
    by_election_years = {2022, 2023, 2024, 2025}

    records: list[dict[str, object]] = []
    for constituency_number in sorted(party_df["constituency_number"].astype(int).unique().tolist()):
        sub = party_df[party_df["constituency_number"] == constituency_number].copy()
        if sub.empty:
            continue

        sub = sub[sub["party"] != "NOTA"].copy()
        if sub.empty:
            continue

        top_rows = (
            sub.sort_values(["year", "vote_share"], ascending=[True, False])
            .groupby("year")
            .head(4)
            .sort_values(["year", "vote_share"], ascending=[True, False])
        )

        rows = [
            {
                "year": int(row["year"]),
                "party": str(row["party"]),
                "vote_share": float(row["vote_share"]),
                "is_by_election": int(row["year"]) in by_election_years,
            }
            for _, row in top_rows.iterrows()
        ]
        if not rows:
            continue

        records.append(
            {
                "constituency_number": int(constituency_number),
                "constituency_name": name_lookup.get(int(constituency_number), f"Seat {int(constituency_number)}"),
                "district": district_lookup.get(int(constituency_number), ""),
                "rows": rows,
            }
        )

    output_path = dashboard_data_dir / "historical_vote_shares.json"
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _compute_model_performance_metrics(all_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, float | int]:
    """
    Compute the same high-level metrics shown in the Streamlit app by aligning
    historical party vote shares with the predicted top-3 rows on
    (constituency_number, party).
    """
    if all_df.empty or pred_df.empty:
        return {
            "Mean Absolute Error (MAE)": 0.0,
            "R-squared": 0.0,
            "Root Mean Squared Error (RMSE)": 0.0,
            "aligned_rows": 0,
        }

    party_df = _compute_party_vote_shares(all_df).rename(columns={"vote_share": "vote_share_true"})
    aligned_df = pred_df.merge(
        party_df[["constituency_number", "party", "vote_share_true"]],
        on=["constituency_number", "party"],
        how="inner",
    )
    if aligned_df.empty:
        return {
            "Mean Absolute Error (MAE)": 0.0,
            "R-squared": 0.0,
            "Root Mean Squared Error (RMSE)": 0.0,
            "aligned_rows": 0,
        }

    true_values = aligned_df["vote_share_true"].to_numpy(dtype=float)
    predicted_values = aligned_df["predicted_vote_share"].to_numpy(dtype=float)
    metrics = evaluate_model_performance(true_values, predicted_values)
    metrics["aligned_rows"] = int(len(aligned_df))
    return metrics


def _write_model_performance_json(
    pred_df: pd.DataFrame,
    all_df: pd.DataFrame,
    base_dir: Path,
    cache_dir: Path,
    cache_suffix: str,
    target_year: int,
) -> None:
    """
    Persist model performance metrics both to the local cache and the dashboard
    data directory.
    """
    metrics = _compute_model_performance_metrics(all_df, pred_df)
    payload = {
        "target_year": int(target_year),
        "cache_suffix": cache_suffix,
        "metrics": metrics,
    }

    cache_output_path = cache_dir / f"model_performance_{cache_suffix}.json"
    cache_output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    dashboard_data_dir = base_dir.parent / "dashboard" / "src" / "data"
    dashboard_data_dir.mkdir(parents=True, exist_ok=True)
    dashboard_output_path = dashboard_data_dir / "model_performance.json"
    dashboard_output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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
    cache_suffix = f"lb{int(round(config.local_body_weight * 100)):02d}v27"
    cache_path = cache_dir / f"pred_top3_{config.target_year}_{cache_suffix}.pkl"
    all_df = load_all_elections(base_dir=base_dir, years=(2011, 2016, 2021))

    if cache_path.exists():
        loaded = joblib.load(cache_path)
        if (
            isinstance(loaded, pd.DataFrame)
            and not loaded.empty
            and "candidate_name" in loaded.columns
        ):
            try:
                _write_model_performance_json(
                    pred_df=loaded,
                    all_df=all_df,
                    base_dir=base_dir,
                    cache_dir=cache_dir,
                    cache_suffix=cache_suffix,
                    target_year=config.target_year,
                )
            except Exception:
                pass
            return loaded
        cache_path.unlink(missing_ok=True)

    pred_df = predict_top3_parties_2026(
        all_elections_df=all_df,
        target_year=config.target_year,
        by_election_years=config.by_election_years,
        lambda_recency=config.lambda_recency,
    )

    local_body_prior = _load_local_body_prior(base_dir)
    default_blend_weight = max(0.0, min(1.0, float(config.local_body_weight)))

    if not local_body_prior.empty:
        # Retrieve per-constituency LB weight overrides (e.g. to skip LB for Thavanur)
        weight_overrides = {
            int(o["constituency_number"]): float(o["weight"])
            for o in get_override_rows("constituency_local_body_weights")
        }

        pred_df = pred_df.merge(
            local_body_prior,
            on=["constituency_number", "party"],
            how="left",
        )
        pred_df["local_body_vote_share"] = pred_df["local_body_vote_share"].fillna(0.0)

        # Map weights to rows; default to global config if no override exists
        pred_df["effective_lb_weight"] = (
            pred_df["constituency_number"].map(weight_overrides).fillna(default_blend_weight)
        )

        # Blending step
        pred_df["predicted_vote_share"] = (
            (1.0 - pred_df["effective_lb_weight"]) * pred_df["predicted_vote_share"]
            + pred_df["effective_lb_weight"] * pred_df["local_body_vote_share"]
        )

        # Front-margin bonus (skipped if effective_lb_weight is 0)
        front_weight = max(0.0, min(1.0, float(config.local_body_front_weight)))
        if front_weight > 0 and {"local_body_front", "local_body_front_margin"}.issubset(pred_df.columns):
            pred_df["local_body_front_margin"] = pred_df["local_body_front_margin"].fillna(0.0)
            front_mask = (pred_df["party"] == pred_df["local_body_front"]) & (pred_df["effective_lb_weight"] > 0)
            pred_df.loc[front_mask, "predicted_vote_share"] = (
                pred_df.loc[front_mask, "predicted_vote_share"]
                + pred_df.loc[front_mask, "local_body_front_margin"] * front_weight
            )

        # Cleanup columns
        pred_df = pred_df.drop(
            columns=[
                "local_body_vote_share",
                "local_body_front",
                "local_body_front_margin",
                "effective_lb_weight",
            ],
            errors="ignore",
        )

        # Let especially strong local-body mandates flip the razor-thin seats.
        for override in get_override_rows("local_body_bonus_overrides"):
            c_num = int(override.get("constituency_number", -1))
            forced_party = str(override.get("party", "")).strip()
            bonus = float(override.get("bonus", 0.0))
            seat_mask = pred_df["constituency_number"].eq(c_num) & pred_df["party"].eq(forced_party)
            if seat_mask.any():
                pred_df.loc[seat_mask, "predicted_vote_share"] = (
                    pred_df.loc[seat_mask, "predicted_vote_share"] + bonus
                )
        pred_df = pred_df.drop(columns=["local_body_vote_share", "local_body_front", "local_body_front_margin"], errors="ignore")

    pred_df = _enrich_with_2026_candidates(pred_df, base_dir)
    pred_df = _apply_candidate_continuity_boost(pred_df, all_df)

    for override in get_override_rows("vote_share_floor_overrides"):
        c_num = int(override.get("constituency_number", -1))
        forced_party = str(override.get("party", "")).strip()
        target_floor = float(override.get("target_floor", 0.0))
        take_from_top_rival = bool(override.get("take_from_top_rival", False))
        seat_mask = pred_df["constituency_number"].eq(c_num)
        if not seat_mask.any():
            continue
        target_mask = seat_mask & pred_df["party"].eq(forced_party)
        if not target_mask.any():
            continue
        current_value = float(pred_df.loc[target_mask, "predicted_vote_share"].iloc[0])
        if current_value >= target_floor:
            continue
        uplift = target_floor - current_value
        if take_from_top_rival:
            rival_mask = seat_mask & pred_df["party"].ne(forced_party)
            if rival_mask.any():
                top_rival_idx = pred_df.loc[rival_mask, "predicted_vote_share"].idxmax()
                pred_df.loc[top_rival_idx, "predicted_vote_share"] = max(
                    0.0, float(pred_df.loc[top_rival_idx, "predicted_vote_share"]) - uplift
                )
        pred_df.loc[target_mask, "predicted_vote_share"] = target_floor

    # Hard local-body override for razor-thin seats where the 2025 result is
    # meant to decide the winner rather than just nudge the margin.
    for override in get_override_rows("force_winner_overrides"):
        c_num = int(override.get("constituency_number", -1))
        forced_party = str(override.get("party", "")).strip()
        margin_over_max = float(override.get("margin_over_max", 0.5))
        seat_mask = pred_df["constituency_number"].eq(c_num)
        if not seat_mask.any():
            continue
        target_mask = seat_mask & pred_df["party"].eq(forced_party)
        if not target_mask.any():
            continue
        current_max = float(pred_df.loc[seat_mask, "predicted_vote_share"].max())
        pred_df.loc[target_mask, "predicted_vote_share"] = current_max + margin_over_max

    pred_df = pred_df.sort_values(["constituency_number", "predicted_vote_share"], ascending=[True, False]).copy()
    pred_df["rank"] = pred_df.groupby("constituency_number").cumcount() + 1

    joblib.dump(pred_df, cache_path)

    csv_path = cache_dir / f"kerala_{config.target_year}_top3_predictions_{cache_suffix}.csv"
    try:
        pred_df.to_csv(csv_path, index=False)
    except Exception:
        pass

    try:
        _write_dashboard_predictions_json(pred_df, base_dir)
        _write_dashboard_history_json(base_dir)
        _write_model_performance_json(
            pred_df=pred_df,
            all_df=all_df,
            base_dir=base_dir,
            cache_dir=cache_dir,
            cache_suffix=cache_suffix,
            target_year=config.target_year,
        )
    except Exception:
        pass

    return pred_df


def load_constituencies(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "data_2021" / "processed" / "constituencies.csv"
    return pd.read_csv(path)

def evaluate_model_performance(true_values: np.ndarray, predicted_values: np.ndarray) -> dict:
    """
    Evaluate model performance using MAE, R-squared, and RMSE.
    """
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))

    return {
        "Mean Absolute Error (MAE)": mae,
        "R-squared": r2,
        "Root Mean Squared Error (RMSE)": rmse,
    }


if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).parent.parent / "main-data"
    if not base_dir.exists():
        base_dir = Path.cwd()
    cache_dir = default_cache_dir(base_dir)
    train_and_predict_all_constituencies(base_dir=base_dir, cache_dir=cache_dir)


