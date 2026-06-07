from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from ml_app.alliance_mapping import format_alliance_tag_for_year
from ml_app.constituency_overrides import get_override_rows


@dataclass(frozen=True)
class PredictionRow:
    constituency_number: int
    party: str
    predicted_vote_share: float


def _weighted_linear_predict(x: np.ndarray, y: np.ndarray, w: np.ndarray, x_pred: float) -> float:
    """
    Weighted linear regression (degree 1).
    If only one point, returns y.
    """
    if len(x) == 1:
        return float(y[0])

    # Fit y = a + b*x using weighted least squares.
    w_sum = float(np.sum(w))
    if w_sum <= 0:
        return float(np.mean(y))

    x_bar = float(np.sum(w * x) / w_sum)
    y_bar = float(np.sum(w * y) / w_sum)
    cov_xy = float(np.sum(w * (x - x_bar) * (y - y_bar)))
    var_x = float(np.sum(w * (x - x_bar) ** 2))

    if var_x <= 1e-12:
        return float(y_bar)
    b = cov_xy / var_x
    a = y_bar - b * x_bar
    return float(a + b * x_pred)


def _event_weight(year: int, by_years: Iterable[int]) -> float:
    """
    Higher weight for general elections, lower for by-elections.
    """
    by_years_set = set(by_years)
    if year in by_years_set:
        return 0.35
    return 1.0


def _compute_party_vote_shares(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate candidate-level rows into party-level vote share per
    (year, constituency_number).
    """
    df = candidate_df.copy()

    def get_entity(row: pd.Series) -> str:
        party_name = str(row["party"]).strip()
        c_num = row.get("constituency_number")
        yr = row.get("year")
        party_upper = party_name.upper()

        # --- Historical Overrides ---
        if c_num == 24:  # Perambra
            # In 2011, 2016 KEC(M) fought for UDF (currently mapped to LDF globally)
            if yr in {2011, 2016} and party_upper in {"KEC(M)", "KEC (M)", "KERALA CONGRESS (M)"}:
                return "UDF"
            # In 2021, the UDF candidate was an Independent (C.H. Ibrahimkutty)
            if yr == 2021 and party_upper == "IND":
                # To be precise, we are making all INDs in 2021 for Perambra count 
                # towards UDF. The minor INDs have negligible impact, while the 
                # main UDF IND's large vote share is correctly captured.
                return "UDF"

        if c_num == 59:  # Nenmara
            cand = str(row.get("candidate", "")).upper()
            if yr == 2011 and party_upper == "CMP":
                return "UDF"
            if yr == 2021 and party_upper == "CMPKSC":
                return "UDF"

        if c_num == 26:  # Elathur
            # In 2021, the UDF candidate was an Independent (Zulphikar mayoori)
            if yr == 2021 and party_upper == "IND":
                return "UDF"

        if c_num == 30:  # Kunnamangalam
            cand = str(row.get("candidate", "")).upper()
            if "RAHIM" in cand:  # ADV. P.T.A. RAHIM (IND -> LDF in 2011, 2016, 2021)
                return "LDF"
            if yr == 2011 and "RAMAN" in cand:
                return "UDF"
            if yr == 2021 and "DINESH" in cand:  # DINESH PERUMANNA (IND -> UDF)
                return "UDF"
                
        if c_num == 38:  # Perinthalmanna
            cand = str(row.get("candidate", "")).upper()
            # 2021: LDF-backed IND candidate
            if yr == 2021 and "MUSTHAFA" in cand and party_upper == "IND":
                return "LDF"

        if c_num == 93:  # Pala
            # KEC(M) was UDF in 2011 and 2016, moved to LDF in 2021
            if yr in {2011, 2016} and party_upper in {"KEC(M)", "KEC (M)", "KERALA CONGRESS (M)"}:
                return "UDF"
            # In 2021, Mani C. Kappan fought under UDF (as NCP/IND depending on the dataset)
            if yr == 2021 and party_upper in {"NCP", "IND"} and "KAPP" in str(row.get("candidate", "")).upper():
                return "UDF"

        if c_num == 84:  # Kunnathunad
            cand = str(row.get("candidate", "")).upper()
            if yr in {2011, 2016, 2021} and "SAJEENDRAN" in cand:
                return "UDF"
            if yr in {2011, 2016} and "SURENDRAN" in cand and "M.A" in cand:
                return "LDF"
            if yr in {2011, 2016, 2021} and "P.V" in cand and "SREENIJIN" in cand:
                return "LDF"
            if yr == 2011 and "M.RAVI" in cand:
                return "NDA"
            if yr == 2016 and "THURAVOOR" in cand:
                return "NDA"
            if yr == 2021 and "JITHIN DEV" in cand:
                return "NDA"

        if c_num == 85:  # Piravom
            cand = str(row.get("candidate", "")).upper()
            if yr == 2021 and party_upper in {"KEC(M)", "KEC (M)", "KERALA CONGRESS (M)"}:
                return "LDF"
            if party_upper in {"KC(J)", "KEC(J)"}:
                return "UDF"
            if yr in {2011, 2016} and "T M JACOB" in cand:
                return "UDF"
            if yr == 2016 and "ANOOP JACOB" in cand:
                return "UDF"

        if c_num == 99:  # Changanassery
            # KEC(M) was UDF in 2011 (in 2016 the user manually changed it to UDF in the dataset)
            if yr == 2011 and party_upper in {"KEC(M)", "KEC (M)", "KERALA CONGRESS (M)"}:
                return "UDF"

        if c_num == 120:  # Pathanapuram
            cand = str(row.get("candidate", "")).upper()
            if yr in {2011, 2016} and "GANESH KUMAR" in cand:
                return "UDF"
            if yr == 2021 and party_upper in {"KEC(B)", "KC(B)", "KCB"}:
                return "LDF"
            if yr == 2021 and "JITHIN DEV" in cand:
                return "NDA"

        if c_num == 82:  # Ernakulam
            # CPI(M) candidates should be LDF for all years
            if party_upper == "CPI(M)" or party_upper == "CPI[M]" or party_upper == "CPI [M]":
                return "LDF"
            # INC candidates should be UDF for all years
            if party_upper == "INC":
                return "UDF"

        if c_num == 117:  # Chavara
            cand = str(row.get("candidate", "")).upper()
            if yr == 2011:
                if "SHIBU" in cand:  # SHIBU BABY JOHN (OTH) was UDF
                    return "UDF"
                if "PREMACHANDRAN" in cand:  # N.K. PREMACHANDRAN (UNKNOWN) was LDF
                    return "LDF"
            if yr == 2016:
                if "VIJAYAN" in cand and party_upper == "CMP":  # N. VIJAYAN PILLAI (LDF)
                    return "LDF"
            if yr == 2021:
                if "SUJITH" in cand and party_upper == "IND":  # Dr. SUJITH VIJAYANPILLAI (LDF)
                    return "LDF"

        tag = format_alliance_tag_for_year(party_name, yr)
        if tag in {"LDF", "UDF", "NDA"}:
            return tag
        # Collapse everything else into OTHER so raw placeholders like
        # UNKNOWN/IND/OTH do not become synthetic parties in the model.
        return "OTHER"

    df["party"] = df.apply(get_entity, axis=1)

    group_cols = ["year", "constituency_number", "party"]
    votes_agg = df.groupby(group_cols, as_index=False).agg(votes=("votes", "sum"))

    # Compute total valid votes per constituency/year (includes NOTA in dataset).
    totals = (
        candidate_df.groupby(["year", "constituency_number"], as_index=False)
        .agg(total_votes=("votes", "sum"))
        .rename(columns={"total_votes": "votes_total"})
    )

    merged = votes_agg.merge(totals, on=["year", "constituency_number"], how="left")
    merged["vote_share"] = np.where(
        merged["votes_total"] > 0,
        (merged["votes"] / merged["votes_total"]) * 100.0,
        0.0,
    )
    return merged.drop(columns=["votes_total"])


def predict_top3_parties_2026(
    all_elections_df: pd.DataFrame,
    target_year: int = 2026,
    by_election_years: Iterable[int] = (2022, 2023, 2024, 2025),
    lambda_recency: float = 0.55,
) -> pd.DataFrame:
    """
    For each constituency, predicts party vote shares for target_year
    and returns top-3 parties (excluding NOTA).

    Model:
      - For each (constituency, party), do weighted linear regression of
        observed vote_share vs year (weights based on recency and whether
        it is by-election).
      - Do the same globally per party to get a prior.
      - Shrink local prediction toward global prediction depending on
        the number of observations for that constituency-party.
      - Predict NOTA too, then renormalize party shares to (100 - NOTA_pred).
    """
    df = all_elections_df.copy()
    df["year"] = df["year"].astype(int)
    df["constituency_number"] = df["constituency_number"].astype(int)
    df["party"] = df["party"].astype(str)

    by_years = set(by_election_years)
    party_df = _compute_party_vote_shares(df)

    # Parties we will predict (exclude NOTA for top3, but NOTA used for renormalization).
    all_parties = sorted(party_df["party"].unique().tolist())

    # Observations for each party in each constituency.
    consts = sorted(party_df["constituency_number"].unique().tolist())

    global_pred_by_party: Dict[str, float] = {}
    for party in all_parties:
        sub = party_df[party_df["party"] == party]
        if sub.empty:
            continue
        x = sub["year"].to_numpy(dtype=float)
        y = sub["vote_share"].to_numpy(dtype=float)
        max_year = float(sub["year"].max())
        w = np.array([_event_weight(int(yr), by_years) * np.exp(-lambda_recency * (max_year - float(yr))) for yr in x], dtype=float)
        global_pred_by_party[party] = _weighted_linear_predict(x, y, w, float(target_year))

    rows: List[PredictionRow] = []

    for c in consts:
        sub_c = party_df[party_df["constituency_number"] == c]
        parties_in_const = sorted(sub_c["party"].unique().tolist())
        if not parties_in_const:
            continue

        # Predict NOTA for renormalization if present historically.
        nota_pred = None
        if (sub_c["party"] == "NOTA").any():
            sub_nota = sub_c[sub_c["party"] == "NOTA"]
            x = sub_nota["year"].to_numpy(dtype=float)
            y = sub_nota["vote_share"].to_numpy(dtype=float)
            max_year = float(sub_nota["year"].max())
            w = np.array(
                [
                    _event_weight(int(yr), by_years) * np.exp(-lambda_recency * (max_year - float(yr)))
                    for yr in x
                ],
                dtype=float,
            )
            nota_pred = _weighted_linear_predict(x, y, w, float(target_year))
        else:
            nota_pred = global_pred_by_party.get("NOTA", 0.0)

        nota_pred = float(np.clip(nota_pred, 0.0, 100.0))
        party_share_preds: Dict[str, float] = {}

        for party in parties_in_const:
            if party == "NOTA":
                continue
            sub_cp = sub_c[sub_c["party"] == party]
            if sub_cp.empty:
                continue

            x_local = sub_cp["year"].to_numpy(dtype=float)
            y_local = sub_cp["vote_share"].to_numpy(dtype=float)
            n_obs = len(x_local)
            max_year = float(sub_cp["year"].max())
            w_local = np.array(
                [
                    _event_weight(int(yr), by_years) * np.exp(-lambda_recency * (max_year - float(yr)))
                    for yr in x_local
                ],
                dtype=float,
            )
            local_pred = _weighted_linear_predict(x_local, y_local, w_local, float(target_year))

            global_pred = float(global_pred_by_party.get(party, np.mean(y_local)))

            # Shrink: more observations => trust local more.
            # With 1 obs, shrink heavily to global.
            gamma = 1.0 - min(1.0, n_obs / 3.0)  # 0 when n>=3
            pred = (1.0 - gamma) * local_pred + gamma * global_pred
            party_share_preds[party] = float(np.clip(pred, 0.0, 100.0))

        sum_pred_parties = float(sum(party_share_preds.values()))
        valid_share_target = max(0.0, 100.0 - nota_pred)
        if sum_pred_parties <= 1e-9:
            # Fallback: equally distribute valid_share_target.
            uniform = valid_share_target / max(1, len(party_share_preds))
            for party in party_share_preds:
                party_share_preds[party] = float(uniform)
        else:
            for party in list(party_share_preds.keys()):
                party_share_preds[party] = (party_share_preds[party] / sum_pred_parties) * valid_share_target

        for override in get_override_rows("predictor_party_adjustments"):
            if int(override.get("constituency_number", -1)) != c:
                continue
            party_deltas = override.get("party_deltas", {})
            if not isinstance(party_deltas, dict):
                continue
            for party, delta in party_deltas.items():
                if party not in party_share_preds:
                    continue
                party_share_preds[party] = max(0.0, float(party_share_preds[party]) + float(delta))

            adjusted_total = float(sum(party_share_preds.values()))
            if adjusted_total > 1e-9:
                for party in list(party_share_preds.keys()):
                    party_share_preds[party] = (party_share_preds[party] / adjusted_total) * valid_share_target

        for party, share in party_share_preds.items():
            rows.append(
                PredictionRow(
                    constituency_number=c,
                    party=party,
                    predicted_vote_share=float(share),
                )
            )

    pred_df = pd.DataFrame([r.__dict__ for r in rows])
    pred_df = pred_df.sort_values(["constituency_number", "predicted_vote_share"], ascending=[True, False])

    # Top-3 per constituency (excluding NOTA by construction).
    top3 = pred_df.groupby("constituency_number").head(3).copy()
    top3["rank"] = top3.groupby("constituency_number").cumcount() + 1
    top3["target_year"] = target_year
    return top3.reset_index(drop=True)


def _ensemble_predict(x: np.ndarray, y: np.ndarray, x_pred: float) -> float:
    """
    Ensemble prediction using Random Forest and Gradient Boosting.
    """
    if len(x) < 2:
        return float(np.mean(y))

    # Prepare data for training
    X = x.reshape(-1, 1)
    y = y

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_pred = rf_model.predict([[x_pred]])[0]

    # Train Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X, y)
    gb_pred = gb_model.predict([[x_pred]])[0]

    # Combine predictions (weighted average)
    ensemble_pred = 0.5 * rf_pred + 0.5 * gb_pred
    return ensemble_pred




