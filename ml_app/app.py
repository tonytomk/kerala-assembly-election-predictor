from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import sys

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
# Ensure `ml_app` is importable regardless of Streamlit working directory.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_app.alliance_mapping import format_alliance_tag
from ml_app.model_pipeline import ModelConfig, default_cache_dir, load_constituencies, train_and_predict_all_constituencies
from ml_app.data_loader import load_all_elections
from ml_app.predictor import _compute_party_vote_shares
from ml_app.ollama_client import generate_explanation_ollama
from ml_app.model_pipeline import evaluate_model_performance


st.set_page_config(page_title="Kerala Election Analytics", layout="wide")
st.title("Kerala Election Analytics (2011–2021 + by-polls) → 2026 Prediction")


@st.cache_data(show_spinner=False)
def load_history_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_df = load_all_elections(base_dir=BASE_DIR, years=(2011, 2016, 2021))
    party_df = _compute_party_vote_shares(all_df)
    const_df = load_constituencies(base_dir=BASE_DIR)
    return party_df, const_df


@st.cache_data(show_spinner=False)
def get_predictions_df() -> pd.DataFrame:
    cache_dir = default_cache_dir(BASE_DIR)
    pred_df = train_and_predict_all_constituencies(
        base_dir=BASE_DIR,
        cache_dir=cache_dir,
        config=ModelConfig(),
    )
    return pred_df


party_df, const_df = load_history_tables()
pred_df = get_predictions_df()

# ---------------------------------------------------------------------------
# Helper: resolve the display alliance from actual_alliance (2026 CSV) when
# available, falling back to the heuristic tag.
# Only trust actual_alliance if it is a known top-level alliance label;
# otherwise the unmatched rows would leak raw party names (IND, UNKNOWN…).
# ---------------------------------------------------------------------------

KNOWN_ALLIANCES = {"LDF", "UDF", "NDA", "AAP"}

def resolve_alliance(row: pd.Series) -> str:
    actual = str(row.get("actual_alliance", "")).strip()
    if actual in KNOWN_ALLIANCES:
        return actual
    # Fall back to heuristic for minor/historical parties not in the 2026 list.
    tag = format_alliance_tag(str(row["party"]))
    return tag if tag in KNOWN_ALLIANCES else "OTHER"


pred_df["Alliance"] = pred_df.apply(resolve_alliance, axis=1)

# ---------------------------------------------------------------------------
# Summary — Projected Seat Counts
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Summary of 2026 Predictions (Projected Seat Counts)")

winners = pred_df[pred_df["rank"] == 1].copy()

# Alliance Summary
alliance_counts = winners["Alliance"].value_counts().reset_index()
alliance_counts.columns = ["Alliance", "Seats"]

# Party Summary (using actual_party when available)
winners["display_party"] = winners["actual_party"].fillna(winners["party"]) if "actual_party" in winners.columns else winners["party"]
party_winners = winners.groupby(["display_party", "Alliance"]).size().reset_index(name="Seats")
party_winners = party_winners.sort_values("Seats", ascending=False)

# Top-3 appearances — use actual_party across full pred_df
top3_all = pred_df.copy()
top3_all["display_party"] = top3_all["actual_party"].fillna(top3_all["party"]) if "actual_party" in top3_all.columns else top3_all["party"]
party_top3 = top3_all.groupby(["display_party", "Alliance"]).size().reset_index(name="Top 3 Count")
party_top3 = party_top3.sort_values("Top 3 Count", ascending=False)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Projected Winners by Alliance**")
    # Always show the four main alliances so totals are visible (even if 0).
    alliance_order = ["LDF", "UDF", "NDA", "AAP", "OTHER"]
    for alliance in alliance_order:
        count = int(alliance_counts[alliance_counts["Alliance"] == alliance]["Seats"].sum())
        st.metric(label=alliance, value=count)

with col2:
    st.markdown("**Alliance Comparison (Wins vs Top-3)**")
    alliance_top3 = pred_df["Alliance"].value_counts().reset_index()
    alliance_top3.columns = ["Alliance", "Top 3 Appearances"]
    comparison_df = alliance_counts.merge(alliance_top3, on="Alliance", how="outer").fillna(0)
    fig_summary = px.bar(
        comparison_df,
        x="Alliance",
        y=["Seats", "Top 3 Appearances"],
        barmode="group",
        title="Projected Seats (Wins) vs Top-3 Appearances",
        color_discrete_sequence=["#e63946", "#457b9d"],
    )
    st.plotly_chart(fig_summary, use_container_width=True)

with st.expander("Show detailed party-wise projections"):
    party_combined = party_winners.merge(
        party_top3,
        on=["display_party", "Alliance"],
        how="outer",
    ).fillna(0)
    party_combined = party_combined.sort_values(["Seats", "Top 3 Count"], ascending=False)
    st.table(
        party_combined.rename(columns={
            "display_party": "Party",
            "Seats": "Seats won",
            "Top 3 Count": "Top 3 finishes",
        })
    )

with st.expander("Show all constituency predictions (2026 top-3)"):
    table = pred_df.copy()
    # Use actual party/candidate when available
    if "candidate_name" in table.columns:
        table["Candidate"] = table["candidate_name"]
    if "actual_party" in table.columns:
        table["Party"] = table["actual_party"].fillna(table["party"])
    else:
        table["Party"] = table["party"]
    table = table.rename(columns={
        "predicted_vote_share": "Predicted Vote Share (%)",
        "target_year": "Year",
        "rank": "Rank",
        "constituency_number": "Constituency No.",
    })
    display_cols = ["Constituency No.", "Rank", "Candidate", "Party", "Alliance", "Predicted Vote Share (%)"]
    display_cols = [c for c in display_cols if c in table.columns]

    show_full = st.checkbox("Display full table", value=False)
    if show_full:
        st.dataframe(table[display_cols], use_container_width=True, height=700)
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="kerala_2026_top3_predictions.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Constituency detail view
# ---------------------------------------------------------------------------

left, right = st.columns([1, 1])
with left:
    const_options = const_df.sort_values("constituency_number")[["constituency_number", "constituency"]].copy()
    const_options["label"] = const_options.apply(
        lambda row: f"{int(row['constituency_number'])} - {str(row['constituency'])}",
        axis=1,
    )
    constituency_label = st.selectbox(
        "Select constituency",
        const_options["label"].tolist(),
    )
    constituency_number = int(const_options.loc[const_options["label"] == constituency_label, "constituency_number"].iloc[0])

with right:
    enable_ollama = st.checkbox("Generate explanation using Ollama", value=False)
    ollama_model = st.text_input("Ollama model", value="llama3")


const_row = const_df[const_df["constituency_number"] == constituency_number].iloc[0]
constituency_name = str(const_row["constituency"])
district = str(const_row.get("district", ""))

pred_sub = pred_df[pred_df["constituency_number"] == constituency_number].copy()
pred_sub = pred_sub.sort_values("rank")

# Build display labels: "Candidate Name (Party)"
def build_label(row: pd.Series) -> str:
    cand = str(row.get("candidate_name", "")).strip()
    party = str(row.get("actual_party", row.get("party", ""))).strip()
    if cand and cand not in {"Unknown", "nan", ""}:
        return f"{cand} ({party})"
    return party

pred_sub["Label"] = pred_sub.apply(build_label, axis=1)

st.markdown("---")
st.subheader(f"{constituency_number} – {constituency_name} ({district})")

# Candidate info table
if "candidate_name" in pred_sub.columns:
    info_table = pred_sub[["rank", "candidate_name", "actual_party", "Alliance", "predicted_vote_share"]].copy()
    info_table.columns = ["Rank", "Candidate", "Party", "Alliance", "Predicted Vote Share (%)"]
    info_table["Predicted Vote Share (%)"] = info_table["Predicted Vote Share (%)"].round(2)
    st.table(info_table.set_index("Rank"))

# Bar chart with candidate labels
chart_df = pred_sub.rename(columns={"predicted_vote_share": "Predicted Vote Share (%)"})[
    ["rank", "Label", "Predicted Vote Share (%)", "Alliance"]
]
ALLIANCE_COLORS = {"LDF": "#e63946", "UDF": "#2a9d8f", "NDA": "#f4a261", "AAP": "#264653", "OTHER": "#adb5bd"}
color_sequence = [ALLIANCE_COLORS.get(a, "#adb5bd") for a in chart_df["Alliance"].tolist()]

fig = px.bar(
    chart_df,
    x="Label",
    y="Predicted Vote Share (%)",
    color="Alliance",
    color_discrete_map=ALLIANCE_COLORS,
    title=f"Predicted top-3 party vote shares for 2026 — {constituency_name}",
    text="Predicted Vote Share (%)",
)
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(
    xaxis_title="Candidate (Party)",
    yaxis_title="Vote Share (%)",
    xaxis_tickangle=-10,
    showlegend=True,
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)


st.markdown("### Historical vote shares (top parties) + by-poll context")
hist_years = sorted(party_df[party_df["constituency_number"] == constituency_number]["year"].unique().tolist())
sub_c = party_df[party_df["constituency_number"] == constituency_number].copy()

# Exclude NOTA from charts.
sub_c = sub_c[sub_c["party"] != "NOTA"].copy()

# Keep only top parties per year.
top_parties = (
    sub_c.sort_values(["year", "vote_share"], ascending=[True, False])
    .groupby("year")
    .head(6)
)

hist_fig = px.line(
    top_parties,
    x="year",
    y="vote_share",
    color="party",
    markers=True,
    title="Party vote share trend across elections (includes by-elections)",
)
hist_fig.update_layout(xaxis_title="Election year", yaxis_title="Vote share (%)")
st.plotly_chart(hist_fig, use_container_width=True)


def build_ollama_prompt() -> str:
    past = (
        party_df[(party_df["constituency_number"] == constituency_number) & (party_df["party"] != "NOTA")]
        .copy()
    )
    top_last = (
        past.sort_values(["year", "vote_share"], ascending=[True, False])
        .groupby("year")
        .head(3)
    )
    last_year = int(past["year"].max())

    lines: List[str] = []
    lines.append(f"Constituency: {constituency_number} – {constituency_name} ({district})")
    lines.append("Past top-3 vote shares by year (excluding NOTA):")
    for y in sorted(top_last["year"].unique().tolist()):
        tmp = top_last[top_last["year"] == y].sort_values("vote_share", ascending=False).head(3)
        for _, r in tmp.iterrows():
            lines.append(f"  - {y}: {r['party']} = {r['vote_share']:.2f}%")

    lines.append("")
    lines.append("Official 2026 candidates and model prediction (top-3):")
    for _, r in pred_sub.iterrows():
        cand = str(r.get("candidate_name", "")).strip()
        actual_p = str(r.get("actual_party", r["party"])).strip()
        alliance = str(r.get("actual_alliance", r["Alliance"])).strip()
        share = float(r["predicted_vote_share"])
        label = f"{cand} ({actual_p}/{alliance})" if cand and cand not in {"Unknown", "nan"} else f"{actual_p} ({alliance})"
        lines.append(f"  - {label} = {share:.2f}%")

    lines.append("")
    lines.append(
        "Write a short, structured explanation (6-10 bullet points) of why these candidates are "
        "predicted to do well, based on the historical vote-share trends and by-election context. "
        "Also mention uncertainties and that these are not official forecasts."
    )
    return "\n".join(lines)


if enable_ollama:
    st.markdown("### Ollama explanation")
    prompt = build_ollama_prompt()
    if st.button("Generate explanation"):
        try:
            response = generate_explanation_ollama(
                prompt=prompt,
                model=ollama_model,
            )
            if response:
                st.write(response)
            else:
                st.warning("Ollama returned empty response.")
        except Exception as e:
            st.error(f"Ollama call failed: {e}")


st.markdown("---")
with st.container(border=True):
    st.markdown("### About This Project")
    st.markdown(
        "Project repo: "
        "[kerala-assembly-election-predictor]"
        "(https://github.com/tonytomk/kerala-assembly-election-predictor)"
    )
    st.markdown(
        "This app is an estimate of the 2026 Kerala Assembly election based on past election results, "
        "recent by-elections, official 2026 candidate lists, and 2025 local-body trends."
    )
    st.markdown(
        "In simple terms, the model looks at how each front or party performed in earlier elections in the "
        "same constituency, gives more importance to more recent results, and then blends that with current "
        "local-body strength to produce a likely vote-share range."
    )
    st.markdown(
        "It also applies a few constituency-level corrections where Kerala politics is highly candidate-driven "
        "or where alliance shifts make the raw history misleading. These numbers are directional projections, "
        "not official forecasts."
    )


def display_metrics(true_values: list, predicted_values: list):
    """
    Display model performance metrics in the Streamlit UI.
    """
    metrics = evaluate_model_performance(np.array(true_values), np.array(predicted_values))
    st.subheader("Model Performance Metrics")
    st.write(metrics)

# Example integration in Streamlit app
# Assuming true_values and predicted_values are available after predictions
# true_values = [...]  # Replace with actual data
# predicted_values = [...]  # Replace with actual data
# display_metrics(true_values, predicted_values)

# Display model performance metrics
st.markdown("---")
st.subheader("Model Performance Metrics")

# Ensure proper column names for merging
if 'vote_share' in party_df.columns:
    party_df = party_df.rename(columns={'vote_share': 'vote_share_true'})
if 'predicted_vote_share' in pred_df.columns:
    pred_df = pred_df.rename(columns={'predicted_vote_share': 'vote_share_pred'})

# Merge DataFrames
aligned_df = pred_df.merge(
    party_df,
    on=['constituency_number', 'party'],
    how='inner'
)

# Extract true and predicted values
true_values = aligned_df['vote_share_true'].values
predicted_values = aligned_df['vote_share_pred'].values

# Display metrics in the Streamlit UI
display_metrics(true_values, predicted_values)
