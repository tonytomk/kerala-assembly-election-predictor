#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ml_app.data_loader import load_all_elections
from ml_app.predictor import _compute_party_vote_shares

base_dir = Path(__file__).parent

print("Testing Ernakulam 82 override...\n")

# Load election data
all_df = load_all_elections(base_dir, years=(2011, 2016, 2021))

# Filter for Ernakulam 82
eranakulam_raw = all_df[all_df['constituency_number'] == 82]
print(f"Raw data for Ernakulam 82 (2021 only):")
print(eranakulam_raw[eranakulam_raw['year'] == 2021][['year', 'candidate', 'party', 'votes']])

# Compute aggregated vote shares
party_df = _compute_party_vote_shares(all_df)

# Check Ernakulam 82 aggregated
eranakulam_agg = party_df[party_df['constituency_number'] == 82]
print("\n\nAggregated Eranakulam 82 (all years):")
print(eranakulam_agg.sort_values(['year', 'vote_share'], ascending=[True, False]))

# Check specifically for LDF
ldf_rows = eranakulam_agg[eranakulam_agg['party'] == 'LDF']
print(f"\n\nLDF rows for Eranakulam 82: {len(ldf_rows)} rows")
if len(ldf_rows) > 0:
    print(ldf_rows)
else:
    print("ERROR: No LDF rows found for Eranakulam 82!")
    print("\nAll parties in aggregated data:")
    print(eranakulam_agg['party'].unique())
