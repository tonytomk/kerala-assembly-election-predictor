"""Fetch first 10 ACs to validate ECI pacing."""
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_2026_actual_results import ECI2026Fetcher, RAW_ROOT

fetcher = ECI2026Fetcher(Path(__file__).resolve().parents[1], cache_html=True)
for ac in range(1, 11):
    rows = fetcher.fetch_top3_for_constituency(ac)
    print(ac, [(r["rank"], r["candidate"][:20], r["votes"]) for r in rows])
    time.sleep(5)

print("cached", len(list(RAW_ROOT.glob("*.html"))))
