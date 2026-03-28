# Kerala Assembly Election Data Pull

This repo now contains a small ingestion pipeline for Kerala assembly election results for `2011`, `2016`, and `2021`.

## Sources

- `2011`: Chief Electoral Officer (Kerala) `candidates/loadhtml/{lac_id}` for candidate, party, ballot votes, postal votes, and total votes.
- `2016`: Chief Electoral Officer (Kerala) `expenditureGE2016/loadhtml/{lac_id}` for candidate-party mapping, plus official Form 20 PDFs for final vote totals.
- `2021`: Election Commission of India constituency result pages for candidate-level vote totals and vote share, with Kerala CEO district/LAC mapping used for constituency metadata.

## Output files

- `data/processed/kerala_assembly_candidate_results.csv`
- `data/processed/state_summary.csv`
- `data/processed/constituencies.csv`
- `data/processed/constituency_results/<year>/<ac>_<slug>.csv`

The main candidate dataset includes:

`year,constituency,candidate,party,votes,vote_share,is_winner`

It also keeps a few useful extras:

- `district`
- `constituency_id`
- `constituency_number`
- `source_url`

## Run

```bash
python -m pip install -r requirements.txt
python scripts/fetch_kerala_assembly_results.py
```

You can also fetch only selected years:

```bash
python scripts/fetch_kerala_assembly_results.py --years 2011 2016
```

## ML / Analytics App (Streamlit + Ollama for explanations)
This app predicts 2026 top-3 parties by vote share for each constituency using your historical
election + by-election vote shares (general: 2011, 2016, 2021; by-polls: 2022-2025).

Run:
```bash
streamlit run ml_app/app.py
```

Ollama usage: Ollama is used only for text explanations (not numeric predictions).
Ensure Ollama is running locally on `http://localhost:11434`.

## License

MIT License

Copyright (c) 2026 Tony Tom K

## Suggested next additions

- Add `candidate_rank` per constituency for faster leaderboard views.
- Store `ballot_votes` and `postal_votes` separately where available.
- Add a `source_type` column such as `eci_html`, `ceo_html`, `form20_pdf`.
- Add party alliance mapping (`LDF`, `UDF`, `NDA`, `Other`) in a separate dimension table.
