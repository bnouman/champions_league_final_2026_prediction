Provenance and reproduction notes
================================

This short note documents how the `data/ucl_matches_2021_22_to_2025_26.csv` snapshot was produced and provides commands you can use to reproduce or re-fetch the raw JSON feeds.

Snapshot info
-------------

- Snapshot file: `data/ucl_matches_2021_22_to_2025_26.csv`
- Snapshot date: 2026-05-15
- SHA256 (CHECKSUMS.txt): 9B9A4FBC63B1E1417632F49E98311BE9A208B7B4CE616F24129CDFDCAC8F1867

Reproducing the raw feeds
-------------------------

The original feeds referenced in the CSV are available on FixtureDownload. Example commands to fetch a season JSON (replace the year token as needed):

```powershell
curl -L "https://fixturedownload.com/feed/json/champions-league-2024" -o champions-league-2024.json
```

Quick CSV assembly (example using Python/pandas)
-----------------------------------------------

1. Download the season JSON files from FixtureDownload.
2. Convert each JSON into a normalized table (fields: date, home_team, away_team, home_goals, away_goals, source_url, etc.).
3. Concatenate season tables and export to CSV:

```python
import pandas as pd
df = pd.concat([pd.read_json('champions-league-2021.json'), pd.read_json('champions-league-2022.json')])
df.to_csv('data/ucl_matches_2021_22_to_2025_26.csv', index=False)
```

Notes
-----

- The feed provider is an aggregator; for official authoritative data and licensing please consult UEFA.
- This repository stores a snapshot for reproducible analysis; do not treat it as the canonical record for publication without verifying rights and completeness.
