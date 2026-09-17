# Official player rating experiments

## Fixed protocol

The tests use market-free margin MAE. Lower values are better. The development years are 2015 to 2022. The validation years are 2023 and 2024. The 2025 season is the final target set and has 216 matches.

Build the source data once:

```sh
uv run python -m src.scrape_afl.wheelo_player_history \
  --first-year 2012 --last-year 2025 \
  --output-dir .context/wheelo-player-history
```

The tested source CSV had this SHA-256 value:

```text
cab47f144922f8f58f39dfc7fd0d7f1792bf6e8bfb4e1d2f6d2101da5e293393
```

Each test used this command shape:

```sh
uv run python -m src.mae_model.run_backtest \
  --player-stats-csv .context/wheelo-player-history/afl_player_ratings.csv \
  --player-control team_only \
  --player-measurement MEASUREMENT \
  --player-signal SIGNAL \
  --player-rating-prior-games PRIOR_GAMES \
  --output-dir OUTPUT_DIR
```

## Fixed initial tests

| Test | Measurement | Signal | Prior games | Development MAE | Validation MAE |
| --- | --- | --- | ---: | ---: | ---: |
| H1 | `official_points` | `rating` | 6 | 27.665443 | 26.250822 |
| H2 | `official_points` | `rating_form` | 6 | 27.666619 | 26.245054 |
| H3 | `official_points_per_time` | `rating_form` | 6 | 27.662897 | 26.240520 |
| H4 | `official_points` | `rating_form` | 12 | 27.667784 | 26.236765 |
| H5 | `official_points` | `missing_leader` | 6 | 27.662805 | 26.245302 |

The control MAE was 27.729524 in development and 26.268117 in validation. H4 had the best validation result. Its 2025 MAE was 26.338423. The control 2025 MAE was 26.348055.

## Later rejected tests

Four larger changes were tested after H4:

| Test | Change | Result |
| --- | --- | --- |
| H6 | Increase the lineup change correction | 2024 MAE was 0.287592 worse. |
| H7 | Use the absolute selected lineup rating level | 2023 MAE was 0.080623 worse and 2024 MAE was 0.146671 worse. |
| H8 | Use recent lineup averages as the reference | It was weaker than H4 in development and validation. |
| H9 | Add team form from prior aggregate Rating Points | 2023 MAE was 0.139583 worse and 2024 MAE was 0.317283 worse. |

These four code variants were reverted. The current branch cannot run them without their experiment patches.

## Limits

The source does not give a publication time for Rating Points. The importer assumes that a rating was available when the repository result became available. The source player rows also identify the final team that played. The replay assumes that this team was known at kickoff.

The H4 correction improved 87 of the 216 target matches, made 97 worse, and did not change 32. A paired bootstrap with 100,000 samples and seed `20260916` gave a 95% interval from -0.0751 to +0.0956 MAE for the improvement. This interval includes zero.
