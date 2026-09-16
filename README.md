# AFL Tipping

Predict AFL margins with team ratings, scoring-shot ratings, market prices, and one simple market blend.

The four control models use no player data. An optional player model adds a small correction for team-selection changes. No model uses observed weather.

## Install and test

```sh
uv sync --group dev
uv run pytest -q
```

## Run a historical comparison

The supplied history covers 2012 to 2025. It has no 2026 results. Update the match history before you use the model during a new season.

Run the internal models without market inputs:

```sh
uv run python -m src.mae_model.run_backtest --output-dir reports
```

The supplied workbook has closing lines without publication times. To use it, select the historical closing-line benchmark explicitly:

```sh
uv run python -m src.mae_model.run_backtest \
  --closing-line-benchmark \
  --market-xlsx src/outputs/afl_betting_history.xlsx \
  --output-dir reports
```

This mode measures performance with historical closing prices. It does not establish what prices were available at your tipping deadline. Do not compare it with earlier-deadline predictions without this qualification.

For a comparison with recorded price times, use a market CSV:

```sh
uv run python -m src.mae_model.run_backtest \
  --market-csv market_quotes.csv \
  --lead-hours 2 \
  --output-dir reports_timed
```

`--lead-hours 2` sets each historical deadline two hours before kickoff. The default is kickoff. The closing-line benchmark requires the default deadline.

The model uses results available by each prediction deadline. If the match history has no result publication time, it uses the next local midnight. The report records this assumption. Matches on the same local date cannot use each other's results under this rule.

## Compare player selections

To add the player model to the historical comparison, supply the player match file:

```sh
uv run python -m src.mae_model.run_backtest \
  --closing-line-benchmark \
  --market-xlsx src/outputs/afl_betting_history.xlsx \
  --player-stats-csv src/outputs/afl_player_stats.csv \
  --player-signal rating \
  --output-dir reports_players
```

The command adds one `player_lineup` row for each `market_scoring_blend` row. The four control rows stay unchanged. Player history starts in 2018. Earlier predictions equal the control margin.

Historical final selections contain player identity only. The model assumes that these selections are available at kickoff. Player backtests therefore require `--lead-hours 0`. This assumption does not establish which players were known before kickoff.

`--player-signal` accepts `rating`, `form`, `missing_leader`, or `rating_form`. The default is `rating`. Reports record the selected signal, fixed weights, fallback counts, and missing regular players. `player_diagnostics.csv` gives one diagnostic row per prediction.

## Predict future fixtures

Create a fixture CSV with these columns:

```csv
match_id,year,round,kickoff,venue,home_team_name,away_team_name
example-2026,2026,1,2026-10-01T19:30:00+10:00,M.C.G.,Sydney,Hawthorn
```

This row shows the input format. It is not an actual fixture announcement. The kickoff must include a UTC offset.

If you have market prices, create a price CSV:

```csv
match_id,observed_at,predicted_margin,home_odds,away_odds
example-2026,2026-10-01T16:00:00+10:00,4.5,1.80,2.10
```

A positive margin predicts a home-team win. A negative margin predicts an away-team win. This is the opposite sign to a home handicap. You can include several price snapshots for one match. The model selects the latest eligible snapshot. Decimal home and away odds are optional.

Run the fixture command with your prediction deadline:

```sh
uv run python -m src.mae_model.predict_fixtures \
  --matches-csv src/outputs/afl_data.csv \
  --fixtures-csv fixtures.csv \
  --market-csv market_quotes.csv \
  --as-of 2026-10-01T17:00:00+10:00 \
  --lead-hours 2 \
  --output-dir predictions
```

For live prediction, `--as-of` sets the deadline for the requested fixtures. `--lead-hours` sets the deadline used for historical blend-training examples. Keep this historical rule consistent with your competition's prediction time.

If you have no market prices, omit `--market-csv`. The blend then uses scoring-shot predictions. The market-only prediction stays empty. The output records the fallback.

The live command does not accept untimed workbook prices. It excludes prices recorded after the deadline. Fixtures must start after the deadline. Use the same deadline for all fixtures when your competition closes tips for the whole round at once.

Live prediction and historical evaluation use the same replay procedure. You do not call rating updates yourself. You do not insert zero scores for future fixtures. Season preparation occurs inside the shared procedure.

### Add dated player selections

For live player predictions, supply both `--player-stats-csv` and `--lineups-csv`. Add these options to the fixture command above. The lineup file has one row per selected player:

```csv
match_id,team_name,player_ref,player_name,observed_at
example-2026,Sydney,../../players/I/Isaac_Heeney.html,Isaac Heeney,2026-10-01T16:00:00+10:00
```

Each team snapshot must contain 22 or 23 unique player references with the same observation time. Include a complete snapshot for each team. The model selects the latest complete snapshot observed at or before `--as-of`. A later or incomplete snapshot cannot supply a forecast. Missing selections return the exact control margin.

Use the same `player_ref` values as the player match file. Player names do not identify players. New players can appear in a selection, but at least 80 percent of each selected team must have five completed player games.

## Model rules

| Output | Method |
| --- | --- |
| `team_only` | Sequential attack and defence ratings from completed scores. |
| `scoring_shots` | Sequential shot-volume ratings with historical points per shot. |
| `market_only` | The eligible market margin, without correction. |
| `market_scoring_blend` | A convex blend of the market margin and scoring-shot prediction. |
| `player_lineup` | An optional correction to the blend for material selection changes. |

The blend fits one market weight from the preceding five seasons. It selects from 0 to 1 in steps of 0.02. Weight 1 gives the market-only prediction. Equal errors favour the larger market weight. Insufficient history also selects weight 1.

The weight stays fixed during the target season. Its fitting cutoff is 1 January at 00:00 in Australia/Sydney, or the request deadline if earlier. Earlier training prices must also pass their own historical prediction deadlines.

The control models do not fit residual corrections, correction limits, or a calibration stack. The target season is excluded from weight fitting. Historical comparisons for 2024 and 2025 are not untouched tests because earlier model development used those results.

The player model compares the selection with regular players from the team's last six completed selections. A regular player appeared in at least half of those selections. The model requires at least three completed selections. It reports the highest reliably rated absent regular and the rating gap to the selected-team median.

Player impact ratings use match outcomes and each player's share of team game time. Forecasts reduce ratings with little player history. Recent form is a fast exponential mean of a fixed box-score score, less a slower career mean. The score uses kicks, handballs, marks, goals, behinds, hit-outs, tackles, clearances, contested possessions, goal assists, and clangers. It divides by at least 50 percent game time. It excludes Brownlow votes. Metadata records all weights.

A correction requires a material rating change or a material replacement gap for an absent regular. The correction limit is four points. The default rating weight is 1.0. The form weight is 0.10. These weights are fixed. The model does not fit them to the target season.

Player statistics enter after the match result becomes available. An optional `statistics_available_at` column must not precede result availability. The player update waits until every player row for that match is available. Current-match statistics cannot change that match's forecast.

Mean absolute error, or MAE, measures margin error in points. Lower values are better. Correct-tip percentage measures winner selection. A draw counts as correct only when the predicted margin is zero. These measures can rank models differently.

Eligible decimal odds can provide a market implied probability after removal of the bookmaker margin. This value is separate from the blend's margin prediction. The model does not claim a calibrated win probability for its internal or blended margins.

Reports include predictions, summaries, and metadata. Metadata records input hashes, code revision, local changes, model settings, timing rules, market coverage, and seasonal weights. Compare market and blend errors on the same market-covered matches. Do not treat a fallback as a market prediction.

The current closing-line benchmark scores 2,258 matches from 2015 to 2025:

| Model | Overall MAE | 2024 MAE | 2025 MAE |
| --- | ---: | ---: | ---: |
| `market_only` | 26.6466 | 26.8519 | 26.1944 |
| `market_scoring_blend` | 26.5694 | 26.6256 | 26.0380 |
| `scoring_shots` | 27.2709 | 26.8564 | 26.6307 |
| `team_only` | 27.3178 | 26.4834 | 26.3481 |

The blend's small historical gain does not establish a future gain. See [the full summary](reports/mae_summary.csv) and [run metadata](reports/metadata.json). The prediction CSV is generated locally and is not tracked in Git.

The old `reports_baseline` and `reports_rethink` outputs are retired. Their results describe earlier code, including an invalid result affected by outcome leakage. The old generated `afl_match_context.csv` is also retired. Git history retains those files. Current results belong in `reports` and include their metadata.

## Historical context export

`build_match_context` exports observed historical weather for research. No prediction model reads this output. Observed daily weather is not a forecast issued before a match.

```sh
uv run python -m src.mae_model.build_match_context \
  --matches-csv src/outputs/afl_data.csv \
  --output-csv .context/observed_match_context.csv \
  --weather-cache .context/open_meteo_daily_cache.json
```

The export marks the weather source. Missing attendance forecasts stay empty. The shared venue table resolves historical names and sponsor aliases. Unknown venues remain missing and produce a diagnostic.

The original historical data source is [the AFL statistics dataset](https://www.kaggle.com/datasets/stoney71/aflstats). Match and player scraping tools remain available under `src/scrape_afl`.

## Earlier match history for research

[AFL Tables season scores](https://afltables.com/afl/seas/season_idx.html) start in 1897. The importer reads one page per season. It keeps the source match IDs, team names, dates, local times, venues, goals, behinds, and scores. It checks each score, all source match IDs, and the regular-season totals. Cached pages and CSV hashes support repeat checks.

```sh
uv run python -m src.scrape_afl.season_scores \
  --first-year 1897 --last-year 2011 \
  --output-dir .context/history-expansion
uv run python -m src.scrape_afl.build_history_input \
  --history-csv .context/history-expansion/season_scores.csv \
  --current-csv src/outputs/afl_data.csv \
  --output-csv .context/history-expansion/afl_data_1897_2025.csv
```

The combined research file contains 16,838 matches. The 2,879 current rows remain unchanged. Fitzroy, Brisbane Bears, and University remain separate clubs. The current team-name aliases still apply when the model reads the file.

The builder uses named venue timezones. Unknown times or timezones stop the build. Most results use the existing next-local-midnight assumption. The 1996 interrupted match uses midnight after its completion day. The 2006 siren dispute uses the verified ruling time. The 1900 result changed after a protest. Its ruling date is unverified, so the research file defers that corrected result until the next season. The manifest lists these assumptions and their sources. These pages are not an archive of information available before each match.

To repeat the source and 2012 overlap checks, save the overlap page and run the verifier. All 207 overlap matches must match the current CSV.

```sh
curl --fail --output .context/history-expansion/cache/2012.html \
  https://afltables.com/afl/seas/2012.html
uv run python -m src.scrape_afl.verify_season_import \
  --output-dir .context/history-expansion \
  --current-csv src/outputs/afl_data.csv
```

Keep evaluation on the original match IDs. With the complete history starting in 1897, use `--min-train-years 118` to keep the first evaluation season at 2015. The default match data remains unchanged. Tests with longer rating history slightly increased development MAE.
