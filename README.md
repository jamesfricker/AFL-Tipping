# AFL Tipping

Predict AFL margins with team ratings, scoring-shot ratings, market prices, and one simple market blend.

The model does not use lineup statistics, actual weather, form corrections, or stacked residual models. These inputs and correction layers did not show a reliable gain in the model investigation.

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

## Model rules

| Output | Method |
| --- | --- |
| `team_only` | Sequential attack and defence ratings from completed scores. |
| `scoring_shots` | Sequential shot-volume ratings with historical points per shot. |
| `market_only` | The eligible market margin, without correction. |
| `market_scoring_blend` | A convex blend of the market margin and scoring-shot prediction. |

The blend fits one market weight from the preceding five seasons. It selects from 0 to 1 in steps of 0.02. Weight 1 gives the market-only prediction. Equal errors favour the larger market weight. Insufficient history also selects weight 1.

The weight stays fixed during the target season. Its fitting cutoff is 1 January at 00:00 in Australia/Sydney, or the request deadline if earlier. Earlier training prices must also pass their own historical prediction deadlines.

The model does not fit residual corrections, correction limits, or a calibration stack. The target season is excluded from weight fitting. Historical comparisons for 2024 and 2025 are not untouched tests because earlier model development used those results.

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
