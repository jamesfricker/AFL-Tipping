# Preseason structural challenger

## Result

This market-free model has a Squiggle-score 2026 MAE of 24.740376 on 215
completed games. The matched Wheelo Ratings MAE is 24.816791. The challenger
has the lower error by 0.076414 points.

The paired bootstrap 95 percent interval for challenger MAE minus Wheelo MAE
is -1.212016 to 1.087856. The interval includes zero. One season does not show
a reliable lead.

| Period | Scoring shots | Challenger | Gain |
| --- | ---: | ---: | ---: |
| 2015 to 2022 development | 27.587261 | 26.790269 | 0.796992 |
| 2023 to 2024 validation | 26.411850 | 26.356214 | 0.055637 |
| 2025 inspected season | 26.630732 | 26.773188 | -0.142456 |
| 2026 inspected season, repository scores | 25.693767 | 24.735725 | 0.958042 |

The 2025 result is worse. The 2024 result is also worse when it is measured by
itself. The aggregate validation gain is small.

## Fixed model rule

The challenger does these operations:

1. It blends two chronological team forecasts with equal weights.
2. One forecast uses the published MoSHBODS score and scoring-shot structure.
3. One forecast uses attack, defence, travel, venue experience, venue form,
   and scoring accuracy.
4. It calculates each team's mean capped preseason scoring-shot margin.
5. Through round 8, it adds 0.2 times the difference between the two teams'
   preseason values.
6. For an absolute predicted margin above 40 points, it adds 0.3 times the
   excess above 40.

The two team configurations, the preseason rule, and the tail rule use the
2015 to 2022 development period. The 2023 and 2024 seasons are validation.
The wider research program inspected the 2026 season before this final replay.
Thus, this is a retrospective challenger. It is not an untouched 2026 test.

The implementation fixes a team identity defect from the first research
script. The common data loader changes `North Melbourne` to `Kangaroos`. The
region table now uses `Kangaroos` too. This fix reduces development MAE from
26.806472 to 26.790269 and validation MAE from 26.476693 to 26.356214.

## Data timing

Regular-season ratings use a result only after its recorded availability time.
Legacy rows use the repository rule of the next local midnight. A preseason
result enters a forecast only when its kickoff is earlier than the prediction
cutoff. The model does not use odds or other models' predictions.

The preseason file contains 242 scored official matches from 2013 to 2026.
The source is the official AFL API competition 2. Eleven returned match records
without scores are excluded. The model calculates scoring shots as goals plus
behinds. It does not use `totalScore`, because old preseason competitions used
super goals.

## Reproduce

```sh
uv run python -m src.mae_model.run_backtest \
  --matches-csv research/preseason_challenger/data/afl_data_2012_2026.csv \
  --preseason-results-csv \
    research/preseason_challenger/data/afl_preseason_results_2013_2026.csv \
  --output-dir research/preseason_challenger/results/run

uv run python research/preseason_challenger/compare_2026.py
```

The source manifests contain the URLs and SHA-256 values. The result directory
contains all model predictions, the period summary, run metadata, the matched
2026 comparison, and its SHA-256 values.

## Live use

Add `--preseason-results-csv` to the fixture command. The same code then adds
`preseason_structural_challenger` to `fixture_predictions.csv`. Live forecasts
use only regular results available at `--as-of` and preseason results with a
kickoff before `--as-of`.

## Sources

- https://www.matterofstats.com/mafl-stats-journal/2026/3/25/moshbods-2026-a-detailed-description
- https://aflapi.afl.com.au/afl/v2/matches
- https://api.squiggle.com.au/?q=games&year=2026
- https://api.squiggle.com.au/?q=tips&year=2026&source=26
