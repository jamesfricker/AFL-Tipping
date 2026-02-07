# Team Residual Lineup Leakage Fix

Date: 2026-02-07
Branch: `jamesfricker/fix-model-leak`

## Request
Patch `team_residual_lineup` to remove leakage and make it pre-game deployable, then compare stats before vs after.

## Implementation Log

1. Captured pre-patch baseline metrics from `reports_rethink/mae_summary.csv`.
2. Reviewed `ShotVolumeConversionModel` in `src/mae_model/sequential_margin.py`.
3. Removed leakage paths in `predict()`:
- `predict()` no longer consumes same-match `home_scoring_shots` / `away_scoring_shots`.
- `predict()` now estimates shot volume from model state only (team attack/defense shots + home advantage).
- lineup conversion effect no longer uses same-match in-game stats (`percent_played`, `disposals`) as weights.
4. Kept post-game updates in `update()` (allowed), including:
- updating shot states from actual scoring shots after results are known;
- updating conversion states from actual points-per-shot after results are known;
- updating player conversion ratings from goals/behinds after results are known.
5. Added regression test to lock behavior:
- `test_hybrid_predict_does_not_use_same_match_outcomes` ensures changing same-match scores/shots does not alter hybrid pre-game prediction.
6. Re-ran tests and full backtest for post-patch metrics.

## Files Changed

- `src/mae_model/sequential_margin.py`
- `tests/test_sequential_margin_model.py`

## Key Model Changes (Technical)

### `ShotVolumeConversionModel` initialization
Added explicit shot-volume state/params:
- `base_scoring_shots`
- `home_adv_scoring_shots`
- `shot_k`
- `shot_residual_cap`
- state dictionaries: `team_attack_shots`, `team_defense_shots`

### `predict()` now uses only pre-game-available state
Old behavior:
- used match row scoring shots (`home_scoring_shots`, `away_scoring_shots`) inside prediction.

New behavior:
- predicts scoring shots from historical team shot ratings;
- predicts conversion (points per shot) from league/team/player conversion state;
- combines predicted shot volume and predicted conversion for expected margin.

### `update()` now trains two residual channels
- shot residuals update shot attack/defense ratings;
- conversion residuals update conversion attack/defense ratings;
- player conversion ratings updated from post-game player goals/behinds.

### Seasonal transition
Now re-centers and carries both shot and conversion states.

## Validation Results

### Tests
Command:

```bash
uv run pytest -q
```

Result:

- `14 passed in 2.40s`

### Backtest rerun
Command:

```bash
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --output-dir .context/leak_fix_reports \
  --min-train-years 3
```

## Metrics Comparison (Pre vs Post)

Source files:
- pre: `reports_rethink/mae_summary.csv`
- post: `.context/leak_fix_reports/mae_summary.csv`

### ALL years summary

- `scoring_shots`: MAE `27.2705 -> 27.2705`, tip% `67.80 -> 67.80`, bits/game `0.1487 -> 0.1487`
- `team_only`: MAE `27.3178 -> 27.3178`, tip% `67.85 -> 67.85`, bits/game `0.1434 -> 0.1434`
- `team_plus_lineup`: MAE `27.3877 -> 27.3877`, tip% `67.71 -> 67.71`, bits/game `0.1418 -> 0.1418`
- `team_residual_lineup`: MAE `14.5606 -> 27.2271`, tip% `84.90 -> 68.11`, bits/game `0.5278 -> 0.1480`

### Interpretation

- Non-hybrid models are unchanged, as expected.
- Hybrid model dropped from unrealistically strong performance to parity with deployable baselines.
- This confirms previous uplift was leakage-driven and that the patched hybrid is now consistent with plausible out-of-sample AFL tipping performance.

## By-Year Hybrid Change (`team_residual_lineup`)

- 2015: MAE `16.2094 -> 31.0973`, tip% `86.41 -> 69.42`
- 2016: MAE `15.2843 -> 30.4808`, tip% `85.51 -> 70.53`
- 2017: MAE `15.0059 -> 29.1191`, tip% `83.57 -> 63.77`
- 2018: MAE `13.8500 -> 26.7463`, tip% `87.44 -> 71.98`
- 2019: MAE `14.8842 -> 27.0366`, tip% `87.92 -> 63.29`
- 2020: MAE `13.6889 -> 23.3652`, tip% `85.80 -> 66.67`
- 2021: MAE `13.6330 -> 27.1875`, tip% `84.54 -> 67.63`
- 2022: MAE `14.0797 -> 24.4752`, tip% `85.02 -> 71.98`
- 2023: MAE `14.6979 -> 26.1630`, tip% `82.87 -> 70.37`
- 2024: MAE `14.9446 -> 26.8232`, tip% `81.94 -> 62.50`
- 2025: MAE `13.7211 -> 26.2875`, tip% `83.33 -> 70.83`

## Remaining Caveats

- Backtest still uses historical lineups keyed by match id; this is acceptable for research but assumes lineup lists are known pre-game in production.
- `team_plus_lineup` still uses same-match `percent_played`/`disposals` weighting and remains not strictly pre-game-safe unless similarly patched.
