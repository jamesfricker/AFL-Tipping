# Avenues Exploration Outcomes (2026-02-07)

## Objective
- Improve the model under strict walk-forward, pre-game-safe constraints.
- Explore the identified avenues and quantify MAE impact against benchmarks:
  - 2024 benchmark: `26.36`
  - 2025 benchmark: `25.97`

## Reproducible Validation
Command:

```bash
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --output-dir .context/avenues_explore_v2 \
  --min-train-years 3
```

Tests:

```bash
uv run pytest -q
```

Result: `17 passed`.

## What Was Implemented

### Avenue 1 - Pace/Territory Chain
- Implemented `TerritoryShotChainModel` (`av1_territory_chain`) in `src/mae_model/sequential_margin.py`.
- Model decomposes expected margin as:
  - expected inside-50 volume
  - expected shot rate (shots per inside-50)
  - expected points-per-shot
- Predict path uses only pre-game model state.
- Update path uses post-game actuals, including lineup inside-50 aggregates.

Outcome:
- 2024 MAE: `26.7297` (gap `+0.3697`)
- 2025 MAE: `26.5792` (gap `+0.6092`)
- Decision: rejected for production benchmark target.

### Avenue 2 - MAE Tail Calibration
- Implemented as `av2_tail_cal`:
  - year-locked threshold+slope shrink of large margins.

Outcome:
- 2024 MAE: `27.0399` (gap `+0.6799`)
- 2025 MAE: `26.3421` (gap `+0.3721`)
- Decision: rejected.

### Avenue 5 - Finals-Specific Head
- Implemented as `av5_finals_head`:
  - finals-only linear head (scale+bias) fit using prior years.

Outcome:
- 2024 MAE: `26.9535` (gap `+0.5935`)
- 2025 MAE: `26.3289` (gap `+0.3589`)
- Decision: rejected.

### Avenue 6 - Matchup/Venue Interactions with Shrinkage
- Implemented as `av6_matchup`:
  - shrunken residual effects for `(home_team, venue)`, `(away_team, venue)`, `(home_team, away_team)`.
  - shrinkage tuned in year-locked manner (prior years only).

Outcome:
- 2024 MAE: `26.9461` (gap `+0.5861`)
- 2025 MAE: `26.3233` (gap `+0.3533`)
- Decision: rejected.

### Avenue 7 - Uncertainty-Based Shrink
- Implemented as `av7_uncertainty`:
  - round-bucket scaling (`early`, `mid`, `late`, `finals`) fit on prior years.

Outcome:
- 2024 MAE: `26.9932` (gap `+0.6332`)
- 2025 MAE: `26.3421` (gap `+0.3721`)
- Decision: rejected.

### Avenue 9 - Small Regularized Ensemble
- Implemented as `av9_ensemble`:
  - year-locked convex blend of base models:
    - `team_only`
    - `team_plus_lineup`
    - `scoring_shots`
    - `team_residual_lineup`

Outcome:
- 2024 MAE: `26.6476` (gap `+0.2876`)
- 2025 MAE: `26.4400` (gap `+0.4700`)
- ALL MAE: `27.2266` (best overall among all evaluated models)
- Decision: keep as an all-years challenger, not as benchmark-years winner.

## Additional Safety Work (Avenue 10 intent)
- Added leakage regression test:
  - `test_territory_chain_predict_does_not_use_same_match_outcomes` in `tests/test_sequential_margin_model.py`
- Existing leakage tests for other models remain in place.

## Not Implemented / Blocked This Pass
- Avenue 3 (expected minutes + multi-skill lineup priors): blocked by lack of explicit pre-game expected minutes/team-sheet signal in current dataset.
- Avenue 4 (team-specific carryover via roster continuity): partially approximated by year-locked fitting choices, but not fully implemented as explicit continuity features.
- Avenue 8 (market prior): blocked; no market line/odds dataset in repo.

## Benchmark Comparison Snapshot (Current Bests)
- Best 2024 MAE among all evaluated models: `team_only` = `26.4834` (gap `+0.1234`)
- Best 2025 MAE among all evaluated models: `team_residual_lineup` = `26.1846` (gap `+0.2146`)
- Best combined 2024+2025 average: `team_plus_lineup` = `26.4147`
- Best ALL-years MAE: `av9_ensemble` = `27.2266`

## Finalization
For the benchmark-binding years (2024 and 2025 together), the best model remains:
- `team_plus_lineup` (best combined 2024+2025 MAE)

For all-years leaderboard only:
- `av9_ensemble` is best overall MAE.

No explored avenue in this pass beat both benchmark targets (`2024 <= 26.36` and `2025 <= 25.97`) simultaneously.
