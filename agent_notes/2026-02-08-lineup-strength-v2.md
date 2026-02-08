# Lineup Strength V2 Experiment (2026-02-08)

## Objective
Attempt to improve the best market-anchored model by using stronger pregame lineup information (known 22) plus existing market/context fields already in dataset.

## Hypothesis
A player-quality-weighted lineup strength signal, computed strictly from prior matches, should help anchor/residual corrections when market and internal disagree.

## Changes made
File: `src/mae_model/sequential_margin.py`

1. Added `LineupStrengthTracker`
- Tracks per-player historical impact from prior games only.
- Impact uses available player stats (`disposals`, `goals`, `behinds`, `clearances`, `tackles`, `inside_50s`, `contested_possessions`, `marks`, `goal_assists`) with TOG weighting.
- Produces pregame team-level features:
  - `lineup_strength_diff`
  - `lineup_experience_diff`
  - `lineup_top_end_diff`

2. Integrated lineup strength into pregame feature flow
- `walk_forward_predictions()` now updates pregame feature map with lineup strength tracker output before prediction.
- Tracker updates only after each match to preserve temporal integrity.

3. Rebalanced feature sets to reduce overfit risk
- Simplified market anchor and residual feature vectors relative to prior over-expanded variant.
- Kept a compact set of movement/total/context and lineup terms.

4. Safety/fit compatibility
- Updated anchor default coefficient dimension to align with new feature vector.

## Validation
- `uv run python -m py_compile src/mae_model/sequential_margin.py` -> pass
- `uv run pytest tests/test_sequential_margin_model.py -q` -> pass
- `uv run pytest -q` -> pass (31 tests)

## Backtest run
Command:
- `PYTHONWARNINGS=ignore uv run python -m src.mae_model.run_backtest --output-dir .context/reports_lineup_strength_v2`

## Results
Compared with prior run (`.context/reports_improved_lineup_market`) and strongest earlier run (`.context/reports_rework`):

### market_line_blend
- lineup_market: 2025 = 26.1864, ALL = 26.6690
- lineup_strength_v2: 2025 = 26.2153, ALL = 26.6801
- delta: worse on both

### market_residual_corrector
- lineup_market: 2025 = 26.1211, ALL = 26.7143
- lineup_strength_v2: 2025 = 26.2274, ALL = 26.7348
- delta: worse on both

### strongest earlier run (reference)
- rework: market_residual_corrector 2025 = 26.0815, ALL = 26.6586
- lineup_strength_v2 is below this benchmark.

## Decision
This variant does not improve the best model; it regresses key metrics.

## Interpretation
The current lineup-strength signal adds noise under year-locked evaluation, likely because player-impact estimation from box score aggregates is not stable enough in this form and/or is redundant with market adjustment features.

## Next recommended direction
- Keep lineup signals only in a *gate* (when to correct), not in correction magnitude.
- Revert residual feature set toward the stronger `reports_rework` configuration and add only 1-2 lineup confidence features with very strong shrinkage.
