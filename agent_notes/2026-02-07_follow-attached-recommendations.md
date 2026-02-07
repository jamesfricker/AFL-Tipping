# Follow-Attachment Execution Log (2026-02-07)

## Request Interpreted
- Follow the attached plan exactly.
- Implement the priority avenues and report outcomes.
- Add a new `agent_notes/*.md` file with decisions, findings, and rationale.

## What I Implemented

### 1) Recency-windowed fitting in meta layers
File: `src/mae_model/sequential_margin.py`

Changes:
- Added `_recent_history_rows(...)` and applied a rolling 3-year fit window to avenue meta-fit steps.
- Updated avenue blend fitting (`_fit_avenue9_blend_weights`) to include exponential recency decay by season.
- Routed all avenue fitting calls in `_build_avenue_experiments(...)` to use recency-filtered history.

Why:
- The attached guidance emphasized modern-era weighting (recent seasons should dominate).

### 2) Expected-minutes lineup weighting (Avenue 3 unblock)
File: `src/mae_model/sequential_margin.py`

Changes in `SequentialMarginModel`:
- Added `player_expected_tog` and per-team player TOG state.
- Prediction-time lineup weighting now uses expected TOG derived from prior matches only.
- Post-match updates ingest observed `percent_played` to update expected TOG state.

Why:
- Uniform lineup weighting was likely too weak.
- This keeps predict-time features pre-game-safe while using historical TOG information.

### 3) Team-specific carryover using lineup continuity (Avenue 4)
File: `src/mae_model/sequential_margin.py`

Changes:
- Added prior-season per-team player TOG snapshots.
- On first team appearance each season, compute returning-minutes continuity from lineup names.
- Apply bounded team-specific carryover adjustment around the global carryover.

Why:
- Global season carryover is too blunt; continuity should influence how much prior strength is retained.

### 4) Leakage guardrail for new chain path
File: `tests/test_sequential_margin_model.py`

Changes:
- Added `test_territory_chain_predict_does_not_use_same_match_outcomes`.

Why:
- Keep the new path aligned with strict pre-game/no-leakage behavior.

## Validation

Commands run:

```bash
uv run pytest -q
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --output-dir .context/avenues_follow_exactly \
  --min-train-years 3
```

Result:
- Tests: `17 passed`.
- Backtest produced updated `mae_summary.csv` in `.context/avenues_follow_exactly`.

## Key Findings (2024 / 2025)

Benchmarks:
- 2024 target: `26.36`
- 2025 target: `25.97`

Selected model outcomes after changes:
- `team_only`: 2024 `26.4834`, 2025 `26.3481`
- `team_plus_lineup`: 2024 `26.6245`, 2025 `26.4005`
- `team_residual_lineup`: 2024 `26.7310`, 2025 `26.1846`
- `av9_ensemble` (recency-weighted): 2024 `26.7791`, 2025 `26.4247`

Delta vs previous run (`.context/avenues_explore_v2`):
- `av9_ensemble`: 2024 worsened by `+0.1315`, 2025 improved by `-0.0153`, ALL worsened by `+0.0367`.
- `team_plus_lineup`: 2024 worsened by `+0.0930`, 2025 worsened by `+0.1025`.

## Decisions and Why

1. Keep the safety constraints intact.
- No predict-time use of same-match outcomes was introduced.
- Reason: prior leakage incidents make this non-negotiable.

2. Retain the implemented mechanisms for further tuning, but do not declare improvement yet.
- Recency and expected-minutes carryover mechanics are now in code and testable.
- Current parameterization did not improve the benchmark years.

3. Best model conclusion did not change.
- No model beats both benchmark targets simultaneously in this run.
- Best by 2024 remains `team_only`.
- Best by 2025 remains `team_residual_lineup`.

## Final Outcome
- The attached plan’s priority avenues were implemented and validated.
- In this first pass with these exact additions, MAE did not improve on the benchmark-binding years enough to cross targets.
