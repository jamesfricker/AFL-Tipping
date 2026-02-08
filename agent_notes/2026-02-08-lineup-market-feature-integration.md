# Lineup + Rich Market Feature Integration Experiment (2026-02-08)

## Goal
Test whether adding:
1) pregame-known lineup structure signals,
2) richer market fields (open/min/max lines, totals, movement),
can improve the current best market-anchored model path.

## Scope of code changes
Primary file:
- `src/mae_model/sequential_margin.py`

Implemented:
- Added `KnownLineupTracker` to derive pregame lineup features without leakage.
  - `lineup_known_diff`
  - `lineup_returning_diff`
  - `lineup_debut_diff`
  - `lineup_size_diff`
  - `lineup_volatility`
- Expanded `load_market_xlsx()` ingestion beyond close line/odds:
  - line open/min/max
  - odds open/close
  - totals open/min/max/close
  - bookmakers surveyed
- Fed these new features into:
  - `_internal_adjustment_features`
  - `_market_anchor_features`
  - `_market_residual_features`
  - `_market_residual_meta_features`
- Extended residual safety damping with:
  - `line_move_k`
  - `lineup_k`
- Wired lineup tracker + rich market row fields through `walk_forward_predictions()`.

## Validation
- `uv run python -m py_compile src/mae_model/sequential_margin.py` -> pass
- `uv run pytest tests/test_sequential_margin_model.py -q` -> pass
- `uv run pytest -q` -> pass (31)

## Backtest
Command:
- `PYTHONWARNINGS=ignore uv run python -m src.mae_model.run_backtest --output-dir .context/reports_improved_lineup_market`

## Comparison vs prior branch baseline (`.context/reports_rework`)

### market_line_blend
- 2025: 26.2158 -> **26.1864** (improved by 0.0294)
- ALL: 26.6412 -> **26.6690** (worse by 0.0278)

### market_residual_corrector
- 2025: 26.0815 -> **26.1211** (worse by 0.0396)
- ALL: 26.6586 -> **26.7143** (worse by 0.0557)

### market_only
- 2025: 26.1944 -> **26.1944** (no change)
- ALL: 26.6466 -> **26.6466** (no change)

### side/internal diagnostic
- `team_residual_lineup_form` 2025: 26.2998 -> **26.2890** (small improvement)

## Decision
Result is mixed and does **not** improve the strongest path consistently:
- slight 2025 improvement for `market_line_blend`,
- but regression on `market_residual_corrector` and overall MAE.

So this integration is not a clear upgrade yet.

## Likely reason
The expanded feature set adds variance and appears to overfit older-year residual dynamics; lineup context and line-movement effects are not robust enough under current year-locked stacking/safety settings.

## Next iteration ideas
1. Keep rich market + lineup features only in anchor model, revert residual feature expansion.
2. Add strong feature shrinkage / stability filtering before residual stack fitting.
3. Restrict new signals to a learned gate (when to correct) instead of direct residual magnitude prediction.
