# Market Residual Corrector Optimization (2026-02-08)

## Request
User asked to incorporate attached feedback and improve model performance further, with explicit focus on market-first strategy and strict pre-game safety. User also requested a full thought process + decision log.

## Constraints and Guardrails
- Keep strict walk-forward year-locked fitting.
- No same-match post-game leakage into prediction path.
- Market data treated as pre-game only.
- Maintain existing backtest framework and comparability.

## Feedback distilled into implementation goals
1. Use market as backbone, not just a weak blend feature.
2. Add `market_only` baseline to quantify true market signal.
3. Build residual-correction model with:
   - small regularized correction,
   - MAE-safe clipping,
   - odds-derived uncertainty damping,
   - year-locked fitting only.
4. Compare against existing best (`market_line_blend`) and keep only if validated.

## Initial state before this iteration
Best deployable model in current code was `market_line_blend` with recency-window weight fitting:
- ALL MAE: `26.5735`
- 2024 MAE: `26.6253`
- 2025 MAE: `26.0380`

## Design options considered
- A) Keep only convex market/internal blend with static or recency weight.
- B) Blend market with context model (`team_context_env`).
- C) Residual-correct around raw market line.
- D) Residual-correct around already-strong market blend anchor.

Observed from prior quick checks:
- B and C underperformed relative to `market_line_blend`.
- D showed potential uplift in offline walk-forward simulation.

Decision: implement D as a first-class model.

## Implementation details
### 1) Added market probability + uncertainty helpers
In `src/mae_model/sequential_margin.py`:
- `_market_implied_home_probability(...)`: removes vig from home/away odds and clamps probability.
- `_market_sigma_from_spread_and_probability(...)`: derives game uncertainty scale from line + implied probability.

### 2) Added residual-corrector feature path
- `_market_residual_features(...)`
- `_fit_ridge_from_rows(...)`
- `_apply_market_residual_corrector(...)`
- `_fit_market_residual_corrector(...)`

Key modeling decisions:
- Anchor = `market_line_blend` margin, not raw market margin.
- Target = residual: `actual_margin - anchor_margin`.
- Features include:
  - anchor margin,
  - internal model margin (`team_residual_lineup`),
  - signed/absolute disagreement,
  - finals flag,
  - implied odds edge,
  - uncertainty deviation.
- Robustness controls:
  - residual target clip to `[-25, +25]` during fit,
  - correction clip candidates in fit (`3` to `8` points),
  - uncertainty damping via `1 / (1 + k * sigma_norm)`.

### 3) Integrated new market models in walk-forward
In `walk_forward_predictions(...)`:
- Added `market_only` prediction output.
- Added `market_residual_corrector` prediction output.
- Added year-locked config fitting for residual corrector using prior years only.
- Kept fallback behavior for missing market rows (use base prediction path).

### 4) Tests added/updated
`tests/test_sequential_margin_model.py`:
- `test_market_probability_removes_vig_and_bounds`
- `test_market_sigma_estimate_from_line_and_probability`
- `test_walk_forward_adds_market_only_and_residual_models`

All tests passing after implementation.

## Experiment chronology and outcomes
### Run A (first residual version)
- `market_residual_corrector` improved over its own initial baseline but still under `market_line_blend`.
- Decision: iterate architecture from raw-market anchor to market-blend anchor.

### Run B (final residual version, market-blend anchored)
Backtest command:
- `uv run python -m src.mae_model.run_backtest --matches-csv src/outputs/afl_data.csv --lineups-csv src/outputs/afl_player_stats.csv --market-xlsx src/outputs/afl_betting_history.xlsx --output-dir .context/market_residual_v2_without_context --min-train-years 3`

Key ALL-years results:
- `market_residual_corrector`: **26.5515**
- `market_line_blend`: 26.5735
- `market_only`: 26.6466

Delta vs previous best (`market_line_blend`):
- ALL MAE: `-0.0220` (improvement)
- Tip%: `68.78` vs `68.56` (improvement)

Benchmark-year note:
- 2024/2025 remain slightly better on `market_line_blend` than `market_residual_corrector`.
- Therefore recommendation is objective-dependent:
  - optimize ALL-years leaderboard -> `market_residual_corrector`
  - optimize 2024/2025-only benchmark years -> `market_line_blend`

## Files changed in this iteration
- `src/mae_model/sequential_margin.py`
- `tests/test_sequential_margin_model.py`
- `README.md`

## Final decision
- Keep `market_residual_corrector` as the new best overall model (ALL MAE).
- Keep `market_line_blend` as a strong benchmark-year challenger.
- Keep `market_only` as an essential market-signal baseline for future iterations.

## Risks / caveats
- Gains are modest; they are real but small.
- 2024/2025-specific targets are still not fully cleared.
- Market timestamp quality still matters; if line timestamps are noisy, real-world performance may drift.

## Next likely improvements
- Add line movement features (open vs close) if reliably available pre-game.
- Add totals (OU) and implied team points into residual features.
- Add stronger year-locked hyperparameter selection focused on 2024/2025 objective, if that remains the acceptance target.
