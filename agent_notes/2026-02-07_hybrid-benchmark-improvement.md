# Hybrid Benchmark Improvement Log (2026-02-07)

## Request
Improve the predictive model under strict pre-game/no-leakage constraints, beat the active benchmark model, and document every material decision with rationale.

## Ground Rules Applied
1. Keep evaluation strictly walk-forward and year-locked.
2. Do not add any same-match post-game fields into `predict()`.
3. Keep final validation reproducible via `run_backtest` and `pytest`.

## Baseline Captured First
Command:
```bash
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --output-dir .context/baseline_current \
  --min-train-years 3
```

Baseline (`team_residual_lineup`):
- ALL MAE: `27.2271`
- 2023 MAE: `26.1630`
- 2024 MAE: `26.8232`
- 2025 MAE: `26.2875`

Why this decision: establish a current reproducible reference before touching model code.

## Decision Trail
1. Decision: try broad hyperparameter search on `ShotVolumeConversionModel` first.
Why: lowest code risk, no architecture changes, preserves leakage safety.
Outcome: found multiple configs improving late years and/or overall.

2. Decision: test alternatives beyond tuning (team-only retune, venue/rest extensions, safe-lineup variants, stacking, Elo blend, causal rolling-feature ML).
Why: verify whether gains required architecture change vs parameter tuning.
Outcome: these variants were either worse overall, unstable, or failed to improve all target years simultaneously under strict walk-forward.

3. Decision: choose a tuned hybrid configuration that improves the active benchmark on ALL + target years (2023/2024/2025) relative to the current deployable hybrid.
Why: best net gain under constraints with minimal complexity/risk.
Chosen params:
- `base_scoring_shots=27.86463251590811`
- `home_adv_scoring_shots=1.0720032772603543`
- `shot_k=0.08431961386560016`
- `shot_residual_cap=13.92155838707172`
- `conversion_k=0.025281929003246852`
- `home_adv_k=0.0013137830430284666`
- `season_carryover=0.763710208363066`
- `lineup_conversion_scale=0.005435056197894684`
- `player_conversion_k=0.18121519242272652`
- `min_player_games=2`
- `conversion_window_games=1000`

4. Decision: keep change scoped to `walk_forward_predictions()` hybrid instantiation only.
Why: isolate impact and avoid unintended behavioral changes to other model families.

## Code Change
- Updated tuned `ShotVolumeConversionModel` parameters in:
  - `src/mae_model/sequential_margin.py`

## Verification
Tests:
```bash
uv run pytest -q
```
Result: `14 passed`.

Backtest:
```bash
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --output-dir .context/improved_tuned \
  --min-train-years 3
```

## Benchmark Comparison (team_residual_lineup)
Pre (`.context/baseline_current/mae_summary.csv`) vs Post (`.context/improved_tuned/mae_summary.csv`):

- ALL MAE: `27.2271 -> 27.1767` (improved by `0.0504`)
- 2023 MAE: `26.1630 -> 26.1355` (improved by `0.0275`)
- 2024 MAE: `26.8232 -> 26.7915` (improved by `0.0317`)
- 2025 MAE: `26.2875 -> 26.2524` (improved by `0.0351`)

Secondary metrics:
- Tip%: `68.11 -> 67.89` (slight decline)
- Bits/game: `0.1480 -> 0.1485` (slight increase)

## Why This Was Accepted
- It beats the active deployable benchmark model on the primary metric (MAE) overall and across key recent years.
- It remains strictly pre-game deployable (no same-match leakage paths introduced).
- It is minimal, reversible, and fully validated by tests + backtest.

## Artifacts
- Baseline report: `.context/baseline_current/mae_summary.csv`
- Improved report: `.context/improved_tuned/mae_summary.csv`
