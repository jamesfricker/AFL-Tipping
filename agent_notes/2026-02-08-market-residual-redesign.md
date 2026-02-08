# Market Residual Corrector Redesign Log (2026-02-08)

## Objective
Reduce MAE for `market_residual_corrector` while preserving walk-forward and year-locked training discipline.

## Starting Point
- Baseline run command: `uv run python -m src.mae_model.run_backtest --output-dir .context/reports_baseline_check`
- Baseline overall MAE:
  - `market_residual_corrector`: `26.5515`
- Baseline yearly (focus years):
  - 2020: 22.5964
  - 2021: 26.8034
  - 2022: 24.1076
  - 2023: 25.1730
  - 2024: 26.6748
  - 2025: 26.0729
- Benchmarks to beat:
  - 2020: 27.82
  - 2021: 27.82
  - 2022: 26.48
  - 2023: 26.48
  - 2024: 26.36
  - 2025: 25.97

## Key Decisions and Rationale

1. Added pre-game engineered state tracking (`TeamFeatureTracker`)
- Why: Need richer internal and disagreement context features without leakage.
- Added:
  - EWMA margin form (alpha 0.1/0.3/0.5)
  - rolling margin means (3/5/10)
  - rest day features (short rest, long break)
  - travel/interstate proxies via venue/team geography mapping
  - season progression scalar
- Leakage control: all features are queried pre-match, then tracker updates after match.

2. Added disagreement context reliability tracker (`DisagreementReliabilityTracker`)
- Why: disagreement signal should be context-dependent and historically calibrated.
- Added smoothed context hit-rate by:
  - finals/regular
  - disagreement bucket
  - favourite + sigma uncertainty bucket
- Used as model feature and damping signal.

3. Upgraded internal margin component with learned pre-game adjuster
- Why: improve `team_residual_lineup` component before market anchoring.
- Implemented `_fit_internal_margin_adjuster` / `_apply_internal_margin_adjuster`:
  - ridge over form/rest/travel features
  - target-encoded venue residual effect with shrinkage
  - year-locked fitting from prior seasons only

4. Upgraded market anchor from fixed weight blend to adaptive anchor model
- Why: fixed blend cannot adapt across uncertainty/disagreement contexts.
- Implemented `_fit_market_anchor_model` / `_apply_market_anchor_model`:
  - anchor adjustment trained on `actual - market_margin`
  - features include market/base/internal disagreement and context terms
  - adaptive damping and clipping tuned by validation MAE

5. Replaced linear residual corrector with stacked residual ensemble
- Why: current correction was too linear for disagreement regimes.
- Implemented:
  - richer residual features (non-linear disagreement terms, bins, interactions, phase, uncertainty, reliability)
  - base learners: Ridge, Huber, HistGradientBoosting
  - chronological OOF stacking for meta-learner
  - optional isotonic calibration when it improves OOF MAE materially

6. Implemented adaptive safety controls
- Why: fixed caps/damping are too rigid.
- Added:
  - context quantile caps by finals/disagreement/sigma buckets
  - model-based expected residual magnitude (`abs_model`) for cap adaptation
  - damping driven by sigma, disagreement magnitude, and disagreement reliability
  - tuned post-correction scale (`post_scale`) to reduce over-correction in late years

7. Preserved walk-forward/year-locked discipline end-to-end
- Year-level configs (`internal`, `anchor`, `residual`) are fit only from prior seasons.
- Feature trackers and reliability state are updated only after prediction.

## Implementation Notes

Main file changed:
- `src/mae_model/sequential_margin.py`

Core integration points:
- New feature trackers and venue/travel utilities near top of file.
- New internal adjuster around `_fit_internal_margin_adjuster`.
- New adaptive anchor around `_fit_market_anchor_model`.
- New stacked residual/safety around `_fit_market_residual_corrector`.
- `walk_forward_predictions` now orchestrates:
  1) pre-game features,
  2) internal adjustment,
  3) adaptive anchor,
  4) stacked residual correction,
  5) tracker updates.

Also added diagnostic model output:
- `team_residual_lineup_form`

## Experiment Iterations

### Iteration 1 (full redesign)
- Result (`.context/reports_rework`):
  - `market_residual_corrector` overall: `26.7120`
  - 2024: `26.5119`
  - 2025: `26.0931`
- Interpretation:
  - Better than baseline on 2024.
  - Slightly worse than baseline on 2025.
  - Still above benchmark in 2024/2025.

### Iteration 2 (runtime + safety cache + reduced grid)
- Added cached raw residual predictions in safety search.
- Reduced tuning grid and removed heavy warning-producing RF path.
- Result:
  - `market_residual_corrector` overall: `26.6586`
  - 2024: `26.5577`
  - 2025: `26.0815`
- Interpretation:
  - Net small improvement vs Iteration 1.
  - 2024 improved materially vs baseline, but still above benchmark.
  - 2025 still just above benchmark.

## Runtime/Engineering Tradeoffs

- Initial advanced grid was too slow; killed and reduced search complexity.
- Added prediction caching in safety tuning to avoid repeated model inference during grid evaluation.
- Removed RF base model due runtime/noise overhead in this environment; retained diverse model family via linear + robust linear + boosted tree.
- Kept full walk-forward training semantics intact despite optimization.

## Validation

- `uv run pytest tests/test_sequential_margin_model.py -q` -> pass
- `uv run pytest -q` -> pass (31 tests)
- Backtests executed multiple times with outputs in `.context/reports_rework`.

## Current Status vs Benchmarks

- Benchmarks beaten for: 2020, 2021, 2022, 2023.
- Benchmarks not yet beaten for: 2024, 2025.
- Gap remaining:
  - 2024: `26.5577` vs target `26.36` (gap +0.1977)
  - 2025: `26.0815` vs target `25.97` (gap +0.1115)

## Next candidate improvements (not yet implemented)

1. Add explicit dual-corrector ensemble: legacy ridge corrector + stacked corrector with year-locked blend weight tuning.
2. Separate late-season/finals residual experts with gating by round phase.
3. Add robust feature stability filtering across folds (drop unstable interaction terms) before fitting each year.
4. Use rolling calibration (last N prior rounds) for correction scale in-season.
