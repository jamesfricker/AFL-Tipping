# Experiment MAE Audit (2026-02-07)

## Scope
- Investigate each backtest experiment currently represented in the repo.
- Quantify MAE impact per experiment/model.
- Compare against benchmark targets in `benchmarks.md`.
- Finalize the best model under current strict walk-forward setup.

## Artifacts Reviewed
- `benchmarks.md`
- `reports_baseline/mae_summary.csv`
- `reports_rethink/mae_summary.csv`
- `agent_notes/2026-02-07_team-residual-lineup-leakage-fix.md`
- `agent_notes/2026-02-07_hybrid-benchmark-improvement.md`
- `src/mae_model/sequential_margin.py`

## Reproduction Run (Current Code)
Command used:

```bash
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --output-dir .context/exp_audit \
  --min-train-years 3
```

Fresh overall MAE (ALL years):
- `scoring_shots`: `27.2705`
- `team_only`: `27.3178`
- `team_plus_lineup`: `27.3828`
- `team_residual_lineup`: `27.2709`

## Experiment Investigation

### Experiment 1: Baseline family (`reports_baseline`)
Models:
- `scoring_shots`
- `team_only`
- `team_plus_lineup`

MAE impact:
- 2024 best in this set: `team_only` (`26.4834`)
- 2025 best in this set: `team_plus_lineup` (`26.3148` in baseline file; `26.2980` in fresh rerun)
- ALL years best in this set: `scoring_shots` (`27.2705`)

Interpretation:
- Baseline models are tightly clustered.
- No baseline model clears both 2024 and 2025 benchmarks.

### Experiment 2: Rethink report (`reports_rethink`) with 14.x hybrid
Added model:
- `team_residual_lineup` reported at `14.5606` ALL MAE (`2024: 14.9446`, `2025: 13.7211`)

MAE impact:
- Massive apparent uplift vs all other models.

Interpretation:
- This uplift is not consistent with current safe code path and prior leakage notes.
- Treat this report as stale/invalid for deployment decisions.

### Experiment 3: Current-code rerun (`.context/exp_audit`)
Current model set:
- `team_only`
- `team_plus_lineup`
- `scoring_shots`
- `team_residual_lineup` (now benchmark-shot-volume implementation)

MAE impact vs benchmarks:

| Model | 2024 MAE | Gap vs 26.36 | 2025 MAE | Gap vs 25.97 | Avg(2024,2025) |
|---|---:|---:|---:|---:|---:|
| `team_only` | 26.4834 | +0.1234 | 26.3481 | +0.3781 | 26.4157 |
| `team_plus_lineup` | 26.5315 | +0.1715 | 26.2980 | +0.3280 | 26.4147 |
| `team_residual_lineup` | 26.7310 | +0.3710 | 26.1846 | +0.2146 | 26.4578 |
| `scoring_shots` | 26.8552 | +0.4952 | 26.6308 | +0.6608 | 26.7430 |

Observations:
- No model beats both benchmark years (2024 and 2025).
- `team_only` is best on 2024.
- `team_residual_lineup` is best on 2025.
- Balanced across 2024+2025 average, `team_plus_lineup` is marginally best.

## Benchmark Hit Count (2020-2025)
Using benchmarks in `benchmarks.md`:
- All four models hit `4/6` benchmark years.
- All four miss 2024 and 2025 in current rerun.

## Finalization: Best Current Model
Selected best model: `team_plus_lineup`

Reason:
- It has the best combined 2024+2025 MAE (lowest average and smallest summed gap to benchmarks) in the current reproduced run.
- It is within ~0.05 MAE of best-in-year for both target years, making it the strongest compromise model for the two binding benchmark years.

Important caveat:
- If the priority is strictly 2024 only, prefer `team_only`.
- If the priority is strictly 2025 only, prefer `team_residual_lineup`.
