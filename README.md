# AFL Tipping

This repository contains AFL data pipelines and margin/tipping models, with a walk-forward backtest framework.

## Setup

Install dependencies with `uv`:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest -vvrP
```

## Data

Primary historical data source:

- https://www.kaggle.com/datasets/stoney71/aflstats

Generated files used by the models:

- `src/outputs/afl_data.csv`: match-level history (scores, goals/behinds, scoring shots)
- `src/outputs/afl_player_stats.csv`: player-level match stats from AFL Tables scraping

## Backtesting Pipeline

Run the walk-forward backtest:

```bash
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --output-dir reports_rethink \
  --min-train-years 3
```

Outputs:

- `reports_rethink/walk_forward_predictions.csv`
- `reports_rethink/mae_summary.csv`

## Model Family (Current)

The backtest currently evaluates four models in `src/mae_model/sequential_margin.py`:

1. `team_only`
- Sequential team offense/defense ratings updated every match.
- Home advantage and season carryover are applied.
- Uses only match metadata available pre-game (teams, venue/home status).

2. `team_plus_lineup`
- Same team engine as `team_only`.
- Adds player-rating aggregation from lineup rows.

3. `scoring_shots`
- Predicts scoring-shot volume first, then maps to scores via points-per-shot history.
- Also sequential with season carryover.

4. `team_residual_lineup`
- Hybrid decomposition model (`ShotVolumeConversionModel`) that combines scoring-shot volume and conversion effects with lineup conversion ratings.

## Deep Audit (Assumptions, Leakage, Overfit)

Audit run date: **February 7, 2026**.

### What passed

- Walk-forward temporal ordering is respected: models are updated sequentially and only scored after warm-up years.
- Probability calibration for bits uses prior years only, not the target year.
- No global scaler/PCA/imputer leakage paths exist in the current code.

### What failed (important)

1. **Hard leakage in `team_residual_lineup`**
- In `ShotVolumeConversionModel.predict`, the model directly reads `home_scoring_shots` and `away_scoring_shots` from the same match row being predicted.
- Those values are post-game outcomes, not pre-game inputs.
- If shots are missing, fallback uses `home_score / 5` and `away_score / 5`, which are even more direct post-game outcomes.

2. **Feature availability mismatch in lineup weighting**
- Lineup effects use `percent_played` and fallback `disposals` from the same match lineup rows.
- These are in-game/post-game stats and are not available pre-bounce.

### Quantitative leakage evidence

Using the same dataset and walk-forward setup (`min_train_years=3`):

- Baseline reported `team_residual_lineup`: **MAE 14.5606**, **tip 84.90%**
- Same model with **no lineup file at all**: **MAE 14.5575**, **tip 84.90%** (nearly unchanged)
- Same model with `home_scoring_shots=away_scoring_shots=0` forcing score fallback: **MAE 0.5677**, **tip 99.11%**

This confirms the extreme performance is leakage-driven, not true pre-game predictive signal.

### Overfit assessment

- Deployable models (`team_only`, `scoring_shots`) are much closer together and stable year-to-year (overall MAE around 27; 2024-2025 MAE around 26.3-26.9).
- The hybrid model's large gap versus all other models is inconsistent with realistic AFL tipping uplift and is explained by leakage above.

## Which Model To Use For 2026

For **pre-game 2026 predictions**, treat `team_residual_lineup` as **not deployable** until leakage is removed.

Recommended production baseline:

- **Primary:** `team_only`
- **Secondary challenger:** `scoring_shots`

Current walk-forward all-years summary:

- `team_only`: MAE 27.3178, tip 67.85%
- `scoring_shots`: MAE 27.2705, tip 67.80%
- `team_plus_lineup`: MAE 27.3877, tip 67.71%
- `team_residual_lineup`: MAE 14.5606, tip 84.90% (**invalid for pre-game use**)

## 2026 Prediction Workflow

Use this weekly cycle during the 2026 season:

1. Update history through the latest completed round.
- Refresh `src/outputs/afl_data.csv` (and optionally player stats).

2. Re-run backtest as a safety check.

```bash
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --output-dir reports_rethink \
  --min-train-years 3
```

3. Generate next-round margins with `team_only` from current model state.
- Replay all completed historical matches with `model.step(...)`.
- For upcoming fixtures, call `model.predict(...)` (do not update state until results are known).

Example script:

```python
import csv
from src.mae_model.sequential_margin import (
    MatchRow,
    SequentialMarginModel,
    canonical_team_name,
    load_matches_csv,
    parse_match_date,
    round_sort_value,
)

history = load_matches_csv("src/outputs/afl_data.csv")

fixtures = []
with open("fixtures_2026_roundN.csv", newline="") as f:
    for row in csv.DictReader(f):
        fixtures.append(
            MatchRow(
                match_id=row["match_id"],
                year=int(row["year"]),
                round_label=row["round"],
                date=parse_match_date(row["date"]),
                venue=row.get("venue", ""),
                home_team=canonical_team_name(row["home_team_name"]),
                away_team=canonical_team_name(row["away_team_name"]),
                home_score=0.0,
                away_score=0.0,
            )
        )

model = SequentialMarginModel(
    use_lineups=False,
    base_score=76.0,
    home_advantage=3.0,
    team_k=0.05,
    defense_k=0.08,
    season_carryover=0.78,
    home_adv_k=0.0,
)

for m in sorted(history, key=lambda r: (r.date, r.year, round_sort_value(r.round_label), r.match_id)):
    model.step(m, {})

for fx in sorted(fixtures, key=lambda r: (r.date, r.year, round_sort_value(r.round_label), r.match_id)):
    _, _, margin = model.predict(fx, {})
    p_home = 1.0 / (1.0 + __import__("math").exp(-margin / 30.0))
    print(f"{fx.match_id},{fx.home_team} vs {fx.away_team},pred_margin={margin:.2f},home_win_prob={p_home:.3f}")
```

4. After round completion, append actual results and replay to update state before the next round.

## Guardrails For Future Model Changes

- Never use same-match post-game fields (`score`, `goals`, `behinds`, `scoring_shots`, `percent_played`, `disposals`) inside `predict(...)`.
- Keep train/validation strictly walk-forward.
- Treat test performance as one-time evaluation, not a tuning loop.
- If performance looks unusually strong, run leakage ablations before accepting it.
