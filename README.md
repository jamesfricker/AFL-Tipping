# AFL Tipping

The aim of this repo is to provide a useful resource for AFL tipping. 

The ELO rating script currently calculates an ELO rating for each AFL team.

## Setup

This project now uses `uv` for environment and dependency management.

```bash
uv sync --group dev
```

Run tests with:

```bash
uv run pytest -vvrP
```

## Data

The data used is taken from here

https://www.kaggle.com/datasets/stoney71/aflstats

## Resources Used

These kaggle resources outline how to perform an ELO rating

- https://www.kaggle.com/code/kplauritzen/elo-ratings-in-python/notebook
- https://www.kaggle.com/code/andreiavadanei/elo-predicting-against-dataset

## Future

- add matchup predictions
- automate data collection for each round
- automate predictions 
- receive predictions via email
- add machine learning to better predict fixture outcomes
- add features for players, so we can take into account injuries etc

## MAE Backtesting (Team + Lineups)

A walk-forward MAE pipeline is now available in `src/mae_model/`:

- `src/mae_model/sequential_margin.py`:
  - team offense/defense sequential ratings
  - optional lineup/player effects from per-match player stats
  - scoring-shots variant (using goals + behinds volume)
  - season carryover and home-advantage updates
- `src/mae_model/run_backtest.py`:
  - runs `team_only`, `team_plus_lineup`, and `scoring_shots` backtests
  - writes:
    - `reports/walk_forward_predictions.csv`
    - `reports/mae_summary.csv`

Example:

```bash
uv run python -m src.mae_model.run_backtest \
  --matches-csv src/outputs/afl_data.csv \
  --lineups-csv src/outputs/afl_player_stats.csv \
  --min-train-years 3
```

To generate lineups/player stats from AFL Tables, use:

```python
from src.scrape_afl import scrape_tables
scrape_tables.write_data_to_csv(
    matches_csv_name="src/outputs/afl_data.csv",
    players_csv_name="src/outputs/afl_player_stats.csv",
)
```

`src/outputs/afl_data.csv` now includes:
- `home_goals`, `home_behinds`, `home_scoring_shots`
- `away_goals`, `away_behinds`, `away_scoring_shots`
