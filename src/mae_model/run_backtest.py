import argparse
from pathlib import Path

from .sequential_margin import (
    load_lineups_csv,
    load_matches_csv,
    summarize_predictions,
    walk_forward_predictions,
    write_prediction_rows,
    write_summary_rows,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run walk-forward MAE backtest for AFL margin models."
    )
    parser.add_argument(
        "--matches-csv",
        default="src/outputs/afl_data.csv",
        help="Path to match-level CSV.",
    )
    parser.add_argument(
        "--lineups-csv",
        default="src/outputs/afl_player_stats.csv",
        help="Path to per-player match stats CSV. If missing, lineup effects default to zero.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for prediction and summary outputs.",
    )
    parser.add_argument(
        "--min-train-years",
        type=int,
        default=3,
        help="Warm-up years before scoring MAE.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    matches = load_matches_csv(args.matches_csv)
    lineups = load_lineups_csv(args.lineups_csv)
    predictions = walk_forward_predictions(
        matches=matches,
        lineups=lineups,
        min_train_years=args.min_train_years,
    )
    summary = summarize_predictions(predictions)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_rows(str(output_dir / "walk_forward_predictions.csv"), predictions)
    write_summary_rows(str(output_dir / "mae_summary.csv"), summary)

    for row in summary:
        if row["year"] == "ALL":
            print(
                f"{row['model_name']}: games={row['num_games']} "
                f"mae_margin={row['mae_margin']} "
                f"tip_pct={row['tip_pct']} "
                f"bits_per_game={row['bits_per_game']}"
            )


if __name__ == "__main__":
    main()
