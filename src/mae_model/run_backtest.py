import argparse
from pathlib import Path

from .data import (
    load_market_csv,
    load_market_xlsx,
    load_matches_csv,
)
from .reporting import write_metadata
from .sequential_margin import (
    summarize_predictions,
    walk_forward_predictions,
    write_prediction_rows,
    write_summary_rows,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay four AFL margin models in time order."
    )
    parser.add_argument("--matches-csv", default="src/outputs/afl_data.csv")
    market = parser.add_mutually_exclusive_group()
    market.add_argument(
        "--market-csv", help="Market snapshots with observed_at timestamps."
    )
    market.add_argument(
        "--market-xlsx",
        help="Untimed closing workbook. Requires --closing-line-benchmark.",
    )
    parser.add_argument("--closing-line-benchmark", action="store_true")
    parser.add_argument(
        "--lead-hours",
        type=float,
        default=0.0,
        help="Prediction deadline in hours before each kickoff.",
    )
    parser.add_argument("--min-train-years", type=int, default=3)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args(argv)
    if args.closing_line_benchmark != bool(args.market_xlsx):
        parser.error("Use --market-xlsx and --closing-line-benchmark together")
    if args.closing_line_benchmark and args.lead_hours != 0:
        parser.error(
            "Closing benchmark cannot represent an earlier deadline; lead-hours must be 0"
        )
    try:
        matches = load_matches_csv(args.matches_csv)
        if not matches:
            raise ValueError("Match history is empty")
        quotes = load_market_csv(args.market_csv) if args.market_csv else []
        if args.market_xlsx:
            quotes = load_market_xlsx(
                args.market_xlsx, matches, closing_line_benchmark=True
            )
        predictions = walk_forward_predictions(
            matches,
            args.min_train_years,
            quotes,
            closing_line_benchmark=args.closing_line_benchmark,
            lead_hours=args.lead_hours,
        )
        if not predictions:
            raise ValueError("No matches remain after the training period")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    summary = summarize_predictions(predictions)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_prediction_rows(str(output / "walk_forward_predictions.csv"), predictions)
    write_summary_rows(str(output / "mae_summary.csv"), summary)
    paths = [
        path for path in (args.matches_csv, args.market_csv, args.market_xlsx) if path
    ]
    write_metadata(output, paths, predictions, matches, vars(args), quotes)
    for row in summary:
        if row["year"] == "ALL" and row["scope"] == "all_matches":
            print(
                f"{row['model_name']}: games={row['num_games']} mae_margin={row['mae_margin']} tip_pct={row['tip_pct']}"
            )


if __name__ == "__main__":
    main()
