import argparse
from pathlib import Path

from .data import (
    load_fixtures_csv,
    load_market_csv,
    load_matches_csv,
    parse_timestamp,
)
from .reporting import write_metadata
from .sequential_margin import (
    predict_fixtures,
    write_prediction_rows,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict future fixtures from information available at as-of."
    )
    parser.add_argument("--matches-csv", default="src/outputs/afl_data.csv")
    parser.add_argument("--fixtures-csv", required=True)
    parser.add_argument("--market-csv")
    parser.add_argument(
        "--as-of",
        required=True,
        help="Prediction time in ISO format with a timezone offset.",
    )
    parser.add_argument(
        "--lead-hours",
        type=float,
        default=0.0,
        help="Historical training deadline in hours before kickoff.",
    )
    parser.add_argument("--output-dir", default="predictions")
    args = parser.parse_args(argv)
    try:
        matches = load_matches_csv(args.matches_csv)
        fixtures = load_fixtures_csv(args.fixtures_csv)
        if not fixtures:
            raise ValueError("Fixture file is empty")
        quotes = load_market_csv(args.market_csv) if args.market_csv else []
        as_of = parse_timestamp(args.as_of)
        predictions = predict_fixtures(
            matches, fixtures, as_of, quotes, lead_hours=args.lead_hours
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_prediction_rows(str(output / "fixture_predictions.csv"), predictions)
    paths = [
        path for path in (args.matches_csv, args.fixtures_csv, args.market_csv) if path
    ]
    write_metadata(output, paths, predictions, matches, vars(args), quotes)
    print(f"Wrote predictions for {len(fixtures)} fixtures to {output}")


if __name__ == "__main__":
    main()
