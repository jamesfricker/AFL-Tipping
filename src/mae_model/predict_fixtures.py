import argparse
from pathlib import Path

from .data import (
    load_fixtures_csv,
    load_market_csv,
    load_matches_csv,
    parse_timestamp,
)
from .player_margin import (
    PlayerModelConfig,
    load_lineup_snapshots_csv,
    load_player_matches_csv,
    replay_player_predictions,
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
    parser.add_argument("--player-stats-csv")
    parser.add_argument("--lineups-csv")
    parser.add_argument(
        "--player-signal",
        default="rating_form",
        choices=("rating", "form", "missing_leader", "rating_form"),
    )
    parser.add_argument("--output-dir", default="predictions")
    args = parser.parse_args(argv)
    if bool(args.player_stats_csv) != bool(args.lineups_csv):
        parser.error("Use --player-stats-csv and --lineups-csv together")
    player_config = None
    player_diagnostics = None
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
        if args.player_stats_csv:
            player_config = PlayerModelConfig(signal=args.player_signal)
            appearances = load_player_matches_csv(args.player_stats_csv, matches)
            lineups = load_lineup_snapshots_csv(args.lineups_csv, fixtures)
            player_rows, player_diagnostics = replay_player_predictions(
                matches, predictions, appearances, lineups, player_config
            )
            predictions.extend(player_rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_prediction_rows(str(output / "fixture_predictions.csv"), predictions)
    paths = [
        path
        for path in (
            args.matches_csv,
            args.fixtures_csv,
            args.market_csv,
            args.player_stats_csv,
            args.lineups_csv,
        )
        if path
    ]
    write_metadata(
        output,
        paths,
        predictions,
        matches,
        vars(args),
        quotes,
        player_config=player_config,
        player_diagnostics=player_diagnostics,
    )
    print(f"Wrote predictions for {len(fixtures)} fixtures to {output}")


if __name__ == "__main__":
    main()
