import argparse
from pathlib import Path

from .data import (
    load_fixtures_csv,
    load_market_csv,
    load_matches_csv,
    parse_timestamp,
)
from .player_margin import (
    HybridPlayerConfig,
    PlayerHistory,
    PlayerModelConfig,
    load_lineup_snapshots_csv,
    load_player_matches_csv,
    replay_hybrid_player_predictions,
    replay_player_predictions,
)
from .reporting import write_metadata
from .selected_team_margin import (
    SelectedTeamConfig,
    replay_selected_team_predictions,
)
from .sequential_margin import (
    predict_fixtures,
    walk_forward_predictions,
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
    parser.add_argument("--official-player-stats-csv")
    parser.add_argument(
        "--official-player-rating-prior-games", type=float, default=12.0
    )
    parser.add_argument("--lineups-csv")
    parser.add_argument("--official-player-lineups-csv")
    parser.add_argument(
        "--player-signal",
        default="rating_form",
        choices=("rating", "form", "missing_leader", "rating_form"),
    )
    parser.add_argument(
        "--player-measurement",
        default="outcome_fantasy",
        choices=(
            "outcome_fantasy",
            "official_points",
            "official_points_per_time",
        ),
    )
    parser.add_argument(
        "--player-control",
        default="market_scoring_blend",
        choices=("team_only", "scoring_shots", "market_scoring_blend"),
    )
    parser.add_argument("--player-rating-prior-games", type=float, default=6.0)
    parser.add_argument("--output-dir", default="predictions")
    args = parser.parse_args(argv)
    if bool(args.player_stats_csv) != bool(args.lineups_csv):
        parser.error("Use --player-stats-csv and --lineups-csv together")
    if args.official_player_stats_csv and (
        not args.player_stats_csv
        or args.player_measurement != "outcome_fantasy"
        or args.player_control != "team_only"
    ):
        parser.error(
            "Hybrid predictions require --player-stats-csv, "
            "--player-measurement outcome_fantasy, and --player-control team_only"
        )
    if bool(args.official_player_stats_csv) != bool(args.official_player_lineups_csv):
        parser.error(
            "Use --official-player-stats-csv and --official-player-lineups-csv together"
        )
    hybrid_config = None
    hybrid_diagnostics = None
    selected_team_config = None
    selected_team_diagnostics = None
    selected_team_fits = None
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
            player_config = PlayerModelConfig(
                signal=args.player_signal,
                measurement=args.player_measurement,
                control_model_name=args.player_control,
                rating_prior_games=args.player_rating_prior_games,
            )
            appearances = load_player_matches_csv(args.player_stats_csv, matches)
            lineups = load_lineup_snapshots_csv(args.lineups_csv, fixtures)
            if args.official_player_stats_csv:
                hybrid_config = HybridPlayerConfig(
                    outcome=player_config,
                    official=PlayerModelConfig(
                        signal=args.player_signal,
                        measurement="official_points",
                        control_model_name="team_only",
                        rating_prior_games=args.official_player_rating_prior_games,
                    ),
                )
                official_appearances = load_player_matches_csv(
                    args.official_player_stats_csv, matches
                )
                official_lineups = load_lineup_snapshots_csv(
                    args.official_player_lineups_csv, fixtures
                )
                outcome_history = PlayerHistory(appearances, lineups)
                official_history = PlayerHistory(
                    official_appearances, official_lineups
                )
                player_rows, hybrid_diagnostics = replay_hybrid_player_predictions(
                    matches,
                    predictions,
                    outcome_history,
                    official_history,
                    hybrid_config,
                )
                historical_controls = walk_forward_predictions(matches, 3)
                historical_hybrid, _ = replay_hybrid_player_predictions(
                    matches,
                    historical_controls,
                    PlayerHistory(appearances, []),
                    PlayerHistory(official_appearances, []),
                    hybrid_config,
                )
                selected_team_config = SelectedTeamConfig()
                selected_all, selected_diagnostics_all, selected_team_fits = (
                    replay_selected_team_predictions(
                        matches,
                        historical_controls + predictions,
                        historical_hybrid + player_rows,
                        official_history,
                        selected_team_config,
                    )
                )
                fixture_ids = {fixture.match_id for fixture in fixtures}
                selected_team_rows = [
                    row for row in selected_all if row.match_id in fixture_ids
                ]
                selected_team_diagnostics = [
                    row
                    for row in selected_diagnostics_all
                    if row.match_id in fixture_ids
                ]
                player_config = None
            else:
                player_rows, player_diagnostics = replay_player_predictions(
                    matches, predictions, appearances, lineups, player_config
                )
            predictions.extend(player_rows)
            if args.official_player_stats_csv:
                predictions.extend(selected_team_rows)
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
            args.official_player_stats_csv,
            args.lineups_csv,
            args.official_player_lineups_csv,
        )
        if path
    ]
    write_metadata(
        output,
        paths,
        predictions,
        matches,
        {
            key: value
            for key, value in vars(args).items()
            if args.official_player_stats_csv or not key.startswith("official_player_")
        },
        quotes,
        player_config=player_config,
        player_diagnostics=player_diagnostics,
        hybrid_config=hybrid_config,
        hybrid_diagnostics=hybrid_diagnostics,
        selected_team_config=selected_team_config,
        selected_team_diagnostics=selected_team_diagnostics,
        selected_team_fits=selected_team_fits,
    )
    print(f"Wrote predictions for {len(fixtures)} fixtures to {output}")


if __name__ == "__main__":
    main()
