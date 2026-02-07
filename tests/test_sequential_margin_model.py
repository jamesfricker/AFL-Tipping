import csv
from pathlib import Path

import pandas as pd

from src.mae_model.sequential_margin import (
    BenchmarkShotVolumeModel,
    MatchRow,
    MatchContextRow,
    SequentialMarginModel,
    ShotVolumeConversionModel,
    TerritoryShotChainModel,
    VenueEnvironmentAdjustmentModel,
    load_match_context_csv,
    load_lineups_csv,
    load_market_xlsx,
    load_matches_csv,
    parse_match_date,
    _market_implied_home_probability,
    _market_sigma_from_spread_and_probability,
    _recent_history_rows,
    summarize_predictions,
    walk_forward_predictions,
)


def _write_synthetic_matches(path: Path):
    headers = [
        "match_id",
        "year",
        "round",
        "date",
        "venue",
        "time",
        "home_team_name",
        "home_team_score",
        "away_team_name",
        "away_team_score",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        game_num = 1
        for year in range(2018, 2023):
            for rnd in range(1, 21):
                is_even = rnd % 2 == 0
                home_team = "B"
                away_team = "A"
                if not is_even:
                    home_team = "A"
                    away_team = "B"

                a_star = rnd % 3 != 0
                b_star = rnd % 4 != 0
                home_adv = 5 if home_team == "A" else -5
                player_delta = (28 if a_star else 0) - (28 if b_star else 0)
                noise = 0
                margin = home_adv + player_delta + noise

                home_score = 80 + (margin / 2.0)
                away_score = 80 - (margin / 2.0)
                month = ((rnd - 1) % 9) + 1
                day = ((rnd - 1) % 27) + 1
                date = f"{day:02d}-{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'][month-1]}-{year}"

                writer.writerow(
                    {
                        "match_id": f"{year}R{rnd:02d}",
                        "year": year,
                        "round": f"R{rnd}",
                        "date": date,
                        "venue": "Synthetic Oval",
                        "time": "1:00 PM",
                        "home_team_name": home_team,
                        "home_team_score": f"{home_score:.1f}",
                        "away_team_name": away_team,
                        "away_team_score": f"{away_score:.1f}",
                    }
                )
                game_num += 1


def _write_synthetic_lineups(path: Path):
    headers = [
        "match_id",
        "team_name",
        "player_name",
        "percent_played",
        "disposals",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for year in range(2018, 2023):
            for rnd in range(1, 21):
                match_id = f"{year}R{rnd:02d}"
                a_star = rnd % 3 != 0
                b_star = rnd % 4 != 0

                a_players = ["A_role_1", "A_role_2", "A_role_3"]
                b_players = ["B_role_1", "B_role_2", "B_role_3"]
                for idx in range(4, 11):
                    a_players.append(f"A_role_{idx}")
                    b_players.append(f"B_role_{idx}")
                if a_star:
                    a_players.append("A_star")
                if b_star:
                    b_players.append("B_star")

                for player in a_players:
                    writer.writerow(
                        {
                            "match_id": match_id,
                            "team_name": "A",
                            "player_name": player,
                            "percent_played": 96 if "star" in player else 42,
                            "disposals": 45 if "star" in player else 2,
                        }
                    )

                for player in b_players:
                    writer.writerow(
                        {
                            "match_id": match_id,
                            "team_name": "B",
                            "player_name": player,
                            "percent_played": 96 if "star" in player else 42,
                            "disposals": 45 if "star" in player else 2,
                        }
                    )


def _write_synthetic_context(path: Path):
    headers = [
        "date",
        "home_team",
        "away_team",
        "venue",
        "weather_temp_c",
        "weather_rain_mm",
        "weather_wind_kmh",
        "weather_humidity_pct",
        "attendance",
        "projected_attendance",
        "venue_length_m",
        "venue_width_m",
        "venue_capacity",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for year in range(2018, 2023):
            for rnd in range(1, 21):
                is_even = rnd % 2 == 0
                home_team = "B" if is_even else "A"
                away_team = "A" if is_even else "B"
                month = ((rnd - 1) % 9) + 1
                day = ((rnd - 1) % 27) + 1
                date = f"{day:02d}-{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'][month-1]}-{year}"
                writer.writerow(
                    {
                        "date": date,
                        "home_team": home_team,
                        "away_team": away_team,
                        "venue": "Synthetic Oval",
                        "weather_temp_c": 15.0 + (rnd % 10),
                        "weather_rain_mm": float(rnd % 5),
                        "weather_wind_kmh": 10.0 + (rnd % 7),
                        "weather_humidity_pct": 45.0 + (rnd % 25),
                        "attendance": 21000 + (rnd * 300),
                        "projected_attendance": 20500 + (rnd * 280),
                        "venue_length_m": 165.0,
                        "venue_width_m": 135.0,
                        "venue_capacity": 52000.0,
                    }
                )


def test_team_plus_lineup_beats_team_only(tmp_path):
    matches_path = tmp_path / "matches.csv"
    players_path = tmp_path / "players.csv"
    _write_synthetic_matches(matches_path)
    _write_synthetic_lineups(players_path)

    matches = load_matches_csv(str(matches_path))
    lineups = load_lineups_csv(str(players_path))
    preds = walk_forward_predictions(matches, lineups, min_train_years=1)
    summary = summarize_predictions(preds)

    overall = {row["model_name"]: row for row in summary if row["year"] == "ALL"}
    assert {"team_only", "team_plus_lineup"}.issubset(set(overall.keys()))
    assert overall["team_plus_lineup"]["mae_margin"] <= overall["team_only"]["mae_margin"] + 0.2


def test_hybrid_predict_does_not_use_same_match_outcomes():
    model = ShotVolumeConversionModel()
    lineups = {
        ("M1", "A"): [{"player_name": "A1", "goals": 0.0, "behinds": 0.0}],
        ("M1", "B"): [{"player_name": "B1", "goals": 0.0, "behinds": 0.0}],
    }
    baseline_match = MatchRow(
        match_id="M1",
        year=2026,
        round_label="R1",
        date=parse_match_date("01-Jan-2026"),
        venue="Test Oval",
        home_team="A",
        away_team="B",
        home_score=0.0,
        away_score=0.0,
        home_scoring_shots=0,
        away_scoring_shots=0,
    )
    changed_outcome_match = MatchRow(
        match_id="M1",
        year=2026,
        round_label="R1",
        date=parse_match_date("01-Jan-2026"),
        venue="Test Oval",
        home_team="A",
        away_team="B",
        home_score=180.0,
        away_score=20.0,
        home_scoring_shots=45,
        away_scoring_shots=8,
    )

    baseline_prediction = model.predict(baseline_match, lineups)
    changed_prediction = model.predict(changed_outcome_match, lineups)
    assert baseline_prediction == changed_prediction


def test_benchmark_shot_volume_predict_does_not_use_same_match_outcomes():
    model = BenchmarkShotVolumeModel()
    baseline_match = MatchRow(
        match_id="M2",
        year=2026,
        round_label="R1",
        date=parse_match_date("01-Jan-2026"),
        venue="Test Oval",
        home_team="A",
        away_team="B",
        home_score=0.0,
        away_score=0.0,
        home_scoring_shots=0,
        away_scoring_shots=0,
    )
    changed_outcome_match = MatchRow(
        match_id="M2",
        year=2026,
        round_label="R1",
        date=parse_match_date("01-Jan-2026"),
        venue="Test Oval",
        home_team="A",
        away_team="B",
        home_score=180.0,
        away_score=20.0,
        home_scoring_shots=45,
        away_scoring_shots=8,
    )

    baseline_prediction = model.predict(baseline_match)
    changed_prediction = model.predict(changed_outcome_match)
    assert baseline_prediction == changed_prediction


def test_territory_chain_predict_does_not_use_same_match_outcomes():
    model = TerritoryShotChainModel()
    baseline_match = MatchRow(
        match_id="M4",
        year=2026,
        round_label="R1",
        date=parse_match_date("01-Jan-2026"),
        venue="Test Oval",
        home_team="A",
        away_team="B",
        home_score=0.0,
        away_score=0.0,
        home_scoring_shots=0,
        away_scoring_shots=0,
    )
    changed_outcome_match = MatchRow(
        match_id="M4",
        year=2026,
        round_label="R1",
        date=parse_match_date("01-Jan-2026"),
        venue="Test Oval",
        home_team="A",
        away_team="B",
        home_score=180.0,
        away_score=20.0,
        home_scoring_shots=45,
        away_scoring_shots=8,
    )

    baseline_prediction = model.predict(baseline_match)
    changed_prediction = model.predict(changed_outcome_match)
    assert baseline_prediction == changed_prediction


def test_team_plus_lineup_predict_does_not_use_same_match_playtime_or_disposals():
    model = SequentialMarginModel(use_lineups=True, min_player_games=0, lineup_scale=8.0)
    model.player_rating["A1"] = 0.3
    model.player_rating["A2"] = -0.1
    model.player_rating["B1"] = -0.2
    model.player_rating["B2"] = 0.4
    model.player_games["A1"] = 5
    model.player_games["A2"] = 5
    model.player_games["B1"] = 5
    model.player_games["B2"] = 5

    match = MatchRow(
        match_id="M3",
        year=2026,
        round_label="R1",
        date=parse_match_date("01-Jan-2026"),
        venue="Test Oval",
        home_team="A",
        away_team="B",
        home_score=0.0,
        away_score=0.0,
    )

    lineups_low_stats = {
        ("M3", "A"): [
            {"player_name": "A1", "percent_played": 1.0, "disposals": 0.0},
            {"player_name": "A2", "percent_played": 1.0, "disposals": 0.0},
        ],
        ("M3", "B"): [
            {"player_name": "B1", "percent_played": 1.0, "disposals": 0.0},
            {"player_name": "B2", "percent_played": 1.0, "disposals": 0.0},
        ],
    }
    lineups_high_stats = {
        ("M3", "A"): [
            {"player_name": "A1", "percent_played": 100.0, "disposals": 45.0},
            {"player_name": "A2", "percent_played": 100.0, "disposals": 45.0},
        ],
        ("M3", "B"): [
            {"player_name": "B1", "percent_played": 100.0, "disposals": 45.0},
            {"player_name": "B2", "percent_played": 100.0, "disposals": 45.0},
        ],
    }

    low_prediction = model.predict(match, lineups_low_stats)
    high_prediction = model.predict(match, lineups_high_stats)
    assert low_prediction == high_prediction


def test_context_model_predict_does_not_use_same_match_actual_attendance():
    model = VenueEnvironmentAdjustmentModel()
    match = MatchRow(
        match_id="M5",
        year=2026,
        round_label="R1",
        date=parse_match_date("01-Jan-2026"),
        venue="Test Oval",
        home_team="A",
        away_team="B",
        home_score=0.0,
        away_score=0.0,
    )
    context_low = MatchContextRow(
        date=match.date.date(),
        home_team="A",
        away_team="B",
        venue="Test Oval",
        weather_temp_c=20.0,
        weather_rain_mm=0.0,
        weather_wind_kmh=12.0,
        weather_humidity_pct=50.0,
        attendance=5000.0,
        venue_length_m=160.0,
        venue_width_m=130.0,
        venue_capacity=30000.0,
    )
    context_high = MatchContextRow(
        date=match.date.date(),
        home_team="A",
        away_team="B",
        venue="Test Oval",
        weather_temp_c=20.0,
        weather_rain_mm=0.0,
        weather_wind_kmh=12.0,
        weather_humidity_pct=50.0,
        attendance=65000.0,
        venue_length_m=160.0,
        venue_width_m=130.0,
        venue_capacity=30000.0,
    )

    low_prediction = model.predict(match, base_margin=3.0, context=context_low)
    high_prediction = model.predict(match, base_margin=3.0, context=context_high)
    assert low_prediction == high_prediction


def test_load_match_context_csv_applies_aliases_and_fields(tmp_path):
    context_path = tmp_path / "context.csv"
    with context_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Date",
                "Home Team",
                "Away Team",
                "Venue",
                "temperature_c",
                "rain_mm",
                "wind_speed_kmh",
                "relative_humidity_pct",
                "crowd",
                "expected_attendance",
                "ground_length_m",
                "ground_width_m",
                "stadium_capacity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Date": "2025-09-27",
                "Home Team": "Brisbane",
                "Away Team": "GWS Giants",
                "Venue": "The Gabba",
                "temperature_c": 23.5,
                "rain_mm": 1.2,
                "wind_speed_kmh": 19.0,
                "relative_humidity_pct": 68.0,
                "crowd": 32910,
                "expected_attendance": 31500,
                "ground_length_m": 156,
                "ground_width_m": 138,
                "stadium_capacity": 42000,
            }
        )

    context = load_match_context_csv(str(context_path))
    key = (pd.Timestamp("2025-09-27").date(), "Brisbane Lions", "Greater Western Sydney")
    assert key in context
    row = context[key]
    assert row.weather_temp_c == 23.5
    assert row.weather_rain_mm == 1.2
    assert row.weather_wind_kmh == 19.0
    assert row.weather_humidity_pct == 68.0
    assert row.attendance == 32910.0
    assert row.projected_attendance == 31500.0
    assert row.venue_length_m == 156.0
    assert row.venue_width_m == 138.0
    assert row.venue_capacity == 42000.0


def test_walk_forward_adds_team_context_env_model(tmp_path):
    matches_path = tmp_path / "matches.csv"
    players_path = tmp_path / "players.csv"
    context_path = tmp_path / "context.csv"
    _write_synthetic_matches(matches_path)
    _write_synthetic_lineups(players_path)
    _write_synthetic_context(context_path)

    matches = load_matches_csv(str(matches_path))
    lineups = load_lineups_csv(str(players_path))
    context_data = load_match_context_csv(str(context_path))
    preds = walk_forward_predictions(
        matches=matches,
        lineups=lineups,
        min_train_years=1,
        match_context_data=context_data,
    )

    model_names = {row.model_name for row in preds}
    assert "team_context_env" in model_names


def test_recent_history_rows_filters_to_recent_years():
    history = [
        {"year": 2020, "value": 1},
        {"year": 2022, "value": 2},
        {"year": 2024, "value": 3},
    ]
    filtered = _recent_history_rows(history, current_year=2025, recent_years=2)
    assert [row["year"] for row in filtered] == [2024]


def test_market_probability_removes_vig_and_bounds():
    p = _market_implied_home_probability(1.80, 2.20)
    assert p is not None
    assert 0.5 < p < 1.0
    assert _market_implied_home_probability(None, 2.0) is None
    assert _market_implied_home_probability(0.9, 2.0) is None


def test_market_sigma_estimate_from_line_and_probability():
    sigma = _market_sigma_from_spread_and_probability(12.0, 0.70)
    assert 12.0 <= sigma <= 80.0
    fallback = _market_sigma_from_spread_and_probability(None, 0.70)
    assert fallback == 30.0


def test_load_market_xlsx_applies_team_aliases(tmp_path):
    xlsx_path = tmp_path / "market.xlsx"
    frame = pd.DataFrame(
        [
            {
                "Date": "2025-09-27",
                "Home Team": "Brisbane",
                "Away Team": "GWS Giants",
                "Home Line Close": -4.5,
                "Home Odds": 1.7,
                "Away Odds": 2.1,
            }
        ]
    )
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Data", index=False, startrow=1)

    market = load_market_xlsx(str(xlsx_path))
    key = (pd.Timestamp("2025-09-27").date(), "Brisbane Lions", "Greater Western Sydney")
    assert key in market
    assert market[key]["home_line_close"] == -4.5


def test_walk_forward_adds_market_only_and_residual_models(tmp_path):
    matches_path = tmp_path / "matches.csv"
    players_path = tmp_path / "players.csv"
    _write_synthetic_matches(matches_path)
    _write_synthetic_lineups(players_path)

    matches = load_matches_csv(str(matches_path))
    lineups = load_lineups_csv(str(players_path))
    preds = walk_forward_predictions(matches, lineups, min_train_years=1)
    model_names = {row.model_name for row in preds}
    assert "market_only" in model_names
    assert "market_residual_corrector" in model_names
