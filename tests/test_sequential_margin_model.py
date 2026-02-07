import csv
from pathlib import Path

from src.mae_model.sequential_margin import (
    MatchRow,
    ShotVolumeConversionModel,
    load_lineups_csv,
    load_matches_csv,
    parse_match_date,
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
    assert overall["team_plus_lineup"]["mae_margin"] < overall["team_only"]["mae_margin"]


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
