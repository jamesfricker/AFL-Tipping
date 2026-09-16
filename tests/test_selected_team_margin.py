from dataclasses import replace
from datetime import datetime

import pytest

from src.mae_model.data import MatchRow
from src.mae_model.player_margin import PlayerHistory, PlayerId, PlayerMatch, PlayerStats
from src.mae_model.selected_team_margin import (
    SelectedTeamConfig,
    replay_selected_team_predictions,
)
from src.mae_model.sequential_margin import walk_forward_predictions


def selected_history():
    matches = []
    appearances = []
    scores = ((100, 80), (72, 90), (110, 70), (85, 81))
    years = (2018, 2019, 2020, 2020)
    for index, (year, scores_for_match) in enumerate(zip(years, scores)):
        kickoff = datetime.fromisoformat(
            f"{year}-{'03' if index < 3 else '04'}-01T19:00:00+11:00"
        )
        match = MatchRow(
            f"m{index}",
            year,
            str(index + 1),
            kickoff,
            "M.C.G.",
            "A",
            "B",
            *scores_for_match,
            home_scoring_shots=25,
            away_scoring_shots=22,
        )
        matches.append(match)
        for team in ("A", "B"):
            for player in range(22):
                appearances.append(
                    PlayerMatch(
                        match.match_id,
                        team,
                        PlayerId(f"{team}{player}"),
                        match.available_at,
                        100,
                        PlayerStats(),
                        14.0 + (4.0 if team == "A" and player < 4 else 0.0),
                    )
                )
    controls = [
        row
        for row in walk_forward_predictions(matches, 0)
        if row.model_name == "team_only"
    ]
    hybrids = [
        replace(
            row,
            model_name="player_hybrid",
            predicted_margin=row.predicted_margin + (1 if row.match_id != "m1" else -1),
            abs_error=abs(
                row.actual_margin
                - (row.predicted_margin + (1 if row.match_id != "m1" else -1))
            ),
        )
        for row in controls
    ]
    return matches, appearances, controls, hybrids


def selected_replay(matches, appearances, controls, hybrids):
    return replay_selected_team_predictions(
        matches,
        controls,
        hybrids,
        PlayerHistory(appearances, []),
        SelectedTeamConfig(
            rating_games=2,
            rating_prior_games=1,
            minimum_training_games=1,
        ),
    )


def test_current_match_rating_cannot_change_its_own_prediction():
    matches, appearances, controls, hybrids = selected_history()
    rows, diagnostics, _ = selected_replay(
        matches, appearances, controls, hybrids
    )
    changed = [
        replace(row, official_rating_points=10000)
        if row.match_id == "m3"
        else row
        for row in appearances
    ]
    changed_rows, changed_diagnostics, _ = selected_replay(
        matches, changed, controls, hybrids
    )
    assert rows[-1].model_name == "selected_team_strength"
    assert rows[-1].predicted_margin == pytest.approx(-7.1833595084)
    assert diagnostics[-1].status == "adjusted"
    assert changed_rows[-1] == rows[-1]
    assert changed_diagnostics[-1] == diagnostics[-1]


def test_missing_selected_team_returns_the_exact_hybrid_prediction():
    matches, appearances, controls, hybrids = selected_history()
    appearances = [row for row in appearances if row.match_id != "m3"]
    rows, diagnostics, _ = selected_replay(
        matches, appearances, controls, hybrids
    )
    assert rows[-1] == replace(
        hybrids[-1],
        model_name="selected_team_strength",
        used_fallback=True,
    )
    assert diagnostics[-1].status == "no_lineup"


def test_same_season_result_does_not_refit_frozen_weights():
    matches, appearances, controls, hybrids = selected_history()
    rows, _, _ = selected_replay(matches, appearances, controls, hybrids)
    changed_controls = [
        replace(row, actual_margin=1000, abs_error=1000)
        if row.match_id == "m2"
        else row
        for row in controls
    ]
    changed_rows, _, _ = selected_replay(
        matches, appearances, changed_controls, hybrids
    )
    assert rows[-1].predicted_margin == pytest.approx(-7.1833595084)
    assert changed_rows[-1].predicted_margin == rows[-1].predicted_margin
