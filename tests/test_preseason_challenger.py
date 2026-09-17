from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.mae_model.data import MatchRow
from src.mae_model.preseason_challenger import (
    PreseasonResult,
    load_preseason_results_csv,
    predict_preseason_challenger,
    replay_preseason_challenger,
)


ZONE = ZoneInfo("Australia/Melbourne")


def match(match_id, year, home, away, home_score, away_score):
    return MatchRow(
        match_id,
        year,
        "1",
        datetime(year, 3, 20, 19, 30, tzinfo=ZONE),
        "M.C.G.",
        home,
        away,
        home_score,
        away_score,
        home_goals=home_score // 6,
        home_behinds=home_score % 6,
        away_goals=away_score // 6,
        away_behinds=away_score % 6,
    )


def history():
    return [
        match("2012", 2012, "Carlton", "Richmond", 90, 70),
        match("2013", 2013, "Richmond", "Carlton", 80, 75),
        match("2014", 2014, "Carlton", "Richmond", 65, 85),
        match("2015", 2015, "Carlton", "Richmond", 88, 74),
    ]


def preseason(kickoff):
    return PreseasonResult(
        "pre-2015",
        2015,
        kickoff,
        "Carlton",
        "Richmond",
        16,
        10,
        5,
        5,
    )


def test_replay_uses_only_preseason_results_before_kickoff():
    rows = history()
    target = rows[-1].fixture.kickoff
    baseline = replay_preseason_challenger(rows, [], first_prediction_year=2015)[0]
    future = replay_preseason_challenger(
        rows,
        [preseason(datetime(2015, 3, 21, 12, tzinfo=ZONE))],
        first_prediction_year=2015,
    )[0]
    prior = replay_preseason_challenger(
        rows,
        [preseason(datetime(2015, 2, 21, 12, tzinfo=ZONE))],
        first_prediction_year=2015,
    )[0]

    assert future.predicted_margin == pytest.approx(baseline.predicted_margin)
    assert prior.predicted_margin > baseline.predicted_margin
    assert prior.cutoff == target


def test_load_preseason_results_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "preseason.csv"
    path.write_text(
        "provider_id,year,kickoff,home_team,home_goals,home_behinds,"
        "away_team,away_goals,away_behinds\n"
        "same,2015,2015-02-01T12:00:00+11:00,Carlton,10,5,Richmond,8,4\n"
        "same,2015,2015-02-02T12:00:00+11:00,Richmond,8,4,Carlton,10,5\n"
    )

    with pytest.raises(ValueError, match="Duplicate preseason provider ID"):
        load_preseason_results_csv(str(path))


def test_live_prediction_matches_walk_forward_prediction():
    rows = history()
    fixture = rows[-1].fixture
    prior = preseason(datetime(2015, 2, 21, 12, tzinfo=ZONE))
    as_of = datetime(2015, 3, 1, 12, tzinfo=ZONE)
    live = predict_preseason_challenger(
        rows[:-1], [fixture], [prior], as_of
    )[0]
    replay = replay_preseason_challenger(
        rows, [prior], first_prediction_year=2015
    )[0]

    assert live.predicted_margin == pytest.approx(replay.predicted_margin)
    assert live.cutoff == as_of
