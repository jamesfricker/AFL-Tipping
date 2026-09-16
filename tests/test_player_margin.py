import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from src.mae_model.data import MatchRow
from src.mae_model.player_margin import (
    PlayerId,
    PlayerMatch,
    PlayerModelConfig,
    PlayerStats,
    derive_historical_lineups,
    load_lineup_snapshots_csv,
    load_player_matches_csv,
    replay_player_predictions,
)
from src.mae_model.predict_fixtures import main as predict_main
from src.mae_model.run_backtest import main as backtest_main
from src.mae_model.sequential_margin import (
    predict_fixtures,
    walk_forward_predictions,
    write_prediction_rows,
)


def write_csv(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_history(count=13, *, year=2019, missing_leader=True):
    matches = []
    appearances = []
    for index in range(count):
        kickoff = datetime.fromisoformat(f"{year}-03-01T19:00:00+11:00") + timedelta(
            days=7 * index
        )
        match = MatchRow(
            f"m{index}",
            kickoff.year,
            str(index + 1),
            kickoff,
            "M.C.G.",
            "A",
            "B",
            110,
            80,
            home_scoring_shots=28,
            away_scoring_shots=21,
        )
        matches.append(match)
        for team in ("A", "B"):
            for player in range(22):
                identity = f"{team}{player}"
                if missing_leader and index == count - 1 and identity == "A0":
                    identity = "replacement"
                appearances.append(
                    PlayerMatch(
                        match.match_id,
                        team,
                        PlayerId(identity),
                        match.available_at,
                        100 if identity == "A0" else 20,
                        PlayerStats(
                            kicks=10 + (index if identity == "A0" else 0), tackles=3
                        ),
                    )
                )
    return matches, appearances


def player_replay(matches, appearances, *, config=None, controls=None, lineups=None):
    controls = (
        controls if controls is not None else walk_forward_predictions(matches, 0)
    )
    return replay_player_predictions(
        matches, controls, appearances, lineups or [], config or PlayerModelConfig()
    )


def target_result(matches, appearances, **kwargs):
    rows, diagnostics = player_replay(matches, appearances, **kwargs)
    return rows[-1], diagnostics[-1]


def test_current_match_statistics_do_not_change_its_forecast():
    matches, appearances = sample_history()
    before = target_result(matches, appearances)
    changed = [
        replace(row, percent_played=100, stats=PlayerStats(kicks=10000, goals=1000))
        if row.match_id == matches[-1].match_id
        else row
        for row in appearances
    ]
    assert before == target_result(matches, changed)
    assert before[1].status == "adjusted"
    assert before[1].correction < 0


def test_future_statistics_results_and_lineups_do_not_change_prior_forecasts():
    matches, appearances = sample_history(14)
    controls = walk_forward_predictions(matches[:-1], 0)
    before = player_replay(matches, appearances, controls=controls)
    future = replace(matches[-1], home_score=0, away_score=200)
    changed = [
        replace(
            row,
            player_id=PlayerId("future-" + row.player_id),
            stats=PlayerStats(kicks=9000),
        )
        if row.match_id == future.match_id
        else row
        for row in appearances
    ]
    assert before == player_replay(matches[:-1] + [future], changed, controls=controls)


def test_missing_leading_regular_reduces_home_margin_and_reports_identity():
    matches, appearances = sample_history()
    row, diagnostic = target_result(matches, appearances)
    control = walk_forward_predictions(matches, 0)[-1]
    assert diagnostic.status == "adjusted"
    assert diagnostic.lineup.home.missing_leader == "A0"
    assert diagnostic.lineup.home.missing_leader_gap > 0.75
    assert -4 <= row.predicted_margin - control.predicted_margin < 0


@pytest.mark.parametrize("signal", ["rating", "form", "missing_leader", "rating_form"])
def test_all_signals_use_the_material_change_gate(signal):
    matches, appearances = sample_history(missing_leader=False)
    rows, diagnostics = player_replay(
        matches, appearances, config=PlayerModelConfig(signal=signal)
    )
    controls = [
        row
        for row in walk_forward_predictions(matches, 0)
        if row.model_name == "market_scoring_blend"
    ]
    assert [row.predicted_margin for row in rows] == [
        row.predicted_margin for row in controls
    ]
    assert diagnostics[-1].status == "stable_lineup"
    assert diagnostics[-1].correction == 0


def test_missing_lineup_returns_exact_control_margin():
    matches, appearances = sample_history()
    appearances = [row for row in appearances if row.match_id != matches[-1].match_id]
    row, diagnostic = target_result(matches, appearances)
    control = walk_forward_predictions(matches, 0)[-1]
    assert row == replace(control, model_name="player_lineup", used_fallback=True)
    assert diagnostic.status == "no_lineup"


def test_pre_2018_rows_return_exact_control_margin():
    matches, appearances = sample_history(year=2017)
    row, diagnostic = target_result(matches, appearances)
    control = walk_forward_predictions(matches, 0)[-1]
    assert row == replace(control, model_name="player_lineup", used_fallback=True)
    assert diagnostic.status == "insufficient_history"


def test_low_player_coverage_returns_exact_control_margin():
    matches, appearances = sample_history()
    changed = [
        replace(row, player_id=PlayerId("new-" + row.player_id))
        if row.match_id == matches[-1].match_id
        else row
        for row in appearances
    ]
    row, diagnostic = target_result(matches, changed)
    control = walk_forward_predictions(matches, 0)[-1]
    assert row.predicted_margin == control.predicted_margin
    assert diagnostic.status == "low_coverage"
    assert diagnostic.lineup.home.missing_leader == "A0"


def test_correction_has_a_four_point_cap():
    matches, appearances = sample_history()
    row, diagnostic = target_result(
        matches, appearances, config=PlayerModelConfig(rating_weight=100)
    )
    assert diagnostic.correction == -4
    assert (
        row.predicted_margin
        == walk_forward_predictions(matches, 0)[-1].predicted_margin - 4
    )


def test_duplicate_updates_are_idempotent_and_changed_sources_fail():
    matches, appearances = sample_history()
    controls = walk_forward_predictions(matches, 0)
    expected = player_replay(matches, appearances, controls=controls)
    assert expected == player_replay(
        matches + matches, appearances + appearances, controls=controls
    )
    with pytest.raises(ValueError, match="Changed source data for player appearance"):
        player_replay(
            matches, appearances + [replace(appearances[0], stats=PlayerStats(goals=9))]
        )
    with pytest.raises(ValueError, match="Changed source data for match result"):
        player_replay(
            matches + [replace(matches[0], home_score=1)],
            appearances,
            controls=controls,
        )


def test_live_and_historical_paths_agree_before_the_target_result():
    matches, appearances = sample_history()
    historical, historical_diagnostic = target_result(matches, appearances)
    target = matches[-1]
    cutoff = target.fixture.kickoff - timedelta(minutes=1)
    live_lineups = [
        replace(row, observed_at=cutoff, source="timed_selection")
        for row in derive_historical_lineups(appearances, matches)
        if row.match_id == target.match_id
    ]
    controls = predict_fixtures(matches[:-1], [target.fixture], cutoff)
    past_players = [row for row in appearances if row.match_id != target.match_id]
    live, diagnostic = target_result(
        matches[:-1], past_players, controls=controls, lineups=live_lineups
    )
    assert live.predicted_margin == historical.predicted_margin
    assert diagnostic.lineup == historical_diagnostic.lineup
    assert diagnostic.correction == historical_diagnostic.correction
    assert live.actual_margin is None
    assert live.abs_error is None


def test_live_uses_latest_complete_snapshot_available_at_cutoff():
    matches, appearances = sample_history()
    target = matches[-1]
    cutoff = target.fixture.kickoff - timedelta(hours=1)
    complete = [
        replace(row, observed_at=cutoff, source="timed_selection")
        for row in derive_historical_lineups(appearances, matches)
        if row.match_id == target.match_id
    ]
    future = [
        replace(row, observed_at=cutoff + timedelta(minutes=1)) for row in complete
    ]
    incomplete = [
        replace(row, player_ids=row.player_ids[:21], observed_at=cutoff)
        for row in complete
    ]
    controls = predict_fixtures(matches[:-1], [target.fixture], cutoff)
    past_players = [row for row in appearances if row.match_id != target.match_id]
    absent, diagnostic = target_result(
        matches[:-1], past_players, controls=controls, lineups=future + incomplete
    )
    assert diagnostic.status == "no_lineup"
    assert absent.predicted_margin == controls[-1].predicted_margin
    older = [
        replace(row, observed_at=cutoff - timedelta(minutes=2)) for row in complete
    ]
    found = target_result(
        matches[:-1],
        past_players,
        controls=controls,
        lineups=older + future + incomplete,
    )
    expected = target_result(
        matches[:-1], past_players, controls=controls, lineups=older
    )
    assert found == expected
    assert found[1].status == "adjusted"


def test_result_at_request_time_is_available_but_later_statistics_are_not():
    matches, appearances = sample_history()
    target = matches[-1]
    prior = replace(matches[-2], result_available_at=target.fixture.kickoff)
    changed_matches = matches[:-2] + [prior, target]
    changed_players = [
        replace(row, statistics_available_at=prior.available_at)
        if row.match_id == prior.match_id
        else row
        for row in appearances
    ]
    available = target_result(changed_matches, changed_players)
    delayed = [
        replace(
            row, statistics_available_at=target.fixture.kickoff + timedelta(seconds=1)
        )
        if row.match_id == prior.match_id
        else row
        for row in changed_players
    ]
    unavailable = target_result(changed_matches, delayed)
    assert (
        available[1].lineup.home.missing_leader_gap
        > unavailable[1].lineup.home.missing_leader_gap
    )
    changed_future = [
        replace(row, stats=PlayerStats(kicks=99999))
        if row.match_id == prior.match_id
        else row
        for row in delayed
    ]
    assert unavailable == target_result(changed_matches, changed_future)


def test_player_rows_leave_control_objects_and_csv_bytes_unchanged(tmp_path):
    matches, appearances = sample_history()
    controls = walk_forward_predictions(matches, 0)
    original = list(controls)
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    write_prediction_rows(str(before), controls)
    players, _ = player_replay(matches, appearances, controls=controls)
    write_prediction_rows(str(after), controls + players)
    assert controls == original
    assert after.read_bytes().startswith(before.read_bytes())
    assert len(players) == len(matches)


def appearance_csv(row):
    return {
        "match_id": row.match_id,
        "team_name": row.team,
        "player_ref": row.player_id,
        "player_name": "Same name",
        "percent_played": row.percent_played,
        "kicks": row.stats.kicks,
        "tackles": row.stats.tackles,
        "statistics_available_at": row.statistics_available_at.isoformat(),
    }


def test_player_csv_boundary_checks_and_ignores_brownlow_votes(tmp_path):
    matches, appearances = sample_history(1)
    path = tmp_path / "players.csv"
    source = appearance_csv(appearances[0])
    write_csv(path, [{**source, "brownlow_votes": "future-value"}])
    assert load_player_matches_csv(str(path), matches) == [appearances[0]]
    second = appearance_csv(appearances[1])
    write_csv(path, [source, second])
    assert {row.player_id for row in load_player_matches_csv(str(path), matches)} == {
        appearances[0].player_id,
        appearances[1].player_id,
    }
    write_csv(path, [source, source])
    assert load_player_matches_csv(str(path), matches) == [appearances[0]]
    for change, error in [
        ({"percent_played": "nan"}, "finite"),
        ({"percent_played": 101}, "exceed 100"),
        ({"player_ref": ""}, "reference"),
        ({"team_name": "C"}, "not in match"),
        ({"match_id": "unknown"}, "Unknown"),
        (
            {"statistics_available_at": matches[0].fixture.kickoff.isoformat()},
            "precede result",
        ),
    ]:
        write_csv(path, [{**source, **change}])
        with pytest.raises(ValueError, match=error):
            load_player_matches_csv(str(path), matches)


def test_lineup_csv_requires_dated_unique_identity(tmp_path):
    matches, _ = sample_history(1)
    target = matches[0].fixture
    path = tmp_path / "lineups.csv"
    source = {
        "match_id": target.match_id,
        "team_name": "A",
        "player_ref": "ref",
        "observed_at": (target.kickoff - timedelta(hours=1)).isoformat(),
    }
    write_csv(path, [source])
    assert load_lineup_snapshots_csv(str(path), [target])[0].player_ids == ("ref",)
    write_csv(path, [source, source])
    with pytest.raises(ValueError, match="Duplicate"):
        load_lineup_snapshots_csv(str(path), [target])
    write_csv(path, [{**source, "observed_at": "2019-03-01T17:00:00"}])
    with pytest.raises(ValueError, match="timezone offset"):
        load_lineup_snapshots_csv(str(path), [target])


def match_csv(match):
    return {
        "match_id": match.match_id,
        "year": match.year,
        "round": match.round_label,
        "kickoff": match.fixture.kickoff.isoformat(),
        "venue": match.venue,
        "home_team_name": match.home_team,
        "away_team_name": match.away_team,
        "home_team_score": match.home_score,
        "away_team_score": match.away_score,
        "home_scoring_shots": match.home_scoring_shots,
        "away_scoring_shots": match.away_scoring_shots,
    }


def test_backtest_command_preserves_control_file_and_records_player_provenance(
    tmp_path,
):
    matches, appearances = sample_history()
    history = tmp_path / "history.csv"
    players = tmp_path / "players.csv"
    write_csv(history, [match_csv(match) for match in matches])
    write_csv(players, [appearance_csv(row) for row in appearances])
    common = ["--matches-csv", str(history), "--min-train-years", "0"]
    base = tmp_path / "base"
    added = tmp_path / "added"
    backtest_main(common + ["--output-dir", str(base)])
    backtest_main(
        common + ["--output-dir", str(added), "--player-stats-csv", str(players)]
    )
    assert (
        (added / "walk_forward_predictions.csv")
        .read_bytes()
        .startswith((base / "walk_forward_predictions.csv").read_bytes())
    )
    metadata = json.loads((added / "metadata.json").read_text())
    assert len(metadata["inputs"]) == 2
    assert metadata["player_model"]["configuration"]["signal"] == "rating_form"
    assert metadata["player_model"]["status_counts"]["adjusted"] == 1
    assert (added / "player_diagnostics.csv").is_file()
    assert "player_model" not in json.loads((base / "metadata.json").read_text())


def test_live_command_requires_both_player_inputs_and_backtest_requires_kickoff(
    tmp_path,
):
    for supplied in ("--player-stats-csv", "--lineups-csv"):
        with pytest.raises(SystemExit) as error:
            predict_main(
                [
                    "--fixtures-csv",
                    "unused",
                    "--as-of",
                    "2019-03-01T17:00:00+11:00",
                    supplied,
                    "unused",
                ]
            )
        assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        backtest_main(["--player-stats-csv", "unused", "--lead-hours", "1"])
    assert error.value.code == 2


def test_live_command_writes_player_rows_for_complete_timed_selections(tmp_path):
    matches, appearances = sample_history()
    target = matches[-1]
    cutoff = target.fixture.kickoff - timedelta(hours=1)
    history = tmp_path / "history.csv"
    players = tmp_path / "players.csv"
    fixtures = tmp_path / "fixtures.csv"
    lineups = tmp_path / "lineups.csv"
    write_csv(history, [match_csv(match) for match in matches[:-1]])
    write_csv(
        players,
        [appearance_csv(row) for row in appearances if row.match_id != target.match_id],
    )
    write_csv(
        fixtures,
        [
            {
                key: value
                for key, value in match_csv(target).items()
                if key
                not in (
                    "home_team_score",
                    "away_team_score",
                    "home_scoring_shots",
                    "away_scoring_shots",
                )
            }
        ],
    )
    write_csv(
        lineups,
        [
            {
                "match_id": row.match_id,
                "team_name": row.team,
                "player_ref": row.player_id,
                "player_name": "Same name",
                "observed_at": cutoff.isoformat(),
            }
            for row in appearances
            if row.match_id == target.match_id
        ],
    )
    output = tmp_path / "live"
    predict_main(
        [
            "--matches-csv",
            str(history),
            "--fixtures-csv",
            str(fixtures),
            "--player-stats-csv",
            str(players),
            "--lineups-csv",
            str(lineups),
            "--as-of",
            cutoff.isoformat(),
            "--output-dir",
            str(output),
        ]
    )
    with (output / "fixture_predictions.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 5
    assert rows[-1]["model_name"] == "player_lineup"
    assert float(rows[-1]["predicted_margin"]) < float(rows[-2]["predicted_margin"])
    assert rows[-1]["actual_margin"] == ""
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["inputs"]) == 4
    assert metadata["player_model"]["status_counts"] == {"adjusted": 1}


def test_historical_snapshot_cannot_be_backdated():
    matches, appearances = sample_history()
    snapshot = derive_historical_lineups(appearances, matches)[-1]
    backdated = replace(snapshot, observed_at=snapshot.observed_at - timedelta(hours=1))
    with pytest.raises(ValueError, match="only at kickoff"):
        player_replay(matches, appearances, lineups=[backdated])


@pytest.mark.parametrize(
    "size, expected",
    [(21, "no_lineup"), (22, "adjusted"), (23, "adjusted"), (24, "no_lineup")],
)
def test_live_selection_accepts_only_22_or_23_players(size, expected):
    matches, appearances = sample_history()
    target = matches[-1]
    cutoff = target.fixture.kickoff - timedelta(hours=1)
    selected = []
    for row in derive_historical_lineups(appearances, matches):
        if row.match_id != target.match_id:
            continue
        players = row.player_ids[:size] + tuple(
            PlayerId(f"{row.team}-new-{index}") for index in range(22, size)
        )
        selected.append(
            replace(
                row, player_ids=players, observed_at=cutoff, source="timed_selection"
            )
        )
    controls = predict_fixtures(matches[:-1], [target.fixture], cutoff)
    past_players = [row for row in appearances if row.match_id != target.match_id]
    _, diagnostic = target_result(
        matches[:-1], past_players, controls=controls, lineups=selected
    )
    assert diagnostic.status == expected


def test_late_player_statistics_enter_only_when_the_complete_match_batch_is_available():
    matches, appearances = sample_history()
    target = matches[-1]
    prior = matches[-2]
    first_cutoff = prior.available_at + timedelta(hours=1)
    second_cutoff = prior.available_at + timedelta(hours=3)
    selected = [
        replace(row, observed_at=first_cutoff, source="timed_selection")
        for row in derive_historical_lineups(appearances, matches)
        if row.match_id == target.match_id
    ]
    past_players = [row for row in appearances if row.match_id != target.match_id]
    delayed = [
        replace(row, statistics_available_at=prior.available_at + timedelta(hours=2))
        if row.match_id == prior.match_id and row.player_id == "A0"
        else row
        for row in past_players
    ]
    first_controls = predict_fixtures(matches[:-1], [target.fixture], first_cutoff)
    second_controls = predict_fixtures(matches[:-1], [target.fixture], second_cutoff)
    _, before = target_result(
        matches[:-1], delayed, controls=first_controls, lineups=selected
    )
    _, after = target_result(
        matches[:-1], delayed, controls=second_controls, lineups=selected
    )
    _, excluded = target_result(
        matches[:-2],
        [row for row in past_players if row.match_id != prior.match_id],
        controls=first_controls,
        lineups=selected,
    )
    _, available = target_result(
        matches[:-1], past_players, controls=second_controls, lineups=selected
    )
    assert before.lineup == excluded.lineup
    assert after.lineup == available.lineup
    assert after.lineup.home.missing_leader_gap > before.lineup.home.missing_leader_gap
    assert after.correction != before.correction
