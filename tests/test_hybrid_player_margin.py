from dataclasses import replace
from datetime import timedelta

import pytest

from src.mae_model.player_margin import (
    HybridPlayerConfig,
    PlayerDiagnostic,
    PlayerHistory,
    PlayerId,
    PlayerModelConfig,
    PlayerStats,
    combine_player_predictions,
    replay_hybrid_player_predictions,
)
from src.mae_model.sequential_margin import walk_forward_predictions
from tests.test_player_margin import sample_history


def hybrid_config():
    return HybridPlayerConfig(
        outcome=PlayerModelConfig(control_model_name="team_only"),
        official=PlayerModelConfig(
            measurement="official_points",
            control_model_name="team_only",
            rating_prior_games=12,
        ),
    )


def control_rows():
    matches, _ = sample_history(2)
    return [
        replace(row, predicted_margin=10, actual_margin=15, abs_error=5)
        for row in walk_forward_predictions(matches, 0)
        if row.model_name == "team_only"
    ]


def diagnostic(row, correction, status="adjusted"):
    return PlayerDiagnostic(row.match_id, row.cutoff, status, correction, None)


def test_additive_formula_matches_keys_and_has_no_final_cap():
    controls = control_rows()
    outcomes = [diagnostic(controls[0], 2), diagnostic(controls[1], 4)]
    officials = [diagnostic(controls[1], 4), diagnostic(controls[0], -1)]
    rows, diagnostics = combine_player_predictions(controls, outcomes, officials)
    assert [row.predicted_margin for row in rows] == [11, 18]
    assert [row.abs_error for row in rows] == [4, 3]
    assert [row.correction for row in diagnostics] == [1, 8]
    assert [row.model_name for row in rows] == ["player_hybrid", "player_hybrid"]
    assert [row.used_fallback for row in rows] == [False, False]
    assert [row.predicted_margin for row in controls] == [10, 10]


def test_missing_component_keeps_the_other_correction():
    controls = control_rows()
    outcomes = [diagnostic(row, 0, "no_lineup") for row in controls]
    officials = [diagnostic(controls[0], -1), diagnostic(controls[1], 0, "no_lineup")]
    rows, _ = combine_player_predictions(controls, outcomes, officials)
    assert [row.predicted_margin for row in rows] == [9, 10]
    assert [row.used_fallback for row in rows] == [False, True]


@pytest.mark.parametrize("change", ["duplicate", "missing", "cutoff"])
def test_invalid_component_keys_fail(change):
    controls = control_rows()
    outcomes = [diagnostic(row, 2) for row in controls]
    officials = [diagnostic(row, -1) for row in controls]
    if change == "duplicate":
        officials.append(officials[0])
    elif change == "missing":
        officials.pop()
    else:
        officials[0] = replace(
            officials[0], cutoff=officials[0].cutoff - timedelta(seconds=1)
        )
    with pytest.raises(ValueError, match="[Kk]ey"):
        combine_player_predictions(controls, outcomes, officials)


def test_hybrid_config_rejects_incompatible_sources_and_controls():
    config = hybrid_config()
    for fields in (
        {"outcome": replace(config.outcome, measurement="official_points")},
        {"official": replace(config.official, measurement="outcome_fantasy")},
        {"outcome": replace(config.outcome, control_model_name="market_scoring_blend")},
        {"official": replace(config.official, control_model_name="scoring_shots")},
    ):
        with pytest.raises(ValueError, match="Hybrid"):
            replace(config, **fields)


def test_independent_sources_ignore_current_match_statistics():
    matches, outcome = sample_history()
    official = [
        replace(
            row,
            player_id=PlayerId("wheelo:" + row.player_id),
            official_rating_points=40 if row.player_id == "A0" else 10,
        )
        for row in outcome
    ]
    controls = walk_forward_predictions(matches, 0)
    expected_rows, expected_diagnostics = replay_hybrid_player_predictions(
        matches,
        controls,
        PlayerHistory(outcome, []),
        PlayerHistory(official, []),
        hybrid_config(),
    )
    changed_outcome = [
        replace(row, stats=PlayerStats(goals=1000))
        if row.match_id == matches[-1].match_id
        else row
        for row in outcome
    ]
    changed_official = [
        replace(row, official_rating_points=1000)
        if row.match_id == matches[-1].match_id
        else row
        for row in official
    ]
    rows, diagnostics = replay_hybrid_player_predictions(
        matches,
        controls,
        PlayerHistory(changed_outcome, []),
        PlayerHistory(changed_official, []),
        hybrid_config(),
    )
    assert rows[-1] == expected_rows[-1]
    assert diagnostics[-1] == expected_diagnostics[-1]
    assert diagnostics[-1].outcome.status == "adjusted"
    assert diagnostics[-1].official.status == "adjusted"
    assert diagnostics[-1].outcome.lineup.home.missing_leader == "A0"
    assert diagnostics[-1].official.lineup.home.missing_leader == "wheelo:A0"


def test_delayed_official_statistics_do_not_delay_outcome_history():
    from src.mae_model.player_margin import derive_historical_lineups
    from src.mae_model.sequential_margin import predict_fixtures

    matches, appearances = sample_history()
    target, prior = matches[-1], matches[-2]
    early = prior.available_at + timedelta(hours=1)
    late = prior.available_at + timedelta(hours=3)
    outcome = [row for row in appearances if row.match_id != target.match_id]
    official = [
        replace(
            row,
            player_id=PlayerId("wheelo:" + row.player_id),
            official_rating_points=40 if row.player_id == "A0" else 10,
            statistics_available_at=prior.available_at + timedelta(hours=2)
            if row.match_id == prior.match_id
            else row.statistics_available_at,
        )
        for row in outcome
    ]
    selected = [
        replace(row, observed_at=early, source="timed_selection")
        for row in derive_historical_lineups(appearances, matches)
        if row.match_id == target.match_id
    ]
    official_selected = [
        replace(
            row,
            player_ids=tuple(PlayerId("wheelo:" + player) for player in row.player_ids),
        )
        for row in selected
    ]
    histories = (
        PlayerHistory(outcome, selected),
        PlayerHistory(official, official_selected),
    )
    results = []
    for cutoff in (early, late):
        controls = predict_fixtures(matches[:-1], [target.fixture], cutoff)
        results.append(
            replay_hybrid_player_predictions(
                matches[:-1], controls, *histories, hybrid_config()
            )
        )
    before, after = results[0][1][0], results[1][1][0]
    assert before.outcome.lineup == after.outcome.lineup
    assert before.outcome.status == after.outcome.status == "adjusted"
    assert (
        before.official.lineup.home.missing_leader_gap
        < after.official.lineup.home.missing_leader_gap
    )
    assert before.official.correction != after.official.correction


def test_backtest_command_records_both_sources_and_components(tmp_path):
    import csv
    import json

    from src.mae_model.run_backtest import main
    from tests.test_player_margin import appearance_csv, match_csv, write_csv

    matches, appearances = sample_history()
    history, outcome, official = (
        tmp_path / name for name in ("history.csv", "outcome.csv", "official.csv")
    )
    write_csv(history, [match_csv(row) for row in matches])
    write_csv(outcome, [appearance_csv(row) for row in appearances])
    write_csv(
        official,
        [
            {
                **appearance_csv(row),
                "player_ref": "wheelo:" + row.player_id,
                "official_rating_points": 40 if row.player_id == "A0" else 10,
            }
            for row in appearances
        ],
    )
    common = ["--matches-csv", str(history), "--min-train-years", "0"]
    base, added = tmp_path / "base", tmp_path / "added"
    main(common + ["--output-dir", str(base)])
    main(
        common
        + [
            "--player-stats-csv",
            str(outcome),
            "--official-player-stats-csv",
            str(official),
            "--player-control",
            "team_only",
            "--official-player-rating-prior-games",
            "12",
            "--output-dir",
            str(added),
        ]
    )
    assert (
        (added / "walk_forward_predictions.csv")
        .read_bytes()
        .startswith((base / "walk_forward_predictions.csv").read_bytes())
    )
    with (added / "walk_forward_predictions.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows[-1]["model_name"] == "selected_team_strength"
    metadata = json.loads((added / "metadata.json").read_text())
    assert len(metadata["inputs"]) == 3
    config = metadata["hybrid_player_model"]["configuration"]
    assert config["outcome"]["measurement"] == "outcome_fantasy"
    assert config["official"]["measurement"] == "official_points"
    assert config["official"]["rating_prior_games"] == 12
    assert metadata["hybrid_player_model"]["status_counts"]["official"]["adjusted"] == 1
    with (added / "hybrid_player_diagnostics.csv").open() as source:
        diagnostics = list(csv.DictReader(source))
    assert (
        diagnostics[-1]["outcome_status"]
        == diagnostics[-1]["official_status"]
        == "adjusted"
    )
    assert float(diagnostics[-1]["correction"]) < float(
        diagnostics[-1]["outcome_correction"]
    )
    assert metadata["selected_team_model"]["status_counts"] == {
        "insufficient_training": 13
    }
    with (added / "selected_team_diagnostics.csv").open() as source:
        selected_diagnostics = list(csv.DictReader(source))
    assert selected_diagnostics[-1]["status"] == "insufficient_training"
    base_metadata = json.loads((base / "metadata.json").read_text())
    assert "hybrid_player_model" not in base_metadata
    assert not any(
        key.startswith("official_player_") for key in base_metadata["configuration"]
    )


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--player-stats-csv", "unused"],
        [
            "--player-stats-csv",
            "unused",
            "--player-control",
            "team_only",
            "--player-measurement",
            "official_points",
        ],
        [
            "--player-stats-csv",
            "unused",
            "--player-control",
            "team_only",
            "--lead-hours",
            "1",
        ],
    ],
)
def test_backtest_rejects_incompatible_hybrid_arguments(extra, capsys):
    from src.mae_model.run_backtest import main

    with pytest.raises(SystemExit) as error:
        main(["--official-player-stats-csv", "unused"] + extra)
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_live_command_uses_separate_lineup_ids_and_matches_historical_margin(tmp_path):
    import csv
    import json

    from src.mae_model.predict_fixtures import main
    from tests.test_player_margin import appearance_csv, match_csv, write_csv

    matches, appearances = sample_history()
    target = matches[-1]
    official_appearances = [
        replace(
            row,
            player_id=PlayerId("wheelo:" + row.player_id),
            official_rating_points=40 if row.player_id == "A0" else 10,
        )
        for row in appearances
    ]
    historical, _ = replay_hybrid_player_predictions(
        matches,
        walk_forward_predictions(matches, 0),
        PlayerHistory(appearances, []),
        PlayerHistory(official_appearances, []),
        hybrid_config(),
    )
    history, fixture_file, outcome_file, official_file, lineups, official_lineups = (
        tmp_path / name
        for name in (
            "history.csv",
            "fixtures.csv",
            "outcome.csv",
            "official.csv",
            "lineups.csv",
            "official_lineups.csv",
        )
    )
    write_csv(history, [match_csv(row) for row in matches[:-1]])
    write_csv(
        fixture_file,
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
    for rows, stats_path, lineup_path in (
        (appearances, outcome_file, lineups),
        (official_appearances, official_file, official_lineups),
    ):
        write_csv(
            stats_path,
            [
                {
                    **appearance_csv(row),
                    "official_rating_points": row.official_rating_points,
                }
                for row in rows
                if row.match_id != target.match_id
            ],
        )
        write_csv(
            lineup_path,
            [
                {
                    "match_id": row.match_id,
                    "team_name": row.team,
                    "player_ref": row.player_id,
                    "observed_at": (
                        target.fixture.kickoff - timedelta(minutes=1)
                    ).isoformat(),
                }
                for row in rows
                if row.match_id == target.match_id
            ],
        )
    output = tmp_path / "live"
    main(
        [
            "--matches-csv",
            str(history),
            "--fixtures-csv",
            str(fixture_file),
            "--player-stats-csv",
            str(outcome_file),
            "--official-player-stats-csv",
            str(official_file),
            "--lineups-csv",
            str(lineups),
            "--official-player-lineups-csv",
            str(official_lineups),
            "--player-control",
            "team_only",
            "--as-of",
            (target.fixture.kickoff - timedelta(minutes=1)).isoformat(),
            "--output-dir",
            str(output),
        ]
    )
    with (output / "fixture_predictions.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert rows[-1]["model_name"] == "selected_team_strength"
    assert float(rows[-1]["predicted_margin"]) == pytest.approx(
        historical[-1].predicted_margin, abs=0.0001
    )
    assert rows[-1]["actual_margin"] == ""
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["inputs"]) == 6
    assert metadata["hybrid_player_model"]["status_counts"]["outcome"] == {
        "adjusted": 1
    }


@pytest.mark.parametrize(
    "flag", ["--official-player-stats-csv", "--official-player-lineups-csv"]
)
def test_live_requires_both_official_inputs(flag, capsys):
    from src.mae_model.predict_fixtures import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--fixtures-csv",
                "unused",
                "--as-of",
                "2019-03-01T17:00:00+11:00",
                "--player-stats-csv",
                "unused",
                "--lineups-csv",
                "unused",
                "--player-control",
                "team_only",
                flag,
                "unused",
            ]
        )
    assert error.value.code == 2
    assert (
        "Use --official-player-stats-csv and --official-player-lineups-csv together"
        in capsys.readouterr().err
    )
