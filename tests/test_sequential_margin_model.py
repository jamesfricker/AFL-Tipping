import csv
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.mae_model.data import (
    MarketQuote,
    MatchRow,
    load_fixtures_csv,
    load_market_csv,
    load_market_xlsx,
    load_matches_csv,
    parse_timestamp,
)
from src.mae_model.predict_fixtures import main as predict_main
from src.mae_model.run_backtest import main as backtest_main
from src.mae_model.sequential_margin import (
    PredictionRequest,
    fit_market_weight,
    predict_fixtures,
    replay_predictions,
    summarize_predictions,
    walk_forward_predictions,
)


def stamp(text):
    return datetime.fromisoformat(text)


def match(
    match_id="first", year=2024, month=3, day=1, hour=19, home="A", away="B", score=100
):
    return MatchRow(
        match_id,
        year,
        "1",
        stamp(f"{year}-{month:02}-{day:02}T{hour:02}:00:00+11:00"),
        "M.C.G.",
        home,
        away,
        score,
        80,
        home_scoring_shots=25,
        away_scoring_shots=20,
    )


def fixture(match_id="future", year=2025, month=3, day=1, hour=19, home="A", away="B"):
    return match(match_id, year, month, day, hour, home, away).fixture


def margins(rows):
    return {row.model_name: row.predicted_margin for row in rows}


def write_csv(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def match_dict():
    return {
        "match_id": "first",
        "year": "2024",
        "round": "1",
        "date": "01-Mar-2024",
        "time": "7:00 PM",
        "venue": "M.C.G.",
        "home_team_name": "A",
        "away_team_name": "B",
        "home_team_score": "100",
        "away_team_score": "80",
        "home_scoring_shots": "25",
        "away_scoring_shots": "20",
    }


def fixture_dict():
    return {
        "match_id": "future",
        "year": "2025",
        "round": "1",
        "kickoff": "2025-03-01T19:00:00+11:00",
        "venue": "M.C.G.",
        "home_team_name": "A",
        "away_team_name": "B",
    }


def test_initial_forecast_has_literal_baseline_and_missing_market():
    target = fixture()
    rows = predict_fixtures([], [target], target.kickoff - timedelta(hours=1))
    assert margins(rows) == {
        "team_only": 6.0,
        "scoring_shots": 9.7,
        "market_only": None,
        "market_scoring_blend": 9.7,
    }
    assert rows[-1].used_fallback
    assert rows[-1].market_weight == 1.0
    assert rows[2].market_status == "missing_quote"
    assert rows[2].market_home_probability is None


def test_season_transition_and_live_backtest_use_same_forecast():
    prior, target = match(), match("future", year=2025)
    cutoff = target.fixture.kickoff - timedelta(hours=1)
    live = predict_fixtures([prior, target], [target.fixture], cutoff, lead_hours=1)
    historical = walk_forward_predictions([prior, target], 1, lead_hours=1)
    assert margins(live) == margins(historical)
    assert margins(live)["team_only"] == pytest.approx(7.4196)
    assert margins(live)["scoring_shots"] == pytest.approx(9.428)
    assert all(row.actual_margin is None for row in live)
    assert all(row.actual_margin == 20 for row in historical)


def test_future_results_and_same_day_results_do_not_change_predictions():
    first = match()
    same_day = match("second", hour=21, home="C", away="D")
    target = fixture("later", year=2024, day=2)
    cutoff = stamp("2024-03-01T22:00:00+11:00")
    original = predict_fixtures([first, same_day], [target], cutoff)
    changed = predict_fixtures(
        [replace(first, home_score=900), replace(same_day, away_score=0)],
        [target],
        cutoff,
    )
    assert margins(original) == margins(changed)
    assert margins(original)["team_only"] == 6.0
    next_day = stamp("2024-03-02T10:00:00+11:00")
    assert predict_fixtures([first, same_day], [target], next_day) == predict_fixtures(
        [same_day, first], [target], next_day
    )
    assert (
        margins(predict_fixtures([first, same_day], [target], next_day))["team_only"]
        != 6.0
    )


def test_explicit_publication_time_controls_result_use():
    prior = replace(match(), result_available_at=stamp("2024-03-01T21:00:00+11:00"))
    target = fixture("later", year=2024, day=2)
    before = predict_fixtures([prior], [target], stamp("2024-03-01T20:59:59+11:00"))
    after = predict_fixtures([prior], [target], stamp("2024-03-01T21:00:00+11:00"))
    assert margins(before)["team_only"] == 6.0
    assert margins(after)["team_only"] == pytest.approx(7.82)
    assert prior.timing_assumption == "explicit_timestamp"


def test_delayed_prior_season_result_does_not_reverse_season():
    old = replace(
        match("old", year=2023), result_available_at=stamp("2025-03-02T12:00:00+11:00")
    )
    recent = match("recent", year=2025)
    target = fixture("later", year=2025, day=3)
    rows = predict_fixtures([old, recent], [target], stamp("2025-03-02T13:00:00+11:00"))
    assert all(
        math.isfinite(value) for value in margins(rows).values() if value is not None
    )
    assert rows == predict_fixtures(
        [recent, old], [target], stamp("2025-03-02T13:00:00+11:00")
    )


def test_latest_eligible_quote_and_probability_use_deadline():
    target = fixture()
    cutoff = target.kickoff - timedelta(hours=1)
    quotes = [
        MarketQuote(target.match_id, cutoff - timedelta(hours=1), 4.5, 2, 4),
        MarketQuote(target.match_id, cutoff + timedelta(minutes=1), 100, 1.1, 20),
    ]
    rows = predict_fixtures([], [target], cutoff, quotes)
    assert margins(rows)["market_only"] == 4.5
    assert margins(rows)["market_scoring_blend"] == 4.5
    assert rows[2].market_home_probability == pytest.approx(2 / 3)
    assert rows[2].market_status == "timed_quote"
    assert not rows[-1].used_fallback
    late = predict_fixtures([], [target], cutoff, quotes[1:])
    assert late[2].predicted_margin is None
    assert late[2].market_status == "no_eligible_quote"


def test_quotes_for_other_future_history_ids_are_safe():
    target = fixture()
    future = match("other", year=2025, day=4)
    quote = MarketQuote("other", future.fixture.kickoff, 999)
    rows = predict_fixtures(
        [future], [target], target.kickoff - timedelta(hours=1), [quote]
    )
    assert margins(rows)["team_only"] == 6.0


def test_weight_defaults_and_ties_prefer_market():
    assert fit_market_weight([(10, 50, 10)] * 119) == 1.0
    assert fit_market_weight([(10, 10, 10)] * 120) == 1.0
    assert fit_market_weight([(10, 50, 10)] * 120) == 0.0
    assert fit_market_weight([(20, 30, 10)] * 120) == 0.5


def test_weight_uses_only_prior_five_years_and_known_results():
    history = [
        match(
            str(i),
            year=2019,
            day=(i % 28) + 1,
            month=(i // 28) + 1,
            home=f"A{i}",
            away=f"B{i}",
        )
        for i in range(120)
    ]
    quotes = [
        MarketQuote(row.match_id, row.fixture.kickoff - timedelta(hours=1), 100)
        for row in history
    ]
    target = fixture(year=2024)
    rows = predict_fixtures(
        history, [target], target.kickoff - timedelta(hours=1), quotes
    )
    assert rows[-1].market_weight == 0.14
    assert rows[-1].weight_training_games == 120
    too_old = predict_fixtures(
        history, [fixture(year=2025)], stamp("2025-03-01T18:00:00+11:00"), quotes
    )
    assert too_old[-1].market_weight == 1.0
    assert too_old[-1].weight_training_games == 0
    late = [
        replace(row, result_available_at=stamp("2024-01-02T00:00:00+11:00"))
        for row in history
    ]
    frozen = predict_fixtures(
        late, [target], target.kickoff - timedelta(hours=1), quotes
    )
    assert frozen[-1].weight_training_games == 0


def test_training_quotes_cannot_arrive_after_the_historical_deadline():
    history = [
        match(
            str(i),
            year=2024,
            day=(i % 28) + 1,
            month=(i // 28) + 1,
            home=f"A{i}",
            away=f"B{i}",
        )
        for i in range(120)
    ]
    quotes = [
        MarketQuote(row.match_id, row.fixture.kickoff + timedelta(minutes=1), 100)
        for row in history
    ]
    target = fixture()
    rows = predict_fixtures(
        history, [target], target.kickoff - timedelta(hours=1), quotes
    )
    assert rows[-1].weight_training_games == 0
    assert rows[-1].market_weight == 1.0


def test_target_season_outcomes_do_not_fit_weight():
    history = [
        match(str(i), year=2025, month=1, day=(i % 28) + 1, home=f"A{i}", away=f"B{i}")
        for i in range(120)
    ]
    quotes = [
        MarketQuote(row.match_id, row.fixture.kickoff - timedelta(hours=1), 100)
        for row in history
    ]
    target = fixture()
    rows = predict_fixtures(
        history, [target], target.kickoff - timedelta(hours=1), quotes
    )
    assert rows[-1].weight_training_games == 0
    assert rows[-1].market_weight == 1.0


def test_raw_market_missing_stays_missing_in_common_subset():
    first, second = match(), match("second", day=2)
    quote = MarketQuote("first", first.fixture.kickoff - timedelta(hours=1), 10)
    summary = summarize_predictions(
        walk_forward_predictions([first, second], 0, [quote])
    )
    market = next(
        row
        for row in summary
        if row["year"] == "ALL"
        and row["model_name"] == "market_only"
        and row["scope"] == "all_matches"
    )
    blend = next(
        row
        for row in summary
        if row["year"] == "ALL"
        and row["model_name"] == "market_scoring_blend"
        and row["scope"] == "all_matches"
    )
    common = [
        row
        for row in summary
        if row["year"] == "ALL" and row["scope"] == "common_market"
    ]
    assert market["num_games"] == 1
    assert market["missing_predictions"] == 1
    assert blend["fallback_count"] == 1
    assert len(common) == 4
    assert {row["num_games"] for row in common} == {1}


@pytest.mark.parametrize(
    "actual,predicted,expected",
    [(0, 1, 0), (-5, 0, 0), (0, 0, 100), (-5, -1, 100), (5, 1, 100)],
)
def test_tips_require_matching_signs(actual, predicted, expected):
    row = walk_forward_predictions([match()], 0)[0]
    row = replace(
        row,
        predicted_margin=predicted,
        actual_margin=actual,
        abs_error=abs(actual - predicted),
    )
    assert summarize_predictions([row])[0]["tip_pct"] == expected


def test_timezone_local_year_and_next_day_use_venue():
    row = replace(match(), date=stamp("2023-12-31T15:00:00+00:00"), year=2024)
    assert row.fixture.year == 2024
    assert row.available_at.isoformat() == "2024-01-02T00:00:00+11:00"
    perth = replace(
        match(), venue="Perth Stadium", date=stamp("2024-03-01T19:00:00+08:00")
    )
    assert perth.available_at.isoformat() == "2024-03-02T00:00:00+08:00"


@pytest.mark.parametrize("value", ["nan", "inf", "bad"])
def test_bad_match_numbers_fail(tmp_path, value):
    row = {**match_dict(), "home_team_score": value}
    path = tmp_path / "matches.csv"
    write_csv(path, [row])
    with pytest.raises(ValueError, match="matches.csv:2"):
        load_matches_csv(str(path))


def test_csv_preserves_kickoff_and_rejects_duplicates(tmp_path):
    path = tmp_path / "matches.csv"
    write_csv(path, [match_dict()])
    assert (
        load_matches_csv(str(path))[0].date.isoformat() == "2024-03-01T19:00:00+11:00"
    )
    write_csv(path, [match_dict(), match_dict()])
    with pytest.raises(ValueError, match="Duplicate"):
        load_matches_csv(str(path))


def test_fixture_outcomes_and_naive_timestamps_fail(tmp_path):
    path = tmp_path / "fixtures.csv"
    write_csv(path, [{**fixture_dict(), "home_team_score": "100"}])
    with pytest.raises(ValueError, match="outcomes"):
        load_fixtures_csv(str(path))
    with pytest.raises(ValueError, match="timezone offset"):
        parse_timestamp("2025-03-01T18:00:00")
    with pytest.raises(ValueError, match="start after"):
        predict_fixtures([], [fixture()], fixture().kickoff)


@pytest.mark.parametrize(
    "change",
    [
        {"predicted_margin": "NaN"},
        {"home_odds": "1", "away_odds": "2"},
        {"observed_at": "2025-03-01T18:00:00"},
    ],
)
def test_invalid_market_csv_fails(tmp_path, change):
    path = tmp_path / "market.csv"
    write_csv(
        path,
        [
            {
                "match_id": "future",
                "observed_at": "2025-03-01T18:00:00+11:00",
                "predicted_margin": "4.5",
                **change,
            }
        ],
    )
    with pytest.raises(ValueError, match="market.csv:2"):
        load_market_csv(str(path))


def test_duplicate_quotes_and_unknown_match_ids_fail():
    target = fixture()
    quote = MarketQuote("future", target.kickoff - timedelta(hours=1), 4.5)
    with pytest.raises(ValueError, match="Duplicate"):
        predict_fixtures([], [target], quote.observed_at, [quote, quote])
    with pytest.raises(ValueError, match="unknown match ID"):
        predict_fixtures(
            [], [target], quote.observed_at, [replace(quote, match_id="unknown")]
        )
    with pytest.raises(ValueError, match="Untimed"):
        predict_fixtures(
            [], [target], quote.observed_at, [replace(quote, observed_at=None)]
        )


def test_workbook_requires_explicit_benchmark_and_reports_missing_margin(tmp_path):
    path = tmp_path / "market.xlsx"
    frame = pd.DataFrame(
        [
            {
                "Date": "2024-03-01",
                "Home Team": "A",
                "Away Team": "B",
                "Home Line Close": None,
                "Home Odds": 2,
                "Away Odds": 2,
            }
        ]
    )
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="Data", startrow=1, index=False)
    with pytest.raises(ValueError, match="closing-line-benchmark"):
        load_market_xlsx(str(path), [match()])
    quotes = load_market_xlsx(str(path), [match()], closing_line_benchmark=True)
    rows = walk_forward_predictions([match()], 0, quotes, closing_line_benchmark=True)
    assert rows[2].predicted_margin is None
    assert rows[2].market_status == "missing_margin"
    assert rows[-1].used_fallback


def test_cli_outputs_and_provenance(tmp_path):
    history, future, quotes = (
        tmp_path / "history.csv",
        tmp_path / "fixtures.csv",
        tmp_path / "quotes.csv",
    )
    write_csv(history, [match_dict()])
    write_csv(future, [fixture_dict()])
    write_csv(
        quotes,
        [
            {
                "match_id": "future",
                "observed_at": "2025-03-01T17:00:00+11:00",
                "predicted_margin": "4.5",
            }
        ],
    )
    output = tmp_path / "live"
    predict_main(
        [
            "--matches-csv",
            str(history),
            "--fixtures-csv",
            str(future),
            "--market-csv",
            str(quotes),
            "--as-of",
            "2025-03-01T18:00:00+11:00",
            "--output-dir",
            str(output),
        ]
    )
    with (output / "fixture_predictions.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert rows[-1]["predicted_margin"] == "4.5"
    assert rows[-1]["actual_margin"] == ""
    metadata = json.loads((output / "metadata.json").read_text())
    assert len(metadata["inputs"]) == 3
    assert all(len(item["sha256"]) == 64 for item in metadata["inputs"])
    assert metadata["source_hashes"]["src/mae_model/sequential_margin.py"]
    assert metadata["common_market_matches"] == 1
    assert metadata["configuration"]["default_market_weight"] == 1.0
    assert metadata["season_weights"][0]["market_weight"] == 1.0
    assert isinstance(metadata["git_dirty"], bool)
    backtest_main(
        [
            "--matches-csv",
            str(history),
            "--min-train-years",
            "0",
            "--output-dir",
            str(tmp_path / "backtest"),
        ]
    )
    assert (tmp_path / "backtest" / "mae_summary.csv").exists()


def test_missing_input_and_untimed_live_cli_fail_without_output(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(SystemExit) as error:
        backtest_main(
            ["--matches-csv", str(tmp_path / "absent.csv"), "--output-dir", str(output)]
        )
    assert error.value.code == 2
    assert not output.exists()
    with pytest.raises(SystemExit):
        predict_main(
            [
                "--fixtures-csv",
                "unused",
                "--as-of",
                "2025-03-01T18:00:00+11:00",
                "--market-xlsx",
                "unused.xlsx",
                "--closing-line-benchmark",
            ]
        )


def test_pure_replay_does_not_change_inputs():
    history = [match()]
    target = fixture()
    request = PredictionRequest(target, target.kickoff - timedelta(hours=1))
    first = replay_predictions(history, [request])
    assert first == replay_predictions(history, [request])
    assert history == [match()]


def test_new_team_forecast_does_not_change_when_zero_residual_result_starts_season():
    prior = match()
    zero_result = replace(
        match("zero", year=2025), home_score=79.7098, away_score=72.2902
    )
    target = fixture("cold", year=2025, day=3, home="C", away="A")
    before = predict_fixtures(
        [prior, zero_result], [target], stamp("2025-03-01T18:00:00+11:00")
    )
    after = predict_fixtures(
        [prior, zero_result], [target], stamp("2025-03-02T01:00:00+11:00")
    )
    assert margins(before)["team_only"] == pytest.approx(5.2902)
    assert margins(after)["team_only"] == pytest.approx(5.2902)


@pytest.mark.parametrize(
    "field",
    [
        "home_behinds",
        "away_behinds",
        "home_scoring_shots",
        "away_scoring_shots",
        "result_available_at",
    ],
)
def test_fixture_rejects_all_result_columns(tmp_path, field):
    path = tmp_path / "fixtures.csv"
    write_csv(path, [{**fixture_dict(), field: "1"}])
    with pytest.raises(ValueError, match="outcomes"):
        load_fixtures_csv(str(path))


def test_missing_line_with_invalid_workbook_odds_has_diagnostic(tmp_path):
    path = tmp_path / "market.xlsx"
    frame = pd.DataFrame(
        [
            {
                "Date": "2024-03-01",
                "Home Team": "A",
                "Away Team": "B",
                "Home Line Close": None,
                "Home Odds": 1,
                "Away Odds": 21,
            }
        ]
    )
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="Data", startrow=1, index=False)
    quotes = load_market_xlsx(str(path), [match()], closing_line_benchmark=True)
    assert quotes[0].predicted_margin is None
    assert quotes[0].home_probability is None
    assert quotes[0].validation_note == "invalid_odds_without_margin"


def test_bad_empty_csv_headers_fail(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("wrong,headers\n")
    with pytest.raises(ValueError, match="missing CSV columns"):
        load_market_csv(str(path))


def test_recorded_benchmark_matches_supplied_data():
    root = Path(__file__).resolve().parents[1]
    history = load_matches_csv(str(root / "src/outputs/afl_data.csv"))
    quotes = load_market_xlsx(
        str(root / "src/outputs/afl_betting_history.xlsx"),
        history,
        closing_line_benchmark=True,
    )
    rows = walk_forward_predictions(
        history, market_quotes=quotes, closing_line_benchmark=True
    )
    summary = {
        row["model_name"]: row
        for row in summarize_predictions(rows)
        if row["year"] == "ALL" and row["scope"] == "common_market"
    }
    assert {row["num_games"] for row in summary.values()} == {2258}
    assert summary["market_only"]["mae_margin"] == pytest.approx(26.646590)
    assert summary["market_scoring_blend"]["mae_margin"] == pytest.approx(26.569416)
