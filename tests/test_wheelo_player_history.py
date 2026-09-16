import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.mae_model.data import MatchRow
from src.scrape_afl import wheelo_player_history as history


def match(match_id="m1", *, home="Richmond", away="Carlton"):
    return MatchRow(
        match_id,
        2012,
        "1",
        datetime(2012, 3, 29, 19, 45, tzinfo=ZoneInfo("Australia/Melbourne")),
        "M.C.G.",
        home,
        away,
        100,
        90,
    )


def columns(rows):
    return {key: [row[key] for row in rows] for key in rows[0]}


def round_body(*, rating=-2.5):
    match_rows = [
        {
            "MatchId": "20120101",
            "MatchDate": "29 Mar",
            "HomeTeam": "Richmond",
            "AwayTeam": "Carlton",
        }
    ]
    players = []
    for team, prefix in (("Richmond", "r"), ("Carlton", "c")):
        for index in range(22):
            players.append(
                {
                    "MatchId": "20120101",
                    "WebsiteId": "stable" if team == "Richmond" and index == 0 else f"{prefix}{index}",
                    "Player": f"{team} {index}",
                    "Team": team,
                    "TimeOnGround": 80 + index % 20,
                    "RatingPoints": rating if index == 0 else float(index),
                    "EstimatedRating": 9999,
                }
            )
    return json.dumps(
        {
            "Matches": [columns(match_rows)],
            "Data": [columns(players)],
            "Summary": [{"Season": "2012", "RoundNumber": 1, "RoundName": "Round 1"}],
        }
    ).encode()


def test_parse_keeps_stable_identity_and_actual_rating_only():
    matches, players = history.parse_round_document(round_body(), 2012)

    assert matches[0].home_team == "Richmond"
    first = players[0]
    assert first.player_ref == "wheelo:stable"
    assert first.official_rating_points == -2.5
    assert not hasattr(first, "estimated_rating")


def test_parse_rejects_different_column_lengths():
    payload = json.loads(round_body())
    payload["Data"][0]["Player"].pop()

    with pytest.raises(ValueError, match="different lengths"):
        history.parse_round_document(json.dumps(payload).encode(), 2012)


def test_match_join_uses_non_result_identity_and_rejects_missing():
    source, _ = history.parse_round_document(round_body(), 2012)
    joined = history.join_source_matches(source, [match()])
    assert joined["20120101"].match_id == "m1"

    with pytest.raises(ValueError, match="missing"):
        history.join_source_matches(source, [match(home="Essendon", away="Carlton")])


def test_match_join_rejects_ambiguous_candidates():
    source, _ = history.parse_round_document(round_body(), 2012)

    with pytest.raises(ValueError, match="ambiguous"):
        history.join_source_matches(source, [match("m1"), match("m2")])


def test_cached_document_restarts_and_detects_changed_body(monkeypatch, tmp_path):
    body = b'{"RoundId":["201201"]}'
    calls = []

    def download(url):
        calls.append(url)
        return body

    monkeypatch.setattr(history, "_download", download)
    first, record = history._cached_document("https://example/2012.json", tmp_path, False)
    second, repeated = history._cached_document("https://example/2012.json", tmp_path, False)
    assert first == second == body
    assert record == repeated
    assert calls == ["https://example/2012.json"]

    cached = tmp_path / record["body_path"]
    cached.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        history._cached_document("https://example/2012.json", tmp_path, False)


def test_builder_writes_repeatable_checked_csv(monkeypatch, tmp_path):
    index = json.dumps({"RoundId": ["201201"]}).encode()
    responses = {
        f"{history.BASE_URL}/2012.json": index,
        f"{history.BASE_URL}/201201.json": round_body(),
    }
    monkeypatch.setattr(history, "_download", responses.__getitem__)
    output = tmp_path / "ratings"

    first = history.build_wheelo_player_history([match()], 2012, 2012, output)
    first_bytes = (output / "afl_player_ratings.csv").read_bytes()
    second = history.build_wheelo_player_history([match()], 2012, 2012, output)

    assert first["output_sha256"] == second["output_sha256"]
    assert (output / "afl_player_ratings.csv").read_bytes() == first_bytes
    with (output / "afl_player_ratings.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 44
    assert rows[0]["player_ref"].startswith("wheelo:")
    assert "EstimatedRating" not in rows[0]
    assert {row["statistics_available_at"] for row in rows} == {
        match().available_at.isoformat()
    }
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["team_size_counts"] == {"22": 2}


def test_player_identity_does_not_include_club():
    _, before = history.parse_round_document(round_body(), 2012)
    payload = json.loads(round_body())
    payload["Data"][0]["Team"][0] = "Carlton"
    _, after = history.parse_round_document(json.dumps(payload).encode(), 2012)

    assert before[0].player_ref == after[0].player_ref == "wheelo:stable"
