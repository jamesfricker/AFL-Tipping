import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup

from src.scrape_afl import player_history, scrape_tables


MATCH_URL = "https://afltables.com/afl/stats/games/2012/162120120324.html"
SEASON_URL = "https://afltables.com/afl/seas/2012.html"


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    matches = tmp_path / "matches.csv"
    with Path("src/outputs/afl_data.csv").open(newline="") as source:
        reader = csv.DictReader(source)
        row = next(reader)
        fields = list(reader.fieldnames or [])
    with matches.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    players = tmp_path / "existing-players.csv"
    with players.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=scrape_tables.PLAYER_FIELDS)
        writer.writeheader()
    return matches, players


def _responses(monkeypatch, match_html: str):
    season_html = (
        '<html><body><a href="../stats/games/2012/162120120324.html">'
        "match</a></body></html>"
    )
    calls = []

    def get(url, timeout):
        calls.append(url)
        content = {SEASON_URL: season_html, MATCH_URL: match_html}[url]
        return SimpleNamespace(text=content, raise_for_status=lambda: None)

    monkeypatch.setattr(player_history.requests, "get", get)
    return calls


def test_builds_checked_player_history(monkeypatch, tmp_path):
    matches, players = _inputs(tmp_path)
    match_html = Path("tests/test_data/afl_tables_match.html").read_text()
    calls = _responses(monkeypatch, match_html)

    result = player_history.build_player_history(
        matches_csv=matches,
        existing_players_csv=players,
        seasons=range(2012, 2013),
        work_dir=tmp_path / "history",
    )

    assert result.added_matches == 1
    assert result.added_players == 44
    assert calls == [SEASON_URL, MATCH_URL]
    with result.players_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 44
    assert {row["match_id"] for row in rows} == {"162120120324"}
    assert "statistics_available_at" not in rows[0]
    report = json.loads(result.report_json.read_text())
    assert report["expected_matches_by_season"] == {"2012": 1}
    assert report["collected_matches_by_season"] == {"2012": 1}
    assert report["output_sha256"]
    assert {source["url"] for source in report["sources"]} == {
        SEASON_URL,
        MATCH_URL,
    }


def test_resumes_from_valid_cached_pages(monkeypatch, tmp_path):
    matches, players = _inputs(tmp_path)
    match_html = Path("tests/test_data/afl_tables_match.html").read_text()
    calls = _responses(monkeypatch, match_html)
    arguments = {
        "matches_csv": matches,
        "existing_players_csv": players,
        "seasons": range(2012, 2013),
        "work_dir": tmp_path / "history",
    }

    first = player_history.build_player_history(**arguments)
    first_bytes = first.players_csv.read_bytes()
    assert len(calls) == 2
    first.players_csv.unlink()
    first.report_json.unlink()

    def fail_get(*args, **kwargs):
        raise AssertionError("A completed resume must not use the network")

    monkeypatch.setattr(player_history.requests, "get", fail_get)
    second = player_history.build_player_history(**arguments)
    assert second == first
    assert second.players_csv.read_bytes() == first_bytes


def test_rejects_incomplete_team(monkeypatch, tmp_path):
    matches, players = _inputs(tmp_path)
    soup = BeautifulSoup(
        Path("tests/test_data/afl_tables_match.html").read_text(), "html.parser"
    )
    first_table = scrape_tables._get_match_stat_tables(soup)[0][1]
    first_table.find("tbody").find_all("tr")[0].decompose()
    _responses(monkeypatch, str(soup))
    output_dir = tmp_path / "history"

    with pytest.raises(ValueError, match="expected 22 unique players per team"):
        player_history.build_player_history(
            matches_csv=matches,
            existing_players_csv=players,
            seasons=range(2012, 2013),
            work_dir=output_dir,
        )

    assert not (output_dir / "players.csv").exists()
    assert not (output_dir / "manifest.json").exists()


def test_date_identity_allows_an_unpadded_day():
    expected = player_history.datetime.fromisoformat("2012-04-01T13:10:00+10:00")

    assert player_history._same_date("1-Apr-2012", expected)
    assert not player_history._same_date("2-Apr-2012", expected)
