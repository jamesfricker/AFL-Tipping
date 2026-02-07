from types import SimpleNamespace

import pytest
import src.scrape_afl.scrape_tables as scrape_tables
from bs4 import BeautifulSoup


def test_get_soup_data(monkeypatch):
    html = "<html><body><p>ok</p></body></html>"

    def _fake_get(url, timeout):
        return SimpleNamespace(
            text=html,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(scrape_tables.requests, "get", _fake_get)
    soup = scrape_tables.get_soup_data("https://example.com")
    assert soup.find("p").text == "ok"


@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("afl_tables_match.html", "1"),
        ("afl_tables_2012_gf.html", "Grand Final"),
    ],
)
def test_get_match_round(file_name, expected):
    f = open(f"tests/test_data/{file_name}")
    content = f.read()
    soup = BeautifulSoup(content, "html.parser")
    match_round = scrape_tables.get_match_round(soup)
    assert match_round == expected


def test_get_all_match_links_from_season():
    f = open("tests/test_data/afl_tables_2012.html")
    content = f.read()
    soup = BeautifulSoup(content, "html.parser")
    match_links = scrape_tables.get_all_match_links_from_season(soup)
    assert len(match_links) == 207


@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("afl_tables_match.html", "24-Mar-2012"),
        ("afl_tables_nm_gws.html", "8-Apr-2012"),
    ],
)
def test_get_match_date(file_name, expected):
    f = open(f"tests/test_data/{file_name}")
    content = f.read()
    soup = BeautifulSoup(content, "html.parser")
    match_date = scrape_tables.get_match_date(soup)
    assert match_date == expected


def test_get_match_time():
    f = open("tests/test_data/afl_tables_match.html")
    content = f.read()
    soup = BeautifulSoup(content, "html.parser")
    match_date = scrape_tables.get_match_time(soup)
    assert match_date == "7:20 PM"


def test_get_data_from_match():
    f = open("tests/test_data/afl_tables_match.html")
    content = f.read()
    soup = BeautifulSoup(content, "html.parser")
    match_data = scrape_tables.get_data_from_match(soup)
    assert match_data["match_id"] == "162120120324"
    assert match_data["year"] == "2012"
    assert match_data["round"] == "1"
    assert match_data["date"] == "24-Mar-2012"
    assert match_data["venue"] == "Stadium Australia"
    assert match_data["time"] == "7:20 PM"
    assert match_data["home_team_name"] == "Greater Western Sydney"
    assert match_data["home_team_score"] == "37"
    assert match_data["home_goals"] == 5
    assert match_data["home_behinds"] == 7
    assert match_data["home_scoring_shots"] == 12
    assert match_data["away_team_name"] == "Sydney"
    assert match_data["away_team_score"] == "100"
    assert match_data["away_goals"] == 14
    assert match_data["away_behinds"] == 16
    assert match_data["away_scoring_shots"] == 30


def test_get_player_stats_from_match():
    f = open("tests/test_data/afl_tables_match.html")
    content = f.read()
    soup = BeautifulSoup(content, "html.parser")
    player_stats = scrape_tables.get_player_stats_from_match(soup)

    assert len(player_stats) > 40
    first = player_stats[0]
    assert first["match_id"] == "162120120324"
    assert first["team_name"] in ("Greater Western Sydney", "Sydney")
    assert first["player_name"] != ""
    assert first["disposals"] >= 0
    assert isinstance(first["subbed_on"], bool)
    assert isinstance(first["subbed_off"], bool)

    names = {p["player_name"] for p in player_stats}
    assert "Bugg, Tomas" in names
    assert "Goodes, Adam" in names


def test_get_match_bundle():
    f = open("tests/test_data/afl_tables_match.html")
    content = f.read()
    soup = BeautifulSoup(content, "html.parser")
    bundle = scrape_tables.get_match_bundle(soup)

    assert "match" in bundle
    assert "players" in bundle
    assert bundle["match"]["match_id"] == "162120120324"
    assert len(bundle["players"]) > 40
