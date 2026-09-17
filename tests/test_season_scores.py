from pathlib import Path

import pytest

from src.scrape_afl.season_scores import import_seasons, parse_season_scores


FIXTURES = Path(__file__).parent / "fixtures" / "season_scores"


def sample(year=1897):
    return (FIXTURES / f"{year}_round1_match.html").read_text()


def test_original_club_identity_and_scoring_split():
    rows, check = parse_season_scores(sample(), 1897)
    assert len(rows) == 1
    assert rows[0] == {
        "match_id": "030618970508",
        "year": 1897,
        "round": "1",
        "date": "08-May-1897",
        "venue": "Brunswick St",
        "time": "3:00 PM",
        "time_status": "source_local_time",
        "source_url": "https://afltables.com/afl/stats/games/1897/030618970508.html",
        "venue_source_url": "https://afltables.com/afl/venues/brunswick_st.html",
        "result_note": "Fitzroy won by 33 pts",
        "home_team_name": "Fitzroy",
        "home_team_score": 49,
        "home_goals": 6,
        "home_behinds": 13,
        "home_scoring_shots": 19,
        "away_team_name": "Carlton",
        "away_team_score": 16,
        "away_goals": 2,
        "away_behinds": 4,
        "away_scoring_shots": 6,
    }
    assert check["source_match_links"] == 1


def test_local_time_precedes_eastern_time_in_parentheses():
    rows, _ = parse_season_scores(sample(2012), 2012)
    assert rows[0]["time"] == "7:20 PM"
    assert rows[0]["match_id"] == "162120120324"
    assert rows[0]["home_team_name"] == "Greater Western Sydney"
    assert rows[0]["away_team_name"] == "Sydney"
    assert rows[0]["home_team_score"] == 37
    assert rows[0]["away_team_score"] == 100


def test_round_heading_can_include_a_notes_link():
    html = sample(2012).replace(
        "Round 1", 'Round 1<a href="/afl/notes.html"> arbitrary link text</a>', 1
    )

    rows, _ = parse_season_scores(html, 2012)

    assert rows[0]["round"] == "1"


def test_unknown_start_time_stays_unknown():
    rows, check = parse_season_scores(sample().replace("3:00 PM", ""), 1897)
    assert rows[0]["time"] == ""
    assert rows[0]["time_status"] == "unknown"
    assert check["unknown_times"] == 1


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda html: html + html, "unique source IDs"),
        (lambda html: html.replace("> 49<", "> 50<"), "score does not equal"),
        (lambda html: html.replace("08-May-1897", "09-May-1897"), "source date"),
        (lambda html: html.replace("../stats/games/", "../not-games/"), "source ID"),
        (
            lambda html: html + "Totals Games: 2, Goals: 8, Behinds: 17",
            "regular season totals",
        ),
    ],
)
def test_invalid_source_fails(mutate, message):
    with pytest.raises(ValueError, match=message):
        parse_season_scores(mutate(sample()), 1897)


def test_sectional_matches_are_separate_from_regular_ladder_totals():
    html = sample() + "Totals Games: 1, Goals: 8, Behinds: 17"
    html += (
        sample()
        .replace("Round 1", "Sectional Round 1")
        .replace("030618970508", "030618970515")
        .replace("08-May-1897", "15-May-1897")
    )
    rows, check = parse_season_scores(html, 1897)
    assert [row["round"] for row in rows] == ["1", "Sectional Round 1"]
    assert check["matches"] == 2
    assert check["source_regular_games"] == 1


def test_cached_import_writes_reproducible_data_and_source_hashes(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "1897.html").write_text(sample())

    def no_network(*args, **kwargs):
        raise AssertionError("Cached import must not use the network")

    monkeypatch.setattr("requests.Session.get", no_network)
    first = import_seasons(1897, 1897, tmp_path)
    second = import_seasons(1897, 1897, tmp_path)
    assert first == second
    assert first["matches"] == 1
    assert len(first["csv_sha256"]) == 64
    assert len(first["sources"][0]["sha256"]) == 64
    assert "Fitzroy" in (tmp_path / "season_scores.csv").read_text()


def test_competition_notes_are_kept():
    html = (
        sample() + "<table><tr><td>*Club forfeited points, Round 6.</td></tr></table>"
    )
    _, check = parse_season_scores(html, 1897)
    assert check["source_notes"] == ["*Club forfeited points, Round 6."]
