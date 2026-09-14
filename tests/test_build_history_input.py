import csv
from datetime import datetime
from pathlib import Path

import pytest

from src.mae_model.data import load_matches_csv
from src.scrape_afl.build_history_input import build_model_input
from src.scrape_afl.scrape_tables import MATCH_FIELDS
from src.scrape_afl.season_scores import parse_season_scores


FIXTURES = Path(__file__).parent / "fixtures" / "season_scores"


def write_rows(path, rows, fields=None):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or list(rows[0]), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def inputs(tmp_path):
    historical = []
    for year in (1900, 1996, 2006):
        rows, _ = parse_season_scores(
            (FIXTURES / f"{year}_exception_match.html").read_bytes(), year
        )
        historical.extend(rows)
    modern, _ = parse_season_scores(
        (FIXTURES / "2012_round1_match.html").read_bytes(), 2012
    )
    return write_rows(tmp_path / "history.csv", historical), write_rows(
        tmp_path / "current.csv", modern, MATCH_FIELDS
    )


def test_timing_exceptions_and_current_data_preservation(tmp_path):
    history, current = inputs(tmp_path)
    original = current.read_bytes()
    output = tmp_path / "combined.csv"
    manifest = build_model_input(history, current, output)
    rows = {match.match_id: match for match in load_matches_csv(str(output))}
    assert len(rows) == 4
    assert rows["111519000505"].available_at == datetime.fromisoformat(
        "1901-01-01T00:00:00+10:00"
    )
    assert rows["051519960608"].available_at == datetime.fromisoformat(
        "1996-06-12T00:00:00+10:00"
    )
    assert rows["081520060430"].available_at == datetime.fromisoformat(
        "2006-05-03T18:08:00+10:00"
    )
    assert rows["162120120324"] == load_matches_csv(str(current))[0]
    assert current.read_bytes() == original
    assert manifest["original_current_fields_unchanged"] is True
    assert (
        manifest["venue_timezones"]["Waverley Park"]["timezone"]
        == "Australia/Melbourne"
    )
    assert manifest["total_matches"] == 4


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("venue", "Unknown Oval", "Unknown timezone"),
        ("time_status", "unknown", "Unknown time"),
        ("year", "2013", "must precede"),
    ],
)
def test_unusable_history_fails_before_output(tmp_path, field, value, message):
    history, current = inputs(tmp_path)
    with history.open() as handle:
        rows = list(csv.DictReader(handle))
    rows[0][field] = value
    write_rows(history, rows)
    output = tmp_path / "combined.csv"
    with pytest.raises(ValueError, match=message):
        build_model_input(history, current, output)
    assert not output.exists()


def test_output_cannot_overwrite_input(tmp_path):
    history, current = inputs(tmp_path)
    original = current.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        build_model_input(history, current, current)
    assert current.read_bytes() == original


def test_invalid_score_does_not_replace_existing_output(tmp_path):
    history, current = inputs(tmp_path)
    with history.open() as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["home_team_score"] = "bad score"
    write_rows(history, rows)
    output = tmp_path / "combined.csv"
    output.write_text("previous valid output\n")
    with pytest.raises(ValueError):
        build_model_input(history, current, output)
    assert output.read_text() == "previous valid output\n"
    assert not list(tmp_path.glob("tmp*"))


def test_output_symlink_cannot_overwrite_input(tmp_path):
    history, current = inputs(tmp_path)
    original = current.read_bytes()
    alias = tmp_path / "alias.csv"
    alias.symlink_to(current)
    with pytest.raises(ValueError, match="must not overwrite"):
        build_model_input(history, current, alias)
    assert current.read_bytes() == original


@pytest.mark.parametrize(
    "field,message",
    [("id", "Missing historical result exception"), ("year", "exception year differs")],
)
def test_mistyped_exception_metadata_is_rejected(tmp_path, monkeypatch, field, message):
    import json
    from src.scrape_afl import build_history_input

    history, current = inputs(tmp_path)
    maps = tmp_path / "maps"
    maps.mkdir()
    for filename in (
        "historical_venue_timezones.json",
        "historical_result_availability.json",
    ):
        (maps / filename).write_bytes(
            (build_history_input.DATA_DIR / filename).read_bytes()
        )
    path = maps / "historical_result_availability.json"
    exceptions = json.loads(path.read_text())
    if field == "id":
        exceptions["typo"] = exceptions.pop("051519960608")
    else:
        exceptions["051519960608"]["year"] = 1997
    path.write_text(json.dumps(exceptions))
    monkeypatch.setattr(build_history_input, "DATA_DIR", maps)
    with pytest.raises(ValueError, match=message):
        build_model_input(history, current, tmp_path / "combined.csv")


def test_subset_does_not_require_exceptions_from_other_seasons(tmp_path):
    history, current = inputs(tmp_path)
    with history.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["year"] == "1996"]
    write_rows(history, rows)
    manifest = build_model_input(history, current, tmp_path / "combined.csv")
    assert set(manifest["result_availability_exceptions"]) == {"051519960608"}
    assert all(
        len(manifest[key]) == 64
        for key in ("builder_sha256", "venue_map_sha256", "availability_map_sha256")
    )
