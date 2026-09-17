import csv
from pathlib import Path

import pytest

from src.scrape_afl.scrape_tables import MATCH_FIELDS
from src.scrape_afl.season_scores import import_seasons, parse_season_scores
from src.scrape_afl.verify_season_import import verify_import


FIXTURES = Path(__file__).parent / "fixtures" / "season_scores"


def prepare_import(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for year in (1897, 2012):
        (cache / f"{year}.html").write_bytes(
            (FIXTURES / f"{year}_round1_match.html").read_bytes()
        )
    import_seasons(1897, 1897, tmp_path)
    modern, _ = parse_season_scores((cache / "2012.html").read_bytes(), 2012)
    current = tmp_path / "current.csv"
    with current.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(modern)
    return cache, current


def test_verification_replays_sources_and_detects_cache_change(tmp_path):
    cache, current = prepare_import(tmp_path)
    result = verify_import(tmp_path, current)
    assert result["verified_imported_matches"] == 1
    assert result["overlap_matches"] == 1
    assert len(result["paired_id_sha256"]) == 64
    (cache / "1897.html").write_text("corrupt source")
    with pytest.raises(ValueError, match="Cache hash mismatch"):
        verify_import(tmp_path, current)


@pytest.mark.parametrize("duplicate", [False, True])
def test_manifest_must_include_each_year_once(tmp_path, duplicate):
    import json

    _, current = prepare_import(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"] = manifest["sources"] * 2 if duplicate else []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source years"):
        verify_import(tmp_path, current)


def test_missing_overlap_page_has_clear_error(tmp_path):
    cache, current = prepare_import(tmp_path)
    (cache / "2012.html").unlink()
    with pytest.raises(ValueError, match="Save the 2012 overlap source"):
        verify_import(tmp_path, current)
