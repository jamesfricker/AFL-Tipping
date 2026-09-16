"""Build a checked AFL Tables player history file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from itertools import chain
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

from src.mae_model.data import (
    MatchRow,
    canonical_team_name,
    load_matches_csv,
    parse_match_date,
)
from src.mae_model.player_margin import load_player_matches_csv
from src.scrape_afl import scrape_tables


_VERSION = 1
_SEASON_URL = "https://afltables.com/afl/seas/{year}.html"
_MODEL_HEADERS = {
    "Player",
    "KI",
    "MK",
    "HB",
    "GL",
    "BH",
    "HO",
    "TK",
    "CL",
    "CG",
    "CP",
    "GA",
    "%P",
}
_MODEL_NUMERIC_FIELDS = {
    "kicks",
    "marks",
    "handballs",
    "goals",
    "behinds",
    "hit_outs",
    "tackles",
    "clearances",
    "clangers",
    "contested_possessions",
    "goal_assists",
    "percent_played",
}
_CSV_NUMERIC_FIELDS = set(scrape_tables.PLAYER_HEADER_TO_KEY.values()) - {
    "player_name",
    "jumper_number",
}


@dataclass(frozen=True)
class PlayerHistoryBuild:
    players_csv: Path
    report_json: Path
    added_matches: int
    added_players: int


@dataclass(frozen=True)
class _CheckedMatchPlayers:
    match_id: str
    source_url: str
    players: tuple[dict[str, object], ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: missing or duplicate CSV headers")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return list(reader.fieldnames), rows


def _new_cache_index(
    matches_csv: Path, existing_players_csv: Path, seasons: tuple[int, ...]
) -> dict:
    return {
        "version": _VERSION,
        "matches_csv_sha256": _file_sha256(matches_csv),
        "existing_players_csv_sha256": _file_sha256(existing_players_csv),
        "seasons": list(seasons),
        "sources": {},
    }


def _load_cache_index(
    output_dir: Path,
    matches_csv: Path,
    existing_players_csv: Path,
    seasons: tuple[int, ...],
) -> tuple[Path, dict]:
    index_path = output_dir / "cache-index.json"
    expected = _new_cache_index(matches_csv, existing_players_csv, seasons)
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for key in (
            "version",
            "matches_csv_sha256",
            "existing_players_csv_sha256",
            "seasons",
        ):
            if index.get(key) != expected[key]:
                raise ValueError(
                    "The output directory belongs to different inputs or seasons"
                )
        if not isinstance(index.get("sources"), dict):
            raise ValueError("The cache index has an invalid source list")
        return index_path, index
    _write_atomic(index_path, _json_bytes(expected))
    return index_path, expected


def _cached_page(
    *, url: str, kind: str, output_dir: Path, index_path: Path, index: dict
) -> bytes:
    source = index["sources"].get(url)
    if source and source.get("kind") == kind:
        cached_path = output_dir / source["path"]
        if cached_path.is_file():
            content = cached_path.read_bytes()
            if _sha256(content) == source.get("sha256"):
                return content

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    content = response.text.encode("utf-8")
    plural = "matches" if kind == "match" else f"{kind}s"
    relative = Path("cache") / plural / f"{_sha256(url.encode())}.html"
    cached_path = output_dir / relative
    _write_atomic(cached_path, content)
    index["sources"][url] = {
        "kind": kind,
        "path": relative.as_posix(),
        "sha256": _sha256(content),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_atomic(index_path, _json_bytes(index))
    return content


def _table_headers(table) -> set[str]:
    head = table.find("thead")
    rows = head.find_all("tr") if head else []
    if len(rows) < 2:
        return set()
    return {
        scrape_tables._normalize_whitespace(cell.get_text(" ", strip=True))
        for cell in rows[1].find_all("th")
    }


def _same_score(parsed: object, expected: float) -> bool:
    try:
        return math.isclose(float(parsed), expected)
    except (TypeError, ValueError):
        return False


def _same_date(parsed: object, expected: datetime) -> bool:
    try:
        return parse_match_date(str(parsed)).date() == expected.date()
    except ValueError:
        return False


def _check_match_players(
    soup: BeautifulSoup, expected: MatchRow, source_url: str
) -> _CheckedMatchPlayers:
    stat_tables = scrape_tables._get_match_stat_tables(soup)
    if len(stat_tables) != 2:
        raise ValueError(
            f"{expected.match_id} at {source_url}: expected two player tables"
        )
    for source_team, table in stat_tables:
        missing = _MODEL_HEADERS - _table_headers(table)
        if missing:
            raise ValueError(
                f"{expected.match_id} at {source_url}: {source_team} is missing "
                f"model columns {sorted(missing)}"
            )

    try:
        bundle = scrape_tables.get_match_bundle(soup)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{expected.match_id} at {source_url}: {exc}") from exc
    parsed = bundle["match"]
    identity_checks = {
        "match ID": parsed.get("match_id") == expected.match_id,
        "season": int(parsed.get("year", 0)) == expected.year,
        "local date": _same_date(parsed.get("date"), expected.date),
        "home team": canonical_team_name(parsed.get("home_team_name", ""))
        == expected.home_team,
        "away team": canonical_team_name(parsed.get("away_team_name", ""))
        == expected.away_team,
        "home score": _same_score(parsed.get("home_team_score"), expected.home_score),
        "away score": _same_score(parsed.get("away_team_score"), expected.away_score),
    }
    failed = [name for name, passed in identity_checks.items() if not passed]
    if failed:
        raise ValueError(
            f"{expected.match_id} at {source_url}: parsed match differs in {failed}"
        )

    expected_teams = {expected.home_team, expected.away_team}
    by_team: dict[str, list[dict[str, object]]] = {
        expected.home_team: [],
        expected.away_team: [],
    }
    all_refs: dict[str, str] = {}
    for row in bundle["players"]:
        team = canonical_team_name(str(row.get("team_name", "")))
        if team not in expected_teams:
            raise ValueError(
                f"{expected.match_id} at {source_url}: unknown player team {team!r}"
            )
        player_ref = str(row.get("player_ref", "")).strip()
        if not player_ref:
            raise ValueError(
                f"{expected.match_id} at {source_url}: player reference is empty"
            )
        if player_ref in all_refs:
            other_team = all_refs[player_ref]
            detail = "both teams" if other_team != team else f"{team} more than once"
            raise ValueError(
                f"{expected.match_id} at {source_url}: {player_ref} appears for {detail}"
            )
        all_refs[player_ref] = team
        for field in _MODEL_NUMERIC_FIELDS:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{expected.match_id} at {source_url}: invalid {field} for {player_ref}"
                )
        if float(row["percent_played"]) > 100:
            raise ValueError(
                f"{expected.match_id} at {source_url}: invalid percent_played for {player_ref}"
            )
        checked = dict(row)
        checked["team_name"] = team
        checked["opponent_name"] = (
            expected.away_team if team == expected.home_team else expected.home_team
        )
        checked["home_team_name"] = expected.home_team
        checked["away_team_name"] = expected.away_team
        checked["statistics_available_at"] = expected.available_at.isoformat()
        by_team[team].append(checked)

    counts = {team: len(rows) for team, rows in by_team.items()}
    if any(count != 22 for count in counts.values()):
        raise ValueError(
            f"{expected.match_id} at {source_url}: expected 22 unique players per team, "
            f"got {counts}"
        )
    players = tuple(by_team[expected.home_team] + by_team[expected.away_team])
    return _CheckedMatchPlayers(expected.match_id, source_url, players)


def _normal_value(field: str, value: object) -> str:
    text = str(value).strip()
    if field in ("subbed_on", "subbed_off"):
        return text.lower()
    if field in _CSV_NUMERIC_FIELDS and text:
        try:
            return str(Decimal(text).normalize())
        except InvalidOperation:
            return text
    if field in ("team_name", "opponent_name", "home_team_name", "away_team_name"):
        return canonical_team_name(text)
    return text


def _merge_rows(
    fields: list[str], existing: list[dict[str, str]], added: Iterable[dict[str, object]]
) -> tuple[list[dict[str, str]], int, int, int]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    added_keys: set[tuple[str, str]] = set()
    added_match_ids: set[str] = set()
    exact_duplicates = 0
    rows = chain(
        ((row, False) for row in existing),
        ((row, True) for row in added),
    )
    for raw, is_added in rows:
        row = {field: str(raw.get(field, "")) for field in fields}
        key = (row["match_id"].strip(), row["player_ref"].strip())
        if not all(key):
            raise ValueError("A player row has an empty match ID or player reference")
        if key in merged:
            prior = merged[key]
            if any(
                _normal_value(field, prior[field]) != _normal_value(field, row[field])
                for field in fields
            ):
                raise ValueError(f"Conflicting duplicate player row: {key}")
            exact_duplicates += 1
            continue
        merged[key] = row
        if is_added:
            added_keys.add(key)
            added_match_ids.add(key[0])
    return list(merged.values()), len(added_match_ids), len(added_keys), exact_duplicates


def _csv_bytes(
    fields: list[str], rows: list[dict[str, str]], *, line_ending: str
) -> bytes:
    from io import StringIO

    target = StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields, lineterminator=line_ending)
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode()


def _completed_result(
    players_csv: Path, report_json: Path
) -> PlayerHistoryBuild | None:
    if not players_csv.exists() and not report_json.exists():
        return None
    if not players_csv.exists() or not report_json.exists():
        return None
    report = json.loads(report_json.read_text())
    if _file_sha256(players_csv) != report.get("output_sha256"):
        raise ValueError("The completed player history does not match its manifest")
    if _file_sha256(Path(__file__)) != report.get("builder_sha256"):
        return None
    return PlayerHistoryBuild(
        players_csv,
        report_json,
        int(report["added_matches"]),
        int(report["added_players"]),
    )


def build_player_history(
    *,
    matches_csv: Path,
    existing_players_csv: Path,
    seasons: range,
    work_dir: Path,
) -> PlayerHistoryBuild:
    seasons_tuple = tuple(seasons)
    if not seasons_tuple or any(year < 1897 for year in seasons_tuple):
        raise ValueError("Supply at least one valid AFL season")
    if seasons_tuple != tuple(range(seasons_tuple[0], seasons_tuple[-1] + 1)):
        raise ValueError("Seasons must be a continuous ascending range")
    matches_csv = matches_csv.resolve()
    existing_players_csv = existing_players_csv.resolve()
    work_dir = work_dir.resolve()
    players_csv = work_dir / "players.csv"
    report_json = work_dir / "manifest.json"
    generated_paths = {players_csv, report_json, work_dir / "cache-index.json"}
    if matches_csv in generated_paths or existing_players_csv in generated_paths:
        raise ValueError("Generated files must not overwrite an input")
    work_dir.mkdir(parents=True, exist_ok=True)
    index_path, index = _load_cache_index(
        work_dir, matches_csv, existing_players_csv, seasons_tuple
    )
    completed = _completed_result(players_csv, report_json)
    if completed:
        return completed
    players_csv.unlink(missing_ok=True)
    report_json.unlink(missing_ok=True)

    all_matches = load_matches_csv(str(matches_csv))
    expected = {
        match.match_id: match for match in all_matches if match.year in seasons_tuple
    }
    counts_by_season = {
        year: sum(match.year == year for match in expected.values())
        for year in seasons_tuple
    }
    missing_seasons = [year for year, count in counts_by_season.items() if count == 0]
    if missing_seasons:
        raise ValueError(f"The match file has no matches for seasons {missing_seasons}")

    links: list[str] = []
    for year in seasons_tuple:
        url = _SEASON_URL.format(year=year)
        content = _cached_page(
            url=url,
            kind="season",
            output_dir=work_dir,
            index_path=index_path,
            index=index,
        )
        soup = BeautifulSoup(content, "html.parser")
        links.extend(scrape_tables.get_all_match_links_from_season(soup))
    links = list(dict.fromkeys(links))

    checked: dict[str, _CheckedMatchPlayers] = {}
    for url in links:
        content = _cached_page(
            url=url,
            kind="match",
            output_dir=work_dir,
            index_path=index_path,
            index=index,
        )
        soup = BeautifulSoup(content, "html.parser")
        try:
            parsed_match = scrape_tables.get_data_from_match(soup)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Could not parse match identity at {url}: {exc}") from exc
        match_id = str(parsed_match["match_id"])
        if match_id not in expected:
            raise ValueError(f"Unexpected source match {match_id} at {url}")
        if match_id in checked:
            raise ValueError(f"More than one source page supplies match {match_id}")
        checked[match_id] = _check_match_players(soup, expected[match_id], url)

    missing_ids = sorted(set(expected) - set(checked))
    if missing_ids:
        raise ValueError(f"Missing source matches: {missing_ids}")

    existing_fields, existing_rows = _read_csv(existing_players_csv)
    required_existing = {"match_id", "player_ref", "team_name", "percent_played"}
    missing_fields = required_existing - set(existing_fields)
    if missing_fields:
        raise ValueError(
            f"{existing_players_csv}: missing CSV columns {sorted(missing_fields)}"
        )
    fields = list(dict.fromkeys(existing_fields + scrape_tables.PLAYER_FIELDS))
    added_rows = [
        row
        for match_id in sorted(checked)
        for row in checked[match_id].players
    ]
    merged_rows, added_matches, added_players, exact_duplicates = _merge_rows(
        fields, existing_rows, added_rows
    )
    line_ending = "\r\n" if b"\r\n" in existing_players_csv.read_bytes() else "\n"
    output_bytes = _csv_bytes(fields, merged_rows, line_ending=line_ending)
    temporary_csv = work_dir / ".players.csv.complete"
    _write_atomic(temporary_csv, output_bytes)

    existing_loaded = load_player_matches_csv(str(existing_players_csv), all_matches)
    merged_loaded = load_player_matches_csv(str(temporary_csv), all_matches)
    existing_by_key = {(row.match_id, row.player_id): row for row in existing_loaded}
    merged_by_key = {(row.match_id, row.player_id): row for row in merged_loaded}
    if any(merged_by_key.get(key) != value for key, value in existing_by_key.items()):
        temporary_csv.unlink(missing_ok=True)
        raise ValueError("Existing model player rows changed during the merge")

    collected_by_season = {
        year: sum(expected[match_id].year == year for match_id in checked)
        for year in seasons_tuple
    }
    source_rows = [
        {"url": url, **source}
        for url, source in sorted(index["sources"].items())
    ]
    report = {
        "version": _VERSION,
        "requested_first_year": seasons_tuple[0],
        "requested_last_year": seasons_tuple[-1],
        "expected_matches": len(expected),
        "collected_matches": len(checked),
        "expected_matches_by_season": counts_by_season,
        "collected_matches_by_season": collected_by_season,
        "missing_match_ids": [],
        "extra_match_ids": [],
        "players_per_team": 22,
        "collected_player_rows": len(added_rows),
        "existing_player_rows": len(existing_rows),
        "merged_player_rows": len(merged_rows),
        "added_matches": added_matches,
        "added_players": added_players,
        "exact_duplicate_rows": exact_duplicates,
        "conflicting_duplicate_rows": 0,
        "duplicate_policy": (
            "Exact rows are merged. Conflicting match_id and player_ref rows fail."
        ),
        "matches_csv_sha256": _file_sha256(matches_csv),
        "existing_players_csv_sha256": _file_sha256(existing_players_csv),
        "builder_sha256": _file_sha256(Path(__file__)),
        "output_sha256": _sha256(output_bytes),
        "sources": source_rows,
        "timing_assumptions": {
            "statistics_available_at": (
                "Use the matched result availability time. When the CSV omits this "
                "column, the model loader uses the same result-availability rule."
            ),
            "default_result_availability": (
                "The match loader assumes the next venue-local day when no "
                "explicit result time exists."
            ),
            "collection_time": (
                "The page fetch time is collection evidence. It is not historical "
                "availability evidence."
            ),
            "lineup": (
                "The players are final participants assumed to be known at kickoff "
                "for this benchmark."
            ),
        },
    }
    temporary_report = work_dir / ".manifest.json.complete"
    _write_atomic(temporary_report, _json_bytes(report))
    temporary_report.replace(report_json)
    temporary_csv.replace(players_csv)
    return PlayerHistoryBuild(players_csv, report_json, added_matches, added_players)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-csv", type=Path, required=True)
    parser.add_argument("--existing-players-csv", type=Path, required=True)
    parser.add_argument("--first-year", type=int, required=True)
    parser.add_argument("--last-year", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.last_year < args.first_year:
        parser.error("--last-year must be at least --first-year")
    result = build_player_history(
        matches_csv=args.matches_csv,
        existing_players_csv=args.existing_players_csv,
        seasons=range(args.first_year, args.last_year + 1),
        work_dir=args.output_dir,
    )
    print(
        f"Wrote {result.added_players} added player rows from "
        f"{result.added_matches} matches to {result.players_csv}"
    )


if __name__ == "__main__":
    main()
