from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from src.mae_model.data import MatchRow, canonical_team_name, load_matches_csv


BASE_URL = "https://www.wheeloratings.com/src/match_stats/table_data"
OUTPUT_FIELDS = (
    "match_id",
    "year",
    "round",
    "team_name",
    "player_name",
    "player_ref",
    "percent_played",
    "official_rating_points",
    "statistics_available_at",
    "source_match_id",
)


@dataclass(frozen=True)
class SourceMatchIdentity:
    source_match_id: str
    year: int
    round_number: int
    round_name: str
    match_date: str
    home_team: str
    away_team: str


@dataclass(frozen=True)
class SourceAppearance:
    source_match_id: str
    team: str
    player_name: str
    player_ref: str
    percent_played: float
    source_time_on_ground: float
    official_rating_points: float | None


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(body)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "AFL player-rating research"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def _cached_document(url: str, cache_dir: Path, refresh: bool) -> tuple[bytes, dict]:
    key = _sha256(url.encode())
    request_path = cache_dir / "requests" / f"{key}.json"
    previous = json.loads(request_path.read_text()) if request_path.exists() else None
    if previous and not refresh:
        if previous.get("url") != url:
            raise ValueError(f"Cached URL key collision: {url}")
        body_path = cache_dir / previous["body_path"]
        try:
            body = body_path.read_bytes()
        except FileNotFoundError as exc:
            raise ValueError(f"Cached body is missing for {url}") from exc
        if _sha256(body) != previous.get("sha256"):
            raise ValueError(f"Cached body changed for {url}; use --refresh")
        return body, previous

    body = _download(url)
    try:
        json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Source did not return JSON: {url}") from exc
    digest = _sha256(body)
    body_path = cache_dir / "bodies" / f"{digest}.json"
    if body_path.exists() and _sha256(body_path.read_bytes()) != digest:
        raise ValueError(f"Content-addressed body changed: {body_path}")
    if not body_path.exists():
        _atomic_write(body_path, body)
    fetched_at = datetime.now(timezone.utc).isoformat()
    record = {
        "url": url,
        "sha256": digest,
        "body_path": str(body_path.relative_to(cache_dir)),
        "fetched_at": fetched_at,
        "previous_sha256": previous.get("sha256") if previous else None,
    }
    _atomic_write(request_path, _json_bytes(record))
    return body, record


def _column_rows(value, label: str) -> list[dict]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(f"{label} must contain one column object")
    columns = value[0]
    lengths = {key: len(items) for key, items in columns.items() if isinstance(items, list)}
    if not lengths:
        return [columns]
    if len(lengths) != len(columns) or len(set(lengths.values())) != 1:
        raise ValueError(f"{label} columns have different lengths")
    size = next(iter(lengths.values()), 0)
    return [{key: columns[key][index] for key in columns} for index in range(size)]


def _text(value, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"Missing {field}")
    return result


def _finite(value, field: str, *, allow_blank: bool = False) -> float | None:
    if value is None or str(value).strip() == "":
        if allow_blank:
            return None
        raise ValueError(f"Missing {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Invalid {field}: {value!r}")
    return result


def parse_round_document(body: bytes, year: int) -> tuple[list[SourceMatchIdentity], list[SourceAppearance]]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid round JSON") from exc
    summary = payload.get("Summary")
    if not isinstance(summary, list) or len(summary) != 1:
        raise ValueError("Round Summary must contain one row")
    if int(summary[0]["Season"]) != year:
        raise ValueError("Round season differs from requested year")
    round_number = int(summary[0]["RoundNumber"])
    round_name = _text(summary[0]["RoundName"], "RoundName")
    matches = []
    for row in _column_rows(payload.get("Matches"), "Matches"):
        matches.append(
            SourceMatchIdentity(
                _text(row.get("MatchId"), "MatchId"),
                year,
                round_number,
                round_name,
                _text(row.get("MatchDate"), "MatchDate"),
                canonical_team_name(_text(row.get("HomeTeam"), "HomeTeam")),
                canonical_team_name(_text(row.get("AwayTeam"), "AwayTeam")),
            )
        )
    match_ids = {row.source_match_id for row in matches}
    if len(match_ids) != len(matches):
        raise ValueError("Duplicate source match identity")
    appearances = []
    for row in _column_rows(payload.get("Data"), "Data"):
        source_match_id = _text(row.get("MatchId"), "MatchId")
        if source_match_id not in match_ids:
            raise ValueError(f"Player row has unknown source match: {source_match_id}")
        website_id = _text(row.get("WebsiteId"), "WebsiteId")
        time_on_ground = _finite(row.get("TimeOnGround"), "TimeOnGround")
        if time_on_ground < 0 or time_on_ground > 200:
            raise ValueError(f"TimeOnGround outside 0 to 200: {time_on_ground}")
        appearances.append(
            SourceAppearance(
                source_match_id,
                canonical_team_name(_text(row.get("Team"), "Team")),
                _text(row.get("Player"), "Player"),
                f"wheelo:{website_id}",
                min(100.0, time_on_ground),
                time_on_ground,
                _finite(row.get("RatingPoints"), "RatingPoints", allow_blank=True),
            )
        )
    return matches, appearances


def _source_date(year: int, value: str):
    try:
        return datetime.strptime(f"{value} {year}", "%d %b %Y").date()
    except ValueError as exc:
        raise ValueError(f"Invalid source match date: {value!r}") from exc


def join_source_matches(
    source_matches: list[SourceMatchIdentity], matches: list[MatchRow]
) -> dict[str, MatchRow]:
    by_identity = defaultdict(list)
    for match in matches:
        key = (
            match.year,
            match.fixture.kickoff.date(),
            frozenset((match.home_team, match.away_team)),
        )
        by_identity[key].append(match)
    joined = {}
    for source in source_matches:
        key = (
            source.year,
            _source_date(source.year, source.match_date),
            frozenset((source.home_team, source.away_team)),
        )
        candidates = by_identity.get(key, [])
        if len(candidates) != 1:
            raise ValueError(
                f"Source match join is {'missing' if not candidates else 'ambiguous'}: "
                f"{source.source_match_id} candidates={len(candidates)}"
            )
        if source.source_match_id in joined:
            raise ValueError(f"Duplicate joined source match: {source.source_match_id}")
        joined[source.source_match_id] = candidates[0]
    return joined


def build_wheelo_player_history(
    matches: list[MatchRow],
    first_year: int,
    last_year: int,
    output_dir: Path,
    *,
    refresh: bool = False,
    base_url: str = BASE_URL,
) -> dict:
    if first_year > last_year:
        raise ValueError("first_year must not be after last_year")
    selected_matches = [m for m in matches if first_year <= m.year <= last_year]
    if not selected_matches:
        raise ValueError("No repository matches in requested years")
    cache_dir = output_dir / "source"
    all_source_matches = []
    all_appearances = []
    source_records = []
    round_ids_seen = set()
    for year in range(first_year, last_year + 1):
        index_url = f"{base_url}/{year}.json"
        index_body, record = _cached_document(index_url, cache_dir, refresh)
        source_records.append(record)
        index = json.loads(index_body)
        round_ids = index.get("RoundId")
        if not isinstance(round_ids, list) or not round_ids:
            raise ValueError(f"Season index has no rounds: {year}")
        if len(round_ids) != len(set(round_ids)):
            raise ValueError(f"Season index has duplicate rounds: {year}")
        for round_id in round_ids:
            round_id = _text(round_id, "RoundId")
            if not round_id.startswith(str(year)):
                raise ValueError(f"Round ID differs from season: {round_id}")
            if round_id in round_ids_seen:
                raise ValueError(f"Round appears in two indexes: {round_id}")
            round_ids_seen.add(round_id)
            round_url = f"{base_url}/{round_id}.json"
            body, record = _cached_document(round_url, cache_dir, refresh)
            source_records.append(record)
            round_matches, appearances = parse_round_document(body, year)
            all_source_matches.extend(round_matches)
            all_appearances.extend(appearances)

    joined = join_source_matches(all_source_matches, selected_matches)
    source_ids = set(joined)
    repository_ids = {match.match_id for match in selected_matches}
    matched_repository_ids = {match.match_id for match in joined.values()}
    if matched_repository_ids != repository_ids:
        missing = sorted(repository_ids - matched_repository_ids)
        extra = sorted(matched_repository_ids - repository_ids)
        raise ValueError(
            f"Source coverage differs from repository: missing={missing[:5]} extra={extra[:5]}"
        )
    appearance_source_ids = {row.source_match_id for row in all_appearances}
    if appearance_source_ids != source_ids:
        raise ValueError("Player rows and source matches cover different match IDs")

    rows_by_key = {}
    team_counts = Counter()
    rating_missing = Counter()
    players = set()
    for appearance in all_appearances:
        match = joined[appearance.source_match_id]
        if appearance.team not in (match.home_team, match.away_team):
            raise ValueError(
                f"Player team is not in joined match: {appearance.source_match_id} {appearance.team}"
            )
        row = {
            "match_id": match.match_id,
            "year": match.year,
            "round": match.round_label,
            "team_name": appearance.team,
            "player_name": appearance.player_name,
            "player_ref": appearance.player_ref,
            "percent_played": f"{appearance.percent_played:g}",
            "official_rating_points": ""
            if appearance.official_rating_points is None
            else f"{appearance.official_rating_points:g}",
            "statistics_available_at": match.available_at.isoformat(),
            "source_match_id": appearance.source_match_id,
        }
        key = (match.match_id, appearance.player_ref)
        if key in rows_by_key and rows_by_key[key] != row:
            raise ValueError(f"Conflicting duplicate player appearance: {key}")
        rows_by_key[key] = row
        team_counts[(match.match_id, appearance.team)] += 1
        rating_missing[match.year] += appearance.official_rating_points is None
        players.add(appearance.player_ref)
    invalid_counts = {
        f"{match_id}:{team}": count
        for (match_id, team), count in team_counts.items()
        if count not in (22, 23)
    }
    if invalid_counts:
        sample = dict(list(sorted(invalid_counts.items()))[:5])
        raise ValueError(f"Invalid selected-team player counts: {sample}")
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (int(row["year"]), row["match_id"], row["team_name"], row["player_ref"]),
    )
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_buffer.seek(0)
    csv_body = csv_buffer.read().encode()
    csv_path = output_dir / "afl_player_ratings.csv"
    manifest_path = output_dir / "manifest.json"
    audit = {
        "status": "pass",
        "requested_years": [first_year, last_year],
        "matches": len(repository_ids),
        "source_matches": len(source_ids),
        "player_rows": len(rows),
        "unique_players": len(players),
        "team_size_counts": dict(sorted(Counter(team_counts.values()).items())),
        "missing_ratings_by_year": dict(sorted(rating_missing.items())),
        "rating_policy": "Blank stays missing. Finite zero and negative values are retained.",
        "time_on_ground_values_capped_at_100": sum(
            row.source_time_on_ground > 100 for row in all_appearances
        ),
        "time_on_ground_policy": "Finite source values above 100 are capped at 100 for the existing percent-played boundary.",
        "availability_policy": "Historical ratings become available with the repository match result.",
        "lineup_policy": "All 22 or 23 source player identities are assumed final at kickoff.",
        "excluded_fields": ["EstimatedRating", "season summary estimates"],
        "join_fields": ["year", "local_date", "unordered_team_pair"],
        "output_sha256": _sha256(csv_body),
        "sources": source_records,
    }
    _atomic_write(csv_path, csv_body)
    _atomic_write(manifest_path, _json_bytes(audit))
    return audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build checked historical AFL player Rating Points data."
    )
    parser.add_argument("--matches-csv", default="src/outputs/afl_data.csv")
    parser.add_argument("--first-year", type=int, default=2012)
    parser.add_argument("--last-year", type=int, default=2025)
    parser.add_argument("--output-dir", default=".context/wheelo-player-history")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    try:
        matches = load_matches_csv(args.matches_csv)
        audit = build_wheelo_player_history(
            matches,
            args.first_year,
            args.last_year,
            Path(args.output_dir),
            refresh=args.refresh,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({key: audit[key] for key in ("status", "matches", "player_rows", "unique_players")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
