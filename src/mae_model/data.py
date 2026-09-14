import csv
import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .venues import get_venue_meta

TEAM_ALIASES = {
    "South Melbourne": "Sydney",
    "Footscray": "Western Bulldogs",
    "North Melbourne": "Kangaroos",
    "North Melbourne Kangaroos": "Kangaroos",
    "Brisbane": "Brisbane Lions",
    "GWS Giants": "Greater Western Sydney",
}


def canonical_team_name(team_name: str) -> str:
    name = team_name.strip()
    return TEAM_ALIASES.get(name, name)


def parse_match_date(raw: str) -> datetime:
    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {raw}")


def parse_timestamp(raw: str) -> datetime:
    stamp = datetime.fromisoformat(raw)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone offset")
    return stamp


def _aware(stamp: datetime, field: str) -> None:
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone offset")


def _number(value, field: str, *, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(
            f"{field} must be finite"
            + (f" and at least {minimum}" if minimum is not None else "")
        )
    return result


@dataclass(frozen=True)
class Fixture:
    match_id: str
    year: int
    round_label: str
    kickoff: datetime
    venue: str
    home_team: str
    away_team: str

    def __post_init__(self):
        _aware(self.kickoff, "kickoff")
        if (
            not self.match_id.strip()
            or not self.home_team.strip()
            or not self.away_team.strip()
        ):
            raise ValueError("Fixture needs a match ID and two team names")
        if self.home_team == self.away_team:
            raise ValueError("Fixture teams must differ")
        meta = get_venue_meta(self.venue)
        local_kickoff = (
            self.kickoff.astimezone(ZoneInfo(meta.timezone)) if meta else self.kickoff
        )
        if self.year != local_kickoff.year:
            raise ValueError("Fixture year must match its local kickoff year")


@dataclass(frozen=True)
class MatchRow:
    match_id: str
    year: int
    round_label: str
    date: datetime
    venue: str
    home_team: str
    away_team: str
    home_score: float
    away_score: float
    home_goals: int = 0
    home_behinds: int = 0
    away_goals: int = 0
    away_behinds: int = 0
    home_scoring_shots: int = 0
    away_scoring_shots: int = 0
    result_available_at: datetime | None = None

    @property
    def timing_assumption(self) -> str:
        return (
            "explicit_timestamp"
            if self.result_available_at is not None
            else "next_local_day"
        )

    @property
    def actual_margin(self) -> float:
        return self.home_score - self.away_score

    @property
    def fixture(self) -> Fixture:
        kickoff = self.date
        if kickoff.tzinfo is None:
            meta = get_venue_meta(self.venue)
            if meta is None:
                raise ValueError(
                    f"Unknown venue timezone for {self.venue!r}; provide an offset kickoff"
                )
            kickoff = kickoff.replace(tzinfo=ZoneInfo(meta.timezone))
        return Fixture(
            self.match_id,
            self.year,
            self.round_label,
            kickoff,
            self.venue,
            self.home_team,
            self.away_team,
        )

    @property
    def available_at(self) -> datetime:
        kickoff = self.fixture.kickoff
        if self.result_available_at is not None:
            _aware(self.result_available_at, "result_available_at")
            if self.result_available_at <= kickoff:
                raise ValueError("Result availability must be after kickoff")
            return self.result_available_at
        meta = get_venue_meta(self.venue)
        zone = ZoneInfo(meta.timezone) if meta else kickoff.tzinfo
        local_day = kickoff.astimezone(zone).date()
        return datetime.combine(local_day + timedelta(days=1), time(), zone)


@dataclass(frozen=True)
class MarketQuote:
    match_id: str
    observed_at: datetime | None
    predicted_margin: float | None
    home_odds: float | None = None
    away_odds: float | None = None
    validation_note: str = ""

    def __post_init__(self):
        if not self.match_id.strip():
            raise ValueError("Market quote needs a match ID")
        if self.observed_at is not None:
            _aware(self.observed_at, "observed_at")
        if self.predicted_margin is not None:
            _number(self.predicted_margin, "predicted_margin")
        if (self.home_odds is None) != (self.away_odds is None):
            raise ValueError("Supply both home_odds and away_odds, or neither")
        for odds in (self.home_odds, self.away_odds):
            if odds is not None and _number(odds, "odds") <= 1:
                raise ValueError("Decimal odds must exceed 1")

    @property
    def home_probability(self) -> float | None:
        if self.home_odds is None:
            return None
        return (1 / self.home_odds) / (1 / self.home_odds + 1 / self.away_odds)


def _csv_rows(path: str, required=()):
    with open(path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise ValueError(f"{path}: missing or duplicate CSV headers")
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            yield line, row


def _unique(items, key, description):
    seen = set()
    for item in items:
        value = key(item)
        if value in seen:
            raise ValueError(f"Duplicate {description}: {value}")
        seen.add(value)


def validate_matches(matches: list[MatchRow]) -> None:
    _unique(matches, lambda row: row.match_id, "match ID")
    _unique(
        matches,
        lambda row: (row.fixture.kickoff, row.home_team, row.away_team),
        "match",
    )
    for match in matches:
        _ = match.available_at
        for name in (
            "home_score",
            "away_score",
            "home_goals",
            "away_goals",
            "home_behinds",
            "away_behinds",
            "home_scoring_shots",
            "away_scoring_shots",
        ):
            _number(getattr(match, name), name, minimum=0)


def _fixture_from_row(row: dict) -> Fixture:
    if row.get("kickoff"):
        kickoff = parse_timestamp(row["kickoff"])
    else:
        meta = get_venue_meta(row["venue"])
        if meta is None:
            raise ValueError(
                f"Unknown venue timezone for {row['venue']!r}; provide kickoff with an offset"
            )
        day = parse_match_date(row["date"])
        raw_time = row.get("time", "").strip()
        if not raw_time:
            raise ValueError("Match time is missing; provide time or kickoff")
        clock = datetime.strptime(raw_time, "%I:%M %p").time()
        kickoff = datetime.combine(day.date(), clock, ZoneInfo(meta.timezone))
    return Fixture(
        row["match_id"].strip(),
        int(row["year"]),
        row["round"],
        kickoff,
        row["venue"],
        canonical_team_name(row["home_team_name"]),
        canonical_team_name(row["away_team_name"]),
    )


def load_matches_csv(path: str) -> list[MatchRow]:
    matches = []
    for line, row in _csv_rows(
        path,
        (
            "match_id",
            "year",
            "round",
            "venue",
            "home_team_name",
            "away_team_name",
            "home_team_score",
            "away_team_score",
        ),
    ):
        try:
            fixture = _fixture_from_row(row)
            counts = {}
            for name in (
                "home_goals",
                "home_behinds",
                "away_goals",
                "away_behinds",
                "home_scoring_shots",
                "away_scoring_shots",
            ):
                number = _number(row.get(name) or 0, name, minimum=0)
                if not number.is_integer():
                    raise ValueError(f"{name} must be an integer")
                counts[name] = int(number)
            stamp = row.get("result_available_at", "").strip()
            matches.append(
                MatchRow(
                    fixture.match_id,
                    fixture.year,
                    fixture.round_label,
                    fixture.kickoff,
                    fixture.venue,
                    fixture.home_team,
                    fixture.away_team,
                    _number(row["home_team_score"], "home_team_score", minimum=0),
                    _number(row["away_team_score"], "away_team_score", minimum=0),
                    **counts,
                    result_available_at=parse_timestamp(stamp) if stamp else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    validate_matches(matches)
    return sorted(matches, key=lambda match: (match.fixture.kickoff, match.match_id))


def load_fixtures_csv(path: str) -> list[Fixture]:
    fixtures = []
    for line, row in _csv_rows(
        path,
        (
            "match_id",
            "year",
            "round",
            "kickoff",
            "venue",
            "home_team_name",
            "away_team_name",
        ),
    ):
        try:
            if any(
                row.get(key, "").strip()
                for key in (
                    "home_team_score",
                    "away_team_score",
                    "actual_margin",
                    "home_goals",
                    "away_goals",
                    "home_behinds",
                    "away_behinds",
                    "home_scoring_shots",
                    "away_scoring_shots",
                    "result_available_at",
                )
            ):
                raise ValueError("Fixture rows must not contain match outcomes")
            if not row.get("kickoff"):
                raise ValueError("Live fixtures require kickoff with a timezone offset")
            fixtures.append(_fixture_from_row(row))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    _unique(fixtures, lambda fixture: fixture.match_id, "fixture ID")
    _unique(
        fixtures,
        lambda fixture: (fixture.kickoff, fixture.home_team, fixture.away_team),
        "fixture",
    )
    return fixtures


def load_market_csv(path: str) -> list[MarketQuote]:
    quotes = []
    for line, row in _csv_rows(path, ("match_id", "observed_at", "predicted_margin")):
        try:
            quotes.append(
                MarketQuote(
                    row["match_id"].strip(),
                    parse_timestamp(row["observed_at"]),
                    _number(row["predicted_margin"], "predicted_margin"),
                    _number(row["home_odds"], "home_odds")
                    if row.get("home_odds")
                    else None,
                    _number(row["away_odds"], "away_odds")
                    if row.get("away_odds")
                    else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    _unique(
        quotes, lambda quote: (quote.match_id, quote.observed_at), "market snapshot"
    )
    return quotes


def load_market_xlsx(
    path: str, matches: list[MatchRow], *, closing_line_benchmark: bool = False
) -> list[MarketQuote]:
    if not closing_line_benchmark:
        raise ValueError("Untimed market history requires --closing-line-benchmark")
    import pandas as pd

    frame = pd.read_excel(Path(path), sheet_name="Data", header=1)
    required = {
        "Date",
        "Home Team",
        "Away Team",
        "Home Line Close",
        "Home Odds",
        "Away Odds",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Market workbook is missing columns: {sorted(required - set(frame.columns))}"
        )

    def local_date(match):
        meta = get_venue_meta(match.venue)
        kickoff = match.fixture.kickoff
        return (
            kickoff.astimezone(ZoneInfo(meta.timezone)).date()
            if meta
            else kickoff.date()
        )

    match_ids = {
        (local_date(match), match.home_team, match.away_team): match.match_id
        for match in matches
    }
    seen = set()
    quotes = []
    for line, row in enumerate(frame.to_dict("records"), 3):
        try:
            if all(pd.isna(row[name]) for name in required):
                continue
            if any(pd.isna(row[name]) for name in ("Date", "Home Team", "Away Team")):
                raise ValueError("Market row needs date and teams")
            key = (
                pd.to_datetime(row["Date"]).date(),
                canonical_team_name(str(row["Home Team"])),
                canonical_team_name(str(row["Away Team"])),
            )
            if key in seen:
                raise ValueError(f"Duplicate market match: {key}")
            seen.add(key)
            margin = (
                None
                if pd.isna(row["Home Line Close"])
                else -_number(row["Home Line Close"], "Home Line Close")
            )
            odds = [
                None if pd.isna(row[name]) else _number(row[name], name)
                for name in ("Home Odds", "Away Odds")
            ]
            note = ""
            if margin is None and any(
                value is not None and value <= 1 for value in odds
            ):
                odds = [None, None]
                note = "invalid_odds_without_margin"
            quote = MarketQuote(
                match_ids.get(key, repr(key)), None, margin, *odds, validation_note=note
            )
            if key in match_ids:
                quotes.append(quote)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    return quotes
