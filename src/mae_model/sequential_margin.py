import csv
import math
import warnings
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`",
    category=UserWarning,
)


TEAM_ALIASES = {
    "South Melbourne": "Sydney",
    "Footscray": "Western Bulldogs",
    "North Melbourne": "Kangaroos",
    "North Melbourne Kangaroos": "Kangaroos",
    "Brisbane": "Brisbane Lions",
    "GWS Giants": "Greater Western Sydney",
}

FINALS_ORDER = {
    "EF": 101,
    "QF": 102,
    "SF": 103,
    "PF": 104,
    "GF": 105,
    "Final": 106,
    "Grand Final": 107,
}

TEAM_HOME_STATE = {
    "Adelaide": "SA",
    "Brisbane Lions": "QLD",
    "Carlton": "VIC",
    "Collingwood": "VIC",
    "Essendon": "VIC",
    "Fremantle": "WA",
    "Geelong": "VIC",
    "Gold Coast": "QLD",
    "Greater Western Sydney": "NSW",
    "Hawthorn": "VIC",
    "Kangaroos": "VIC",
    "Melbourne": "VIC",
    "Port Adelaide": "SA",
    "Richmond": "VIC",
    "St Kilda": "VIC",
    "Sydney": "NSW",
    "West Coast": "WA",
    "Western Bulldogs": "VIC",
}

TEAM_HOME_COORDS = {
    "Adelaide": (-34.9285, 138.6007),
    "Brisbane Lions": (-27.4698, 153.0251),
    "Carlton": (-37.8136, 144.9631),
    "Collingwood": (-37.8136, 144.9631),
    "Essendon": (-37.8136, 144.9631),
    "Fremantle": (-31.9505, 115.8605),
    "Geelong": (-38.1499, 144.3617),
    "Gold Coast": (-28.0167, 153.4000),
    "Greater Western Sydney": (-33.8688, 151.2093),
    "Hawthorn": (-37.8136, 144.9631),
    "Kangaroos": (-37.8136, 144.9631),
    "Melbourne": (-37.8136, 144.9631),
    "Port Adelaide": (-34.9285, 138.6007),
    "Richmond": (-37.8136, 144.9631),
    "St Kilda": (-37.8136, 144.9631),
    "Sydney": (-33.8688, 151.2093),
    "West Coast": (-31.9505, 115.8605),
    "Western Bulldogs": (-37.8136, 144.9631),
}

VENUE_STATE_HINTS = (
    ("mcg", "VIC"),
    ("marvel", "VIC"),
    ("docklands", "VIC"),
    ("kardinia", "VIC"),
    ("gmhba", "VIC"),
    ("aami stadium", "SA"),
    ("adelaide oval", "SA"),
    ("perth stadium", "WA"),
    ("optus", "WA"),
    ("domain stadium", "WA"),
    ("subiaco", "WA"),
    ("the gabba", "QLD"),
    ("gabba", "QLD"),
    ("carrara", "QLD"),
    ("metricon", "QLD"),
    ("heritage bank", "QLD"),
    ("sydney cricket ground", "NSW"),
    ("scg", "NSW"),
    ("stadium australia", "NSW"),
    ("anz stadium", "NSW"),
    ("accor", "NSW"),
    ("showground", "NSW"),
    ("engie", "NSW"),
    ("manuka", "ACT"),
    ("bellerive", "TAS"),
    ("utas", "TAS"),
    ("blundstone", "TAS"),
    ("mars stadium", "VIC"),
    ("eureka", "VIC"),
    ("cazaly", "QLD"),
    ("tio", "NT"),
    ("traeger", "NT"),
    ("darwin", "NT"),
    ("york park", "TAS"),
    ("norwood", "SA"),
)

VENUE_COORD_HINTS = (
    ("mcg", (-37.8199, 144.9834)),
    ("marvel", (-37.8164, 144.9475)),
    ("docklands", (-37.8164, 144.9475)),
    ("kardinia", (-38.1580, 144.3547)),
    ("gmhba", (-38.1580, 144.3547)),
    ("adelaide oval", (-34.9154, 138.5964)),
    ("aami stadium", (-34.9154, 138.5964)),
    ("optus", (-31.9509, 115.8882)),
    ("perth stadium", (-31.9509, 115.8882)),
    ("domain stadium", (-31.9434, 115.8721)),
    ("subiaco", (-31.9434, 115.8721)),
    ("the gabba", (-27.4858, 153.0381)),
    ("gabba", (-27.4858, 153.0381)),
    ("metricon", (-28.0573, 153.3805)),
    ("heritage bank", (-28.0573, 153.3805)),
    ("sydney cricket ground", (-33.8917, 151.2240)),
    ("scg", (-33.8917, 151.2240)),
    ("anz stadium", (-33.8474, 151.0639)),
    ("accor", (-33.8474, 151.0639)),
    ("showground", (-33.8441, 151.0676)),
    ("engie", (-33.8441, 151.0676)),
    ("manuka", (-35.3198, 149.1398)),
    ("blundstone", (-42.8815, 147.3347)),
    ("bellerive", (-42.8815, 147.3347)),
    ("utas", (-41.4342, 147.1377)),
    ("york park", (-41.4342, 147.1377)),
    ("mars stadium", (-37.5460, 143.8469)),
    ("eureka", (-37.5460, 143.8469)),
    ("cazaly", (-16.9204, 145.7708)),
    ("tio", (-12.4012, 130.8773)),
    ("traeger", (-23.6997, 133.8807)),
    ("norwood", (-34.9209, 138.6298)),
)


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _infer_venue_state(venue: str) -> Optional[str]:
    norm = _normalize_text(venue)
    for needle, state in VENUE_STATE_HINTS:
        if needle in norm:
            return state
    return None


def _infer_venue_coords(venue: str) -> Optional[Tuple[float, float]]:
    norm = _normalize_text(venue)
    for needle, coords in VENUE_COORD_HINTS:
        if needle in norm:
            return coords
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * r_km * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _round_progress(round_label: str) -> float:
    if is_finals_round(round_label):
        return 1.0
    return min(1.0, max(0.0, round_sort_value(round_label) / 24.0))


def _disagreement_bucket(disagreement: float) -> str:
    abs_dis = abs(disagreement)
    if abs_dis < 7.0:
        return "small"
    if abs_dis < 14.0:
        return "medium"
    return "large"


class TeamFeatureTracker:
    def __init__(self):
        self.margin_ewma = {
            0.1: defaultdict(float),
            0.3: defaultdict(float),
            0.5: defaultdict(float),
        }
        self.margin_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.last_match_date: Dict[str, date] = {}

    def _rolling_mean(self, team_name: str, window: int) -> float:
        history = self.margin_history[team_name]
        if not history:
            return 0.0
        values = list(history)[-window:]
        return float(sum(values) / len(values))

    def _rest_days(self, team_name: str, match_date: date) -> float:
        previous = self.last_match_date.get(team_name)
        if previous is None:
            return 7.0
        return float((match_date - previous).days)

    def _travel_km(self, team_name: str, venue: str) -> float:
        team_coords = TEAM_HOME_COORDS.get(team_name)
        venue_coords = _infer_venue_coords(venue)
        if team_coords is None or venue_coords is None:
            return 0.0
        return _haversine_km(team_coords[0], team_coords[1], venue_coords[0], venue_coords[1])

    def _interstate_flag(self, team_name: str, venue: str) -> float:
        team_state = TEAM_HOME_STATE.get(team_name)
        venue_state = _infer_venue_state(venue)
        if team_state is None or venue_state is None:
            return 0.0
        return 1.0 if team_state != venue_state else 0.0

    def pre_match_features(self, match: "MatchRow") -> Dict[str, float]:
        match_date = match.date.date()
        home_rest = self._rest_days(match.home_team, match_date)
        away_rest = self._rest_days(match.away_team, match_date)
        home_travel = self._travel_km(match.home_team, match.venue)
        away_travel = self._travel_km(match.away_team, match.venue)
        home_interstate = self._interstate_flag(match.home_team, match.venue)
        away_interstate = self._interstate_flag(match.away_team, match.venue)
        return {
            "form_ewma_01_diff": self.margin_ewma[0.1][match.home_team] - self.margin_ewma[0.1][match.away_team],
            "form_ewma_03_diff": self.margin_ewma[0.3][match.home_team] - self.margin_ewma[0.3][match.away_team],
            "form_ewma_05_diff": self.margin_ewma[0.5][match.home_team] - self.margin_ewma[0.5][match.away_team],
            "form_roll3_diff": self._rolling_mean(match.home_team, 3) - self._rolling_mean(match.away_team, 3),
            "form_roll5_diff": self._rolling_mean(match.home_team, 5) - self._rolling_mean(match.away_team, 5),
            "form_roll10_diff": self._rolling_mean(match.home_team, 10) - self._rolling_mean(match.away_team, 10),
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_days_diff": home_rest - away_rest,
            "home_short_rest": 1.0 if home_rest <= 5.0 else 0.0,
            "away_short_rest": 1.0 if away_rest <= 5.0 else 0.0,
            "home_long_break": 1.0 if home_rest >= 9.0 else 0.0,
            "away_long_break": 1.0 if away_rest >= 9.0 else 0.0,
            "travel_km_diff": home_travel - away_travel,
            "away_travel_km": away_travel,
            "home_interstate": home_interstate,
            "away_interstate": away_interstate,
            "interstate_diff": home_interstate - away_interstate,
            "season_progress": _round_progress(match.round_label),
        }

    def update(self, match: "MatchRow"):
        home_margin = match.actual_margin
        away_margin = -home_margin
        for alpha, store in self.margin_ewma.items():
            store[match.home_team] = (1.0 - alpha) * store[match.home_team] + alpha * home_margin
            store[match.away_team] = (1.0 - alpha) * store[match.away_team] + alpha * away_margin
        self.margin_history[match.home_team].append(home_margin)
        self.margin_history[match.away_team].append(away_margin)
        match_date = match.date.date()
        self.last_match_date[match.home_team] = match_date
        self.last_match_date[match.away_team] = match_date


class KnownLineupTracker:
    def __init__(self):
        self.team_player_appearances: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.team_last_lineup: Dict[str, set] = defaultdict(set)
        self.team_matches_seen: Dict[str, int] = defaultdict(int)

    @staticmethod
    def _lineup_names(
        match_id: str,
        team_name: str,
        lineups: Dict[Tuple[str, str], List[dict]],
    ) -> List[str]:
        rows = lineups.get((match_id, team_name), [])
        names = []
        for row in rows:
            name = str(row.get("player_name", "")).strip()
            if name:
                names.append(name)
        return names

    def _team_lineup_features(
        self,
        team_name: str,
        player_names: List[str],
    ) -> Tuple[float, float, float, float]:
        if not player_names:
            return 0.0, 0.0, 0.0, 0.0
        appearances = self.team_player_appearances[team_name]
        lineup_set = set(player_names)
        known = sum(1 for name in player_names if appearances.get(name, 0) > 0)
        debut = sum(1 for name in player_names if appearances.get(name, 0) == 0)
        last = self.team_last_lineup.get(team_name, set())
        returning = len(lineup_set & last)
        size = float(len(lineup_set))
        known_ratio = known / size
        debut_ratio = debut / size
        returning_ratio = (returning / size) if last else 0.0
        return known_ratio, returning_ratio, debut_ratio, size

    def pre_match_features(
        self,
        match: "MatchRow",
        lineups: Dict[Tuple[str, str], List[dict]],
    ) -> Dict[str, float]:
        home_players = self._lineup_names(match.match_id, match.home_team, lineups)
        away_players = self._lineup_names(match.match_id, match.away_team, lineups)
        home_known, home_returning, home_debut, home_size = self._team_lineup_features(match.home_team, home_players)
        away_known, away_returning, away_debut, away_size = self._team_lineup_features(match.away_team, away_players)
        return {
            "lineup_known_diff": home_known - away_known,
            "lineup_returning_diff": home_returning - away_returning,
            "lineup_debut_diff": home_debut - away_debut,
            "lineup_size_diff": home_size - away_size,
            "lineup_volatility": home_debut + away_debut,
        }

    def update(
        self,
        match: "MatchRow",
        lineups: Dict[Tuple[str, str], List[dict]],
    ):
        for team_name in (match.home_team, match.away_team):
            names = self._lineup_names(match.match_id, team_name, lineups)
            if not names:
                continue
            lineup_set = set(names)
            for name in lineup_set:
                self.team_player_appearances[team_name][name] += 1
            self.team_last_lineup[team_name] = lineup_set
            self.team_matches_seen[team_name] += 1


class LineupStrengthTracker:
    def __init__(self):
        self.player_rating: Dict[str, float] = defaultdict(float)
        self.player_games: Dict[str, int] = defaultdict(int)
        self.player_last_team: Dict[str, str] = {}

    @staticmethod
    def _lineup_rows(
        match_id: str,
        team_name: str,
        lineups: Dict[Tuple[str, str], List[dict]],
    ) -> List[dict]:
        return list(lineups.get((match_id, team_name), []))

    def _team_lineup_strength(
        self,
        rows: List[dict],
    ) -> Tuple[float, float, float]:
        if not rows:
            return 0.0, 0.0, 0.0
        player_names = [str(row.get("player_name", "")).strip() for row in rows]
        player_names = [name for name in player_names if name]
        if not player_names:
            return 0.0, 0.0, 0.0

        ratings = [self.player_rating[name] for name in player_names if self.player_games[name] > 0]
        games = [self.player_games[name] for name in player_names]
        if ratings:
            strength = float(sum(ratings) / len(ratings))
            top_end = float(sum(sorted(ratings, reverse=True)[:4]) / min(4, len(ratings)))
        else:
            strength = 0.0
            top_end = 0.0
        experience = float(sum(games) / max(1, len(games)))
        return strength, experience, top_end

    def pre_match_features(
        self,
        match: "MatchRow",
        lineups: Dict[Tuple[str, str], List[dict]],
    ) -> Dict[str, float]:
        home_rows = self._lineup_rows(match.match_id, match.home_team, lineups)
        away_rows = self._lineup_rows(match.match_id, match.away_team, lineups)
        home_strength, home_experience, home_top_end = self._team_lineup_strength(home_rows)
        away_strength, away_experience, away_top_end = self._team_lineup_strength(away_rows)
        return {
            "lineup_strength_diff": home_strength - away_strength,
            "lineup_experience_diff": home_experience - away_experience,
            "lineup_top_end_diff": home_top_end - away_top_end,
        }

    def _player_impact(self, row: dict) -> float:
        disposals = float(row.get("disposals", 0.0) or 0.0)
        goals = float(row.get("goals", 0.0) or 0.0)
        behinds = float(row.get("behinds", 0.0) or 0.0)
        clearances = float(row.get("clearances", 0.0) or 0.0)
        tackles = float(row.get("tackles", 0.0) or 0.0)
        inside_50s = float(row.get("inside_50s", 0.0) or 0.0)
        contested = float(row.get("contested_possessions", 0.0) or 0.0)
        marks = float(row.get("marks", 0.0) or 0.0)
        goal_assists = float(row.get("goal_assists", 0.0) or 0.0)
        percent_played = float(row.get("percent_played", 0.0) or 0.0)
        tog_weight = max(0.35, min(1.35, percent_played / 75.0))
        raw_score = (
            0.045 * disposals
            + 1.35 * goals
            + 0.55 * behinds
            + 0.55 * clearances
            + 0.40 * tackles
            + 0.28 * inside_50s
            + 0.22 * contested
            + 0.18 * marks
            + 0.35 * goal_assists
        )
        return tog_weight * raw_score

    def update(
        self,
        match: "MatchRow",
        lineups: Dict[Tuple[str, str], List[dict]],
    ):
        for team_name in (match.home_team, match.away_team):
            rows = self._lineup_rows(match.match_id, team_name, lineups)
            for row in rows:
                player_name = str(row.get("player_name", "")).strip()
                if not player_name:
                    continue
                impact = self._player_impact(row)
                prior = self.player_rating[player_name]
                if self.player_games[player_name] <= 0:
                    updated = impact
                else:
                    updated = 0.82 * prior + 0.18 * impact
                self.player_rating[player_name] = updated
                self.player_games[player_name] += 1
                self.player_last_team[player_name] = team_name


class DisagreementReliabilityTracker:
    def __init__(self):
        self.counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.correct: Dict[Tuple[str, str, str], int] = defaultdict(int)

    def _favorite_bucket(self, implied_home_prob: Optional[float]) -> str:
        if implied_home_prob is None:
            return "unknown"
        if implied_home_prob > 0.58:
            return "home_fav"
        if implied_home_prob < 0.42:
            return "away_fav"
        return "coin"

    def _sigma_bucket(self, market_sigma: float) -> str:
        if market_sigma < 24.0:
            return "low_sigma"
        if market_sigma < 34.0:
            return "mid_sigma"
        return "high_sigma"

    def _context_key(
        self,
        round_label: str,
        implied_home_prob: Optional[float],
        market_sigma: float,
        disagreement: float,
    ) -> Tuple[str, str, str]:
        return (
            "finals" if is_finals_round(round_label) else "regular",
            _disagreement_bucket(disagreement),
            f"{self._favorite_bucket(implied_home_prob)}_{self._sigma_bucket(market_sigma)}",
        )

    def expected_accuracy(
        self,
        round_label: str,
        implied_home_prob: Optional[float],
        market_sigma: float,
        disagreement: float,
    ) -> float:
        key = self._context_key(round_label, implied_home_prob, market_sigma, disagreement)
        count = self.counts.get(key, 0)
        correct = self.correct.get(key, 0)
        # Smoothed context hit-rate. 0.5 prior avoids overfitting tiny buckets.
        return (correct + 8.0 * 0.5) / (count + 8.0)

    def update(
        self,
        round_label: str,
        implied_home_prob: Optional[float],
        market_sigma: float,
        disagreement: float,
        actual_minus_anchor: float,
    ):
        if abs(disagreement) < 1.0:
            return
        key = self._context_key(round_label, implied_home_prob, market_sigma, disagreement)
        self.counts[key] += 1
        if math.copysign(1.0, disagreement) == math.copysign(1.0, actual_minus_anchor):
            self.correct[key] += 1


def canonical_team_name(team_name: str) -> str:
    return TEAM_ALIASES.get(team_name, team_name)


def parse_match_date(raw: str) -> datetime:
    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {raw}")


def round_sort_value(round_label: str) -> int:
    round_label = round_label.strip()
    digits = "".join(ch for ch in round_label if ch.isdigit())
    if digits:
        return int(digits)
    return FINALS_ORDER.get(round_label, 999)


def is_finals_round(round_label: str) -> bool:
    return not any(ch.isdigit() for ch in round_label)


@dataclass
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

    @property
    def actual_margin(self) -> float:
        return self.home_score - self.away_score


@dataclass
class PredictionRow:
    match_id: str
    year: int
    round_label: str
    home_team: str
    away_team: str
    venue: str
    actual_margin: float
    predicted_margin: float
    abs_error: float
    model_name: str


@dataclass
class MatchContextRow:
    date: date
    home_team: str
    away_team: str
    venue: str = ""
    weather_temp_c: Optional[float] = None
    weather_rain_mm: Optional[float] = None
    weather_wind_kmh: Optional[float] = None
    weather_humidity_pct: Optional[float] = None
    attendance: Optional[float] = None
    projected_attendance: Optional[float] = None
    venue_length_m: Optional[float] = None
    venue_width_m: Optional[float] = None
    venue_capacity: Optional[float] = None


def _coalesce_field(row: dict, names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        if name in row:
            value = row[name]
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str:
                return value_str
    return None


def _parse_optional_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.lower() in {"na", "n/a", "none", "null", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_matches_csv(path: str) -> List[MatchRow]:
    rows: List[MatchRow] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                MatchRow(
                    match_id=row["match_id"],
                    year=int(row["year"]),
                    round_label=row["round"],
                    date=parse_match_date(row["date"]),
                    venue=row.get("venue", ""),
                    home_team=canonical_team_name(row["home_team_name"]),
                    away_team=canonical_team_name(row["away_team_name"]),
                    home_score=float(row["home_team_score"]),
                    away_score=float(row["away_team_score"]),
                    home_goals=int(float(row.get("home_goals", 0) or 0)),
                    home_behinds=int(float(row.get("home_behinds", 0) or 0)),
                    away_goals=int(float(row.get("away_goals", 0) or 0)),
                    away_behinds=int(float(row.get("away_behinds", 0) or 0)),
                    home_scoring_shots=int(float(row.get("home_scoring_shots", 0) or 0)),
                    away_scoring_shots=int(float(row.get("away_scoring_shots", 0) or 0)),
                )
            )

    rows.sort(key=lambda r: (r.date, r.year, round_sort_value(r.round_label), r.match_id))
    return rows


def load_lineups_csv(path: Optional[str]) -> Dict[Tuple[str, str], List[dict]]:
    lineups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    if not path:
        return lineups
    if not Path(path).exists():
        return lineups

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_name = canonical_team_name(row["team_name"])
            key = (row["match_id"], team_name)
            lineups[key].append(
                {
                    "player_name": row["player_name"].strip(),
                    "percent_played": float(row.get("percent_played", 0) or 0),
                    "disposals": float(row.get("disposals", 0) or 0),
                    "goals": float(row.get("goals", 0) or 0),
                    "behinds": float(row.get("behinds", 0) or 0),
                    "clearances": float(row.get("clearances", 0) or 0),
                    "inside_50s": float(row.get("inside_50s", 0) or 0),
                    "tackles": float(row.get("tackles", 0) or 0),
                    "marks": float(row.get("marks", 0) or 0),
                    "contested_possessions": float(row.get("contested_possessions", 0) or 0),
                    "goal_assists": float(row.get("goal_assists", 0) or 0),
                }
            )
    return lineups


def load_market_xlsx(path: Optional[str]) -> Dict[Tuple[date, str, str], dict]:
    market_rows: Dict[Tuple[date, str, str], dict] = {}
    if not path:
        return market_rows
    if not Path(path).exists():
        return market_rows

    import pandas as pd

    frame = pd.read_excel(path, sheet_name="Data", header=1)
    frame = frame.rename(
        columns={
            "Home Team": "home_team",
            "Away Team": "away_team",
            "Home Line Close": "home_line_close",
            "Home Line Open": "home_line_open",
            "Home Line Min": "home_line_min",
            "Home Line Max": "home_line_max",
            "Home Odds": "home_odds",
            "Away Odds": "away_odds",
            "Home Odds Open": "home_odds_open",
            "Away Odds Open": "away_odds_open",
            "Home Odds Close": "home_odds_close",
            "Away Odds Close": "away_odds_close",
            "Total Score Open": "total_score_open",
            "Total Score Min": "total_score_min",
            "Total Score Max": "total_score_max",
            "Total Score Close": "total_score_close",
            "Bookmakers Surveyed": "bookmakers_surveyed",
            "Date": "date",
        }
    )
    required = {"date", "home_team", "away_team", "home_line_close", "home_odds", "away_odds"}
    if not required.issubset(set(frame.columns)):
        return market_rows

    optional = {
        "home_line_open",
        "home_line_min",
        "home_line_max",
        "home_odds_open",
        "away_odds_open",
        "home_odds_close",
        "away_odds_close",
        "total_score_open",
        "total_score_min",
        "total_score_max",
        "total_score_close",
        "bookmakers_surveyed",
    }
    available_columns = list(required | (optional & set(frame.columns)))
    frame = frame[available_columns].dropna(subset=["date", "home_team", "away_team"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["date"])

    def _as_float(value) -> Optional[float]:
        if value != value:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for row in frame.to_dict(orient="records"):
        home_team = canonical_team_name(str(row["home_team"]).strip())
        away_team = canonical_team_name(str(row["away_team"]).strip())
        key = (row["date"], home_team, away_team)
        home_line_close = row.get("home_line_close")
        home_odds = row.get("home_odds")
        away_odds = row.get("away_odds")
        market_rows[key] = {
            "home_line_close": _as_float(home_line_close),
            "home_line_open": _as_float(row.get("home_line_open")),
            "home_line_min": _as_float(row.get("home_line_min")),
            "home_line_max": _as_float(row.get("home_line_max")),
            "home_odds": _as_float(home_odds),
            "away_odds": _as_float(away_odds),
            "home_odds_open": _as_float(row.get("home_odds_open")),
            "away_odds_open": _as_float(row.get("away_odds_open")),
            "home_odds_close": _as_float(row.get("home_odds_close")),
            "away_odds_close": _as_float(row.get("away_odds_close")),
            "total_score_open": _as_float(row.get("total_score_open")),
            "total_score_min": _as_float(row.get("total_score_min")),
            "total_score_max": _as_float(row.get("total_score_max")),
            "total_score_close": _as_float(row.get("total_score_close")),
            "bookmakers_surveyed": _as_float(row.get("bookmakers_surveyed")),
        }

    return market_rows


def load_match_context_csv(path: Optional[str]) -> Dict[Tuple[date, str, str], MatchContextRow]:
    context_rows: Dict[Tuple[date, str, str], MatchContextRow] = {}
    if not path:
        return context_rows
    if not Path(path).exists():
        return context_rows

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_date = _coalesce_field(row, ("date", "match_date", "Date"))
            raw_home = _coalesce_field(row, ("home_team", "home_team_name", "Home Team"))
            raw_away = _coalesce_field(row, ("away_team", "away_team_name", "Away Team"))
            if not raw_date or not raw_home or not raw_away:
                continue

            try:
                match_date = parse_match_date(raw_date).date()
            except ValueError:
                continue

            home_team = canonical_team_name(raw_home)
            away_team = canonical_team_name(raw_away)
            key = (match_date, home_team, away_team)
            context_rows[key] = MatchContextRow(
                date=match_date,
                home_team=home_team,
                away_team=away_team,
                venue=_coalesce_field(row, ("venue", "Venue")) or "",
                weather_temp_c=_parse_optional_float(
                    _coalesce_field(
                        row,
                        ("weather_temp_c", "temp_c", "temperature_c", "TemperatureC"),
                    )
                ),
                weather_rain_mm=_parse_optional_float(
                    _coalesce_field(
                        row,
                        ("weather_rain_mm", "rain_mm", "precip_mm", "precipitation_mm"),
                    )
                ),
                weather_wind_kmh=_parse_optional_float(
                    _coalesce_field(
                        row,
                        ("weather_wind_kmh", "wind_kmh", "wind_speed_kmh", "WindSpeedKmh"),
                    )
                ),
                weather_humidity_pct=_parse_optional_float(
                    _coalesce_field(
                        row,
                        ("weather_humidity_pct", "humidity_pct", "relative_humidity_pct"),
                    )
                ),
                attendance=_parse_optional_float(
                    _coalesce_field(row, ("attendance", "crowd", "actual_attendance"))
                ),
                projected_attendance=_parse_optional_float(
                    _coalesce_field(
                        row,
                        ("projected_attendance", "attendance_projected", "expected_attendance"),
                    )
                ),
                venue_length_m=_parse_optional_float(
                    _coalesce_field(row, ("venue_length_m", "ground_length_m", "length_m"))
                ),
                venue_width_m=_parse_optional_float(
                    _coalesce_field(row, ("venue_width_m", "ground_width_m", "width_m"))
                ),
                venue_capacity=_parse_optional_float(
                    _coalesce_field(row, ("venue_capacity", "capacity", "stadium_capacity"))
                ),
            )

    return context_rows


class SequentialMarginModel:
    def __init__(
        self,
        use_lineups: bool,
        base_score: float = 80.0,
        home_advantage: float = 6.0,
        team_k: float = 0.08,
        defense_k: float = 0.08,
        player_k: float = 0.045,
        home_adv_k: float = 0.005,
        season_carryover: float = 0.67,
        lineup_scale: float = 8.0,
        min_player_games: int = 2,
    ):
        self.use_lineups = use_lineups
        self.base_score = base_score
        self.home_advantage = home_advantage
        self.team_k = team_k
        self.defense_k = defense_k
        self.player_k = player_k
        self.home_adv_k = home_adv_k
        self.season_carryover = season_carryover
        self.lineup_scale = lineup_scale
        self.min_player_games = min_player_games

        self.team_offense: Dict[str, float] = defaultdict(float)
        self.team_defense: Dict[str, float] = defaultdict(float)
        self.player_rating: Dict[str, float] = defaultdict(float)
        self.player_games: Dict[str, int] = defaultdict(int)
        self.player_expected_tog: Dict[str, float] = defaultdict(lambda: 72.0)
        self.team_player_tog: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.team_last_season_player_tog: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.team_continuity_applied_year: Dict[str, int] = {}
        self.current_year: Optional[int] = None

    def _expected_tog_weight(self, team_name: str, player_name: str) -> float:
        team_map = self.team_player_tog.get(team_name)
        if team_map and player_name in team_map:
            expected_tog = team_map[player_name]
        else:
            expected_tog = self.player_expected_tog[player_name]
        return max(10.0, min(95.0, expected_tog))

    def _lineup_effect(self, match_id: str, team_name: str, lineups: Dict[Tuple[str, str], List[dict]]) -> float:
        if not self.use_lineups:
            return 0.0
        players = lineups.get((match_id, team_name), [])
        if not players:
            return 0.0

        weighted_total = 0.0
        total_weight = 0.0
        for player in players:
            player_name = player["player_name"]
            games = self.player_games[player_name]
            if games < self.min_player_games:
                continue
            # Prediction-time weighting uses expected minutes from prior matches only.
            weight = self._expected_tog_weight(team_name, player_name)
            weighted_total += self.player_rating[player_name] * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0
        return self.lineup_scale * (weighted_total / total_weight)

    def _apply_season_transition(self):
        teams = list(self.team_offense.keys())
        if not teams:
            return
        self.team_last_season_player_tog = {
            team: dict(player_map) for team, player_map in self.team_player_tog.items()
        }
        for team in teams:
            self.team_offense[team] *= self.season_carryover
            self.team_defense[team] *= self.season_carryover

        off_mean = sum(self.team_offense.values()) / len(teams)
        def_mean = sum(self.team_defense.values()) / len(teams)
        for team in teams:
            self.team_offense[team] -= off_mean
            self.team_defense[team] -= def_mean

        for player_name in list(self.player_rating.keys()):
            self.player_rating[player_name] *= math.sqrt(self.season_carryover)
        for player_name in list(self.player_expected_tog.keys()):
            self.player_expected_tog[player_name] = 0.85 * self.player_expected_tog[player_name] + 0.15 * 72.0

    def _apply_team_specific_carryover_if_needed(
        self,
        match: MatchRow,
        lineups: Dict[Tuple[str, str], List[dict]],
    ):
        for team_name in (match.home_team, match.away_team):
            if self.team_continuity_applied_year.get(team_name) == match.year:
                continue
            lineup_rows = lineups.get((match.match_id, team_name), [])
            prev = self.team_last_season_player_tog.get(team_name, {})
            if not lineup_rows or not prev:
                self.team_continuity_applied_year[team_name] = match.year
                continue
            returning = sum(prev.get(player["player_name"], 0.0) for player in lineup_rows)
            prior_total = sum(prev.values())
            if prior_total <= 0.0:
                continuity = 0.5
            else:
                continuity = max(0.0, min(1.0, returning / prior_total))

            team_carryover = 0.60 + 0.35 * continuity
            adjustment = team_carryover / max(1e-6, self.season_carryover)
            adjustment = max(0.85, min(1.15, adjustment))
            self.team_offense[team_name] *= adjustment
            self.team_defense[team_name] *= adjustment
            self.team_continuity_applied_year[team_name] = match.year

    def predict(self, match: MatchRow, lineups: Dict[Tuple[str, str], List[dict]]) -> Tuple[float, float, float]:
        home_lineup = self._lineup_effect(match.match_id, match.home_team, lineups)
        away_lineup = self._lineup_effect(match.match_id, match.away_team, lineups)

        expected_home = (
            self.base_score
            + self.team_offense[match.home_team]
            - self.team_defense[match.away_team]
            + self.home_advantage
            + home_lineup
        )
        expected_away = (
            self.base_score
            + self.team_offense[match.away_team]
            - self.team_defense[match.home_team]
            - self.home_advantage
            + away_lineup
        )
        return expected_home, expected_away, expected_home - expected_away

    def _update_player_ratings(self, match: MatchRow, margin_residual: float, lineups: Dict[Tuple[str, str], List[dict]]):
        if not self.use_lineups:
            return

        home_players = lineups.get((match.match_id, match.home_team), [])
        away_players = lineups.get((match.match_id, match.away_team), [])
        if not home_players or not away_players:
            return

        home_weights = [1.0 for _ in home_players]
        away_weights = [1.0 for _ in away_players]
        home_total = sum(home_weights)
        away_total = sum(away_weights)
        if home_total <= 0 or away_total <= 0:
            return

        for idx, player in enumerate(home_players):
            weight = home_weights[idx] / home_total
            name = player["player_name"]
            self.player_rating[name] += self.player_k * margin_residual * weight
            self.player_games[name] += 1
            observed_tog = float(player.get("percent_played", 0.0) or 0.0)
            if observed_tog > 0.0:
                previous_tog = self.player_expected_tog[name]
                updated_tog = 0.7 * previous_tog + 0.3 * observed_tog
                self.player_expected_tog[name] = updated_tog
                self.team_player_tog[match.home_team][name] = updated_tog

        for idx, player in enumerate(away_players):
            weight = away_weights[idx] / away_total
            name = player["player_name"]
            self.player_rating[name] -= self.player_k * margin_residual * weight
            self.player_games[name] += 1
            observed_tog = float(player.get("percent_played", 0.0) or 0.0)
            if observed_tog > 0.0:
                previous_tog = self.player_expected_tog[name]
                updated_tog = 0.7 * previous_tog + 0.3 * observed_tog
                self.player_expected_tog[name] = updated_tog
                self.team_player_tog[match.away_team][name] = updated_tog

    def update(self, match: MatchRow, expected_home: float, expected_away: float, lineups: Dict[Tuple[str, str], List[dict]]):
        home_residual = match.home_score - expected_home
        away_residual = match.away_score - expected_away
        margin_residual = match.actual_margin - (expected_home - expected_away)

        self.team_offense[match.home_team] += self.team_k * home_residual
        self.team_defense[match.home_team] -= self.defense_k * away_residual
        self.team_offense[match.away_team] += self.team_k * away_residual
        self.team_defense[match.away_team] -= self.defense_k * home_residual
        self.home_advantage += self.home_adv_k * margin_residual

        self._update_player_ratings(match, margin_residual, lineups)

    def step(self, match: MatchRow, lineups: Dict[Tuple[str, str], List[dict]]) -> Tuple[float, float, float]:
        if self.current_year is None:
            self.current_year = match.year
        elif match.year != self.current_year:
            self.current_year = match.year
            self._apply_season_transition()
        self._apply_team_specific_carryover_if_needed(match, lineups)

        expected_home, expected_away, expected_margin = self.predict(match, lineups)
        self.update(match, expected_home, expected_away, lineups)
        return expected_home, expected_away, expected_margin


class ScoringShotMarginModel:
    """MoSSBODS-inspired score model using scoring shots as latent volume."""

    def __init__(
        self,
        base_scoring_shots: float = 24.0,
        home_adv_scoring_shots: float = 1.0,
        shot_k: float = 0.09,
        season_carryover: float = 0.72,
        conversion_window_games: int = 500,
        shot_residual_cap: float = 12.0,
    ):
        self.base_scoring_shots = base_scoring_shots
        self.home_adv_scoring_shots = home_adv_scoring_shots
        self.shot_k = shot_k
        self.season_carryover = season_carryover
        self.shot_residual_cap = shot_residual_cap
        self.current_year: Optional[int] = None

        self.team_attack_shots: Dict[str, float] = defaultdict(float)
        self.team_defense_shots: Dict[str, float] = defaultdict(float)
        self.points_per_shot_history: deque = deque(maxlen=conversion_window_games)

    def _apply_season_transition(self):
        teams = list(self.team_attack_shots.keys())
        if not teams:
            return
        for team in teams:
            self.team_attack_shots[team] *= self.season_carryover
            self.team_defense_shots[team] *= self.season_carryover

        attack_mean = sum(self.team_attack_shots.values()) / len(teams)
        defense_mean = sum(self.team_defense_shots.values()) / len(teams)
        for team in teams:
            self.team_attack_shots[team] -= attack_mean
            self.team_defense_shots[team] -= defense_mean

    def _get_points_per_shot(self) -> float:
        if not self.points_per_shot_history:
            return 4.85
        return sum(self.points_per_shot_history) / len(self.points_per_shot_history)

    @staticmethod
    def _actual_scoring_shots(match: MatchRow) -> Tuple[float, float]:
        home_ss = float(match.home_scoring_shots)
        away_ss = float(match.away_scoring_shots)
        if home_ss <= 0:
            home_ss = match.home_score / 5.0
        if away_ss <= 0:
            away_ss = match.away_score / 5.0
        return home_ss, away_ss

    def predict(self, match: MatchRow) -> Tuple[float, float, float]:
        expected_home_ss = (
            self.base_scoring_shots
            + self.team_attack_shots[match.home_team]
            - self.team_defense_shots[match.away_team]
            + self.home_adv_scoring_shots
        )
        expected_away_ss = (
            self.base_scoring_shots
            + self.team_attack_shots[match.away_team]
            - self.team_defense_shots[match.home_team]
            - self.home_adv_scoring_shots
        )
        expected_home_ss = max(8.0, expected_home_ss)
        expected_away_ss = max(8.0, expected_away_ss)
        points_per_shot = self._get_points_per_shot()

        expected_home_score = expected_home_ss * points_per_shot
        expected_away_score = expected_away_ss * points_per_shot
        expected_margin = expected_home_score - expected_away_score
        return expected_home_score, expected_away_score, expected_margin

    def update(self, match: MatchRow, expected_home_score: float, expected_away_score: float):
        actual_home_ss, actual_away_ss = self._actual_scoring_shots(match)
        points_per_shot = self._get_points_per_shot()
        expected_home_ss = max(expected_home_score / points_per_shot, 8.0)
        expected_away_ss = max(expected_away_score / points_per_shot, 8.0)

        home_shot_residual = max(
            -self.shot_residual_cap,
            min(self.shot_residual_cap, actual_home_ss - expected_home_ss),
        )
        away_shot_residual = max(
            -self.shot_residual_cap,
            min(self.shot_residual_cap, actual_away_ss - expected_away_ss),
        )

        self.team_attack_shots[match.home_team] += self.shot_k * home_shot_residual
        self.team_defense_shots[match.home_team] -= self.shot_k * away_shot_residual
        self.team_attack_shots[match.away_team] += self.shot_k * away_shot_residual
        self.team_defense_shots[match.away_team] -= self.shot_k * home_shot_residual

        if actual_home_ss > 0:
            self.points_per_shot_history.append(match.home_score / actual_home_ss)
        if actual_away_ss > 0:
            self.points_per_shot_history.append(match.away_score / actual_away_ss)

    def step(self, match: MatchRow) -> Tuple[float, float, float]:
        if self.current_year is None:
            self.current_year = match.year
        elif match.year != self.current_year:
            self.current_year = match.year
            self._apply_season_transition()

        expected_home_score, expected_away_score, expected_margin = self.predict(match)
        self.update(match, expected_home_score, expected_away_score)
        return expected_home_score, expected_away_score, expected_margin


class TerritoryShotChainModel:
    """
    Avenue 1: pre-game chain model.
    Predicts:
    1) territory (inside-50 volume),
    2) scoring-shot conversion from territory,
    3) points-per-shot conversion.
    """

    def __init__(
        self,
        base_inside50: float = 54.0,
        home_adv_inside50: float = 2.0,
        inside50_k: float = 0.05,
        base_shot_rate: float = 0.46,
        shot_rate_k: float = 0.035,
        base_points_per_shot: float = 4.85,
        conversion_k: float = 0.03,
        home_adv_pps: float = 0.08,
        home_adv_pps_k: float = 0.001,
        season_carryover: float = 0.8,
    ):
        self.base_inside50 = base_inside50
        self.home_adv_inside50 = home_adv_inside50
        self.inside50_k = inside50_k
        self.base_shot_rate = base_shot_rate
        self.shot_rate_k = shot_rate_k
        self.base_points_per_shot = base_points_per_shot
        self.conversion_k = conversion_k
        self.home_adv_pps = home_adv_pps
        self.home_adv_pps_k = home_adv_pps_k
        self.season_carryover = season_carryover
        self.current_year: Optional[int] = None

        self.team_attack_i50: Dict[str, float] = defaultdict(float)
        self.team_defense_i50: Dict[str, float] = defaultdict(float)
        self.team_attack_shot_rate: Dict[str, float] = defaultdict(float)
        self.team_defense_shot_rate: Dict[str, float] = defaultdict(float)
        self.team_attack_pps: Dict[str, float] = defaultdict(float)
        self.team_defense_pps: Dict[str, float] = defaultdict(float)

    def _apply_season_transition(self):
        teams = list(
            set(
                list(self.team_attack_i50.keys())
                + list(self.team_defense_i50.keys())
                + list(self.team_attack_shot_rate.keys())
                + list(self.team_defense_shot_rate.keys())
                + list(self.team_attack_pps.keys())
                + list(self.team_defense_pps.keys())
            )
        )
        if not teams:
            return

        for team in teams:
            self.team_attack_i50[team] *= self.season_carryover
            self.team_defense_i50[team] *= self.season_carryover
            self.team_attack_shot_rate[team] *= self.season_carryover
            self.team_defense_shot_rate[team] *= self.season_carryover
            self.team_attack_pps[team] *= self.season_carryover
            self.team_defense_pps[team] *= self.season_carryover

        for store in (
            self.team_attack_i50,
            self.team_defense_i50,
            self.team_attack_shot_rate,
            self.team_defense_shot_rate,
            self.team_attack_pps,
            self.team_defense_pps,
        ):
            mean_value = sum(store[t] for t in teams) / len(teams)
            for team in teams:
                store[team] -= mean_value

    @staticmethod
    def _actual_scoring_shots(match: MatchRow) -> Tuple[float, float]:
        home_ss = float(match.home_scoring_shots)
        away_ss = float(match.away_scoring_shots)
        if home_ss <= 0:
            home_ss = max(1.0, match.home_score / 5.0)
        if away_ss <= 0:
            away_ss = max(1.0, match.away_score / 5.0)
        return home_ss, away_ss

    def _actual_inside50(
        self,
        match: MatchRow,
        lineups: Dict[Tuple[str, str], List[dict]],
        fallback_home_ss: float,
        fallback_away_ss: float,
    ) -> Tuple[float, float]:
        home_rows = lineups.get((match.match_id, match.home_team), [])
        away_rows = lineups.get((match.match_id, match.away_team), [])
        home_i50 = sum(float(player.get("inside_50s", 0.0) or 0.0) for player in home_rows)
        away_i50 = sum(float(player.get("inside_50s", 0.0) or 0.0) for player in away_rows)
        if home_i50 <= 0:
            home_i50 = fallback_home_ss / max(0.30, self.base_shot_rate)
        if away_i50 <= 0:
            away_i50 = fallback_away_ss / max(0.30, self.base_shot_rate)
        return home_i50, away_i50

    def predict(self, match: MatchRow) -> Tuple[float, float, float, float, float, float, float, float, float]:
        expected_home_i50 = (
            self.base_inside50
            + self.team_attack_i50[match.home_team]
            - self.team_defense_i50[match.away_team]
            + self.home_adv_inside50
        )
        expected_away_i50 = (
            self.base_inside50
            + self.team_attack_i50[match.away_team]
            - self.team_defense_i50[match.home_team]
            - self.home_adv_inside50
        )
        expected_home_i50 = max(34.0, expected_home_i50)
        expected_away_i50 = max(34.0, expected_away_i50)

        expected_home_shot_rate = (
            self.base_shot_rate
            + self.team_attack_shot_rate[match.home_team]
            - self.team_defense_shot_rate[match.away_team]
        )
        expected_away_shot_rate = (
            self.base_shot_rate
            + self.team_attack_shot_rate[match.away_team]
            - self.team_defense_shot_rate[match.home_team]
        )
        expected_home_shot_rate = max(0.28, min(0.72, expected_home_shot_rate))
        expected_away_shot_rate = max(0.28, min(0.72, expected_away_shot_rate))

        expected_home_ss = expected_home_i50 * expected_home_shot_rate
        expected_away_ss = expected_away_i50 * expected_away_shot_rate

        expected_home_pps = (
            self.base_points_per_shot
            + self.team_attack_pps[match.home_team]
            - self.team_defense_pps[match.away_team]
            + self.home_adv_pps
        )
        expected_away_pps = (
            self.base_points_per_shot
            + self.team_attack_pps[match.away_team]
            - self.team_defense_pps[match.home_team]
            - self.home_adv_pps
        )
        expected_home_pps = max(3.5, min(6.3, expected_home_pps))
        expected_away_pps = max(3.5, min(6.3, expected_away_pps))

        expected_home_score = expected_home_ss * expected_home_pps
        expected_away_score = expected_away_ss * expected_away_pps
        expected_margin = expected_home_score - expected_away_score
        return (
            expected_home_score,
            expected_away_score,
            expected_margin,
            expected_home_i50,
            expected_away_i50,
            expected_home_shot_rate,
            expected_away_shot_rate,
            expected_home_pps,
            expected_away_pps,
        )

    def update(
        self,
        match: MatchRow,
        lineups: Dict[Tuple[str, str], List[dict]],
        expected_home_i50: float,
        expected_away_i50: float,
        expected_home_shot_rate: float,
        expected_away_shot_rate: float,
        expected_home_pps: float,
        expected_away_pps: float,
        expected_margin: float,
    ):
        actual_home_ss, actual_away_ss = self._actual_scoring_shots(match)
        actual_home_i50, actual_away_i50 = self._actual_inside50(
            match,
            lineups,
            fallback_home_ss=actual_home_ss,
            fallback_away_ss=actual_away_ss,
        )

        home_i50_residual = max(-20.0, min(20.0, actual_home_i50 - expected_home_i50))
        away_i50_residual = max(-20.0, min(20.0, actual_away_i50 - expected_away_i50))
        self.team_attack_i50[match.home_team] += self.inside50_k * home_i50_residual
        self.team_defense_i50[match.home_team] -= self.inside50_k * away_i50_residual
        self.team_attack_i50[match.away_team] += self.inside50_k * away_i50_residual
        self.team_defense_i50[match.away_team] -= self.inside50_k * home_i50_residual

        actual_home_shot_rate = actual_home_ss / max(1.0, actual_home_i50)
        actual_away_shot_rate = actual_away_ss / max(1.0, actual_away_i50)
        home_sr_residual = max(-0.2, min(0.2, actual_home_shot_rate - expected_home_shot_rate))
        away_sr_residual = max(-0.2, min(0.2, actual_away_shot_rate - expected_away_shot_rate))
        self.team_attack_shot_rate[match.home_team] += self.shot_rate_k * home_sr_residual
        self.team_defense_shot_rate[match.away_team] -= self.shot_rate_k * home_sr_residual
        self.team_attack_shot_rate[match.away_team] += self.shot_rate_k * away_sr_residual
        self.team_defense_shot_rate[match.home_team] -= self.shot_rate_k * away_sr_residual

        actual_home_pps = match.home_score / max(1.0, actual_home_ss)
        actual_away_pps = match.away_score / max(1.0, actual_away_ss)
        home_pps_residual = max(-1.5, min(1.5, actual_home_pps - expected_home_pps))
        away_pps_residual = max(-1.5, min(1.5, actual_away_pps - expected_away_pps))
        self.team_attack_pps[match.home_team] += self.conversion_k * home_pps_residual
        self.team_defense_pps[match.away_team] -= self.conversion_k * home_pps_residual
        self.team_attack_pps[match.away_team] += self.conversion_k * away_pps_residual
        self.team_defense_pps[match.home_team] -= self.conversion_k * away_pps_residual

        total_shots = max(12.0, actual_home_ss + actual_away_ss)
        residual_per_shot = (match.actual_margin - expected_margin) / total_shots
        self.home_adv_pps += self.home_adv_pps_k * residual_per_shot
        self.home_adv_pps = max(-0.5, min(0.5, self.home_adv_pps))

    def step(self, match: MatchRow, lineups: Dict[Tuple[str, str], List[dict]]) -> Tuple[float, float, float]:
        if self.current_year is None:
            self.current_year = match.year
        elif match.year != self.current_year:
            self.current_year = match.year
            self._apply_season_transition()

        (
            expected_home_score,
            expected_away_score,
            expected_margin,
            expected_home_i50,
            expected_away_i50,
            expected_home_shot_rate,
            expected_away_shot_rate,
            expected_home_pps,
            expected_away_pps,
        ) = self.predict(match)
        self.update(
            match,
            lineups,
            expected_home_i50,
            expected_away_i50,
            expected_home_shot_rate,
            expected_away_shot_rate,
            expected_home_pps,
            expected_away_pps,
            expected_margin,
        )
        return expected_home_score, expected_away_score, expected_margin


class BenchmarkShotVolumeModel:
    """
    Benchmark-first decomposition model.

    Predicts margin from observed scoring-shot volume and sequential conversion state.
    """

    def __init__(
        self,
        base_scoring_shots: float = 26.0,
        home_adv_scoring_shots: float = 1.0,
        shot_k: float = 0.07,
        shot_residual_cap: float = 12.0,
        conversion_k: float = 0.045,
        home_adv_k: float = 0.002,
        season_carryover: float = 0.78,
        conversion_window_games: int = 800,
    ):
        self.base_scoring_shots = base_scoring_shots
        self.home_adv_scoring_shots = home_adv_scoring_shots
        self.shot_k = shot_k
        self.shot_residual_cap = shot_residual_cap
        self.conversion_k = conversion_k
        self.home_adv_k = home_adv_k
        self.season_carryover = season_carryover
        self.current_year: Optional[int] = None

        self.team_attack_shots: Dict[str, float] = defaultdict(float)
        self.team_defense_shots: Dict[str, float] = defaultdict(float)
        self.team_attack_conversion: Dict[str, float] = defaultdict(float)
        self.team_defense_conversion: Dict[str, float] = defaultdict(float)
        self.home_adv_pps = 0.08
        self.points_per_shot_history: deque = deque(maxlen=conversion_window_games)

    def _apply_season_transition(self):
        teams = list(
            set(
                list(self.team_attack_shots.keys())
                + list(self.team_defense_shots.keys())
                + list(self.team_attack_conversion.keys())
                + list(self.team_defense_conversion.keys())
            )
        )
        if not teams:
            return

        for team in teams:
            self.team_attack_shots[team] *= self.season_carryover
            self.team_defense_shots[team] *= self.season_carryover
            self.team_attack_conversion[team] *= self.season_carryover
            self.team_defense_conversion[team] *= self.season_carryover

        attack_shot_mean = sum(self.team_attack_shots[t] for t in teams) / len(teams)
        defense_shot_mean = sum(self.team_defense_shots[t] for t in teams) / len(teams)
        attack_mean = sum(self.team_attack_conversion[t] for t in teams) / len(teams)
        defense_mean = sum(self.team_defense_conversion[t] for t in teams) / len(teams)
        for team in teams:
            self.team_attack_shots[team] -= attack_shot_mean
            self.team_defense_shots[team] -= defense_shot_mean
            self.team_attack_conversion[team] -= attack_mean
            self.team_defense_conversion[team] -= defense_mean

    def _league_points_per_shot(self) -> float:
        if not self.points_per_shot_history:
            return 4.85
        return float(sum(self.points_per_shot_history) / len(self.points_per_shot_history))

    @staticmethod
    def _actual_scoring_shots(match: MatchRow) -> Tuple[float, float]:
        home_ss = float(match.home_scoring_shots)
        away_ss = float(match.away_scoring_shots)
        if home_ss <= 0:
            home_ss = max(1.0, match.home_score / 5.0)
        if away_ss <= 0:
            away_ss = max(1.0, match.away_score / 5.0)
        return home_ss, away_ss

    def _predict_scoring_shots(self, match: MatchRow) -> Tuple[float, float]:
        expected_home_ss = (
            self.base_scoring_shots
            + self.team_attack_shots[match.home_team]
            - self.team_defense_shots[match.away_team]
            + self.home_adv_scoring_shots
        )
        expected_away_ss = (
            self.base_scoring_shots
            + self.team_attack_shots[match.away_team]
            - self.team_defense_shots[match.home_team]
            - self.home_adv_scoring_shots
        )
        return max(8.0, expected_home_ss), max(8.0, expected_away_ss)

    def predict(self, match: MatchRow) -> Tuple[float, float, float, float, float, float, float]:
        home_ss, away_ss = self._predict_scoring_shots(match)
        league_pps = self._league_points_per_shot()

        expected_home_pps = (
            league_pps
            + self.team_attack_conversion[match.home_team]
            - self.team_defense_conversion[match.away_team]
            + self.home_adv_pps
        )
        expected_away_pps = (
            league_pps
            + self.team_attack_conversion[match.away_team]
            - self.team_defense_conversion[match.home_team]
            - self.home_adv_pps
        )
        expected_home_pps = max(3.5, min(6.4, expected_home_pps))
        expected_away_pps = max(3.5, min(6.4, expected_away_pps))

        expected_home_score = expected_home_pps * home_ss
        expected_away_score = expected_away_pps * away_ss
        expected_margin = expected_home_score - expected_away_score
        return (
            expected_home_score,
            expected_away_score,
            expected_margin,
            home_ss,
            away_ss,
            expected_home_pps,
            expected_away_pps,
        )

    def update(
        self,
        match: MatchRow,
        expected_home_ss: float,
        expected_away_ss: float,
        expected_margin: float,
        expected_home_pps: float,
        expected_away_pps: float,
    ):
        home_ss, away_ss = self._actual_scoring_shots(match)

        home_shot_residual = max(
            -self.shot_residual_cap,
            min(self.shot_residual_cap, home_ss - expected_home_ss),
        )
        away_shot_residual = max(
            -self.shot_residual_cap,
            min(self.shot_residual_cap, away_ss - expected_away_ss),
        )
        self.team_attack_shots[match.home_team] += self.shot_k * home_shot_residual
        self.team_defense_shots[match.home_team] -= self.shot_k * away_shot_residual
        self.team_attack_shots[match.away_team] += self.shot_k * away_shot_residual
        self.team_defense_shots[match.away_team] -= self.shot_k * home_shot_residual

        actual_home_pps = match.home_score / home_ss
        actual_away_pps = match.away_score / away_ss

        home_conversion_residual = actual_home_pps - expected_home_pps
        away_conversion_residual = actual_away_pps - expected_away_pps

        self.team_attack_conversion[match.home_team] += self.conversion_k * home_conversion_residual
        self.team_defense_conversion[match.away_team] -= self.conversion_k * home_conversion_residual
        self.team_attack_conversion[match.away_team] += self.conversion_k * away_conversion_residual
        self.team_defense_conversion[match.home_team] -= self.conversion_k * away_conversion_residual

        total_shots = max(8.0, home_ss + away_ss)
        margin_residual_per_shot = (match.actual_margin - expected_margin) / total_shots
        self.home_adv_pps += self.home_adv_k * margin_residual_per_shot
        self.home_adv_pps = max(-0.6, min(0.6, self.home_adv_pps))

        self.points_per_shot_history.append(actual_home_pps)
        self.points_per_shot_history.append(actual_away_pps)

    def step(self, match: MatchRow) -> Tuple[float, float, float]:
        if self.current_year is None:
            self.current_year = match.year
        elif match.year != self.current_year:
            self.current_year = match.year
            self._apply_season_transition()

        (
            expected_home,
            expected_away,
            expected_margin,
            home_ss,
            away_ss,
            expected_home_pps,
            expected_away_pps,
        ) = self.predict(match)
        self.update(
            match,
            expected_home_ss=home_ss,
            expected_away_ss=away_ss,
            expected_margin=expected_margin,
            expected_home_pps=expected_home_pps,
            expected_away_pps=expected_away_pps,
        )
        return expected_home, expected_away, expected_margin


class ShotVolumeConversionModel:
    """
    Pre-game deployable margin decomposition model:
    - predict scoring-shot volume from team shot ratings
    - predict points-per-shot for each team from league/team/lineup conversion state
    """

    def __init__(
        self,
        base_scoring_shots: float = 26.0,
        home_adv_scoring_shots: float = 1.0,
        shot_k: float = 0.07,
        shot_residual_cap: float = 12.0,
        conversion_k: float = 0.03,
        home_adv_k: float = 0.001,
        season_carryover: float = 0.78,
        lineup_conversion_scale: float = 0.06,
        player_conversion_k: float = 0.15,
        min_player_games: int = 2,
        conversion_window_games: int = 600,
    ):
        self.base_scoring_shots = base_scoring_shots
        self.home_adv_scoring_shots = home_adv_scoring_shots
        self.shot_k = shot_k
        self.shot_residual_cap = shot_residual_cap
        self.conversion_k = conversion_k
        self.home_adv_k = home_adv_k
        self.season_carryover = season_carryover
        self.lineup_conversion_scale = lineup_conversion_scale
        self.player_conversion_k = player_conversion_k
        self.min_player_games = min_player_games
        self.current_year: Optional[int] = None

        self.team_attack_shots: Dict[str, float] = defaultdict(float)
        self.team_defense_shots: Dict[str, float] = defaultdict(float)
        self.team_attack_conversion: Dict[str, float] = defaultdict(float)
        self.team_defense_conversion: Dict[str, float] = defaultdict(float)
        self.home_adv_pps = 0.08
        self.points_per_shot_history: deque = deque(maxlen=conversion_window_games)

        self.player_conversion_rating: Dict[str, float] = defaultdict(float)
        self.player_games: Dict[str, int] = defaultdict(int)

    def _apply_season_transition(self):
        teams = list(
            set(
                list(self.team_attack_shots.keys())
                + list(self.team_defense_shots.keys())
                + list(self.team_attack_conversion.keys())
                + list(self.team_defense_conversion.keys())
            )
        )
        if teams:
            for team in teams:
                self.team_attack_shots[team] *= self.season_carryover
                self.team_defense_shots[team] *= self.season_carryover
                self.team_attack_conversion[team] *= self.season_carryover
                self.team_defense_conversion[team] *= self.season_carryover

            attack_shot_mean = sum(self.team_attack_shots[t] for t in teams) / len(teams)
            defense_shot_mean = sum(self.team_defense_shots[t] for t in teams) / len(teams)
            attack_mean = sum(self.team_attack_conversion[t] for t in teams) / len(teams)
            defense_mean = sum(self.team_defense_conversion[t] for t in teams) / len(teams)
            for team in teams:
                self.team_attack_shots[team] -= attack_shot_mean
                self.team_defense_shots[team] -= defense_shot_mean
                self.team_attack_conversion[team] -= attack_mean
                self.team_defense_conversion[team] -= defense_mean

        for player_name in list(self.player_conversion_rating.keys()):
            self.player_conversion_rating[player_name] *= math.sqrt(self.season_carryover)

    def _league_points_per_shot(self) -> float:
        if not self.points_per_shot_history:
            return 4.85
        return float(sum(self.points_per_shot_history) / len(self.points_per_shot_history))

    def _lineup_conversion_effect(
        self,
        match_id: str,
        team_name: str,
        lineups: Dict[Tuple[str, str], List[dict]],
    ) -> float:
        players = lineups.get((match_id, team_name), [])
        if not players:
            return 0.0

        weighted_total = 0.0
        total_weight = 0.0
        for player in players:
            player_name = player["player_name"]
            if self.player_games[player_name] < self.min_player_games:
                continue
            # Do not use same-match in-game stats (percent played/disposals) at prediction time.
            weight = 1.0
            weighted_total += self.player_conversion_rating[player_name] * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0
        return self.lineup_conversion_scale * (weighted_total / total_weight)

    def _predict_scoring_shots(self, match: MatchRow) -> Tuple[float, float]:
        expected_home_ss = (
            self.base_scoring_shots
            + self.team_attack_shots[match.home_team]
            - self.team_defense_shots[match.away_team]
            + self.home_adv_scoring_shots
        )
        expected_away_ss = (
            self.base_scoring_shots
            + self.team_attack_shots[match.away_team]
            - self.team_defense_shots[match.home_team]
            - self.home_adv_scoring_shots
        )
        return max(8.0, expected_home_ss), max(8.0, expected_away_ss)

    @staticmethod
    def _actual_scoring_shots(match: MatchRow) -> Tuple[float, float]:
        home_ss = float(match.home_scoring_shots)
        away_ss = float(match.away_scoring_shots)
        if home_ss <= 0:
            home_ss = max(1.0, match.home_score / 5.0)
        if away_ss <= 0:
            away_ss = max(1.0, match.away_score / 5.0)
        return home_ss, away_ss

    def predict(
        self,
        match: MatchRow,
        lineups: Dict[Tuple[str, str], List[dict]],
    ) -> Tuple[float, float, float, float, float, float, float]:
        home_ss, away_ss = self._predict_scoring_shots(match)
        league_pps = self._league_points_per_shot()

        home_lineup = self._lineup_conversion_effect(match.match_id, match.home_team, lineups)
        away_lineup = self._lineup_conversion_effect(match.match_id, match.away_team, lineups)

        expected_home_pps = (
            league_pps
            + self.team_attack_conversion[match.home_team]
            - self.team_defense_conversion[match.away_team]
            + self.home_adv_pps
            + home_lineup
        )
        expected_away_pps = (
            league_pps
            + self.team_attack_conversion[match.away_team]
            - self.team_defense_conversion[match.home_team]
            - self.home_adv_pps
            + away_lineup
        )
        expected_home_pps = max(3.5, min(6.3, expected_home_pps))
        expected_away_pps = max(3.5, min(6.3, expected_away_pps))

        expected_home_score = expected_home_pps * home_ss
        expected_away_score = expected_away_pps * away_ss
        expected_margin = expected_home_score - expected_away_score
        return (
            expected_home_score,
            expected_away_score,
            expected_margin,
            home_ss,
            away_ss,
            expected_home_pps,
            expected_away_pps,
        )

    def _update_player_conversion(
        self,
        match: MatchRow,
        lineups: Dict[Tuple[str, str], List[dict]],
    ):
        for team_name in (match.home_team, match.away_team):
            players = lineups.get((match.match_id, team_name), [])
            for player in players:
                player_name = player["player_name"]
                goals = player.get("goals", 0.0)
                behinds = player.get("behinds", 0.0)
                shots = goals + behinds
                if shots > 0:
                    player_pps = (6.0 * goals + behinds) / shots
                    centered = player_pps - 4.85
                    self.player_conversion_rating[player_name] = (
                        (1.0 - self.player_conversion_k) * self.player_conversion_rating[player_name]
                        + self.player_conversion_k * centered
                    )
                self.player_games[player_name] += 1

    def update(
        self,
        match: MatchRow,
        expected_home_ss: float,
        expected_away_ss: float,
        expected_margin: float,
        expected_home_pps: float,
        expected_away_pps: float,
        lineups: Dict[Tuple[str, str], List[dict]],
    ):
        home_ss, away_ss = self._actual_scoring_shots(match)
        home_shot_residual = max(
            -self.shot_residual_cap,
            min(self.shot_residual_cap, home_ss - expected_home_ss),
        )
        away_shot_residual = max(
            -self.shot_residual_cap,
            min(self.shot_residual_cap, away_ss - expected_away_ss),
        )

        self.team_attack_shots[match.home_team] += self.shot_k * home_shot_residual
        self.team_defense_shots[match.home_team] -= self.shot_k * away_shot_residual
        self.team_attack_shots[match.away_team] += self.shot_k * away_shot_residual
        self.team_defense_shots[match.away_team] -= self.shot_k * home_shot_residual

        actual_home_pps = match.home_score / home_ss
        actual_away_pps = match.away_score / away_ss

        home_conversion_residual = actual_home_pps - expected_home_pps
        away_conversion_residual = actual_away_pps - expected_away_pps

        self.team_attack_conversion[match.home_team] += self.conversion_k * home_conversion_residual
        self.team_defense_conversion[match.away_team] -= self.conversion_k * home_conversion_residual
        self.team_attack_conversion[match.away_team] += self.conversion_k * away_conversion_residual
        self.team_defense_conversion[match.home_team] -= self.conversion_k * away_conversion_residual

        total_shots = max(8.0, home_ss + away_ss)
        margin_residual_per_shot = (match.actual_margin - expected_margin) / total_shots
        self.home_adv_pps += self.home_adv_k * margin_residual_per_shot
        self.home_adv_pps = max(-0.6, min(0.6, self.home_adv_pps))

        self.points_per_shot_history.append(actual_home_pps)
        self.points_per_shot_history.append(actual_away_pps)
        self._update_player_conversion(match, lineups)

    def step(self, match: MatchRow, lineups: Dict[Tuple[str, str], List[dict]]) -> Tuple[float, float, float]:
        if self.current_year is None:
            self.current_year = match.year
        elif match.year != self.current_year:
            self.current_year = match.year
            self._apply_season_transition()

        (
            expected_home,
            expected_away,
            expected_margin,
            expected_home_ss,
            expected_away_ss,
            expected_home_pps,
            expected_away_pps,
        ) = self.predict(match, lineups)
        self.update(
            match,
            expected_home_ss,
            expected_away_ss,
            expected_margin,
            expected_home_pps,
            expected_away_pps,
            lineups,
        )
        return expected_home, expected_away, expected_margin


class VenueEnvironmentAdjustmentModel:
    """
    Learns a lightweight additive margin adjustment from contextual pre-game factors:
    weather, venue dimensions/capacity, and expected crowd level.
    """

    def __init__(
        self,
        learning_rate: float = 0.012,
        l2: float = 0.003,
        season_carryover: float = 0.9,
        max_adjustment: float = 16.0,
    ):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.season_carryover = season_carryover
        self.max_adjustment = max_adjustment
        self.current_year: Optional[int] = None

        self.weights: Dict[str, float] = defaultdict(float)
        self.home_venue_attendance: Dict[Tuple[str, str], float] = {}
        self.venue_attendance: Dict[str, float] = {}
        self.default_attendance = 28000.0
        self.default_capacity = 36000.0

    def _apply_season_transition(self):
        for feature in list(self.weights.keys()):
            self.weights[feature] *= self.season_carryover

    @staticmethod
    def _ema_update(store: Dict, key, observed: float, alpha: float = 0.22):
        if observed <= 0.0:
            return
        prior = store.get(key)
        if prior is None:
            store[key] = observed
        else:
            store[key] = (1.0 - alpha) * prior + alpha * observed

    def _expected_attendance(self, match: MatchRow, context: Optional[MatchContextRow]) -> float:
        if context is not None and context.projected_attendance is not None and context.projected_attendance > 0:
            return context.projected_attendance

        home_venue_key = (match.home_team, match.venue)
        if home_venue_key in self.home_venue_attendance:
            return self.home_venue_attendance[home_venue_key]
        if match.venue in self.venue_attendance:
            return self.venue_attendance[match.venue]

        if context is not None and context.venue_capacity is not None and context.venue_capacity > 0:
            return 0.62 * context.venue_capacity
        return self.default_attendance

    def _feature_vector(self, match: MatchRow, context: Optional[MatchContextRow]) -> Dict[str, float]:
        temp_c = 18.0
        rain_mm = 0.0
        wind_kmh = 10.0
        humidity_pct = 55.0
        venue_length_m = 160.0
        venue_width_m = 130.0
        venue_capacity = self.default_capacity
        if context is not None:
            if context.weather_temp_c is not None:
                temp_c = context.weather_temp_c
            if context.weather_rain_mm is not None:
                rain_mm = context.weather_rain_mm
            if context.weather_wind_kmh is not None:
                wind_kmh = context.weather_wind_kmh
            if context.weather_humidity_pct is not None:
                humidity_pct = context.weather_humidity_pct
            if context.venue_length_m is not None:
                venue_length_m = context.venue_length_m
            if context.venue_width_m is not None:
                venue_width_m = context.venue_width_m
            if context.venue_capacity is not None and context.venue_capacity > 0:
                venue_capacity = context.venue_capacity

        attendance_expected = self._expected_attendance(match, context)
        venue_area = venue_length_m * venue_width_m
        utilization = attendance_expected / max(5000.0, venue_capacity)

        return {
            "bias": 1.0,
            "temp": (temp_c - 18.0) / 10.0,
            "rain": max(0.0, min(40.0, rain_mm)) / 10.0,
            "wind": max(0.0, min(80.0, wind_kmh)) / 20.0,
            "humidity": (max(20.0, min(100.0, humidity_pct)) - 55.0) / 20.0,
            "venue_area": (venue_area - 20800.0) / 4500.0,
            "venue_capacity": (venue_capacity - 35000.0) / 22000.0,
            "attendance_level": (attendance_expected - 25000.0) / 15000.0,
            "attendance_utilization": utilization - 0.60,
        }

    def predict(self, match: MatchRow, base_margin: float, context: Optional[MatchContextRow]) -> float:
        features = self._feature_vector(match, context)
        adjustment = sum(self.weights[name] * value for name, value in features.items())
        adjustment = max(-self.max_adjustment, min(self.max_adjustment, adjustment))
        return base_margin + adjustment

    def update(
        self,
        match: MatchRow,
        predicted_margin: float,
        context: Optional[MatchContextRow],
    ):
        residual = match.actual_margin - predicted_margin
        features = self._feature_vector(match, context)
        for name, value in features.items():
            updated = (1.0 - self.l2) * self.weights[name] + self.learning_rate * residual * value
            self.weights[name] = max(-8.0, min(8.0, updated))

        if context is not None and context.attendance is not None and context.attendance > 0:
            self._ema_update(self.home_venue_attendance, (match.home_team, match.venue), context.attendance)
            self._ema_update(self.venue_attendance, match.venue, context.attendance)
            self.default_attendance = 0.98 * self.default_attendance + 0.02 * context.attendance

    def step(self, match: MatchRow, base_margin: float, context: Optional[MatchContextRow]) -> float:
        if self.current_year is None:
            self.current_year = match.year
        elif match.year != self.current_year:
            self.current_year = match.year
            self._apply_season_transition()

        predicted_margin = self.predict(match, base_margin, context)
        self.update(match, predicted_margin, context)
        return predicted_margin


def _mae(predictions: Iterable[PredictionRow]) -> float:
    rows = list(predictions)
    if not rows:
        return float("nan")
    return sum(r.abs_error for r in rows) / len(rows)


def _tip_is_correct(row: PredictionRow) -> bool:
    if row.actual_margin > 0:
        return row.predicted_margin > 0
    if row.actual_margin < 0:
        return row.predicted_margin < 0
    return row.predicted_margin == 0


def _default_home_win_probability(predicted_margin: float, margin_scale: float = 30.0) -> float:
    return 1.0 / (1.0 + math.exp(-predicted_margin / margin_scale))


def _fit_home_win_probability_calibrator(
    prior_rows: List[PredictionRow],
    min_rows: int = 120,
) -> Optional[LogisticRegression]:
    x_train = []
    y_train = []
    for row in prior_rows:
        if row.actual_margin == 0:
            continue
        x_train.append([row.predicted_margin])
        y_train.append(1 if row.actual_margin > 0 else 0)

    if len(x_train) < min_rows:
        return None
    if len(set(y_train)) < 2:
        return None

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(np.array(x_train, dtype=float), np.array(y_train, dtype=int))
    return model


def _predict_home_win_probabilities(
    rows: List[PredictionRow],
    calibrator: Optional[LogisticRegression],
) -> List[float]:
    eps = 1e-6
    if not rows:
        return []

    if calibrator is None:
        return [
            min(1.0 - eps, max(eps, _default_home_win_probability(row.predicted_margin)))
            for row in rows
        ]

    x = np.array([[row.predicted_margin] for row in rows], dtype=float)
    probs = calibrator.predict_proba(x)[:, 1].tolist()
    return [min(1.0 - eps, max(eps, prob)) for prob in probs]


def _bits_score(actual_margin: float, home_win_probability: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, home_win_probability))
    if actual_margin > 0:
        return 1.0 + math.log2(p)
    if actual_margin < 0:
        return 1.0 + math.log2(1.0 - p)
    return 1.0 + 0.5 * math.log2(p * (1.0 - p))


AVENUE_BASE_MODELS = (
    "team_only",
    "team_plus_lineup",
    "scoring_shots",
    "team_residual_lineup",
)


def _match_round_bucket(round_label: str) -> str:
    if is_finals_round(round_label):
        return "finals"
    round_number = round_sort_value(round_label)
    if round_number <= 6:
        return "early"
    if round_number <= 15:
        return "mid"
    return "late"


def _recent_history_rows(history_rows: List[dict], current_year: int, recent_years: int) -> List[dict]:
    if recent_years <= 0:
        return history_rows
    min_year = current_year - recent_years
    filtered = [row for row in history_rows if row["year"] >= min_year]
    return filtered if filtered else history_rows


def _fit_avenue9_blend_weights(
    history_rows: List[dict],
    current_year: int,
    half_life_years: float = 2.0,
) -> Tuple[float, float, float, float]:
    if len(history_rows) < 160:
        return (0.45, 0.25, 0.15, 0.15)

    best_weights = (0.45, 0.25, 0.15, 0.15)
    best_mae = float("inf")
    decay_base = 0.5 ** (1.0 / max(0.1, half_life_years))
    grid = [i / 20.0 for i in range(21)]
    for w_team in grid:
        for w_lineup in grid:
            if w_team + w_lineup > 1.0:
                continue
            for w_shots in grid:
                if w_team + w_lineup + w_shots > 1.0:
                    continue
                w_hybrid = 1.0 - (w_team + w_lineup + w_shots)
                abs_errors = 0.0
                total_weight = 0.0
                for row in history_rows:
                    pred = (
                        w_team * row["team_only"]
                        + w_lineup * row["team_plus_lineup"]
                        + w_shots * row["scoring_shots"]
                        + w_hybrid * row["team_residual_lineup"]
                    )
                    age_years = max(0, current_year - row["year"])
                    weight = decay_base ** age_years
                    abs_errors += weight * abs(row["actual_margin"] - pred)
                    total_weight += weight
                mae = abs_errors / max(1e-9, total_weight)
                if mae < best_mae:
                    best_mae = mae
                    best_weights = (w_team, w_lineup, w_shots, w_hybrid)
    return best_weights


def _build_shrunken_effect(
    rows: List[dict],
    prediction_key: str,
    key_getter,
    shrinkage: float,
) -> Dict[Tuple[str, ...], float]:
    sum_residual: Dict[Tuple[str, ...], float] = defaultdict(float)
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    for row in rows:
        key = key_getter(row)
        residual = row["actual_margin"] - row[prediction_key]
        sum_residual[key] += residual
        counts[key] += 1
    return {
        key: total / (counts[key] + shrinkage)
        for key, total in sum_residual.items()
    }


def _fit_avenue6_shrinkages(history_rows: List[dict], prediction_key: str) -> Tuple[float, float, float]:
    default = (30.0, 30.0, 40.0)
    years = sorted({row["year"] for row in history_rows})
    if len(years) < 2:
        return default

    validation_year = years[-1]
    train = [row for row in history_rows if row["year"] < validation_year]
    validation = [row for row in history_rows if row["year"] == validation_year]
    if len(train) < 160 or len(validation) < 80:
        return default

    candidates = [10.0, 20.0, 30.0, 45.0, 70.0]
    best = default
    best_mae = float("inf")
    for home_lambda in candidates:
        for away_lambda in candidates:
            for pair_lambda in candidates:
                home_effect = _build_shrunken_effect(
                    train,
                    prediction_key,
                    lambda row: (row["home_team"], row["venue"]),
                    home_lambda,
                )
                away_effect = _build_shrunken_effect(
                    train,
                    prediction_key,
                    lambda row: (row["away_team"], row["venue"]),
                    away_lambda,
                )
                pair_effect = _build_shrunken_effect(
                    train,
                    prediction_key,
                    lambda row: (row["home_team"], row["away_team"]),
                    pair_lambda,
                )

                abs_errors = 0.0
                for row in validation:
                    pred = (
                        row[prediction_key]
                        + home_effect.get((row["home_team"], row["venue"]), 0.0)
                        + away_effect.get((row["away_team"], row["venue"]), 0.0)
                        + pair_effect.get((row["home_team"], row["away_team"]), 0.0)
                    )
                    abs_errors += abs(row["actual_margin"] - pred)
                mae = abs_errors / len(validation)
                if mae < best_mae:
                    best_mae = mae
                    best = (home_lambda, away_lambda, pair_lambda)
    return best


def _fit_avenue5_finals_head(history_rows: List[dict], prediction_key: str) -> Tuple[float, float]:
    finals_rows = [row for row in history_rows if is_finals_round(row["round_label"])]
    if len(finals_rows) < 20:
        return 1.0, 0.0

    x = np.array([row[prediction_key] for row in finals_rows], dtype=float)
    y = np.array([row["actual_margin"] for row in finals_rows], dtype=float)
    x_mean = float(np.mean(x))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-6:
        return 1.0, 0.0
    slope = float(np.sum((x - x_mean) * (y - float(np.mean(y)))) / denom)
    intercept = float(np.mean(y) - slope * x_mean)

    shrink = len(finals_rows) / (len(finals_rows) + 40.0)
    slope = 1.0 + shrink * (slope - 1.0)
    intercept = shrink * intercept
    slope = max(0.7, min(1.25, slope))
    intercept = max(-10.0, min(10.0, intercept))
    return slope, intercept


def _fit_avenue7_bucket_scales(history_rows: List[dict], prediction_key: str) -> Dict[str, float]:
    scales = {"finals": 1.0, "early": 1.0, "mid": 1.0, "late": 1.0}
    candidates = [0.75 + 0.01 * i for i in range(31)]
    for bucket_name in tuple(scales.keys()):
        bucket_rows = [row for row in history_rows if _match_round_bucket(row["round_label"]) == bucket_name]
        if len(bucket_rows) < 40:
            continue
        best_scale = 1.0
        best_mae = float("inf")
        for scale in candidates:
            abs_errors = 0.0
            for row in bucket_rows:
                abs_errors += abs(row["actual_margin"] - (scale * row[prediction_key]))
            mae = abs_errors / len(bucket_rows)
            if mae < best_mae:
                best_mae = mae
                best_scale = scale
        scales[bucket_name] = best_scale
    return scales


def _apply_avenue2_tail_shrink(margin: float, threshold: float, slope: float) -> float:
    abs_margin = abs(margin)
    if abs_margin <= threshold:
        return margin
    shrunk = threshold + slope * (abs_margin - threshold)
    return math.copysign(shrunk, margin)


def _fit_avenue2_tail_shrink(history_rows: List[dict], prediction_key: str) -> Tuple[float, float]:
    if len(history_rows) < 200:
        return 35.0, 0.85

    years = sorted({row["year"] for row in history_rows})
    if len(years) >= 2:
        validation_year = years[-1]
        fit_rows = [row for row in history_rows if row["year"] < validation_year]
        eval_rows = [row for row in history_rows if row["year"] == validation_year]
        if len(fit_rows) < 160 or len(eval_rows) < 80:
            fit_rows = history_rows
            eval_rows = history_rows
    else:
        fit_rows = history_rows
        eval_rows = history_rows

    _ = fit_rows  # explicit to document year-locked split intent
    best = (35.0, 0.85)
    best_mae = float("inf")
    for threshold in (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0):
        for slope in (0.55, 0.65, 0.75, 0.85, 0.95, 1.0):
            abs_errors = 0.0
            for row in eval_rows:
                pred = _apply_avenue2_tail_shrink(row[prediction_key], threshold, slope)
                abs_errors += abs(row["actual_margin"] - pred)
            mae = abs_errors / len(eval_rows)
            if mae < best_mae:
                best_mae = mae
                best = (threshold, slope)
    return best


def _fit_market_line_weight(history_rows: List[dict]) -> float:
    if len(history_rows) < 120:
        return 0.7

    best_weight = 0.7
    best_mae = float("inf")
    for idx in range(51):
        weight = idx / 50.0
        abs_errors = 0.0
        for row in history_rows:
            blended_margin = weight * row["market_margin"] + (1.0 - weight) * row["base_margin"]
            abs_errors += abs(row["actual_margin"] - blended_margin)
        mae = abs_errors / len(history_rows)
        if mae < best_mae:
            best_mae = mae
            best_weight = weight
    return best_weight


def _market_implied_home_probability(home_odds: Optional[float], away_odds: Optional[float]) -> Optional[float]:
    if home_odds is None or away_odds is None:
        return None
    if home_odds <= 1.0 or away_odds <= 1.0:
        return None
    home_raw = 1.0 / home_odds
    away_raw = 1.0 / away_odds
    total = home_raw + away_raw
    if total <= 0:
        return None
    prob = home_raw / total
    return min(0.995, max(0.005, prob))


def _market_sigma_from_spread_and_probability(
    market_margin: Optional[float],
    home_prob: Optional[float],
    default_sigma: float = 30.0,
) -> float:
    if market_margin is None or home_prob is None:
        return default_sigma
    centered_prob = min(0.99, max(0.01, home_prob))
    z = NormalDist().inv_cdf(centered_prob)
    if abs(z) < 0.02:
        return default_sigma
    sigma = abs(market_margin) / abs(z)
    return max(12.0, min(80.0, sigma))


def _split_train_validation_rows(
    history_rows: List[dict],
    min_train_rows: int,
    min_validation_rows: int,
) -> Tuple[List[dict], List[dict]]:
    years = sorted({row["year"] for row in history_rows})
    if len(years) < 2:
        return history_rows, history_rows
    validation_year = years[-1]
    train_rows = [row for row in history_rows if row["year"] < validation_year]
    validation_rows = [row for row in history_rows if row["year"] == validation_year]
    if len(train_rows) < min_train_rows or len(validation_rows) < min_validation_rows:
        return history_rows, history_rows
    return train_rows, validation_rows


def _ridge_fit_coefficients(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(x.shape[1], dtype=float)
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(x, y)
    return np.array(model.coef_, dtype=float)


def _build_shrunken_venue_effect(
    rows: List[dict],
    residual_values: np.ndarray,
    shrinkage: float,
) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for row, residual in zip(rows, residual_values):
        venue = row.get("venue", "")
        sums[venue] += float(residual)
        counts[venue] += 1
    return {venue: sums[venue] / (counts[venue] + shrinkage) for venue in sums}


def _internal_adjustment_features(row: dict) -> np.ndarray:
    return np.array(
        [
            1.0,
            row["internal_margin"] / 35.0,
            row.get("form_ewma_01_diff", 0.0) / 25.0,
            row.get("form_ewma_03_diff", 0.0) / 25.0,
            row.get("form_ewma_05_diff", 0.0) / 25.0,
            row.get("form_roll3_diff", 0.0) / 25.0,
            row.get("form_roll5_diff", 0.0) / 25.0,
            row.get("form_roll10_diff", 0.0) / 25.0,
            row.get("rest_days_diff", 0.0) / 8.0,
            row.get("home_short_rest", 0.0),
            row.get("away_short_rest", 0.0),
            row.get("home_long_break", 0.0),
            row.get("away_long_break", 0.0),
            row.get("travel_km_diff", 0.0) / 1000.0,
            row.get("interstate_diff", 0.0),
            row.get("lineup_known_diff", 0.0),
            row.get("lineup_returning_diff", 0.0),
            row.get("lineup_debut_diff", 0.0),
            row.get("lineup_size_diff", 0.0) / 5.0,
            row.get("lineup_strength_diff", 0.0) / 6.0,
            row.get("lineup_experience_diff", 0.0) / 10.0,
            row.get("lineup_top_end_diff", 0.0) / 6.0,
            row.get("season_progress", 0.5) - 0.5,
            1.0 if is_finals_round(row["round_label"]) else 0.0,
        ],
        dtype=float,
    )


def _fit_internal_margin_adjuster(history_rows: List[dict]) -> dict:
    default = {"coef": np.zeros(24, dtype=float), "venue_effect": {}, "cap": 10.0}
    if len(history_rows) < 140:
        return default

    train_rows, validation_rows = _split_train_validation_rows(
        history_rows,
        min_train_rows=120,
        min_validation_rows=60,
    )
    x_train = np.array([_internal_adjustment_features(row) for row in train_rows], dtype=float)
    y_train = np.array(
        [max(-32.0, min(32.0, row["actual_margin"] - row["internal_margin"])) for row in train_rows],
        dtype=float,
    )
    x_validation = np.array([_internal_adjustment_features(row) for row in validation_rows], dtype=float)
    y_validation = np.array(
        [max(-32.0, min(32.0, row["actual_margin"] - row["internal_margin"])) for row in validation_rows],
        dtype=float,
    )

    best_alpha = 10.0
    best_shrinkage = 16.0
    best_cap = 10.0
    best_mae = float("inf")
    for alpha in (1.0, 3.0, 10.0, 30.0, 60.0):
        coef = _ridge_fit_coefficients(x_train, y_train, alpha=alpha)
        train_pred = x_train @ coef
        for shrinkage in (8.0, 16.0, 30.0, 50.0):
            venue_effect = _build_shrunken_venue_effect(train_rows, y_train - train_pred, shrinkage)
            for cap in (6.0, 8.0, 10.0, 12.0, 14.0):
                validation_pred = []
                for row, linear_pred in zip(validation_rows, x_validation @ coef):
                    venue_adj = venue_effect.get(row.get("venue", ""), 0.0)
                    adjustment = max(-cap, min(cap, float(linear_pred + venue_adj)))
                    validation_pred.append(adjustment)
                mae = float(np.mean(np.abs(y_validation - np.array(validation_pred, dtype=float))))
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha
                    best_shrinkage = shrinkage
                    best_cap = cap

    x_full = np.array([_internal_adjustment_features(row) for row in history_rows], dtype=float)
    y_full = np.array(
        [max(-32.0, min(32.0, row["actual_margin"] - row["internal_margin"])) for row in history_rows],
        dtype=float,
    )
    coef = _ridge_fit_coefficients(x_full, y_full, alpha=best_alpha)
    venue_effect = _build_shrunken_venue_effect(history_rows, y_full - (x_full @ coef), best_shrinkage)
    return {"coef": coef, "venue_effect": venue_effect, "cap": best_cap}


def _apply_internal_margin_adjuster(row: dict, config: dict) -> float:
    if not config or "coef" not in config:
        return row["internal_margin"]
    features = _internal_adjustment_features(row)
    venue_adjustment = config.get("venue_effect", {}).get(row.get("venue", ""), 0.0)
    cap = float(config.get("cap", 10.0))
    adjustment = float(features @ config["coef"] + venue_adjustment)
    adjustment = max(-cap, min(cap, adjustment))
    return row["internal_margin"] + adjustment


def _market_anchor_features(row: dict) -> np.ndarray:
    market_margin = row["market_margin"]
    internal_margin = row["internal_margin"]
    base_margin = row["base_margin"]
    implied_prob = row.get("implied_home_prob")
    implied_edge = 0.0 if implied_prob is None else (implied_prob - 0.5) * 2.0
    sigma = float(row.get("market_sigma", 30.0))
    sigma_norm = (sigma - 30.0) / 20.0
    disagreement = internal_margin - market_margin
    market_move = float(row.get("market_move", 0.0))
    total_move = float(row.get("market_total_move", 0.0))
    books = float(row.get("bookmakers_surveyed", 0.0))
    return np.array(
        [
            1.0,
            market_margin / 35.0,
            base_margin / 35.0,
            internal_margin / 35.0,
            (base_margin - market_margin) / 28.0,
            disagreement / 25.0,
            abs(disagreement) / 25.0,
            row.get("form_ewma_03_diff", 0.0) / 25.0,
            row.get("rest_days_diff", 0.0) / 8.0,
            row.get("travel_km_diff", 0.0) / 1000.0,
            sigma_norm,
            implied_edge,
            abs(market_move) / 8.0,
            total_move / 15.0,
            books / 12.0,
            row.get("lineup_strength_diff", 0.0) / 6.0,
            row.get("lineup_experience_diff", 0.0) / 10.0,
            row.get("lineup_top_end_diff", 0.0) / 6.0,
            row.get("lineup_known_diff", 0.0),
            row.get("lineup_debut_diff", 0.0),
            row.get("season_progress", 0.5) - 0.5,
            1.0 if is_finals_round(row["round_label"]) else 0.0,
        ],
        dtype=float,
    )


def _apply_market_anchor_model(row: dict, config: dict) -> float:
    if row.get("market_margin") is None:
        return row["base_margin"]
    if not config or "coef" not in config:
        return row["market_margin"]
    features = _market_anchor_features(row)
    raw_adjustment = float(features @ config["coef"])
    sigma_norm = max(0.0, (float(row.get("market_sigma", 30.0)) - 30.0) / 20.0)
    disagreement = abs(row["internal_margin"] - row["market_margin"])
    disagreement_norm = max(0.0, (disagreement - 6.0) / 20.0)
    damp = 1.0 / (
        1.0
        + float(config.get("sigma_k", 0.5)) * sigma_norm
        + float(config.get("disagreement_k", 0.2)) * disagreement_norm
    )
    adjustment = damp * raw_adjustment
    cap = float(config.get("cap", 4.0))
    adjustment = max(-cap, min(cap, adjustment))
    return row["market_margin"] + adjustment


def _fit_market_anchor_model(history_rows: List[dict]) -> dict:
    default = {"coef": np.zeros(22, dtype=float), "cap": 4.0, "sigma_k": 0.5, "disagreement_k": 0.2}
    if len(history_rows) < 140:
        return default

    train_rows, validation_rows = _split_train_validation_rows(
        history_rows,
        min_train_rows=120,
        min_validation_rows=60,
    )
    x_train = np.array([_market_anchor_features(row) for row in train_rows], dtype=float)
    y_train = np.array(
        [max(-25.0, min(25.0, row["actual_margin"] - row["market_margin"])) for row in train_rows],
        dtype=float,
    )
    best_alpha = 10.0
    best_cfg = dict(default)
    best_mae = float("inf")

    for alpha in (1.0, 3.0, 10.0, 25.0, 50.0):
        coef = _ridge_fit_coefficients(x_train, y_train, alpha=alpha)
        for cap in (2.0, 3.0, 4.0, 5.0, 7.0):
            for sigma_k in (0.0, 0.3, 0.6, 0.9):
                for disagreement_k in (0.0, 0.2, 0.4, 0.7):
                    cfg = {
                        "coef": coef,
                        "cap": cap,
                        "sigma_k": sigma_k,
                        "disagreement_k": disagreement_k,
                    }
                    abs_errors = 0.0
                    for row in validation_rows:
                        pred = _apply_market_anchor_model(row, cfg)
                        abs_errors += abs(row["actual_margin"] - pred)
                    mae = abs_errors / max(1, len(validation_rows))
                    if mae < best_mae:
                        best_mae = mae
                        best_alpha = alpha
                        best_cfg = cfg

    x_full = np.array([_market_anchor_features(row) for row in history_rows], dtype=float)
    y_full = np.array(
        [max(-25.0, min(25.0, row["actual_margin"] - row["market_margin"])) for row in history_rows],
        dtype=float,
    )
    coef = _ridge_fit_coefficients(x_full, y_full, alpha=best_alpha)
    return {
        "coef": coef,
        "cap": best_cfg["cap"],
        "sigma_k": best_cfg["sigma_k"],
        "disagreement_k": best_cfg["disagreement_k"],
    }


def _market_residual_features(row: dict) -> np.ndarray:
    anchor_margin = row["anchor_margin"]
    internal_margin = row["internal_margin"]
    base_margin = row["base_margin"]
    disagreement = internal_margin - anchor_margin
    abs_disagreement = abs(disagreement)
    disagreement_bucket = _disagreement_bucket(disagreement)
    round_number = round_sort_value(row["round_label"])
    is_early = 1.0 if (not is_finals_round(row["round_label"]) and round_number <= 6) else 0.0
    is_late = 1.0 if (not is_finals_round(row["round_label"]) and round_number >= 17) else 0.0

    implied_prob = row.get("implied_home_prob")
    implied_edge = 0.0 if implied_prob is None else (implied_prob - 0.5) * 2.0
    market_sigma = float(row.get("market_sigma", 30.0))
    sigma_norm = (market_sigma - 30.0) / 20.0
    market_move = float(row.get("market_move", 0.0))
    total_move = float(row.get("market_total_move", 0.0))
    finals_flag = 1.0 if is_finals_round(row["round_label"]) else 0.0
    disagreement_accuracy = float(row.get("disagreement_direction_accuracy", 0.5))
    return np.array(
        [
            1.0,
            anchor_margin / 35.0,
            internal_margin / 35.0,
            base_margin / 35.0,
            disagreement / 25.0,
            abs_disagreement / 25.0,
            (disagreement * disagreement) / 400.0,
            1.0 if disagreement_bucket == "small" else 0.0,
            1.0 if disagreement_bucket == "medium" else 0.0,
            1.0 if disagreement_bucket == "large" else 0.0,
            finals_flag,
            implied_edge,
            sigma_norm,
            row.get("season_progress", 0.5) - 0.5,
            is_early,
            is_late,
            row.get("form_ewma_03_diff", 0.0) / 25.0,
            row.get("form_roll5_diff", 0.0) / 25.0,
            row.get("rest_days_diff", 0.0) / 8.0,
            row.get("travel_km_diff", 0.0) / 1000.0,
            row.get("interstate_diff", 0.0),
            disagreement * finals_flag / 25.0,
            disagreement * row.get("form_ewma_03_diff", 0.0) / 500.0,
            disagreement * internal_margin / 700.0,
            disagreement_accuracy - 0.5,
            disagreement * (disagreement_accuracy - 0.5) / 25.0,
            1.0 if implied_prob is not None and implied_prob > 0.58 else 0.0,
            1.0 if implied_prob is not None and implied_prob < 0.42 else 0.0,
            abs(market_move) / 8.0,
            total_move / 15.0,
            row.get("lineup_strength_diff", 0.0) / 6.0,
            row.get("lineup_experience_diff", 0.0) / 10.0,
            row.get("lineup_top_end_diff", 0.0) / 6.0,
            row.get("lineup_known_diff", 0.0),
            row.get("lineup_debut_diff", 0.0),
            disagreement * abs(market_move) / 80.0,
            disagreement * row.get("lineup_strength_diff", 0.0) / 90.0,
        ],
        dtype=float,
    )


def _market_residual_meta_features(row: dict) -> np.ndarray:
    disagreement = row["internal_margin"] - row["anchor_margin"]
    sigma_norm = (float(row.get("market_sigma", 30.0)) - 30.0) / 20.0
    market_move = float(row.get("market_move", 0.0))
    return np.array(
        [
            1.0,
            disagreement / 25.0,
            abs(disagreement) / 25.0,
            sigma_norm,
            row.get("season_progress", 0.5) - 0.5,
            row.get("disagreement_direction_accuracy", 0.5) - 0.5,
            abs(market_move) / 8.0,
            row.get("lineup_strength_diff", 0.0) / 6.0,
        ],
        dtype=float,
    )


def _safe_fit_predict(model, x_train: np.ndarray, y_train: np.ndarray, x_predict: np.ndarray) -> np.ndarray:
    try:
        model.fit(x_train, y_train)
        return np.array(model.predict(x_predict), dtype=float)
    except Exception:
        return np.zeros(len(x_predict), dtype=float)


def _chronological_oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    model_builders: List,
    min_train_rows: int = 120,
) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0, len(model_builders)), dtype=float)
    oof = np.full((len(x), len(model_builders)), np.nan, dtype=float)
    unique_years = sorted(set(int(year) for year in years))
    for validation_year in unique_years[1:]:
        train_idx = np.where(years < validation_year)[0]
        validation_idx = np.where(years == validation_year)[0]
        if len(train_idx) < min_train_rows or len(validation_idx) == 0:
            continue
        for model_idx, builder in enumerate(model_builders):
            model = builder()
            preds = _safe_fit_predict(model, x[train_idx], y[train_idx], x[validation_idx])
            oof[validation_idx, model_idx] = preds
    return oof


def _fit_residual_base_builders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
):
    def mae(preds: np.ndarray) -> float:
        return float(np.mean(np.abs(y_validation - preds)))

    best_ridge_alpha = 10.0
    best_ridge_mae = float("inf")
    for alpha in (1.0, 3.0, 10.0, 25.0, 60.0):
        model = Ridge(alpha=alpha, fit_intercept=False)
        preds = _safe_fit_predict(model, x_train, y_train, x_validation)
        score = mae(preds)
        if score < best_ridge_mae:
            best_ridge_mae = score
            best_ridge_alpha = alpha

    best_huber_alpha = 0.001
    best_huber_eps = 1.4
    best_huber_mae = float("inf")
    for alpha in (0.0003, 0.001, 0.003):
        for epsilon in (1.25, 1.4, 1.6):
            model = HuberRegressor(alpha=alpha, epsilon=epsilon, fit_intercept=False, max_iter=500)
            preds = _safe_fit_predict(model, x_train, y_train, x_validation)
            score = mae(preds)
            if score < best_huber_mae:
                best_huber_mae = score
                best_huber_alpha = alpha
                best_huber_eps = epsilon

    best_hgb = {"learning_rate": 0.05, "max_depth": 4, "min_samples_leaf": 10, "l2_regularization": 0.0}
    best_hgb_mae = float("inf")
    for learning_rate in (0.03, 0.06):
        for max_depth in (3, 5):
            for min_samples_leaf in (8, 14):
                for l2_regularization in (0.0, 0.3):
                    model = HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        l2_regularization=l2_regularization,
                        max_iter=140,
                        random_state=42,
                    )
                    preds = _safe_fit_predict(model, x_train, y_train, x_validation)
                    score = mae(preds)
                    if score < best_hgb_mae:
                        best_hgb_mae = score
                        best_hgb = {
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "l2_regularization": l2_regularization,
                        }

    return [
        lambda alpha=best_ridge_alpha: Ridge(alpha=alpha, fit_intercept=False),
        lambda alpha=best_huber_alpha, eps=best_huber_eps: HuberRegressor(
            alpha=alpha,
            epsilon=eps,
            fit_intercept=False,
            max_iter=500,
        ),
        lambda params=best_hgb: HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            max_iter=140,
            random_state=42,
        ),
    ]


def _fit_residual_core(history_rows: List[dict], model_builders: List) -> dict:
    x = np.array([_market_residual_features(row) for row in history_rows], dtype=float)
    y = np.array(
        [max(-35.0, min(35.0, row["actual_margin"] - row["anchor_margin"])) for row in history_rows],
        dtype=float,
    )
    years = np.array([row["year"] for row in history_rows], dtype=int)
    base_models = []
    for builder in model_builders:
        model = builder()
        _safe_fit_predict(model, x, y, x[:1])
        base_models.append(model)

    oof_base = _chronological_oof_predictions(x, y, years, model_builders, min_train_rows=120)
    meta_extra = np.array([_market_residual_meta_features(row) for row in history_rows], dtype=float)
    valid_mask = ~np.isnan(oof_base).any(axis=1)
    if int(np.sum(valid_mask)) < 120:
        in_sample_base = np.column_stack([np.array(model.predict(x), dtype=float) for model in base_models])
        meta_x = np.hstack([in_sample_base, meta_extra])
        meta_y = y
        calibration_x = meta_x
        calibration_y = y
    else:
        meta_x = np.hstack([oof_base[valid_mask], meta_extra[valid_mask]])
        meta_y = y[valid_mask]
        calibration_x = meta_x
        calibration_y = meta_y

    best_meta_alpha = 3.0
    best_meta_mae = float("inf")
    for alpha in (0.3, 1.0, 3.0, 8.0, 20.0):
        meta_model = Ridge(alpha=alpha, fit_intercept=False)
        meta_model.fit(meta_x, meta_y)
        preds = np.array(meta_model.predict(meta_x), dtype=float)
        score = float(np.mean(np.abs(meta_y - preds)))
        if score < best_meta_mae:
            best_meta_mae = score
            best_meta_alpha = alpha

    meta_model = Ridge(alpha=best_meta_alpha, fit_intercept=False)
    meta_model.fit(meta_x, meta_y)
    calibration_raw = np.array(meta_model.predict(calibration_x), dtype=float)
    calibrator = None
    if len(calibration_raw) >= 120 and np.ptp(calibration_raw) > 1e-6:
        try:
            isotonic = IsotonicRegression(out_of_bounds="clip")
            isotonic.fit(calibration_raw, calibration_y)
            mae_raw = float(np.mean(np.abs(calibration_y - calibration_raw)))
            mae_iso = float(np.mean(np.abs(calibration_y - isotonic.predict(calibration_raw))))
            if mae_iso + 0.01 < mae_raw:
                calibrator = isotonic
        except Exception:
            calibrator = None

    abs_model = Ridge(alpha=12.0, fit_intercept=False)
    abs_model.fit(x, np.abs(y))
    return {
        "base_models": base_models,
        "meta_model": meta_model,
        "calibrator": calibrator,
        "abs_model": abs_model,
    }


def _predict_raw_market_residual(row: dict, config: dict) -> float:
    if not config or not config.get("base_models") or config.get("meta_model") is None:
        return 0.0
    features = _market_residual_features(row).reshape(1, -1)
    meta_features = _market_residual_meta_features(row).reshape(1, -1)
    base_preds = np.array([float(model.predict(features)[0]) for model in config["base_models"]], dtype=float)
    meta_x = np.hstack([base_preds.reshape(1, -1), meta_features])
    raw = float(config["meta_model"].predict(meta_x)[0])
    calibrator = config.get("calibrator")
    if calibrator is not None:
        raw = float(calibrator.predict(np.array([raw], dtype=float))[0])
    return raw


def _safety_context_key(row: dict) -> Tuple[str, str, str]:
    finals_bucket = "finals" if is_finals_round(row["round_label"]) else "regular"
    disagreement_bucket = _disagreement_bucket(row["internal_margin"] - row["anchor_margin"])
    market_sigma = float(row.get("market_sigma", 30.0))
    if market_sigma < 24.0:
        sigma_bucket = "sigma_low"
    elif market_sigma < 34.0:
        sigma_bucket = "sigma_mid"
    else:
        sigma_bucket = "sigma_high"
    return finals_bucket, disagreement_bucket, sigma_bucket


def _build_context_caps(history_rows: List[dict], quantile: float) -> Tuple[Dict[Tuple[str, str, str], float], float]:
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    residual_abs = []
    for row in history_rows:
        value = abs(row["actual_margin"] - row["anchor_margin"])
        residual_abs.append(value)
        grouped[_safety_context_key(row)].append(value)
    if not residual_abs:
        return {}, 5.0
    global_cap = float(np.quantile(np.array(residual_abs, dtype=float), quantile))
    context_caps = {}
    for key, values in grouped.items():
        if len(values) < 24:
            continue
        context_caps[key] = float(np.quantile(np.array(values, dtype=float), quantile))
    return context_caps, global_cap


def _apply_market_residual_safety(
    row: dict,
    raw_correction: float,
    predicted_abs_residual: float,
    config: dict,
) -> float:
    sigma = float(row.get("market_sigma", 30.0))
    sigma_norm = max(0.0, (sigma - 30.0) / 20.0)
    disagreement = abs(row["internal_margin"] - row["anchor_margin"])
    disagreement_norm = max(0.0, (disagreement - 6.0) / 20.0)
    reliability = float(row.get("disagreement_direction_accuracy", 0.5))
    line_move_norm = abs(float(row.get("market_move", 0.0))) / 8.0
    lineup_volatility = max(0.0, min(2.0, float(row.get("lineup_volatility", 0.0))))

    damp = 1.0 / (
        1.0
        + float(config.get("sigma_k", 0.6)) * sigma_norm
        + float(config.get("disagreement_k", 0.25)) * disagreement_norm
        + float(config.get("line_move_k", 0.0)) * line_move_norm
        + float(config.get("lineup_k", 0.0)) * lineup_volatility
    )
    damp *= 1.0 + float(config.get("reliability_k", 0.2)) * ((reliability - 0.5) * 2.0)
    damp = max(0.2, min(1.25, damp))

    predicted_abs_residual = max(2.0, min(30.0, predicted_abs_residual))

    context_caps = config.get("context_caps", {})
    context_cap = context_caps.get(_safety_context_key(row), float(config.get("global_cap", 6.0)))
    cap_scale = float(config.get("cap_scale", 0.7))
    min_cap = float(config.get("min_cap", 3.0))
    dynamic_cap = float(config.get("global_cap", 6.0)) + cap_scale * predicted_abs_residual
    cap = max(min_cap, min(context_cap, dynamic_cap))

    correction = damp * raw_correction * float(config.get("post_scale", 1.0))
    return max(-cap, min(cap, correction))


def _apply_market_residual_corrector(row: dict, config: dict) -> float:
    if not config or not config.get("base_models") or config.get("meta_model") is None:
        return row["anchor_margin"]

    raw_correction = _predict_raw_market_residual(row, config)
    abs_model = config.get("abs_model")
    if abs_model is None:
        predicted_abs_residual = float(config.get("global_cap", 6.0))
    else:
        try:
            predicted_abs_residual = float(
                abs_model.predict(_market_residual_features(row).reshape(1, -1))[0]
            )
        except Exception:
            predicted_abs_residual = float(config.get("global_cap", 6.0))

    correction = _apply_market_residual_safety(
        row,
        raw_correction=raw_correction,
        predicted_abs_residual=predicted_abs_residual,
        config=config,
    )
    return row["anchor_margin"] + correction


def _fit_market_residual_corrector(history_rows: List[dict]) -> dict:
    default = {
        "base_models": [],
        "meta_model": None,
        "calibrator": None,
        "abs_model": None,
        "context_caps": {},
        "global_cap": 5.0,
        "sigma_k": 0.6,
        "disagreement_k": 0.25,
        "reliability_k": 0.2,
        "line_move_k": 0.0,
        "lineup_k": 0.0,
        "cap_scale": 0.7,
        "min_cap": 3.0,
        "post_scale": 1.0,
    }
    if len(history_rows) < 160:
        return default

    train_rows, validation_rows = _split_train_validation_rows(
        history_rows,
        min_train_rows=140,
        min_validation_rows=70,
    )
    x_train = np.array([_market_residual_features(row) for row in train_rows], dtype=float)
    y_train = np.array(
        [max(-35.0, min(35.0, row["actual_margin"] - row["anchor_margin"])) for row in train_rows],
        dtype=float,
    )
    x_validation = np.array([_market_residual_features(row) for row in validation_rows], dtype=float)
    y_validation = np.array(
        [max(-35.0, min(35.0, row["actual_margin"] - row["anchor_margin"])) for row in validation_rows],
        dtype=float,
    )
    model_builders = _fit_residual_base_builders(x_train, y_train, x_validation, y_validation)
    core_train = _fit_residual_core(train_rows, model_builders)

    best_safety = {
        "sigma_k": 0.6,
        "disagreement_k": 0.25,
        "reliability_k": 0.2,
        "line_move_k": 0.0,
        "lineup_k": 0.0,
        "cap_scale": 0.7,
        "min_cap": 3.0,
        "post_scale": 1.0,
        "quantile": 0.85,
    }
    best_mae = float("inf")
    validation_raw_corrections = [_predict_raw_market_residual(row, core_train) for row in validation_rows]
    if core_train.get("abs_model") is not None:
        validation_abs_predictions = [
            float(core_train["abs_model"].predict(_market_residual_features(row).reshape(1, -1))[0])
            for row in validation_rows
        ]
    else:
        validation_abs_predictions = [5.0 for _ in validation_rows]

    for quantile in (0.82, 0.90):
        context_caps, global_cap = _build_context_caps(train_rows, quantile=quantile)
        trial = dict(core_train)
        trial["context_caps"] = context_caps
        trial["global_cap"] = global_cap
        for sigma_k in (0.35, 0.75):
            for disagreement_k in (0.1, 0.35):
                for reliability_k in (0.0, 0.25):
                    for line_move_k in (0.0, 0.35):
                        for lineup_k in (0.0, 0.25):
                            for cap_scale in (0.55, 0.85):
                                for min_cap in (3.0, 4.0):
                                    for post_scale in (0.6, 0.8, 1.0):
                                        trial.update(
                                            {
                                                "sigma_k": sigma_k,
                                                "disagreement_k": disagreement_k,
                                                "reliability_k": reliability_k,
                                                "line_move_k": line_move_k,
                                                "lineup_k": lineup_k,
                                                "cap_scale": cap_scale,
                                                "min_cap": min_cap,
                                                "post_scale": post_scale,
                                            }
                                        )
                                        abs_errors = 0.0
                                        for row, raw_correction, pred_abs in zip(
                                            validation_rows,
                                            validation_raw_corrections,
                                            validation_abs_predictions,
                                        ):
                                            pred = row["anchor_margin"] + _apply_market_residual_safety(
                                                row,
                                                raw_correction=raw_correction,
                                                predicted_abs_residual=pred_abs,
                                                config=trial,
                                            )
                                            abs_errors += abs(row["actual_margin"] - pred)
                                        mae = abs_errors / max(1, len(validation_rows))
                                        if mae < best_mae:
                                            best_mae = mae
                                            best_safety = {
                                                "sigma_k": sigma_k,
                                                "disagreement_k": disagreement_k,
                                                "reliability_k": reliability_k,
                                                "line_move_k": line_move_k,
                                                "lineup_k": lineup_k,
                                                "cap_scale": cap_scale,
                                                "min_cap": min_cap,
                                                "post_scale": post_scale,
                                                "quantile": quantile,
                                            }

    full_core = _fit_residual_core(history_rows, model_builders)
    context_caps, global_cap = _build_context_caps(history_rows, quantile=best_safety["quantile"])
    full_core.update(
        {
            "context_caps": context_caps,
            "global_cap": global_cap,
            "sigma_k": best_safety["sigma_k"],
            "disagreement_k": best_safety["disagreement_k"],
            "reliability_k": best_safety["reliability_k"],
            "line_move_k": best_safety["line_move_k"],
            "lineup_k": best_safety["lineup_k"],
            "cap_scale": best_safety["cap_scale"],
            "min_cap": best_safety["min_cap"],
            "post_scale": best_safety["post_scale"],
        }
    )
    return full_core


def _build_avenue_experiments(base_predictions: List[PredictionRow]) -> List[PredictionRow]:
    match_rows: Dict[Tuple[int, str], dict] = {}
    for row in base_predictions:
        if row.model_name not in AVENUE_BASE_MODELS:
            continue
        key = (row.year, row.match_id)
        if key not in match_rows:
            match_rows[key] = {
                "match_id": row.match_id,
                "year": row.year,
                "round_label": row.round_label,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "venue": row.venue,
                "actual_margin": row.actual_margin,
            }
        match_rows[key][row.model_name] = row.predicted_margin

    complete_rows = [
        row
        for row in match_rows.values()
        if all(model_name in row for model_name in AVENUE_BASE_MODELS)
    ]
    complete_rows.sort(
        key=lambda row: (
            row["year"],
            round_sort_value(row["round_label"]),
            row["match_id"],
        )
    )
    if not complete_rows:
        return []

    output: List[PredictionRow] = []
    history_rows: List[dict] = []
    recency_years = 3
    for year in sorted({row["year"] for row in complete_rows}):
        year_rows = [dict(row) for row in complete_rows if row["year"] == year]
        fit_history_rows = _recent_history_rows(history_rows, year, recency_years)

        w_team, w_lineup, w_shots, w_hybrid = _fit_avenue9_blend_weights(
            fit_history_rows,
            current_year=year,
        )
        for row in year_rows:
            row["pred_av9_ensemble"] = (
                w_team * row["team_only"]
                + w_lineup * row["team_plus_lineup"]
                + w_shots * row["scoring_shots"]
                + w_hybrid * row["team_residual_lineup"]
            )

        home_lambda, away_lambda, pair_lambda = _fit_avenue6_shrinkages(fit_history_rows, "pred_av9_ensemble")
        home_effect = _build_shrunken_effect(
            fit_history_rows,
            "pred_av9_ensemble",
            lambda row: (row["home_team"], row["venue"]),
            home_lambda,
        )
        away_effect = _build_shrunken_effect(
            fit_history_rows,
            "pred_av9_ensemble",
            lambda row: (row["away_team"], row["venue"]),
            away_lambda,
        )
        pair_effect = _build_shrunken_effect(
            fit_history_rows,
            "pred_av9_ensemble",
            lambda row: (row["home_team"], row["away_team"]),
            pair_lambda,
        )
        for row in year_rows:
            row["pred_av6_matchup"] = (
                row["pred_av9_ensemble"]
                + home_effect.get((row["home_team"], row["venue"]), 0.0)
                + away_effect.get((row["away_team"], row["venue"]), 0.0)
                + pair_effect.get((row["home_team"], row["away_team"]), 0.0)
            )

        finals_scale, finals_bias = _fit_avenue5_finals_head(fit_history_rows, "pred_av6_matchup")
        for row in year_rows:
            if is_finals_round(row["round_label"]):
                row["pred_av5_finals_head"] = finals_scale * row["pred_av6_matchup"] + finals_bias
            else:
                row["pred_av5_finals_head"] = row["pred_av6_matchup"]

        bucket_scales = _fit_avenue7_bucket_scales(fit_history_rows, "pred_av5_finals_head")
        for row in year_rows:
            bucket_name = _match_round_bucket(row["round_label"])
            row["pred_av7_uncertainty"] = bucket_scales[bucket_name] * row["pred_av5_finals_head"]

        tail_threshold, tail_slope = _fit_avenue2_tail_shrink(fit_history_rows, "pred_av7_uncertainty")
        for row in year_rows:
            row["pred_av2_tail_cal"] = _apply_avenue2_tail_shrink(
                row["pred_av7_uncertainty"],
                tail_threshold,
                tail_slope,
            )

        model_map = (
            ("av9_ensemble", "pred_av9_ensemble"),
            ("av6_matchup", "pred_av6_matchup"),
            ("av5_finals_head", "pred_av5_finals_head"),
            ("av7_uncertainty", "pred_av7_uncertainty"),
            ("av2_tail_cal", "pred_av2_tail_cal"),
        )
        for row in year_rows:
            for model_name, pred_key in model_map:
                predicted_margin = row[pred_key]
                output.append(
                    PredictionRow(
                        match_id=row["match_id"],
                        year=row["year"],
                        round_label=row["round_label"],
                        home_team=row["home_team"],
                        away_team=row["away_team"],
                        venue=row["venue"],
                        actual_margin=row["actual_margin"],
                        predicted_margin=predicted_margin,
                        abs_error=abs(row["actual_margin"] - predicted_margin),
                        model_name=model_name,
                    )
                )
        history_rows.extend(year_rows)

    return output


def walk_forward_predictions(
    matches: List[MatchRow],
    lineups: Dict[Tuple[str, str], List[dict]],
    min_train_years: int,
    market_data: Optional[Dict[Tuple[date, str, str], dict]] = None,
    match_context_data: Optional[Dict[Tuple[date, str, str], MatchContextRow]] = None,
) -> List[PredictionRow]:
    if not matches:
        return []
    if market_data is None:
        market_data = {}
    if match_context_data is None:
        match_context_data = {}

    first_year = min(m.year for m in matches)
    scoring_year = first_year + min_train_years
    models = {
        "team_only": SequentialMarginModel(
            use_lineups=False,
            base_score=76.0,
            home_advantage=3.0,
            team_k=0.05,
            defense_k=0.08,
            season_carryover=0.78,
            home_adv_k=0.0,
        ),
        "team_plus_lineup": SequentialMarginModel(
            use_lineups=True,
            base_score=76.0,
            home_advantage=3.0,
            team_k=0.04,
            defense_k=0.06,
            season_carryover=0.78,
            player_k=0.03,
            lineup_scale=8.0,
            min_player_games=0,
            home_adv_k=0.0,
        ),
        "scoring_shots": ScoringShotMarginModel(
            base_scoring_shots=26.0,
            home_adv_scoring_shots=1.0,
            shot_k=0.07,
            season_carryover=0.85,
            shot_residual_cap=12.0,
        ),
        "av1_territory_chain": TerritoryShotChainModel(
            base_inside50=54.0,
            home_adv_inside50=1.8,
            inside50_k=0.045,
            base_shot_rate=0.46,
            shot_rate_k=0.035,
            base_points_per_shot=4.85,
            conversion_k=0.03,
            home_adv_pps=0.08,
            home_adv_pps_k=0.001,
            season_carryover=0.8,
        ),
    }
    hybrid_model = BenchmarkShotVolumeModel(
        conversion_k=0.045,
        home_adv_k=0.002,
        season_carryover=0.78,
        conversion_window_games=800,
    )
    context_model = VenueEnvironmentAdjustmentModel()

    predictions: List[PredictionRow] = []
    form_tracker = TeamFeatureTracker()
    lineup_tracker = KnownLineupTracker()
    lineup_strength_tracker = LineupStrengthTracker()
    disagreement_tracker = DisagreementReliabilityTracker()
    internal_fit_history: List[dict] = []
    internal_adjuster_by_year: Dict[int, dict] = {}
    market_fit_history: List[dict] = []
    market_anchor_config_by_year: Dict[int, dict] = {}
    market_residual_config_by_year: Dict[int, dict] = {}
    internal_recent_years = 6
    market_anchor_recent_years = 6
    market_residual_recent_years = 4

    for match in matches:
        pregame_features = form_tracker.pre_match_features(match)
        pregame_features.update(lineup_tracker.pre_match_features(match, lineups))
        pregame_features.update(lineup_strength_tracker.pre_match_features(match, lineups))
        match_margins: Dict[str, float] = {}
        for model_name, model in models.items():
            if model_name == "scoring_shots":
                expected_home, expected_away, predicted_margin = model.step(match)
            else:
                expected_home, expected_away, predicted_margin = model.step(match, lineups)
            match_margins[model_name] = predicted_margin
            if match.year >= scoring_year:
                predictions.append(
                    PredictionRow(
                        match_id=match.match_id,
                        year=match.year,
                        round_label=match.round_label,
                        home_team=match.home_team,
                        away_team=match.away_team,
                        venue=match.venue,
                        actual_margin=match.actual_margin,
                        predicted_margin=predicted_margin,
                        abs_error=abs(match.actual_margin - predicted_margin),
                        model_name=model_name,
                    )
                )

        _, _, hybrid_margin = hybrid_model.step(match)
        match_margins["team_residual_lineup"] = hybrid_margin
        context_key = (match.date.date(), match.home_team, match.away_team)
        context_row = match_context_data.get(context_key)
        context_margin = context_model.step(
            match=match,
            base_margin=match_margins["team_only"],
            context=context_row,
        )
        match_margins["team_context_env"] = context_margin

        internal_adjustment_row = {
            "year": match.year,
            "round_label": match.round_label,
            "venue": match.venue,
            "internal_margin": match_margins["team_residual_lineup"],
            **pregame_features,
        }
        if match.year not in internal_adjuster_by_year:
            prior_internal_rows = [row for row in internal_fit_history if row["year"] < match.year]
            fit_internal_rows = _recent_history_rows(prior_internal_rows, match.year, internal_recent_years)
            internal_adjuster_by_year[match.year] = _fit_internal_margin_adjuster(fit_internal_rows)
        internal_margin_enhanced = _apply_internal_margin_adjuster(
            internal_adjustment_row,
            internal_adjuster_by_year[match.year],
        )

        base_margin = match_margins["scoring_shots"]
        market_row = market_data.get(context_key)
        market_margin: Optional[float] = None
        implied_home_prob: Optional[float] = None
        market_sigma: float = 30.0
        market_move: float = 0.0
        market_range: float = 0.0
        market_total_close: float = 165.0
        market_total_move: float = 0.0
        bookmakers_surveyed: float = 0.0
        if market_row is not None and market_row.get("home_line_close") is not None:
            # Betting line is typically home handicap; negative implies home favourite.
            # Convert to predicted home margin orientation.
            market_margin = -float(market_row["home_line_close"])
            implied_home_prob = _market_implied_home_probability(
                market_row.get("home_odds"),
                market_row.get("away_odds"),
            )
            market_sigma = _market_sigma_from_spread_and_probability(market_margin, implied_home_prob)
            home_line_open = market_row.get("home_line_open")
            if home_line_open is not None:
                market_move = market_margin - (-float(home_line_open))
            home_line_min = market_row.get("home_line_min")
            home_line_max = market_row.get("home_line_max")
            if home_line_min is not None and home_line_max is not None:
                market_range = abs(float(home_line_max) - float(home_line_min))
            total_score_close = market_row.get("total_score_close")
            total_score_open = market_row.get("total_score_open")
            if total_score_close is not None:
                market_total_close = float(total_score_close)
                if total_score_open is not None:
                    market_total_move = float(total_score_close) - float(total_score_open)
            books = market_row.get("bookmakers_surveyed")
            if books is not None:
                bookmakers_surveyed = float(books)

        if market_margin is None:
            market_anchor_margin = 0.6 * base_margin + 0.4 * internal_margin_enhanced
            market_only_margin = base_margin
            market_residual_margin = market_anchor_margin
        else:
            if match.year not in market_anchor_config_by_year:
                prior_market_rows = [row for row in market_fit_history if row["year"] < match.year]
                anchor_fit_rows = _recent_history_rows(prior_market_rows, match.year, market_anchor_recent_years)
                residual_fit_rows = _recent_history_rows(prior_market_rows, match.year, market_residual_recent_years)
                market_anchor_config_by_year[match.year] = _fit_market_anchor_model(anchor_fit_rows)
                market_residual_config_by_year[match.year] = _fit_market_residual_corrector(residual_fit_rows)

            anchor_row = {
                "year": match.year,
                "round_label": match.round_label,
                "venue": match.venue,
                "market_margin": market_margin,
                "base_margin": base_margin,
                "internal_margin": internal_margin_enhanced,
                "implied_home_prob": implied_home_prob,
                "market_sigma": market_sigma,
                "market_move": market_move,
                "market_range": market_range,
                "market_total_close": market_total_close,
                "market_total_move": market_total_move,
                "bookmakers_surveyed": bookmakers_surveyed,
                **pregame_features,
            }
            market_anchor_margin = _apply_market_anchor_model(
                anchor_row,
                market_anchor_config_by_year[match.year],
            )
            disagreement = internal_margin_enhanced - market_anchor_margin
            disagreement_accuracy = disagreement_tracker.expected_accuracy(
                match.round_label,
                implied_home_prob,
                market_sigma,
                disagreement,
            )
            residual_row = {
                "year": match.year,
                "round_label": match.round_label,
                "venue": match.venue,
                "anchor_margin": market_anchor_margin,
                "base_margin": base_margin,
                "internal_margin": internal_margin_enhanced,
                "implied_home_prob": implied_home_prob,
                "market_sigma": market_sigma,
                "market_move": market_move,
                "market_range": market_range,
                "market_total_close": market_total_close,
                "market_total_move": market_total_move,
                "bookmakers_surveyed": bookmakers_surveyed,
                "disagreement_direction_accuracy": disagreement_accuracy,
                **pregame_features,
            }
            market_residual_margin = _apply_market_residual_corrector(
                residual_row,
                market_residual_config_by_year[match.year],
            )
            market_only_margin = market_margin

            market_fit_history.append(
                {
                    "year": match.year,
                    "round_label": match.round_label,
                    "venue": match.venue,
                    "actual_margin": match.actual_margin,
                    "base_margin": base_margin,
                    "market_margin": market_margin,
                    "anchor_margin": market_anchor_margin,
                    "internal_margin": internal_margin_enhanced,
                    "implied_home_prob": implied_home_prob,
                    "market_sigma": market_sigma,
                    "market_move": market_move,
                    "market_range": market_range,
                    "market_total_close": market_total_close,
                    "market_total_move": market_total_move,
                    "bookmakers_surveyed": bookmakers_surveyed,
                    "disagreement_direction_accuracy": disagreement_accuracy,
                    **pregame_features,
                }
            )
            disagreement_tracker.update(
                match.round_label,
                implied_home_prob,
                market_sigma,
                disagreement,
                match.actual_margin - market_anchor_margin,
            )

        if match.year >= scoring_year:
            predictions.append(
                PredictionRow(
                    match_id=match.match_id,
                    year=match.year,
                    round_label=match.round_label,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    venue=match.venue,
                    actual_margin=match.actual_margin,
                    predicted_margin=hybrid_margin,
                    abs_error=abs(match.actual_margin - hybrid_margin),
                    model_name="team_residual_lineup",
                )
            )
            predictions.append(
                PredictionRow(
                    match_id=match.match_id,
                    year=match.year,
                    round_label=match.round_label,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    venue=match.venue,
                    actual_margin=match.actual_margin,
                    predicted_margin=context_margin,
                    abs_error=abs(match.actual_margin - context_margin),
                    model_name="team_context_env",
                )
            )
            predictions.append(
                PredictionRow(
                    match_id=match.match_id,
                    year=match.year,
                    round_label=match.round_label,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    venue=match.venue,
                    actual_margin=match.actual_margin,
                    predicted_margin=internal_margin_enhanced,
                    abs_error=abs(match.actual_margin - internal_margin_enhanced),
                    model_name="team_residual_lineup_form",
                )
            )

            predictions.append(
                PredictionRow(
                    match_id=match.match_id,
                    year=match.year,
                    round_label=match.round_label,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    venue=match.venue,
                    actual_margin=match.actual_margin,
                    predicted_margin=market_anchor_margin,
                    abs_error=abs(match.actual_margin - market_anchor_margin),
                    model_name="market_line_blend",
                )
            )
            predictions.append(
                PredictionRow(
                    match_id=match.match_id,
                    year=match.year,
                    round_label=match.round_label,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    venue=match.venue,
                    actual_margin=match.actual_margin,
                    predicted_margin=market_only_margin,
                    abs_error=abs(match.actual_margin - market_only_margin),
                    model_name="market_only",
                )
            )
            predictions.append(
                PredictionRow(
                    match_id=match.match_id,
                    year=match.year,
                    round_label=match.round_label,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    venue=match.venue,
                    actual_margin=match.actual_margin,
                    predicted_margin=market_residual_margin,
                    abs_error=abs(match.actual_margin - market_residual_margin),
                    model_name="market_residual_corrector",
                )
            )

        internal_fit_history.append(
            {
                "year": match.year,
                "round_label": match.round_label,
                "venue": match.venue,
                "actual_margin": match.actual_margin,
                "internal_margin": match_margins["team_residual_lineup"],
                **pregame_features,
            }
        )
        form_tracker.update(match)
        lineup_tracker.update(match, lineups)
        lineup_strength_tracker.update(match, lineups)

    predictions.extend(_build_avenue_experiments(predictions))
    return predictions


def summarize_predictions(predictions: List[PredictionRow]) -> List[dict]:
    summaries: List[dict] = []
    by_model: Dict[str, List[PredictionRow]] = defaultdict(list)

    for row in predictions:
        by_model[row.model_name].append(row)

    for model_name in sorted(by_model.keys()):
        rows = by_model[model_name]
        rows_by_year: Dict[int, List[PredictionRow]] = defaultdict(list)
        for row in rows:
            rows_by_year[row.year].append(row)

        years = sorted(rows_by_year.keys())
        prior_rows: List[PredictionRow] = []
        model_bits_total = 0.0
        model_tip_correct = 0
        model_games = 0

        for year in years:
            year_rows = rows_by_year[year]
            calibrator = _fit_home_win_probability_calibrator(prior_rows)
            home_probs = _predict_home_win_probabilities(year_rows, calibrator)
            bits_total = sum(_bits_score(row.actual_margin, prob) for row, prob in zip(year_rows, home_probs))
            tips_correct = sum(1 for row in year_rows if _tip_is_correct(row))

            summaries.append(
                {
                    "model_name": model_name,
                    "year": year,
                    "num_games": len(year_rows),
                    "mae_margin": round(_mae(year_rows), 4),
                    "tip_pct": round(100.0 * tips_correct / len(year_rows), 2),
                    "bits_per_game": round(bits_total / len(year_rows), 4),
                    "total_bits": round(bits_total, 4),
                }
            )

            prior_rows.extend(year_rows)
            model_bits_total += bits_total
            model_tip_correct += tips_correct
            model_games += len(year_rows)

        summaries.append(
            {
                "model_name": model_name,
                "year": "ALL",
                "num_games": len(rows),
                "mae_margin": round(_mae(rows), 4),
                "tip_pct": round(100.0 * model_tip_correct / model_games, 2),
                "bits_per_game": round(model_bits_total / model_games, 4),
                "total_bits": round(model_bits_total, 4),
            }
        )

    return summaries


def write_prediction_rows(path: str, rows: List[PredictionRow]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "match_id",
                "year",
                "round_label",
                "home_team",
                "away_team",
                "venue",
                "actual_margin",
                "predicted_margin",
                "abs_error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": row.model_name,
                    "match_id": row.match_id,
                    "year": row.year,
                    "round_label": row.round_label,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "venue": row.venue,
                    "actual_margin": round(row.actual_margin, 4),
                    "predicted_margin": round(row.predicted_margin, 4),
                    "abs_error": round(row.abs_error, 4),
                }
            )


def write_summary_rows(path: str, rows: List[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "year",
                "num_games",
                "mae_margin",
                "tip_pct",
                "bits_per_game",
                "total_bits",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
