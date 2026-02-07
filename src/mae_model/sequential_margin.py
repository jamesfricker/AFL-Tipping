import csv
import math
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


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
            "Home Odds": "home_odds",
            "Away Odds": "away_odds",
            "Date": "date",
        }
    )
    required = {"date", "home_team", "away_team", "home_line_close", "home_odds", "away_odds"}
    if not required.issubset(set(frame.columns)):
        return market_rows

    frame = frame[list(required)].dropna(subset=["date", "home_team", "away_team"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["date"])

    for row in frame.to_dict(orient="records"):
        home_team = canonical_team_name(str(row["home_team"]).strip())
        away_team = canonical_team_name(str(row["away_team"]).strip())
        key = (row["date"], home_team, away_team)
        home_line_close = row["home_line_close"]
        home_odds = row["home_odds"]
        away_odds = row["away_odds"]
        market_rows[key] = {
            "home_line_close": float(home_line_close) if home_line_close == home_line_close else None,
            "home_odds": float(home_odds) if home_odds == home_odds else None,
            "away_odds": float(away_odds) if away_odds == away_odds else None,
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


def _market_residual_features(row: dict) -> np.ndarray:
    internal_margin = row["internal_margin"]
    anchor_margin = row["anchor_margin"]
    disagreement = internal_margin - anchor_margin
    implied_prob = row.get("implied_home_prob")
    market_sigma = row.get("market_sigma", 30.0)
    implied_edge = 0.0 if implied_prob is None else (implied_prob - 0.5) * 2.0
    finals_flag = 1.0 if is_finals_round(row["round_label"]) else 0.0
    return np.array(
        [
            1.0,
            anchor_margin / 40.0,
            internal_margin / 30.0,
            disagreement / 25.0,
            abs(disagreement) / 25.0,
            finals_flag,
            implied_edge,
            (market_sigma - 30.0) / 20.0,
        ],
        dtype=float,
    )


def _fit_ridge_from_rows(rows: List[dict], alpha: float) -> np.ndarray:
    if not rows:
        return np.zeros(8, dtype=float)
    x = np.array([_market_residual_features(row) for row in rows], dtype=float)
    y = np.array(
        [max(-25.0, min(25.0, row["actual_margin"] - row["anchor_margin"])) for row in rows],
        dtype=float,
    )
    ridge = alpha * np.eye(x.shape[1], dtype=float)
    ridge[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    return np.clip(coef, -15.0, 15.0)


def _apply_market_residual_corrector(row: dict, config: dict) -> float:
    raw_correction = float(_market_residual_features(row) @ config["coef"])
    sigma = float(row.get("market_sigma", 30.0))
    sigma_norm = max(0.0, (sigma - 30.0) / 20.0)
    damp = 1.0 / (1.0 + config["sigma_k"] * sigma_norm)
    correction = damp * raw_correction
    cap = config["cap"]
    correction = max(-cap, min(cap, correction))
    return row["anchor_margin"] + correction


def _fit_market_residual_corrector(history_rows: List[dict]) -> dict:
    default = {"coef": np.zeros(8, dtype=float), "cap": 5.0, "sigma_k": 0.6}
    if len(history_rows) < 140:
        return default

    years = sorted({row["year"] for row in history_rows})
    if len(years) >= 2:
        validation_year = years[-1]
        train_rows = [row for row in history_rows if row["year"] < validation_year]
        validation_rows = [row for row in history_rows if row["year"] == validation_year]
        if len(train_rows) < 120 or len(validation_rows) < 60:
            train_rows = history_rows
            validation_rows = history_rows
    else:
        train_rows = history_rows
        validation_rows = history_rows

    alphas = (1.0, 3.0, 10.0, 30.0)
    caps = (3.0, 4.0, 5.0, 6.0, 8.0)
    sigma_ks = (0.0, 0.3, 0.6, 1.0)

    best_cfg = default
    best_alpha = 3.0
    best_mae = float("inf")
    for alpha in alphas:
        coef = _fit_ridge_from_rows(train_rows, alpha)
        for cap in caps:
            for sigma_k in sigma_ks:
                cfg = {"coef": coef, "cap": cap, "sigma_k": sigma_k}
                abs_errors = 0.0
                for row in validation_rows:
                    pred = _apply_market_residual_corrector(row, cfg)
                    abs_errors += abs(row["actual_margin"] - pred)
                mae = abs_errors / len(validation_rows)
                if mae < best_mae:
                    best_mae = mae
                    best_cfg = cfg
                    best_alpha = alpha

    final_coef = _fit_ridge_from_rows(history_rows, alpha=best_alpha)
    return {"coef": final_coef, "cap": best_cfg["cap"], "sigma_k": best_cfg["sigma_k"]}


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
    market_fit_history: List[dict] = []
    market_weight_by_year: Dict[int, float] = {}
    market_residual_config_by_year: Dict[int, dict] = {}
    market_fit_recent_years = 5
    market_residual_recent_years = 3

    for match in matches:
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

            market_row = market_data.get(context_key)
            market_margin: Optional[float] = None
            implied_home_prob: Optional[float] = None
            market_sigma: float = 30.0
            if market_row is not None and market_row.get("home_line_close") is not None:
                # Betting line is typically home handicap; negative implies home favourite.
                # Convert to predicted home margin orientation.
                market_margin = -float(market_row["home_line_close"])
                implied_home_prob = _market_implied_home_probability(
                    market_row.get("home_odds"),
                    market_row.get("away_odds"),
                )
                market_sigma = _market_sigma_from_spread_and_probability(market_margin, implied_home_prob)

            if match.year not in market_weight_by_year:
                prior_rows = [row for row in market_fit_history if row["year"] < match.year]
                fit_rows = _recent_history_rows(prior_rows, match.year, market_fit_recent_years)
                market_weight_by_year[match.year] = _fit_market_line_weight(fit_rows)
                residual_fit_rows = _recent_history_rows(prior_rows, match.year, market_residual_recent_years)
                market_residual_config_by_year[match.year] = _fit_market_residual_corrector(residual_fit_rows)

            base_margin = match_margins["scoring_shots"]
            if market_margin is None:
                market_blend_margin = base_margin
            else:
                weight = market_weight_by_year[match.year]
                market_blend_margin = weight * market_margin + (1.0 - weight) * base_margin

            market_only_margin = market_margin if market_margin is not None else base_margin
            if market_margin is None:
                market_residual_margin = market_blend_margin
            else:
                residual_row = {
                    "year": match.year,
                    "round_label": match.round_label,
                    "anchor_margin": market_blend_margin,
                    "internal_margin": match_margins["team_residual_lineup"],
                    "implied_home_prob": implied_home_prob,
                    "market_sigma": market_sigma,
                }
                cfg = market_residual_config_by_year[match.year]
                market_residual_margin = _apply_market_residual_corrector(residual_row, cfg)

            predictions.append(
                PredictionRow(
                    match_id=match.match_id,
                    year=match.year,
                    round_label=match.round_label,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    venue=match.venue,
                    actual_margin=match.actual_margin,
                    predicted_margin=market_blend_margin,
                    abs_error=abs(match.actual_margin - market_blend_margin),
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

            if market_margin is not None:
                market_fit_history.append(
                    {
                        "year": match.year,
                        "round_label": match.round_label,
                        "actual_margin": match.actual_margin,
                        "base_margin": base_margin,
                        "market_margin": market_margin,
                        "anchor_margin": market_blend_margin,
                        "internal_margin": match_margins["team_residual_lineup"],
                        "implied_home_prob": implied_home_prob,
                        "market_sigma": market_sigma,
                    }
                )

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
