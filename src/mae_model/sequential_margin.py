import csv
import math
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


TEAM_ALIASES = {
    "South Melbourne": "Sydney",
    "Footscray": "Western Bulldogs",
    "North Melbourne": "Kangaroos",
    "North Melbourne Kangaroos": "Kangaroos",
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
    actual_margin: float
    predicted_margin: float
    abs_error: float
    model_name: str


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
        self.current_year: Optional[int] = None

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
            # Prediction-time weighting must not depend on same-match in-game stats.
            weight = 1.0
            weighted_total += self.player_rating[player_name] * weight
            total_weight += weight

        if total_weight <= 0:
            return 0.0
        return self.lineup_scale * (weighted_total / total_weight)

    def _apply_season_transition(self):
        teams = list(self.team_offense.keys())
        if not teams:
            return
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

        for idx, player in enumerate(away_players):
            weight = away_weights[idx] / away_total
            name = player["player_name"]
            self.player_rating[name] -= self.player_k * margin_residual * weight
            self.player_games[name] += 1

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


def walk_forward_predictions(
    matches: List[MatchRow],
    lineups: Dict[Tuple[str, str], List[dict]],
    min_train_years: int,
) -> List[PredictionRow]:
    if not matches:
        return []

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
    }
    hybrid_model = BenchmarkShotVolumeModel(
        conversion_k=0.045,
        home_adv_k=0.002,
        season_carryover=0.78,
        conversion_window_games=800,
    )

    predictions: List[PredictionRow] = []

    for match in matches:
        for model_name, model in models.items():
            if model_name == "scoring_shots":
                expected_home, expected_away, predicted_margin = model.step(match)
            else:
                expected_home, expected_away, predicted_margin = model.step(match, lineups)
            if match.year >= scoring_year:
                predictions.append(
                    PredictionRow(
                        match_id=match.match_id,
                        year=match.year,
                        round_label=match.round_label,
                        home_team=match.home_team,
                        away_team=match.away_team,
                        actual_margin=match.actual_margin,
                        predicted_margin=predicted_margin,
                        abs_error=abs(match.actual_margin - predicted_margin),
                        model_name=model_name,
                    )
                )

        _, _, hybrid_margin = hybrid_model.step(match)

        if match.year >= scoring_year:
            predictions.append(
                PredictionRow(
                    match_id=match.match_id,
                    year=match.year,
                    round_label=match.round_label,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    actual_margin=match.actual_margin,
                    predicted_margin=hybrid_margin,
                    abs_error=abs(match.actual_margin - hybrid_margin),
                    model_name="team_residual_lineup",
                )
            )

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
