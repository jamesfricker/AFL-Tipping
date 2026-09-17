"""Market-free margin model with official preseason scoring-shot results."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean

from .data import (
    Fixture,
    MatchRow,
    _aware,
    _csv_rows,
    _unique,
    canonical_team_name,
    parse_timestamp,
    validate_matches,
)
from .sequential_margin import PredictionRow
from .venues import get_venue_meta, normalize_venue_name


MODEL_NAME = "preseason_structural_challenger"

MOSHBODS_CONFIG = {
    "score_window_years": 5.0,
    "adjusted_shot_weight": 0.3596021186362973,
    "carryover": 0.5722986956547784,
    "alpha_early": 0.1891793477128674,
    "alpha_late": 0.055957029668312605,
    "alpha_decay": 5.747455174647012,
    "alpha_finals": 0.06906146017264701,
    "vpv_window_years": 7.5,
    "vpv_mean_reg": 0.14528685629948412,
    "vpv_reference_games": 15.906159219593142,
    "away_region_default": -0.37730376771286367,
    "vpv_weight": 1.4964794434159354,
}

STRUCTURE_CONFIG = {
    "rating_rate": 0.0791659346299017,
    "carryover": 0.6344105485750351,
    "actual_score_weight": 0.23970754502833902,
    "accuracy_prior_shots": 56.47814359356514,
    "home_advantage": 2.548440727215546,
    "travel_weight": -0.18211392136911064,
    "experience_weight": 0.002296722296663667,
    "experience_prior": 6.719205034263909,
    "venue_performance_rate": 0.29675263170077176,
    "venue_performance_weight": 0.20674317060433206,
}

PRESEASON_SHOT_VALUE = 4.85
PRESEASON_CAP = 40.0
PRESEASON_WEIGHT = 0.2
PRESEASON_LAST_ROUND = 8
TAIL_THRESHOLD = 40.0
TAIL_SLOPE = 0.3

TEAM_REGION = {
    "Adelaide": "adelaide",
    "Brisbane Lions": "queensland",
    "Carlton": "melbourne",
    "Collingwood": "melbourne",
    "Essendon": "melbourne",
    "Fremantle": "perth",
    "Geelong": "geelong",
    "Gold Coast": "queensland",
    "Greater Western Sydney": "sydney",
    "Hawthorn": "melbourne",
    "Melbourne": "melbourne",
    "Kangaroos": "melbourne",
    "Port Adelaide": "adelaide",
    "Richmond": "melbourne",
    "St Kilda": "melbourne",
    "Sydney": "sydney",
    "West Coast": "perth",
    "Western Bulldogs": "melbourne",
}

TEAM_BASES = {
    "Adelaide": (-34.894, 138.52),
    "Brisbane Lions": (-27.485, 153.0381),
    "Carlton": (-37.8164, 144.9475),
    "Collingwood": (-37.8199, 144.9834),
    "Essendon": (-37.8164, 144.9475),
    "Fremantle": (-31.9431, 115.8329),
    "Geelong": (-38.1561, 144.3548),
    "Gold Coast": (-28.0064, 153.3669),
    "Greater Western Sydney": (-33.8474, 151.0674),
    "Hawthorn": (-37.8199, 144.9834),
    "Kangaroos": (-37.8164, 144.9475),
    "Melbourne": (-37.8199, 144.9834),
    "Port Adelaide": (-34.894, 138.52),
    "Richmond": (-37.8199, 144.9834),
    "St Kilda": (-37.8164, 144.9475),
    "Sydney": (-33.8917, 151.224),
    "West Coast": (-31.9431, 115.8329),
    "Western Bulldogs": (-37.8164, 144.9475),
}


@dataclass(frozen=True)
class PreseasonResult:
    provider_id: str
    year: int
    kickoff: datetime
    home_team: str
    away_team: str
    home_goals: int
    home_behinds: int
    away_goals: int
    away_behinds: int

    @property
    def shot_margin(self) -> float:
        home = self.home_goals + self.home_behinds
        away = self.away_goals + self.away_behinds
        return PRESEASON_SHOT_VALUE * (home - away)


def load_preseason_results_csv(path: str) -> list[PreseasonResult]:
    required = (
        "provider_id",
        "year",
        "kickoff",
        "home_team",
        "home_goals",
        "home_behinds",
        "away_team",
        "away_goals",
        "away_behinds",
    )
    rows = []
    for line, row in _csv_rows(path, required):
        try:
            result = PreseasonResult(
                row["provider_id"].strip(),
                int(row["year"]),
                parse_timestamp(row["kickoff"]),
                canonical_team_name(row["home_team"]),
                canonical_team_name(row["away_team"]),
                int(row["home_goals"]),
                int(row["home_behinds"]),
                int(row["away_goals"]),
                int(row["away_behinds"]),
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
        if result.kickoff.year != result.year:
            raise ValueError(f"{path}:{line}: year differs from kickoff")
        if result.home_team == result.away_team:
            raise ValueError(f"{path}:{line}: teams must differ")
        if min(
            result.home_goals,
            result.home_behinds,
            result.away_goals,
            result.away_behinds,
        ) < 0:
            raise ValueError(f"{path}:{line}: scores must be non-negative")
        rows.append(result)
    ids = [row.provider_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate preseason provider ID")
    return sorted(rows, key=lambda row: (row.kickoff, row.provider_id))


def _round_number(label: str) -> int:
    value = label.lower()
    if "opening" in value:
        return 0
    found = re.search(r"\d+", value)
    return int(found.group()) if found else 99


def _is_final(label: str) -> bool:
    value = label.lower()
    return "final" in value and not re.search(r"round\s+\d+", value)


def _venue_region(name: str) -> str:
    key = normalize_venue_name(name)
    if key in {
        "kardiniapark",
        "gmhba",
        "gmhbastadium",
        "simondsstadium",
        "skilledstadium",
    }:
        return "geelong"
    meta = get_venue_meta(name)
    if meta is None:
        return "unknown"
    return {
        "Australia/Melbourne": "melbourne",
        "Australia/Adelaide": "adelaide",
        "Australia/Brisbane": "queensland",
        "Australia/Perth": "perth",
        "Australia/Sydney": "sydney",
    }.get(meta.timezone, "unknown")


def _haversine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    value = math.sin((lat2 - lat1) / 2) ** 2
    value += math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(value))


def _moshbods_predictions(
    matches: list[MatchRow], fixtures: list[Fixture] | None = None
) -> dict[str, float]:
    config = MOSHBODS_CONFIG
    offence = defaultdict(float)
    defence = defaultdict(float)
    league = deque()
    league_totals = [0.0, 0.0, 0.0, 0.0]
    venue_excess = defaultdict(deque)
    forecasts = {}
    predictions = {}
    current_year = None
    events = []
    for match in matches:
        events.append((match.fixture.kickoff, 1, match.match_id, match))
        events.append((match.available_at, 0, match.match_id, match))
    for fixture in fixtures or []:
        events.append((fixture.kickoff, 2, fixture.match_id, fixture))
    events.sort(key=lambda event: event[:3])

    def begin_year(year):
        nonlocal current_year
        if current_year is None:
            current_year = year
            return
        if year <= current_year:
            return
        elapsed = year - current_year
        active = set(offence) | set(defence)
        if active:
            offset = sum(offence[t] + defence[t] for t in active) / (2 * len(active))
            for team in active:
                offence[team] = (offence[team] - offset) * config["carryover"] ** elapsed
                defence[team] = (defence[team] - offset) * config["carryover"] ** elapsed
        current_year = year

    def league_stats(at):
        score_sum, score_sq, shot_sum, shot_sq = league_totals
        oldest = at - timedelta(days=365.25 * config["score_window_years"])
        while league and league[0][0] < oldest:
            _, score, shots = league.popleft()
            score_sum -= score
            score_sq -= score * score
            shot_sum -= shots
            shot_sq -= shots * shots
        league_totals[:] = score_sum, score_sq, shot_sum, shot_sq
        count = len(league)
        score_mean = score_sum / count if count else 76.0
        shot_mean = shot_sum / count if count else 26.0
        score_var = score_sq / count - score_mean**2 if count > 1 else 900.0
        shot_var = shot_sq / count - shot_mean**2 if count > 1 else 64.0
        return score_mean, math.sqrt(max(score_var, 1.0)), shot_mean, math.sqrt(max(shot_var, 1.0))

    def vpv(team, venue, at):
        history = venue_excess[(team, normalize_venue_name(venue))]
        oldest = at - timedelta(days=365.25 * config["vpv_window_years"])
        while history and history[0][0] < oldest:
            history.popleft()
        count = len(history)
        sample_weight = min(count, config["vpv_reference_games"]) / config["vpv_reference_games"]
        average = config["vpv_mean_reg"] * fmean(row[1] for row in history) if history else 0.0
        default = 0.0 if TEAM_REGION.get(team) == _venue_region(venue) else config["away_region_default"]
        return config["vpv_weight"] * (sample_weight * average + (1 - sample_weight) * default)

    for at, kind, _, item in events:
        fixture = item.fixture if isinstance(item, MatchRow) else item
        begin_year(fixture.year)
        home, away = fixture.home_team, fixture.away_team
        net_vpv = vpv(home, fixture.venue, at) - vpv(away, fixture.venue, at)
        score_mean, score_sd, shot_mean, shot_sd = league_stats(at)
        point_margin = (offence[home] - defence[away] - offence[away] + defence[home] + net_vpv) * score_sd
        if fixture.year == 2020:
            point_margin /= 1.25
        if kind in (1, 2):
            predictions[fixture.match_id] = point_margin
            if kind == 2:
                continue
            forecasts[fixture.match_id] = (
                offence[home], defence[home], offence[away], defence[away],
                net_vpv, score_mean, score_sd, shot_mean, shot_sd,
            )
            continue
        match = item
        off_h, def_h, off_a, def_a, net_vpv, score_mean, score_sd, shot_mean, shot_sd = forecasts[match.match_id]
        duration = 1.25 if fixture.year == 2020 else 1.0
        home_score, away_score = match.home_score * duration, match.away_score * duration
        home_shots = (match.home_scoring_shots or match.home_goals + match.home_behinds) * duration
        away_shots = (match.away_scoring_shots or match.away_goals + match.away_behinds) * duration
        weight = config["adjusted_shot_weight"]
        home_adj = weight * ((home_shots - shot_mean) / shot_sd) + (1 - weight) * ((home_score - score_mean) / score_sd)
        away_adj = weight * ((away_shots - shot_mean) / shot_sd) + (1 - weight) * ((away_score - score_mean) / score_sd)
        if _is_final(fixture.round_label):
            rate = config["alpha_finals"]
        else:
            progress = min(_round_number(fixture.round_label), 24) / 24
            rate = config["alpha_late"] + (config["alpha_early"] - config["alpha_late"]) * math.exp(-config["alpha_decay"] * progress)
        offence[home] = off_h + rate * (home_adj - (off_h - def_a + net_vpv / 2))
        offence[away] = off_a + rate * (away_adj - (off_a - def_h - net_vpv / 2))
        defence[home] = def_h + rate * (-away_adj - (def_h - off_a + net_vpv / 2))
        defence[away] = def_a + rate * (-home_adj - (def_a - off_h - net_vpv / 2))
        rating_margin = off_h + def_h - off_a - def_a
        home_excess = (home_adj - away_adj) - rating_margin
        venue_key = normalize_venue_name(fixture.venue)
        venue_excess[(home, venue_key)].append((fixture.kickoff, home_excess))
        venue_excess[(away, venue_key)].append((fixture.kickoff, -home_excess))
        league.append((fixture.kickoff, home_score, home_shots))
        league.append((fixture.kickoff, away_score, away_shots))
        league_totals[0] += home_score + away_score
        league_totals[1] += home_score**2 + away_score**2
        league_totals[2] += home_shots + away_shots
        league_totals[3] += home_shots**2 + away_shots**2
    return predictions


def _structure_predictions(
    matches: list[MatchRow], fixtures: list[Fixture] | None = None
) -> dict[str, float]:
    config = STRUCTURE_CONFIG
    attack = defaultdict(float)
    defense = defaultdict(float)
    venue_games = defaultdict(deque)
    venue_performance = defaultdict(float)
    league_scores = deque(maxlen=500)
    team_goals = defaultdict(float)
    team_shots = defaultdict(float)
    league_goals = 0.0
    league_shots = 0.0
    current_year = None
    forecasts = {}
    predictions = {}
    events = []
    for match in matches:
        events.append((match.fixture.kickoff, 1, match.match_id, match))
        events.append((match.available_at, 0, match.match_id, match))
    for fixture in fixtures or []:
        events.append((fixture.kickoff, 2, fixture.match_id, fixture))
    events.sort(key=lambda event: event[:3])

    def begin_year(year):
        nonlocal current_year, team_goals, team_shots, league_goals, league_shots
        if current_year is None:
            current_year = year
            return
        if year <= current_year:
            return
        elapsed = year - current_year
        for ratings in (attack, defense):
            mean = fmean(ratings.values()) if ratings else 0.0
            for team in ratings:
                ratings[team] = (ratings[team] - mean) * config["carryover"] ** elapsed
        team_goals, team_shots = defaultdict(float), defaultdict(float)
        league_goals = league_shots = 0.0
        current_year = year

    for _, kind, _, item in events:
        fixture = item.fixture if isinstance(item, MatchRow) else item
        begin_year(fixture.year)
        home, away = fixture.home_team, fixture.away_team
        venue_key = normalize_venue_name(fixture.venue)
        venue = get_venue_meta(fixture.venue)
        target = (venue.latitude, venue.longitude) if venue else None
        home_distance = _haversine(TEAM_BASES.get(home), target)
        away_distance = _haversine(TEAM_BASES.get(away), target)
        cutoff_year = fixture.year - 4
        for team in (home, away):
            history = venue_games[(team, venue_key)]
            while history and history[0] < cutoff_year:
                history.popleft()
        travel = config["travel_weight"] * (math.sqrt(home_distance) - math.sqrt(away_distance))
        experience = config["experience_weight"] * (
            math.sqrt(len(venue_games[(home, venue_key)]) + config["experience_prior"])
            - math.sqrt(len(venue_games[(away, venue_key)]) + config["experience_prior"])
        )
        base_venue = config["home_advantage"] + travel + experience
        venue_form = config["venue_performance_weight"] * (
            venue_performance[(home, venue_key)] - venue_performance[(away, venue_key)]
        )
        team_margin = attack[home] + defense[home] - attack[away] - defense[away]
        prediction = team_margin + base_venue + venue_form
        average_score = fmean(league_scores) if league_scores else 76.0
        expected_home = average_score + attack[home] - defense[away]
        expected_away = average_score + attack[away] - defense[home]
        if kind in (1, 2):
            predictions[fixture.match_id] = prediction
            if kind == 2:
                continue
            forecasts[fixture.match_id] = prediction, team_margin, base_venue, expected_home, expected_away
            continue
        match = item
        _, team_margin, base_venue, expected_home, expected_away = forecasts[match.match_id]
        league_accuracy = league_goals / league_shots if league_shots else 0.5

        def adjusted(team, score, goals, behinds):
            shots = goals + behinds
            prior = config["accuracy_prior_shots"]
            accuracy = (team_goals[team] + prior * league_accuracy) / (team_shots[team] + prior)
            neutral = shots * (1.0 + 5.0 * accuracy)
            return config["actual_score_weight"] * score + (1 - config["actual_score_weight"]) * neutral

        adjusted_home = adjusted(home, match.home_score, match.home_goals, match.home_behinds)
        adjusted_away = adjusted(away, match.away_score, match.away_goals, match.away_behinds)
        home_residual, away_residual = adjusted_home - expected_home, adjusted_away - expected_away
        attack[home] += config["rating_rate"] * home_residual
        defense[away] -= config["rating_rate"] * home_residual
        attack[away] += config["rating_rate"] * away_residual
        defense[home] -= config["rating_rate"] * away_residual
        venue_residual = (adjusted_home - adjusted_away) - team_margin - base_venue
        rate = config["venue_performance_rate"]
        for team, target_value in ((home, venue_residual / 2), (away, -venue_residual / 2)):
            key = (team, venue_key)
            venue_performance[key] += rate * (target_value - venue_performance[key])
            venue_games[key].append(fixture.year)
        for team, score, goals, behinds in (
            (home, match.home_score, match.home_goals, match.home_behinds),
            (away, match.away_score, match.away_goals, match.away_behinds),
        ):
            shots = goals + behinds
            team_goals[team] += goals
            team_shots[team] += shots
            league_goals += goals
            league_shots += shots
            league_scores.append(score)
    return predictions


def _preseason_signal(
    preseason: list[PreseasonResult], year: int, team: str, cutoff: datetime
) -> float:
    values = []
    for match in preseason:
        if match.year != year or match.kickoff >= cutoff:
            continue
        value = max(-PRESEASON_CAP, min(PRESEASON_CAP, match.shot_margin))
        if match.home_team == team:
            values.append(value)
        elif match.away_team == team:
            values.append(-value)
    return fmean(values) if values else 0.0


def _final_prediction(
    fixture: Fixture,
    cutoff: datetime,
    preseason: list[PreseasonResult],
    moshbods: dict[str, float],
    structure: dict[str, float],
) -> float:
    prediction = 0.5 * (
        moshbods[fixture.match_id] + structure[fixture.match_id]
    )
    if _round_number(fixture.round_label) <= PRESEASON_LAST_ROUND:
        home = _preseason_signal(
            preseason, fixture.year, fixture.home_team, cutoff
        )
        away = _preseason_signal(
            preseason, fixture.year, fixture.away_team, cutoff
        )
        prediction += PRESEASON_WEIGHT * (home - away)
    return math.copysign(
        abs(prediction)
        + TAIL_SLOPE * max(0.0, abs(prediction) - TAIL_THRESHOLD),
        prediction,
    )


def replay_preseason_challenger(
    matches: list[MatchRow],
    preseason: list[PreseasonResult],
    *,
    first_prediction_year: int | None = None,
) -> list[PredictionRow]:
    """Replay the fixed challenger with information available before kickoff."""
    validate_matches(matches)
    if not matches:
        return []
    moshbods = _moshbods_predictions(matches)
    structure = _structure_predictions(matches)
    first_year = min(match.year for match in matches)
    target_year = first_prediction_year if first_prediction_year is not None else first_year + 3
    rows = []
    for match in sorted(matches, key=lambda row: (row.fixture.kickoff, row.match_id)):
        if match.year < target_year:
            continue
        prediction = _final_prediction(
            match.fixture,
            match.fixture.kickoff,
            preseason,
            moshbods,
            structure,
        )
        rows.append(
            PredictionRow(
                match.match_id,
                match.year,
                match.round_label,
                match.home_team,
                match.away_team,
                match.venue,
                match.fixture.kickoff,
                match.fixture.kickoff,
                MODEL_NAME,
                prediction,
                "not_used",
                None,
                None,
                0.0,
                0,
                match.fixture.kickoff,
                actual_margin=match.actual_margin,
                abs_error=abs(match.actual_margin - prediction),
                result_available_at=match.available_at,
                result_timing_assumption=match.timing_assumption,
            )
        )
    return rows


def predict_preseason_challenger(
    matches: list[MatchRow],
    fixtures: list[Fixture],
    preseason: list[PreseasonResult],
    as_of: datetime,
) -> list[PredictionRow]:
    """Predict future fixtures with results that were available at ``as_of``."""
    _aware(as_of, "as_of")
    validate_matches(matches)
    _unique(fixtures, lambda row: row.match_id, "fixture ID")
    if any(fixture.kickoff <= as_of for fixture in fixtures):
        raise ValueError("Live fixtures must start after as_of")
    history = [match for match in matches if match.available_at <= as_of]
    moshbods = _moshbods_predictions(history, fixtures)
    structure = _structure_predictions(history, fixtures)
    rows = []
    for fixture in fixtures:
        prediction = _final_prediction(
            fixture, as_of, preseason, moshbods, structure
        )
        rows.append(
            PredictionRow(
                fixture.match_id,
                fixture.year,
                fixture.round_label,
                fixture.home_team,
                fixture.away_team,
                fixture.venue,
                fixture.kickoff,
                as_of,
                MODEL_NAME,
                prediction,
                "not_used",
                None,
                None,
                0.0,
                0,
                as_of,
            )
        )
    return rows
