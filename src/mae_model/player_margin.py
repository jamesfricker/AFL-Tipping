import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime
from statistics import fmean, median
from typing import Literal, NewType

from .data import (
    Fixture,
    MatchRow,
    _aware,
    _csv_rows,
    _number,
    canonical_team_name,
    parse_timestamp,
    validate_matches,
)
from .sequential_margin import PredictionRow

PlayerId = NewType("PlayerId", str)
PlayerSignal = Literal["rating", "form", "missing_leader", "rating_form"]
PlayerMeasurement = Literal[
    "outcome_fantasy", "official_points", "official_points_per_time"
]


@dataclass(frozen=True)
class PlayerStats:
    kicks: float = 0.0
    handballs: float = 0.0
    marks: float = 0.0
    goals: float = 0.0
    behinds: float = 0.0
    hit_outs: float = 0.0
    tackles: float = 0.0
    clearances: float = 0.0
    contested_possessions: float = 0.0
    goal_assists: float = 0.0
    clangers: float = 0.0


@dataclass(frozen=True)
class PlayerMatch:
    match_id: str
    team: str
    player_id: PlayerId
    statistics_available_at: datetime
    percent_played: float
    stats: PlayerStats
    official_rating_points: float | None = None


@dataclass(frozen=True)
class LineupSnapshot:
    match_id: str
    team: str
    player_ids: tuple[PlayerId, ...]
    observed_at: datetime
    source: Literal["timed_selection", "assumed_final_at_kickoff"]


@dataclass(frozen=True)
class PlayerState:
    impact_rating: float = 0.0
    experience: float = 0.0
    career_performance: float = 0.0
    recent_performance: float = 0.0
    official_games: int = 0
    official_average: float = 0.0
    official_recent: float = 0.0


@dataclass(frozen=True)
class TeamLineupForecast:
    rating_change: float
    form_change: float
    coverage: float
    missing_leader: PlayerId | None
    missing_leader_gap: float


@dataclass(frozen=True)
class LineupForecast:
    home: TeamLineupForecast
    away: TeamLineupForecast
    rating_change_difference: float
    form_change_difference: float


@dataclass(frozen=True)
class PlayerDiagnostic:
    match_id: str
    cutoff: datetime
    status: Literal[
        "adjusted",
        "no_lineup",
        "low_coverage",
        "stable_lineup",
        "insufficient_history",
    ]
    correction: float
    lineup: LineupForecast | None


@dataclass(frozen=True)
class PlayerModelConfig:
    signal: PlayerSignal = "rating_form"
    measurement: PlayerMeasurement = "outcome_fantasy"
    control_model_name: str = "market_scoring_blend"
    reference_lineups: int = 4
    minimum_reference_lineups: int = 3
    minimum_coverage: float = 0.80
    minimum_player_games: float = 5.0
    rating_rate: float = 24.0
    rating_prior_games: float = 6.0
    form_rate: float = 0.25
    official_average_rate: float = 0.05
    material_change: float = 0.75
    correction_cap: float = 4.0
    rating_weight: float = 1.0
    form_weight: float = 0.20

    def __post_init__(self):
        if self.signal not in ("rating", "form", "missing_leader", "rating_form"):
            raise ValueError(f"Unknown player signal: {self.signal}")
        if self.measurement not in (
            "outcome_fantasy",
            "official_points",
            "official_points_per_time",
        ):
            raise ValueError(f"Unknown player measurement: {self.measurement}")
        if not 1 <= self.minimum_reference_lineups <= self.reference_lineups:
            raise ValueError("Player reference counts must be positive and ordered")
        if (
            not 0 <= self.minimum_coverage <= 1
            or not 0.05 < self.form_rate <= 1
            or not 0 < self.official_average_rate <= 1
        ):
            raise ValueError("Invalid player coverage or form rate")
        for name in (
            "minimum_player_games",
            "rating_rate",
            "rating_prior_games",
            "material_change",
            "correction_cap",
            "rating_weight",
            "form_weight",
        ):
            _number(getattr(self, name), name, minimum=0)
        if self.rating_prior_games == 0:
            raise ValueError("rating_prior_games must be positive")


def _deduplicate(items, key, description):
    seen = {}
    for item in items:
        identity = key(item)
        if identity in seen and seen[identity] != item:
            raise ValueError(f"Changed source data for {description}: {identity}")
        seen[identity] = item
    return list(seen.values())


def load_player_matches_csv(path: str, matches: list[MatchRow]) -> list[PlayerMatch]:
    history = {match.match_id: match for match in matches}
    appearances = []
    required = ("match_id", "team_name", "player_ref", "percent_played")
    for line, row in _csv_rows(path, required):
        try:
            match_id = row["match_id"].strip()
            if match_id not in history:
                raise ValueError(f"Unknown player match ID: {match_id}")
            match = history[match_id]
            team = canonical_team_name(row["team_name"])
            if team not in (match.home_team, match.away_team):
                raise ValueError(f"Player team is not in match: {team}")
            player_id = row["player_ref"].strip()
            if not player_id:
                raise ValueError("Player reference is empty")
            percent = _number(row["percent_played"] or 0, "percent_played", minimum=0)
            if percent > 100:
                raise ValueError("percent_played must not exceed 100")
            stamp = row.get("statistics_available_at", "").strip()
            available = parse_timestamp(stamp) if stamp else match.available_at
            if available < match.available_at:
                raise ValueError("Player statistics cannot precede result availability")
            stats = PlayerStats(
                **{
                    name: _number(row.get(name) or 0, name, minimum=0)
                    for name in PlayerStats.__dataclass_fields__
                }
            )
            official_value = row.get("official_rating_points", "").strip()
            official_rating = (
                _number(official_value, "official_rating_points")
                if official_value
                else None
            )
            appearances.append(
                PlayerMatch(
                    match_id,
                    team,
                    PlayerId(player_id),
                    available,
                    percent,
                    stats,
                    official_rating,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    return _deduplicate(
        appearances, lambda row: (row.match_id, row.player_id), "player appearance"
    )


def load_lineup_snapshots_csv(
    path: str, fixtures: list[Fixture]
) -> list[LineupSnapshot]:
    known = {fixture.match_id: fixture for fixture in fixtures}
    groups = defaultdict(list)
    for line, row in _csv_rows(
        path, ("match_id", "team_name", "player_ref", "observed_at")
    ):
        try:
            match_id = row["match_id"].strip()
            if match_id not in known:
                raise ValueError(f"Unknown lineup match ID: {match_id}")
            fixture = known[match_id]
            team = canonical_team_name(row["team_name"])
            if team not in (fixture.home_team, fixture.away_team):
                raise ValueError(f"Lineup team is not in fixture: {team}")
            player_id = PlayerId(row["player_ref"].strip())
            if not player_id:
                raise ValueError("Player reference is empty")
            stamp = parse_timestamp(row["observed_at"])
            if stamp > fixture.kickoff:
                raise ValueError("Lineup observation must not be after kickoff")
            key = (match_id, team, stamp)
            if player_id in groups[key]:
                raise ValueError(f"Duplicate lineup player: {player_id}")
            groups[key].append(player_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line}: {exc}") from exc
    return [
        LineupSnapshot(match_id, team, tuple(sorted(players)), stamp, "timed_selection")
        for (match_id, team, stamp), players in groups.items()
    ]


def derive_historical_lineups(
    appearances: list[PlayerMatch], matches: list[MatchRow]
) -> list[LineupSnapshot]:
    history = {match.match_id: match for match in matches}
    groups = defaultdict(set)
    for appearance in appearances:
        groups[(appearance.match_id, appearance.team)].add(appearance.player_id)
    return [
        LineupSnapshot(
            match_id,
            team,
            tuple(sorted(players)),
            history[match_id].fixture.kickoff,
            "assumed_final_at_kickoff",
        )
        for (match_id, team), players in groups.items()
    ]


def _rating(state, config):
    if config.measurement != "outcome_fantasy":
        return (
            state.official_average
            * state.official_games
            / (state.official_games + config.rating_prior_games)
        )
    return (
        state.impact_rating
        * state.experience
        / (state.experience + config.rating_prior_games)
    )


def _form(state, config):
    if config.measurement != "outcome_fantasy":
        return state.official_recent - state.official_average
    return state.recent_performance - state.career_performance


def _experience(state, config):
    return (
        state.official_games
        if config.measurement != "outcome_fantasy"
        else state.experience
    )


def _team_forecast(selected, references, states, config):
    counts = Counter(player for lineup in references for player in lineup)
    regular = sorted(
        player for player, count in counts.items() if count * 2 >= len(references)
    )

    def state(player):
        return states.get(player, PlayerState())

    def average(players, value):
        return fmean(value(state(player)) for player in players) if players else 0.0

    absent = [
        player
        for player in regular
        if player not in selected
        and _experience(state(player), config) >= config.minimum_player_games
    ]
    leader = max(
        absent,
        key=lambda player: (_rating(state(player), config), player),
        default=None,
    )
    selected_median = median(_rating(state(player), config) for player in selected)
    return TeamLineupForecast(
        average(selected, lambda item: _rating(item, config))
        - average(regular, lambda item: _rating(item, config)),
        average(selected, lambda item: _form(item, config))
        - average(regular, lambda item: _form(item, config)),
        sum(
            _experience(state(player), config) >= config.minimum_player_games
            for player in selected
        )
        / len(selected),
        leader,
        _rating(state(leader), config) - selected_median if leader is not None else 0.0,
    )


def _performance(appearance):
    stats = appearance.stats
    value = (
        3 * stats.kicks
        + 2 * stats.handballs
        + 3 * stats.marks
        + 6 * stats.goals
        + stats.behinds
        + stats.hit_outs
        + 4 * stats.tackles
        + 3 * stats.clearances
        + 0.5 * stats.contested_possessions
        + 3 * stats.goal_assists
        - 3 * stats.clangers
    )
    return value / max(0.5, appearance.percent_played / 100)


def _update_official_rating(state, appearance, config):
    value = appearance.official_rating_points
    if value is None:
        return state.official_games, state.official_average, state.official_recent
    if config.measurement == "official_points_per_time":
        value /= max(0.5, appearance.percent_played / 100)
    games = state.official_games + 1
    if state.official_games == 0:
        return games, value, value
    average = state.official_average + config.official_average_rate * (
        value - state.official_average
    )
    recent = state.official_recent + config.form_rate * (
        value - state.official_recent
    )
    return games, average, recent


def replay_player_predictions(
    matches: list[MatchRow],
    control_rows: list[PredictionRow],
    appearances: list[PlayerMatch],
    lineups: list[LineupSnapshot],
    config: PlayerModelConfig,
) -> tuple[list[PredictionRow], list[PlayerDiagnostic]]:
    matches = _deduplicate(matches, lambda match: match.match_id, "match result")
    validate_matches(matches)
    history = {match.match_id: match for match in matches}
    controls = [
        row for row in control_rows if row.model_name == config.control_model_name
    ]
    appearances = _deduplicate(
        appearances, lambda row: (row.match_id, row.player_id), "player appearance"
    )
    by_match = defaultdict(list)
    for appearance in appearances:
        match = history.get(appearance.match_id)
        if match is None or appearance.team not in (match.home_team, match.away_team):
            raise ValueError(f"Unknown player match or team: {appearance.match_id}")
        _aware(appearance.statistics_available_at, "statistics_available_at")
        if appearance.statistics_available_at < match.available_at:
            raise ValueError("Player statistics cannot precede result availability")
        by_match[appearance.match_id].append(appearance)
    snapshots = defaultdict(list)
    fixtures = {match.match_id: match.fixture for match in matches}
    for row in controls:
        _aware(row.cutoff, "cutoff")
        if row.cutoff > row.kickoff:
            raise ValueError("Player prediction cutoff must be at or before kickoff")
        fixture = Fixture(
            row.match_id,
            row.year,
            row.round_label,
            row.kickoff,
            row.venue,
            row.home_team,
            row.away_team,
        )
        if row.match_id in fixtures and fixtures[row.match_id] != fixture:
            raise ValueError(f"Player fixture differs from history: {row.match_id}")
        fixtures[row.match_id] = fixture
    combined = derive_historical_lineups(appearances, matches) + lineups
    for snapshot in _deduplicate(
        combined,
        lambda row: (row.match_id, row.team, row.observed_at, row.source),
        "lineup snapshot",
    ):
        fixture = fixtures.get(snapshot.match_id)
        if fixture is None or snapshot.team not in (
            fixture.home_team,
            fixture.away_team,
        ):
            raise ValueError(f"Unknown lineup match or team: {snapshot.match_id}")
        _aware(snapshot.observed_at, "observed_at")
        if snapshot.observed_at > fixture.kickoff:
            raise ValueError("Lineup observation must not be after kickoff")
        if (
            snapshot.source == "assumed_final_at_kickoff"
            and snapshot.observed_at != fixture.kickoff
        ):
            raise ValueError("Historical final lineups are known only at kickoff")
        if len(snapshot.player_ids) != len(set(snapshot.player_ids)):
            raise ValueError("Duplicate player in lineup snapshot")
        snapshots[(snapshot.match_id, snapshot.team)].append(snapshot)

    def selection(fixture, cutoff):
        teams = []
        for team in (fixture.home_team, fixture.away_team):
            eligible = [
                snapshot
                for snapshot in snapshots[(fixture.match_id, team)]
                if snapshot.observed_at <= cutoff
                and len(snapshot.player_ids) in (22, 23)
            ]
            newest = max(
                eligible,
                key=lambda snapshot: (snapshot.observed_at, snapshot.source),
                default=None,
            )
            teams.append(newest.player_ids if newest else ())
        if set(teams[0]) & set(teams[1]):
            raise ValueError(f"Player selected for both teams: {fixture.match_id}")
        return teams

    events = []
    for match in matches:
        events.append((match.fixture.kickoff, 1, match.match_id, match))
        player_available_at = max(
            (row.statistics_available_at for row in by_match[match.match_id]),
            default=match.available_at,
        )
        events.append(
            (max(match.available_at, player_available_at), 0, match.match_id, match)
        )
    for index, row in enumerate(controls):
        events.append((row.cutoff, 2, row.match_id, index))
    events.sort(key=lambda event: event[:3])
    last_cutoff = max((row.cutoff for row in controls), default=None)
    states = {}
    references = defaultdict(lambda: deque(maxlen=config.reference_lineups))
    expectations = {}
    outputs = {}
    diagnostics = {}
    for stamp, kind, match_id, item in events:
        if last_cutoff is None or stamp > last_cutoff:
            break
        if kind == 1:
            home, away = selection(item.fixture, stamp)
            if home and away:
                difference = fmean(
                    _rating(states.get(player, PlayerState()), config)
                    for player in home
                ) - fmean(
                    _rating(states.get(player, PlayerState()), config)
                    for player in away
                )
                expectations[match_id] = 1 / (1 + math.exp(-difference / 400))
            continue
        if kind == 0:
            rows = [
                row
                for row in by_match[match_id]
                if row.statistics_available_at <= stamp
            ]
            expectation = expectations.get(match_id)
            actual = (
                1.0
                if item.actual_margin > 0
                else 0.0
                if item.actual_margin < 0
                else 0.5
            )
            updated = {}
            for team, side in ((item.home_team, 1), (item.away_team, -1)):
                team_rows = [
                    row for row in rows if row.team == team and row.percent_played > 0
                ]
                total_time = sum(row.percent_played for row in team_rows)
                for appearance in team_rows:
                    prior = states.get(appearance.player_id, PlayerState())
                    performance = _performance(appearance)
                    official = _update_official_rating(prior, appearance, config)
                    change = (
                        side
                        * config.rating_rate
                        * (actual - expectation)
                        * appearance.percent_played
                        / total_time
                        if expectation is not None
                        else 0.0
                    )
                    updated[appearance.player_id] = PlayerState(
                        prior.impact_rating + change,
                        prior.experience + 1,
                        prior.career_performance
                        + 0.05 * (performance - prior.career_performance)
                        if prior.experience
                        else performance,
                        prior.recent_performance
                        + config.form_rate * (performance - prior.recent_performance)
                        if prior.experience
                        else performance,
                        *official,
                    )
            states.update(updated)
            for team, selected in zip(
                (item.home_team, item.away_team),
                selection(item.fixture, item.fixture.kickoff),
            ):
                if selected:
                    references[team].append(selected)
            continue

        row = controls[item]
        fixture = fixtures[match_id]
        home_ids, away_ids = selection(fixture, stamp)
        forecast = None
        correction = 0.0
        if not home_ids or not away_ids:
            status = "no_lineup"
        elif any(
            len(references[team]) < config.minimum_reference_lineups
            for team in (row.home_team, row.away_team)
        ):
            status = "insufficient_history"
        else:
            home = _team_forecast(home_ids, references[row.home_team], states, config)
            away = _team_forecast(away_ids, references[row.away_team], states, config)
            material = any(
                abs(team.rating_change) >= config.material_change
                or (
                    team.missing_leader is not None
                    and team.missing_leader_gap >= config.material_change
                )
                for team in (home, away)
            )
            forecast = LineupForecast(
                home,
                away,
                home.rating_change - away.rating_change,
                home.form_change - away.form_change,
            )
            if min(home.coverage, away.coverage) < config.minimum_coverage:
                status = "low_coverage"
            elif not material or (
                config.signal == "missing_leader"
                and not any(
                    team.missing_leader is not None
                    and team.missing_leader_gap >= config.material_change
                    for team in (home, away)
                )
            ):
                status = "stable_lineup"
            else:
                if config.signal in ("rating", "missing_leader", "rating_form"):
                    correction += (
                        config.rating_weight * forecast.rating_change_difference
                    )
                if config.signal in ("form", "rating_form"):
                    correction += config.form_weight * forecast.form_change_difference
                correction = max(
                    -config.correction_cap, min(config.correction_cap, correction)
                )
                status = "adjusted"
        margin = row.predicted_margin
        if correction and margin is not None:
            margin += correction
        outputs[item] = replace(
            row,
            model_name="player_lineup",
            predicted_margin=margin,
            abs_error=abs(row.actual_margin - margin)
            if row.actual_margin is not None and margin is not None
            else None,
            used_fallback=status != "adjusted" or row.used_fallback,
        )
        diagnostics[item] = PlayerDiagnostic(
            match_id, stamp, status, correction, forecast
        )
    return [outputs[index] for index in range(len(controls))], [
        diagnostics[index] for index in range(len(controls))
    ]
