from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

import numpy as np

from .data import Fixture, MatchRow, _aware, _number, validate_matches
from .player_margin import (
    PlayerHistory,
    PlayerId,
    derive_historical_lineups,
)
from .sequential_margin import PredictionRow


@dataclass(frozen=True)
class SelectedTeamConfig:
    rating_games: int = 20
    rating_prior_games: float = 10.0
    fit_penalty: float = 1.0
    minimum_training_games: int = 150

    def __post_init__(self):
        if not isinstance(self.rating_games, int) or self.rating_games < 1:
            raise ValueError("Selected-team rating games must be a positive integer")
        if (
            not isinstance(self.minimum_training_games, int)
            or self.minimum_training_games < 1
        ):
            raise ValueError("Selected-team minimum training games must be positive")
        _number(self.rating_prior_games, "rating_prior_games", minimum=0)
        _number(self.fit_penalty, "fit_penalty", minimum=0)


@dataclass(frozen=True)
class ConservativeSelectedTeamConfig:
    selected_team_weight: float = 0.7

    def __post_init__(self):
        weight = _number(
            self.selected_team_weight, "selected_team_weight", minimum=0
        )
        if weight > 1:
            raise ValueError("Selected-team weight must be at most 1")
        object.__setattr__(self, "selected_team_weight", weight)


@dataclass(frozen=True)
class SelectedTeamDiagnostic:
    match_id: str
    cutoff: datetime
    status: Literal["adjusted", "no_lineup", "insufficient_training"]
    home_rating: float | None
    away_rating: float | None
    rating_difference: float | None


@dataclass(frozen=True)
class SelectedTeamFit:
    year: int
    training_games: int
    fitted: bool
    intercept: float | None = None
    team_weight: float | None = None
    hybrid_change_weight: float | None = None
    selected_player_weight: float | None = None


@dataclass(frozen=True)
class _SelectedTeamFeature:
    control: PredictionRow
    hybrid: PredictionRow
    home_rating: float | None
    away_rating: float | None

    @property
    def rating_difference(self):
        if self.home_rating is None or self.away_rating is None:
            return None
        return self.home_rating - self.away_rating


def _deduplicate(items, key, description):
    seen = {}
    for item in items:
        identity = key(item)
        if identity in seen and seen[identity] != item:
            raise ValueError(f"Changed source data for {description}: {identity}")
        seen[identity] = item
    return list(seen.values())


def _prediction_index(rows, model_name):
    result = {}
    for row in rows:
        if row.model_name != model_name:
            continue
        key = (row.match_id, row.cutoff)
        if key in result:
            raise ValueError(f"Duplicate {model_name} prediction key: {key}")
        result[key] = row
    return result


def _lineup_index(matches, controls, history):
    fixtures = {match.match_id: match.fixture for match in matches}
    for row in controls.values():
        _aware(row.cutoff, "cutoff")
        if row.cutoff > row.kickoff:
            raise ValueError("Selected-team cutoff must be at or before kickoff")
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
            raise ValueError(f"Selected-team fixture differs from history: {row.match_id}")
        fixtures[row.match_id] = fixture
    snapshots = defaultdict(list)
    combined = derive_historical_lineups(history.appearances, matches) + history.lineups
    for snapshot in _deduplicate(
        combined,
        lambda row: (row.match_id, row.team, row.observed_at, row.source),
        "selected-team lineup snapshot",
    ):
        fixture = fixtures.get(snapshot.match_id)
        if fixture is None or snapshot.team not in (
            fixture.home_team,
            fixture.away_team,
        ):
            raise ValueError(f"Unknown selected-team lineup: {snapshot.match_id}")
        _aware(snapshot.observed_at, "observed_at")
        if snapshot.observed_at > fixture.kickoff:
            raise ValueError("Selected-team lineup must not be after kickoff")
        if (
            snapshot.source == "assumed_final_at_kickoff"
            and snapshot.observed_at != fixture.kickoff
        ):
            raise ValueError("Historical selected teams are known only at kickoff")
        if len(snapshot.player_ids) != len(set(snapshot.player_ids)):
            raise ValueError("Duplicate player in selected-team lineup")
        snapshots[(snapshot.match_id, snapshot.team)].append(snapshot)
    return fixtures, snapshots


def _selection(fixture, cutoff, snapshots):
    teams = []
    for team in (fixture.home_team, fixture.away_team):
        eligible = [
            row
            for row in snapshots[(fixture.match_id, team)]
            if row.observed_at <= cutoff and len(row.player_ids) in (22, 23)
        ]
        newest = max(
            eligible,
            key=lambda row: (row.observed_at, row.source),
            default=None,
        )
        teams.append(newest.player_ids if newest else ())
    if set(teams[0]) & set(teams[1]):
        raise ValueError(f"Player selected for both teams: {fixture.match_id}")
    return teams


def _player_estimate(history, league_mean, prior_games):
    if not history:
        return league_mean
    return (sum(history) + prior_games * league_mean) / (
        len(history) + prior_games
    )


def _build_features(matches, controls, hybrids, history, config):
    matches = _deduplicate(matches, lambda row: row.match_id, "match result")
    validate_matches(matches)
    match_index = {match.match_id: match for match in matches}
    fixtures, snapshots = _lineup_index(matches, controls, history)
    appearances = _deduplicate(
        history.appearances,
        lambda row: (row.match_id, row.player_id),
        "selected-team player appearance",
    )
    by_match = defaultdict(list)
    for row in appearances:
        match = match_index.get(row.match_id)
        if match is None or row.team not in (match.home_team, match.away_team):
            raise ValueError(f"Unknown selected-team player match: {row.match_id}")
        _aware(row.statistics_available_at, "statistics_available_at")
        if row.statistics_available_at < match.available_at:
            raise ValueError("Player statistics cannot precede result availability")
        by_match[row.match_id].append(row)
    events = []
    for match in matches:
        available = max(
            (row.statistics_available_at for row in by_match[match.match_id]),
            default=match.available_at,
        )
        events.append((max(match.available_at, available), 0, match.match_id, match))
    for key, row in controls.items():
        events.append((row.cutoff, 1, row.match_id, key))
    events.sort(key=lambda item: item[:3])
    histories = defaultdict(lambda: deque(maxlen=config.rating_games))
    league_total = 0.0
    league_count = 0
    features = {}
    for stamp, kind, match_id, item in events:
        if kind == 0:
            for appearance in by_match[match_id]:
                if (
                    appearance.statistics_available_at <= stamp
                    and appearance.official_rating_points is not None
                ):
                    value = appearance.official_rating_points
                    histories[appearance.player_id].append(value)
                    league_total += value
                    league_count += 1
            continue
        control = controls[item]
        fixture = fixtures[match_id]
        home, away = _selection(fixture, stamp, snapshots)
        if not home or not away:
            features[item] = _SelectedTeamFeature(
                control, hybrids[item], None, None
            )
            continue
        league_mean = league_total / league_count if league_count else 0.0

        def total(players: tuple[PlayerId, ...]):
            return sum(
                _player_estimate(
                    histories[player], league_mean, config.rating_prior_games
                )
                for player in players
            )

        features[item] = _SelectedTeamFeature(
            control,
            hybrids[item],
            total(home),
            total(away),
        )
    return features


def _design(features):
    return np.asarray(
        [
            [
                row.control.predicted_margin,
                row.hybrid.predicted_margin - row.control.predicted_margin,
                row.rating_difference,
            ]
            for row in features
        ],
        dtype=float,
    )


def _fit_robust(train_x, train_y, test_x, penalty):
    means = train_x.mean(axis=0)
    scales = train_x.std(axis=0)
    scales[scales < 1e-9] = 1.0
    x = np.column_stack([np.ones(len(train_x)), (train_x - means) / scales])
    z = np.column_stack([np.ones(len(test_x)), (test_x - means) / scales])
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    weights = np.ones(len(x))
    beta = np.zeros(x.shape[1])
    for _ in range(30):
        weighted_x = x * weights[:, None]
        updated = np.linalg.solve(
            x.T @ weighted_x + regularizer,
            x.T @ (weights * train_y),
        )
        residuals = train_y - x @ updated
        next_weights = 1.0 / np.maximum(np.abs(residuals), 2.0)
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
        weights = next_weights
    slopes = beta[1:] / scales
    intercept = beta[0] - np.dot(slopes, means)
    return z @ beta, (float(intercept), *map(float, slopes))


def replay_selected_team_predictions(
    matches: list[MatchRow],
    control_rows: list[PredictionRow],
    hybrid_rows: list[PredictionRow],
    official_history: PlayerHistory,
    config: SelectedTeamConfig | None = None,
) -> tuple[
    list[PredictionRow], list[SelectedTeamDiagnostic], list[SelectedTeamFit]
]:
    config = config or SelectedTeamConfig()
    controls = _prediction_index(control_rows, "team_only")
    hybrids = _prediction_index(hybrid_rows, "player_hybrid")
    if controls.keys() != hybrids.keys():
        raise ValueError("Selected-team control and hybrid prediction keys must match")
    features = _build_features(matches, controls, hybrids, official_history, config)
    outputs = {}
    diagnostics = {}
    fits = []
    years = sorted({row.year for row in controls.values()})
    for year in years:
        current = [
            (key, features[key])
            for key, row in controls.items()
            if row.year == year
        ]
        train = [
            feature
            for feature in features.values()
            if feature.control.year < year
            and feature.control.actual_margin is not None
            and feature.rating_difference is not None
        ]
        valid = [item for item in current if item[1].rating_difference is not None]
        predictions = {}
        coefficients = None
        if len(train) >= config.minimum_training_games and valid:
            values, coefficients = _fit_robust(
                _design(train),
                np.asarray([row.control.actual_margin for row in train]),
                _design([feature for _, feature in valid]),
                config.fit_penalty,
            )
            predictions.update(
                (key, float(value)) for (key, _), value in zip(valid, values)
            )
        values = coefficients or (None, None, None, None)
        fits.append(
            SelectedTeamFit(
                year=year,
                training_games=len(train),
                fitted=coefficients is not None,
                intercept=values[0],
                team_weight=values[1],
                hybrid_change_weight=values[2],
                selected_player_weight=values[3],
            )
        )
        for key, feature in current:
            hybrid = feature.hybrid
            margin = predictions.get(key, hybrid.predicted_margin)
            if feature.rating_difference is None:
                status = "no_lineup"
            elif coefficients is None:
                status = "insufficient_training"
            else:
                status = "adjusted"
            outputs[key] = replace(
                hybrid,
                model_name="selected_team_strength",
                predicted_margin=margin,
                abs_error=(
                    abs(hybrid.actual_margin - margin)
                    if hybrid.actual_margin is not None and margin is not None
                    else None
                ),
                used_fallback=feature.control.used_fallback or status != "adjusted",
            )
            diagnostics[key] = SelectedTeamDiagnostic(
                hybrid.match_id,
                hybrid.cutoff,
                status,
                feature.home_rating,
                feature.away_rating,
                feature.rating_difference,
            )
    ordered_keys = list(controls)
    return (
        [outputs[key] for key in ordered_keys],
        [diagnostics[key] for key in ordered_keys],
        fits,
    )


def replay_conservative_selected_team_predictions(
    selected_rows: list[PredictionRow],
    scoring_rows: list[PredictionRow],
    config: ConservativeSelectedTeamConfig | None = None,
) -> list[PredictionRow]:
    config = config or ConservativeSelectedTeamConfig()
    selected = _prediction_index(selected_rows, "selected_team_strength")
    scoring = _prediction_index(scoring_rows, "scoring_shots")
    if selected.keys() != scoring.keys():
        raise ValueError("Selected-team and scoring prediction keys must match")
    weight = config.selected_team_weight
    output = []
    for key, selected_row in selected.items():
        scoring_row = scoring[key]
        if (
            selected_row.predicted_margin is None
            or scoring_row.predicted_margin is None
        ):
            raise ValueError("Conservative model requires two finite predictions")
        selected_margin = _number(
            selected_row.predicted_margin, "selected_team_prediction"
        )
        scoring_margin = _number(
            scoring_row.predicted_margin, "scoring_shots_prediction"
        )
        margin = (
            weight * selected_margin
            + (1 - weight) * scoring_margin
        )
        output.append(
            replace(
                selected_row,
                model_name="conservative_selected_team",
                predicted_margin=margin,
                abs_error=(
                    abs(selected_row.actual_margin - margin)
                    if selected_row.actual_margin is not None
                    else None
                ),
                used_fallback=(
                    selected_row.used_fallback or scoring_row.used_fallback
                ),
            )
        )
    return output
