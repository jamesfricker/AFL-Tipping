import csv
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from statistics import fmean
from zoneinfo import ZoneInfo

from .data import (
    Fixture,
    MarketQuote,
    MatchRow,
    _aware,
    _number,
    _unique,
    validate_matches,
)

MODEL_NAMES = ("team_only", "scoring_shots", "market_only", "market_scoring_blend")
MODEL_CONFIG = {
    "team_base_score": 76.0,
    "team_home_advantage": 3.0,
    "team_attack_rate": 0.05,
    "team_defense_rate": 0.08,
    "team_carryover": 0.78,
    "base_scoring_shots": 26.0,
    "home_advantage_shots": 1.0,
    "shot_rate": 0.07,
    "shot_carryover": 0.85,
    "shot_residual_cap": 12.0,
    "conversion_window_team_games": 500,
    "initial_points_per_shot": 4.85,
    "minimum_expected_shots": 8.0,
    "missing_shots_points_per_shot": 5.0,
    "blend_window_years": 5,
    "minimum_blend_games": 120,
    "market_weight_step": 0.02,
    "default_market_weight": 1.0,
}


@dataclass(frozen=True)
class PredictionRequest:
    fixture: Fixture
    cutoff: datetime

    def __post_init__(self):
        _aware(self.cutoff, "cutoff")
        if self.cutoff > self.fixture.kickoff:
            raise ValueError("Prediction cutoff must be at or before kickoff")


@dataclass(frozen=True)
class PredictionRow:
    match_id: str
    year: int
    round_label: str
    home_team: str
    away_team: str
    venue: str
    kickoff: datetime
    cutoff: datetime
    model_name: str
    predicted_margin: float | None
    market_status: str
    market_observed_at: datetime | None
    market_home_probability: float | None
    market_weight: float
    weight_training_games: int
    weight_cutoff: datetime
    used_fallback: bool = False
    actual_margin: float | None = None
    abs_error: float | None = None
    result_available_at: datetime | None = None
    result_timing_assumption: str = ""


@dataclass(frozen=True)
class _RatingForecast:
    home_score: float
    away_score: float
    home_shots: float
    away_shots: float
    points_per_shot: float

    @property
    def team_margin(self):
        return self.home_score - self.away_score

    @property
    def shot_margin(self):
        return (self.home_shots - self.away_shots) * self.points_per_shot


class _Ratings:
    def __init__(self):
        self.year = None
        self.attack = defaultdict(float)
        self.defense = defaultdict(float)
        self.shot_attack = defaultdict(float)
        self.shot_defense = defaultdict(float)
        self.conversion = deque(maxlen=MODEL_CONFIG["conversion_window_team_games"])

    def _value(self, ratings, team, year, carryover):
        if team not in ratings:
            return 0.0
        elapsed = max(0, year - self.year) if self.year is not None else 0
        mean = fmean(ratings.values()) if elapsed and ratings else 0.0
        return (ratings.get(team, 0.0) - mean) * carryover**elapsed

    def forecast(self, fixture: Fixture) -> _RatingForecast:
        home, away, year = fixture.home_team, fixture.away_team, fixture.year
        config = MODEL_CONFIG
        def attack(team):
            return self._value(self.attack, team, year, config["team_carryover"])

        def defense(team):
            return self._value(self.defense, team, year, config["team_carryover"])

        def shots(team):
            return self._value(self.shot_attack, team, year, config["shot_carryover"])

        def shot_defense(team):
            return self._value(self.shot_defense, team, year, config["shot_carryover"])

        return _RatingForecast(
            config["team_base_score"]
            + config["team_home_advantage"]
            + attack(home)
            - defense(away),
            config["team_base_score"]
            - config["team_home_advantage"]
            + attack(away)
            - defense(home),
            max(
                config["minimum_expected_shots"],
                config["base_scoring_shots"]
                + config["home_advantage_shots"]
                + shots(home)
                - shot_defense(away),
            ),
            max(
                config["minimum_expected_shots"],
                config["base_scoring_shots"]
                - config["home_advantage_shots"]
                + shots(away)
                - shot_defense(home),
            ),
            fmean(self.conversion)
            if self.conversion
            else config["initial_points_per_shot"],
        )

    def update(self, match: MatchRow, forecast: _RatingForecast) -> None:
        config = MODEL_CONFIG
        if self.year is not None and match.year > self.year:
            for ratings, carryover in (
                (self.attack, config["team_carryover"]),
                (self.defense, config["team_carryover"]),
                (self.shot_attack, config["shot_carryover"]),
                (self.shot_defense, config["shot_carryover"]),
            ):
                mean = fmean(ratings.values()) if ratings else 0.0
                for team in ratings:
                    ratings[team] = (ratings[team] - mean) * carryover ** (
                        match.year - self.year
                    )
        self.year = max(self.year or match.year, match.year)
        elapsed = self.year - match.year
        home_residual = match.home_score - forecast.home_score
        away_residual = match.away_score - forecast.away_score
        home_shots = match.home_scoring_shots or match.home_goals + match.home_behinds
        away_shots = match.away_scoring_shots or match.away_goals + match.away_behinds
        home_shots = (
            home_shots or match.home_score / config["missing_shots_points_per_shot"]
        )
        away_shots = (
            away_shots or match.away_score / config["missing_shots_points_per_shot"]
        )
        home_shot_residual = max(
            -config["shot_residual_cap"],
            min(config["shot_residual_cap"], home_shots - forecast.home_shots),
        )
        away_shot_residual = max(
            -config["shot_residual_cap"],
            min(config["shot_residual_cap"], away_shots - forecast.away_shots),
        )
        for team, own, other, own_shots, other_shots in (
            (
                match.home_team,
                home_residual,
                away_residual,
                home_shot_residual,
                away_shot_residual,
            ),
            (
                match.away_team,
                away_residual,
                home_residual,
                away_shot_residual,
                home_shot_residual,
            ),
        ):
            self.attack[team] += (
                config["team_attack_rate"] * own * config["team_carryover"] ** elapsed
            )
            self.defense[team] -= (
                config["team_defense_rate"]
                * other
                * config["team_carryover"] ** elapsed
            )
            self.shot_attack[team] += (
                config["shot_rate"] * own_shots * config["shot_carryover"] ** elapsed
            )
            self.shot_defense[team] -= (
                config["shot_rate"] * other_shots * config["shot_carryover"] ** elapsed
            )
        if home_shots:
            self.conversion.append(match.home_score / home_shots)
        if away_shots:
            self.conversion.append(match.away_score / away_shots)


def _market_at(fixture, cutoff, quotes, closing_line_benchmark):
    eligible = [
        quote
        for quote in quotes
        if (quote.observed_at is None and closing_line_benchmark)
        or (
            quote.observed_at is not None
            and quote.observed_at <= cutoff
            and quote.observed_at <= fixture.kickoff
        )
    ]
    if not eligible:
        return None, "no_eligible_quote" if quotes else "missing_quote"
    quote = max(eligible, key=lambda quote: quote.observed_at or fixture.kickoff)
    if quote.predicted_margin is None:
        return quote, "missing_margin"
    return (
        quote,
        "untimed_closing_benchmark" if quote.observed_at is None else "timed_quote",
    )


def fit_market_weight(training: list[tuple[float, float, float]]) -> float:
    """Fit one convex weight to earlier predictions. Ties keep the market."""
    if len(training) < MODEL_CONFIG["minimum_blend_games"]:
        return MODEL_CONFIG["default_market_weight"]
    best_weight = MODEL_CONFIG["default_market_weight"]
    best_error = sum(abs(actual - market) for actual, market, shots in training)
    steps = round(1 / MODEL_CONFIG["market_weight_step"])
    for index in range(steps - 1, -1, -1):
        weight = index / steps
        error = sum(
            abs(actual - (weight * market + (1 - weight) * shots))
            for actual, market, shots in training
        )
        if error < best_error - 1e-10:
            best_error, best_weight = error, weight
    return best_weight


def replay_predictions(
    matches: list[MatchRow],
    requests: list[PredictionRequest],
    market_quotes: list[MarketQuote] | None = None,
    *,
    closing_line_benchmark: bool = False,
    lead_hours: float = 0.0,
) -> list[PredictionRow]:
    """Return forecasts without changing inputs or keeping caller-owned state."""
    validate_matches(matches)
    _number(lead_hours, "lead_hours", minimum=0)
    quotes = market_quotes or []
    _unique(requests, lambda request: request.fixture.match_id, "request ID")
    _unique(
        quotes, lambda quote: (quote.match_id, quote.observed_at), "market snapshot"
    )
    if (
        any(quote.observed_at is None for quote in quotes)
        and not closing_line_benchmark
    ):
        raise ValueError(
            "Untimed market quotes require explicit closing benchmark mode"
        )
    history = {match.match_id: match for match in matches}
    for request in requests:
        if (
            request.fixture.match_id in history
            and request.fixture != history[request.fixture.match_id].fixture
        ):
            raise ValueError(
                f"Fixture differs from history: {request.fixture.match_id}"
            )
    known_ids = set(history) | {request.fixture.match_id for request in requests}
    if any(quote.match_id not in known_ids for quote in quotes):
        raise ValueError("Market quote has an unknown match ID")
    quote_index = defaultdict(list)
    for quote in quotes:
        quote_index[quote.match_id].append(quote)
    events = []
    for match in matches:
        events.append(
            (
                match.fixture.kickoff - timedelta(hours=lead_hours),
                1,
                match.match_id,
                match,
            )
        )
        events.append((match.available_at, 0, match.match_id, match))
    for request in requests:
        events.append((request.cutoff, 2, request.fixture.match_id, request))
    events.sort(key=lambda event: event[:3])
    ratings = _Ratings()
    forecasts = {}
    training_history = []
    weight_cache = {}
    outputs = {}
    last_cutoff = max((request.cutoff for request in requests), default=None)
    for stamp, kind, match_id, item in events:
        if last_cutoff is None or stamp > last_cutoff:
            break
        if kind == 0:
            ratings.update(item, forecasts[match_id])
            continue
        if kind == 1:
            forecast = ratings.forecast(item.fixture)
            forecasts[match_id] = forecast
            quote, status = _market_at(
                item.fixture, stamp, quote_index[match_id], closing_line_benchmark
            )
            if quote is not None and quote.predicted_margin is not None:
                training_history.append((item, forecast.shot_margin, quote))
            continue
        fixture = item.fixture
        forecast = ratings.forecast(fixture)
        season_cutoff = min(
            stamp, datetime(fixture.year, 1, 1, tzinfo=ZoneInfo("Australia/Sydney"))
        )
        cache_key = (fixture.year, season_cutoff)
        if cache_key not in weight_cache:
            training = [
                (match.actual_margin, quote.predicted_margin, shots)
                for match, shots, quote in training_history
                if fixture.year - MODEL_CONFIG["blend_window_years"]
                <= match.year
                < fixture.year
                and match.available_at <= season_cutoff
                and (quote.observed_at is None or quote.observed_at <= season_cutoff)
            ]
            weight_cache[cache_key] = (fit_market_weight(training), len(training))
        weight, count = weight_cache[cache_key]
        quote, status = _market_at(
            fixture, stamp, quote_index[match_id], closing_line_benchmark
        )
        market = quote.predicted_margin if quote is not None else None
        margins = (
            forecast.team_margin,
            forecast.shot_margin,
            market,
            weight * market + (1 - weight) * forecast.shot_margin
            if market is not None
            else forecast.shot_margin,
        )
        outputs[match_id] = [
            PredictionRow(
                match_id,
                fixture.year,
                fixture.round_label,
                fixture.home_team,
                fixture.away_team,
                fixture.venue,
                fixture.kickoff,
                stamp,
                name,
                margin,
                status,
                quote.observed_at if quote else None,
                quote.home_probability if quote else None,
                weight,
                count,
                season_cutoff,
                used_fallback=name == "market_scoring_blend" and market is None,
            )
            for name, margin in zip(MODEL_NAMES, margins)
        ]
    return [row for request in requests for row in outputs[request.fixture.match_id]]


def predict_fixtures(
    matches: list[MatchRow],
    fixtures: list[Fixture],
    as_of: datetime,
    market_quotes: list[MarketQuote] | None = None,
    *,
    lead_hours: float = 0.0,
) -> list[PredictionRow]:
    _aware(as_of, "as_of")
    if any(fixture.kickoff <= as_of for fixture in fixtures):
        raise ValueError("Live fixtures must start after as_of")
    validate_matches(matches)
    _unique(
        fixtures,
        lambda fixture: (fixture.kickoff, fixture.home_team, fixture.away_team),
        "fixture",
    )
    return replay_predictions(
        matches,
        [PredictionRequest(fixture, as_of) for fixture in fixtures],
        market_quotes,
        lead_hours=lead_hours,
    )


def walk_forward_predictions(
    matches: list[MatchRow],
    min_train_years: int = 3,
    market_quotes: list[MarketQuote] | None = None,
    *,
    closing_line_benchmark: bool = False,
    lead_hours: float = 0.0,
) -> list[PredictionRow]:
    if not isinstance(min_train_years, int) or min_train_years < 0:
        raise ValueError("min_train_years must be a non-negative integer")
    _number(lead_hours, "lead_hours", minimum=0)
    first_year = min((match.year for match in matches), default=0)
    requests = [
        PredictionRequest(
            match.fixture, match.fixture.kickoff - timedelta(hours=lead_hours)
        )
        for match in matches
        if match.year >= first_year + min_train_years
    ]
    rows = replay_predictions(
        matches,
        requests,
        market_quotes,
        closing_line_benchmark=closing_line_benchmark,
        lead_hours=lead_hours,
    )
    history = {match.match_id: match for match in matches}
    return [
        replace(
            row,
            actual_margin=history[row.match_id].actual_margin,
            abs_error=abs(history[row.match_id].actual_margin - row.predicted_margin)
            if row.predicted_margin is not None
            else None,
            result_available_at=history[row.match_id].available_at,
            result_timing_assumption=history[row.match_id].timing_assumption,
        )
        for row in rows
    ]


def summarize_predictions(predictions: list[PredictionRow]) -> list[dict]:
    groups = defaultdict(list)
    market_ids = {
        row.match_id
        for row in predictions
        if row.model_name == "market_only" and row.predicted_margin is not None
    }
    for row in predictions:
        for year in (str(row.year), "ALL"):
            groups[(year, row.model_name, "all_matches")].append(row)
            common = groups[(year, row.model_name, "common_market")]
            if row.match_id in market_ids:
                common.append(row)
    summary = []
    for (year, name, scope), rows in sorted(groups.items()):
        scored = [row for row in rows if row.abs_error is not None]
        correct = sum(
            ((row.predicted_margin > 0) - (row.predicted_margin < 0))
            == ((row.actual_margin > 0) - (row.actual_margin < 0))
            for row in scored
        )
        summary.append(
            {
                "year": year,
                "model_name": name,
                "scope": scope,
                "num_matches": len(rows),
                "num_games": len(scored),
                "missing_predictions": sum(
                    row.predicted_margin is None for row in rows
                ),
                "fallback_count": sum(row.used_fallback for row in rows),
                "mae_margin": round(fmean(row.abs_error for row in scored), 6)
                if scored
                else None,
                "tip_pct": round(100 * correct / len(scored), 4) if scored else None,
            }
        )
    return summary


def write_prediction_rows(path: str, rows: list[PredictionRow]):
    _write_csv(
        path, [asdict(row) for row in rows], list(PredictionRow.__dataclass_fields__)
    )


def write_summary_rows(path: str, rows: list[dict]):
    _write_csv(
        path,
        rows,
        [
            "year",
            "model_name",
            "scope",
            "num_matches",
            "num_games",
            "missing_predictions",
            "fallback_count",
            "mae_margin",
            "tip_pct",
        ],
    )


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: value.isoformat() if isinstance(value, datetime) else value
                    for key, value in row.items()
                }
            )
