import csv
from datetime import date

from src.mae_model.build_match_context import (
    MultiWindowRateLimiter,
    _split_missing_dates_into_year_ranges,
    build_context_rows,
    fetch_open_meteo_daily_range,
    get_venue_meta,
    load_attendance_csv,
    normalize_venue_name,
)
from src.mae_model.sequential_margin import MatchRow, parse_match_date


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.headers = {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        raise RuntimeError(f"bad status {self.status_code}")


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return self._responses.pop(0)


def test_normalize_venue_name():
    assert normalize_venue_name("M.C.G.") == "mcg"
    assert normalize_venue_name("Cazaly's Stadium") == "cazalysstadium"


def test_get_venue_meta_for_known_ground():
    meta = get_venue_meta("Docklands")
    assert meta is not None
    assert meta.capacity == 53000.0


def test_load_attendance_csv_aliases(tmp_path):
    path = tmp_path / "attendance.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["date", "home_team_name", "away_team_name", "crowd", "expected_attendance"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "date": "2025-09-27",
                "home_team_name": "Brisbane",
                "away_team_name": "GWS Giants",
                "crowd": 32123,
                "expected_attendance": 30000,
            }
        )
    rows = load_attendance_csv(str(path))
    key = (date(2025, 9, 27), "Brisbane Lions", "Greater Western Sydney")
    assert rows[key]["attendance"] == 32123.0
    assert rows[key]["projected_attendance"] == 30000.0


def test_fetch_open_meteo_daily_range_parses_payload():
    payload = {
        "daily": {
            "time": ["2025-09-27", "2025-09-28"],
            "temperature_2m_mean": [18.0, 20.0],
            "precipitation_sum": [0.8, 4.2],
            "wind_speed_10m_max": [19.0, 33.0],
            "relative_humidity_2m_mean": [61.0, 67.0],
        }
    }
    session = _FakeSession([_FakeResponse(200, payload)])
    limiter = MultiWindowRateLimiter(per_minute=999, per_hour=9999, per_day=9999, per_month=99999)
    out = fetch_open_meteo_daily_range(
        session=session,
        limiter=limiter,
        latitude=-33.0,
        longitude=151.0,
        timezone_name="Australia/Sydney",
        start_date=date(2025, 9, 27),
        end_date=date(2025, 9, 28),
    )
    assert out["2025-09-27"]["weather_temp_c"] == 18.0
    assert out["2025-09-28"]["weather_rain_mm"] == 4.2


def test_split_missing_dates_into_year_ranges():
    days = [date(2024, 3, 1), date(2024, 8, 2), date(2025, 4, 10)]
    ranges = _split_missing_dates_into_year_ranges(days)
    assert ranges == [(date(2024, 3, 1), date(2024, 8, 2)), (date(2025, 4, 10), date(2025, 4, 10))]


def test_build_context_rows_uses_cache_and_projected_fallback():
    matches = [
        MatchRow(
            match_id="m1",
            year=2025,
            round_label="R1",
            date=parse_match_date("27-Sep-2025"),
            venue="M.C.G.",
            home_team="Geelong",
            away_team="Brisbane Lions",
            home_score=90.0,
            away_score=80.0,
        )
    ]
    weather_cache = {
        "mcg:2025-09-27": {
            "weather_temp_c": 16.1,
            "weather_rain_mm": 0.0,
            "weather_wind_kmh": 20.0,
            "weather_humidity_pct": 58.0,
        }
    }
    session = _FakeSession([])
    limiter = MultiWindowRateLimiter(per_minute=999, per_hour=9999, per_day=9999, per_month=99999)

    rows = build_context_rows(
        matches=matches,
        attendance_rows={},
        weather_cache=weather_cache,
        session=session,
        limiter=limiter,
        max_retries=1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["weather_temp_c"] == 16.1
    assert row["venue_capacity"] == 100024.0
    assert row["projected_attendance"] == 0.62 * 100024.0
    assert session.calls == []
