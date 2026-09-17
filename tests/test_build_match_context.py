import csv
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pytest

from src.mae_model.build_match_context import (
    MultiWindowRateLimiter,
    _split_missing_dates_into_year_ranges,
    build_context_rows,
    fetch_open_meteo_daily_range,
    load_attendance_csv,
    write_context_csv,
)
from src.mae_model.data import MatchRow, load_matches_csv, parse_match_date
from src.mae_model.venues import get_venue_meta, normalize_venue_name


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
            fieldnames=[
                "date",
                "home_team_name",
                "away_team_name",
                "crowd",
                "expected_attendance",
            ],
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
    limiter = MultiWindowRateLimiter(
        per_minute=999, per_hour=9999, per_day=9999, per_month=99999
    )
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
    assert ranges == [
        (date(2024, 3, 1), date(2024, 8, 2)),
        (date(2025, 4, 10), date(2025, 4, 10)),
    ]


def test_build_context_rows_preserves_missing_projection_and_weather_provenance(
    tmp_path,
):
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
        "mcg:-37.8199:144.9834:Australia/Melbourne:2025-09-27": {
            "weather_temp_c": 16.1,
            "weather_rain_mm": 0.0,
            "weather_wind_kmh": 20.0,
            "weather_humidity_pct": 58.0,
        }
    }
    session = _FakeSession([])
    limiter = MultiWindowRateLimiter(
        per_minute=999, per_hour=9999, per_day=9999, per_month=99999
    )

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
    assert row["projected_attendance"] is None
    assert row["weather_provenance"] == "observed_weather"
    assert session.calls == []
    output = tmp_path / "research.csv"
    write_context_csv(str(output), rows)
    with output.open(newline="") as handle:
        written = next(csv.DictReader(handle))
    assert written["projected_attendance"] == ""
    assert written["weather_provenance"] == "observed_weather"


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("M.C.G.", "M.C.G."),
        ("S.C.G.", "S.C.G."),
        ("Carrara", "Carrara"),
        ("Metricon Stadium", "Carrara"),
        ("Heritage Bank Stadium", "Carrara"),
        ("People First Stadium", "Carrara"),
        ("Marvel Stadium", "Docklands"),
        ("AAMI Stadium", "Football Park"),
        ("Optus Stadium", "Perth Stadium"),
        ("Domain Stadium", "Subiaco"),
        ("GMHBA Stadium", "Kardinia Park"),
        ("ENGIE Stadium", "Sydney Showground"),
        ("Accor Stadium", "Stadium Australia"),
        ("UTAS Stadium", "York Park"),
        ("Blundstone Arena", "Bellerive Oval"),
        ("TIO Stadium", "Marrara Oval"),
        ("TIO Traeger Park", "Traeger Park"),
        ("Mars Stadium", "Eureka Stadium"),
        ("Summit Sport and Recreation Park", "Summit Sports Park"),
    ],
)
def test_venue_aliases_resolve_to_the_correct_ground(name, canonical):
    assert get_venue_meta(name).canonical_name == canonical


def test_all_historical_venues_resolve():
    matches_path = Path(__file__).resolve().parents[1] / "src/outputs/afl_data.csv"
    matches = load_matches_csv(str(matches_path))
    assert matches
    unresolved = {
        match.venue for match in matches if get_venue_meta(match.venue) is None
    }
    assert unresolved == set()


def test_summit_and_carrara_have_separate_locations():
    summit = get_venue_meta("Summit Sports Park")
    assert (summit.latitude, summit.longitude) == (-35.0754, 138.8943)
    assert summit.timezone == "Australia/Adelaide"
    carrara = get_venue_meta("Carrara")
    assert (carrara.latitude, carrara.longitude) == (-28.0064, 153.3669)
    assert carrara.timezone == "Australia/Brisbane"


def _research_match(venue):
    return MatchRow(
        match_id="m1",
        year=2025,
        round_label="R1",
        date=parse_match_date("27-Sep-2025"),
        venue=venue,
        home_team="Geelong",
        away_team="Brisbane Lions",
        home_score=90.0,
        away_score=80.0,
    )


def test_unknown_venue_has_no_coordinates_weather_or_capacity(capsys):
    assert get_venue_meta("Unknown Stadium") is None
    session = _FakeSession([])
    rows = build_context_rows(
        matches=[_research_match("Unknown Stadium")],
        attendance_rows={},
        weather_cache={"unknown:unknownstadium:2025-09-27": {"weather_temp_c": 99.0}},
        session=session,
        limiter=MultiWindowRateLimiter(999, 9999, 9999, 99999),
        max_retries=1,
    )
    assert rows[0]["venue_capacity"] is None
    assert rows[0]["weather_temp_c"] is None
    assert rows[0]["weather_provenance"] is None
    assert rows[0]["projected_attendance"] is None
    assert session.calls == []
    assert "missing venue metadata" in capsys.readouterr().out


def test_old_weather_cache_is_refetched_at_corrected_location():
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                {
                    "daily": {
                        "time": ["2025-09-27"],
                        "temperature_2m_mean": [16.0],
                    }
                },
            )
        ]
    )
    cache = {"summitsportspark:2025-09-27": {"weather_temp_c": 30.0}}
    match = _research_match("Summit Sports Park")
    rows = build_context_rows(
        matches=[match],
        attendance_rows={
            (match.date.date(), match.home_team, match.away_team): {
                "attendance": 4700.0,
                "projected_attendance": 4500.0,
            }
        },
        weather_cache=cache,
        session=session,
        limiter=MultiWindowRateLimiter(999, 9999, 9999, 99999),
        max_retries=1,
    )
    assert len(session.calls) == 1
    assert session.calls[0][1]["latitude"] == -35.0754
    assert session.calls[0][1]["longitude"] == 138.8943
    assert session.calls[0][1]["timezone"] == "Australia/Adelaide"
    assert rows[0]["weather_temp_c"] == 16.0
    assert rows[0]["weather_provenance"] == "observed_weather"
    assert rows[0]["projected_attendance"] == 4500.0


def test_venue_aliases_share_one_weather_request():
    match = _research_match("Carrara")
    session = _FakeSession(
        [
            _FakeResponse(
                200,
                {
                    "daily": {
                        "time": ["2025-09-27"],
                        "temperature_2m_mean": [25.0],
                    }
                },
            )
        ]
    )
    rows = build_context_rows(
        matches=[match, replace(match, match_id="m2", venue="Metricon Stadium")],
        attendance_rows={},
        weather_cache={},
        session=session,
        limiter=MultiWindowRateLimiter(999, 9999, 9999, 99999),
        max_retries=1,
    )
    assert len(session.calls) == 1
    assert [row["weather_temp_c"] for row in rows] == [25.0, 25.0]


def test_context_uses_venue_local_day_for_utc_kickoff():
    match = replace(
        _research_match("M.C.G."),
        date=datetime.fromisoformat("2025-09-27T19:00:00+00:00"),
    )
    session = _FakeSession([])
    rows = build_context_rows(
        matches=[match],
        attendance_rows={
            (date(2025, 9, 28), match.home_team, match.away_team): {
                "projected_attendance": 50000.0,
            }
        },
        weather_cache={
            "mcg:-37.8199:144.9834:Australia/Melbourne:2025-09-28": {
                "weather_temp_c": 17.0,
            }
        },
        session=session,
        limiter=MultiWindowRateLimiter(999, 9999, 9999, 99999),
        max_retries=1,
    )
    assert rows[0]["date"] == "2025-09-28"
    assert rows[0]["weather_temp_c"] == 17.0
    assert rows[0]["projected_attendance"] == 50000.0
    assert session.calls == []
