import argparse
import csv
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from .sequential_margin import MatchRow, canonical_team_name, load_matches_csv, parse_match_date


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_DAILY_FIELDS = (
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
)


@dataclass(frozen=True)
class VenueMeta:
    canonical_name: str
    latitude: float
    longitude: float
    timezone: str
    length_m: Optional[float]
    width_m: Optional[float]
    capacity: Optional[float]


# Dimensions/capacity are approximate and can be overridden downstream if desired.
VENUE_METADATA: Dict[str, VenueMeta] = {
    "mcg": VenueMeta("M.C.G.", -37.8199, 144.9834, "Australia/Melbourne", 160.0, 141.0, 100024.0),
    "docklands": VenueMeta("Docklands", -37.8164, 144.9475, "Australia/Melbourne", 159.5, 128.8, 53000.0),
    "adelaideoval": VenueMeta("Adelaide Oval", -34.9154, 138.5967, "Australia/Adelaide", 167.0, 123.0, 53500.0),
    "gabba": VenueMeta("Gabba", -27.4850, 153.0381, "Australia/Brisbane", 156.0, 138.0, 42000.0),
    "perthstadium": VenueMeta("Perth Stadium", -31.9509, 115.8890, "Australia/Perth", 165.0, 130.0, 60000.0),
    "carrara": VenueMeta("Carrara", -28.0064, 153.3670, "Australia/Brisbane", 159.0, 134.0, 27000.0),
    "scg": VenueMeta("S.C.G.", -33.8917, 151.2240, "Australia/Sydney", 155.0, 136.0, 48000.0),
    "subiaco": VenueMeta("Subiaco", -31.9431, 115.8329, "Australia/Perth", 175.0, 122.0, 43000.0),
    "kardiniapark": VenueMeta("Kardinia Park", -38.1561, 144.3548, "Australia/Melbourne", 170.0, 115.0, 40000.0),
    "sydneyshowground": VenueMeta(
        "Sydney Showground", -33.8474, 151.0674, "Australia/Sydney", 164.0, 128.0, 24000.0
    ),
    "yorkpark": VenueMeta("York Park", -41.4262, 147.1344, "Australia/Hobart", 175.0, 145.0, 21000.0),
    "footballpark": VenueMeta("Football Park", -34.8940, 138.5200, "Australia/Adelaide", 167.0, 123.0, 51000.0),
    "belleriveoval": VenueMeta("Bellerive Oval", -42.8752, 147.3706, "Australia/Hobart", 175.0, 135.0, 20000.0),
    "manukaoval": VenueMeta("Manuka Oval", -35.3211, 149.1460, "Australia/Sydney", 170.0, 130.0, 15000.0),
    "marraraoval": VenueMeta("Marrara Oval", -12.4013, 130.8835, "Australia/Darwin", 177.0, 145.0, 14000.0),
    "stadiumaustralia": VenueMeta(
        "Stadium Australia", -33.8477, 151.0631, "Australia/Sydney", 170.0, 145.0, 83500.0
    ),
    "eurekastadium": VenueMeta("Eureka Stadium", -37.5516, 143.8513, "Australia/Melbourne", 160.0, 130.0, 11000.0),
    "cazalysstadium": VenueMeta(
        "Cazaly's Stadium", -16.9200, 145.7460, "Australia/Brisbane", 164.0, 137.0, 13000.0
    ),
    "traegerpark": VenueMeta("Traeger Park", -23.7068, 133.8830, "Australia/Darwin", 175.0, 145.0, 10000.0),
    "norwoodoval": VenueMeta("Norwood Oval", -34.9205, 138.6360, "Australia/Adelaide", 167.0, 123.0, 22000.0),
    "wellington": VenueMeta("Wellington", -41.2725, 174.7853, "Pacific/Auckland", 165.0, 135.0, 34500.0),
    "jiangwanstadium": VenueMeta("Jiangwan Stadium", 31.3027, 121.5045, "Asia/Shanghai", 160.0, 130.0, 25000.0),
    "summitsportspark": VenueMeta(
        "Summit Sports Park", -28.0045, 153.3650, "Australia/Brisbane", 160.0, 130.0, 5000.0
    ),
    "barossaoval": VenueMeta("Barossa Oval", -34.5270, 138.9580, "Australia/Adelaide", 165.0, 130.0, 5000.0),
    "blacktown": VenueMeta("Blacktown", -33.7706, 150.8573, "Australia/Sydney", 160.0, 130.0, 10000.0),
    "riverwaystadium": VenueMeta(
        "Riverway Stadium", -19.3022, 146.7299, "Australia/Brisbane", 165.0, 135.0, 10000.0
    ),
    "handsoval": VenueMeta("Hands Oval", -33.3389, 115.6433, "Australia/Perth", 165.0, 130.0, 8000.0),
}


def normalize_venue_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", raw.strip().lower())


def get_venue_meta(venue_name: str) -> Optional[VenueMeta]:
    return VENUE_METADATA.get(normalize_venue_name(venue_name))


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


def _coalesce_field(row: dict, names: Tuple[str, ...]) -> Optional[str]:
    for name in names:
        if name in row and row[name] is not None:
            text = str(row[name]).strip()
            if text:
                return text
    return None


def load_attendance_csv(path: Optional[str]) -> Dict[Tuple[date, str, str], dict]:
    rows: Dict[Tuple[date, str, str], dict] = {}
    if not path:
        return rows
    file_path = Path(path)
    if not file_path.exists():
        return rows

    with file_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_date = _coalesce_field(row, ("date", "match_date", "Date"))
            raw_home = _coalesce_field(row, ("home_team", "home_team_name", "Home Team"))
            raw_away = _coalesce_field(row, ("away_team", "away_team_name", "Away Team"))
            if not raw_date or not raw_home or not raw_away:
                continue
            try:
                match_date = date.fromisoformat(raw_date)
            except ValueError:
                try:
                    match_date = date.fromisoformat(raw_date.replace("/", "-"))
                except ValueError:
                    try:
                        match_date = parse_match_date(raw_date).date()
                    except ValueError:
                        continue
            key = (match_date, canonical_team_name(raw_home), canonical_team_name(raw_away))
            rows[key] = {
                "attendance": _parse_optional_float(
                    _coalesce_field(row, ("attendance", "crowd", "actual_attendance"))
                ),
                "projected_attendance": _parse_optional_float(
                    _coalesce_field(row, ("projected_attendance", "expected_attendance", "attendance_projected"))
                ),
            }
    return rows


def load_weather_cache(path: str) -> Dict[str, dict]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    with cache_path.open() as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}


def persist_weather_cache(path: str, cache: Dict[str, dict]):
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump(cache, f, sort_keys=True)


class MultiWindowRateLimiter:
    def __init__(
        self,
        per_minute: int,
        per_hour: int,
        per_day: int,
        per_month: int,
    ):
        self.per_minute = per_minute
        self.per_hour = per_hour
        self.per_day = per_day
        self.per_month = per_month
        self._minute = deque()
        self._hour = deque()
        self._day = deque()
        self._month = deque()

    @staticmethod
    def _trim(window: deque, now: float, seconds: float):
        while window and (now - window[0]) >= seconds:
            window.popleft()

    def wait_for_slot(self):
        while True:
            now = time.monotonic()
            self._trim(self._minute, now, 60.0)
            self._trim(self._hour, now, 3600.0)
            self._trim(self._day, now, 86400.0)
            self._trim(self._month, now, 30.0 * 86400.0)

            waits = []
            if len(self._minute) >= self.per_minute:
                waits.append(60.0 - (now - self._minute[0]) + 0.01)
            if len(self._hour) >= self.per_hour:
                waits.append(3600.0 - (now - self._hour[0]) + 0.01)
            if len(self._day) >= self.per_day:
                waits.append(86400.0 - (now - self._day[0]) + 0.01)
            if len(self._month) >= self.per_month:
                waits.append(30.0 * 86400.0 - (now - self._month[0]) + 0.01)

            if not waits:
                self._minute.append(now)
                self._hour.append(now)
                self._day.append(now)
                self._month.append(now)
                return
            time.sleep(max(0.01, max(waits)))


def fetch_open_meteo_daily_range(
    session: requests.Session,
    limiter: MultiWindowRateLimiter,
    latitude: float,
    longitude: float,
    timezone_name: str,
    start_date: date,
    end_date: date,
    max_retries: int = 5,
) -> Dict[str, dict]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_name,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(OPEN_METEO_DAILY_FIELDS),
        "wind_speed_unit": "kmh",
    }

    attempt = 0
    while True:
        limiter.wait_for_slot()
        response = session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=40)
        if response.status_code == 200:
            payload = response.json()
            daily = payload.get("daily", {})
            times = daily.get("time", [])
            temp = daily.get("temperature_2m_mean", [])
            rain = daily.get("precipitation_sum", [])
            wind = daily.get("wind_speed_10m_max", [])
            humidity = daily.get("relative_humidity_2m_mean", [])
            out = {}
            for idx, day in enumerate(times):
                out[day] = {
                    "weather_temp_c": temp[idx] if idx < len(temp) else None,
                    "weather_rain_mm": rain[idx] if idx < len(rain) else None,
                    "weather_wind_kmh": wind[idx] if idx < len(wind) else None,
                    "weather_humidity_pct": humidity[idx] if idx < len(humidity) else None,
                }
            return out

        retryable = response.status_code in {429, 500, 502, 503, 504}
        if not retryable or attempt >= max_retries:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                sleep_seconds = max(1.0, float(retry_after))
            except ValueError:
                sleep_seconds = min(120.0, 5.0 * (2.0**attempt))
        else:
            sleep_seconds = min(120.0, 5.0 * (2.0**attempt))
        time.sleep(sleep_seconds)
        attempt += 1


def _weather_cache_key(venue_name: str, day: date) -> str:
    return f"{normalize_venue_name(venue_name)}:{day.isoformat()}"


def _group_matches_by_venue(matches: Iterable[MatchRow]) -> Dict[str, List[MatchRow]]:
    grouped: Dict[str, List[MatchRow]] = defaultdict(list)
    for match in matches:
        grouped[match.venue].append(match)
    return grouped


def _split_missing_dates_into_year_ranges(days: List[date]) -> List[Tuple[date, date]]:
    by_year: Dict[int, List[date]] = defaultdict(list)
    for day in days:
        by_year[day.year].append(day)
    ranges: List[Tuple[date, date]] = []
    for year in sorted(by_year.keys()):
        values = sorted(by_year[year])
        ranges.append((values[0], values[-1]))
    return ranges


def build_context_rows(
    matches: List[MatchRow],
    attendance_rows: Dict[Tuple[date, str, str], dict],
    weather_cache: Dict[str, dict],
    session: requests.Session,
    limiter: MultiWindowRateLimiter,
    max_retries: int,
) -> List[dict]:
    grouped = _group_matches_by_venue(matches)
    unresolved_venues = []

    for venue_name, venue_matches in grouped.items():
        meta = get_venue_meta(venue_name)
        if meta is None:
            unresolved_venues.append(venue_name)
            continue

        missing_dates = sorted(
            {
                match.date.date()
                for match in venue_matches
                if _weather_cache_key(venue_name, match.date.date()) not in weather_cache
            }
        )
        if not missing_dates:
            continue

        for start_day, end_day in _split_missing_dates_into_year_ranges(missing_dates):
            fetched = fetch_open_meteo_daily_range(
                session=session,
                limiter=limiter,
                latitude=meta.latitude,
                longitude=meta.longitude,
                timezone_name=meta.timezone,
                start_date=start_day,
                end_date=end_day,
                max_retries=max_retries,
            )
            for day_text, payload in fetched.items():
                day = date.fromisoformat(day_text)
                weather_cache[_weather_cache_key(venue_name, day)] = payload

    if unresolved_venues:
        unresolved_list = ", ".join(sorted(unresolved_venues))
        print(f"warning: missing venue metadata for {len(unresolved_venues)} venues: {unresolved_list}")

    rows: List[dict] = []
    for match in matches:
        meta = get_venue_meta(match.venue)
        weather = weather_cache.get(_weather_cache_key(match.venue, match.date.date()), {})
        attendance_key = (match.date.date(), match.home_team, match.away_team)
        attendance_payload = attendance_rows.get(attendance_key, {})
        projected_attendance = attendance_payload.get("projected_attendance")
        if projected_attendance is None and meta is not None and meta.capacity is not None:
            projected_attendance = 0.62 * meta.capacity

        rows.append(
            {
                "date": match.date.date().isoformat(),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "venue": match.venue,
                "weather_temp_c": weather.get("weather_temp_c"),
                "weather_rain_mm": weather.get("weather_rain_mm"),
                "weather_wind_kmh": weather.get("weather_wind_kmh"),
                "weather_humidity_pct": weather.get("weather_humidity_pct"),
                "attendance": attendance_payload.get("attendance"),
                "projected_attendance": projected_attendance,
                "venue_length_m": meta.length_m if meta is not None else None,
                "venue_width_m": meta.width_m if meta is not None else None,
                "venue_capacity": meta.capacity if meta is not None else None,
            }
        )

    return rows


def write_context_csv(path: str, rows: List[dict]):
    fieldnames = [
        "date",
        "home_team",
        "away_team",
        "venue",
        "weather_temp_c",
        "weather_rain_mm",
        "weather_wind_kmh",
        "weather_humidity_pct",
        "attendance",
        "projected_attendance",
        "venue_length_m",
        "venue_width_m",
        "venue_capacity",
    ]
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build optional match context CSV (weather, attendance, venue traits) using Open-Meteo."
    )
    parser.add_argument(
        "--matches-csv",
        default="src/outputs/afl_data.csv",
        help="Path to match-level CSV.",
    )
    parser.add_argument(
        "--attendance-csv",
        default="",
        help="Optional attendance/projection CSV keyed by date/home/away.",
    )
    parser.add_argument(
        "--output-csv",
        default="src/outputs/afl_match_context.csv",
        help="Destination context CSV path.",
    )
    parser.add_argument(
        "--weather-cache",
        default=".context/open_meteo_daily_cache.json",
        help="Persistent weather cache JSON path.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retry attempts for retryable Open-Meteo responses.",
    )
    parser.add_argument(
        "--limit-minute",
        type=int,
        default=550,
        help="Max Open-Meteo calls per minute (free/open-access shown as 600/min).",
    )
    parser.add_argument(
        "--limit-hour",
        type=int,
        default=4500,
        help="Max Open-Meteo calls per hour (free/open-access shown as 5000/hour).",
    )
    parser.add_argument(
        "--limit-day",
        type=int,
        default=9500,
        help="Max Open-Meteo calls per day (free/open-access shown as 10000/day).",
    )
    parser.add_argument(
        "--limit-month",
        type=int,
        default=280000,
        help="Max Open-Meteo calls per month (free/open-access shown as 300000/month).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    matches = load_matches_csv(args.matches_csv)
    attendance_rows = load_attendance_csv(args.attendance_csv)
    weather_cache = load_weather_cache(args.weather_cache)
    limiter = MultiWindowRateLimiter(
        per_minute=args.limit_minute,
        per_hour=args.limit_hour,
        per_day=args.limit_day,
        per_month=args.limit_month,
    )

    with requests.Session() as session:
        rows = build_context_rows(
            matches=matches,
            attendance_rows=attendance_rows,
            weather_cache=weather_cache,
            session=session,
            limiter=limiter,
            max_retries=args.max_retries,
        )

    write_context_csv(args.output_csv, rows)
    persist_weather_cache(args.weather_cache, weather_cache)
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    print(f"weather cache entries: {len(weather_cache)} ({args.weather_cache})")


if __name__ == "__main__":
    main()
