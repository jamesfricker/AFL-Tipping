"""Import season score tables without a request for each match."""

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.scrape_afl.scrape_tables import MATCH_FIELDS


SOURCE_ROOT = "https://afltables.com/afl/seas/"
EXTRA_FIELDS = ["time_status", "source_url", "venue_source_url", "result_note"]
ROUND_PATTERN = re.compile(r"^(?:Round \d+|.*Final.*|Section.*)$")


def parse_season_scores(html: bytes | str, year: int) -> tuple[list[dict], dict]:
    soup = BeautifulSoup(html, "html.parser")
    source_url = f"{SOURCE_ROOT}{year}.html"
    rows = []
    current_round = None
    source_regular_games = None
    for element in soup.find_all(["b", "table"]):
        if element.name == "b":
            label = " ".join(
                element.find_all(string=True, recursive=False)
            ).strip()
            if ROUND_PATTERN.fullmatch(label):
                current_round = label.removeprefix("Round ")
            continue
        team_rows = element.find_all("tr", recursive=False)
        if len(team_rows) != 2:
            continue
        cells = [row.find_all("td", recursive=False) for row in team_rows]
        if any(len(row) != 4 for row in cells):
            continue
        if not all(row[0].find("a", href=re.compile(r"teams/")) for row in cells):
            continue
        if current_round is None:
            raise ValueError(f"{year}: match has no round")
        links = element.find_all("a", href=re.compile(r"stats/games/"))
        if len(links) != 1:
            raise ValueError(f"{year}: match must have one source ID")
        match_url = urljoin(source_url, links[0]["href"])
        match_id = Path(match_url).stem
        if not re.fullmatch(r"\d{12}", match_id):
            raise ValueError(f"{year}: invalid source match ID {match_id}")
        logistics = cells[0][3].get_text(" ", strip=True)
        date_match = re.search(r"\b(\d{2}-[A-Za-z]{3}-\d{4})\b", logistics)
        if not date_match:
            raise ValueError(f"{match_id}: missing date")
        match_date = datetime.strptime(date_match[1], "%d-%b-%Y")
        if match_date.year != year or match_date.strftime("%Y%m%d") != match_id[-8:]:
            raise ValueError(f"{match_id}: source date does not match ID or season")
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s*[AP]M)\b", logistics)
        venue = cells[0][3].find("a", href=re.compile(r"venues/"))
        if venue is None:
            raise ValueError(f"{match_id}: missing venue")
        row = {
            "match_id": match_id,
            "year": year,
            "round": current_round,
            "date": date_match[1],
            "venue": venue.get_text(" ", strip=True),
            "time": time_match[1] if time_match else "",
            "time_status": "source_local_time" if time_match else "unknown",
            "source_url": match_url,
            "venue_source_url": urljoin(source_url, venue["href"]),
            "result_note": cells[1][3]
            .get_text(" ", strip=True)
            .replace("[ Match stats ]", "")
            .strip(),
        }
        for side, team_cells in zip(("home", "away"), cells, strict=True):
            splits = re.findall(
                r"(\d+)\.(\d+)", team_cells[1].get_text(" ", strip=True)
            )
            if not splits:
                raise ValueError(f"{match_id}: missing {side} scoring split")
            goals, behinds = map(int, splits[-1])
            score = int(team_cells[2].get_text(strip=True))
            if 6 * goals + behinds != score:
                raise ValueError(
                    f"{match_id}: {side} score does not equal goals and behinds"
                )
            row.update(
                {
                    f"{side}_team_name": team_cells[0].get_text(" ", strip=True),
                    f"{side}_team_score": score,
                    f"{side}_goals": goals,
                    f"{side}_behinds": behinds,
                    f"{side}_scoring_shots": goals + behinds,
                }
            )
        rows.append(row)
    ids = [row["match_id"] for row in rows]
    source_ids = [
        Path(link["href"]).stem
        for link in soup.find_all("a", href=re.compile(r"stats/games/"))
    ]
    if not rows or len(set(ids)) != len(ids) or sorted(ids) != sorted(source_ids):
        raise ValueError(f"{year}: match count or unique source IDs do not match")
    totals = re.search(
        r"Totals Games:\s*(\d+), Goals:\s*(\d+), Behinds:\s*(\d+)",
        soup.get_text(" ", strip=True),
    )
    if totals:
        source_regular_games = int(totals[1])
        regular = [
            row
            for row in rows
            if "Final" not in row["round"] and "Section" not in row["round"]
        ]
        actual = (
            len(regular),
            sum(row[f"{side}_goals"] for row in regular for side in ("home", "away")),
            sum(row[f"{side}_behinds"] for row in regular for side in ("home", "away")),
        )
        expected = tuple(map(int, totals.groups()))
        if actual != expected:
            raise ValueError(
                f"{year}: regular season totals {actual} differ from source {expected}"
            )
    return rows, {
        "year": year,
        "matches": len(rows),
        "source_match_links": len(source_ids),
        "source_regular_games": source_regular_games,
        "unknown_times": sum(row["time_status"] == "unknown" for row in rows),
        "source_notes": [
            cell.get_text(" ", strip=True)
            for cell in soup.find_all("td")
            if re.search(
                r"forfeit|protest|abandon|cancel",
                cell.get_text(" ", strip=True),
                re.IGNORECASE,
            )
            and not cell.find("table")
        ],
    }


def import_seasons(first_year: int, last_year: int, output_dir: Path) -> dict:
    if first_year < 1897 or first_year > last_year:
        raise ValueError("Invalid season range")
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    all_rows, sources, checks = [], [], []
    with requests.Session() as session:
        for year in range(first_year, last_year + 1):
            path = cache / f"{year}.html"
            url = f"{SOURCE_ROOT}{year}.html"
            if not path.exists():
                response = session.get(url, timeout=45)
                response.raise_for_status()
                parse_season_scores(response.content, year)
                path.write_bytes(response.content)
                path.with_suffix(".fetch.json").write_text(
                    json.dumps(
                        {
                            "url": url,
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "sha256": hashlib.sha256(response.content).hexdigest(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                time.sleep(1)
            content = path.read_bytes()
            rows, check = parse_season_scores(content, year)
            all_rows.extend(rows)
            checks.append(check)
            sources.append(
                {
                    "year": year,
                    "url": url,
                    "cache_path": str(path),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                }
            )
    if len({row["match_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate match IDs across seasons")
    output_path = output_dir / "season_scores.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCH_FIELDS + EXTRA_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    manifest = {
        "first_year": first_year,
        "last_year": last_year,
        "matches": len(all_rows),
        "csv_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "team_identity": "Original source names. No merged clubs.",
        "time_policy": "First source time is venue-local. Missing source times remain unknown.",
        "sources": sources,
        "season_checks": checks,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=1897)
    parser.add_argument("--last-year", type=int, default=2011)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = import_seasons(args.first_year, args.last_year, args.output_dir)
    print(
        f"Imported {manifest['matches']} matches. See {args.output_dir / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
