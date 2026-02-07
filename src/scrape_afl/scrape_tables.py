import csv
import re
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup


MATCH_FIELDS = [
    "match_id",
    "year",
    "round",
    "date",
    "venue",
    "time",
    "home_team_name",
    "home_team_score",
    "home_goals",
    "home_behinds",
    "home_scoring_shots",
    "away_team_name",
    "away_team_score",
    "away_goals",
    "away_behinds",
    "away_scoring_shots",
]

PLAYER_FIELDS = [
    "match_id",
    "year",
    "round",
    "date",
    "venue",
    "team_name",
    "opponent_name",
    "home_team_name",
    "away_team_name",
    "player_name",
    "player_ref",
    "jumper_number",
    "subbed_on",
    "subbed_off",
    "kicks",
    "marks",
    "handballs",
    "disposals",
    "goals",
    "behinds",
    "hit_outs",
    "tackles",
    "rebound_50s",
    "inside_50s",
    "clearances",
    "clangers",
    "free_kicks_for",
    "free_kicks_against",
    "brownlow_votes",
    "contested_possessions",
    "uncontested_possessions",
    "contested_marks",
    "marks_inside_50",
    "one_percenters",
    "bounces",
    "goal_assists",
    "percent_played",
]


TEAM_ID_LOOKUP = {
    "Adelaide": "01",
    "Brisbane Bears": "02",
    "Carlton": "03",
    "Collingwood": "04",
    "Essendon": "05",
    "Fitzroy": "06",
    "Footscray": "07",
    "Western Bulldogs": "07",
    "Fremantle": "08",
    "Geelong": "09",
    "Hawthorn": "10",
    "Melbourne": "11",
    "North Melbourne": "12",
    "Kangaroos": "12",
    "Port Adelaide": "13",
    "Richmond": "14",
    "St Kilda": "15",
    "South Melbourne": "16",
    "Sydney": "16",
    "West Coast": "18",
    "Brisbane Lions": "19",
    "Gold Coast": "20",
    "Greater Western Sydney": "21",
    "GW Sydney": "21",
    "University": "22",
}


PLAYER_HEADER_TO_KEY = {
    "#": "jumper_number",
    "Player": "player_name",
    "KI": "kicks",
    "MK": "marks",
    "HB": "handballs",
    "DI": "disposals",
    "GL": "goals",
    "BH": "behinds",
    "HO": "hit_outs",
    "TK": "tackles",
    "RB": "rebound_50s",
    "IF": "inside_50s",
    "CL": "clearances",
    "CG": "clangers",
    "FF": "free_kicks_for",
    "FA": "free_kicks_against",
    "BR": "brownlow_votes",
    "CP": "contested_possessions",
    "UP": "uncontested_possessions",
    "CM": "contested_marks",
    "MI": "marks_inside_50",
    "1%": "one_percenters",
    "BO": "bounces",
    "GA": "goal_assists",
    "%P": "percent_played",
}


def get_soup_data(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_all_match_links_from_season(soup):
    # season pages like:
    # https://afltables.com/afl/seas/2012.html
    match_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "stats/games" not in href:
            continue
        normalized = href.replace("../", "https://afltables.com/afl/")
        match_links.append(normalized)
    return match_links


def convert_date(date):
    return datetime.strptime(date, "%d-%b-%Y").strftime("%Y%m%d")


def _normalize_whitespace(value):
    return re.sub(r"\s+", " ", value or "").strip()


def _coerce_int(value):
    cleaned = _normalize_whitespace(value).replace("\xa0", "")
    if cleaned in ("", "-", "—"):
        return 0
    return int(cleaned)


def _get_main_scoreboard_table(soup):
    return soup.find_all("table", {"style": "font: 12px Verdana;"})[0]


def get_match_round(soup):
    table = _get_main_scoreboard_table(soup)
    logistics = _normalize_whitespace(table.find_all("td")[1].get_text(" ", strip=True))
    match = re.search(r"Round:\s*(.*?)\s*Venue:", logistics)
    if not match:
        raise ValueError(f"Could not parse round from '{logistics}'")
    return match.group(1).strip()


def get_venue(soup):
    table = _get_main_scoreboard_table(soup)
    venue_anchor = table.find_all("td")[1].find("a")
    return _normalize_whitespace(venue_anchor.text)


def get_match_date(soup):
    table = _get_main_scoreboard_table(soup)
    logistics = _normalize_whitespace(table.find_all("td")[1].get_text(" ", strip=True))
    match = re.search(r"Date:\s*\w{3},\s*(\d{1,2}-[A-Za-z]{3}-\d{4})", logistics)
    if not match:
        raise ValueError(f"Could not parse match date from '{logistics}'")
    return match.group(1)


def get_match_time(soup):
    table = _get_main_scoreboard_table(soup)
    logistics = _normalize_whitespace(table.find_all("td")[1].get_text(" ", strip=True))
    match = re.search(r"(\d{1,2}:\d{2}\s*[AP]M)", logistics)
    if not match:
        raise ValueError(f"Could not parse match time from '{logistics}'")
    return match.group(1)


def get_logisitics_row(soup):
    table = _get_main_scoreboard_table(soup)
    return _normalize_whitespace(table.find_all("td")[0].get_text(" ", strip=True))


def team_name_to_id_map(team_name):
    if team_name not in TEAM_ID_LOOKUP:
        raise KeyError(f"Unknown team name '{team_name}'")
    return TEAM_ID_LOOKUP[team_name]


def create_match_id(match_date, home_team_id, away_team_id):
    lower_id = min(home_team_id, away_team_id)
    higher_id = max(home_team_id, away_team_id)
    return f"{lower_id}{higher_id}{match_date}"


def get_data_from_match(soup):
    # match pages like:
    # https://afltables.com/afl/stats/games/2012/162120120324.html
    match_date = get_match_date(soup)
    converted_date = convert_date(match_date)
    scoreboard = _get_main_scoreboard_table(soup)

    home_row = scoreboard.find_all("tr")[1].find_all("td")
    away_row = scoreboard.find_all("tr")[2].find_all("td")
    home_team_name = _normalize_whitespace(home_row[0].get_text(" ", strip=True))
    away_team_name = _normalize_whitespace(away_row[0].get_text(" ", strip=True))
    final_home_score = _normalize_whitespace(home_row[-1].find("b").text)
    final_away_score = _normalize_whitespace(away_row[-1].find("b").text)
    final_home_split = _normalize_whitespace(home_row[-1].get_text(" ", strip=True))
    final_away_split = _normalize_whitespace(away_row[-1].get_text(" ", strip=True))
    home_goal_match = re.search(r"(\d+)\.(\d+)\.\s*\d+", final_home_split)
    away_goal_match = re.search(r"(\d+)\.(\d+)\.\s*\d+", final_away_split)
    home_goals = int(home_goal_match.group(1)) if home_goal_match else 0
    home_behinds = int(home_goal_match.group(2)) if home_goal_match else 0
    away_goals = int(away_goal_match.group(1)) if away_goal_match else 0
    away_behinds = int(away_goal_match.group(2)) if away_goal_match else 0

    return {
        "match_id": create_match_id(
            match_date=converted_date,
            home_team_id=team_name_to_id_map(home_team_name),
            away_team_id=team_name_to_id_map(away_team_name),
        ),
        "year": match_date.split("-")[2],
        "round": get_match_round(soup),
        "date": match_date,
        "venue": get_venue(soup),
        "time": get_match_time(soup),
        "home_team_name": home_team_name,
        "home_team_score": final_home_score,
        "home_goals": home_goals,
        "home_behinds": home_behinds,
        "home_scoring_shots": home_goals + home_behinds,
        "away_team_name": away_team_name,
        "away_team_score": final_away_score,
        "away_goals": away_goals,
        "away_behinds": away_behinds,
        "away_scoring_shots": away_goals + away_behinds,
    }


def _extract_stat_table_team_name(table):
    first_header = table.find("th")
    if not first_header:
        return ""
    title = _normalize_whitespace(first_header.get_text(" ", strip=True))
    if "Match Statistics" not in title:
        return ""
    return title.split("Match Statistics")[0].strip()


def _get_match_stat_tables(soup):
    match_tables = []
    for table in soup.find_all("table", class_="sortable"):
        team_name = _extract_stat_table_team_name(table)
        if team_name:
            match_tables.append((team_name, table))
    return match_tables


def _parse_player_row(cells, headers):
    parsed = {}
    for idx, header in enumerate(headers):
        key = PLAYER_HEADER_TO_KEY.get(header)
        if key is None:
            continue
        cell = cells[idx]
        text = _normalize_whitespace(cell.get_text(" ", strip=True))
        if header == "Player":
            player_link = cell.find("a")
            parsed["player_ref"] = player_link["href"] if player_link else ""
            parsed[key] = text
        elif header == "#":
            parsed[key] = _normalize_whitespace(text.replace("↑", "").replace("↓", ""))
            parsed["subbed_on"] = "↑" in text
            parsed["subbed_off"] = "↓" in text
        else:
            parsed[key] = _coerce_int(text)
    return parsed


def get_player_stats_from_match(soup):
    match_data = get_data_from_match(soup)
    stat_tables = _get_match_stat_tables(soup)
    player_rows = []

    for team_name, table in stat_tables:
        header_rows = table.find("thead").find_all("tr")
        headers = [_normalize_whitespace(th.get_text(" ", strip=True)) for th in header_rows[1].find_all("th")]
        body_rows = table.find("tbody").find_all("tr")

        for body_row in body_rows:
            cells = body_row.find_all("td")
            if len(cells) != len(headers):
                continue

            parsed_row = _parse_player_row(cells, headers)
            if not parsed_row.get("player_name"):
                continue

            row = {
                "match_id": match_data["match_id"],
                "year": int(match_data["year"]),
                "round": match_data["round"],
                "date": match_data["date"],
                "venue": match_data["venue"],
                "team_name": team_name,
                "home_team_name": match_data["home_team_name"],
                "away_team_name": match_data["away_team_name"],
            }
            row.update(parsed_row)
            row["opponent_name"] = (
                match_data["away_team_name"]
                if team_name == match_data["home_team_name"]
                else match_data["home_team_name"]
            )
            player_rows.append(row)

    return player_rows


def get_match_bundle(soup):
    return {
        "match": get_data_from_match(soup),
        "players": get_player_stats_from_match(soup),
    }


def write_header(csv_name):
    with open(csv_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_FIELDS)
        writer.writeheader()


def write_player_header(csv_name):
    with open(csv_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PLAYER_FIELDS)
        writer.writeheader()


def write_csv_row(csv_name, data, fieldnames):
    with open(csv_name, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({key: data.get(key, "") for key in fieldnames})


def get_date_from_url(url):
    date = url.split("/")[-1].split(".")[0]
    return date[-8:]


def _dedupe_matches(csv_name):
    seen = set()
    deduped_rows = []
    with open(csv_name, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            match_id = row["match_id"]
            if match_id in seen:
                continue
            seen.add(match_id)
            deduped_rows.append(row)

    with open(csv_name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped_rows)


def _read_scrape_index(index_path):
    if not index_path.exists():
        return 0
    return int(index_path.read_text().strip())


def write_data_to_csv(
    matches_csv_name="../outputs/afl_data.csv",
    players_csv_name=None,
    season_start=2012,
    season_end=2024,
    scraping_index="scraping_index.txt",
):
    index_path = Path(scraping_index)
    index = _read_scrape_index(index_path)

    if index == 0:
        write_header(matches_csv_name)
        if players_csv_name:
            write_player_header(players_csv_name)

    for season in range(season_start, season_end):
        season_url = f"https://afltables.com/afl/seas/{season}.html"
        season_soup = get_soup_data(season_url)
        match_links = get_all_match_links_from_season(season_soup)
        for match_link in match_links:
            match_date = int(get_date_from_url(match_link))
            if match_date < index:
                continue

            match_soup = get_soup_data(match_link)
            bundle = get_match_bundle(match_soup)
            write_csv_row(csv_name=matches_csv_name, data=bundle["match"], fieldnames=MATCH_FIELDS)
            if players_csv_name:
                for player_row in bundle["players"]:
                    write_csv_row(csv_name=players_csv_name, data=player_row, fieldnames=PLAYER_FIELDS)
            index_path.write_text(str(match_date))

    _dedupe_matches(matches_csv_name)


if __name__ == "__main__":
    write_data_to_csv()
