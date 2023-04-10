import requests

from bs4 import BeautifulSoup
from datetime import datetime
import csv
import re
import pandas as pd


def get_soup_data(url):
    data = requests.get(url)
    soup = BeautifulSoup(data.text, "html.parser")
    return soup


def parse_afl_tables_season_soup(soup):
    tables = soup.find_all("table")
    table_data = []
    for table in tables:
        table_data.append(parse_afl_table_soup(table))
    return table_data


def get_all_match_links_from_season(soup):
    # soup from seasons like
    # https://afltables.com/afl/stats/seasons/2012.html

    all_links = soup.find_all("a", href=True)
    match_links = []
    for link in all_links:
        href = link["href"]
        if "stats/games" in href:
            href = href.replace("../", "https://afltables.com/afl/")
            match_links.append(href)
    return match_links


def convert_date(date) -> str:
    date_obj = datetime.strptime(date, "%d-%b-%Y")
    return date_obj.strftime("%Y%m%d")


def get_data_from_match(soup):
    # soup from games like
    # https://afltables.com/afl/stats/games/2012/162120120324.html
    logistics_row = get_logisitics_row(soup)
    match_date = get_match_date(soup)
    converted_date = convert_date(match_date)
    # Find the table containing the team names
    table = soup.find_all("table", {"style": "font: 12px Verdana;"})[0]

    # Extract the team names
    home_team_name = table.find_all("tr")[1].find_all("td")[0].text
    away_team_name = table.find_all("tr")[2].find_all("td")[0].text

    final_home_score = table.find_all("tr")[1].find_all("td")[-1].find("b").text
    final_away_score = table.find_all("tr")[2].find_all("td")[-1].find("b").text

    # venue = table.find_all("td", colspan=5)[0].find("a").text

    return {
        "match_id": create_match_id(
            match_date=converted_date,
            home_team_id=team_name_to_id_map(home_team_name),
            away_team_id=team_name_to_id_map(away_team_name),
        ),
        "year": match_date.split("-")[2],
        "round": get_match_round(soup),
        "date": get_match_date(soup),
        "venue": get_venue(soup),
        "time": get_match_time(soup),
        "home_team_name": home_team_name,
        "home_team_score": final_home_score,
        "away_team_name": away_team_name,
        "away_team_score": final_away_score,
    }


def get_match_round(soup) -> str:
    # Find the table containing the round information
    table = soup.find_all("table", {"style": "font: 12px Verdana;"})[0]
    round_info = table.find_all("td")[1].text
    # Extract the round number as an integer
    round_num = round_info.split("Venue:")[0].split("Round:")[1].strip()
    return round_num


def get_venue(soup) -> str:
    table = soup.find_all("table", {"style": "font: 12px Verdana;"})[0]
    td = table.find_all("td")[1]
    venue = td.find("a").text
    return venue


def get_match_time(soup) -> str:
    # (" ").join(logistics_row.split("Date:")[1].strip().split(" ")[2:4]),
    table = soup.find_all("table", {"style": "font: 12px Verdana;"})[0]
    td = table.find_all("td")[1].text
    time = (" ").join(td.split("Date:")[1].strip().split(" ")[2:4])
    return time


def get_logisitics_row(soup) -> str:
    # Find the table containing the round information
    table = soup.find("table")
    # Extract the round information from the table
    logistics_row = table.find_all("td")[0].text
    return logistics_row


def get_match_date(soup) -> str:
    # Find the table containing the date information
    table = soup.find_all("table", {"style": "font: 12px Verdana;"})[0]
    td = table.find_all("td")[1].text
    date_str = td.split("Date: ")[1].split(" ")[1]
    return date_str


def get_match_time(soup) -> str:
    # Find the table containing the round information
    table = soup.find_all("table", {"style": "font: 12px Verdana;"})[0]
    target_row = table.find_all("td")[1].text
    # Extract the round information from the table
    date_info = target_row.split("Date:")[1].strip()
    # Extract the round number as an integer
    time = (" ").join(date_info.split(" ")[2:4])
    return time


def team_name_to_id_map(team_name):
    # using afl tables team ids
    return {
        "Adelaide": "01",
        "Brisbane Lions": "19",
        "Carlton": "03",
        "Collingwood": "04",
        "Essendon": "05",
        "Fremantle": "08",
        "Geelong": "09",
        "Greater Western Sydney": "21",
        "Sydney": "16",
        "North Melbourne": "12",
        "Port Adelaide": "13",
        "Richmond": "14",
        "St Kilda": "15",
        "West Coast": "18",
        "Western Bulldogs": "07",
        "Melbourne": "11",
        "Hawthorn": "10",
        "Gold Coast": "20",
    }[team_name]


def create_match_id(match_date, home_team_id, away_team_id):
    # match date is form
    # YYYYMMDD
    lower_id = min(home_team_id, away_team_id)
    higher_id = max(home_team_id, away_team_id)
    return f"{lower_id}{higher_id}{match_date}"


def write_header(csv_name):
    with open(csv_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "match_id",
                "year",
                "round",
                "date",
                "venue",
                "time",
                "home_team_name",
                "home_team_score",
                "away_team_name",
                "away_team_score",
            ]
        )


def write_csv_row(csv_name, data):
    with open(csv_name, "a") as f:
        writer = csv.writer(f)
        writer.writerow(data.values())


def get_date_from_url(url):
    date = url.split("/")[-1].split(".")[0]
    return date[-8:]


def write_data_to_csv():
    # read scraping index txt
    with open("scraping_index.txt", "r") as f:
        f = f.read()
        index = int(f)

    csv_name = "../outputs/afl_data.csv"

    if index:
        print("Scraping has already been done. Don't write a header")
    else:
        write_header(csv_name)
    for season in range(2012, 2024):
        # https://afltables.com/afl/seas/2012.html
        season_url = f"https://afltables.com/afl/seas/{season}.html"
        season_soup = get_soup_data(season_url)
        match_links = get_all_match_links_from_season(season_soup)
        for match_link in match_links:
            print(match_link)
            match_date = int(get_date_from_url(match_link))
            if match_date < index:
                print("Already scraped this match")
                continue
            match_soup = get_soup_data(match_link)
            match_data = get_data_from_match(match_soup)
            write_csv_row(csv_name=csv_name, data=match_data)
            with open("scraping_index.txt", "w") as f:
                f.write(str(match_date))

    # remove match duplicates
    df = pd.read_csv(csv_name)
    df.drop_duplicates(subset="match_id", inplace=True)
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    write_data_to_csv()
