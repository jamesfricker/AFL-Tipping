# https://www.footywire.com/
# https://www.afl.com.au
# https://afltables.com/

# players
# use https://www.footywire.com/afl/footy/tp-adelaide-crows
# to get player id
# and then scrape match data from afl to get the lineups

# get matches

# get details from a match
# player details
# match result?

from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
import pytest
from datetime import datetime
import csv


def get_soup_data(url):
    browser = webdriver.Firefox()
    browser.get(url)
    # TODO: Don't wait 2 seconds if the page has loaded
    time.sleep(2)
    html = browser.page_source
    soup = BeautifulSoup(html, "html.parser")
    browser.close()
    return soup


def get_all_links_from_afl_homepage(homepage_soup):
    relevant_links = []
    links = homepage_soup.find_all("a")
    for l in links:
        href = l.get("href")
        # must not be None
        # must contain /matches
        # must not contain # or - because there are other links that have these
        # must be unique and not already in the list
        if (
            href != None
            and "/matches" in href
            and ("#" and "-") not in href
            and href not in relevant_links
        ):
            relevant_links.append(href)
    return relevant_links


def get_info_from_match_soup(soup):
    home_team_score = -1
    away_team_score = -1
    home_team_name = "NA"
    away_team_name = "NA"
    match_date = "NA"
    home_team_goals = -1
    home_team_behinds = -1
    away_team_goals = -1
    away_team_behinds = -1

    # get team names
    match_title = soup.find("div", class_="mc-header__round-wrapper").text
    match_title = match_title.replace("\n", "")

    match_title_split = match_title.split("•")
    match_round = match_title_split[0].strip()

    team_names = match_title_split[1].split("v")
    home_team_name = team_names[0].strip()
    away_team_name = team_names[1].strip()

    # get the scores
    scores = soup.find_all("span", attrs={"class": "mc-header__score-main"})
    if len(scores) == 2:
        home_team_score = scores[0].text
        away_team_score = scores[1].text

    detailed_scores = soup.find_all("span", attrs={"class": "mc-header__score-split"})
    if len(detailed_scores) == 2:
        home_team_goals = detailed_scores[0].text.split(".")[0].strip()
        home_team_behinds = detailed_scores[0].text.split(".")[1].strip()
        away_team_goals = detailed_scores[1].text.split(".")[0].strip()
        away_team_behinds = detailed_scores[1].text.split(".")[1].strip()

    # get the location
    venue = soup.find("span", class_="mc-header__venue-highlight").text

    # get the date
    match_date = soup.find(
        "div", class_="mc-header__date-wrapper js-match-start-time"
    ).text

    match_information = {
        "match_date": match_date,
        "venue": venue,
        "home_team_name": home_team_name,
        "home_team_score": home_team_score,
        "away_team_name": away_team_name,
        "away_team_score": away_team_score,
        "home_team_goals": home_team_goals,
        "home_team_behinds": home_team_behinds,
        "away_team_goals": away_team_goals,
        "away_team_behinds": away_team_behinds,
    }
    return match_information


def get_relevant_content_from_match(match_url):
    match_url = "https://www.afl.com.au" + match_url
    soup = get_soup_data(match_url)
    return get_info_from_match_soup(soup)


def get_all_matches_from_afl_homepage(homepage_soup):
    relevant_links = get_all_links_from_afl_homepage(homepage_soup)
    matches = []
    for link in relevant_links:
        match_information = get_relevant_content_from_match(link)
        original_match_date = match_information["match_date"]
        # fix match date
        datetimeobject = datetime.strptime(
            original_match_date.split("•")[0].strip(), "%A %d %B %Y"
        )
        match_information["match_date"] = datetimeobject.strftime("%d-%b-%Y")
        match_information["match_time"] = original_match_date.split("•")[1].strip()

        matches.append(match_information)
    return matches


def year_to_comp_season_map(year):
    return {
        2012: 2,
        2013: 4,
        2014: 5,
        2015: 7,
        2016: 9,
        2017: 11,
        2018: 14,
        2019: 18,
        2020: 20,
        2021: 34,
        2022: 43,
        2023: 52,
    }[year]


def get_gameweek_matches(gameweek, year):
    COMPETITION = 1
    COMP_SEASON = year_to_comp_season_map(year)
    GAMEWEEK = gameweek
    url = f"https://www.afl.com.au/fixture?Competition={COMPETITION}&CompSeason={COMP_SEASON}&MatchTimezone=VENUE_TIME&Regions=2&GameWeeks={GAMEWEEK}&Teams=1&Venues=13"
    s = get_soup_data(url)
    return get_all_matches_from_afl_homepage(s)


def get_all_matches(year):
    COMPETITION = 1
    COMP_SEASON = year_to_comp_season_map(year)
    matches = []
    for GAMEWEEK in range(1, 28):
        url = f"https://www.afl.com.au/fixture?Competition={COMPETITION}&CompSeason={COMP_SEASON}&MatchTimezone=VENUE_TIME&Regions=2&GameWeeks={GAMEWEEK}&Teams=1&Venues=13"
        s = get_soup_data(url)
        matches.extend(get_all_matches_from_afl_homepage(s))
    return matches


def write_data_to_csv(csv_name):
    # read scraping index txt
    with open("scraping_index.txt", "r") as f:
        f = f.read()
        index_year = int(f.split(" ")[0])
        index_gameweek = int(f.split(" ")[1])

    if index_year:
        print("Scraping has already been done. Don't write a header")
    else:
        # write header
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
                    "margin",
                ]
            )
    for year in range(2012, 2024):
        for gameweek in range(1, 28):
            if year <= index_year and gameweek <= index_gameweek:
                continue
            matches = get_gameweek_matches(gameweek, year)
            rnd = "R" + str(gameweek)
            count = 1
            for match in matches:
                # 2022R101,2022,R1,16-Mar-2022,MCG,7:25 PM,0,Melbourne,97,Western Bulldogs,71,0.0
                with open(csv_name, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            str(year) + rnd + "0" + str(count),
                            year,
                            rnd,
                            match["match_date"],
                            match["venue"],
                            match["match_time"],
                            match["home_team_name"],
                            match["home_team_score"],
                            match["away_team_name"],
                            match["away_team_score"],
                            abs(
                                int(match["home_team_score"])
                                - int(match["away_team_score"])
                            ),
                        ]
                    )
                count += 1
            # write current position to file
            # so we can save our progress
            with open("scraping_index.txt", "w") as f:
                f.write(str(year) + " " + str(gameweek))


if __name__ == "__main__":
    write_data_to_csv("../outputs/afl.csv")
