
# https://www.footywire.com/
# https://www.afl.com.au
# https://afltables.com/

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

def get_soup_data(url):
    browser = webdriver.Firefox()

    browser.get(url)
    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')
    browser.close()
    return soup

def get_all_links_from_afl_homepage(homepage_soup):
    relevant_links = []
    links = homepage_soup.find_all('a')
    for l in links:
        href = l.get('href')
        # must not be None
        # must contain /matches
        # must not contain # or - because there are other links that have these
        # must be unique and not already in the list
        if href != None and '/matches' in href and ('#' and '-') not in href and href not in relevant_links:
            relevant_links.append(href)
    return relevant_links

def get_info_from_match_soup(soup):
    home_team_score = -1
    away_team_score = -1 
    home_team_name = 'NA'
    away_team_name = 'NA'
    match_date = 'NA'
    home_team_goals = -1
    home_team_behinds = -1
    away_team_goals = -1
    away_team_behinds = -1

    # get team names
    match_title = soup.find('div', class_='mc-header__round-wrapper').text
    match_title = match_title.replace('\n','')

    match_title_split = match_title.split('•')
    match_round = match_title_split[0].strip()

    team_names = match_title_split[1].split('v')
    home_team_name = team_names[0].strip()
    away_team_name = team_names[1].strip()

    # get the scores
    scores = soup.find_all('span',attrs={'class': 'mc-header__score-main'})
    if len(scores) == 2:
        home_team_score = scores[0].text
        away_team_score = scores[1].text

    detailed_scores = soup.find_all('span',attrs={'class': 'mc-header__score-split'})
    if len(detailed_scores) == 2:
        home_team_goals = detailed_scores[0].text.split('.')[0].strip()
        home_team_behinds = detailed_scores[0].text.split('.')[1].strip()
        away_team_goals = detailed_scores[1].text.split('.')[0].strip()
        away_team_behinds = detailed_scores[1].text.split('.')[1].strip()
    
    # get the location
    venue = soup.find('span',class_='mc-header__venue-highlight').text

    # get the date
    match_date = soup.find('div', class_='mc-header__date-wrapper js-match-start-time').text

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
                "away_team_behinds": away_team_behinds
    }
    return match_information

def get_relevant_content_from_match(match_url):
    match_url = 'https://www.afl.com.au' + match_url
    soup = get_soup_data(match_url)
    return get_info_from_match_soup(soup)

def map_team_name(team_name):
    if team_name == "West Coast Eagles":
        return "West Coast"
    if team_name == "Geelong Cats":
        return "Geelong"
    if team_name == "Sydney Swans":
        return "Sydney"
    if team_name == "Gold Coast Suns":
        return "Gold Coast"
    if team_name == "Adelaide Crows":
        return "Adelaide"
    if team_name == "GWS Giants":
        return "Greater Western Sydney"
    return team_name

def get_all_matches_from_afl_homepage(homepage_soup):
    relevant_links = get_all_links_from_afl_homepage(homepage_soup)
    matches = []
    for link in relevant_links:
        match_information = get_relevant_content_from_match(link)
        # fix team names
        match_information['home_team_name'] = map_team_name(match_information['home_team_name'])
        match_information['away_team_name'] = map_team_name(match_information['away_team_name'])
        # fix match date
        datetimeobject = datetime.strptime(match_information['match_date'].split('•')[0].strip(),'%A %d %B %Y')
        match_information['match_date'] = datetimeobject.strftime('%d-%b-%Y')
        matches.append(match_information)
    return matches

def get_gameweek_matches(GAMEWEEK):
    YEAR = 2022
    COMPETITION = 1
    COMP_SEASON = 43    
    url = "https://www.afl.com.au/fixture?Competition=%s&CompSeason=%s&MatchTimezone=VENUE_TIME&Regions=2&GameWeeks=%s&Teams=1&Venues=12" % (COMPETITION,COMP_SEASON, GAMEWEEK)
    s = get_soup_data(url)
    return get_all_matches_from_afl_homepage(s)

if __name__ == '__main__':
    matches = get_gameweek_matches(5)
    count = 1
    for match in matches:
        rnd = "R"+str(GAMEWEEK)
        # 2022R101,2022,R1,16-Mar-2022,MCG,7:25 PM,0,Melbourne,97,Western Bulldogs,71,0.0
        print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
            str(YEAR)+rnd+"0"+str(count),
            YEAR,
            rnd,
            match['match_date'],
            match['venue'],
            "7:25 PM",
            0,
            match['home_team_name'],
            match['home_team_score'],
            match['away_team_name'],
            match['away_team_score'],
            0.0))
        count+=1