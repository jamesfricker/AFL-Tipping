
# get matches

# get details from a match
# player details
# match result?

from bs4 import BeautifulSoup
import requests
import time
from selenium import webdriver
import pytest

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
    
    # get the date
    match_date = soup.find('div', class_='mc-header__date-wrapper js-match-start-time').text

    match_information = {
                "home_team_name": home_team_name,
                "away_team_name": away_team_name,
                "home_team_score": home_team_score,
                "home_team_goals": home_team_goals,
                "home_team_behinds": home_team_behinds,
                "away_team_score": away_team_score,
                "away_team_goals": away_team_goals,
                "away_team_behinds": away_team_behinds,
                "match_date": match_date
    }
    return match_information

def get_relevant_content_from_match(match_url):
    match_url = 'https://www.afl.com.au' + match_url
    soup = get_soup_data(match_url)
    return get_info_from_match_soup(soup)


if __name__ == '__main__':
    COMPETITION = 1
    COMP_SEASON = 43
    GAMEWEEK = 3
    url = "https://www.afl.com.au/fixture?Competition=%s&CompSeason=%s&MatchTimezone=MY_TIME&Regions=2&GameWeeks=%s&Teams=1&Venues=12" % (COMPETITION,COMP_SEASON, GAMEWEEK)
    s = get_soup_data(url)
    relevant_links = get_all_links_from_afl_homepage(s)
    for link in relevant_links:
        print(link)
        match_info = get_relevant_content_from_match(link)
        if match_info != None:
            print(match_info)