from scrape_afl import scrape_data
from bs4 import BeautifulSoup
import requests
import pytest

def test_scrape_return():
    f = open('tests/test_data/round3_2022.html')
    content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    print(scrape_data.get_all_links_from_afl_homepage(soup))
    assert len(scrape_data.get_all_links_from_afl_homepage(soup))>0

def test_get_info_from_match():
    f = open('tests/test_data/dogs_sydney_r3_22.html')
    content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    match_information = {
                "home_team_name": 'Western Bulldogs',
                "away_team_name": 'Sydney Swans',
                "home_team_score": '71',
                "home_team_goals": '9',
                "home_team_behinds": '17',
                "away_team_score": '60',
                "away_team_goals": '9',
                "away_team_behinds": '6',
                "venue": 'Marvel Stadium',
                "match_date": 'Thursday 31 March 2022 • 7:20 PM (GMT+11)'
    }
    assert scrape_data.get_info_from_match_soup(soup) == match_information

