from scrape_afl import scrape_data
from bs4 import BeautifulSoup
import requests
import pytest

def test_scrape_return():
    f = open('tests/test_data/round3_2022.html')
    content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    assert len(scrape_data.get_all_links_from_afl_homepage(soup))>0

