from scrape_afl import scrape_data as SCRAPE_DATA
import elo_ratings.calculate_elo_rating as CALCULATE_ELO

import pandas as pd
import numpy as np

ROUND_TO_PREDICT = 7

def read_elo_csv():
    df_elo = pd.read_csv("elo_ratings/elo_ratings.csv")
    return df_elo

def get_predict_round_matches():
    return SCRAPE_DATA.get_gameweek_matches(ROUND_TO_PREDICT)

def get_team_elo(team_name,team_elos):
    return team_elos.loc[team_elos['team'] == team_name]['elo'].values[0]

def predict_matches():
    matches = get_predict_round_matches()
    team_elos = read_elo_csv()
    match_predictions = []
    for match in matches:
        home_team_elo = get_team_elo(match['home_team_name'],team_elos)
        away_team_elo = get_team_elo(match['away_team_name'],team_elos)
        home_win_prob = CALCULATE_ELO.expected_margin(home_team_elo, away_team_elo)
        match_prediction = {
            'home_team_name': match['home_team_name'],
            'away_team_name': match['away_team_name'],
            'home_win_prob': home_win_prob
        }
        match_predictions.append(match_prediction)
    return match_predictions

if __name__ == '__main__':
    print(pd.DataFrame(predict_matches()))