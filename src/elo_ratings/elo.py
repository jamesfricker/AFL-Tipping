import pandas as pd


def elo_probability(rating1, rating2):
    Q1 = 10 ** (rating1 / 400)
    Q2 = 10 ** (rating2 / 400)
    E1 = Q1 / (Q1 + Q2)
    E2 = Q2 / (Q1 + Q2)
    return E1, E2


def margin_factor(margin):
    # gpt4 generated this function
    return ((margin + 3) ** 0.8) / 22


def update_elo_ratings(
    home_team_elo, away_team_elo, home_team_score, away_team_score, K=32
):
    margin = abs(home_team_score - away_team_score)
    mf = margin_factor(margin)

    if home_team_score > away_team_score:
        home_team_result, away_team_result = 1, 0
    elif home_team_score < away_team_score:
        home_team_result, away_team_result = 0, 1
    else:
        home_team_result, away_team_result = 0.5, 0.5

    expected_home_team, expected_away_team = elo_probability(
        home_team_elo, away_team_elo
    )
    home_team_elo += K * mf * (home_team_result - expected_home_team)
    away_team_elo += K * mf * (away_team_result - expected_away_team)

    return home_team_elo, away_team_elo


def end_of_season_adjustment(elo_ratings):
    # at the end of the season
    # revert elos 1/3 of the way to the base elo
    base_elo = 1500
    for team in elo_ratings:
        elo_ratings[team] = base_elo + (elo_ratings[team] - base_elo) * 2 / 3
    return elo_ratings


def compute_elo_ratings(games):
    elo_ratings = {}
    season = ""
    for _, game in games.iterrows():
        home_team = game["home_team_name"]
        away_team = game["away_team_name"]
        home_team_score = game["home_team_score"]
        away_team_score = game["away_team_score"]
        if home_team not in elo_ratings:
            elo_ratings[home_team] = 1500
        if away_team not in elo_ratings:
            elo_ratings[away_team] = 1500

        if game["year"] != season:
            # end of season, adjust elo ratings
            elo_ratings = end_of_season_adjustment(elo_ratings)

        season = game["year"]

        home_team_elo, away_team_elo = elo_ratings[home_team], elo_ratings[away_team]
        new_home_team_elo, new_away_team_elo = update_elo_ratings(
            home_team_elo, away_team_elo, home_team_score, away_team_score
        )

        elo_ratings[home_team], elo_ratings[away_team] = (
            new_home_team_elo,
            new_away_team_elo,
        )

    return elo_ratings


games = pd.read_csv("../outputs/afl_data.csv")
# Sort the DataFrame by year and round
games = games.sort_values(by=["year", "round"], ascending=[True, True])

# Reset the index of the DataFrame
games = games.reset_index(drop=True)

elo_ratings = compute_elo_ratings(games)
# sort by key
elo_ratings = {k: v for k, v in sorted(elo_ratings.items(), key=lambda item: item[1])}
# reverse
elo_ratings = dict(reversed(list(elo_ratings.items())))
print(elo_ratings)
