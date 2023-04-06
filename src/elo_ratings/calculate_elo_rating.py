import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_games = pd.read_csv("games_current.csv")
df_players = pd.read_csv("players.csv")
df_stats = pd.read_csv("stats.csv")

#print(df_games.columns)

# margin method for elo rating
# https://sci-hub.st/10.1016/j.ijforecast.2020.01.006

# we need to get 
# team A
# team B
# winning Team
# these will be used to create the ELO ratings

# create this as a twitter bot?
# https://twitter.com/thecruncherau
# https://twitter.com/AFLxScore
# https://twitter.com/AflLadder
# https://twitter.com/AFLLab
# https://twitter.com/SquiggleAFL
# https://twitter.com/AFLalytics

# there is an AFL algorithm competition
# https://squiggle.com.au/leaderboard/

def find_winner(x):
    if x.homeTeamScore > x.awayTeamScore:
        return x.homeTeam
    elif x.homeTeamScore < x.awayTeamScore:
        return x.awayTeam
    return 'NA'

# create a winning team column 
df_games['winningTeam'] = df_games.apply(lambda x: find_winner(x),axis=1)

# create a losing team columns
df_games['losingTeam'] = df_games.apply(lambda x: x.awayTeam if x.homeTeam==x.winningTeam else x.homeTeam ,axis=1)

# create a year_round_id column
df_games['yearRoundId'] = df_games.apply(lambda x: x.gameId[:-2],axis=1)

# create margin column
df_games['margin'] = df_games.apply(lambda x: abs(x.homeTeamScore - x.awayTeamScore),axis=1)

# convert date to a date object so we can sort
df_games['date'] = pd.to_datetime(df_games['date'])

# earliest games are now first
df_games = df_games.sort_values(by=['date'],ascending=True)

#print(df_games.columns)
teams = df_games['homeTeam'].unique()
teams.sort()
teams = teams.tolist()
teams.append('NA')
#print(teams.index('Adelaide'))
#print(df_games)
# now we can run our algorithm to get the ELO ratings
# ELO consists of 1. ESTIMATION 2. UPDATE

# from https://www.kaggle.com/code/kplauritzen/elo-ratings-in-python/notebook
# and https://www.kaggle.com/code/andreiavadanei/elo-predicting-against-dataset

alpha = 12
#sigma_1 = df_games['margin'].std() # 26.62
sigma_1 = 26
sigma_2 = 400

def update_elo(winner_elo, loser_elo,margin):
    """
    update the ELO ratings
    """
    # expected_win = expected_result(winner_elo, loser_elo)
    # change_in_elo = k_factor * (margin-expected_win)

    # use margin method
    expected_marg = expected_margin(winner_elo, loser_elo)
    l_step = 1.0/(1+alpha**((-margin)/sigma_1))
    change_in_elo = k_factor * (l_step-expected_marg)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a

def expected_margin(elo_a, elo_b):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    sigma_2 = 400
    expect_a = 1.0/(1+alpha**((elo_b - elo_a)/sigma_2))
    return expect_a

def update_end_of_season(elos):
    """Regression towards the mean
    
    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    diff_from_mean = elos - mean_elo
    elos -= diff_from_mean/3
    return elos

# these are also from this kaggle notebook
#default settings for elo
mean_elo = 1500
elo_width = 400
n_samples = 8000 #used for predicting
std_margin = df_games['margin'].std()
k_factor = 32 

df_games['w_elo_before_game'] = 0
df_games['w_elo_after_game'] = 0
df_games['l_elo_before_game'] = 0
df_games['l_elo_after_game'] = 0

elo_per_season = {}

# we add 1 for the NA team that happens when teams draw
num_teams = len(df_games['homeTeam'].unique()) + 1
current_elos = np.ones(shape=(num_teams)) * mean_elo

# add df for ELOs for each team
df_team_elos = pd.DataFrame(index=df_games['yearRoundId'].unique(), columns=range(num_teams))
df_team_elos.iloc[0, :] = current_elos

# where the magic happens
current_season = df_games.at[0, 'year']
current_round = df_games.at[0, 'round']

for row in df_games.itertuples():
    if row.year != current_season:
        # Check if we are starting a new season. 
        # Regress all ratings towards the mean
        current_elos = update_end_of_season(current_elos)
        # Write the beginning of new season ratings to a dict for later lookups.
        elo_per_season[row.year] = current_elos.copy()
        current_season = row.year

    idx = row.Index
    w_id = teams.index(row.winningTeam)
    l_id = teams.index(row.losingTeam)
    # Get current elos
    w_elo_before = current_elos[w_id]
    l_elo_before = current_elos[l_id]

    # Update on game results
    w_elo_after, l_elo_after = update_elo(w_elo_before, l_elo_before,row.margin)
        
    # Save updated elos
    df_games.at[idx, 'w_elo_before_game'] = w_elo_before
    df_games.at[idx, 'l_elo_before_game'] = l_elo_before
    df_games.at[idx, 'w_elo_after_game'] = w_elo_after
    df_games.at[idx, 'l_elo_after_game'] = l_elo_after
    current_elos[w_id] = w_elo_after
    current_elos[l_id] = l_elo_after

    # Save elos to team DataFrame
    round_id = row.yearRoundId
    df_team_elos.at[round_id, w_id] = w_elo_after
    df_team_elos.at[round_id, l_id] = l_elo_after
    

cols = df_games['homeTeam'].unique().tolist()
cols.sort()
cols.append("NA")
df_team_elos.columns = cols
df_team_elos
plt.plot(df_team_elos)
plt.legend(cols)
#plt.show()

final_elos = df_team_elos.drop("NA",axis=1).dropna().iloc[-1]
#print(final_elos)

team_names = [i for i in df_team_elos.columns if i != 'NA']
final_elos = [i for i in final_elos]

zipped = list(zip(team_names,final_elos))

elos_df = pd.DataFrame(zipped, columns=['team','elo'])
elos_df = elos_df.sort_values(by='elo',ascending=False)
elos_df.to_csv('elo_ratings/elo_ratings.csv')
#print(final_elos.sort_values(ascending=False))

n_samples = 1000
samples = df_games[df_games.year > 2015].sample(n_samples)
loss=0
expected_list = []
for row in samples.itertuples():
    w_elo = row.w_elo_before_game
    l_elo = row.l_elo_before_game
    w_expected = expected_margin(w_elo, l_elo)
    expected_list.append(w_expected)
    loss += np.log(w_expected)


print(sigma_1,sigma_2,loss/n_samples)

sns.displot(expected_list, kde=False, bins=20)
plt.xlabel('Elo Expected Wins for Actual Winner')
plt.ylabel('Counts')
#plt.show()

df_team_elos.fillna(method='ffill', inplace=True)
