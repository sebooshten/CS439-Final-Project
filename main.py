import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

games = pd.read_csv('C:/Users/seb/Documents/CS439/CS439_Final_Project/csgo_games.csv')
# sort data by increasing rating and then plot some key data points and see if theres any correlation

# dont care about t1_world_rank, t1_h2h_win_perc
drops = ['t1_world_rank', 't2_world_rank', 't1_h2h_win_perc', 't2_h2h_win_perc']
games = games.drop(columns=drops)

# Tidy the data and separate into a different table to analyze stats per player
player_stats = [
    "rating", "impact", "kdr", "dmr", "kpr", "apr", "dpr", "spr",
    "opk_ratio", "opk_rating", "multikill_perc", "clutch_win_perc"]

# restructure data to be team 1, team 2, players 1 2 3 4 5, and then the stats
players = pd.melt(
    games,
    id_vars=["match_date"],
    value_vars=[f"t{team}_player{i}_{stat}" for team in [1, 2] for i in range(1, 6) for stat in player_stats],
    var_name="player",
    value_name="value")

players["team"] = players["player"].str.extract(r"t(\d)_")[0].astype(int)
players["player_number"] = players["player"].str.extract(r"player(\d)_")[0].astype(int)
players["stat"] = players["player"].str.replace(r"t\d_player\d_", "", regex=True)

players = players.pivot_table(
    index=["match_date", "team", "player_number"],
    columns="stat",
    values="value").reset_index()

# Split based on teams if needed
team_1 = players[players["team"] == 1].drop(columns='team')
team_2 = players[players["team"] == 2].drop(columns='team')


# At this point we have all the stats that we want to see, and organized per player, so we can clearly compare rating vs performance
# Heat map provides correlation between stats

def plotCorrelation_between_Stats(df):
    stats_correlation = df.drop(columns=['match_date', 'player_number', 'team']).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(stats_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Between Stats")
    plt.tight_layout()
    plt.show()


plotCorrelation_between_Stats(players)


def plotRating_vs_Stats(df):
    stats = [stat for stat in player_stats if stat != "rating"]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(14, 12))
    axes = axes.flatten()

    for i, stat in enumerate(stats):
        sns.regplot(x="rating", y=stat, data=df, ax=axes[i], line_kws={'color': 'black'})

        axes[i].set_title(stat.upper() + " vs Rating", fontsize=10, y=0.98)
        axes[i].set_xlabel('Rating', fontsize=9)
        axes[i].set_ylabel(stat, fontsize=9)

    plt.tight_layout()
    plt.show()


plotRating_vs_Stats(players)


def plotSkill_Grouping(df):
    df['skill'] = pd.cut(df['rating'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    stats = [stat for stat in player_stats if stat != "rating"]
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 12))
    axes = axes.flatten()

    for i, stat in enumerate(stats):
        sns.boxplot(x='skill', y=stat, data=df, ax=axes[i])

        axes[i].set_title(stat.upper() + " by Rating Category", fontsize=10)
        axes[i].set_xlabel('Rating Category', fontsize=9)
        axes[i].set_ylabel(stat, fontsize=9)
        axes[i].tick_params(axis='x')

    plt.tight_layout()
    plt.show()


plotSkill_Grouping(players)

# Also we can categorize players into roles based on high values of certain stats
# Star player, Entry Fragger, Support, Flex (Well Rounded)
playerRoles = {
    "Entry Fragger": ["opk_ratio", "opk_rating"],
    "Support": ['apr', 'spr'],
    "Flex": ['kdr', 'apr'],
    "Star": ['multikill_perc', 'kpr']}


# Entry Fragger should have higher stats in the opening kill categories
# Support should have higher stats in the assists and saving teamates
# Flex should have a higher avg in kdr and apr together
# Star player will have a lot of multi kills and kills per round

def assignRoles(df):
    roleStats = []
    for stat in playerRoles.values():
        roleStats.extend(stat)
    roleStats = list(set(roleStats))

    scaled = StandardScaler().fit_transform(df[roleStats])
    scaledDf = pd.DataFrame(scaled, columns=roleStats, index=df.index)

    roles = []
    for index, _ in df.iterrows():
        scores = {}
        for role, stat in playerRoles.items():
            scores[role] = scaledDf.loc[index, stat].mean()

        topRole = max(scores.items(), key=lambda x: x[1])[0]
        roles.append(topRole)

    df = df.copy()
    df['role'] = roles
    return df


newPlayers = assignRoles(players);


def plotRoles_vs_Rating(df):
    plt.figure(figsize=(12, 10))
    sns.boxplot(x='role', y='rating', data=df)
    plt.title('Roles vs Rating')
    plt.tight_layout()
    plt.show()

    avgRoles = df.groupby('role')['rating'].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(x=avgRoles.index, y=avgRoles.values)
    plt.title('Average Rating by Role')
    plt.tight_layout()
    plt.show()


plotRoles_vs_Rating(newPlayers)
