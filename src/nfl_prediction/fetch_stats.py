import nfl_data_py as nfl
import pandas as pd

# Fetch team stats for specified seasons
seasons = [2018, 2019, 2020, 2021, 2022, 2023]

# Get player stats and rosters to map to teams
stats = nfl.import_seasonal_data(seasons)
rosters = nfl.import_seasonal_rosters(seasons)

# Merge with rosters to get team information
stats = pd.merge(
    stats,
    rosters[["player_id", "season", "team"]],
    on=["player_id", "season"],
    how="left",
)

# Aggregate stats by team and season
team_stats = (
    stats.groupby(["team", "season"])
    .agg(
        {
            "passing_yards": "sum",
            "passing_tds": "sum",
            "interceptions": "sum",
            "rushing_yards": "sum",
            "rushing_tds": "sum",
            "rushing_fumbles_lost": "sum",
            "receiving_yards": "sum",
            "receiving_tds": "sum",
            "games": "max",  # Use max since this should be same for all players on a team
        }
    )
    .reset_index()
)

# Calculate per-game stats
team_stats["total_yards_per_game"] = (
    team_stats["passing_yards"] + team_stats["rushing_yards"]
) / team_stats["games"]
team_stats["points_per_game"] = (
    (
        team_stats["passing_tds"]
        + team_stats["rushing_tds"]
        + team_stats["receiving_tds"]
    )
    * 6
) / team_stats["games"]
team_stats["turnover_ratio"] = (
    -(team_stats["interceptions"] + team_stats["rushing_fumbles_lost"])
    / team_stats["games"]
)

# Display information about the data
print("Available columns:")
print(team_stats.columns.tolist())
print("\nSample data:")
print(team_stats.head())

# Save to CSV for inspection
team_stats.to_csv("team_stats_processed.csv", index=False)
print("\nData saved to team_stats_processed.csv")
