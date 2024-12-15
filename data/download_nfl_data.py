import pandas as pd
import nfl_data_py as nfl
import numpy as np
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_nfl_data():
    """Download and prepare NFL game data"""
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(__file__), exist_ok=True)

    # Calculate date range (last 5 seasons)
    end_year = datetime.now().year
    start_year = end_year - 5

    logger.info(f"Downloading NFL game data from {start_year} to {end_year}...")

    # Download game data
    games = nfl.import_schedules([year for year in range(start_year, end_year + 1)])

    # Filter for completed regular season and playoff games
    games = games[
        (games["game_type"].isin(["REG", "POST"]))
        & (~games["home_score"].isna())
        & (~games["away_score"].isna())  # Game is completed if score exists
    ].copy()

    # Calculate team statistics (rolling averages)
    def calculate_team_stats(group):
        group["win_pct"] = group["result"].expanding().mean()
        group["points_per_game"] = group["points"].expanding().mean()
        group["points_allowed_per_game"] = group["opp_points"].expanding().mean()
        return group

    # Calculate stats for each team
    team_stats = []
    for team in games["home_team"].unique():
        team_games = games[
            (games["home_team"] == team) | (games["away_team"] == team)
        ].sort_values("gameday")

        # Add result column (1 for win, 0 for loss)
        team_games["result"] = (
            (team_games["home_team"] == team)
            & (team_games["home_score"] > team_games["away_score"])
        ) | (
            (team_games["away_team"] == team)
            & (team_games["away_score"] > team_games["home_score"])
        )

        # Add points columns
        team_games["points"] = np.where(
            team_games["home_team"] == team,
            team_games["home_score"],
            team_games["away_score"],
        )
        team_games["opp_points"] = np.where(
            team_games["home_team"] == team,
            team_games["away_score"],
            team_games["home_score"],
        )

        # Calculate rolling stats
        team_stats.append(calculate_team_stats(team_games))

    team_stats_df = pd.concat(team_stats)

    # Prepare final dataset
    final_games = []
    for _, game in games.iterrows():
        # Get team stats before the game
        home_stats = (
            team_stats_df[
                (team_stats_df.index < game.name)
                & (
                    (team_stats_df["home_team"] == game["home_team"])
                    | (team_stats_df["away_team"] == game["home_team"])
                )
            ].iloc[-1]
            if not team_stats_df[
                (team_stats_df.index < game.name)
                & (
                    (team_stats_df["home_team"] == game["home_team"])
                    | (team_stats_df["away_team"] == game["home_team"])
                )
            ].empty
            else pd.Series(
                {"win_pct": 0.5, "points_per_game": 20, "points_allowed_per_game": 20}
            )
        )

        away_stats = (
            team_stats_df[
                (team_stats_df.index < game.name)
                & (
                    (team_stats_df["home_team"] == game["away_team"])
                    | (team_stats_df["away_team"] == game["away_team"])
                )
            ].iloc[-1]
            if not team_stats_df[
                (team_stats_df.index < game.name)
                & (
                    (team_stats_df["home_team"] == game["away_team"])
                    | (team_stats_df["away_team"] == game["away_team"])
                )
            ].empty
            else pd.Series(
                {"win_pct": 0.5, "points_per_game": 20, "points_allowed_per_game": 20}
            )
        )

        # Get stadium coordinates (if available)
        stadium_lat = game.get("stadium_latitude", 0)
        stadium_lon = game.get("stadium_longitude", 0)

        final_games.append(
            {
                "game_datetime": game["gameday"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_team_score": game["home_score"],
                "away_team_score": game["away_score"],
                "home_team_win_pct": home_stats["win_pct"],
                "away_team_win_pct": away_stats["win_pct"],
                "home_team_points_per_game": home_stats["points_per_game"],
                "away_team_points_per_game": away_stats["points_per_game"],
                "home_team_points_allowed_per_game": home_stats[
                    "points_allowed_per_game"
                ],
                "away_team_points_allowed_per_game": away_stats[
                    "points_allowed_per_game"
                ],
                "stadium_latitude": stadium_lat,
                "stadium_longitude": stadium_lon,
            }
        )

    final_df = pd.DataFrame(final_games)

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "nfl_games.csv")
    final_df.to_csv(output_path, index=False)
    logger.info(f"Saved NFL game data to {output_path}")


if __name__ == "__main__":
    download_nfl_data()
