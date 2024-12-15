"""Check available historical NFL data."""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_available_data():
    """Check what historical data is available through nfl_data_py."""
    try:
        # Check available seasons for schedules
        logger.info("Fetching schedules for last 5 seasons...")
        current_year = 2024
        seasons = list(range(current_year - 4, current_year + 1))  # Last 5 seasons
        schedules = nfl.import_schedules(seasons)

        logger.info(f"\nSchedule Data Available:")
        logger.info(f"Total games: {len(schedules)}")
        logger.info(f"Seasons: {schedules['season'].unique()}")
        logger.info(f"Available columns: {schedules.columns.tolist()}")

        # Check if we have stadium/location data
        stadium_cols = [
            col
            for col in schedules.columns
            if "stadium" in col.lower() or "location" in col.lower()
        ]
        logger.info(f"\nStadium/Location columns: {stadium_cols}")

        # Sample game data to show available information
        sample_game = schedules.iloc[0]
        logger.info("\nSample Game Data:")
        for col in schedules.columns:
            logger.info(f"{col}: {sample_game[col]}")

        # Check team info
        logger.info("\nFetching team information...")
        teams = nfl.import_team_desc()
        logger.info(f"Team Info Available:")
        logger.info(f"Total teams: {len(teams)}")
        logger.info(f"Team info columns: {teams.columns.tolist()}")

        # Check player stats
        logger.info("\nFetching player stats for last season...")
        player_stats = nfl.import_seasonal_data([2023])
        logger.info(f"Player Stats Available:")
        logger.info(f"Total player records: {len(player_stats)}")
        logger.info(f"Player stats columns: {player_stats.columns.tolist()}")

        # Check weekly data
        logger.info("\nFetching weekly data for last season...")
        weekly_data = nfl.import_weekly_data([2023])
        logger.info(f"Weekly Data Available:")
        logger.info(f"Total weekly records: {len(weekly_data)}")
        logger.info(f"Weekly data columns: {weekly_data.columns.tolist()}")

    except Exception as e:
        logger.error(f"Error checking available data: {str(e)}")


if __name__ == "__main__":
    check_available_data()
