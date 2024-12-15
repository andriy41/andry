"""Check current NFL season status."""

import nfl_data_py as nfl
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_current_week_games():
    """Fetch current week games using nfl_data_py."""
    try:
        # Get all games for 2024 season
        logger.info("Fetching 2024 season schedule...")
        schedule = nfl.import_schedules([2024])

        if schedule is None or schedule.empty:
            logger.error("No schedule data found for 2024 season")
            return []

        # Filter for week 14 games
        week_14_games = schedule[schedule["week"] == 14].copy()
        logger.info(f"Found {len(week_14_games)} Week 14 games")

        if week_14_games.empty:
            logger.warning("No Week 14 games found")
            return []

        games = []
        for _, game in week_14_games.iterrows():
            game_dict = {
                "team1": game["home_team"],
                "team2": game["away_team"],
                "status": "Final"
                if pd.notna(game["result"])
                else f"Scheduled for {game['gameday']}",
            }

            # Add scores if game is completed
            if pd.notna(game["result"]):
                if game["result"] > 0:  # Home team won
                    game_dict.update(
                        {
                            "team1_score": str(game["home_score"]),
                            "team2_score": str(game["away_score"]),
                        }
                    )
                else:  # Away team won
                    game_dict.update(
                        {
                            "team1_score": str(game["home_score"]),
                            "team2_score": str(game["away_score"]),
                        }
                    )

            games.append(game_dict)
            logger.info(f"Parsed game: {game_dict['team1']} vs {game_dict['team2']}")

        return games

    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return []


def check_season_status():
    """Check the current status of the NFL season."""
    logger.info("Fetching Week 14 games (2024 season)...")

    games = fetch_current_week_games()

    if games:
        logger.info("\nWeek 14 Games (2024 Season):")
        for game in games:
            if game["status"].lower() == "final":
                logger.info(
                    f"[Final] {game['team1']} {game['team1_score']} - "
                    f"{game['team2']} {game['team2_score']}"
                )
            else:
                logger.info(f"{game['status']}: {game['team1']} vs {game['team2']}")
    else:
        logger.info("No games found or error fetching games")


if __name__ == "__main__":
    check_season_status()
