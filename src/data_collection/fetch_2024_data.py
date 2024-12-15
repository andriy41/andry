"""Script to fetch and analyze 2024 NFL season data."""

import pandas as pd
import nfl_data_py as nfl
from nfl_data_collector import NFLDataCollector
from datetime import datetime
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_2024_data():
    """Fetch comprehensive data for the 2024 NFL season."""
    collector = NFLDataCollector()
    data_dir = Path("data/2024_season")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Get 2024 schedule
        logger.info("Fetching 2024 schedule...")
        schedule_2024 = nfl.import_schedules([2024])
        schedule_2024.to_csv(data_dir / "schedule_2024.csv", index=False)
        logger.info(f"Saved schedule with {len(schedule_2024)} games")

        # 2. Get team stats
        logger.info("Fetching team stats...")
        team_stats = {}
        for team in collector.team_name_map.keys():
            try:
                # Using PFR data since nfl_data_py doesn't have 2024 team stats yet
                stats = collector._fetch_pfr_team_stats(team, 2024)
                team_stats[team] = stats
                logger.info(f"Fetched stats for {team}")
            except Exception as e:
                logger.error(f"Error fetching stats for {team}: {str(e)}")

        pd.DataFrame.from_dict(team_stats, orient="index").to_csv(
            data_dir / "team_stats_2024.csv"
        )

        # 3. Get current player data
        logger.info("Fetching player data...")
        players = nfl.import_players()
        players_2024 = players[players["status"] == "ACT"]  # Only active players
        players_2024.to_csv(data_dir / "players_2024.csv", index=False)

        # 4. Get weekly data (if available)
        logger.info("Fetching weekly data...")
        try:
            weekly_data = nfl.import_weekly_data([2024])
            weekly_data.to_csv(data_dir / "weekly_data_2024.csv", index=False)
        except Exception as e:
            logger.warning(f"Weekly data not available yet: {str(e)}")

        # 5. Get injuries
        logger.info("Fetching injury reports...")
        injuries = collector.fetch_injury_reports()
        injuries.to_csv(data_dir / "injuries_current.csv", index=False)

        # 6. Get play-by-play data (if available)
        logger.info("Fetching play-by-play data...")
        try:
            pbp_data = nfl.import_pbp_data([2024])
            if not pbp_data.empty:
                pbp_data.to_csv(data_dir / "pbp_2024.csv", index=False)
        except Exception as e:
            logger.warning(f"Play-by-play data not available yet: {str(e)}")

        logger.info("Data collection complete!")
        return True

    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        return False


def analyze_2024_data():
    """Analyze the collected 2024 NFL season data."""
    data_dir = Path("data/2024_season")

    try:
        # Load available data
        analysis = {}

        # Schedule analysis
        if (data_dir / "schedule_2024.csv").exists():
            schedule = pd.read_csv(data_dir / "schedule_2024.csv")
            analysis["total_games"] = len(schedule)
            analysis["games_completed"] = (
                len(schedule[schedule["completed"] == True])
                if "completed" in schedule.columns
                else 0
            )
            analysis["games_remaining"] = (
                analysis["total_games"] - analysis["games_completed"]
            )

        # Team stats analysis
        if (data_dir / "team_stats_2024.csv").exists():
            team_stats = pd.read_csv(data_dir / "team_stats_2024.csv")
            analysis["teams_with_data"] = len(team_stats)

        # Player analysis
        if (data_dir / "players_2024.csv").exists():
            players = pd.read_csv(data_dir / "players_2024.csv")
            analysis["total_players"] = len(players)
            if "position" in players.columns:
                analysis["players_by_position"] = (
                    players["position"].value_counts().to_dict()
                )

        # Injury analysis
        if (data_dir / "injuries_current.csv").exists():
            injuries = pd.read_csv(data_dir / "injuries_current.csv")
            analysis["current_injuries"] = len(injuries)
            if "status" in injuries.columns:
                analysis["injuries_by_status"] = (
                    injuries["status"].value_counts().to_dict()
                )

        # Save analysis
        pd.DataFrame([analysis]).to_csv(data_dir / "analysis_summary.csv", index=False)

        logger.info("Analysis complete!")
        logger.info(f"Summary: {analysis}")
        return analysis

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return None


if __name__ == "__main__":
    logger.info("Starting 2024 NFL data collection and analysis...")

    if fetch_2024_data():
        analysis = analyze_2024_data()
        if analysis:
            logger.info("Process completed successfully!")
        else:
            logger.error("Analysis failed!")
    else:
        logger.error("Data collection failed!")
