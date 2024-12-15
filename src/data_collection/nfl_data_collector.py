"""NFL Data Collection Module."""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import nfl_data_py as nfl
import time
from typing import Dict, List, Optional, Union, Tuple
import pytz
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLDataCollector:
    """Collects NFL game data, team stats, and player information."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the NFL data collector."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".nfl_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Headers for web scraping
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        self.team_timezones = {
            "ARI": "US/Arizona",  # Arizona doesn't observe DST
            "ATL": "US/Eastern",
            "BAL": "US/Eastern",
            "BUF": "US/Eastern",
            "CAR": "US/Eastern",
            "CHI": "US/Central",
            "CIN": "US/Eastern",
            "CLE": "US/Eastern",
            "DAL": "US/Central",
            "DEN": "US/Mountain",
            "DET": "US/Eastern",
            "GB": "US/Central",
            "HOU": "US/Central",
            "IND": "US/Eastern",
            "JAX": "US/Eastern",
            "KC": "US/Central",
            "LAC": "US/Pacific",
            "LA": "US/Pacific",
            "LV": "US/Pacific",
            "MIA": "US/Eastern",
            "MIN": "US/Central",
            "NE": "US/Eastern",
            "NO": "US/Central",
            "NYG": "US/Eastern",
            "NYJ": "US/Eastern",
            "OAK": "US/Pacific",  # Historical
            "PHI": "US/Eastern",
            "PIT": "US/Eastern",
            "SEA": "US/Pacific",
            "SF": "US/Pacific",
            "TB": "US/Eastern",
            "TEN": "US/Central",
            "WAS": "US/Eastern",
            "SD": "US/Pacific",  # Historical
            "STL": "US/Central",  # Historical
        }

    def _safe_get_data(
        self, data_fn, *args, **kwargs
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Safely fetch data from nfl_data_py with error handling."""
        try:
            df = data_fn(*args, **kwargs)
            if df is None or df.empty:
                return None, "No data returned"
            return df, None
        except Exception as e:
            error_msg = f"Error fetching data: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def get_local_game_time(self, game_time: str, home_team: str) -> datetime:
        """Convert game time to local time based on home team's timezone."""
        try:
            # Get home team's timezone
            team_tz = pytz.timezone(self.team_timezones.get(home_team, "US/Eastern"))

            # Parse the time string
            if isinstance(game_time, str):
                if len(game_time) == 5:  # Format: "HH:MM"
                    hour, minute = map(int, game_time.split(":"))
                    # Create a datetime object for today with the given time
                    naive_dt = datetime.now().replace(
                        hour=hour, minute=minute, second=0, microsecond=0
                    )
                    # Localize to team's timezone
                    game_time = team_tz.localize(naive_dt)
                else:
                    # Try parsing as full datetime string with timezone
                    try:
                        game_time = datetime.strptime(game_time, "%Y-%m-%dT%H:%M%z")
                        game_time = game_time.astimezone(team_tz)
                    except ValueError:
                        # Try parsing without timezone and assume team's timezone
                        naive_dt = datetime.strptime(game_time, "%Y-%m-%dT%H:%M")
                        game_time = team_tz.localize(naive_dt)
            elif isinstance(game_time, pd.Timestamp):
                # Convert pandas timestamp to datetime
                naive_dt = game_time.to_pydatetime()
                if naive_dt.tzinfo is None:
                    game_time = team_tz.localize(naive_dt)
                else:
                    game_time = naive_dt.astimezone(team_tz)

            return game_time

        except Exception as e:
            logger.error(f"Error converting game time: {str(e)}")
            return None

    def get_season_games(self, year: int) -> pd.DataFrame:
        """Fetch all games for a given season."""
        try:
            games_data = []

            # Fetch regular season games
            url = (
                f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
            )
            params = {
                "limit": 1000,
                "dates": f"{year}0901-{year+1}0228",  # Sept 1 to Feb 28
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            for event in data.get("events", []):
                game_data = {}
                competition = event["competitions"][0]

                # Get teams
                home_team = None
                away_team = None
                for team in competition["competitors"]:
                    if team["homeAway"] == "home":
                        home_team = team["team"]["abbreviation"]
                        game_data["home_team"] = home_team
                        game_data["home_score"] = int(team.get("score", 0))
                    else:
                        away_team = team["team"]["abbreviation"]
                        game_data["away_team"] = away_team
                        game_data["away_score"] = int(team.get("score", 0))

                # Get game date and time
                game_date = competition["date"]
                try:
                    # Parse the full datetime string from ESPN
                    game_datetime = datetime.strptime(game_date, "%Y-%m-%dT%H:%M%z")
                    # Convert to home team's timezone
                    team_tz = pytz.timezone(
                        self.team_timezones.get(home_team, "US/Eastern")
                    )
                    local_time = game_datetime.astimezone(team_tz)

                    game_data["game_date"] = local_time.strftime("%Y-%m-%d")
                    game_data["game_time"] = local_time.strftime("%H:%M")
                    game_data["timezone"] = self.team_timezones.get(
                        home_team, "US/Eastern"
                    )
                except Exception as e:
                    logger.error(f"Error parsing game time: {str(e)}")
                    game_data["game_date"] = None
                    game_data["game_time"] = None
                    game_data["timezone"] = None

                game_data["season"] = year
                game_data["week"] = competition.get("week", {}).get("number", 0)
                game_data["game_id"] = event["id"]

                games_data.append(game_data)

            return pd.DataFrame(games_data)

        except Exception as e:
            logger.error(f"Error fetching season games: {str(e)}")
            return pd.DataFrame()

    def fetch_season_games(self, year: int) -> pd.DataFrame:
        """Fetch all games for a specific season using nfl_data_py."""
        logger.info(f"Fetching season games for {year}")

        try:
            # Get games data
            games_df = self.get_season_games(year)

            if not games_df.empty:
                # Add additional stats and features
                games_df["score_diff"] = games_df["home_score"].fillna(0) - games_df[
                    "away_score"
                ].fillna(0)
                games_df["is_dome"] = (
                    games_df["roof"].str.contains(
                        "dome|closed|retractable", case=False, na=False
                    )
                    if "roof" in games_df.columns
                    else False
                )
                games_df["is_primetime"] = games_df["game_time"].apply(
                    lambda x: x >= "20:00" if pd.notnull(x) else False
                )

            return games_df

        except Exception as e:
            logger.error(f"Error fetching season games: {str(e)}")
            return pd.DataFrame()

    def fetch_team_stats(self, team: str, year: int) -> Optional[pd.DataFrame]:
        """Fetch team statistics for a specific season."""
        logger.info(f"Fetching team stats for {team} {year}")

        try:
            # Get team schedule
            schedule_data, error = self._safe_get_data(nfl.import_schedules, [year])
            if error or schedule_data.empty:
                logger.warning(f"No schedule data found for {year}")
                return None

            # Convert game_id to string
            schedule_data["game_id"] = schedule_data["game_id"].astype(str)

            # Filter for regular season games
            schedule_data = schedule_data[
                schedule_data["game_type"].str.contains("REG", na=False)
            ]

            # Get home and away games
            home_games = schedule_data[schedule_data["home_team"] == team]
            away_games = schedule_data[schedule_data["away_team"] == team]

            if home_games.empty and away_games.empty:
                logger.warning(f"No games found for {team} in {year}")
                return None

            # Calculate stats
            stats = {}

            if not home_games.empty:
                home_stats = {
                    "home_games_played": len(home_games),
                    "home_points_scored": home_games["home_score"].mean(),
                    "home_points_allowed": home_games["away_score"].mean(),
                    "home_wins": home_games[
                        home_games["home_score"] > home_games["away_score"]
                    ].shape[0],
                    "home_losses": home_games[
                        home_games["home_score"] < home_games["away_score"]
                    ].shape[0],
                    "home_ties": home_games[
                        home_games["home_score"] == home_games["away_score"]
                    ].shape[0],
                }
                stats.update(home_stats)

            if not away_games.empty:
                away_stats = {
                    "away_games_played": len(away_games),
                    "away_points_scored": away_games["away_score"].mean(),
                    "away_points_allowed": away_games["home_score"].mean(),
                    "away_wins": away_games[
                        away_games["away_score"] > away_games["home_score"]
                    ].shape[0],
                    "away_losses": away_games[
                        away_games["away_score"] < away_games["home_score"]
                    ].shape[0],
                    "away_ties": away_games[
                        away_games["away_score"] == away_games["home_score"]
                    ].shape[0],
                }
                stats.update(away_stats)

            # Calculate overall stats
            total_games = stats.get("home_games_played", 0) + stats.get(
                "away_games_played", 0
            )
            if total_games > 0:
                total_points_scored = (
                    stats.get("home_points_scored", 0)
                    * stats.get("home_games_played", 0)
                    + stats.get("away_points_scored", 0)
                    * stats.get("away_games_played", 0)
                ) / total_games

                total_points_allowed = (
                    stats.get("home_points_allowed", 0)
                    * stats.get("home_games_played", 0)
                    + stats.get("away_points_allowed", 0)
                    * stats.get("away_games_played", 0)
                ) / total_games

                total_wins = stats.get("home_wins", 0) + stats.get("away_wins", 0)
                total_losses = stats.get("home_losses", 0) + stats.get("away_losses", 0)
                total_ties = stats.get("home_ties", 0) + stats.get("away_ties", 0)

                overall_stats = {
                    "total_games": total_games,
                    "avg_points_scored": total_points_scored,
                    "avg_points_allowed": total_points_allowed,
                    "total_wins": total_wins,
                    "total_losses": total_losses,
                    "total_ties": total_ties,
                    "win_percentage": (total_wins + 0.5 * total_ties) / total_games,
                }
                stats.update(overall_stats)

            return pd.DataFrame([stats])

        except Exception as e:
            logger.error(f"Error fetching team stats: {str(e)}")
            return None


def main():
    """Main function to collect NFL data."""
    logger.info("Starting data collection")

    try:
        # Initialize collector
        collector = NFLDataCollector()

        # Create data directory if it doesn't exist
        data_dir = Path("../data")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Collect game data for recent seasons
        all_games = []
        team_stats_records = []

        for year in range(2016, 2024):
            logger.info(f"Fetching season games for {year}")
            games = collector.fetch_season_games(year)
            if not games.empty:
                all_games.append(games)

            # Fetch team stats for each team
            for team in collector.team_timezones.keys():
                if team not in ["OAK", "SD", "STL"]:  # Skip historical teams
                    logger.info(f"Fetching team stats for {team} {year}")
                    team_stats = collector.fetch_team_stats(team, year)
                    if team_stats is not None:
                        team_stats_records.append(team_stats)

        if all_games:
            # Combine all seasons
            combined_games = pd.concat(all_games, ignore_index=True)

            # Save games to CSV
            output_path = "data/enhanced_historical_games.csv"
            logger.info(f"Saving {len(combined_games)} games to {output_path}")
            combined_games.to_csv(output_path, index=False)

            # Save team stats to CSV
            if team_stats_records:
                team_stats_df = pd.DataFrame(team_stats_records)
                team_stats_path = "data/team_stats.csv"
                logger.info(
                    f"Saving {len(team_stats_records)} team stats records to {team_stats_path}"
                )
                team_stats_df.to_csv(team_stats_path, index=False)

            logger.info("Data collection complete!")
        else:
            logger.error("No game data collected")

    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")


if __name__ == "__main__":
    main()
