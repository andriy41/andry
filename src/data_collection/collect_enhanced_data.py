"""Module for collecting enhanced NFL data including astrological factors."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import json
from pathlib import Path
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import warnings
import traceback

from .nfl_data_collector import NFLDataCollector
from src.models.vedic_astrology.nfl_vedic_calculator import NFLVedicCalculator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_vedic_features(
    df: pd.DataFrame, vedic_calc: NFLVedicCalculator
) -> pd.DataFrame:
    """Add Vedic astrological features to the dataframe."""

    # Initialize columns for Vedic features
    vedic_features = []

    # Initialize data collector for timezone mapping
    collector = NFLDataCollector()

    for _, row in df.iterrows():
        try:
            # Get home team's timezone from the mapping
            home_team = row["home_team"]
            timezone = collector.team_timezones.get(home_team, "US/Eastern")

            # Convert team abbreviations to full names
            home_team_full = vedic_calc.team_aliases.get(home_team, home_team)
            away_team_full = vedic_calc.team_aliases.get(
                row["away_team"], row["away_team"]
            )

            features = vedic_calc.calculate_game_features(
                game_date=row["game_date"],
                game_time=row["game_time"],
                timezone=timezone,
                home_team=home_team_full,
                away_team=away_team_full,
            )
            vedic_features.append(features)

        except Exception as e:
            logger.error(
                f"Error calculating Vedic features for game {row['game_date']} {home_team} vs {row['away_team']}: {str(e)}"
            )
            # Add default features
            vedic_features.append(
                {
                    "moon_phase": 0.5,
                    "home_planet_strength": 0.5,
                    "away_planet_strength": 0.5,
                    "home_zodiac_strength": 0.5,
                    "away_zodiac_strength": 0.5,
                    "beneficial_aspects": 0,
                    "malefic_aspects": 0,
                    "home_nakshatra_score": 0.5,
                    "away_nakshatra_score": 0.5,
                    "planetary_alignment": 0.5,
                }
            )

    # Convert list of dictionaries to DataFrame
    vedic_df = pd.DataFrame(vedic_features)

    # Combine with original DataFrame
    return pd.concat([df, vedic_df], axis=1)


def collect_data(start_year: int = 2016, end_year: int = 2023) -> None:
    """Collect NFL data including astrological features for specified years."""

    # Initialize collectors
    collector = NFLDataCollector()
    vedic_calc = NFLVedicCalculator()

    all_games = []
    all_team_stats = []

    for year in range(start_year, end_year + 1):
        try:
            # Fetch season games
            season_games = collector.fetch_season_games(year)
            if not season_games.empty:
                # Add local game times
                season_games["game_date"] = pd.to_datetime(
                    season_games["game_date"]
                ).dt.date
                # Handle both time formats (HH:MM and HH:MM:SS)
                season_games["game_time"] = pd.to_datetime(
                    season_games["game_time"]
                ).dt.strftime("%H:%M")

                # Add Vedic features (timezone will be determined per game based on home team)
                season_games = add_vedic_features(season_games, vedic_calc)
                all_games.append(season_games)

            # Fetch team stats
            if not season_games.empty:
                home_teams = pd.Series(season_games["home_team"].unique())
                away_teams = pd.Series(season_games["away_team"].unique())
                teams = pd.concat([home_teams, away_teams]).unique()

                for team in teams:
                    team_stats = collector.fetch_team_stats(team, year)
                    if team_stats is not None and not team_stats.empty:
                        team_stats["year"] = year
                        team_stats["team"] = team
                        all_team_stats.append(team_stats)

        except Exception as e:
            logger.error(f"Error processing year {year}: {str(e)}")
            continue

    try:
        # Combine and save all games data
        if all_games:
            games_df = pd.concat(all_games, ignore_index=True)

            # Downcasting floats
            float_cols = games_df.select_dtypes(include=["float64"]).columns
            games_df[float_cols] = games_df[float_cols].astype("float32")

            # Create data directory if it doesn't exist
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            games_df.to_csv(data_dir / "enhanced_historical_games.csv", index=False)
            logger.info(f"Saved {len(games_df)} games to enhanced_historical_games.csv")

        # Combine and save team stats
        if all_team_stats:
            stats_df = pd.concat(all_team_stats, ignore_index=True)
            stats_df.to_csv(data_dir / "team_stats.csv", index=False)
            logger.info(f"Saved {len(stats_df)} team stats records to team_stats.csv")

    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")


if __name__ == "__main__":
    collect_data()
