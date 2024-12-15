"""
NFL Data Processor for feature engineering and data preparation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import json
from models.vedic_astrology.nfl_vedic_calculator import NFLVedicCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLDataProcessor:
    def __init__(self, lookback_weeks: int = 8):
        """
        Initialize the NFL data processor

        Args:
            lookback_weeks (int): Number of weeks to look back for rolling statistics
        """
        self.lookback_weeks = lookback_weeks
        self.scaler = StandardScaler()
        self.vedic_calculator = NFLVedicCalculator()
        self.team_code_mapping = {
            "WSH": "WAS",  # Washington team code change
            "OAK": "LV",  # Oakland Raiders to Las Vegas Raiders
            "STL": "LAR",  # St. Louis Rams to Los Angeles Rams
            "SD": "LAC",  # San Diego Chargers to Los Angeles Chargers
        }

    def load_data(self, input_file: str) -> pd.DataFrame:
        """Load and preprocess the raw data."""
        logger.info(f"Loading data from {input_file}")

        # Read the CSV file
        df = pd.read_csv(input_file)

        # Print column names for debugging
        logger.info(f"Available columns: {df.columns.tolist()}")

        # Filter out Pro Bowl and special games
        df = df[
            ~df["home_team"].isin(["AFC", "NFC"])
            & ~df["away_team"].isin(["AFC", "NFC"])
        ]

        # Convert date to datetime
        df["game_date"] = pd.to_datetime(df["game_date"])

        # Sort by date
        df = df.sort_values("game_date")

        # Extract season and week
        df["season"] = df["game_date"].dt.year
        df["week"] = df.groupby("season").cumcount() + 1

        # Add win/loss columns
        df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
        df["away_win"] = (df["away_score"] > df["home_score"]).astype(int)
        df["tie"] = (df["home_score"] == df["away_score"]).astype(int)

        return df

    def clean_numeric_data(self, value: str) -> float:
        """Clean numeric data that might be in string format."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                if "-" in value:
                    nums = value.split("-")
                    if len(nums) == 2:
                        try:
                            return float(nums[0]) / float(nums[1])
                        except (ValueError, ZeroDivisionError):
                            return 0.0
                elif "/" in value:
                    nums = value.split("/")
                    if len(nums) == 2:
                        try:
                            return float(nums[0]) / float(nums[1])
                        except (ValueError, ZeroDivisionError):
                            return 0.0
            return 0.0
        except Exception:
            return 0.0

    def create_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team statistics for both home and away teams."""
        logger.info("Creating team statistics")

        # First, get team information and game context
        team_info = df[
            [
                "game_date",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
            ]
        ]

        # Create team stats dictionary structure
        for idx, row in df.iterrows():
            try:
                # Home team stats
                home_stats = {
                    "points_per_game": row["home_score"],
                    "points_allowed_per_game": row["away_score"],
                    "total_yards": row.get(
                        "away_totalyards", 0
                    ),  # Using away team's total yards allowed
                    "passing_yards": row.get(
                        "away_netpassingyards", 0
                    ),  # Using away team's passing yards allowed
                    "rushing_yards": row.get(
                        "away_rush", 0
                    ),  # Using away team's rushing yards allowed
                    "first_downs": row.get(
                        "away_firstdowns", 0
                    ),  # Using away team's first downs allowed
                    "third_down_efficiency": self.clean_numeric_data(
                        row.get("away_thirddowneff", "0-0")
                    ),
                    "fourth_down_efficiency": self.clean_numeric_data(
                        row.get("away_fourthdowneff", "0-0")
                    ),
                    "total_plays": row.get("away_totaloffensiveplays", 0),
                    "yards_per_play": row.get("away_yardsperplay", 0),
                    "turnovers": row.get("away_turnovers", 0),
                    "time_of_possession": row.get("away_possessiontime", "0:00"),
                }

                # Away team stats
                away_stats = {
                    "points_per_game": row["away_score"],
                    "points_allowed_per_game": row["home_score"],
                    "total_yards": row.get("away_totalyards", 0),
                    "passing_yards": row.get("away_netpassingyards", 0),
                    "rushing_yards": row.get("away_rush", 0),
                    "first_downs": row.get("away_firstdowns", 0),
                    "third_down_efficiency": self.clean_numeric_data(
                        row.get("away_thirddowneff", "0-0")
                    ),
                    "fourth_down_efficiency": self.clean_numeric_data(
                        row.get("away_fourthdowneff", "0-0")
                    ),
                    "total_plays": row.get("away_totaloffensiveplays", 0),
                    "yards_per_play": row.get("away_yardsperplay", 0),
                    "turnovers": row.get("away_turnovers", 0),
                    "time_of_possession": row.get("away_possessiontime", "0:00"),
                }

                # Add stats to DataFrame
                df.at[idx, "home_team_stats"] = json.dumps(home_stats)
                df.at[idx, "away_team_stats"] = json.dumps(away_stats)

            except Exception as e:
                logger.error(f"Error processing stats for game {idx}: {str(e)}")
                # Add default empty stats
                df.at[idx, "home_team_stats"] = json.dumps({})
                df.at[idx, "away_team_stats"] = json.dumps({})

        return df

    def calculate_rolling_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling averages for team statistics."""
        logger.info("Calculating rolling averages")

        # List of statistics to calculate rolling averages for
        stats = [
            "total_yards",
            "first_downs",
            "offensive_plays",
            "yards_per_play",
            "passing_yards",
            "rushing_yards",
            "turnovers",
            "penalty_yards",
        ]

        # Calculate rolling averages for each statistic
        for stat in stats:
            if f"home_{stat}" in df.columns and f"away_{stat}" in df.columns:
                # Home team rolling averages
                df[f"home_{stat}_rolling"] = df.groupby("home_team")[
                    f"home_{stat}"
                ].transform(
                    lambda x: x.shift()
                    .rolling(window=self.lookback_weeks, min_periods=1)
                    .mean()
                )

                # Away team rolling averages
                df[f"away_{stat}_rolling"] = df.groupby("away_team")[
                    f"away_{stat}"
                ].transform(
                    lambda x: x.shift()
                    .rolling(window=self.lookback_weeks, min_periods=1)
                    .mean()
                )

        # Fill NaN values with 0
        df = df.fillna(0)

        return df

    def create_head_to_head_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features based on head-to-head history"""
        logger.info("Creating head-to-head features")

        # Create unique game identifier
        df["game_id"] = range(len(df))

        h2h_features = []

        for idx, row in df.iterrows():
            # Get previous meetings
            prev_meetings = df[
                (df["game_date"] < row["game_date"])
                & (
                    (
                        (df["home_team"] == row["home_team"])
                        & (df["away_team"] == row["away_team"])
                    )
                    | (
                        (df["home_team"] == row["away_team"])
                        & (df["away_team"] == row["home_team"])
                    )
                )
            ]

            # Calculate head-to-head features
            h2h_stats = {
                "game_id": row["game_id"],
                "h2h_games": len(prev_meetings),
                "h2h_home_wins": 0,
                "h2h_away_wins": 0,
                "h2h_avg_point_diff": 0,
            }

            if len(prev_meetings) > 0:
                home_wins = sum(
                    (prev_meetings["home_team"] == row["home_team"])
                    & (prev_meetings["home_score"] > prev_meetings["away_score"])
                )
                away_wins = sum(
                    (prev_meetings["home_team"] == row["away_team"])
                    & (prev_meetings["home_score"] > prev_meetings["away_score"])
                )

                h2h_stats.update(
                    {
                        "h2h_home_wins": home_wins,
                        "h2h_away_wins": away_wins,
                        "h2h_avg_point_diff": (
                            prev_meetings["home_score"] - prev_meetings["away_score"]
                        ).mean(),
                    }
                )

            h2h_features.append(h2h_stats)

        h2h_df = pd.DataFrame(h2h_features)
        return df.merge(h2h_df, on="game_id")

    def create_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create winning/losing streak features"""
        logger.info("Creating streak features")

        def calculate_streak(wins):
            """Calculate streak from win/loss series"""
            streak = 0
            streaks = []

            for win in wins:
                if win == 1:  # Win
                    streak = streak + 1 if streak >= 0 else 1
                else:  # Loss
                    streak = streak - 1 if streak <= 0 else -1
                streaks.append(streak)

            return streaks

        # Initialize streak columns
        df["home_streak"] = 0
        df["away_streak"] = 0

        # Calculate streaks for each team
        for team in df["home_team"].unique():
            # Get team's games in chronological order
            team_games = df[
                (df["home_team"] == team) | (df["away_team"] == team)
            ].sort_values("game_date")

            # Create win/loss series
            wins = []
            for _, game in team_games.iterrows():
                if game["home_team"] == team:
                    wins.append(1 if game["home_score"] > game["away_score"] else 0)
                else:
                    wins.append(1 if game["away_score"] > game["home_score"] else 0)

            # Calculate streaks
            streaks = calculate_streak(wins)

            # Map streaks back to original dataframe
            for i, (_, game) in enumerate(team_games.iterrows()):
                if i > 0:  # Use previous game's streak
                    if game["home_team"] == team:
                        df.loc[game.name, "home_streak"] = streaks[i - 1]
                    else:
                        df.loc[game.name, "away_streak"] = streaks[i - 1]

        return df

    def add_weather_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather impact features based on stadium type."""
        logger.info("Adding weather impact features")

        # Weather impact is based on whether the stadium is indoor or outdoor
        df["weather_impact"] = ~df["venue_indoor"].astype(bool)

        return df

    def add_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced statistics."""
        logger.info("Adding advanced statistics")

        try:
            # Offensive efficiency (yards per play)
            df["home_yards_per_play"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_totalyards", 0))
                / self.clean_numeric_data(row.get("away_totaloffensiveplays", 1))
                if self.clean_numeric_data(row.get("away_totaloffensiveplays", 0)) > 0
                else 0,
                axis=1,
            )

            df["away_yards_per_play"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_totalyards", 0))
                / self.clean_numeric_data(row.get("away_totaloffensiveplays", 1))
                if self.clean_numeric_data(row.get("away_totaloffensiveplays", 0)) > 0
                else 0,
                axis=1,
            )

            # Third down conversion rate
            df["home_third_down_rate"] = df.apply(
                lambda row: self.clean_numeric_data(
                    row.get("away_thirddowneff", "0-0")
                ),
                axis=1,
            )

            df["away_third_down_rate"] = df.apply(
                lambda row: self.clean_numeric_data(
                    row.get("away_thirddowneff", "0-0")
                ),
                axis=1,
            )

            # Red zone efficiency
            df["home_redzone_rate"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_redzoneattempts", 0))
                / self.clean_numeric_data(row.get("away_totaldrives", 1))
                if self.clean_numeric_data(row.get("away_totaldrives", 0)) > 0
                else 0,
                axis=1,
            )

            df["away_redzone_rate"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_redzoneattempts", 0))
                / self.clean_numeric_data(row.get("away_totaldrives", 1))
                if self.clean_numeric_data(row.get("away_totaldrives", 0)) > 0
                else 0,
                axis=1,
            )

            # Turnover rate per drive
            df["home_turnover_rate"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_turnovers", 0))
                / self.clean_numeric_data(row.get("away_totaldrives", 1))
                if self.clean_numeric_data(row.get("away_totaldrives", 0)) > 0
                else 0,
                axis=1,
            )

            df["away_turnover_rate"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_turnovers", 0))
                / self.clean_numeric_data(row.get("away_totaldrives", 1))
                if self.clean_numeric_data(row.get("away_totaldrives", 0)) > 0
                else 0,
                axis=1,
            )

            # Offensive balance (pass/run ratio)
            df["home_pass_ratio"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_netpassingyards", 0))
                / self.clean_numeric_data(row.get("away_totalyards", 1))
                if self.clean_numeric_data(row.get("away_totalyards", 0)) > 0
                else 0,
                axis=1,
            )

            df["away_pass_ratio"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_netpassingyards", 0))
                / self.clean_numeric_data(row.get("away_totalyards", 1))
                if self.clean_numeric_data(row.get("away_totalyards", 0)) > 0
                else 0,
                axis=1,
            )

            # Yards per pass attempt
            df["home_yards_per_pass"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_netpassingyards", 0))
                / self._get_pass_attempts(row.get("away_completionattempts", "0/0"))
                if self._get_pass_attempts(row.get("away_completionattempts", "0/0"))
                > 0
                else 0,
                axis=1,
            )

            df["away_yards_per_pass"] = df.apply(
                lambda row: self.clean_numeric_data(row.get("away_netpassingyards", 0))
                / self._get_pass_attempts(row.get("away_completionattempts", "0/0"))
                if self._get_pass_attempts(row.get("away_completionattempts", "0/0"))
                > 0
                else 0,
                axis=1,
            )

            # Completion percentage
            df["home_completion_pct"] = df.apply(
                lambda row: self._get_completion_pct(
                    row.get("away_completionattempts", "0/0")
                ),
                axis=1,
            )

            df["away_completion_pct"] = df.apply(
                lambda row: self._get_completion_pct(
                    row.get("away_completionattempts", "0/0")
                ),
                axis=1,
            )

            # Sack rate
            df["home_sack_rate"] = df.apply(
                lambda row: self._get_sacks(row.get("away_sacksyardslost", "0-0"))
                / self._get_pass_attempts(row.get("away_completionattempts", "0/0"))
                if self._get_pass_attempts(row.get("away_completionattempts", "0/0"))
                > 0
                else 0,
                axis=1,
            )

            df["away_sack_rate"] = df.apply(
                lambda row: self._get_sacks(row.get("away_sacksyardslost", "0-0"))
                / self._get_pass_attempts(row.get("away_completionattempts", "0/0"))
                if self._get_pass_attempts(row.get("away_completionattempts", "0/0"))
                > 0
                else 0,
                axis=1,
            )

            # Fill NaN values that may result from division by zero
            df = df.fillna(0)

            return df

        except Exception as e:
            logger.error(f"Error processing advanced stats: {str(e)}")
            return df

    def _get_pass_attempts(self, completionattempts: str) -> int:
        """Extract pass attempts from completion/attempts string."""
        try:
            return int(completionattempts.split("/")[1])
        except:
            return 0

    def _get_completion_pct(self, completionattempts: str) -> float:
        """Calculate completion percentage from completion/attempts string."""
        try:
            completions, attempts = map(int, completionattempts.split("/"))
            return completions / attempts if attempts > 0 else 0
        except:
            return 0

    def _get_sacks(self, sacksyardslost: str) -> int:
        """Extract number of sacks from sacks-yards lost string."""
        try:
            return int(sacksyardslost.split("-")[0])
        except:
            return 0

    def add_player_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add player impact features."""
        logger.info("Adding player impact features")

        try:
            # Offensive efficiency
            df["home_offensive_efficiency"] = df.apply(
                lambda row: row.get("away_totalyards", 0)
                / row.get("away_totaloffensiveplays", 1)
                if row.get("away_totaloffensiveplays", 0) > 0
                else 0,
                axis=1,
            )

            df["away_offensive_efficiency"] = df.apply(
                lambda row: row.get("away_totalyards", 0)
                / row.get("away_totaloffensiveplays", 1)
                if row.get("away_totaloffensiveplays", 0) > 0
                else 0,
                axis=1,
            )

            # Passing efficiency
            df["home_passing_efficiency"] = df.apply(
                lambda row: row.get("away_netpassingyards", 0)
                / row.get("away_totaloffensiveplays", 1)
                if row.get("away_totaloffensiveplays", 0) > 0
                else 0,
                axis=1,
            )

            df["away_passing_efficiency"] = df.apply(
                lambda row: row.get("away_netpassingyards", 0)
                / row.get("away_totaloffensiveplays", 1)
                if row.get("away_totaloffensiveplays", 0) > 0
                else 0,
                axis=1,
            )

            # First down efficiency
            df["home_first_down_rate"] = df.apply(
                lambda row: row.get("away_firstdowns", 0)
                / row.get("away_totaloffensiveplays", 1)
                if row.get("away_totaloffensiveplays", 0) > 0
                else 0,
                axis=1,
            )

            df["away_first_down_rate"] = df.apply(
                lambda row: row.get("away_firstdowns", 0)
                / row.get("away_totaloffensiveplays", 1)
                if row.get("away_totaloffensiveplays", 0) > 0
                else 0,
                axis=1,
            )

            # Fill NaN values
            df = df.fillna(0)

            return df

        except Exception as e:
            logger.error(f"Error processing player impact features: {str(e)}")
            return df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        logger.info("Adding momentum features")

        try:
            # Sort by date for each team
            df = df.sort_values("game_date")

            # Calculate point differential trends (last 3 games)
            for team in df["home_team"].unique():
                try:
                    team_games = df[
                        (df["home_team"] == team) | (df["away_team"] == team)
                    ].sort_values("game_date")

                    team_games["point_diff"] = team_games.apply(
                        lambda row: float(row["home_score"]) - float(row["away_score"])
                        if row["home_team"] == team
                        else float(row["away_score"]) - float(row["home_score"]),
                        axis=1,
                    )

                    # Calculate rolling stats
                    team_games["rolling_point_diff"] = (
                        team_games["point_diff"].rolling(window=3, min_periods=1).mean()
                    )

                    team_games["point_diff_trend"] = (
                        team_games["point_diff"]
                        .rolling(window=3, min_periods=1)
                        .apply(
                            lambda x: 1
                            if (len(x) > 1 and x.iloc[-1] > x.iloc[0])
                            else 0
                        )
                    )

                    # Map back to main dataframe
                    for idx, row in team_games.iterrows():
                        if row["home_team"] == team:
                            df.loc[idx, "home_rolling_point_diff"] = row[
                                "rolling_point_diff"
                            ]
                            df.loc[idx, "home_point_diff_trend"] = row[
                                "point_diff_trend"
                            ]
                        else:
                            df.loc[idx, "away_rolling_point_diff"] = row[
                                "rolling_point_diff"
                            ]
                            df.loc[idx, "away_point_diff_trend"] = row[
                                "point_diff_trend"
                            ]
                except Exception as e:
                    logger.error(f"Error processing momentum for team {team}: {str(e)}")
                    continue

            # Calculate win streak values
            df["home_win_streak"] = df.apply(
                lambda row: self._get_win_streak(
                    df, row["home_team"], row["game_date"]
                ),
                axis=1,
            )

            df["away_win_streak"] = df.apply(
                lambda row: self._get_win_streak(
                    df, row["away_team"], row["game_date"]
                ),
                axis=1,
            )

            # Calculate recent performance score (weighted average of last 5 games)
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Most recent game has highest weight
            for team in df["home_team"].unique():
                try:
                    team_games = df[
                        (df["home_team"] == team) | (df["away_team"] == team)
                    ].sort_values("game_date")

                    team_games["win"] = team_games.apply(
                        lambda row: 1
                        if (
                            (
                                row["home_team"] == team
                                and float(row["home_score"]) > float(row["away_score"])
                            )
                            or (
                                row["away_team"] == team
                                and float(row["away_score"]) > float(row["home_score"])
                            )
                        )
                        else 0,
                        axis=1,
                    )

                    # Calculate weighted performance
                    team_games["weighted_performance"] = (
                        team_games["win"]
                        .rolling(window=5, min_periods=1)
                        .apply(
                            lambda x: sum(w * v for w, v in zip(weights[-len(x) :], x))
                            / sum(weights[-len(x) :])
                        )
                    )

                    # Map back to main dataframe
                    for idx, row in team_games.iterrows():
                        if row["home_team"] == team:
                            df.loc[idx, "home_weighted_performance"] = row[
                                "weighted_performance"
                            ]
                        else:
                            df.loc[idx, "away_weighted_performance"] = row[
                                "weighted_performance"
                            ]
                except Exception as e:
                    logger.error(
                        f"Error processing weighted performance for team {team}: {str(e)}"
                    )
                    continue

            # Fill NaN values
            momentum_columns = [
                "home_rolling_point_diff",
                "away_rolling_point_diff",
                "home_point_diff_trend",
                "away_point_diff_trend",
                "home_win_streak",
                "away_win_streak",
                "home_weighted_performance",
                "away_weighted_performance",
            ]
            df[momentum_columns] = df[momentum_columns].fillna(0)

            return df

        except Exception as e:
            logger.error(f"Error processing momentum features: {str(e)}")
            return df

    def _get_win_streak(self, df: pd.DataFrame, team: str, game_date: str) -> int:
        """Calculate the win streak for a team up to a specific date."""
        try:
            # Get previous games
            prev_games = df[
                ((df["home_team"] == team) | (df["away_team"] == team))
                & (df["game_date"] < game_date)
            ].sort_values("game_date", ascending=False)

            streak = 0
            for _, game in prev_games.iterrows():
                if game["home_team"] == team:
                    if game["home_score"] > game["away_score"]:
                        streak += 1
                    else:
                        break
                else:  # away team
                    if game["away_score"] > game["home_score"]:
                        streak += 1
                    else:
                        break
            return streak

        except Exception as e:
            logger.error(f"Error calculating win streak: {str(e)}")
            return 0

    def create_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to rest days between games"""
        logger.info("Creating rest features")

        # Sort by date for accurate calculations
        df = df.sort_values("game_date")

        # Calculate days since last game for each team
        for team_type in ["home", "away"]:
            df[f"{team_type}_last_game"] = df.groupby(f"{team_type}_team")[
                "game_date"
            ].shift(1)
            df[f"{team_type}_rest_days"] = (
                df["game_date"] - df[f"{team_type}_last_game"]
            ).dt.days.fillna(
                7
            )  # Default to 7 days for first game

            # Create categorical rest features
            df[f"{team_type}_short_rest"] = df[f"{team_type}_rest_days"] <= 5
            df[f"{team_type}_long_rest"] = df[f"{team_type}_rest_days"] >= 10

        # Rest advantage
        df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]

        return df

    def add_division_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add division and rivalry-based features"""
        logger.info("Adding division features")

        # NFL division mappings
        divisions = {
            "AFC_EAST": ["NE", "BUF", "MIA", "NYJ"],
            "AFC_NORTH": ["BAL", "PIT", "CLE", "CIN"],
            "AFC_SOUTH": ["TEN", "IND", "HOU", "JAX"],
            "AFC_WEST": ["KC", "LV", "LAC", "DEN"],
            "NFC_EAST": ["DAL", "PHI", "NYG", "WAS"],
            "NFC_NORTH": ["GB", "MIN", "CHI", "DET"],
            "NFC_SOUTH": ["TB", "NO", "ATL", "CAR"],
            "NFC_WEST": ["SF", "SEA", "LAR", "ARI"],
        }

        # Create division mapping dictionary
        team_to_division = {}
        for division, teams in divisions.items():
            for team in teams:
                team_to_division[team] = division

        # Add division columns
        df["home_division"] = df["home_team"].map(team_to_division)
        df["away_division"] = df["away_team"].map(team_to_division)

        # Division game indicator
        df["division_game"] = df["home_division"] == df["away_division"]

        # Division record features
        df["home_div_record"] = (
            df[df["division_game"]]
            .groupby(["season", "home_team"])["home_win"]
            .transform("mean")
        )
        df["away_div_record"] = (
            df[df["division_game"]]
            .groupby(["season", "away_team"])
            .apply(lambda x: 1 - x["home_win"].mean())
            .reset_index(level=[0, 1], drop=True)
        )

        return df

    def add_primetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add primetime and special game features"""
        logger.info("Adding primetime features")

        # Extract hour from game time
        df["game_hour"] = df["game_date"].dt.hour

        # Primetime indicators
        df["is_primetime"] = df["game_hour"] >= 19  # 7 PM or later
        df["is_afternoon"] = (df["game_hour"] >= 16) & (df["game_hour"] < 19)
        df["is_early"] = df["game_hour"] < 16

        # Day of week features
        df["is_sunday"] = df["game_date"].dt.dayofweek == 6
        df["is_monday"] = df["game_date"].dt.dayofweek == 0
        df["is_thursday"] = df["game_date"].dt.dayofweek == 3

        # Team performance in primetime
        for team_type in ["home", "away"]:
            df[f"{team_type}_primetime_record"] = (
                df[df["is_primetime"]]
                .groupby(f"{team_type}_team")["home_win"]
                .transform("mean")
            )
            if team_type == "away":
                df[f"{team_type}_primetime_record"] = (
                    1 - df[f"{team_type}_primetime_record"]
                )

        return df

    def add_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team ranking and performance features"""
        logger.info("Adding ranking features")

        try:
            # Calculate season-to-date win percentages
            for team_type in ["home", "away"]:
                # Season record
                df[f"{team_type}_season_wins"] = df.groupby(
                    ["season", f"{team_type}_team"]
                )["home_win"].transform(lambda x: x.expanding().sum())
                df[f"{team_type}_season_games"] = (
                    df.groupby(["season", f"{team_type}_team"]).cumcount() + 1
                )
                df[f"{team_type}_season_winpct"] = (
                    df[f"{team_type}_season_wins"] / df[f"{team_type}_season_games"]
                )

                # Points scored/allowed
                df[f"{team_type}_points_scored"] = df.groupby(
                    ["season", f"{team_type}_team"]
                )[f"{team_type}_score"].transform(lambda x: x.expanding().mean())

            # Calculate point differential
            df["point_differential"] = df["home_score"] - df["away_score"]

            # Point differential by team
            for team_type in ["home", "away"]:
                df[f"{team_type}_point_diff"] = df.groupby(
                    ["season", f"{team_type}_team"]
                )["point_differential"].transform(lambda x: x.expanding().mean())
                if team_type == "away":
                    df[f"{team_type}_point_diff"] = -df[f"{team_type}_point_diff"]

            # Create relative ranking features
            df["win_pct_diff"] = df["home_season_winpct"] - df["away_season_winpct"]
            df["point_diff_advantage"] = df["home_point_diff"] - df["away_point_diff"]
            df["scoring_diff"] = df["home_points_scored"] - df["away_points_scored"]

            # Create ranking tiers (1-8 scale, 1 being best)
            for metric in ["season_winpct", "points_scored", "point_diff"]:
                for team_type in ["home", "away"]:
                    df[f"{team_type}_{metric}_rank"] = df.groupby("season")[
                        f"{team_type}_{metric}"
                    ].transform(
                        lambda x: pd.qcut(x, q=8, labels=False, duplicates="drop") + 1
                    )

            # Fill NaN values
            df = df.fillna(0)

            return df

        except Exception as e:
            logger.error(f"Error processing ranking features: {str(e)}")
            return df

    def add_game_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add game context features."""
        logger.info("Adding game context features")

        try:
            # Calculate current win percentages using home_win_percentage and away_win_percentage
            df["home_win_pct_season"] = df["home_win_percentage"]
            df["away_win_pct_season"] = df["away_win_percentage"]

            # Calculate point differential
            df["point_differential"] = df["home_score"] - df["away_score"]
            df["abs_point_differential"] = abs(df["point_differential"])

            # Game importance features
            df["late_season"] = df["week"] >= 13  # Last quarter of regular season
            df["must_win_home"] = (
                df["late_season"]
                & (df["home_win_pct_season"] >= 0.4)
                & (  # Still in playoff contention
                    df["home_win_pct_season"] < 0.6
                )  # Below comfortable playoff position
            )
            df["must_win_away"] = (
                df["late_season"]
                & (df["away_win_pct_season"] >= 0.4)
                & (df["away_win_pct_season"] < 0.6)
            )

            # Game closeness features
            df["close_game"] = df["abs_point_differential"] <= 7
            df["one_score_game"] = df["abs_point_differential"] <= 8
            df["two_score_game"] = df["abs_point_differential"] <= 14

            # Fill NaN values
            df = df.fillna(0)

            return df

        except Exception as e:
            logger.error(f"Error processing game context features: {str(e)}")
            return df

    def process_vedic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Vedic astrological features for each game."""
        logger.info("Processing Vedic astrological features")

        try:
            # Calculate astrological features for each game
            for idx, row in df.iterrows():
                try:
                    # Get game date and location
                    game_date = pd.to_datetime(row["game_date"])
                    venue_lat = float(row["venue_latitude"])
                    venue_long = float(row["venue_longitude"])

                    # Calculate planetary positions and aspects
                    planet_positions = self.vedic_calculator.get_planet_positions(
                        game_date, venue_lat, venue_long
                    )

                    # Calculate Nakshatra effects
                    moon_nakshatra = self.vedic_calculator.get_moon_nakshatra(
                        planet_positions["Moon"]["longitude"]
                    )

                    # Calculate zodiac influences
                    zodiac_strengths = self.vedic_calculator.calculate_zodiac_strengths(
                        planet_positions
                    )

                    # Calculate team-specific astrological scores
                    home_team = row["home_team"]
                    away_team = row["away_team"]

                    home_score = self.vedic_calculator.calculate_team_score(
                        home_team, planet_positions, zodiac_strengths, moon_nakshatra
                    )

                    away_score = self.vedic_calculator.calculate_team_score(
                        away_team, planet_positions, zodiac_strengths, moon_nakshatra
                    )

                    # Calculate planetary aspects
                    beneficial_aspects = self.vedic_calculator.count_beneficial_aspects(
                        planet_positions
                    )
                    malefic_aspects = self.vedic_calculator.count_malefic_aspects(
                        planet_positions
                    )

                    # Store features in dataframe
                    df.loc[idx, "vedic_home_score"] = home_score
                    df.loc[idx, "vedic_away_score"] = away_score
                    df.loc[idx, "vedic_score_diff"] = home_score - away_score
                    df.loc[idx, "moon_nakshatra"] = moon_nakshatra
                    df.loc[idx, "beneficial_aspects"] = beneficial_aspects
                    df.loc[idx, "malefic_aspects"] = malefic_aspects

                    # Store individual planet positions
                    for planet, data in planet_positions.items():
                        df.loc[idx, f"{planet.lower()}_longitude"] = data["longitude"]
                        df.loc[idx, f"{planet.lower()}_latitude"] = data["latitude"]
                        df.loc[idx, f"{planet.lower()}_distance"] = data["distance"]

                    # Store zodiac strengths
                    for zodiac, strength in zodiac_strengths.items():
                        df.loc[idx, f"zodiac_{zodiac.lower()}_strength"] = strength

                except Exception as e:
                    logger.error(
                        f"Error processing Vedic features for game {idx}: {str(e)}"
                    )
                    # Set default values
                    df.loc[idx, "vedic_home_score"] = 0.5
                    df.loc[idx, "vedic_away_score"] = 0.5
                    df.loc[idx, "vedic_score_diff"] = 0
                    df.loc[idx, "moon_nakshatra"] = 0
                    df.loc[idx, "beneficial_aspects"] = 0
                    df.loc[idx, "malefic_aspects"] = 0

            return df

        except Exception as e:
            logger.error(f"Error processing Vedic features: {str(e)}")
            return df

    def process_data(self, input_path: str, output_path: str) -> None:
        """
        Main method to process NFL data and create features

        Args:
            input_path (str): Path to raw NFL game data
            output_path (str): Path to save processed data
        """
        logger.info("Starting data processing pipeline")

        try:
            # Load data
            df = self.load_data(input_path)

            # Create team statistics
            team_stats = self.create_team_stats(df)

            # Calculate rolling averages
            team_stats = self.calculate_rolling_averages(team_stats)

            # Create head-to-head features
            df = self.create_head_to_head_features(df)

            # Create streak features
            team_stats = self.create_streak_features(team_stats)

            # Merge team stats back to main dataframe
            df = df.merge(
                team_stats,
                on=["game_date", "season", "week"],
                suffixes=("", "_opponent"),
            )

            # Add various feature groups
            df = self.add_weather_impact(df)
            df = self.add_advanced_stats(df)
            df = self.add_momentum_features(df)
            df = self.add_player_impact(df)
            df = self.add_division_features(df)
            df = self.add_primetime_features(df)
            df = self.add_ranking_features(df)
            df = self.add_game_context(df)
            df = self.process_vedic_features(df)

            # Handle missing values
            df = self._handle_missing_values(df)

            # Log feature statistics
            self._log_feature_stats(df)

            # Save processed data
            logger.info(f"Saving processed data to {output_path}")
            df.to_csv(output_path, index=False)

            return df

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values")

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns

        # Fill numeric missing values with median
        for col in numeric_cols:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logger.info(
                    f"Filled {df[col].isnull().sum()} missing values in {col} with median {median_value:.2f}"
                )

        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                logger.info(
                    f"Filled {df[col].isnull().sum()} missing values in {col} with mode {mode_value}"
                )

        return df

    def _log_feature_stats(self, df: pd.DataFrame) -> None:
        """Log basic statistics about the features"""
        logger.info("\nFeature Statistics:")
        logger.info(f"Total number of features: {len(df.columns)}")
        logger.info(f"Number of games: {len(df)}")

        # Log correlation with target
        if "home_win" in df.columns:
            correlations = df.corr()["home_win"].sort_values(ascending=False)
            logger.info("\nTop 10 features by correlation with home_win:")
            for feature, corr in correlations[:10].items():
                logger.info(f"{feature}: {corr:.3f}")

        # Log missing value statistics
        missing_stats = df.isnull().sum()
        if missing_stats.any():
            logger.info("\nColumns with missing values:")
            for col, count in missing_stats[missing_stats > 0].items():
                logger.info(f"{col}: {count} missing values ({count/len(df)*100:.2f}%)")


def main():
    """Main function to process NFL data."""
    logger.info("Starting data processing pipeline")

    # Initialize processor
    processor = NFLDataProcessor()

    # Process data
    processor.process_data(
        input_path="../data/raw/nfl_games_2019_2024_progress.csv",
        output_path="../data/processed/nfl_processed_data.csv",
    )


if __name__ == "__main__":
    main()
