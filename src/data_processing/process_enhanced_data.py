"""
Process collected NFL data into enhanced format for model training
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the collected game and team stats data."""
    # Fix the path to point to the project root data directory
    data_dir = Path(__file__).parent.parent.parent / "data"
    games_df = pd.read_csv(data_dir / "games.csv")
    team_stats_df = pd.read_csv(data_dir / "team_stats.csv")
    return games_df, team_stats_df


def calculate_rolling_stats(
    df: pd.DataFrame, team_col: str, stat_cols: List[str], window: int = 3
) -> pd.DataFrame:
    """Calculate rolling averages for team statistics."""
    for stat in stat_cols:
        col_name = f"{team_col}_{stat}_rolling_{window}"
        df[col_name] = df.groupby(team_col)[stat].transform(
            lambda x: x.shift().rolling(window=window, min_periods=1).mean()
        )
    return df


def calculate_momentum(
    df: pd.DataFrame, team_col: str, points_col: str, opp_points_col: str
) -> pd.DataFrame:
    """Calculate team momentum based on recent point differentials."""
    df[f"{team_col}_momentum"] = df.groupby(team_col)[points_col].transform(
        lambda x: x.shift().rolling(window=3, min_periods=1).mean()
    ) - df.groupby(team_col)[opp_points_col].transform(
        lambda x: x.shift().rolling(window=3, min_periods=1).mean()
    )
    return df


def calculate_strength_of_schedule(
    df: pd.DataFrame, team_col: str, opp_col: str
) -> pd.DataFrame:
    """Calculate strength of schedule based on opponent win percentages."""
    # Calculate team win percentages
    team_records = {}

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        game_date = row["gameday"]

        if home_team not in team_records:
            team_records[home_team] = {"wins": 0, "games": 0}
        if away_team not in team_records:
            team_records[away_team] = {"wins": 0, "games": 0}

        team_records[home_team]["games"] += 1
        team_records[away_team]["games"] += 1

        if row["home_score"] > row["away_score"]:
            team_records[home_team]["wins"] += 1
        elif row["away_score"] > row["home_score"]:
            team_records[away_team]["wins"] += 1

    # Calculate win percentages
    win_pcts = {
        team: record["wins"] / max(record["games"], 1)
        for team, record in team_records.items()
    }

    # Calculate SOS for each team
    df[f"{team_col}_sos"] = df.apply(
        lambda row: win_pcts.get(row[opp_col], 0.5), axis=1
    )

    return df


def process_enhanced_data():
    """Process the collected data into enhanced format."""
    logger.info("Loading collected data...")
    games_df, team_stats_df = load_data()

    # Convert date to datetime
    games_df["gameday"] = pd.to_datetime(games_df["gameday"])

    # Sort by date
    games_df = games_df.sort_values("gameday")

    # Calculate total points
    games_df["total_points"] = games_df["total"].fillna(
        games_df["home_score"] + games_df["away_score"]
    )

    # Add game context features
    games_df["is_division_game"] = games_df["div_game"].fillna(0).astype(int)
    games_df["is_primetime"] = games_df["is_primetime"].fillna(False).astype(int)
    games_df["is_dome"] = games_df["is_dome"].fillna(False).astype(int)

    # Extract numeric stats
    games_df["home_total_yards"] = pd.to_numeric(
        games_df["yards_gained"], errors="coerce"
    )
    games_df["away_total_yards"] = pd.to_numeric(
        games_df["yards_gained"], errors="coerce"
    )
    games_df["home_turnovers"] = pd.to_numeric(
        games_df["fumble"].fillna(0), errors="coerce"
    ) + pd.to_numeric(games_df["interception"].fillna(0), errors="coerce")
    games_df["away_turnovers"] = games_df[
        "home_turnovers"
    ]  # Since these are game totals
    games_df["home_penalty_yards"] = pd.to_numeric(
        games_df["penalty_yards"], errors="coerce"
    )
    games_df["away_penalty_yards"] = games_df[
        "home_penalty_yards"
    ]  # Since these are game totals

    # Calculate third down efficiency
    games_df["home_third_down_efficiency"] = pd.to_numeric(
        games_df["third_down_converted"], errors="coerce"
    ) / (
        pd.to_numeric(games_df["third_down_converted"], errors="coerce")
        + pd.to_numeric(games_df["third_down_failed"], errors="coerce")
    ).clip(
        lower=1
    )
    games_df["away_third_down_efficiency"] = games_df[
        "home_third_down_efficiency"
    ]  # Since these are game totals

    # Calculate rolling stats for both home and away teams
    stat_cols = ["total_yards", "turnovers", "penalty_yards", "third_down_efficiency"]
    for team_type in ["home", "away"]:
        for stat in stat_cols:
            games_df = calculate_rolling_stats(
                games_df, f"{team_type}_team", [f"{team_type}_{stat}"]
            )

    # Calculate scoring rolling averages
    games_df = calculate_rolling_stats(games_df, "home_team", ["home_score"], window=3)
    games_df = calculate_rolling_stats(games_df, "home_team", ["home_score"], window=5)
    games_df = calculate_rolling_stats(games_df, "away_team", ["away_score"], window=3)
    games_df = calculate_rolling_stats(games_df, "away_team", ["away_score"], window=5)

    # Calculate momentum
    games_df = calculate_momentum(games_df, "home_team", "home_score", "away_score")
    games_df = calculate_momentum(games_df, "away_team", "away_score", "home_score")

    # Calculate strength of schedule
    games_df = calculate_strength_of_schedule(games_df, "home_team", "away_team")
    games_df = calculate_strength_of_schedule(games_df, "away_team", "home_team")

    # Calculate offensive efficiency
    games_df["home_off_efficiency"] = games_df["home_score"] / games_df[
        "home_total_yards"
    ].clip(lower=1)
    games_df["away_off_efficiency"] = games_df["away_score"] / games_df[
        "away_total_yards"
    ].clip(lower=1)

    # Fill missing values with 0
    numeric_cols = games_df.select_dtypes(include=[np.number]).columns
    games_df[numeric_cols] = games_df[numeric_cols].fillna(0)

    # Save enhanced data
    output_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "historical"
        / "nfl_games_training.csv"
    )
    os.makedirs(output_path.parent, exist_ok=True)
    games_df.to_csv(output_path, index=False)
    logger.info(f"Enhanced data saved to {output_path}")

    # Log feature statistics
    logger.info("\nFeature Statistics:")
    for col in numeric_cols:
        stats = games_df[col].describe()
        logger.info(f"\n{col}:")
        logger.info(f"Mean: {stats['mean']:.2f}")
        logger.info(f"Std: {stats['std']:.2f}")
        logger.info(f"Min: {stats['min']:.2f}")
        logger.info(f"Max: {stats['max']:.2f}")


if __name__ == "__main__":
    process_enhanced_data()
