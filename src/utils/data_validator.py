"""Data validation utilities for NFL prediction models."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validates data quality for NFL prediction models."""

    # Define valid ranges for various metrics
    VALID_RANGES = {
        "temperature": (-50, 120),  # Fahrenheit
        "wind_speed": (0, 100),  # MPH
        "points": (0, 100),  # Game points
        "yards": (0, 1000),  # Game yards
        "turnover_margin": (-10, 10),
        "third_down_conversion": (0, 100),  # Percentage
        "moon_phase": (0, 1),  # 0 to 1
        "planet_strength": (0, 100),  # Percentage
        "zodiac_strength": (0, 100),  # Percentage
        "nakshatra_score": (0, 100),  # Percentage
        "week": (1, 22),  # NFL week number
        "latitude": (25, 50),  # US stadium latitudes
        "longitude": (-125, -65),  # US stadium longitudes
    }

    # Required columns for different data types
    REQUIRED_COLUMNS = {
        "team_stats": [
            "team",
            "total_yards_per_game",
            "points_per_game",
            "yards_allowed_per_game",
            "points_allowed_per_game",
            "turnover_margin",
            "third_down_conversion",
        ],
        "game_data": [
            "game_id",
            "home_team",
            "away_team",
            "result",
            "stadium",
            "temp",
            "wind",
            "roof",
            "week",
        ],
        "vedic_features": [
            "moon_phase",
            "home_planet_strength",
            "away_planet_strength",
            "home_zodiac_strength",
            "away_zodiac_strength",
            "beneficial_aspects",
            "malefic_aspects",
            "home_nakshatra_score",
            "away_nakshatra_score",
            "planetary_alignment",
            "raja_yoga",
            "dhana_yoga",
            "vipreet_yoga",
            "kesari_yoga",
        ],
    }

    @staticmethod
    def validate_numeric_range(value: float, metric_name: str) -> Tuple[bool, str]:
        """Validate if a numeric value falls within the expected range."""
        if metric_name not in DataQualityValidator.VALID_RANGES:
            return True, ""

        min_val, max_val = DataQualityValidator.VALID_RANGES[metric_name]
        if not isinstance(value, (int, float)) or pd.isna(value):
            return False, f"Invalid {metric_name}: {value} (not numeric)"

        if not min_val <= value <= max_val:
            return (
                False,
                f"Invalid {metric_name}: {value} (outside range [{min_val}, {max_val}])",
            )

        return True, ""

    @staticmethod
    def validate_team_stats(stats_df: pd.DataFrame) -> List[str]:
        """Validate team statistics dataframe."""
        errors = []

        # Check required columns
        missing_cols = [
            col
            for col in DataQualityValidator.REQUIRED_COLUMNS["team_stats"]
            if col not in stats_df.columns
        ]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check for missing team names
        null_teams = stats_df[stats_df["team"].isnull()]
        if not null_teams.empty:
            errors.append(f"Found {len(null_teams)} rows with missing team names")

        # Validate numeric ranges
        for col in stats_df.columns:
            if col == "team":
                continue

            metric_name = col.replace("_per_game", "")
            for idx, value in stats_df[col].items():
                try:
                    is_valid, error_msg = DataQualityValidator.validate_numeric_range(
                        float(value), metric_name
                    )
                    if not is_valid:
                        errors.append(f"Row {idx}: {error_msg}")
                except ValueError:
                    errors.append(f"Row {idx}: Invalid numeric value in {col}: {value}")

        return errors

    @staticmethod
    def validate_game_data(game_df: pd.DataFrame) -> List[str]:
        """Validate game data dataframe."""
        errors = []

        # Check required columns
        missing_cols = [
            col
            for col in DataQualityValidator.REQUIRED_COLUMNS["game_data"]
            if col not in game_df.columns
        ]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check for complete game records
        incomplete_games = game_df[
            game_df[["home_team", "away_team", "stadium"]].isnull().any(axis=1)
        ]
        if not incomplete_games.empty:
            errors.append(f"Found {len(incomplete_games)} incomplete game records")

        # Validate numeric ranges for environmental data
        for idx, row in game_df.iterrows():
            # Temperature
            if pd.notna(row["temp"]):
                is_valid, error_msg = DataQualityValidator.validate_numeric_range(
                    float(row["temp"]), "temperature"
                )
                if not is_valid:
                    errors.append(f"Game {row['game_id']}: {error_msg}")

            # Wind speed
            if pd.notna(row["wind"]):
                is_valid, error_msg = DataQualityValidator.validate_numeric_range(
                    float(row["wind"]), "wind_speed"
                )
                if not is_valid:
                    errors.append(f"Game {row['game_id']}: {error_msg}")

            # Week number
            try:
                is_valid, error_msg = DataQualityValidator.validate_numeric_range(
                    int(row["week"]), "week"
                )
                if not is_valid:
                    errors.append(f"Game {row['game_id']}: {error_msg}")
            except ValueError:
                errors.append(
                    f"Game {row['game_id']}: Invalid week number: {row['week']}"
                )

        return errors

    @staticmethod
    def validate_vedic_features(features: Dict[str, Any]) -> List[str]:
        """Validate Vedic features dictionary."""
        errors = []

        # Check required features
        missing_features = [
            f
            for f in DataQualityValidator.REQUIRED_COLUMNS["vedic_features"]
            if f not in features
        ]
        if missing_features:
            errors.append(f"Missing Vedic features: {missing_features}")

        # Validate numeric ranges
        for feature, value in features.items():
            if "planet_strength" in feature:
                is_valid, error_msg = DataQualityValidator.validate_numeric_range(
                    value, "planet_strength"
                )
            elif "zodiac_strength" in feature:
                is_valid, error_msg = DataQualityValidator.validate_numeric_range(
                    value, "zodiac_strength"
                )
            elif "nakshatra_score" in feature:
                is_valid, error_msg = DataQualityValidator.validate_numeric_range(
                    value, "nakshatra_score"
                )
            elif feature == "moon_phase":
                is_valid, error_msg = DataQualityValidator.validate_numeric_range(
                    value, "moon_phase"
                )
            else:
                continue

            if not is_valid:
                errors.append(error_msg)

        return errors

    @staticmethod
    def validate_model_data(
        X: np.ndarray, y: np.ndarray, min_samples: int = 100
    ) -> List[str]:
        """Validate model input data."""
        errors = []

        # Check minimum sample size
        if len(X) < min_samples:
            errors.append(f"Insufficient samples: {len(X)} < {min_samples}")

        # Check for NaN or infinite values
        if np.any(np.isnan(X)):
            nan_count = np.isnan(X).sum()
            errors.append(f"Found {nan_count} NaN values in features")

        if np.any(np.isinf(X)):
            inf_count = np.isinf(X).sum()
            errors.append(f"Found {inf_count} infinite values in features")

        # Check target variable
        if not np.all(np.isin(y, [0, 1])):
            invalid_count = np.sum(~np.isin(y, [0, 1]))
            errors.append(f"Found {invalid_count} invalid target values (not 0 or 1)")

        # Check class balance
        if len(y) > 0:
            class_counts = np.bincount(y)
            majority_class = max(class_counts)
            minority_class = min(class_counts)
            imbalance_ratio = majority_class / minority_class
            if imbalance_ratio > 3:
                errors.append(f"Severe class imbalance: ratio = {imbalance_ratio:.2f}")

        return errors

    @staticmethod
    def log_validation_summary(errors: List[str], data_type: str):
        """Log a summary of validation errors."""
        if not errors:
            logger.info(f"✓ {data_type} validation passed: No errors found")
            return

        logger.warning(f"⚠ {data_type} validation found {len(errors)} issues:")
        for i, error in enumerate(errors, 1):
            logger.warning(f"  {i}. {error}")

        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more issues")
