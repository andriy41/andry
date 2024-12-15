"""Train and evaluate the Vedic model using historical NFL data."""

import argparse
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import sys
from models.vedic_basic.vedic_model import VedicModel
from models.vedic_basic.calculations.nfl_vedic_calculator import NFLVedicCalculator
import os
from utils.data_monitor import DataMonitor
from utils.data_validator import DataQualityValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Stadium coordinates (latitude, longitude)
STADIUM_COORDS = {
    "Arrowhead Stadium": (39.0489, -94.4839),
    "Allegiant Stadium": (36.0909, -115.1833),
    "Levi's Stadium": (37.4033, -121.9694),
    "State Farm Stadium": (33.5276, -112.2626),
    "SoFi Stadium": (33.9534, -118.3387),
    "Lumen Field": (47.5952, -122.3316),
    "Empower Field at Mile High": (39.7439, -105.0201),
    "GEHA Field at Arrowhead Stadium": (39.0489, -94.4839),
    "Highmark Stadium": (42.7738, -78.7870),
    "Hard Rock Stadium": (25.9580, -80.2389),
    "Gillette Stadium": (42.0909, -71.2643),
    "MetLife Stadium": (40.8135, -74.0744),
    "Acrisure Stadium": (40.4468, -80.0158),
    "M&T Bank Stadium": (39.2780, -76.6227),
    "Paycor Stadium": (39.0955, -84.5161),
    "FirstEnergy Stadium": (41.5061, -81.6995),
    "Lucas Oil Stadium": (39.7601, -86.1639),
    "TIAA Bank Field": (30.3239, -81.6373),
    "NRG Stadium": (29.6847, -95.4107),
    "Nissan Stadium": (36.1665, -86.7713),
    "U.S. Bank Stadium": (44.9735, -93.2575),
    "Ford Field": (42.3400, -83.0456),
    "Lambeau Field": (44.5013, -88.0622),
    "Soldier Field": (41.8623, -87.6167),
    "Mercedes-Benz Stadium": (33.7553, -84.4006),
    "Bank of America Stadium": (35.2258, -80.8528),
    "Caesars Superdome": (29.9511, -90.0814),
    "Raymond James Stadium": (27.9758, -82.5033),
    "Lincoln Financial Field": (39.9013, -75.1674),
    "FedExField": (38.9077, -76.8645),
    "AT&T Stadium": (32.7473, -97.0945),
}


def validate_data_files():
    """Validate that all required data files exist and are accessible."""
    required_files = [
        ("data/stadium_coordinates.csv", "Stadium coordinates file"),
        ("models/trained", "Model output directory"),
    ]

    for file_path, description in required_files:
        if not os.path.exists(file_path):
            if file_path.endswith("/"):
                os.makedirs(file_path, exist_ok=True)
                logger.info(f"Created missing directory: {file_path}")
            else:
                raise FileNotFoundError(f"Missing {description}: {file_path}")


def validate_team_stats(team_stats_df):
    """Validate team statistics dataframe."""
    required_columns = [
        "team",
        "total_yards_per_game",
        "points_per_game",
        "yards_allowed_per_game",
        "points_allowed_per_game",
        "turnover_margin",
        "third_down_conversion",
    ]

    missing_columns = [
        col for col in required_columns if col not in team_stats_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in team_stats.csv: {missing_columns}"
        )

    # Check for missing values in critical columns
    null_teams = team_stats_df[team_stats_df["team"].isnull()]
    if not null_teams.empty:
        raise ValueError(f"Found {len(null_teams)} rows with missing team names")


def prepare_game_data(seasons=None, min_games=100):
    """Prepare NFL game data for Vedic analysis."""
    monitor = DataMonitor()

    try:
        monitor.start_phase("Data Loading")

        if seasons is None:
            seasons = list(range(2020, 2025))

        logger.info(f"Fetching game data for seasons: {seasons}")

        # Get schedule data
        try:
            schedules = nfl.import_schedules(seasons)

            # Check for data quality issues
            monitor.check_nulls(
                schedules, ["home_team", "away_team", "stadium", "result"]
            )
            monitor.check_ranges(
                schedules, {"temp": (-50, 120), "wind": (0, 100), "week": (1, 22)}
            )
            monitor.check_distributions(schedules, ["temp", "wind"])

        except Exception as e:
            raise RuntimeError(f"Failed to fetch NFL schedules: {str(e)}")

        # Get team stats
        try:
            # Get player stats and rosters to map to teams
            player_stats = nfl.import_seasonal_data(seasons)
            rosters = nfl.import_seasonal_rosters(seasons)

            # Merge with rosters to get team information
            stats = pd.merge(
                player_stats,
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
            team_stats["yards_allowed_per_game"] = team_stats[
                "total_yards_per_game"
            ]  # Will need opponent stats for this
            team_stats["points_allowed_per_game"] = team_stats[
                "points_per_game"
            ]  # Will need opponent stats for this
            team_stats["turnover_margin"] = (
                -(team_stats["interceptions"] + team_stats["rushing_fumbles_lost"])
                / team_stats["games"]
            )
            team_stats[
                "third_down_conversion"
            ] = 50.0  # Default value since we don't have this data

            # Check team stats quality
            monitor.check_nulls(
                team_stats, ["team", "total_yards_per_game", "points_per_game"]
            )
            monitor.check_ranges(
                team_stats,
                {
                    "total_yards_per_game": (0, 1000),
                    "points_per_game": (0, 100),
                    "turnover_margin": (-10, 10),
                    "third_down_conversion": (0, 100),
                },
            )
            monitor.check_distributions(
                team_stats,
                [
                    "total_yards_per_game",
                    "points_per_game",
                    "turnover_margin",
                    "third_down_conversion",
                ],
            )
            monitor.check_correlations(team_stats)

        except Exception as e:
            raise RuntimeError(f"Failed to load team stats: {str(e)}")

        monitor.end_phase()

        # Filter and prepare data
        monitor.start_phase("Data Preparation")

        schedules = schedules[schedules["game_type"].isin(["REG", "POST"])]
        if schedules.empty:
            raise ValueError("No valid games found after filtering")

        X = []
        y = []
        calculator = NFLVedicCalculator()

        total_games = len(schedules)
        for idx, game in schedules.iterrows():
            try:
                if pd.isna(game["result"]):
                    monitor.track_progress(
                        idx + 1,
                        total_games,
                        False,
                        f"Missing result for game {game['game_id']}",
                    )
                    continue

                # Process game data
                features = process_game(game, team_stats, calculator)
                if features is not None:
                    X.append(features)
                    y.append(1 if game["result"] > 0 else 0)
                    monitor.track_progress(idx + 1, total_games, True)
                else:
                    monitor.track_progress(
                        idx + 1,
                        total_games,
                        False,
                        f"Failed to process game {game['game_id']}",
                    )

            except Exception as e:
                monitor.track_progress(
                    idx + 1,
                    total_games,
                    False,
                    f"Error processing game {game['game_id']}: {str(e)}",
                )
                continue

        monitor.end_phase()

        # Final validation
        monitor.start_phase("Final Validation")

        X = np.array(X)
        y = np.array(y)

        # Check class balance
        monitor.check_class_balance(y)

        # Check feature correlations
        if len(X) > 0:
            feature_df = pd.DataFrame(X, columns=get_feature_names())
            monitor.check_correlations(feature_df)

        # Generate and save report
        report = monitor.get_report()
        report.save("models/reports/data_quality_report.json")
        report.print_summary()

        monitor.end_phase()

        if report.error_count > total_games * 0.2:  # More than 20% errors
            raise ValueError("Too many errors during data processing")

        if len(X) < min_games:
            raise ValueError(f"Insufficient valid games: {len(X)} < {min_games}")

        return X, y

    except Exception as e:
        logger.error(f"Failed to prepare game data: {str(e)}")
        raise


def process_game(game, team_stats, calculator):
    """Process a single game with validation."""
    try:
        # Validate stadium coordinates
        stadium_name = game["stadium"]
        if stadium_name not in STADIUM_COORDS:
            return None

        lat, lon = STADIUM_COORDS[stadium_name]

        # Get team stats
        home_stats = team_stats[team_stats["team"] == game["home_team"]].iloc[0]
        away_stats = team_stats[team_stats["team"] == game["away_team"]].iloc[0]

        # Calculate Vedic features
        game_date = pd.to_datetime(game["gameday"])
        game_time = pd.to_datetime(game["gametime"], format="%H:%M").time()

        vedic_features = calculator.calculate_game_features(
            game_date.strftime("%Y-%m-%d"),
            game_time.strftime("%H:%M"),
            "US/Eastern",
            game["home_team"],
            game["away_team"],
            week_number=int(game["week"]),
            is_playoff=(game["game_type"] == "POST"),
        )

        # Create feature vector
        return create_feature_vector(
            vedic_features, home_stats, away_stats, game, lat, lon
        )

    except Exception as e:
        logger.error(f"Failed to process game: {str(e)}")
        return None


def create_feature_vector(vedic_features, home_stats, away_stats, game, lat, lon):
    """Create a feature vector with validation."""
    validator = DataQualityValidator()
    errors = []

    # Extract and validate Vedic features
    vedic_feature_list = []
    for feature in validator.REQUIRED_COLUMNS["vedic_features"]:
        value = vedic_features.get(feature)
        is_valid, error_msg = validator.validate_numeric_range(value, feature)
        if not is_valid:
            errors.append(error_msg)
        vedic_feature_list.append(value)

    # Extract and validate team stats
    team_features = []
    for stats, prefix in [(home_stats, "home"), (away_stats, "away")]:
        for stat in [
            "total_yards_per_game",
            "points_per_game",
            "yards_allowed_per_game",
            "points_allowed_per_game",
            "turnover_margin",
            "third_down_conversion",
        ]:
            value = float(stats.get(stat, 0))
            is_valid, error_msg = validator.validate_numeric_range(
                value, stat.replace("_per_game", "")
            )
            if not is_valid:
                errors.append(f"{prefix}_{error_msg}")
            team_features.append(value)

    # Extract and validate environmental features
    env_features = []

    # Temperature
    temp = float(game["temp"]) if pd.notna(game["temp"]) else 70.0
    is_valid, error_msg = validator.validate_numeric_range(temp, "temperature")
    if not is_valid:
        errors.append(error_msg)
    env_features.append(temp)

    # Wind speed
    wind = float(game["wind"]) if pd.notna(game["wind"]) else 0.0
    is_valid, error_msg = validator.validate_numeric_range(wind, "wind_speed")
    if not is_valid:
        errors.append(error_msg)
    env_features.append(wind)

    # Binary features
    env_features.extend(
        [
            1.0 if game["roof"] == "dome" else 0.0,
            1.0 if game["div_game"] == 1 else 0.0,
            1.0 if game["game_type"] == "POST" else 0.0,
        ]
    )

    # Week number
    week = float(game["week"])
    is_valid, error_msg = validator.validate_numeric_range(week, "week")
    if not is_valid:
        errors.append(error_msg)
    env_features.append(week)

    # Stadium coordinates
    for coord, name in [(lat, "latitude"), (lon, "longitude")]:
        is_valid, error_msg = validator.validate_numeric_range(coord, name)
        if not is_valid:
            errors.append(error_msg)
        env_features.append(coord)

    if errors:
        error_msg = "; ".join(errors)
        raise ValueError(f"Invalid feature values: {error_msg}")

    return vedic_feature_list + team_features + env_features


def train_and_evaluate(seasons=None, min_games=100):
    """Train and evaluate the Vedic model."""
    monitor = DataMonitor()

    try:
        monitor.start_phase("Data Preparation")
        X, y = prepare_game_data(seasons, min_games)
        monitor.end_phase()

        monitor.start_phase("Model Training")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        model = VedicModel(feature_names=get_feature_names())
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        monitor.report.add_metric("accuracy", accuracy)
        monitor.report.add_metric("train_samples", len(X_train))
        monitor.report.add_metric("test_samples", len(X_test))

        # Print results
        logger.info(f"\nModel Accuracy: {accuracy:.2%}")
        logger.info("\nClassification Report:")
        logger.info(
            classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"])
        )

        # Save model
        model.save("models/trained/vedic_model.pkl")
        monitor.end_phase()

        # Save final report
        monitor.report.save("models/reports/training_report.json")
        monitor.report.print_summary()

        return model, accuracy

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


def get_feature_names():
    """Get the list of feature names."""
    validator = DataQualityValidator()
    return (
        validator.REQUIRED_COLUMNS["vedic_features"]
        + [f"home_{stat}" for stat in validator.REQUIRED_COLUMNS["team_stats"][1:]]
        + [f"away_{stat}" for stat in validator.REQUIRED_COLUMNS["team_stats"][1:]]
        + [
            "temperature",
            "wind_speed",
            "is_dome",
            "is_division_game",
            "is_playoff",
            "week_number",
            "stadium_lat",
            "stadium_lon",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the NFL Vedic prediction model")
    parser.add_argument(
        "--start_season",
        type=int,
        default=2020,
        help="Starting season year (default: 2020)",
    )
    parser.add_argument(
        "--end_season",
        type=int,
        default=2024,
        help="Ending season year (default: 2024)",
    )
    parser.add_argument(
        "--min_games",
        type=int,
        default=100,
        help="Minimum number of games required for training (default: 100)",
    )

    args = parser.parse_args()

    try:
        # Create output directories
        os.makedirs("models/trained", exist_ok=True)
        os.makedirs("models/reports", exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("models/reports/training.log"),
            ],
        )

        # Validate arguments
        if args.start_season > args.end_season:
            raise ValueError("Start season cannot be after end season")

        if args.start_season < 2010:
            logger.warning("Data before 2010 may be incomplete")

        if args.min_games < 50:
            logger.warning(
                "Training with less than 50 games may lead to poor model performance"
            )

        seasons = list(range(args.start_season, args.end_season + 1))
        logger.info(
            f"Training model with seasons from {args.start_season} to {args.end_season}"
        )

        model, accuracy = train_and_evaluate(seasons, args.min_games)
        logger.info("Training complete!")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
