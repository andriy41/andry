"""Test NFL prediction system with sample data."""

import logging
import pandas as pd
from datetime import datetime
from src.models.prediction_system import NFLPredictionSystem
from src.models.vedic_astrology.nfl_vedic_calculator import NFLVedicCalculator
from src.data.team_metadata import get_team_info
import numpy as np
from src.utils.model_evaluation import cross_validate_model, validate_input_data
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create a sample dataset of NFL games with enhanced features."""
    games = [
        # Strong favorite with historical dominance
        {
            "date": "2023-09-24",
            "home_team": "SF",
            "away_team": "NYG",
            "home_score": 30,
            "away_score": 12,
            "total_points": 42,
            "spread": 18,
            "home_win": 1,
            "season": 2023,
            "week": 3,
            "home_dome": 0,
            "away_dome": 0,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "moderate",
            "mars_strength": 0.9,
            "jupiter_strength": 0.8,
            "saturn_strength": 0.7,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.9,
            "away_nakshatra_score": 0.4,
            "planetary_alignment": 0.9,
            "moon_phase_score": 0.8,
            # Performance metrics
            "home_win_streak": 3,
            "away_win_streak": 0,
            "home_points_for": 30.5,
            "away_points_for": 14.2,
            "home_points_against": 15.3,
            "away_points_against": 28.7,
            # Historical matchup data
            "h2h_wins_home": 4,  # Home team wins in last 5 meetings
            "h2h_wins_away": 1,  # Away team wins in last 5 meetings
            "h2h_avg_margin": 12.4,  # Average margin in last 5 meetings
            # Defensive rankings (1-32, lower is better)
            "home_def_rank": 2,
            "away_def_rank": 28,
            "home_pass_def_rank": 3,
            "away_pass_def_rank": 25,
            "home_rush_def_rank": 4,
            "away_rush_def_rank": 30,
            # Weather conditions
            "temperature": 72,
            "wind_speed": 8,
            "precipitation": 0,
            # Key player availability (percentage of starters available)
            "home_starters_available": 0.95,
            "away_starters_available": 0.85,
        },
        # Division rivalry with weather impact
        {
            "date": "2023-10-01",
            "home_team": "BUF",
            "away_team": "MIA",
            "home_score": 35,
            "away_score": 14,
            "total_points": 49,
            "spread": 21,
            "home_win": 1,
            "season": 2023,
            "week": 4,
            "home_dome": 0,
            "away_dome": 0,
            "is_division_game": 1,
            "is_conference_game": 1,
            "home_climate": "cold",
            "away_climate": "hot",
            "mars_strength": 0.9,
            "jupiter_strength": 0.8,
            "saturn_strength": 0.7,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.95,
            "away_nakshatra_score": 0.3,
            "planetary_alignment": 0.9,
            "moon_phase_score": 0.9,
            # Performance metrics
            "home_win_streak": 4,
            "away_win_streak": 0,
            "home_points_for": 32.5,
            "away_points_for": 15.8,
            "home_points_against": 14.2,
            "away_points_against": 27.9,
            # Historical matchup data
            "h2h_wins_home": 3,
            "h2h_wins_away": 2,
            "h2h_avg_margin": 8.6,
            # Defensive rankings
            "home_def_rank": 1,
            "away_def_rank": 24,
            "home_pass_def_rank": 2,
            "away_pass_def_rank": 22,
            "home_rush_def_rank": 3,
            "away_rush_def_rank": 26,
            # Weather conditions
            "temperature": 45,
            "wind_speed": 15,
            "precipitation": 0,
            # Key player availability
            "home_starters_available": 0.98,
            "away_starters_available": 0.82,
        },
        # Dome game with injury impact
        {
            "date": "2023-10-08",
            "home_team": "DAL",
            "away_team": "LAR",
            "home_score": 28,
            "away_score": 10,
            "total_points": 38,
            "spread": 18,
            "home_win": 1,
            "season": 2023,
            "week": 5,
            "home_dome": 1,
            "away_dome": 1,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "moderate",
            "mars_strength": 0.85,
            "jupiter_strength": 0.75,
            "saturn_strength": 0.8,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.88,
            "away_nakshatra_score": 0.45,
            "planetary_alignment": 0.85,
            "moon_phase_score": 0.82,
            # Performance metrics
            "home_win_streak": 3,
            "away_win_streak": 0,
            "home_points_for": 29.8,
            "away_points_for": 16.5,
            "home_points_against": 15.8,
            "away_points_against": 26.2,
            # Historical matchup data
            "h2h_wins_home": 4,
            "h2h_wins_away": 1,
            "h2h_avg_margin": 10.2,
            # Defensive rankings
            "home_def_rank": 3,
            "away_def_rank": 25,
            "home_pass_def_rank": 4,
            "away_pass_def_rank": 23,
            "home_rush_def_rank": 5,
            "away_rush_def_rank": 28,
            # Weather conditions (indoor)
            "temperature": 72,
            "wind_speed": 0,
            "precipitation": 0,
            # Key player availability
            "home_starters_available": 0.92,
            "away_starters_available": 0.75,
        },
        {
            "date": "2023-09-24",
            "home_team": "DAL",
            "away_team": "ARI",
            "home_score": 28,
            "away_score": 16,
            "total_points": 44,
            "spread": 12,
            "home_win": 1,
            "season": 2023,
            "week": 3,
            "home_dome": 1,
            "away_dome": 1,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "hot",
            "mars_strength": 0.8,
            "jupiter_strength": 0.7,
            "saturn_strength": 0.9,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.8,
            "away_nakshatra_score": 0.5,
            "planetary_alignment": 0.8,
            "moon_phase_score": 0.7,
            "home_win_streak": 2,
            "away_win_streak": 0,
            "home_points_for": 28.3,
            "away_points_for": 17.5,
            "home_points_against": 16.8,
            "away_points_against": 25.4,
            "h2h_wins_home": 3,
            "h2h_wins_away": 2,
            "h2h_avg_margin": 6.2,
            "home_def_rank": 5,
            "away_def_rank": 20,
            "home_pass_def_rank": 6,
            "away_pass_def_rank": 18,
            "home_rush_def_rank": 7,
            "away_rush_def_rank": 22,
            "temperature": 75,
            "wind_speed": 5,
            "precipitation": 0,
            "home_starters_available": 0.95,
            "away_starters_available": 0.85,
        },
        {
            "date": "2023-09-07",
            "home_team": "KC",
            "away_team": "DET",
            "home_score": 20,
            "away_score": 21,
            "total_points": 41,
            "spread": -1,
            "home_win": 0,
            "season": 2023,
            "week": 1,
            "home_dome": 0,
            "away_dome": 1,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "moderate",
            "mars_strength": 0.7,
            "jupiter_strength": 0.6,
            "saturn_strength": 0.5,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.8,
            "away_nakshatra_score": 0.6,
            "planetary_alignment": 0.7,
            "moon_phase_score": 0.6,
            "home_win_streak": 1,
            "away_win_streak": 1,
            "home_points_for": 25.8,
            "away_points_for": 23.4,
            "home_points_against": 22.1,
            "away_points_against": 24.3,
            "h2h_wins_home": 2,
            "h2h_wins_away": 3,
            "h2h_avg_margin": 4.5,
            "home_def_rank": 10,
            "away_def_rank": 15,
            "home_pass_def_rank": 11,
            "away_pass_def_rank": 12,
            "home_rush_def_rank": 9,
            "away_rush_def_rank": 18,
            "temperature": 60,
            "wind_speed": 10,
            "precipitation": 0,
            "home_starters_available": 0.92,
            "away_starters_available": 0.88,
        },
        {
            "date": "2023-09-10",
            "home_team": "CHI",
            "away_team": "GB",
            "home_score": 20,
            "away_score": 38,
            "total_points": 58,
            "spread": -18,
            "home_win": 0,
            "season": 2023,
            "week": 1,
            "home_dome": 0,
            "away_dome": 0,
            "is_division_game": 1,
            "is_conference_game": 1,
            "home_climate": "cold",
            "away_climate": "cold",
            "mars_strength": 0.5,
            "jupiter_strength": 0.8,
            "saturn_strength": 0.6,
            "home_team_yoga": 0,
            "away_team_yoga": 1,
            "home_nakshatra_score": 0.5,
            "away_nakshatra_score": 0.9,
            "planetary_alignment": 0.8,
            "moon_phase_score": 0.7,
            "home_win_streak": 0,
            "away_win_streak": 1,
            "home_points_for": 20.3,
            "away_points_for": 26.8,
            "home_points_against": 25.9,
            "away_points_against": 20.5,
            "h2h_wins_home": 2,
            "h2h_wins_away": 3,
            "h2h_avg_margin": 5.1,
            "home_def_rank": 20,
            "away_def_rank": 8,
            "home_pass_def_rank": 22,
            "away_pass_def_rank": 5,
            "home_rush_def_rank": 18,
            "away_rush_def_rank": 11,
            "temperature": 50,
            "wind_speed": 12,
            "precipitation": 0,
            "home_starters_available": 0.90,
            "away_starters_available": 0.95,
        },
        {
            "date": "2023-09-10",
            "home_team": "MIN",
            "away_team": "TB",
            "home_score": 17,
            "away_score": 20,
            "total_points": 37,
            "spread": -3,
            "home_win": 0,
            "season": 2023,
            "week": 1,
            "home_dome": 1,
            "away_dome": 0,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "hot",
            "mars_strength": 0.6,
            "jupiter_strength": 0.7,
            "saturn_strength": 0.8,
            "home_team_yoga": 0,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.6,
            "away_nakshatra_score": 0.7,
            "planetary_alignment": 0.6,
            "moon_phase_score": 0.8,
            "home_win_streak": 0,
            "away_win_streak": 1,
            "home_points_for": 22.5,
            "away_points_for": 24.9,
            "home_points_against": 23.8,
            "away_points_against": 22.1,
            "h2h_wins_home": 2,
            "h2h_wins_away": 3,
            "h2h_avg_margin": 3.9,
            "home_def_rank": 15,
            "away_def_rank": 12,
            "home_pass_def_rank": 16,
            "away_pass_def_rank": 9,
            "home_rush_def_rank": 14,
            "away_rush_def_rank": 15,
            "temperature": 70,
            "wind_speed": 8,
            "precipitation": 0,
            "home_starters_available": 0.92,
            "away_starters_available": 0.90,
        },
        {
            "date": "2023-09-10",
            "home_team": "WSH",
            "away_team": "ARI",
            "home_score": 20,
            "away_score": 16,
            "total_points": 36,
            "spread": 4,
            "home_win": 1,
            "season": 2023,
            "week": 1,
            "home_dome": 0,
            "away_dome": 1,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "hot",
            "mars_strength": 0.8,
            "jupiter_strength": 0.5,
            "saturn_strength": 0.6,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.9,
            "away_nakshatra_score": 0.5,
            "planetary_alignment": 0.6,
            "moon_phase_score": 0.7,
            "home_win_streak": 1,
            "away_win_streak": 0,
            "home_points_for": 24.8,
            "away_points_for": 18.2,
            "home_points_against": 20.5,
            "away_points_against": 25.9,
            "h2h_wins_home": 3,
            "h2h_wins_away": 2,
            "h2h_avg_margin": 4.8,
            "home_def_rank": 8,
            "away_def_rank": 22,
            "home_pass_def_rank": 7,
            "away_pass_def_rank": 20,
            "home_rush_def_rank": 10,
            "away_rush_def_rank": 24,
            "temperature": 75,
            "wind_speed": 5,
            "precipitation": 0,
            "home_starters_available": 0.95,
            "away_starters_available": 0.85,
        },
        {
            "date": "2023-09-17",
            "home_team": "DAL",
            "away_team": "NYJ",
            "home_score": 30,
            "away_score": 10,
            "total_points": 40,
            "spread": 20,
            "home_win": 1,
            "season": 2023,
            "week": 2,
            "home_dome": 1,
            "away_dome": 0,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "moderate",
            "mars_strength": 0.9,
            "jupiter_strength": 0.8,
            "saturn_strength": 0.7,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.9,
            "away_nakshatra_score": 0.4,
            "planetary_alignment": 0.9,
            "moon_phase_score": 0.8,
            "home_win_streak": 2,
            "away_win_streak": 0,
            "home_points_for": 28.3,
            "away_points_for": 15.8,
            "home_points_against": 16.8,
            "away_points_against": 25.4,
            "h2h_wins_home": 4,
            "h2h_wins_away": 1,
            "h2h_avg_margin": 10.5,
            "home_def_rank": 4,
            "away_def_rank": 26,
            "home_pass_def_rank": 5,
            "away_pass_def_rank": 24,
            "home_rush_def_rank": 6,
            "away_rush_def_rank": 28,
            "temperature": 70,
            "wind_speed": 8,
            "precipitation": 0,
            "home_starters_available": 0.98,
            "away_starters_available": 0.80,
        },
        {
            "date": "2023-09-17",
            "home_team": "BUF",
            "away_team": "LV",
            "home_score": 38,
            "away_score": 10,
            "total_points": 48,
            "spread": 28,
            "home_win": 1,
            "season": 2023,
            "week": 2,
            "home_dome": 0,
            "away_dome": 1,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "hot",
            "mars_strength": 0.9,
            "jupiter_strength": 0.7,
            "saturn_strength": 0.8,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.8,
            "away_nakshatra_score": 0.3,
            "planetary_alignment": 0.8,
            "moon_phase_score": 0.9,
            "home_win_streak": 2,
            "away_win_streak": 0,
            "home_points_for": 32.5,
            "away_points_for": 15.8,
            "home_points_against": 14.2,
            "away_points_against": 27.9,
            "h2h_wins_home": 3,
            "h2h_wins_away": 2,
            "h2h_avg_margin": 6.8,
            "home_def_rank": 2,
            "away_def_rank": 27,
            "home_pass_def_rank": 3,
            "away_pass_def_rank": 26,
            "home_rush_def_rank": 4,
            "away_rush_def_rank": 29,
            "temperature": 65,
            "wind_speed": 10,
            "precipitation": 0,
            "home_starters_available": 0.96,
            "away_starters_available": 0.82,
        },
        {
            "date": "2023-10-01",
            "home_team": "BUF",
            "away_team": "MIA",
            "home_score": 35,
            "away_score": 14,
            "total_points": 49,
            "spread": 21,
            "home_win": 1,
            "season": 2023,
            "week": 4,
            "home_dome": 0,
            "away_dome": 0,
            "is_division_game": 1,
            "is_conference_game": 1,
            "home_climate": "cold",
            "away_climate": "hot",
            "mars_strength": 0.9,
            "jupiter_strength": 0.8,
            "saturn_strength": 0.7,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.95,
            "away_nakshatra_score": 0.3,
            "planetary_alignment": 0.9,
            "moon_phase_score": 0.9,
            "home_win_streak": 4,
            "away_win_streak": 0,
            "home_points_for": 32.5,
            "away_points_for": 15.8,
            "home_points_against": 14.2,
            "away_points_against": 27.9,
            "h2h_wins_home": 4,
            "h2h_wins_away": 1,
            "h2h_avg_margin": 11.2,
            "home_def_rank": 1,
            "away_def_rank": 28,
            "home_pass_def_rank": 2,
            "away_pass_def_rank": 27,
            "home_rush_def_rank": 3,
            "away_rush_def_rank": 30,
            "temperature": 40,
            "wind_speed": 15,
            "precipitation": 0,
            "home_starters_available": 0.98,
            "away_starters_available": 0.80,
        },
    ]

    return pd.DataFrame(games)


def calculate_features(df):
    """Calculate features for each game."""
    try:
        # Initialize Vedic calculator
        vedic_calc = NFLVedicCalculator()

        # Basic features
        df["home_dome"] = df["home_team"].apply(
            lambda x: int(get_team_info(x).get("dome", False))
        )
        df["away_dome"] = df["away_team"].apply(
            lambda x: int(get_team_info(x).get("dome", False))
        )
        df["is_division_game"] = df.apply(
            lambda x: get_team_info(x["home_team"])["division"]
            == get_team_info(x["away_team"])["division"],
            axis=1,
        )
        df["is_conference_game"] = df.apply(
            lambda x: get_team_info(x["home_team"])["conference"]
            == get_team_info(x["away_team"])["conference"],
            axis=1,
        )

        # Climate features
        df["home_climate"] = df["home_team"].apply(
            lambda x: get_team_info(x).get("climate", "moderate")
        )
        df["away_climate"] = df["away_team"].apply(
            lambda x: get_team_info(x).get("climate", "moderate")
        )

        # Convert dates to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Calculate Vedic features
        for idx, row in df.iterrows():
            game_date = row["date"].strftime("%Y-%m-%d")
            try:
                vedic_features = vedic_calc.calculate_game_features(
                    game_date, row["home_team"], row["away_team"]
                )
                for feature, value in vedic_features.items():
                    df.at[idx, feature] = value
            except Exception as e:
                logger.warning(
                    f"Failed to calculate Vedic features for game {idx}: {e}"
                )
                # Set default values for Vedic features
                default_vedic_features = {
                    "mars_strength": 0.5,
                    "jupiter_strength": 0.5,
                    "saturn_strength": 0.5,
                    "team_yoga": 0.5,
                    "nakshatra_score": 0.5,
                    "planetary_alignment": 0.5,
                    "moon_phase_score": 0.5,
                }
                for feature, value in default_vedic_features.items():
                    df.at[idx, feature] = value

        return df
    except Exception as e:
        logger.error(f"Error calculating features: {e}")
        raise


def train_and_evaluate():
    """Train and evaluate the NFL prediction system."""
    try:
        # Create sample dataset
        df = create_sample_data()

        # Initialize prediction system
        system = NFLPredictionSystem()

        # Encode climate features
        climate_types = ["hot", "warm", "moderate", "cold"]
        for climate in climate_types:
            df[f"home_climate_{climate}"] = (df["home_climate"] == climate).astype(int)
            df[f"away_climate_{climate}"] = (df["away_climate"] == climate).astype(int)

        # Prepare features and targets for cross-validation
        X = df[system.feature_columns]
        y = {
            "total": df["total_points"].values,
            "spread": df["spread"].values,
            "win": df["home_win"].values,
        }

        # Perform cross-validation
        metrics = cross_validate_model(
            system, X, y, n_splits=2
        )  # Using 2 splits for small sample

        # Log average metrics
        logger.info("Cross-validation results:")
        for metric, values in metrics.items():
            logger.info(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")

        # Train on full dataset for final evaluation
        system.train(df)

        # Make predictions on test cases
        test_games = create_test_cases()
        predictions = []

        for game in test_games:
            # Add climate encoding to test data
            for climate in climate_types:
                game[f"home_climate_{climate}"] = (
                    1 if game.get("home_climate") == climate else 0
                )
                game[f"away_climate_{climate}"] = (
                    1 if game.get("away_climate") == climate else 0
                )

            # Validate input data
            game_df = pd.DataFrame([game])
            is_valid, error_msg = validate_input_data(game_df, system.feature_columns)

            if not is_valid:
                logger.error(f"Invalid input data: {error_msg}")
                continue

            # Make prediction
            pred = system.predict(game_df)
            predictions.append(pred)

            logger.info(f"Prediction for {game['home_team']} vs {game['away_team']}:")
            logger.info(f"Total Points: {pred['total_points']:.1f}")
            logger.info(f"Spread: {pred['spread']:.1f}")
            logger.info(f"Home Win Probability: {pred['win_prob']:.1%}")
            logger.info(f"Confidence: {pred['confidence']:.1%}\n")

    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {str(e)}")
        raise


def create_test_cases():
    """Create test cases for prediction evaluation."""
    return [
        {
            "date": "2023-12-10",
            "home_team": "BUF",
            "away_team": "KC",
            "season": 2023,
            "week": 14,
            "home_dome": 0,
            "away_dome": 0,
            "is_division_game": 0,
            "is_conference_game": 1,
            "home_climate": "cold",
            "away_climate": "moderate",
            "mars_strength": 0.7,
            "jupiter_strength": 0.8,
            "saturn_strength": 0.6,
            "home_team_yoga": 1,
            "away_team_yoga": 1,
            "home_nakshatra_score": 0.8,
            "away_nakshatra_score": 0.7,
            "planetary_alignment": 0.6,
            "moon_phase_score": 0.7,
            "home_win_streak": 4,
            "away_win_streak": 0,
            "home_points_for": 32.5,
            "away_points_for": 15.8,
            "home_points_against": 14.2,
            "away_points_against": 27.9,
            "h2h_wins_home": 3,
            "h2h_wins_away": 2,
            "h2h_avg_margin": 5.5,
            "home_def_rank": 1,
            "away_def_rank": 25,
            "home_pass_def_rank": 2,
            "away_pass_def_rank": 23,
            "home_rush_def_rank": 3,
            "away_rush_def_rank": 28,
            "temperature": 40,
            "wind_speed": 15,
            "precipitation": 0,
            "home_starters_available": 0.98,
            "away_starters_available": 0.80,
        },
        {
            "date": "2023-12-10",
            "home_team": "LAR",
            "away_team": "BAL",
            "season": 2023,
            "week": 14,
            "home_dome": 1,
            "away_dome": 0,
            "is_division_game": 0,
            "is_conference_game": 0,
            "home_climate": "warm",
            "away_climate": "moderate",
            "mars_strength": 0.6,
            "jupiter_strength": 0.7,
            "saturn_strength": 0.8,
            "home_team_yoga": 0,
            "away_team_yoga": 1,
            "home_nakshatra_score": 0.6,
            "away_nakshatra_score": 0.8,
            "planetary_alignment": 0.7,
            "moon_phase_score": 0.8,
            "home_win_streak": 3,
            "away_win_streak": 0,
            "home_points_for": 29.8,
            "away_points_for": 16.5,
            "home_points_against": 15.8,
            "away_points_against": 26.2,
            "h2h_wins_home": 2,
            "h2h_wins_away": 3,
            "h2h_avg_margin": 4.2,
            "home_def_rank": 3,
            "away_def_rank": 24,
            "home_pass_def_rank": 4,
            "away_pass_def_rank": 22,
            "home_rush_def_rank": 5,
            "away_rush_def_rank": 26,
            "temperature": 60,
            "wind_speed": 10,
            "precipitation": 0,
            "home_starters_available": 0.92,
            "away_starters_available": 0.85,
        },
        {
            "date": "2023-12-10",
            "home_team": "DAL",
            "away_team": "PHI",
            "season": 2023,
            "week": 14,
            "home_dome": 1,
            "away_dome": 0,
            "is_division_game": 1,
            "is_conference_game": 1,
            "home_climate": "moderate",
            "away_climate": "moderate",
            "mars_strength": 0.8,
            "jupiter_strength": 0.6,
            "saturn_strength": 0.7,
            "home_team_yoga": 1,
            "away_team_yoga": 0,
            "home_nakshatra_score": 0.9,
            "away_nakshatra_score": 0.6,
            "planetary_alignment": 0.8,
            "moon_phase_score": 0.6,
            "home_win_streak": 4,
            "away_win_streak": 0,
            "home_points_for": 29.8,
            "away_points_for": 16.5,
            "home_points_against": 15.8,
            "away_points_against": 26.2,
            "h2h_wins_home": 3,
            "h2h_wins_away": 2,
            "h2h_avg_margin": 5.1,
            "home_def_rank": 2,
            "away_def_rank": 23,
            "home_pass_def_rank": 3,
            "away_pass_def_rank": 21,
            "home_rush_def_rank": 4,
            "away_rush_def_rank": 25,
            "temperature": 50,
            "wind_speed": 12,
            "precipitation": 0,
            "home_starters_available": 0.95,
            "away_starters_available": 0.85,
        },
    ]


def test_full_dataset():
    """Test the prediction system on the full dataset."""
    try:
        # Load and prepare data
        data = pd.read_csv("data/nfl_games.csv")
        data["game_datetime"] = pd.to_datetime(data["game_datetime"])

        # Sort by date to maintain temporal order
        data = data.sort_values("game_datetime")

        # Calculate target variables
        data["total_points"] = data["home_team_score"] + data["away_team_score"]
        data["spread"] = data["home_team_score"] - data["away_team_score"]
        data["home_win"] = (data["spread"] > 0).astype(int)

        # Initialize prediction system
        system = NFLPredictionSystem()

        # Initialize metrics
        total_predictions = 0
        correct_predictions = 0
        high_confidence_predictions = 0
        high_confidence_correct = 0
        total_points_mae = []
        spread_mae = []
        confidences = []

        # Use rolling window for training to simulate real-world predictions
        window_size = 400  # Use 400 games for initial training

        # Ensure we have enough data for initial training
        if len(data) <= window_size:
            logging.error("Not enough data for training window")
            return

        for i in range(window_size, len(data)):
            try:
                # Get training data
                train_data = data.iloc[i - window_size : i].copy()
                test_game = data.iloc[i : i + 1].copy()  # Keep as DataFrame

                # Skip if not enough unique values for qcut
                if (
                    len(train_data["home_team_points_per_game"].unique()) < 10
                    or len(train_data["away_team_points_per_game"].unique()) < 10
                ):
                    logging.warning(
                        f"Skipping game {i}: Not enough unique values for ranking"
                    )
                    continue

                # Train models
                system.train(train_data)

                # Make prediction
                prediction = system.predict(test_game)

                # Update metrics
                total_predictions += 1

                # Win/loss accuracy
                predicted_win = prediction["spread"] > 0
                actual_win = test_game["spread"].iloc[0] > 0
                if predicted_win == actual_win:
                    correct_predictions += 1

                # High confidence predictions
                if prediction["confidence"] >= 0.85:
                    high_confidence_predictions += 1
                    if predicted_win == actual_win:
                        high_confidence_correct += 1

                # MAE for total points and spread
                total_points_mae.append(
                    abs(prediction["total_points"] - test_game["total_points"].iloc[0])
                )
                spread_mae.append(
                    abs(prediction["spread"] - test_game["spread"].iloc[0])
                )
                confidences.append(prediction["confidence"])

                # Log progress every 100 games
                if total_predictions % 100 == 0:
                    logging.info(f"Processed {total_predictions} predictions")
                    logging.info(
                        f"Current accuracy: {correct_predictions/total_predictions:.3f}"
                    )
                    if high_confidence_predictions > 0:
                        logging.info(
                            f"High confidence accuracy: {high_confidence_correct/high_confidence_predictions:.3f}"
                        )
                    logging.info(
                        f"Average total points MAE: {np.mean(total_points_mae):.1f}"
                    )
                    logging.info(f"Average spread MAE: {np.mean(spread_mae):.1f}")
                    logging.info(f"Average confidence: {np.mean(confidences):.3f}")

            except Exception as e:
                logging.error(f"Error processing game {i}: {str(e)}")
                logging.error(f"Traceback: {traceback.format_exc()}")
                continue

        # Calculate final metrics
        if total_predictions == 0:
            logging.error("No predictions were made")
            return

        overall_accuracy = correct_predictions / total_predictions
        high_confidence_accuracy = (
            high_confidence_correct / high_confidence_predictions
            if high_confidence_predictions > 0
            else 0
        )
        avg_total_points_mae = np.mean(total_points_mae)
        avg_spread_mae = np.mean(spread_mae)
        avg_confidence = np.mean(confidences)

        # Print results
        print("\nFinal Results:")
        print(f"Total predictions: {total_predictions}")
        print(f"Overall accuracy: {overall_accuracy:.3f}")
        print(f"High confidence predictions: {high_confidence_predictions}")
        print(f"High confidence accuracy: {high_confidence_accuracy:.3f}")
        print(f"Average total points MAE: {avg_total_points_mae:.1f}")
        print(f"Average spread MAE: {avg_spread_mae:.1f}")
        print(f"Average confidence: {avg_confidence:.3f}")

        # Additional analysis
        if (
            len(confidences) >= 10
        ):  # Only do confidence analysis if we have enough predictions
            try:
                confidence_bins = pd.qcut(confidences, q=10)
                accuracy_by_confidence = pd.DataFrame(
                    {
                        "confidence_range": confidence_bins.unique(),
                        "accuracy": [
                            sum(
                                (np.array(confidences) >= bin.left)
                                & (np.array(confidences) < bin.right)
                                & (
                                    np.array([p > 0 for p in spread_mae])
                                    == np.array(
                                        [
                                            a > 0
                                            for a in data.iloc[window_size:]["spread"]
                                        ]
                                    )
                                )
                            )
                            / sum(
                                (np.array(confidences) >= bin.left)
                                & (np.array(confidences) < bin.right)
                            )
                            for bin in confidence_bins.unique()
                        ],
                    }
                )

                print("\nAccuracy by Confidence Range:")
                print(accuracy_by_confidence.to_string(index=False))
            except Exception as e:
                logging.error(f"Error in confidence analysis: {str(e)}")

        return {
            "overall_accuracy": overall_accuracy,
            "high_confidence_accuracy": high_confidence_accuracy,
            "avg_total_points_mae": avg_total_points_mae,
            "avg_spread_mae": avg_spread_mae,
            "avg_confidence": avg_confidence,
        }

    except Exception as e:
        print(f"Error in test_full_dataset: {str(e)}")
        raise


def test_high_confidence_predictions():
    """Test the prediction system with focus on finding high-confidence consensus predictions."""
    # Create sample data
    df = create_sample_data()

    # Initialize prediction systems
    nfl_system = NFLPredictionSystem()
    vedic_calc = NFLVedicCalculator()

    # Split data for testing
    train_data = df.iloc[:4]  # First 4 games for training
    test_data = df.iloc[4:]  # Last 2 games for testing

    # Train the systems
    nfl_system.train(train_data)
    vedic_calc.train(train_data)

    # Make predictions on test data
    predictions = []
    confidences = []
    actuals = []

    for _, game in test_data.iterrows():
        nfl_pred = nfl_system.predict(game)
        vedic_pred = vedic_calc.predict(game)

        consensus = {
            "all_agree": nfl_pred["win_prob"] > 0.5 and vedic_pred["win_prob"] > 0.5,
            "predicted_winner": game["home_team"]
            if nfl_pred["win_prob"] > 0.5
            else game["away_team"],
            "avg_confidence": (nfl_pred["confidence"] + vedic_pred["confidence"]) / 2,
            "max_confidence": max(nfl_pred["confidence"], vedic_pred["confidence"]),
            "nfl_prediction": nfl_pred,
            "vedic_prediction": vedic_pred,
        }

        predictions.append(consensus)
        confidences.append(consensus["avg_confidence"])
        actuals.append(
            {
                "total_points": game["total_points"],
                "spread": game["spread"],
                "home_win": game["home_win"],
            }
        )

    # Analyze predictions
    print("\nPrediction Analysis:")
    print("-" * 50)

    for i, (pred, conf, actual) in enumerate(zip(predictions, confidences, actuals)):
        print(f"\nGame {i+1}:")
        print(f"Confidence: {conf:.2%}")
        print(f"Predicted Winner: {pred['predicted_winner']}")
        print(
            f"Actual Winner: {game['home_team'] if actual['home_win'] else game['away_team']}"
        )

        if pred["all_agree"]:
            print("*** ALL MODELS AGREE ***")

        if conf >= 0.85:
            print("*** HIGH CONFIDENCE PREDICTION ***")

    # Calculate overall metrics
    high_conf_count = sum(1 for conf in confidences if conf >= 0.85)
    print(f"\nNumber of high confidence predictions (≥85%): {high_conf_count}")

    avg_confidence = sum(confidences) / len(confidences)
    print(f"Average confidence: {avg_confidence:.2%}")


def predict_game(
    game_date, game_time, home_team, away_team, timezone="America/New_York"
):
    """Make predictions using all available models."""
    try:
        # Initialize prediction systems
        nfl_system = NFLPredictionSystem()
        vedic_calc = NFLVedicCalculator()

        # Get predictions from each model
        predictions = {}

        # 1. Statistical Model Predictions
        try:
            stats_pred = nfl_system.predict_game(
                home_team=home_team, away_team=away_team, game_date=game_date
            )
            predictions["statistical"] = {
                "home_win_prob": stats_pred["win_probability"],
                "confidence": stats_pred["confidence"],
                "model_agreement": stats_pred.get("model_agreement", False),
            }
        except Exception as e:
            logger.error(f"Statistical prediction error: {str(e)}")
            predictions["statistical"] = {
                "home_win_prob": 0.5,
                "confidence": 0.0,
                "model_agreement": False,
            }

        # 2. Vedic Astrology Prediction
        try:
            vedic_pred = vedic_calc.predict_influence(
                game_date=game_date,
                game_time=game_time,
                timezone=timezone,
                home_team=home_team,
                away_team=away_team,
            )
            predictions["vedic"] = {
                "home_win_prob": vedic_pred["home_win_probability"],
                "confidence": vedic_pred["confidence"],
                "features": vedic_pred.get("features", {}),
            }
        except Exception as e:
            logger.error(f"Vedic prediction error: {str(e)}")
            predictions["vedic"] = {
                "home_win_prob": 0.5,
                "confidence": 0.0,
                "features": {},
            }

        # Calculate overall consensus
        home_win_predictions = [
            pred["home_win_prob"] > 0.5 for pred in predictions.values()
        ]
        confidence_scores = [pred["confidence"] for pred in predictions.values()]

        consensus = {
            "all_agree": len(set(home_win_predictions)) == 1,
            "predicted_winner": home_team
            if np.mean([p["home_win_prob"] for p in predictions.values()]) > 0.5
            else away_team,
            "avg_confidence": np.mean(confidence_scores),
            "max_confidence": max(confidence_scores),
            "predictions": predictions,
        }

        return consensus

    except Exception as e:
        logger.error(f"Error in predict_game: {str(e)}")
        return {
            "all_agree": False,
            "predicted_winner": None,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "predictions": {},
        }


def test_predictions():
    """Test predictions for upcoming games."""

    # List of games to predict
    games = [
        {
            "date": "2024-12-08",
            "time": "13:00:00",
            "home": "KC",
            "away": "SF",
            "description": "Super Bowl LIV Rematch",
        },
        {
            "date": "2024-12-08",
            "time": "16:25:00",
            "home": "GB",
            "away": "NYG",
            "description": "Classic NFC Matchup",
        },
        {
            "date": "2024-12-08",
            "time": "20:20:00",
            "home": "NE",
            "away": "NYJ",
            "description": "AFC East Rivalry",
        },
    ]

    print("\nNFL Game Predictions - Multi-Model Analysis")
    print("===========================================")

    for game in games:
        try:
            print(f"\n{game['description']}")
            print(f"{game['away']} @ {game['home']}")
            print(f"Date: {game['date']} {game['time']} ET")

            result = predict_game(
                game_date=game["date"],
                game_time=game["time"],
                home_team=game["home"],
                away_team=game["away"],
            )

            if result["predicted_winner"]:
                print("\nPrediction Results:")
                print(f"Consensus Winner: {result['predicted_winner']}")
                print(f"All Models Agree: {'Yes' if result['all_agree'] else 'No'}")
                print(f"Average Confidence: {result['avg_confidence']:.2%}")
                print(f"Maximum Confidence: {result['max_confidence']:.2%}")

                print("\nDetailed Model Predictions:")
                for model, pred in result["predictions"].items():
                    winner = (
                        game["home"] if pred["home_win_prob"] > 0.5 else game["away"]
                    )
                    print(f"{model.title()} Model:")
                    print(f"  Winner: {winner}")
                    print(
                        f"  Win Probability: {abs(pred['home_win_prob'] - 0.5) * 2 + 0.5:.2%}"
                    )
                    print(f"  Confidence: {pred['confidence']:.2%}")

                # Check for high-confidence consensus
                high_confidence = (
                    result["all_agree"]
                    and result["avg_confidence"] > 0.80
                    and result["max_confidence"] > 0.85
                )

                if high_confidence:
                    print("\n🌟 HIGH CONFIDENCE PREDICTION 🌟")
                    print(f"Strong consensus for {result['predicted_winner']} victory")
            else:
                print("\nUnable to generate prediction")

            print("\n" + "=" * 45)

        except Exception as e:
            logger.error(f"Error processing game: {str(e)}")
            print("\nError generating prediction")
            print("=" * 45)


if __name__ == "__main__":
    test_predictions()
