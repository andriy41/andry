"""
Training module for Vedic NFL prediction model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime
import logging
import os
from typing import Dict, Any, List, Tuple

from .vedic_model import VedicModel
from .calculations import (
    calculate_planet_strength,
    calculate_planetary_alignment_score,
    calculate_sarvashtakavarga,
    check_shadbala,
    check_vimshottari_dasa,
    calculate_divisional_strength,
    calculate_bhava_chalit_aspects,
    calculate_special_lagnas,
    calculate_victory_yogas,
    calculate_nakshatra_tara,
    calculate_sublords,
    calculate_retrograde_impact,
    calculate_moon_phase,
    calculate_muhurta_score,
    calculate_hora_score,
    get_house_lord,
    get_nakshatra_lord,
    PLANETS,
    LATITUDE,
    LONGITUDE,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_nfl_data(data_path: str) -> pd.DataFrame:
    """
    Load NFL game data from CSV

    Args:
        data_path: Path to CSV file containing NFL game data

    Returns:
        DataFrame containing NFL game data

    Raises:
        FileNotFoundError: If data file does not exist
        pd.errors.EmptyDataError: If data file is empty
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise pd.errors.EmptyDataError("Data file is empty")

    return df


def prepare_vedic_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare Vedic astrological features for all games

    Args:
        df: DataFrame containing NFL game data

    Returns:
        Tuple containing:
            - Array of Vedic features
            - List of feature names
    """
    features_list = []
    feature_names = []

    for _, row in df.iterrows():
        try:
            # Convert game datetime (handle both date and datetime formats)
            game_date_str = row["game_date"]
            try:
                game_time = datetime.strptime(game_date_str, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                # If no time is provided, use noon of that day
                game_time = datetime.strptime(game_date_str.split("T")[0], "%Y-%m-%d")
                game_time = game_time.replace(hour=12)

            # Calculate Vedic features
            features = {
                "sun_strength": calculate_planet_strength("Sun", game_time),
                "moon_strength": calculate_planet_strength("Moon", game_time),
                "mars_strength": calculate_planet_strength("Mars", game_time),
                "jupiter_strength": calculate_planet_strength("Jupiter", game_time),
                "venus_strength": calculate_planet_strength("Venus", game_time),
                "saturn_strength": calculate_planet_strength("Saturn", game_time),
                "planetary_alignment": calculate_planetary_alignment_score(game_time),
                "sarvashtakavarga": calculate_sarvashtakavarga(game_time),
                "shadbala": check_shadbala(game_time),
                "vimshottari_dasa": check_vimshottari_dasa(game_time),
                "divisional_strength": calculate_divisional_strength(game_time),
                "bhava_chalit": calculate_bhava_chalit_aspects(game_time),
                "special_lagnas": calculate_special_lagnas(game_time),
                "victory_yogas": calculate_victory_yogas(game_time),
                "nakshatra_tara": calculate_nakshatra_tara(game_time),
                "sublords": calculate_sublords(game_time),
                "retrograde_impact": calculate_retrograde_impact(game_time),
                "moon_phase": calculate_moon_phase(game_time),
                "muhurta_score": calculate_muhurta_score(game_time),
                "hora_score": calculate_hora_score(game_time),
            }

            if not feature_names:
                feature_names = list(features.keys())

            features_list.append(list(features.values()))

        except Exception as e:
            logger.error(
                f"Error calculating features for game {game_date_str}: {str(e)}"
            )
            # Use neutral values for failed calculations
            if not feature_names:
                feature_names = ["feature_" + str(i) for i in range(20)]
            features_list.append([0.5] * 20)

    return np.array(features_list), feature_names


def prepare_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare labels (home team win = 1, loss = 0)

    Args:
        df: DataFrame containing NFL game data

    Returns:
        Array of binary labels
    """
    return (df["home_score"] > df["away_score"]).astype(int).values


def train_model(data_file: str, model_save_path: str = None) -> VedicModel:
    """
    Train the Vedic model on NFL game data

    Args:
        data_file: Path to CSV file containing NFL game data
        model_save_path: Optional path to save trained model

    Returns:
        Trained VedicModel instance

    Raises:
        FileNotFoundError: If data file does not exist
        ValueError: If data processing fails
    """
    try:
        logger.info("Loading NFL game data...")
        df = load_nfl_data(data_file)

        logger.info("Preparing features...")
        X, feature_names = prepare_vedic_features(df)

        logger.info("Preparing labels...")
        y = prepare_labels(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize and train model
        logger.info("Training Vedic model...")
        model = VedicModel()
        model.train({"features": X_train, "labels": y_train})

        # Evaluate model
        logger.info("Evaluating model...")
        metrics = model.evaluate({"features": X_test, "labels": y_test})
        logger.info(f"Model performance:\n{metrics}")

        # Save model if path provided
        if model_save_path:
            logger.info(f"Saving model to {model_save_path}...")
            model.save_model(model_save_path)

        return model

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise


def main():
    """Main entry point for training the Vedic model"""
    try:
        # Set paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, "data", "nfl_games.csv")
        model_save_path = os.path.join(
            current_dir, "saved_models", "vedic_model.joblib"
        )

        # Create save directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # Train model
        train_model(data_file, model_save_path)
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
