"""
NFL prediction model trainer using Vedic astrology.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime
import pytz

from ..nfl_vedic_calculator import NFLVedicCalculator
from ..utils.team_data import TEAM_DATA, TEAM_ALIASES
from ..utils.astro_calculations import calculate_planet_positions, calculate_aspects
from ..utils.planet_calculations import calculate_planet_strength
from ..utils.house_calculations import calculate_house_strength

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLPredictionModelTrainer:
    """Trainer for the NFL prediction model using Vedic Astrology features."""

    def __init__(self, seasons=None, model_dir="saved_models"):
        """Initialize the prediction model trainer.

        Args:
            seasons (list): List of seasons to train on (e.g., [2020, 2021, 2022])
            model_dir (str): Directory to save trained models
        """
        self.seasons = seasons or list(range(2020, 2024))
        self.model_dir = model_dir
        self.calculator = NFLVedicCalculator()

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize containers
        self.X = []  # Features
        self.y = []  # Labels (1 for home win, 0 for away win)
        self.feature_names = None
        self.model = None
        self.scaler = None

    def process_historical_data(self):
        """Process historical NFL game data for model training."""
        logger.info(f"Processing NFL data for seasons: {self.seasons}")

        try:
            # Load game data
            games = nfl.import_schedules(self.seasons)

            # Filter for completed games with scores
            games = games[
                (games["home_score"].notna())
                & (games["away_score"].notna())
                & (games["gameday"].notna())
                & (games["gametime"].notna())
            ]

            # Process each game
            for _, game in games.iterrows():
                try:
                    # Extract astrological features
                    features = self._extract_game_features(game)
                    if features:
                        self.X.append(features)
                        # 1 if home team won, 0 if away team won
                        self.y.append(
                            1 if game["home_score"] > game["away_score"] else 0
                        )

                except Exception as e:
                    logger.error(f"Error processing game {game['gameday']}: {str(e)}")
                    continue

            # Convert to numpy arrays
            self.X = np.array(self.X)
            self.y = np.array(self.y)

            logger.info(f"Successfully processed {len(self.X)} games")

        except Exception as e:
            logger.error(f"Error processing NFL data: {str(e)}")
            raise

    def _extract_game_features(self, game):
        """Extract Vedic astrology features for a game."""
        try:
            # Parse game datetime
            game_date = game["gameday"]
            game_time = game["gametime"]

            # Get week number and playoff status
            week_num = game["week"]
            is_playoff = week_num > 18 if week_num else False

            # Calculate astrological features
            features = self.calculator.calculate_game_features(
                game_date=game_date,
                game_time=game_time,
                timezone="America/New_York",  # NFL games use Eastern Time
                home_team=game["home_team"],
                away_team=game["away_team"],
                week_number=week_num,
                is_playoff=is_playoff,
            )

            if not features:
                return None

            # Extract features in a consistent order
            feature_list = [
                features["moon_phase"],
                features["home_planet_strength"],
                features["away_planet_strength"],
                features["beneficial_aspects"],
                features["malefic_aspects"],
                features["home_strength"],
                features["away_strength"],
                features["raja_yoga"],
                features["dhana_yoga"],
                features["vipreet_yoga"],
                features["kesari_yoga"],
            ]

            # Store feature names if not already stored
            if self.feature_names is None:
                self.feature_names = list(features.keys())

            return feature_list

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None

    def build_and_train_model(self, test_size=0.2, random_state=42):
        """Build and train the prediction model."""
        try:
            logger.info("Building and training model...")

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )

            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
            )

            self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)

            logger.info(f"Training accuracy: {train_score:.3f}")
            logger.info(f"Testing accuracy: {test_score:.3f}")

            # Print detailed classification report
            y_pred = self.model.predict(X_test_scaled)
            logger.info("\nClassification Report:")
            logger.info(classification_report(y_test, y_pred))

            # Analyze feature importance
            self._analyze_feature_importance()

            return train_score, test_score

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def _analyze_feature_importance(self):
        """Analyze and log feature importance."""
        if self.model and self.feature_names:
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame(
                {"feature": self.feature_names, "importance": importances}
            )
            feature_imp = feature_imp.sort_values("importance", ascending=False)

            logger.info("\nFeature Importance Rankings:")
            for _, row in feature_imp.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.3f}")

    def export_model(self, model_name="nfl_prediction_model"):
        """Export the trained model and scaler."""
        try:
            if self.model is None:
                raise ValueError("No trained model to export")

            # Save model
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            joblib.dump(self.model, model_path)

            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)

            logger.info(f"Model exported to {model_path}")
            logger.info(f"Scaler exported to {scaler_path}")

        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            raise


def main():
    """Main execution function."""
    try:
        # Initialize trainer
        trainer = NFLPredictionModelTrainer(seasons=[2020, 2021, 2022, 2023])

        # Process historical data
        trainer.process_historical_data()

        # Build and train model
        train_score, test_score = trainer.build_and_train_model()

        # Export model
        trainer.export_model()

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
