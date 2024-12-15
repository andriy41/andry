"""
Train the NFL Vedic prediction model.
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


class NFLVedicModelTrainer:
    """Trainer for the NFL Vedic Astrology prediction model."""

    def __init__(self, seasons=None, model_dir="saved_models"):
        """Initialize the trainer.

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

    def load_nfl_data(self):
        """Load NFL game data for specified seasons."""
        logging.info(f"Loading NFL data for seasons: {self.seasons}")

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
                    # Calculate game features
                    features = self._calculate_game_features(game)
                    if features:
                        self.X.append(features)
                        # 1 if home team won, 0 if away team won
                        self.y.append(
                            1 if game["home_score"] > game["away_score"] else 0
                        )

                except Exception as e:
                    logging.error(f"Error processing game {game['gameday']}: {str(e)}")
                    continue

            # Convert to numpy arrays
            self.X = np.array(self.X)
            self.y = np.array(self.y)

            logging.info(f"Processed {len(self.X)} games successfully")

        except Exception as e:
            logging.error(f"Error loading NFL data: {str(e)}")
            raise

    def _calculate_game_features(self, game):
        """Calculate Vedic astrology features for a game."""
        try:
            # Parse game datetime
            game_date = game["gameday"]
            game_time = game["gametime"]

            # Get week number and playoff status
            week_num = game["week"]
            is_playoff = week_num > 18 if week_num else False

            # Calculate features using our calculator
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
            logging.error(f"Error calculating features: {str(e)}")
            return None

    def train_model(self, test_size=0.2, random_state=42):
        """Train the prediction model."""
        try:
            logging.info("Starting model training...")

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

            logging.info(f"Training accuracy: {train_score:.3f}")
            logging.info(f"Testing accuracy: {test_score:.3f}")

            # Print detailed classification report
            y_pred = self.model.predict(X_test_scaled)
            logging.info("\nClassification Report:")
            logging.info(classification_report(y_test, y_pred))

            # Save feature importance
            self._analyze_feature_importance()

            return train_score, test_score

        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

    def _analyze_feature_importance(self):
        """Analyze and log feature importance."""
        if self.model and self.feature_names:
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame(
                {"feature": self.feature_names, "importance": importances}
            )
            feature_imp = feature_imp.sort_values("importance", ascending=False)

            logging.info("\nFeature Importance:")
            for _, row in feature_imp.iterrows():
                logging.info(f"{row['feature']}: {row['importance']:.3f}")

    def save_model(self, model_name="nfl_vedic_model"):
        """Save the trained model and scaler."""
        try:
            if self.model is None:
                raise ValueError("No trained model to save")

            # Save model
            model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
            joblib.dump(self.model, model_path)

            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)

            logging.info(f"Model saved to {model_path}")
            logging.info(f"Scaler saved to {scaler_path}")

        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise


def main():
    """Main training function."""
    try:
        # Initialize trainer
        trainer = NFLVedicModelTrainer(seasons=[2020, 2021, 2022, 2023])

        # Load and process data
        trainer.load_nfl_data()

        # Train model
        train_score, test_score = trainer.train_model()

        # Save model
        trainer.save_model()

        logging.info("Training completed successfully!")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
