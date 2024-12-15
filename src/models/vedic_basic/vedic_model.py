"""NFL Vedic Astrology Prediction Model"""

import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import logging
import pandas as pd
from src.models.vedic_basic.astro_calculator import AstroCalculator
from src.models.base_model import NFLPredictionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VedicModel(NFLPredictionModel):
    """Enhanced Vedic astrology model for NFL prediction using ensemble of models"""

    def __init__(self):
        """Initialize the Vedic model with enhanced ensemble components"""
        super().__init__()
        self.scaler = StandardScaler()
        self.is_trained = False

        # Initialize ensemble models
        self.xgb_model = None
        self.rf_model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
        )

        self.gb_model = GradientBoostingClassifier(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
        )

        # Historical performance tracking
        self.team_performance_cache = {}
        self.confidence_threshold = 0.65  # Lower confidence threshold for predictions

        # Team stadium mapping including relocated teams
        self.team_stadium_mapping = {
            "OAK": "LV",  # Oakland -> Las Vegas
            "STL": "LA",  # St. Louis -> Los Angeles
            "SD": "LAC",  # San Diego -> Los Angeles Chargers
        }

        # Enhanced feature list
        self.feature_names = [
            # Basic planetary positions (0-360 degrees)
            "Sun_longitude",
            "Moon_longitude",
            "Mars_longitude",
            "Mercury_longitude",
            "Jupiter_longitude",
            "Venus_longitude",
            "Saturn_longitude",
            "Rahu_longitude",
            "Ketu_longitude",
            # Aspects between planets
            "Sun_Moon_aspect",
            "Sun_Mars_aspect",
            "Sun_Mercury_aspect",
            "Sun_Jupiter_aspect",
            "Sun_Venus_aspect",
            "Sun_Saturn_aspect",
            "Sun_Rahu_aspect",
            "Sun_Ketu_aspect",
            "Moon_Mars_aspect",
            "Moon_Mercury_aspect",
            "Moon_Jupiter_aspect",
            "Moon_Venus_aspect",
            "Moon_Saturn_aspect",
            "Moon_Rahu_aspect",
            "Moon_Ketu_aspect",
            "Mars_Mercury_aspect",
            "Mars_Jupiter_aspect",
            "Mars_Venus_aspect",
            "Mars_Saturn_aspect",
            "Mars_Rahu_aspect",
            "Mars_Ketu_aspect",
            "Mercury_Jupiter_aspect",
            "Mercury_Venus_aspect",
            "Mercury_Saturn_aspect",
            "Mercury_Rahu_aspect",
            "Mercury_Ketu_aspect",
            "Jupiter_Venus_aspect",
            "Jupiter_Saturn_aspect",
            "Jupiter_Rahu_aspect",
            "Jupiter_Ketu_aspect",
            "Venus_Saturn_aspect",
            "Venus_Rahu_aspect",
            "Venus_Ketu_aspect",
            "Saturn_Rahu_aspect",
            "Saturn_Ketu_aspect",
            "Rahu_Ketu_aspect",
            # Planetary strengths
            "Sun_strength",
            "Moon_strength",
            "Mars_strength",
            "Mercury_strength",
            "Jupiter_strength",
            "Venus_strength",
            "Saturn_strength",
            # Team historical performance features
            "home_win_rate",
            "home_recent_form",
            "away_win_rate",
            "away_recent_form",
            "home_advantage",
        ]

        # Initialize neural network
        self.nn_model = self._build_neural_network()

        # Initialize astro calculator
        self.astro = AstroCalculator()

    def _build_neural_network(self):
        """Build an enhanced neural network model"""
        model = Sequential(
            [
                Input(shape=(len(self.feature_names),)),
                Dense(256, activation="relu"),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.1),
                Dense(32, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def calculate_vedic_features(self, game_data):
        """Calculate Vedic astrology features for a game"""
        home_team = game_data.get("home_team")
        away_team = game_data.get("away_team")
        game_date = game_data.get("date")

        if not all([home_team, away_team, game_date]):
            logger.error(
                f"Missing required game data: home_team={home_team}, away_team={away_team}, date={game_date}"
            )
            return None

        try:
            # Convert date string to datetime
            game_datetime = pd.to_datetime(game_date)

            # Get stadium coordinates directly from game data
            stadium_coords = {
                "lat": float(game_data.get("stadium_latitude", 0)),
                "lon": float(game_data.get("stadium_longitude", 0)),
            }

            # Calculate planetary positions and aspects
            planets = self.astro.calculate_planet_positions(
                game_datetime, stadium_coords
            )
            if not planets:
                logger.error("Could not calculate planetary positions")
                return self._get_default_vedic_features()

            # Enhanced feature engineering
            features = {}

            # Basic planetary positions (0-360 degrees)
            for planet, pos in planets.items():
                features[f"{planet}_longitude"] = float(pos % 360)

            # Calculate aspects between planets
            for p1 in planets:
                for p2 in planets:
                    if p1 < p2:  # Avoid duplicate calculations
                        aspect = abs(planets[p1] - planets[p2]) % 360
                        features[f"{p1}_{p2}_aspect"] = float(min(aspect, 360 - aspect))

            # Calculate planetary strengths
            for planet, pos in planets.items():
                # Exaltation points (simplified)
                exalt_points = {
                    "Sun": 0,  # Aries
                    "Moon": 30,  # Taurus
                    "Mars": 300,  # Capricorn
                    "Mercury": 150,  # Virgo
                    "Jupiter": 90,  # Cancer
                    "Venus": 330,  # Pisces
                    "Saturn": 180,  # Libra
                }

                if planet in exalt_points:
                    dist_from_exalt = abs(pos - exalt_points[planet]) % 360
                    strength = 1 - (min(dist_from_exalt, 360 - dist_from_exalt) / 180)
                    features[f"{planet}_strength"] = float(strength)

            # Team historical performance features
            home_perf = self._get_team_performance(home_team, game_datetime)
            away_perf = self._get_team_performance(away_team, game_datetime)

            features.update(
                {
                    "home_win_rate": float(home_perf["win_rate"]),
                    "home_recent_form": float(home_perf["recent_form"]),
                    "away_win_rate": float(away_perf["win_rate"]),
                    "away_recent_form": float(away_perf["recent_form"]),
                    "home_advantage": 1.0,  # Home field advantage factor
                }
            )

            # Validate all features are present
            missing_features = set(self.feature_names) - set(features.keys())
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    features[feature] = 0.5  # Default value for missing features

            # Ensure all features are float values
            for feature in self.feature_names:
                if not isinstance(features[feature], float):
                    features[feature] = float(features[feature])

            return features

        except Exception as e:
            logger.error(f"Error calculating Vedic features: {str(e)}")
            return self._get_default_vedic_features()

    def _get_team_performance(self, team, date):
        """Get team's historical performance metrics"""
        if team not in self.team_performance_cache:
            self.team_performance_cache[team] = {
                "win_rate": 0.5,  # Default to neutral
                "recent_form": 0.5,
            }
        return self.team_performance_cache[team]

    def predict(self, game_data):
        """Enhanced prediction with ensemble and confidence thresholding"""
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return None

        # Calculate features
        features = self.calculate_vedic_features(game_data)
        if not features:
            return None

        # Convert features to array
        feature_values = [features[name] for name in self.feature_names]
        X = np.array([feature_values])

        # Get predictions from both models
        xgb_prob = self.xgb_model.predict_proba(X)[0]
        nn_prob = self.nn_model.predict(X)[0]

        # Ensemble prediction (weighted average)
        final_prob = 0.6 * xgb_prob[1] + 0.4 * nn_prob[0]

        # Check confidence threshold
        if abs(final_prob - 0.5) >= (self.confidence_threshold - 0.5):
            prediction = int(final_prob >= 0.5)
            confidence = max(final_prob, 1 - final_prob)

            return {
                "prediction": prediction,
                "confidence": confidence,
                "features": features,
            }
        else:
            logger.warning("Prediction confidence below threshold")
            return None

    def predict_with_confidence(self, game_data, confidence_threshold=0.75):
        """
        Make a prediction only if confidence exceeds the threshold
        Returns:
        - prediction: 1 for home team win, 0 for away team win, None if below confidence threshold
        - confidence: probability score between 0 and 1
        """
        try:
            features = self.calculate_vedic_features(game_data)
            if features is None:
                return None, 0.0

            # Scale features
            X = np.array([[features[name] for name in self.feature_names]])
            X_scaled = self.scaler.transform(X)

            # Get predictions from both models
            xgb_prob = self.xgb_model.predict_proba(X_scaled)[0]
            nn_prob = self.nn_model.predict(X_scaled)[0]

            # Average the probabilities
            home_win_prob = (xgb_prob[1] + nn_prob) / 2

            # Calculate confidence as distance from 0.5
            confidence = abs(home_win_prob - 0.5) * 2  # Scale to [0,1]

            if confidence >= confidence_threshold:
                prediction = 1 if home_win_prob > 0.5 else 0
                return prediction, confidence
            else:
                return None, confidence

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None, 0.0

    def predict_batch_with_confidence(self, games, confidence_threshold=0.75):
        """
        Make predictions for multiple games, returning only high-confidence predictions
        Returns list of (game, prediction, confidence) tuples for games above threshold
        """
        high_confidence_predictions = []

        for game in games:
            prediction, confidence = self.predict_with_confidence(
                game, confidence_threshold
            )
            if prediction is not None:  # Only include predictions above threshold
                high_confidence_predictions.append(
                    {
                        "game": game,
                        "prediction": prediction,
                        "confidence": confidence,
                        "home_team": game.get("home_team"),
                        "away_team": game.get("away_team"),
                        "date": game.get("date"),
                    }
                )

        return high_confidence_predictions

    def train_xgb(self, X_train, y_train):
        """Train XGBoost model"""
        params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        self.xgb_model = XGBClassifier(**params)
        self.xgb_model.fit(X_train, y_train)

    def train(self, train_data):
        """Enhanced training with cross-validation and early stopping"""
        if not train_data:
            logger.error("No training data provided")
            return False

        try:
            # Extract features and labels
            X = []
            y = []

            for game in train_data:
                features = self.calculate_vedic_features(game)
                if features is not None:
                    feature_values = [features[name] for name in self.feature_names]
                    if len(feature_values) == len(
                        self.feature_names
                    ):  # Ensure all features are present
                        X.append(feature_values)
                        y.append(
                            1
                            if float(game.get("home_score", 0))
                            > float(game.get("away_score", 0))
                            else 0
                        )

            if not X or not y:
                logger.error("No valid features extracted from training data")
                return False

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)

            logger.info(f"Training on {len(X)} samples")

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train XGBoost model
            self.train_xgb(X_scaled, y)

            # Train neural network model
            self.nn_model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=1)

            # Calculate training metrics
            train_metrics = self._calculate_metrics(X_scaled, y)
            logger.info(f"Training metrics: {train_metrics}")

            # Set trained flag
            self.is_trained = True

            return train_metrics["accuracy"] >= self.min_accuracy

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False

    def load_model(self, model_path: str) -> bool:
        """Load a trained model from disk"""
        try:
            with open(model_path, "rb") as f:
                loaded_data = joblib.load(f)
                self.xgb_model = loaded_data["xgb_model"]
                self.nn_model = loaded_data["nn_model"]
                self.scaler = loaded_data["scaler"]
                self.feature_names = loaded_data["feature_names"]
                self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def save_model(self, model_path: str) -> bool:
        """Save the trained model to disk"""
        if not self.is_trained:
            logger.error("Model not trained yet")
            return False

        try:
            model_data = {
                "xgb_model": self.xgb_model,
                "nn_model": self.nn_model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            }
            with open(model_path, "wb") as f:
                joblib.dump(model_data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def evaluate(self, test_data):
        """
        Evaluate model performance on test data

        Args:
            test_data: Dictionary containing test data similar to training_data

        Returns:
            Dictionary containing performance metrics:
                - accuracy: float
                - precision: float
                - recall: float
                - f1_score: float
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        predictions = []
        actuals = []

        for game, label in zip(test_data["games"], test_data["labels"]):
            pred = self.predict(game)
            if pred is not None:  # Only include predictions where model is confident
                predictions.append(pred["prediction"])
                actuals.append(label)

        if len(predictions) == 0:
            logger.warning("No confident predictions made during evaluation")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        # Calculate metrics only on games where we made predictions
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        metrics = {
            "accuracy": accuracy_score(actuals, predictions),
            "precision": precision_score(actuals, predictions),
            "recall": recall_score(actuals, predictions),
            "f1_score": f1_score(actuals, predictions),
        }

        return metrics

    def _get_default_vedic_features(self):
        """Return default values for Vedic features"""
        features = {}

        # Basic planetary positions (0-360 degrees)
        for planet in [
            "Sun",
            "Moon",
            "Mars",
            "Mercury",
            "Jupiter",
            "Venus",
            "Saturn",
            "Rahu",
            "Ketu",
        ]:
            features[f"{planet}_longitude"] = 0.5

        # Aspects between planets
        for p1 in [
            "Sun",
            "Moon",
            "Mars",
            "Mercury",
            "Jupiter",
            "Venus",
            "Saturn",
            "Rahu",
            "Ketu",
        ]:
            for p2 in [
                "Sun",
                "Moon",
                "Mars",
                "Mercury",
                "Jupiter",
                "Venus",
                "Saturn",
                "Rahu",
                "Ketu",
            ]:
                if p1 < p2:  # Avoid duplicate calculations
                    features[f"{p1}_{p2}_aspect"] = 0.5

        # Planetary strengths
        for planet in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]:
            features[f"{planet}_strength"] = 0.5

        # Team historical performance features
        features.update(
            {
                "home_win_rate": 0.5,
                "home_recent_form": 0.5,
                "away_win_rate": 0.5,
                "away_recent_form": 0.5,
                "home_advantage": 1.0,
            }
        )

        return features

    def _get_stadium_coordinates(self, team):
        """Get stadium coordinates from game data."""
        return {
            "lat": float(team.get("stadium_latitude", 0)),
            "lon": float(team.get("stadium_longitude", 0)),
        }
