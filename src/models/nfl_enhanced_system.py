"""Enhanced NFL Prediction System combining multiple models and analysis methods."""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLEnhancedSystem:
    """Advanced NFL prediction system combining multiple models and analysis methods."""

    def __init__(self):
        """Initialize the enhanced NFL prediction system."""
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_tracker = {}

        # Model weights for ensemble
        self.model_weights = {
            "vedic": 0.15,
            "advanced": 0.25,
            "ml": 0.35,
            "statistical": 0.25,
        }

        # NFL-specific factors
        self.situational_factors = {
            "prime_time": 1.05,
            "division_game": 1.02,
            "playoff_game": 1.08,
            "must_win": 1.03,
            "short_rest": 0.97,
        }

    def load_models(self, model_dir):
        """Load all required models and scalers."""
        try:
            # Load Vedic model
            self.models["vedic"] = joblib.load(f"{model_dir}/nfl_vedic_model.joblib")

            # Load Advanced Analytics model
            self.models["advanced"] = joblib.load(
                f"{model_dir}/nfl_advanced_model.joblib"
            )

            # Load ML model
            self.models["ml"] = joblib.load(f"{model_dir}/nfl_ml_model.joblib")

            # Load Statistical model
            self.models["statistical"] = joblib.load(
                f"{model_dir}/nfl_statistical_model.joblib"
            )

            # Load scalers
            self.scalers["standard"] = joblib.load(f"{model_dir}/nfl_scaler.joblib")

            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def prepare_features(self, game_data):
        """Prepare features for prediction."""
        try:
            features = {}

            # Basic features
            features["basic"] = self.prepare_basic_features(game_data)

            # Advanced features
            features["advanced"] = self.prepare_advanced_features(game_data)

            # Situational features
            features["situational"] = self.prepare_situational_features(game_data)

            # Weather features if outdoor game
            if game_data.get("is_outdoor", False):
                features["weather"] = self.prepare_weather_features(game_data)

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None

    def prepare_basic_features(self, game_data):
        """Prepare basic statistical features."""
        try:
            features = {
                "home_win_pct": game_data.get("home_win_pct", 0.5),
                "away_win_pct": game_data.get("away_win_pct", 0.5),
                "home_points_per_game": game_data.get("home_points_per_game", 0),
                "away_points_per_game": game_data.get("away_points_per_game", 0),
                "home_points_allowed": game_data.get("home_points_allowed", 0),
                "away_points_allowed": game_data.get("away_points_allowed", 0),
            }
            return features

        except Exception as e:
            logger.error(f"Error preparing basic features: {str(e)}")
            return None

    def prepare_advanced_features(self, game_data):
        """Prepare advanced analytical features."""
        try:
            features = {
                "home_dvoa": game_data.get("home_dvoa", 0),
                "away_dvoa": game_data.get("away_dvoa", 0),
                "home_qb_rating": game_data.get("home_qb_rating", 0),
                "away_qb_rating": game_data.get("away_qb_rating", 0),
                "home_turnover_diff": game_data.get("home_turnover_diff", 0),
                "away_turnover_diff": game_data.get("away_turnover_diff", 0),
            }
            return features

        except Exception as e:
            logger.error(f"Error preparing advanced features: {str(e)}")
            return None

    def prepare_situational_features(self, game_data):
        """Prepare situational and context features."""
        try:
            features = {
                "is_division_game": game_data.get("is_division_game", False),
                "is_prime_time": game_data.get("is_prime_time", False),
                "home_rest_days": game_data.get("home_rest_days", 7),
                "away_rest_days": game_data.get("away_rest_days", 7),
                "home_travel_distance": game_data.get("home_travel_distance", 0),
                "away_travel_distance": game_data.get("away_travel_distance", 0),
            }
            return features

        except Exception as e:
            logger.error(f"Error preparing situational features: {str(e)}")
            return None

    def prepare_weather_features(self, game_data):
        """Prepare weather-related features for outdoor games."""
        try:
            features = {
                "temperature": game_data.get("temperature", 70),
                "wind_speed": game_data.get("wind_speed", 0),
                "precipitation": game_data.get("precipitation", 0),
                "humidity": game_data.get("humidity", 50),
            }
            return features

        except Exception as e:
            logger.error(f"Error preparing weather features: {str(e)}")
            return None

    def predict(self, game_data):
        """Make prediction using all models and combine results."""
        try:
            predictions = {}
            features = self.prepare_features(game_data)

            if not features:
                raise ValueError("Failed to prepare features")

            # Get predictions from each model
            for model_name, model in self.models.items():
                predictions[model_name] = self.get_model_prediction(
                    model, features, model_name
                )

            # Combine predictions with weights
            final_prediction = self.combine_predictions(predictions)

            # Apply situational adjustments
            final_prediction = self.apply_situational_adjustments(
                final_prediction, features["situational"]
            )

            return {
                "win_probability": final_prediction,
                "individual_predictions": predictions,
                "confidence": self.calculate_confidence(predictions),
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None

    def get_model_prediction(self, model, features, model_name):
        """Get prediction from a specific model."""
        try:
            if model_name == "vedic":
                return model.predict(features["basic"])
            elif model_name == "advanced":
                return model.predict(features["advanced"])
            elif model_name == "ml":
                # Combine all features for ML model
                all_features = {
                    **features["basic"],
                    **features["advanced"],
                    **features["situational"],
                }
                if "weather" in features:
                    all_features.update(features["weather"])
                return model.predict(all_features)
            elif model_name == "statistical":
                return model.predict(features["basic"])

        except Exception as e:
            logger.error(f"Error getting prediction from {model_name}: {str(e)}")
            return 0.5

    def combine_predictions(self, predictions):
        """Combine predictions from different models using weights."""
        try:
            final_prediction = 0
            for model_name, pred in predictions.items():
                weight = self.model_weights.get(model_name, 0.25)
                final_prediction += pred * weight
            return final_prediction

        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return 0.5

    def apply_situational_adjustments(self, prediction, situational_features):
        """Apply situational adjustments to the prediction."""
        try:
            adjustment = 1.0

            if situational_features.get("is_division_game"):
                adjustment *= self.situational_factors["division_game"]

            if situational_features.get("is_prime_time"):
                adjustment *= self.situational_factors["prime_time"]

            # Apply rest advantage/disadvantage
            home_rest = situational_features.get("home_rest_days", 7)
            if home_rest < 7:
                adjustment *= self.situational_factors["short_rest"]

            return prediction * adjustment

        except Exception as e:
            logger.error(f"Error applying situational adjustments: {str(e)}")
            return prediction

    def calculate_confidence(self, predictions):
        """Calculate prediction confidence based on model agreement."""
        try:
            # Convert predictions to binary
            binary_preds = {name: pred > 0.5 for name, pred in predictions.items()}

            # Calculate agreement ratio
            agreement = sum(
                1
                for pred in binary_preds.values()
                if pred == list(binary_preds.values())[0]
            )
            agreement_ratio = agreement / len(binary_preds)

            # Calculate average distance from 0.5
            avg_distance = np.mean([abs(pred - 0.5) for pred in predictions.values()])

            # Combine metrics
            confidence = (agreement_ratio + 2 * avg_distance) / 3

            return confidence

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def update_performance_tracker(self, prediction_results):
        """Update model performance tracking."""
        try:
            for model_name in self.models.keys():
                if model_name not in self.performance_tracker:
                    self.performance_tracker[model_name] = {
                        "total_predictions": 0,
                        "correct_predictions": 0,
                    }

                self.performance_tracker[model_name]["total_predictions"] += 1
                if prediction_results[model_name]["correct"]:
                    self.performance_tracker[model_name]["correct_predictions"] += 1

            self.update_model_weights()

        except Exception as e:
            logger.error(f"Error updating performance tracker: {str(e)}")

    def update_model_weights(self):
        """Update model weights based on recent performance."""
        try:
            total_accuracy = 0
            accuracies = {}

            # Calculate accuracy for each model
            for model_name, stats in self.performance_tracker.items():
                accuracy = stats["correct_predictions"] / stats["total_predictions"]
                accuracies[model_name] = accuracy
                total_accuracy += accuracy

            # Update weights based on relative accuracy
            for model_name in self.model_weights.keys():
                self.model_weights[model_name] = accuracies[model_name] / total_accuracy

        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")

    def save_models(self, model_dir):
        """Save all models and their current state."""
        try:
            for name, model in self.models.items():
                joblib.dump(model, f"{model_dir}/nfl_{name}_model.joblib")

            joblib.dump(self.scalers["standard"], f"{model_dir}/nfl_scaler.joblib")
            joblib.dump(self.model_weights, f"{model_dir}/nfl_weights.joblib")
            joblib.dump(self.performance_tracker, f"{model_dir}/nfl_performance.joblib")

            logger.info("Models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
