"""NFL Combined Model integrating various prediction methods."""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLCombinedModel:
    """Combines various NFL prediction models into a unified prediction system."""

    def __init__(self):
        """Initialize the combined model with component weights."""
        self.model_weights = {
            "vedic": 0.25,
            "astro": 0.15,
            "ml": 0.35,
            "statistical": 0.25,
        }

        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.model_performance = {}

    def prepare_features(self, game_data):
        """Prepare features for prediction."""
        try:
            features = {
                "team_stats": self._prepare_team_stats(game_data),
                "game_context": self._prepare_game_context(game_data),
                "historical": self._prepare_historical_features(game_data),
                "astrological": self._prepare_astro_features(game_data),
            }

            return pd.concat([v for k, v in features.items()], axis=1)

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def _prepare_team_stats(self, game_data):
        """Prepare team statistical features."""
        features = pd.DataFrame()

        try:
            # Offensive stats
            features["off_yards_diff"] = (
                game_data["home_off_yards"] - game_data["away_off_yards"]
            )
            features["pass_yards_diff"] = (
                game_data["home_pass_yards"] - game_data["away_pass_yards"]
            )
            features["rush_yards_diff"] = (
                game_data["home_rush_yards"] - game_data["away_rush_yards"]
            )
            features["turnover_diff"] = (
                game_data["home_turnovers"] - game_data["away_turnovers"]
            )

            # Defensive stats
            features["def_yards_diff"] = (
                game_data["home_def_yards"] - game_data["away_def_yards"]
            )
            features["sacks_diff"] = game_data["home_sacks"] - game_data["away_sacks"]
            features["interceptions_diff"] = (
                game_data["home_ints"] - game_data["away_ints"]
            )

            # Special teams
            features["special_teams_diff"] = (
                game_data["home_special"] - game_data["away_special"]
            )

        except Exception as e:
            logger.error(f"Error preparing team stats: {str(e)}")

        return features

    def _prepare_game_context(self, game_data):
        """Prepare game context features."""
        features = pd.DataFrame()

        try:
            # Home/Away context
            features[
                "is_home"
            ] = 1  # Since we're always predicting from home team perspective
            features["home_rest_days"] = game_data["home_rest_days"]
            features["away_rest_days"] = game_data["away_rest_days"]

            # Game importance
            features["is_division"] = game_data["is_division_game"]
            features["is_conference"] = game_data["is_conference_game"]
            features["week_number"] = game_data["week"]
            features["is_primetime"] = game_data["is_primetime"]

            # Weather features (if available)
            if "temperature" in game_data.columns:
                features["temperature"] = game_data["temperature"]
                features["is_precipitation"] = game_data["is_precipitation"]
                features["wind_speed"] = game_data["wind_speed"]

        except Exception as e:
            logger.error(f"Error preparing game context: {str(e)}")

        return features

    def _prepare_historical_features(self, game_data):
        """Prepare historical matchup features."""
        features = pd.DataFrame()

        try:
            # Recent form
            features["home_form"] = game_data["home_last5_wins"] / 5
            features["away_form"] = game_data["away_last5_wins"] / 5

            # Head-to-head history
            features["h2h_home_wins"] = game_data["h2h_home_wins"]
            features["h2h_total_games"] = game_data["h2h_total_games"]

            # Season performance
            features["home_win_pct"] = (
                game_data["home_season_wins"] / game_data["home_games_played"]
            )
            features["away_win_pct"] = (
                game_data["away_season_wins"] / game_data["away_games_played"]
            )

        except Exception as e:
            logger.error(f"Error preparing historical features: {str(e)}")

        return features

    def _prepare_astro_features(self, game_data):
        """Prepare astrological features."""
        features = pd.DataFrame()

        try:
            # Team zodiac compatibility
            features["zodiac_compatibility"] = game_data["zodiac_compatibility"]

            # Planetary influences
            features["home_planetary_strength"] = game_data["home_planetary_strength"]
            features["away_planetary_strength"] = game_data["away_planetary_strength"]

            # Stadium astrological influence
            features["stadium_astro_power"] = game_data["stadium_astro_power"]

        except Exception as e:
            logger.error(f"Error preparing astrological features: {str(e)}")

        return features

    def combine_predictions(self, predictions_dict):
        """Combine predictions from different models using weighted average."""
        try:
            final_prediction = 0
            total_weight = 0

            for model_name, pred in predictions_dict.items():
                if model_name in self.model_weights:
                    weight = self.model_weights[model_name]
                    final_prediction += pred * weight
                    total_weight += weight

            if total_weight > 0:
                final_prediction /= total_weight
            else:
                final_prediction = 0.5  # Default to uncertain prediction

            return final_prediction

        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return 0.5

    def adjust_for_special_circumstances(self, prediction, game_data):
        """Adjust predictions for special circumstances."""
        try:
            # Injury adjustments
            if game_data["home_qb_out"]:
                prediction *= 0.8
            if game_data["away_qb_out"]:
                prediction *= 1.2

            # Weather adjustments
            if game_data.get("extreme_weather", False):
                prediction = 0.6 * prediction + 0.4 * 0.5  # Regress towards uncertainty

            # Late season motivation
            if game_data["week"] > 14:
                if game_data["home_playoff_odds"] > game_data["away_playoff_odds"]:
                    prediction *= 1.1
                else:
                    prediction *= 0.9

            return min(max(prediction, 0), 1)  # Ensure prediction stays between 0 and 1

        except Exception as e:
            logger.error(f"Error adjusting prediction: {str(e)}")
            return prediction

    def save_model(self, path="models/nfl_combined_model.joblib"):
        """Save the combined model to disk."""
        try:
            model_data = {
                "weights": self.model_weights,
                "scaler": self.scaler,
                "feature_importance": self.feature_importance,
                "model_performance": self.model_performance,
            }

            joblib.dump(model_data, path)
            logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path="models/nfl_combined_model.joblib"):
        """Load the combined model from disk."""
        try:
            model_data = joblib.load(path)

            self.model_weights = model_data["weights"]
            self.scaler = model_data["scaler"]
            self.feature_importance = model_data["feature_importance"]
            self.model_performance = model_data["model_performance"]

            logger.info(f"Model loaded successfully from {path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def update_weights(self, performance_metrics):
        """Update model weights based on recent performance."""
        try:
            total_performance = sum(performance_metrics.values())

            if total_performance > 0:
                self.model_weights = {
                    model: score / total_performance
                    for model, score in performance_metrics.items()
                }

            logger.info("Model weights updated successfully")

        except Exception as e:
            logger.error(f"Error updating weights: {str(e)}")

    def predict(self, game_data):
        """Generate final prediction for a game."""
        try:
            # Prepare features
            features = self.prepare_features(game_data)

            # Get predictions from each model
            predictions = {
                "vedic": self._get_vedic_prediction(game_data),
                "astro": self._get_astro_prediction(game_data),
                "ml": self._get_ml_prediction(features),
                "statistical": self._get_statistical_prediction(features),
            }

            # Combine predictions
            combined_pred = self.combine_predictions(predictions)

            # Adjust for special circumstances
            final_pred = self.adjust_for_special_circumstances(combined_pred, game_data)

            return {
                "prediction": final_pred,
                "component_predictions": predictions,
                "confidence": self._calculate_confidence(predictions),
            }

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            return None

    def _get_vedic_prediction(self, game_data):
        """Get prediction from Vedic astrology model."""
        try:
            # Implementation would come from vedic_calculations.py
            return 0.5  # Placeholder
        except Exception as e:
            logger.error(f"Error in Vedic prediction: {str(e)}")
            return 0.5

    def _get_astro_prediction(self, game_data):
        """Get prediction from astronomical model."""
        try:
            # Implementation would come from astro_calculations.py
            return 0.5  # Placeholder
        except Exception as e:
            logger.error(f"Error in Astro prediction: {str(e)}")
            return 0.5

    def _get_ml_prediction(self, features):
        """Get prediction from machine learning model."""
        try:
            # Implementation would come from ml_model.py
            return 0.5  # Placeholder
        except Exception as e:
            logger.error(f"Error in ML prediction: {str(e)}")
            return 0.5

    def _get_statistical_prediction(self, features):
        """Get prediction from statistical model."""
        try:
            # Implementation would come from statistical_model.py
            return 0.5  # Placeholder
        except Exception as e:
            logger.error(f"Error in Statistical prediction: {str(e)}")
            return 0.5

    def _calculate_confidence(self, predictions):
        """Calculate confidence level in the prediction."""
        try:
            # Calculate standard deviation of predictions
            pred_values = list(predictions.values())
            std_dev = np.std(pred_values)

            # Higher agreement (lower std_dev) = higher confidence
            confidence = 1 - (std_dev * 2)  # Scale to 0-1
            return max(min(confidence, 1), 0)  # Ensure between 0 and 1

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
