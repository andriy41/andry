"""NFL prediction model integrator."""

import sys
import os
import logging
from datetime import datetime
import numpy as np

# Add parent directory to path for shared utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.combined_predictor import CombinedPredictor
from src.utils.trend_analyzer import TrendAnalyzer
from src.models.total_prediction.neuro_total_model import NeuroTotalModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLModelIntegrator:
    def __init__(self):
        """Initialize NFL model integrator."""
        self.combined_predictor = CombinedPredictor()
        self.trend_analyzer = TrendAnalyzer()
        self.total_predictor = NeuroTotalModel()

        # NFL-specific weight adjustments
        self.model_weights = {
            "vedic": 0.20,
            "advanced": 0.35,  # Higher for NFL due to importance of matchups and stats
            "ml": 0.25,
            "basic": 0.20,
        }

        # NFL-specific factors
        self.specific_factors = [
            "quarterback_rating",
            "weather_conditions",
            "injury_impact",
            "rest_days",
            "home_field",
            "divisional_game",
        ]

    def predict_game(self, home_team, away_team, game_date, game_info=None):
        """Generate comprehensive NFL game prediction."""
        try:
            # Get predictions from each model
            vedic_pred = self._get_vedic_prediction(home_team, away_team, game_date)
            advanced_pred = self._get_advanced_prediction(
                home_team, away_team, game_info
            )
            ml_pred = self._get_ml_prediction(home_team, away_team, game_info)
            basic_pred = self._get_basic_prediction(home_team, away_team)

            # Get total points prediction
            total_pred = self._get_total_prediction(home_team, away_team, game_info)

            # Analyze trends
            home_trends = self.trend_analyzer.analyze_team_trends(
                home_team.recent_games, "NFL"
            )
            away_trends = self.trend_analyzer.analyze_team_trends(
                away_team.recent_games, "NFL"
            )

            # Get QB and weather information
            qb_comparison = self._compare_quarterbacks(home_team, away_team)
            weather_impact = self._analyze_weather_impact(game_info)

            # Combine predictions with NFL-specific adjustments
            combined_prediction = self.combined_predictor.predict_game(
                vedic_pred, advanced_pred, ml_pred, basic_pred, "NFL"
            )

            # Apply NFL-specific adjustments
            final_prediction = self._adjust_for_nfl_factors(
                combined_prediction,
                home_trends,
                away_trends,
                qb_comparison,
                weather_impact,
                game_info,
            )

            # Calculate prediction variance and confidence metrics
            variance_metrics = self._calculate_prediction_variance(
                [vedic_pred, advanced_pred, ml_pred, basic_pred]
            )

            return {
                "win_probability": final_prediction,
                "confidence": variance_metrics["confidence"],
                "confidence_level": variance_metrics["mean"],
                "total_points": total_pred["predicted_total"],
                "total_confidence": total_pred["confidence_score"],
                "total_factors": total_pred["factors"],
                "significant_trends": {
                    "home": self.trend_analyzer.get_trend_summary(home_trends),
                    "away": self.trend_analyzer.get_trend_summary(away_trends),
                },
                "qb_factor": qb_comparison,
                "weather_impact": weather_impact,
                "model_contributions": combined_prediction["model_contributions"],
            }

        except Exception as e:
            logger.error(f"Error in NFL prediction: {str(e)}")
            return None

    def _get_vedic_prediction(self, home_team, away_team, game_date):
        """Get prediction from Vedic astrology model."""
        try:
            from src.models.vedic_basic.vedic_model import VedicModel

            model = VedicModel()
            prediction = model.predict_game(home_team, away_team, game_date)
            return prediction["probability"]
        except Exception as e:
            logger.error(f"Error in Vedic prediction: {str(e)}")
            return 0.5

    def _get_advanced_prediction(self, home_team, away_team, game_info):
        """Get prediction from advanced stats model."""
        try:
            from src.models.advanced_system.advanced_model import AdvancedModel

            model = AdvancedModel()
            prediction = model.predict_game(home_team, away_team, game_info)
            return prediction["probability"]
        except Exception as e:
            logger.error(f"Error in advanced prediction: {str(e)}")
            return 0.5

    def _get_ml_prediction(self, home_team, away_team, game_info):
        """Get prediction from machine learning model."""
        try:
            from src.models.combined_ml.combined_model import CombinedMLModel

            model = CombinedMLModel()
            prediction = model.predict_game(home_team, away_team, game_info)
            return prediction["probability"]
        except Exception as e:
            logger.error(f"Error in ML prediction: {str(e)}")
            return 0.5

    def _get_basic_prediction(self, home_team, away_team):
        """Get prediction from basic stats model."""
        try:
            from src.models.sports_only.sports_model import SportsModel

            model = SportsModel()
            prediction = model.predict_game(home_team, away_team)
            return prediction["probability"]
        except Exception as e:
            logger.error(f"Error in basic prediction: {str(e)}")
            return 0.5

    def _get_total_prediction(self, home_team, away_team, game_info):
        """Get total points prediction."""
        try:
            # Prepare game data for total prediction
            game_data = {
                "home_stats": home_team.stats,
                "away_stats": away_team.stats,
                "is_division_game": game_info.get("is_divisional_game", False),
                "is_primetime": game_info.get("is_primetime", False),
                "days_rest": game_info.get("rest_days", {"home": 7, "away": 7}),
                "playoff_implications": game_info.get("playoff_implications", False),
            }

            # Add weather info if available
            if game_info and "weather" in game_info:
                game_data.update(
                    {
                        "temperature": game_info["weather"].get("temperature", 70),
                        "wind_speed": game_info["weather"].get("wind_speed", 0),
                        "is_dome": game_info["weather"].get("is_dome", False),
                        "precipitation_chance": game_info["weather"].get(
                            "precipitation", 0
                        ),
                        "altitude": game_info["weather"].get("altitude", 0),
                    }
                )

            prediction = self.total_predictor.predict(game_data)
            return prediction

        except Exception as e:
            logger.error(f"Error in total points prediction: {str(e)}")
            return {
                "predicted_total": 44.5,  # League average as fallback
                "confidence_score": 0.5,
                "factors": {},
            }

    def _compare_quarterbacks(self, home_team, away_team):
        """Compare quarterback statistics and performance."""
        try:
            # Implementation would compare actual QB stats
            return {
                "qb_rating_difference": 0.0,
                "performance_trend": 0.0,
                "injury_factor": 0.0,
            }
        except Exception as e:
            logger.error(f"Error comparing quarterbacks: {str(e)}")
            return None

    def _analyze_weather_impact(self, game_info):
        """Analyze impact of weather conditions."""
        try:
            if not game_info or "weather" not in game_info:
                return None

            weather = game_info["weather"]
            impact = {
                "passing_game": self._calculate_weather_impact(weather, "passing"),
                "running_game": self._calculate_weather_impact(weather, "running"),
                "kicking_game": self._calculate_weather_impact(weather, "kicking"),
            }
            return impact
        except Exception as e:
            logger.error(f"Error analyzing weather impact: {str(e)}")
            return None

    def _adjust_for_nfl_factors(
        self,
        prediction,
        home_trends,
        away_trends,
        qb_comparison,
        weather_impact,
        game_info,
    ):
        """Apply NFL-specific adjustments to prediction."""
        try:
            base_prob = prediction["combined_probability"]
            adjustment = 0.0

            # QB impact
            if qb_comparison:
                qb_factor = qb_comparison["qb_rating_difference"] * 0.15
                adjustment += qb_factor

            # Weather impact
            if weather_impact:
                weather_factor = (
                    weather_impact["passing_game"] * 0.4
                    + weather_impact["running_game"] * 0.4
                    + weather_impact["kicking_game"] * 0.2
                )
                adjustment += weather_factor * 0.1

            # Divisional game impact
            if game_info.get("is_divisional_game"):
                adjustment *= 0.8  # Reduce impact of other factors in divisional games

            # Rest advantage
            if "rest_days" in game_info:
                rest_difference = (
                    game_info["rest_days"]["home"] - game_info["rest_days"]["away"]
                )
                adjustment += rest_difference * 0.02

            # Apply adjustment with limits
            final_prob = max(0.1, min(0.9, base_prob + adjustment))
            return final_prob

        except Exception as e:
            logger.error(f"Error adjusting for NFL factors: {str(e)}")
            return prediction["combined_probability"]

    def _calculate_weather_impact(self, weather, game_aspect):
        """Calculate weather impact on specific aspect of the game."""
        try:
            impact = 0.0

            if game_aspect == "passing":
                impact -= weather.get("wind_speed", 0) * 0.02
                impact -= weather.get("precipitation", 0) * 0.05
            elif game_aspect == "running":
                impact += weather.get("wind_speed", 0) * 0.01
                impact -= weather.get("precipitation", 0) * 0.02
            elif game_aspect == "kicking":
                impact -= weather.get("wind_speed", 0) * 0.03
                impact -= weather.get("precipitation", 0) * 0.04

            return max(-1, min(1, impact))  # Normalize to [-1, 1]
        except Exception as e:
            logger.error(f"Error calculating weather impact: {str(e)}")
            return 0.0

    def _calculate_prediction_variance(self, predictions):
        """Calculate prediction variance and confidence metrics."""
        try:
            # Convert predictions to numpy array
            pred_array = np.array(predictions)

            # Calculate basic statistics
            mean_pred = np.mean(pred_array)
            std_pred = np.std(pred_array)

            # Check if all models agree on winner
            all_agree = np.all(pred_array > 0.5) or np.all(pred_array < 0.5)

            # Check confidence thresholds
            high_confidence = np.all(
                np.abs(pred_array - 0.5) >= 0.3
            )  # All models >80% confident
            very_high_confidence = np.any(
                np.abs(pred_array - 0.5) >= 0.35
            )  # At least one >85% confident

            # Calculate confidence score
            if all_agree and high_confidence and very_high_confidence:
                confidence = "Strong"
            elif all_agree and high_confidence:
                confidence = "Good"
            elif all_agree:
                confidence = "Moderate"
            else:
                confidence = "Weak"

            return {
                "confidence": confidence,
                "mean": mean_pred,
                "std": std_pred,
                "agreement": all_agree,
            }

        except Exception as e:
            logger.error(f"Error calculating prediction variance: {str(e)}")
            return {"confidence": "Weak", "mean": 0.5, "std": 0.0, "agreement": False}

    def update_model_accuracy(self, game_result):
        """Update accuracy tracking for all models."""
        try:
            self.combined_predictor.update_model_accuracy(
                game_result["predictions"], game_result["actual_result"]
            )
        except Exception as e:
            logger.error(f"Error updating model accuracy: {str(e)}")
