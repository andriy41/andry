"""
Vedic astrology model for total points prediction.
"""

from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
import swisseph as swe
from .total_model import TotalPredictionModel
import logging

logger = logging.getLogger(__name__)


class VedicTotalModel(TotalPredictionModel):
    def __init__(self):
        super().__init__()
        self.model_name = "Vedic Total Prediction Model"
        self.is_trained = False

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make Vedic astrology-based predictions for total points.

        Args:
            game_data: Dictionary containing game information

        Returns:
            Dictionary containing predictions
        """
        # Calculate Vedic factors
        vedic_factors = self._calculate_vedic_factors(game_data)

        # Make prediction
        predicted_total = self._predict_total(vedic_factors)
        confidence = self._calculate_confidence(vedic_factors)

        return {
            "predicted_total": predicted_total,
            "confidence": confidence,
            "recommendation": self._get_recommendation(predicted_total, confidence),
            "vedic_factors": vedic_factors,
        }

    def train(self, training_data: Dict[str, Any]) -> None:
        """
        Train the model using historical data

        Args:
            training_data: Dictionary containing training data
        """
        try:
            # Extract features and labels
            X = training_data["features"]
            y = training_data["labels"]

            # Train the model (implement your training logic here)
            # For now, we'll just set some baseline weights
            self.weights = np.ones(X.shape[1]) / X.shape[1]
            self.is_trained = True

        except Exception as e:
            raise ValueError(f"Training failed: {str(e)}")

    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            test_data: Dictionary containing test data

        Returns:
            Dictionary of performance metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        try:
            # Get predictions
            X_test = test_data["features"]
            y_test = test_data["labels"]

            predictions = np.array(
                [self.predict({"features": x})["predicted_total"] for x in X_test]
            )

            # Calculate metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))

            return {"mse": float(mse), "mae": float(mae), "rmse": float(np.sqrt(mse))}

        except Exception as e:
            raise ValueError(f"Evaluation failed: {str(e)}")

    def _calculate_vedic_factors(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Vedic astrological factors for the game."""
        try:
            game_time = game_data.get("game_time", datetime.now())
            jd = float(
                swe.julday(
                    game_time.year,
                    game_time.month,
                    game_time.day,
                    game_time.hour
                    + game_time.minute / 60.0
                    + game_time.second / 3600.0,
                )
            )

            # Get planet positions in sidereal zodiac
            swe.set_sid_mode(swe.SIDM_LAHIRI)  # Use Lahiri ayanamsa

            # Calculate Sun position
            sun_result = swe.calc_ut(jd, swe.SUN, swe.FLG_SIDEREAL)
            if sun_result is None:
                raise ValueError("Could not calculate Sun position")
            sun_values = sun_result[0] if isinstance(sun_result, tuple) else sun_result
            if not isinstance(sun_values, (list, tuple)) or len(sun_values) < 1:
                raise ValueError("Invalid Sun position data")
            sun_pos = float(sun_values[0])

            # Calculate Moon position
            moon_result = swe.calc_ut(jd, swe.MOON, swe.FLG_SIDEREAL)
            if moon_result is None:
                raise ValueError("Could not calculate Moon position")
            moon_values = (
                moon_result[0] if isinstance(moon_result, tuple) else moon_result
            )
            if not isinstance(moon_values, (list, tuple)) or len(moon_values) < 1:
                raise ValueError("Invalid Moon position data")
            moon_pos = float(moon_values[0])

            # Calculate Mars position
            mars_result = swe.calc_ut(jd, swe.MARS, swe.FLG_SIDEREAL)
            if mars_result is None:
                raise ValueError("Could not calculate Mars position")
            mars_values = (
                mars_result[0] if isinstance(mars_result, tuple) else mars_result
            )
            if not isinstance(mars_values, (list, tuple)) or len(mars_values) < 1:
                raise ValueError("Invalid Mars position data")
            mars_pos = float(mars_values[0])

            # Calculate Jupiter position
            jupiter_result = swe.calc_ut(jd, swe.JUPITER, swe.FLG_SIDEREAL)
            if jupiter_result is None:
                raise ValueError("Could not calculate Jupiter position")
            jupiter_values = (
                jupiter_result[0]
                if isinstance(jupiter_result, tuple)
                else jupiter_result
            )
            if not isinstance(jupiter_values, (list, tuple)) or len(jupiter_values) < 1:
                raise ValueError("Invalid Jupiter position data")
            jupiter_pos = float(jupiter_values[0])

            # Calculate aspects and strengths
            moon_phase = (moon_pos - sun_pos) % 360
            mars_jupiter_aspect = abs((mars_pos - jupiter_pos) % 180 - 90)

            return {
                "moon_phase": moon_phase / 360.0,  # Normalize to [0,1]
                "mars_jupiter_harmony": 1.0 - (mars_jupiter_aspect / 90.0),
                "sun_strength": self._calculate_dignity(sun_pos),
                "moon_strength": self._calculate_dignity(moon_pos),
                "mars_strength": self._calculate_dignity(mars_pos),
                "jupiter_strength": self._calculate_dignity(jupiter_pos),
            }

        except Exception as e:
            logger.error(f"Error calculating Vedic factors: {str(e)}")
            # Return neutral values on error
            return {
                "moon_phase": 0.5,
                "mars_jupiter_harmony": 0.5,
                "sun_strength": 0.5,
                "moon_strength": 0.5,
                "mars_strength": 0.5,
                "jupiter_strength": 0.5,
            }

    def _calculate_dignity(self, position: float) -> float:
        """Calculate planetary dignity (simplified)"""
        house = int(position / 30)
        return 0.5 + 0.5 * np.sin(np.radians(position))  # Simple cyclic strength

    def _predict_total(self, vedic_factors: Dict[str, float]) -> float:
        """Predict total points based on Vedic factors"""
        if not self.is_trained:
            return 45.0  # League average as fallback

        # Convert factors to features array
        features = np.array(
            [
                vedic_factors["moon_phase"],
                vedic_factors["mars_jupiter_harmony"],
                vedic_factors["sun_strength"],
                vedic_factors["moon_strength"],
                vedic_factors["mars_strength"],
                vedic_factors["jupiter_strength"],
            ]
        )

        # Apply weights and base prediction
        base_prediction = np.dot(features, self.weights) * 100

        # Ensure reasonable range
        return max(30.0, min(80.0, base_prediction))

    def _calculate_confidence(self, vedic_factors: Dict[str, float]) -> float:
        """Calculate prediction confidence based on Vedic factors"""
        # Higher confidence when planets are strong and well-aspected
        return min(
            1.0,
            max(
                0.0,
                np.mean(
                    [
                        vedic_factors["mars_jupiter_harmony"],
                        vedic_factors["sun_strength"],
                        vedic_factors["moon_strength"],
                    ]
                ),
            ),
        )

    def _get_recommendation(self, predicted_total: float, confidence: float) -> str:
        """Get betting recommendation based on prediction and confidence"""
        if confidence < 0.6:
            return "No strong indication - avoid betting"
        elif predicted_total > 50:
            return "Consider OVER if line is significantly lower"
        else:
            return "Consider UNDER if line is significantly higher"
