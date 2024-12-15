"""
Astrological model for total points prediction.
"""

from typing import Dict, Any
import numpy as np
from .total_model import TotalPredictionModel
import swisseph as swe


class AstroTotalModel(TotalPredictionModel):
    def __init__(self):
        super().__init__()
        self.name = "Astrological Total Prediction Model"

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make astrological predictions for total points.

        Args:
            game_data: Dictionary containing game information

        Returns:
            Dictionary containing predictions
        """
        # Calculate astrological factors
        astro_factors = self._calculate_astro_factors(game_data)

        # Make prediction
        predicted_total = self._predict_total(astro_factors)
        confidence = self._calculate_confidence(astro_factors)

        return {
            "predicted_total": predicted_total,
            "confidence": confidence,
            "recommendation": self._get_recommendation(predicted_total, confidence),
        }

    def _calculate_astro_factors(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate astrological factors for the game."""
        # Get game time in Julian date
        jd = float(game_data.get("julian_date", swe.julday(2024, 12, 7, 1.5)))

        # Get planet positions
        sun_pos = swe.calc_ut(jd, swe.SUN)[0]
        moon_pos = swe.calc_ut(jd, swe.MOON)[0]
        mars_pos = swe.calc_ut(jd, swe.MARS)[0]
        jupiter_pos = swe.calc_ut(jd, swe.JUPITER)[0]

        return {
            "sun_long": sun_pos[0],
            "moon_long": moon_pos[0],
            "mars_long": mars_pos[0],
            "jupiter_long": jupiter_pos[0],
        }

    def _predict_total(self, astro_factors: Dict[str, float]) -> float:
        """Predict total points based on astrological factors."""
        # Basic prediction based on Moon-Jupiter angle
        moon_jupiter_angle = abs(
            astro_factors["moon_long"] - astro_factors["jupiter_long"]
        )
        if moon_jupiter_angle > 180:
            moon_jupiter_angle = 360 - moon_jupiter_angle

        # Higher angles tend to indicate higher scoring games
        base_total = 45.0 + (moon_jupiter_angle / 180.0) * 10.0

        return base_total

    def _calculate_confidence(self, astro_factors: Dict[str, float]) -> float:
        """Calculate confidence based on astrological factors."""
        # Higher confidence when Mars aspects Jupiter
        mars_jupiter_angle = abs(
            astro_factors["mars_long"] - astro_factors["jupiter_long"]
        )
        if mars_jupiter_angle > 180:
            mars_jupiter_angle = 360 - mars_jupiter_angle

        # Higher confidence for strong aspects (0°, 60°, 90°, 120°, 180°)
        aspect_strengths = [0, 60, 90, 120, 180]
        min_angle_diff = min(
            abs(mars_jupiter_angle - aspect) for aspect in aspect_strengths
        )

        confidence = 1.0 - (min_angle_diff / 30.0)  # Within 30° of aspect
        return max(0.5, min(confidence, 0.9))  # Keep between 0.5 and 0.9

    def _get_recommendation(self, predicted_total: float, confidence: float) -> str:
        """Get betting recommendation based on prediction and confidence."""
        if confidence < 0.6:
            return "PASS"
        return "OVER" if predicted_total > 45 else "UNDER"
