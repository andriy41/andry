"""Combined predictor for NFL games."""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class CombinedPredictor:
    def __init__(self):
        """Initialize combined predictor."""
        pass

    def predict_game(self, vedic_pred, advanced_pred, ml_pred, basic_pred, sport="NFL"):
        """Combine predictions from multiple models."""
        try:
            predictions = [vedic_pred, advanced_pred, ml_pred, basic_pred]
            weights = [0.2, 0.35, 0.25, 0.2]  # Default NFL weights

            # Calculate weighted average
            combined_prob = np.average(predictions, weights=weights)

            # Calculate model contributions
            contributions = {
                "vedic": vedic_pred * weights[0],
                "advanced": advanced_pred * weights[1],
                "ml": ml_pred * weights[2],
                "basic": basic_pred * weights[3],
            }

            return {
                "combined_probability": combined_prob,
                "model_contributions": contributions,
            }

        except Exception as e:
            logger.error(f"Error combining predictions: {str(e)}")
            return {
                "combined_probability": 0.5,
                "model_contributions": {
                    "vedic": 0.125,
                    "advanced": 0.125,
                    "ml": 0.125,
                    "basic": 0.125,
                },
            }

    def get_confidence_level(self, variance_metrics):
        """Get confidence level based on prediction variance."""
        try:
            # Higher variance = lower confidence
            confidence = 1 - np.clip(variance_metrics["std"], 0, 0.5)
            return confidence

        except Exception as e:
            logger.error(f"Error calculating confidence level: {str(e)}")
            return 0.5
