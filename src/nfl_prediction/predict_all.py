"""
Comprehensive NFL prediction script that outputs all betting predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from typing import Dict, Any, List, Tuple
import logging

from src.models import (
    TotalPredictionModel,
    MLTotalModel,
    AstroTotalModel,
    VedicTotalModel,
)
from src.models.vedic_astrology.nfl_vedic_calculator import NFLVedicCalculator
from src.utils.astro_utils import (
    calculate_astrological_advantage,
    calculate_yogas,
    get_planet_positions,
)


class NFLComprehensivePredictor:
    def __init__(self):
        self.total_model = TotalPredictionModel()
        self.ml_model = MLTotalModel()
        self.astro_model = AstroTotalModel()
        self.vedic_model = VedicTotalModel()
        self.vedic_calc = NFLVedicCalculator()

    def predict_game(
        self,
        game_date: str,
        game_time: str,
        home_team: str,
        away_team: str,
        spread: float = None,
        total_line: float = None,
        home_ml: float = None,
        away_ml: float = None,
    ) -> Dict[str, Any]:
        """
        Make comprehensive predictions for an NFL game.

        Args:
            game_date: Date of the game (YYYY-MM-DD)
            game_time: Time of the game (HH:MM)
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            spread: Current spread (optional)
            total_line: Current over/under line (optional)
            home_ml: Home team moneyline odds (optional)
            away_ml: Away team moneyline odds (optional)

        Returns:
            Dictionary containing all predictions
        """
        # Convert to datetime
        game_dt = datetime.strptime(f"{game_date} {game_time}", "%Y-%m-%d %H:%M")
        eastern = pytz.timezone("America/New_York")
        game_dt = eastern.localize(game_dt)

        # Get astrological factors
        astro_adv = calculate_astrological_advantage(game_dt, home_team, away_team)
        yogas = calculate_yogas(game_dt)
        planet_pos = get_planet_positions(game_dt)

        # Get Vedic predictions
        vedic_pred = self.vedic_calc.predict_game_outcome(game_dt, home_team, away_team)

        # Combine all model predictions
        total_pred = self.total_model.predict(
            {
                "game_datetime": game_dt,
                "home_team": home_team,
                "away_team": away_team,
                "astro_advantage": astro_adv,
                "yogas": yogas,
                "planet_positions": planet_pos,
            }
        )

        ml_pred = self.ml_model.predict_proba(
            {"game_datetime": game_dt, "home_team": home_team, "away_team": away_team}
        )

        # Calculate consensus predictions
        home_win_prob = (ml_pred["home_win"] + vedic_pred["home_win_prob"]) / 2
        predicted_spread = -1 * (home_win_prob - 0.5) * 14  # Convert to points
        predicted_total = total_pred["total_points"]

        # Generate betting recommendations
        recommendations = {
            "game_info": {
                "date": game_date,
                "time": game_time,
                "home_team": home_team,
                "away_team": away_team,
            },
            "predictions": {
                "home_win_probability": round(home_win_prob * 100, 1),
                "predicted_spread": round(predicted_spread, 1),
                "predicted_total": round(predicted_total, 1),
            },
            "betting_advice": {},
        }

        # Add spread recommendation if line is provided
        if spread is not None:
            edge = predicted_spread - spread
            recommendations["betting_advice"]["spread"] = {
                "current_line": spread,
                "edge": round(edge, 1),
                "recommendation": "HOME"
                if edge > 2
                else "AWAY"
                if edge < -2
                else "PASS",
            }

        # Add total recommendation if line is provided
        if total_line is not None:
            total_edge = predicted_total - total_line
            recommendations["betting_advice"]["total"] = {
                "current_line": total_line,
                "edge": round(total_edge, 1),
                "recommendation": "OVER"
                if total_edge > 2
                else "UNDER"
                if total_edge < -2
                else "PASS",
            }

        # Add moneyline recommendation if odds are provided
        if home_ml is not None and away_ml is not None:
            recommendations["betting_advice"]["moneyline"] = {
                "home_odds": home_ml,
                "away_odds": away_ml,
                "recommendation": "HOME"
                if home_win_prob > 0.55
                else "AWAY"
                if home_win_prob < 0.45
                else "PASS",
            }

        return recommendations


def print_predictions(predictions: Dict[str, Any]) -> None:
    """Print predictions in a clear format."""
    game = predictions["game_info"]
    preds = predictions["predictions"]
    advice = predictions["betting_advice"]

    print("\n=== NFL Game Prediction ===")
    print(f"\nGame: {game['away_team']} @ {game['home_team']}")
    print(f"Date: {game['date']} {game['time']}")

    print("\n--- Predictions ---")
    print(f"Home Win Probability: {preds['home_win_probability']}%")
    print(f"Predicted Spread: {game['home_team']} {preds['predicted_spread']}")
    print(f"Predicted Total: {preds['predicted_total']}")

    print("\n--- Betting Advice ---")
    if "spread" in advice:
        spread = advice["spread"]
        print(f"\nSpread ({spread['current_line']}):")
        print(f"Edge: {spread['edge']} points")
        print(f"Recommendation: {spread['recommendation']}")

    if "total" in advice:
        total = advice["total"]
        print(f"\nTotal ({total['current_line']}):")
        print(f"Edge: {total['edge']} points")
        print(f"Recommendation: {total['recommendation']}")

    if "moneyline" in advice:
        ml = advice["moneyline"]
        print(f"\nMoneyline ({ml['home_odds']}/{ml['away_odds']}):")
        print(f"Recommendation: {ml['recommendation']}")


def main():
    predictor = NFLComprehensivePredictor()

    # Example usage for an upcoming game
    predictions = predictor.predict_game(
        game_date="2024-12-07",
        game_time="13:00",
        home_team="NE",
        away_team="LAC",
        spread=-3.5,
        total_line=40.5,
        home_ml=+165,
        away_ml=-185,
    )

    print_predictions(predictions)


if __name__ == "__main__":
    main()
