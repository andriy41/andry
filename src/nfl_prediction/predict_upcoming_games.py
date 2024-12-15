"""Predict upcoming NFL games using our trained models."""

import os
import sys

# Add the project root to the Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, List

from src.models.nfl_model_integrator import NFLModelIntegrator
from src.models.vedic_basic.vedic_model import VedicModel
from src.models.advanced_system.advanced_model import AdvancedModel
from src.models.combined_ml.combined_model import CombinedModel
from src.models.sports_only.sports_model import SportsModel
from src.models.total_prediction.neuro_total_model import NeuroTotalModel
from src.utils.trend_analyzer import TrendAnalyzer
from src.utils.combined_predictor import CombinedPredictor
from src.utils.performance_tracker import PerformanceTracker  # Add this import


def get_upcoming_games() -> pd.DataFrame:
    """Get upcoming NFL games with enhanced metadata"""
    # Add Vegas lines and weather data
    upcoming_games = [
        {
            "game_date": "2024-12-07",
            "home_team": "CHI",
            "away_team": "DET",
            "is_division_game": True,
            "is_conference_game": True,
            "vegas_spread": -3.5,
            "vegas_total": 47.5,
            "weather_temp": 35,
            "weather_condition": "SNOW",
        },
        # Add remaining games with enhanced data
    ]
    return pd.DataFrame(upcoming_games)


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate the weighted average of a list of values."""
    return np.average(values, weights=weights)


def analyze_predictions(predictions_df: pd.DataFrame):
    """Enhanced prediction analysis with betting angles"""
    results = []

    for _, game in predictions_df.iterrows():
        # Calculate model consensus with weighted confidence
        model_weights = {
            "vedic": 0.2,
            "advanced": 0.3,
            "combined": 0.25,
            "sports": 0.15,
            "neuro": 0.1,
        }

        weighted_predictions = {
            "win_prob": weighted_average(
                [game[f"{m}_win_prob"] for m in model_weights.keys()],
                list(model_weights.values()),
            ),
            "spread": weighted_average(
                [game[f"{m}_spread"] for m in model_weights.keys()],
                list(model_weights.values()),
            ),
            "total": weighted_average(
                [game[f"{m}_total"] for m in model_weights.keys()],
                list(model_weights.values()),
            ),
        }

        # Add betting value analysis
        vegas_edge = abs(weighted_predictions["total"] - game["vegas_total"])
        value_rating = (
            "HIGH" if vegas_edge > 4 else "MEDIUM" if vegas_edge > 2 else "LOW"
        )

        result = {
            "matchup": f"{game['away_team']} @ {game['home_team']}",
            "prediction": generate_prediction_summary(weighted_predictions, game),
            "value_rating": value_rating,
            "consensus_strength": calculate_consensus_strength(game),
            "betting_recommendation": generate_betting_recommendation(
                weighted_predictions, game
            ),
        }
        results.append(result)

    return pd.DataFrame(results)


def calculate_consensus_strength(game: pd.Series) -> str:
    """Calculate the consensus strength of the prediction."""
    confidence_levels = [
        game[f"{model}_confidence"]
        for model in ["vedic", "advanced", "combined", "sports", "neuro"]
    ]
    average_confidence = np.mean(confidence_levels)

    if average_confidence >= 0.7:
        return "Strong"
    elif average_confidence >= 0.5:
        return "Moderate"
    else:
        return "Weak"


def generate_betting_recommendation(
    weighted_predictions: Dict[str, float], game: pd.Series
) -> str:
    """Generate a betting recommendation based on predictions and game data."""
    if weighted_predictions["win_prob"] > 0.6:
        return f"Bet on {game['home_team']} to win"
    elif weighted_predictions["spread"] > 3:
        return f"Bet on {game['home_team']} to cover the spread"
    elif weighted_predictions["total"] > game["vegas_total"]:
        return "Bet on the over"
    else:
        return "No strong recommendation"


def generate_prediction_summary(
    weighted_predictions: Dict[str, float], game: pd.Series
) -> str:
    """Generate a summary of the prediction based on weighted predictions."""
    return (
        f"Win Probability: {weighted_predictions['win_prob']:.2f}, "
        f"Spread: {weighted_predictions['spread']:.2f}, "
        f"Total: {weighted_predictions['total']:.2f}"
    )


def main():
    """Enhanced main function with performance tracking"""
    # Initialize performance tracker
    tracker = PerformanceTracker()

    # Load models with validation
    models = load_models()
    # Validate models
    for name, model in models.items():
        if not model.is_valid():
            logger.error(f"{name} model is not valid")
            raise ValueError(f"{name} model is not valid")

    # Get games and make predictions
    games_df = get_upcoming_games()
    predictions_df = predict_games(games_df, models)
    results_df = analyze_predictions(predictions_df)

    # Generate detailed reports
    # generate_prediction_report(results_df)
    # generate_confidence_analysis(results_df)
    # generate_betting_angles(results_df)

    # Track performance
    tracker.update(results_df)
    tracker.save_history()


def get_upcoming_games() -> pd.DataFrame:
    """Get the upcoming NFL games for prediction."""
    # Current week's games
    upcoming_games = [
        {
            "game_date": "2024-12-07",  # Week 14 games
            "home_team": "CHI",
            "away_team": "DET",
            "is_division_game": True,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "CIN",
            "away_team": "IND",
            "is_division_game": False,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "NO",
            "away_team": "CAR",
            "is_division_game": True,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "NYJ",
            "away_team": "HOU",
            "is_division_game": False,
            "is_conference_game": False,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "BAL",
            "away_team": "LAR",
            "is_division_game": False,
            "is_conference_game": False,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "LV",
            "away_team": "MIN",
            "is_division_game": False,
            "is_conference_game": False,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "CLE",
            "away_team": "JAX",
            "is_division_game": False,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "TB",
            "away_team": "ATL",
            "is_division_game": True,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "TEN",
            "away_team": "MIA",
            "is_division_game": False,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "KC",
            "away_team": "BUF",
            "is_division_game": False,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "DEN",
            "away_team": "LAC",
            "is_division_game": True,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "SF",
            "away_team": "SEA",
            "is_division_game": True,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "DAL",
            "away_team": "PHI",
            "is_division_game": True,
            "is_conference_game": True,
        },
        {
            "game_date": "2024-12-07",
            "home_team": "GB",
            "away_team": "NYG",
            "is_division_game": False,
            "is_conference_game": True,
        },
    ]

    return pd.DataFrame(upcoming_games)


def load_models():
    """Load and initialize all prediction models."""
    models = {
        "vedic": VedicModel(),
        "advanced": AdvancedModel(),
        "combined": CombinedModel(),
        "sports": SportsModel(),
        "neuro": NeuroTotalModel(),
    }

    # Load the latest trained models
    for name, model in models.items():
        try:
            model.load_latest()
            logger.info(f"Loaded {name} model successfully")
        except Exception as e:
            logger.error(f"Error loading {name} model: {str(e)}")

    return models


def predict_games(games_df: pd.DataFrame, models: Dict[str, Any]):
    """Make predictions for each game using all models."""
    predictions = []

    for _, game in games_df.iterrows():
        game_predictions = {
            "game_date": game["game_date"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "is_division_game": game["is_division_game"],
            "is_conference_game": game["is_conference_game"],
        }

        # Get predictions from each model
        for name, model in models.items():
            try:
                pred = model.predict(game)

                # Store predictions
                game_predictions[f"{name}_win_prob"] = pred.get("win_probability", 0.5)
                game_predictions[f"{name}_spread"] = pred.get("spread_prediction", 0)
                game_predictions[f"{name}_total"] = pred.get("total_prediction", 0)
                game_predictions[f"{name}_confidence"] = pred.get("confidence", 0)

            except Exception as e:
                logger.error(
                    f"Error getting prediction from {name} model for {game['home_team']} vs {game['away_team']}: {str(e)}"
                )
                game_predictions[f"{name}_win_prob"] = 0.5
                game_predictions[f"{name}_spread"] = 0
                game_predictions[f"{name}_total"] = 0
                game_predictions[f"{name}_confidence"] = 0

        predictions.append(game_predictions)

    return pd.DataFrame(predictions)


def analyze_predictions(predictions_df: pd.DataFrame):
    """Analyze and combine predictions from all models."""
    results = []

    for _, game in predictions_df.iterrows():
        # Calculate consensus win probability
        win_probs = [
            game[f"{model}_win_prob"] * game[f"{model}_confidence"]
            for model in ["vedic", "advanced", "combined", "sports", "neuro"]
        ]
        confidences = [
            game[f"{model}_confidence"]
            for model in ["vedic", "advanced", "combined", "sports", "neuro"]
        ]

        consensus_win_prob = np.average(win_probs, weights=confidences)

        # Calculate consensus spread
        spreads = [
            game[f"{model}_spread"] * game[f"{model}_confidence"]
            for model in ["vedic", "advanced", "combined", "sports", "neuro"]
        ]
        consensus_spread = np.average(spreads, weights=confidences)

        # Calculate consensus total
        totals = [
            game[f"{model}_total"] * game[f"{model}_confidence"]
            for model in ["vedic", "advanced", "combined", "sports", "neuro"]
        ]
        consensus_total = np.average(totals, weights=confidences)

        # Determine prediction strength
        confidence_level = np.mean(confidences)
        if confidence_level >= 0.7:
            strength = "Strong"
        elif confidence_level >= 0.5:
            strength = "Moderate"
        else:
            strength = "Weak"

        result = {
            "game_date": game["game_date"],
            "matchup": f"{game['away_team']} @ {game['home_team']}",
            "prediction": f"{game['home_team'] if consensus_win_prob > 0.5 else game['away_team']} Win",
            "win_probability": f"{max(consensus_win_prob, 1-consensus_win_prob):.1%}",
            "spread": f"{abs(consensus_spread):.1f} {game['home_team'] if consensus_spread > 0 else game['away_team']}",
            "total": f"O/U {consensus_total:.1f}",
            "confidence": strength,
            "is_division_game": game["is_division_game"],
            "is_conference_game": game["is_conference_game"],
        }

        results.append(result)

    return pd.DataFrame(results)


def main():
    """Main function to predict upcoming games."""
    # Get upcoming games
    games_df = get_upcoming_games()
    logger.info(f"Found {len(games_df)} upcoming games to predict")

    # Load models
    models = load_models()

    # Make predictions
    logger.info("Making predictions...")
    predictions_df = predict_games(games_df, models)

    # Analyze predictions
    logger.info("Analyzing predictions...")
    results_df = analyze_predictions(predictions_df)

    # Save predictions
    output_path = "data/predictions"
    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(
        f'{output_path}/predictions_{datetime.now().strftime("%Y%m%d")}.csv',
        index=False,
    )

    # Print predictions
    print("\nNFL Game Predictions:")
    print("=" * 80)
    for _, pred in results_df.iterrows():
        print(f"\nGame: {pred['matchup']}")
        print(f"Date: {pred['game_date']}")
        print(
            f"Prediction: {pred['prediction']} ({pred['win_probability']} probability)"
        )
        print(f"Spread: {pred['spread']}")
        print(f"Total Points: {pred['total']}")
        print(f"Confidence: {pred['confidence']}")
        if pred["is_division_game"]:
            print("*Division Game*")
        elif pred["is_conference_game"]:
            print("*Conference Game*")
        print("-" * 40)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
