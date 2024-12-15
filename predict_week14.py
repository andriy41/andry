import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.models.vedic_basic.vedic_model import VedicModel
from src.models.total_prediction.enhanced_total_model import EnhancedTotalModel
from src.utils.combined_predictor import CombinedPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_games():
    """Load Week 14 games"""
    games = [
        {
            "game_datetime": "2023-12-07 20:15:00",
            "home_team": "New England Patriots",
            "away_team": "Pittsburgh Steelers",
            "stadium_latitude": 42.0909,
            "stadium_longitude": -71.2643,
        },
        {
            "game_datetime": "2023-12-10 13:00:00",
            "home_team": "Chicago Bears",
            "away_team": "Detroit Lions",
            "stadium_latitude": 41.8623,
            "stadium_longitude": -87.6167,
        },
        {
            "game_datetime": "2023-12-10 13:00:00",
            "home_team": "Cincinnati Bengals",
            "away_team": "Indianapolis Colts",
            "stadium_latitude": 39.0955,
            "stadium_longitude": -84.5161,
        },
        {
            "game_datetime": "2023-12-10 13:00:00",
            "home_team": "Cleveland Browns",
            "away_team": "Jacksonville Jaguars",
            "stadium_latitude": 41.5061,
            "stadium_longitude": -81.6995,
        },
        {
            "game_datetime": "2023-12-10 13:00:00",
            "home_team": "New Orleans Saints",
            "away_team": "Carolina Panthers",
            "stadium_latitude": 29.9511,
            "stadium_longitude": -90.0814,
        },
        {
            "game_datetime": "2023-12-10 13:00:00",
            "home_team": "New York Jets",
            "away_team": "Houston Texans",
            "stadium_latitude": 40.8135,
            "stadium_longitude": -74.0745,
        },
        {
            "game_datetime": "2023-12-10 13:00:00",
            "home_team": "Tampa Bay Buccaneers",
            "away_team": "Atlanta Falcons",
            "stadium_latitude": 27.9759,
            "stadium_longitude": -82.5033,
        },
        {
            "game_datetime": "2023-12-10 16:05:00",
            "home_team": "Las Vegas Raiders",
            "away_team": "Minnesota Vikings",
            "stadium_latitude": 36.0909,
            "stadium_longitude": -115.1833,
        },
        {
            "game_datetime": "2023-12-10 16:05:00",
            "home_team": "Los Angeles Rams",
            "away_team": "Baltimore Ravens",
            "stadium_latitude": 34.0142,
            "stadium_longitude": -118.2878,
        },
        {
            "game_datetime": "2023-12-10 16:25:00",
            "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills",
            "stadium_latitude": 39.0489,
            "stadium_longitude": -94.4839,
        },
        {
            "game_datetime": "2023-12-10 16:25:00",
            "home_team": "San Francisco 49ers",
            "away_team": "Seattle Seahawks",
            "stadium_latitude": 37.4032,
            "stadium_longitude": -121.9697,
        },
        {
            "game_datetime": "2023-12-10 16:25:00",
            "home_team": "Denver Broncos",
            "away_team": "Los Angeles Chargers",
            "stadium_latitude": 39.7439,
            "stadium_longitude": -105.0201,
        },
        {
            "game_datetime": "2023-12-10 20:20:00",
            "home_team": "Dallas Cowboys",
            "away_team": "Philadelphia Eagles",
            "stadium_latitude": 32.7473,
            "stadium_longitude": -97.0945,
        },
        {
            "game_datetime": "2023-12-11 20:15:00",
            "home_team": "Miami Dolphins",
            "away_team": "Tennessee Titans",
            "stadium_latitude": 25.9580,
            "stadium_longitude": -80.2389,
        },
    ]
    return pd.DataFrame(games)


def format_prediction(home_team, away_team, prediction, confidence):
    """Format prediction output"""
    winner = home_team if prediction == 1 else away_team
    return f"{home_team} vs {away_team}: {winner} wins ({confidence:.1%} confident)"


def main():
    # Load games
    games_df = load_games()

    # Initialize models
    vedic_model = VedicModel()
    enhanced_model = EnhancedTotalModel()
    combined_predictor = CombinedPredictor()

    # Load trained models
    model_dir = os.path.join(project_root, "models")
    vedic_model.load_model(os.path.join(model_dir, "vedic_basic/trained_model.pkl"))
    enhanced_model.load_model(os.path.join(model_dir, "enhanced_total_model.joblib"))

    logger.info("\n=== Week 14 NFL Game Predictions ===\n")
    logger.info("Comparing predictions from multiple models:\n")

    for _, game in games_df.iterrows():
        game_dict = game.to_dict()

        logger.info(f"\n{game['home_team']} vs {game['away_team']}")
        logger.info("-" * 50)

        # Get predictions from each model
        try:
            # Vedic Model prediction
            ved_pred, ved_conf = vedic_model.predict_with_confidence(game_dict, 0.70)
            if ved_pred is not None:
                logger.info(
                    f"Vedic Model: {format_prediction(game['home_team'], game['away_team'], ved_pred, ved_conf)}"
                )

            # Enhanced Model prediction
            enh_pred = enhanced_model.predict_with_confidence(game_dict)
            if enh_pred and enh_pred["confidence"] >= 0.70:
                logger.info(
                    f"Enhanced Model: {enh_pred['prediction']} ({enh_pred['confidence']:.1%} confident)"
                )

            # Combined Predictor
            comb_pred = combined_predictor.predict_game(game_dict)
            if comb_pred and comb_pred.get("confidence", 0) >= 0.70:
                logger.info(
                    f"Combined Model: {comb_pred['prediction']} ({comb_pred['confidence']:.1%} confident)"
                )

        except Exception as e:
            logger.error(f"Error predicting game: {str(e)}")
            continue

        logger.info("-" * 50)


if __name__ == "__main__":
    main()
