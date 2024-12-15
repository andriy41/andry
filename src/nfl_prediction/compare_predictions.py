import logging
import pandas as pd
from datetime import datetime
import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.models.vedic_basic.vedic_model import VedicModel
from src.models.unified_predictor import UnifiedPredictor
from src.nfl_prediction.predict_total import TotalPredictor
from src.utils.combined_predictor import CombinedPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_games(file_path):
    """Load games from CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading games: {str(e)}")
        return pd.DataFrame()


def format_prediction(home_team, away_team, prediction, confidence):
    """Format prediction output"""
    winner = home_team if prediction == 1 else away_team
    return f"{home_team} vs {away_team}: {winner} wins ({confidence:.1%} confident)"


def main():
    # Load games
    games_df = load_games("upcoming_games.csv")
    if games_df.empty:
        logger.error("No games loaded")
        return

    # Initialize predictors
    vedic_model = VedicModel()
    total_predictor = TotalPredictor()
    combined_predictor = CombinedPredictor()

    # Load models
    if not vedic_model.load_model("models/vedic_basic/trained_model.pkl"):
        logger.error("Failed to load Vedic model")
        return

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

            # Total Predictor
            total_pred = total_predictor.predict_game(game_dict)
            if total_pred and total_pred.get("high_confidence"):
                logger.info(
                    f"Total Model: {total_pred['prediction']} ({total_pred['confidence']:.1%} confident)"
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
