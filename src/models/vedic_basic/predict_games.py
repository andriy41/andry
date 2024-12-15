import pandas as pd
import logging
from datetime import datetime
from vedic_model import VedicModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_games(file_path):
    """Load games from CSV file"""
    try:
        df = pd.read_csv(file_path)
        games = df.to_dict("records")
        return games
    except Exception as e:
        logger.error(f"Error loading games: {str(e)}")
        return []


def main():
    # Initialize model
    model = VedicModel()

    # Load the trained model
    if not model.load_model("trained_model.pkl"):
        logger.error("Failed to load model")
        return

    # Load games to predict
    games = load_games("upcoming_games.csv")
    if not games:
        logger.error("No games loaded")
        return

    # Make predictions with confidence threshold
    confidence_threshold = 0.75  # Only predict games with 75% or higher confidence
    predictions = model.predict_batch_with_confidence(games, confidence_threshold)

    # Display high confidence predictions
    logger.info(f"\nHigh Confidence Predictions (threshold: {confidence_threshold}):")
    logger.info("-" * 80)

    for pred in predictions:
        confidence_pct = round(pred["confidence"] * 100, 2)
        winner = pred["home_team"] if pred["prediction"] == 1 else pred["away_team"]
        logger.info(f"Date: {pred['date']}")
        logger.info(f"{pred['home_team']} vs {pred['away_team']}")
        logger.info(f"Predicted Winner: {winner}")
        logger.info(f"Confidence: {confidence_pct}%")
        logger.info("-" * 80)

    logger.info(f"\nTotal high confidence predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
