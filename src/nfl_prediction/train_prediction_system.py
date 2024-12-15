"""Train and evaluate the NFL prediction system."""

import logging
from models.prediction_system import NFLPredictionSystem
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Initialize and train prediction system
    predictor = NFLPredictionSystem()
    logger.info("Training prediction models...")
    predictor.train_all_models("data/processed_vedic/nfl_games_with_vedic.csv")

    # Save trained models
    logger.info("Saving trained models...")
    predictor.save_models("models")

    # Test with a sample game
    logger.info("\nTesting prediction system with sample game:")
    sample_game = {
        "game_date": "2023-12-10",
        "home_team": "SF",
        "away_team": "SEA",
        "home_firstdowns": 24,
        "away_firstdowns": 18,
        "home_firstdownspassing": 15,
        "away_firstdownspassing": 12,
        "home_firstdownsrushing": 9,
        "away_firstdownsrushing": 6,
        "home_passing_yards": 280,
        "away_passing_yards": 220,
        "home_passing_yards_rolling": 265.5,
        "away_passing_yards_rolling": 205.8,
        "home_rushing_yards": 145,
        "away_rushing_yards": 95,
        "home_rushing_yards_rolling": 138.2,
        "away_rushing_yards_rolling": 89.5,
        "home_penalty_yards": 45,
        "away_penalty_yards": 65,
        "home_penalty_yards_rolling": 42.5,
        "away_penalty_yards_rolling": 58.8,
        "home_division_rank": 1,
        "away_division_rank": 2,
        "home_conference_rank": 2,
        "away_conference_rank": 6,
        "home_playoff_seed": 2,
        "away_playoff_seed": 6,
        "home_point_diff_rank": 1,
        "away_point_diff_rank": 8,
        "home_win_percentage": 0.846,
        "away_win_percentage": 0.615,
        "abs_point_differential": 12.5,
    }

    logger.info(f"Home Team: {sample_game['home_team']}")
    logger.info(f"Away Team: {sample_game['away_team']}")

    result = predictor.predict_game(sample_game)
    logger.info("\nPrediction Results:")
    logger.info(f"Winner: {'Home Team' if result['prediction'] == 1 else 'Away Team'}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"Model Agreement: {result['agreement']:.2%}")


if __name__ == "__main__":
    main()
