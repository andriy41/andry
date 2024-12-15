"""
Evaluate NFL prediction accuracy on historical games
"""
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import pandas as pd
from datetime import datetime
import logging
from src.models.nfl_model_integrator import NFLModelIntegrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_predictions():
    # Load historical games
    logger.info("Loading historical games...")
    df = pd.read_csv("data/processed_vedic/nfl_games_with_vedic.csv")

    # Initialize predictor
    predictor = NFLModelIntegrator()

    # Track predictions
    results = []

    # Categories for confidence levels
    category_results = {
        "Strong": {"correct": 0, "total": 0},
        "Good": {"correct": 0, "total": 0},
        "Moderate": {"correct": 0, "total": 0},
        "Weak": {"correct": 0, "total": 0},
    }

    logger.info("Making predictions...")
    for idx, game in df.iterrows():
        try:
            # Convert game time to datetime, removing timezone info
            game_time_str = game["game_date"].split("+")[0].strip()
            game_time = datetime.strptime(game_time_str, "%Y-%m-%d %H:%M:%S")

            # Get prediction
            prediction = predictor.predict_game(
                home_team=game["home_team"],
                away_team=game["away_team"],
                game_date=game_time,
                game_info={
                    "weather": game.get("weather", None),
                    "is_divisional_game": game.get("is_divisional", False),
                    "rest_days": {
                        "home": game.get("home_rest_days", 7),
                        "away": game.get("away_rest_days", 7),
                    },
                },
            )

            if prediction:
                # Get actual result
                actual_home_win = game["home_score"] > game["away_score"]

                # Get predicted winner and confidence
                pred_home_win = prediction["win_probability"] > 0.5
                confidence = prediction["confidence"]

                # Track result
                correct = pred_home_win == actual_home_win
                category_results[confidence]["total"] += 1
                if correct:
                    category_results[confidence]["correct"] += 1

                # Store detailed result
                results.append(
                    {
                        "game_date": game["game_date"],
                        "home_team": game["home_team"],
                        "away_team": game["away_team"],
                        "actual_home_win": actual_home_win,
                        "predicted_home_win": pred_home_win,
                        "confidence": prediction["confidence_level"],
                        "confidence_category": confidence,
                        "correct": correct,
                        "home_score": game["home_score"],
                        "away_score": game["away_score"],
                    }
                )

                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1} games...")

        except Exception as e:
            logger.error(f"Error processing game {idx}: {str(e)}")
            continue

    # Calculate accuracy by category
    logger.info("\nPrediction Results by Confidence Level:")
    overall_correct = 0
    overall_total = 0

    for category in category_results:
        correct = category_results[category]["correct"]
        total = category_results[category]["total"]
        accuracy = (correct / total * 100) if total > 0 else 0

        overall_correct += correct
        overall_total += total

        logger.info(f"{category} Predictions:")
        logger.info(f"  Total Games: {total}")
        logger.info(f"  Correct: {correct}")
        logger.info(f"  Accuracy: {accuracy:.1f}%\n")

    overall_accuracy = (
        (overall_correct / overall_total * 100) if overall_total > 0 else 0
    )
    logger.info(f"Overall Accuracy: {overall_accuracy:.1f}%")

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv("prediction_results.csv", index=False)
    logger.info("\nDetailed results saved to prediction_results.csv")


if __name__ == "__main__":
    evaluate_predictions()
