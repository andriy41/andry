import logging
from typing import Dict, Any, List
from predict_total import TotalPredictor
from data_collectors.totals_collector import NFLTotalsCollector
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def is_high_confidence_pick(prediction: Dict[str, Any]) -> bool:
    """
    Determine if this is a high-confidence pick based on model agreement
    and confidence levels.

    Requirements:
    - All models predict the same direction (over/under)
    - Each model shows >80% confidence
    - At least one model shows >85% confidence
    """
    if not prediction or "model_predictions" not in prediction:
        return False

    predictions = prediction["model_predictions"]
    confidences = prediction["model_confidences"]
    vegas_total = prediction.get("vegas_total", 0)

    # Check if all models predict the same direction
    pred_values = list(predictions.values())
    all_over = all(p > vegas_total for p in pred_values)
    all_under = all(p < vegas_total for p in pred_values)

    if not (all_over or all_under):
        return False

    # Check confidence thresholds
    conf_values = list(confidences.values())
    all_above_80 = all(c >= 0.80 for c in conf_values)
    any_above_85 = any(c >= 0.85 for c in conf_values)

    return all_above_80 and any_above_85


def format_model_name(name: str) -> str:
    """Format model name for display"""
    name_map = {
        "advanced": "Advanced System",
        "vedic": "Vedic System",
        "ml": "Combined ML",
        "stats": "Sports-Only",
        "neuro": "Neural Network",
    }
    return name_map.get(name, name.title())


def main():
    try:
        # Initialize predictor and collector
        predictor = TotalPredictor()
        collector = NFLTotalsCollector()

        # Get Week 14 games
        logging.info("Fetching Week 14 games data...")
        games = collector.get_week_14_2024_data()

        if not games:
            logging.error("No games data found for Week 14")
            return

        print("\nNFL WEEK 14 PREDICTIONS (December 5-8, 2024)")
        print("=" * 60)

        high_confidence_picks = []

        for game in games:
            try:
                print(f"\n{game['away_team']} @ {game['home_team']}")
                print(f"Date: {game['date']}")
                print(f"Venue: {game['venue']}")
                print("-" * 40)

                # Validate required game data
                required_fields = ["home_team", "away_team", "vegas_total"]
                if not all(field in game for field in required_fields):
                    logging.error(f"Missing required fields for game: {game}")
                    continue

                # Use game data directly instead of fetching from ESPN
                prediction = predictor.predict(game)

                if not prediction or "model_predictions" not in prediction:
                    logging.error(f"Failed to get prediction for game: {game}")
                    continue

                print(f"Vegas Total: {game['vegas_total']}")
                print("\nModel Predictions:")

                # Sort models by confidence
                models = list(prediction["model_predictions"].keys())
                models.sort(
                    key=lambda x: prediction["model_confidences"][x], reverse=True
                )

                for model in models:
                    pred = prediction["model_predictions"][model]
                    conf = prediction["model_confidences"][model]
                    print(
                        f"- {format_model_name(model)}: {pred:.1f} (Confidence: {conf:.2f})"
                    )

                consensus = prediction["consensus"]
                print(f"\nConsensus Total: {consensus['total']:.1f}")
                print(f"Rating: {consensus['rating']}")

                # Calculate the difference from Vegas line
                diff = consensus["total"] - game["vegas_total"]
                lean = "OVER" if diff > 0 else "UNDER"

                is_high_conf = is_high_confidence_pick(prediction)
                if is_high_conf:
                    high_confidence_picks.append(
                        {
                            "game": f"{game['away_team']} @ {game['home_team']}",
                            "pick": f"{lean} {game['vegas_total']}",
                            "edge": abs(diff),
                            "consensus": consensus["total"],
                        }
                    )

                if abs(diff) >= 2:
                    confidence_note = " (HIGH CONFIDENCE)" if is_high_conf else ""
                    print(
                        f"\nRecommendation: {lean} {game['vegas_total']}{confidence_note}"
                    )
                    print(f"Edge: {abs(diff):.1f} points")
                else:
                    print("\nRecommendation: PASS - Not enough edge")

                print(f"Explanation: {consensus['explanation']}")

                # Additional factors
                print("\nKey Factors:")
                if "temperature" in game and game["temperature"] < 40:
                    print("- Cold weather could impact scoring")
                if "wind_speed" in game and game["wind_speed"] > 10:
                    print(
                        f"- High winds ({game['wind_speed']} mph) could affect passing/kicking"
                    )
                if "precipitation" in game and game["precipitation"] > 0:
                    print("- Precipitation in forecast")

            except Exception as e:
                logging.error(
                    f"Error processing game {game.get('home_team', 'Unknown')} vs {game.get('away_team', 'Unknown')}: {str(e)}"
                )
                continue

        # Print high confidence picks summary
        if high_confidence_picks:
            print("\nHIGH CONFIDENCE PICKS SUMMARY")
            print("=" * 60)
            for pick in high_confidence_picks:
                print(f"\n{pick['game']}")
                print(f"Pick: {pick['pick']}")
                print(f"Edge: {pick['edge']:.1f} points")
                print(f"Consensus: {pick['consensus']:.1f}")

    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    main()
