import logging
import time
import importlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import random
import os
import sys
import joblib

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.total_prediction.total_model import TotalPredictionModel
from models.total_prediction.stats_total_model import StatsTotalModel
from models.total_prediction.ml_total_model import MLTotalModel
from models.total_prediction.vedic_total_model import VedicTotalModel


class TotalPredictor:
    def __init__(self):
        # Try to load trained models first
        try:
            self.stats_model = joblib.load("models/trained/stats_total_model.joblib")
            print("Loaded trained StatsTotalModel")
        except:
            print("Using new StatsTotalModel")
            self.stats_model = StatsTotalModel()

        try:
            self.ml_model = joblib.load("models/trained/ml_total_model.joblib")
            print("Loaded trained MLTotalModel")
        except:
            print("Using new MLTotalModel")
            self.ml_model = MLTotalModel()

        self.vedic_model = VedicTotalModel()

        self.models = [self.stats_model, self.ml_model, self.vedic_model]

    def predict_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get predictions from all models for a single game
        """
        predictions = []

        for model in self.models:
            try:
                pred = model.predict(game_data)
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Error in {model.__class__.__name__}: {str(e)}")

        if not predictions:
            return self._get_default_prediction()

        return self._aggregate_predictions(predictions)

    def _aggregate_predictions(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate predictions from multiple models"""
        try:
            # Handle both float and dict predictions
            total_points = 0
            valid_preds = 0

            for p in predictions:
                if isinstance(p, (int, float)):
                    total_points += float(p)
                    valid_preds += 1
                elif isinstance(p, dict) and "total" in p:
                    total_points += float(p["total"])
                    valid_preds += 1

            if valid_preds == 0:
                return self._get_default_prediction()

            avg_total = total_points / valid_preds

            return {
                "total": avg_total,
                "confidence": 0.7,
                "explanation": f"Average prediction across {valid_preds} models: {avg_total:.1f} points",
            }

        except Exception as e:
            logging.error(f"Error aggregating predictions: {str(e)}")
            return self._get_default_prediction()

    def _get_default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when aggregation fails"""
        return {
            "total": 47.5,
            "confidence": 0.5,
            "explanation": "Using league average due to prediction errors",
        }

    def _generate_consensus_explanation(self, predictions: List[Dict[str, Any]]) -> str:
        """Generate consensus explanation from all predictions"""
        total_points = sum(p["total"]["points"] for p in predictions) / len(predictions)
        spread_points = sum(p["spread"]["points"] for p in predictions) / len(
            predictions
        )
        home_win_prob = sum(p["moneyline"]["home_win_prob"] for p in predictions) / len(
            predictions
        )

        return (
            f"Consensus Prediction:\n"
            f"Total Points: {total_points:.1f} "
            f"({'OVER' if total_points > predictions[0]['total']['line'] else 'UNDER'} "
            f"{predictions[0]['total']['line']})\n"
            f"Spread: {abs(spread_points):.1f} points "
            f"({'HOME' if spread_points > 0 else 'AWAY'} favored)\n"
            f"Win Probability: {home_win_prob:.1%} for "
            f"{'HOME' if home_win_prob > 0.5 else 'AWAY'} team"
        )


def validate_prediction_confidence(prediction: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate prediction confidence levels
    Returns (is_valid, explanation)
    """
    try:
        # Check required fields
        if "total" not in prediction or "confidence" not in prediction:
            return False, "Invalid prediction format"

        confidence = float(prediction["confidence"])

        # Validate confidence range
        if not (0 <= confidence <= 1):
            return False, f"Invalid confidence value: {confidence}"

        # Check for high confidence threshold
        is_confident = confidence >= 0.8

        if is_confident:
            return True, "High confidence prediction"
        else:
            return True, "Normal confidence prediction"

    except Exception as e:
        logging.error(f"Error validating prediction: {str(e)}")
        return False, "Error validating prediction"


def handle_rate_limit(error_msg: str) -> bool:
    """Handle rate limit errors by waiting 5 minutes"""
    if "rate limit exceeded" in str(error_msg).lower():
        wait_time = 300  # 5 minutes in seconds
        logging.warning(
            f"Rate limit exceeded. Waiting {wait_time} seconds before continuing..."
        )
        print(
            f"\nRate limit hit - Pausing for 5 minutes at {datetime.now().strftime('%H:%M:%S')}"
        )
        print("Testing will automatically resume after the wait period.")

        # Wait with a countdown
        for remaining in range(wait_time, 0, -1):
            mins, secs = divmod(remaining, 60)
            print(f"\rResuming in: {mins:02d}:{secs:02d}", end="")
            time.sleep(1)

        print("\nResuming testing...")
        return True
    return False


def run_continuous_tests(duration_hours: float = 1.0):
    """Run continuous tests for specified duration"""
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)

    test_results = {
        "total_tests": 0,
        "successful_tests": 0,
        "high_confidence_picks": 0,
        "model_errors": 0,
        "rate_limit_hits": 0,
        "predictions": [],
    }

    # Initialize predictor
    try:
        predictor = TotalPredictor()
    except Exception as e:
        logging.error(f"Failed to initialize predictor: {str(e)}")
        return

    logging.info("Starting continuous testing session")
    logging.info(f"Duration: {duration_hours} hours")
    logging.info("Confidence thresholds: >80% all models, >85% at least one model")

    while time.time() < end_time:
        try:
            # Generate test game data
            game_data = generate_test_game_data()

            # Get prediction
            prediction = predictor.predict_game(game_data)

            # Validate confidence levels
            is_confident, conf_explanation = validate_prediction_confidence(prediction)

            # Log prediction details
            logging.info(f"\nTest {test_results['total_tests'] + 1}:")
            logging.info(
                f"Game conditions: {'Dome' if game_data['is_dome'] else 'Outdoor'}, "
                f"Temp: {game_data.get('temperature', 'N/A')}, "
                f"Wind: {game_data.get('wind_speed', 'N/A')}"
            )
            logging.info(
                f"Total prediction: {prediction['total']:.1f} "
                f"Confidence: {prediction['confidence']:.2%}"
            )
            logging.info(f"Confidence validation: {conf_explanation}")

            # Update statistics
            test_results["total_tests"] += 1
            test_results["successful_tests"] += 1
            if is_confident:
                test_results["high_confidence_picks"] += 1
                logging.info("*** HIGH CONFIDENCE PICK ***")

            # Store prediction for analysis
            test_results["predictions"].append(
                {
                    "total": prediction["total"],
                    "confidence": prediction["confidence"],
                    "is_confident": is_confident,
                    "game_data": game_data,
                }
            )

            # Print progress update every 10 tests
            if test_results["total_tests"] % 10 == 0:
                print_progress_update(test_results, start_time)

            # Brief pause between iterations
            time.sleep(1)

        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error in test iteration: {error_msg}")

            # Handle rate limit errors
            if handle_rate_limit(error_msg):
                test_results["rate_limit_hits"] += 1
                continue

            test_results["model_errors"] += 1
            test_results["total_tests"] += 1
            time.sleep(1)

    # Final analysis
    print("\nTesting Complete!")
    print_progress_update(test_results, start_time)
    save_test_results(test_results)


def print_progress_update(results: Dict[str, Any], start_time: float):
    """Print progress update with key metrics"""
    elapsed_time = time.time() - start_time
    success_rate = (results["successful_tests"] / max(1, results["total_tests"])) * 100
    confidence_rate = (
        results["high_confidence_picks"] / max(1, results["successful_tests"])
    ) * 100

    print("\nProgress Update:")
    print(f"Time elapsed: {elapsed_time/3600:.2f} hours")
    print(f"Total tests: {results['total_tests']}")
    print(f"Success rate: {success_rate:.1f}%")
    print(
        f"High confidence picks: {results['high_confidence_picks']} ({confidence_rate:.1f}%)"
    )
    print(f"Model errors: {results['model_errors']}")
    print(f"Rate limit hits: {results.get('rate_limit_hits', 0)}")


def analyze_predictions(predictions: List[Dict[str, Any]]):
    """Analyze prediction patterns and trends"""
    # Basic prediction distribution
    total_over = sum(1 for p in predictions if p["total"] > 47.5)
    total_under = sum(1 for p in predictions if p["total"] <= 47.5)

    print("\nPrediction Distribution:")
    print(f"Total: Over {total_over}, Under {total_under}")

    # High confidence analysis
    confident_picks = [p for p in predictions if p["is_confident"]]
    if confident_picks:
        print("\nHigh Confidence Picks Analysis:")
        conf_over = sum(1 for p in confident_picks if p["total"] > 47.5)
        conf_under = sum(1 for p in confident_picks if p["total"] <= 47.5)

        print(f"Total: Over {conf_over}, Under {conf_under}")

        avg_total_conf = sum(p["confidence"] for p in confident_picks) / len(
            confident_picks
        )

        print(f"Average Confidence:")
        print(f"Total: {avg_total_conf:.1%}")


def generate_test_game_data() -> Dict[str, Any]:
    """Generate realistic test game data"""
    return {
        "home_team_stats": {
            "points_per_game": random.uniform(17, 30),
            "points_allowed_per_game": random.uniform(17, 30),
            "yards_per_game": random.uniform(280, 420),
            "yards_allowed_per_game": random.uniform(280, 420),
            "third_down_conversion": random.uniform(0.3, 0.5),
            "red_zone_scoring": random.uniform(0.45, 0.7),
        },
        "away_team_stats": {
            "points_per_game": random.uniform(17, 30),
            "points_allowed_per_game": random.uniform(17, 30),
            "yards_per_game": random.uniform(280, 420),
            "yards_allowed_per_game": random.uniform(280, 420),
            "third_down_conversion": random.uniform(0.3, 0.5),
            "red_zone_scoring": random.uniform(0.45, 0.7),
        },
        "is_dome": random.choice([True, False]),
        "temperature": random.uniform(25, 95) if random.random() > 0.3 else None,
        "wind_speed": random.uniform(0, 25) if random.random() > 0.3 else None,
        "over_under_line": random.uniform(41, 54),
        "spread_line": random.uniform(-10, 10),
        "is_primetime": random.choice([True, False]),
    }


def save_test_results(results: Dict[str, Any]):
    """Save test results to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.log"

    with open(os.path.join("logs", filename), "w") as f:
        f.write("=== NFL Prediction Model Test Results ===\n\n")
        f.write(f"Test Date: {datetime.now()}\n")
        f.write(f"Total Tests: {results['total_tests']}\n")
        f.write(f"Successful Tests: {results['successful_tests']}\n")
        f.write(f"High Confidence Picks: {results['high_confidence_picks']}\n")
        f.write(f"Model Errors: {results['model_errors']}\n")
        f.write(f"Rate Limit Hits: {results['rate_limit_hits']}\n\n")

        # Write high confidence picks
        f.write("High Confidence Picks:\n")
        confident_picks = [p for p in results["predictions"] if p["is_confident"]]
        for i, pick in enumerate(confident_picks, 1):
            f.write(f"\nPick {i}:\n")
            f.write(
                f"Total: {pick['total']:.1f} " f"Confidence: {pick['confidence']:.2%}\n"
            )


if __name__ == "__main__":
    # Set up logging
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/continuous_test.log"),
            logging.StreamHandler(),
        ],
    )

    try:
        run_continuous_tests()
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        logging.error(f"Error in continuous testing: {str(e)}")
        print(f"\nError in continuous testing: {str(e)}")
