import logging
import os
import json
import pandas as pd
from datetime import datetime
from predict_real_games import predict_real_games
from predict_synthetic_games import predict_synthetic_games

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
        logging.StreamHandler(),
    ],
)


def ensure_directories():
    """Ensure required directories exist"""
    dirs = ["predictions", "logs"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logging.info(f"Created directory: {d}")


def analyze_results(real_results, synthetic_results):
    """Analyze and compare results from both prediction sets"""
    try:
        # Convert to DataFrames
        real_df = pd.DataFrame(real_results)
        synthetic_df = pd.DataFrame(synthetic_results)

        # Calculate metrics
        metrics = {
            "real_games": {
                "count": len(real_df),
                "mae": abs(real_df["predicted_total"] - real_df["actual_total"]).mean(),
                "rmse": (
                    (real_df["predicted_total"] - real_df["actual_total"]) ** 2
                ).mean()
                ** 0.5,
                "avg_confidence": real_df["confidence"].mean(),
            },
            "synthetic_games": {
                "count": len(synthetic_df),
                "mae": abs(
                    synthetic_df["predicted_total"] - synthetic_df["actual_total"]
                ).mean(),
                "rmse": (
                    (synthetic_df["predicted_total"] - synthetic_df["actual_total"])
                    ** 2
                ).mean()
                ** 0.5,
                "avg_confidence": synthetic_df["confidence"].mean(),
            },
        }

        # Log results
        logging.info("\n=== Test Results ===")
        logging.info("\nReal Games:")
        logging.info(f"Number of games: {metrics['real_games']['count']}")
        logging.info(f"Mean Absolute Error: {metrics['real_games']['mae']:.2f}")
        logging.info(f"Root Mean Square Error: {metrics['real_games']['rmse']:.2f}")
        logging.info(
            f"Average Confidence: {metrics['real_games']['avg_confidence']:.2%}"
        )

        logging.info("\nSynthetic Games:")
        logging.info(f"Number of games: {metrics['synthetic_games']['count']}")
        logging.info(f"Mean Absolute Error: {metrics['synthetic_games']['mae']:.2f}")
        logging.info(
            f"Root Mean Square Error: {metrics['synthetic_games']['rmse']:.2f}"
        )
        logging.info(
            f"Average Confidence: {metrics['synthetic_games']['avg_confidence']:.2%}"
        )

        # Save metrics
        output_file = (
            f"predictions/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"\nSaved metrics to {output_file}")

        return metrics

    except Exception as e:
        logging.error(f"Error analyzing results: {str(e)}")
        raise


def run_tests():
    """Run predictions on both real and synthetic games"""
    try:
        ensure_directories()

        logging.info("Starting test run...")

        # Run predictions
        logging.info("\nPredicting real games...")
        real_results = predict_real_games()

        logging.info("\nPredicting synthetic games...")
        synthetic_results = predict_synthetic_games()

        # Analyze results
        metrics = analyze_results(real_results, synthetic_results)

        logging.info("\nTest run completed successfully!")

        return metrics

    except Exception as e:
        logging.error(f"Error in test run: {str(e)}")
        raise


if __name__ == "__main__":
    run_tests()
