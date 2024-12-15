"""
Train the NFL prediction ensemble model
"""
import logging
from datetime import datetime
from data_collection.data_fetcher import NFLDataFetcher
from models.ensemble import NFLEnsemble
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(start_year: int = 2018, save_path: str = "saved_models/ensemble"):
    """Train the ensemble model using historical NFL data

    Args:
        start_year: First season to include in training data
        save_path: Where to save the trained model
    """
    current_year = datetime.now().year
    years = list(range(start_year, current_year))

    logger.info(f"Fetching NFL data for years {years}")

    # Initialize components
    data_fetcher = NFLDataFetcher()
    ensemble = NFLEnsemble(use_lstm=True)

    try:
        # Fetch and prepare training data
        df = data_fetcher.fetch_training_data(years)
        features, labels = data_fetcher.prepare_training_features(df)

        logger.info(f"Training with {len(features)} games")

        # Train the ensemble
        metrics = ensemble.train(features, labels)

        # Create save directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the trained model
        ensemble.save_models(save_path)

        logger.info("Training metrics:")
        for model_name, model_metrics in metrics.items():
            if isinstance(model_metrics, dict):
                logger.info(f"\n{model_name}:")
                for metric_name, value in model_metrics.items():
                    logger.info(f"  {metric_name}: {value}")
            else:
                logger.info(f"{model_name}: {model_metrics}")

        return metrics

    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise


def main():
    """Main training function with error handling"""
    try:
        metrics = train_model()
        logger.info("Training completed successfully!")
        return metrics
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None


if __name__ == "__main__":
    main()
