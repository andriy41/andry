"""
Train all NFL prediction models at once
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import time
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.vedic_basic.vedic_model import VedicModel
from src.models.advanced_system.advanced_model import AdvancedModel
from src.models.combined_ml.combined_model import CombinedModel
from src.models.sports_only.sports_model import SportsModel
from src.models.total_prediction.neuro_total_model import NeuroTotalModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("model_training.log")],
)
logger = logging.getLogger(__name__)


def load_data():
    """Load and prepare the historical games data"""
    data_dir = project_root / "data" / "historical"
    data = pd.read_csv(data_dir / "nfl_games_training.csv")
    data["gameday"] = pd.to_datetime(data["gameday"])

    # Create home_team_won column
    data["home_team_won"] = (data["result"] > 0).astype(int)

    # Sort by date and split into train/test
    data = data.sort_values("gameday")
    train_data = data.iloc[:-500].copy()
    test_data = data.iloc[-500:].copy()

    # Convert DataFrames to the format expected by models
    def prepare_data(df):
        """Convert DataFrame to the format expected by models"""
        # Create dictionary with both games and labels
        return {
            "df": df,  # Original DataFrame
            "games": df.to_dict("records"),  # List of game dictionaries
            "labels": df["home_team_won"].values,  # Labels array
            "features": {  # Add basic features that all models need
                "home_points_per_game": df["home_points_per_game"].values
                if "home_points_per_game" in df.columns
                else np.zeros(len(df)),
                "away_points_per_game": df["away_points_per_game"].values
                if "away_points_per_game" in df.columns
                else np.zeros(len(df)),
                "home_yards_per_game": df["home_yards_per_game"].values
                if "home_yards_per_game" in df.columns
                else np.zeros(len(df)),
                "away_yards_per_game": df["away_yards_per_game"].values
                if "away_yards_per_game" in df.columns
                else np.zeros(len(df)),
                "spread": df["spread"].values
                if "spread" in df.columns
                else np.zeros(len(df)),
                "total_points": df["total_points"].values
                if "total_points" in df.columns
                else np.zeros(len(df)),
            },
        }

    train_dict = prepare_data(train_data)
    test_dict = prepare_data(test_data)

    logger.info(
        f"Loaded {len(test_dict['games'])} games for testing and {len(train_dict['games'])} for training"
    )
    logger.info(
        f"Date range for training data: {train_data['gameday'].min()} to {train_data['gameday'].max()}"
    )

    return train_dict, test_dict


def train_model(name, model_class, train_data, test_data):
    """Train and evaluate a single model"""
    start_time = time.time()
    logger.info(f"\nTraining {name} model...")

    try:
        # Initialize model
        model = model_class()

        # Train model
        model.train(train_data)

        # Evaluate on test data
        metrics = model.evaluate(test_data)

        # Log results
        duration = time.time() - start_time
        logger.info(f"{name} model training completed in {duration:.2f} seconds")
        logger.info(f"{name} model metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

        return model, metrics

    except Exception as e:
        logger.error(f"Error training {name} model: {str(e)}")
        return None, None


def main():
    """Train all models"""
    logger.info("Starting model training pipeline...")

    # Load data
    train_data, test_data = load_data()

    # Define models to train
    models = {
        "Sports": SportsModel,  # Start with simpler model
        "Vedic": VedicModel,
        "Advanced": AdvancedModel,
        "Combined": CombinedModel,
        "Neuro": NeuroTotalModel,
    }

    # Train all models
    trained_models = {}
    all_metrics = {}

    for name, model_class in models.items():
        model, metrics = train_model(name, model_class, train_data, test_data)
        if model is not None:
            trained_models[name] = model
            all_metrics[name] = metrics

    # Print comparative results
    logger.info("\nComparative Model Performance:")
    logger.info("-" * 60)
    metrics_names = set()
    for metrics in all_metrics.values():
        if metrics:
            metrics_names.update(metrics.keys())

    for metric in sorted(metrics_names):
        logger.info(f"\n{metric}:")
        for model_name in models.keys():
            if model_name in all_metrics and all_metrics[model_name]:
                value = all_metrics[model_name].get(metric, "N/A")
                if isinstance(value, (int, float)):
                    logger.info(f"  {model_name}: {value:.4f}")
                else:
                    logger.info(f"  {model_name}: {value}")

    logger.info("\nTraining pipeline completed!")


if __name__ == "__main__":
    main()
