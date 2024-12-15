"""Train the enhanced NFL prediction model."""
import pandas as pd
import numpy as np
from models.total_prediction.enhanced_total_model import EnhancedTotalModel
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple
from pathlib import Path


def load_enhanced_data() -> Tuple[pd.DataFrame, List[str]]:
    """Load enhanced historical NFL data."""
    data_dir = Path(os.path.dirname(__file__)) / "data"
    df = pd.read_csv(data_dir / "enhanced_historical_games.csv")

    # Define features to use
    features = [
        # Basic game info
        "is_primetime",
        "is_division_game",
        "is_dome",
        # Team performance metrics
        "home_total_yards",
        "away_total_yards",
        "home_turnovers",
        "away_turnovers",
        "home_penalty_yards",
        "away_penalty_yards",
        "home_third_down_efficiency",
        "away_third_down_efficiency",
        "home_off_efficiency",
        "away_off_efficiency",
        # Rolling averages
        "home_team_home_score_rolling_3",
        "away_team_away_score_rolling_3",
        "home_team_home_score_rolling_5",
        "away_team_away_score_rolling_5",
        "home_team_home_total_yards_rolling_3",
        "away_team_away_total_yards_rolling_3",
        # Momentum and strength of schedule
        "home_team_momentum",
        "away_team_momentum",
        "home_team_sos",
        "away_team_sos",
    ]

    # Add weather features if available
    if "temp" in df.columns and "wind" in df.columns:
        features.extend(["temp", "wind"])

    return df, features


def optimize_hyperparameters(
    model: EnhancedTotalModel, X: pd.DataFrame, y: pd.DataFrame
) -> Dict[str, Any]:
    """Optimize model hyperparameters using RandomizedSearchCV."""

    # Define parameter distributions for each model
    param_distributions = {
        "gbm__n_estimators": [100, 200, 300, 400, 500],
        "gbm__learning_rate": [0.01, 0.05, 0.1],
        "gbm__max_depth": [3, 4, 5, 6],
        "rf__n_estimators": [100, 200, 300, 400, 500],
        "rf__max_depth": [5, 10, 15, 20],
        "xgb__n_estimators": [100, 200, 300, 400, 500],
        "xgb__learning_rate": [0.01, 0.05, 0.1],
        "xgb__max_depth": [3, 4, 5, 6],
        "lgb__n_estimators": [100, 200, 300, 400, 500],
        "lgb__learning_rate": [0.01, 0.05, 0.1],
        "lgb__max_depth": [3, 4, 5, 6],
    }

    # Use TimeSeriesSplit for validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform randomized search
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=50,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=2,
    )

    random_search.fit(X, y)
    logging.info(f"Best parameters: {random_search.best_params_}")
    logging.info(f"Best score: {-random_search.best_score_:.2f} MSE")

    return random_search.best_params_


def evaluate_model(
    model: EnhancedTotalModel, X: pd.DataFrame, y: pd.DataFrame
) -> Dict[str, float]:
    """Evaluate model using time series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {"mse": [], "mae": [], "within_3": [], "within_7": []}

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        predictions, confidence = model.predict(X_test)

        # Calculate metrics
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        within_3 = np.mean(np.abs(predictions - y_test) <= 3)
        within_7 = np.mean(np.abs(predictions - y_test) <= 7)

        metrics["mse"].append(mse)
        metrics["mae"].append(mae)
        metrics["within_3"].append(within_3)
        metrics["within_7"].append(within_7)

    # Calculate average metrics
    return {
        "mse": np.mean(metrics["mse"]),
        "mae": np.mean(metrics["mae"]),
        "within_3": np.mean(metrics["within_3"]),
        "within_7": np.mean(metrics["within_7"]),
    }


def train_enhanced_model():
    """Train the enhanced prediction model with optimized parameters."""
    logging.info("Loading enhanced data...")
    data, features = load_enhanced_data()

    # Prepare features and target
    X = data[features]
    y = data["total_points"]

    # Remove rows with missing values
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Initialize model
    model = EnhancedTotalModel()

    # Optimize hyperparameters
    logging.info("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(model, X, y)

    # Update model with best parameters
    model.set_params(**best_params)

    # Evaluate model
    logging.info("Evaluating model...")
    metrics = evaluate_model(model, X, y)

    logging.info("Model Performance:")
    logging.info(f"Mean Squared Error: {metrics['mse']:.2f}")
    logging.info(f"Mean Absolute Error: {metrics['mae']:.2f}")
    logging.info(f"Within 3 points: {metrics['within_3']*100:.1f}%")
    logging.info(f"Within 7 points: {metrics['within_7']*100:.1f}%")

    # Train final model on all data
    logging.info("Training final model...")
    model.fit(X, y)

    # Save model
    model_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "total_prediction",
        "enhanced_total_model.joblib",
    )
    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_enhanced_model()
