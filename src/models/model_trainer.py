import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import joblib
from typing import Tuple, List, Dict
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "rf": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "gb": GradientBoostingClassifier(
                n_estimators=100, random_state=random_state
            ),
            "lr": LogisticRegression(random_state=random_state, max_iter=1000),
        }
        self.trained_models = {}
        self.feature_importances = {}
        self.scaler = StandardScaler()

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        # Define target
        y = (df["home_score"] > df["away_score"]).astype(int)

        # Select features
        feature_columns = [
            # Efficiency metrics
            "pass_efficiency_diff",
            "completion_pct_diff",
            "rush_efficiency_diff",
            "success_rate_diff",
            "turnover_rate_diff",
            "off_efficiency_diff",
            # Team performance metrics
            "home_win_streak",
            "away_win_streak",
            "home_points_streak",
            "away_points_streak",
            "home_yards_streak",
            "away_yards_streak",
            # Game context
            "home_rest_days",
            "away_rest_days",
            "home_travel_distance",
            "away_travel_distance",
            # Historical matchup features
            "historical_matchup_wins",
            "points_diff_history",
            # New momentum and efficiency features
            "momentum_diff",
            "home_momentum",
            "away_momentum",
            "home_off_efficiency",
            "away_off_efficiency",
            # Season phase performance
            "early_season_win_pct_diff",
            "mid_season_win_pct_diff",
            "late_season_win_pct_diff",
            # Strength of schedule
            "strength_of_schedule_diff",
        ]

        X = df[feature_columns].copy()

        # Handle missing values
        X = X.fillna(0)

        return X, y

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train all models in the ensemble"""
        logger.info("Training models...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_scaled, y)
            self.trained_models[name] = model

            # Store feature importances for tree-based models
            if hasattr(model, "feature_importances_"):
                self.feature_importances[name] = pd.Series(
                    model.feature_importances_, index=X.columns
                ).sort_values(ascending=False)

        return self.trained_models

    def evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate all trained models"""
        logger.info("Evaluating models...")

        results = {}
        X_scaled = self.scaler.transform(X)

        for name, model in self.trained_models.items():
            y_pred = model.predict(X_scaled)

            results[name] = {
                "accuracy": accuracy_score(y, y_pred),
                "classification_report": classification_report(y, y_pred),
                "confusion_matrix": confusion_matrix(y, y_pred),
            }

            logger.info(f"{name} model accuracy: {results[name]['accuracy']:.4f}")

        return results

    def save_models(self, output_dir: str):
        """Save trained models and scaler"""
        os.makedirs(output_dir, exist_ok=True)

        # Save models
        for name, model in self.trained_models.items():
            model_path = os.path.join(output_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")

        # Save scaler
        scaler_path = os.path.join(output_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

        # Save feature importances
        for name, importances in self.feature_importances.items():
            importances.to_csv(
                os.path.join(output_dir, f"{name}_feature_importances.csv")
            )
            logger.info(f"Saved {name} feature importances")


def main():
    # Load processed data
    logger.info("Loading data...")
    df = pd.read_csv("../data/processed/nfl_processed_data.csv")

    # Initialize trainer
    trainer = NFLModelTrainer()

    # Prepare features
    X, y = trainer.prepare_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Train models
    trainer.train_models(X_train, y_train)

    # Evaluate models
    results = trainer.evaluate_models(X_test, y_test)

    # Save models
    trainer.save_models("trained_models")

    logger.info("Training complete!")
    return results


if __name__ == "__main__":
    main()
