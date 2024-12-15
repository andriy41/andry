"""
NFL Model Trainer
Trains multiple prediction models for NFL games using various data sources
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLModelTrainer:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )
        os.makedirs(self.models_dir, exist_ok=True)

        self.models = {"total_points": None, "spread": None, "moneyline": None}

        # Define feature columns
        self.feature_columns = [
            "home_total_yards",
            "away_total_yards",
            "home_turnovers",
            "away_turnovers",
            "home_penalty_yards",
            "away_penalty_yards",
            "home_third_down_efficiency",
            "away_third_down_efficiency",
            "home_team_home_score_rolling_3",
            "home_team_home_score_rolling_5",
            "away_team_away_score_rolling_3",
            "away_team_away_score_rolling_5",
            "home_team_momentum",
            "away_team_momentum",
            "home_team_sos",
            "away_team_sos",
            "home_off_efficiency",
            "away_off_efficiency",
            "is_division_game",
            "is_dome",
            "is_primetime",
            "temp",
            "wind",
        ]

    def load_data(self):
        """Load historical game data and features"""
        historical_data_path = os.path.join(
            self.data_dir, "historical", "nfl_games_training.csv"
        )
        if os.path.exists(historical_data_path):
            return pd.read_csv(historical_data_path)
        return None

    def prepare_features(self, data):
        """Prepare feature matrix for training"""
        # Select features
        X = data[self.feature_columns].copy()

        # Handle missing values
        X = X.fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled

    def prepare_targets(self, data):
        """Prepare target variables for training"""
        # Total points (over/under)
        y_total = (data["total_points"] > data["total_line"]).astype(int)

        # Spread (home team covers)
        y_spread = (
            (data["home_score"] - data["away_score"]) > data["spread_line"]
        ).astype(int)

        # Moneyline (home team wins)
        y_moneyline = (data["home_score"] > data["away_score"]).astype(int)

        return y_total, y_spread, y_moneyline

    def train_models(self):
        """Train all prediction models"""
        data = self.load_data()
        if data is None:
            logger.error("No training data found")
            return

        logger.info("\n=== Training Information ===")
        logger.info(f"Total number of games: {len(data)}")
        logger.info(f"Date range: {data['gameday'].min()} to {data['gameday'].max()}")
        logger.info(f"\nFeature columns ({len(self.feature_columns)}):")
        for col in self.feature_columns:
            logger.info(f"- {col}")

        logger.info("\nPreparing features and targets...")
        X = self.prepare_features(data)
        y_total, y_spread, y_moneyline = self.prepare_targets(data)

        # Split data
        X_train, X_test, y_total_train, y_total_test = train_test_split(
            X, y_total, test_size=0.2, random_state=42
        )
        _, _, y_spread_train, y_spread_test = train_test_split(
            X, y_spread, test_size=0.2, random_state=42
        )
        _, _, y_moneyline_train, y_moneyline_test = train_test_split(
            X, y_moneyline, test_size=0.2, random_state=42
        )

        logger.info(f"\nTraining set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        # Train models
        logger.info("\nTraining models...")

        # Total points model
        logger.info("\n=== Total Points Model ===")
        logger.info("Class distribution:")
        logger.info(f"Over: {(y_total == 1).sum()} ({(y_total == 1).mean():.1%})")
        logger.info(f"Under: {(y_total == 0).sum()} ({(y_total == 0).mean():.1%})")

        self.models["total_points"] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3
        )
        self.models["total_points"].fit(X_train, y_total_train)
        y_total_pred = self.models["total_points"].predict(X_test)
        logger.info("\nTotal points model performance:")
        logger.info(f"Accuracy: {accuracy_score(y_total_test, y_total_pred):.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_total_test, y_total_pred))

        # Spread model
        logger.info("\n=== Spread Model ===")
        logger.info("Class distribution:")
        logger.info(f"Cover: {(y_spread == 1).sum()} ({(y_spread == 1).mean():.1%})")
        logger.info(
            f"Not Cover: {(y_spread == 0).sum()} ({(y_spread == 0).mean():.1%})"
        )

        self.models["spread"] = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=5
        )
        self.models["spread"].fit(X_train, y_spread_train)
        y_spread_pred = self.models["spread"].predict(X_test)
        logger.info("\nSpread model performance:")
        logger.info(f"Accuracy: {accuracy_score(y_spread_test, y_spread_pred):.3f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_spread_test, y_spread_pred))

        # Moneyline model
        logger.info("\n=== Moneyline Model ===")
        logger.info("Class distribution:")
        logger.info(
            f"Home Win: {(y_moneyline == 1).sum()} ({(y_moneyline == 1).mean():.1%})"
        )
        logger.info(
            f"Away Win: {(y_moneyline == 0).sum()} ({(y_moneyline == 0).mean():.1%})"
        )

        self.models["moneyline"] = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3
        )
        self.models["moneyline"].fit(X_train, y_moneyline_train)
        y_moneyline_pred = self.models["moneyline"].predict(X_test)
        logger.info("\nMoneyline model performance:")
        logger.info(
            f"Accuracy: {accuracy_score(y_moneyline_test, y_moneyline_pred):.3f}"
        )
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_moneyline_test, y_moneyline_pred))

        # Feature importance
        logger.info("\n=== Feature Importance ===")
        for model_name, model in self.models.items():
            logger.info(f"\n{model_name.title()} Model - Top 10 Important Features:")
            if isinstance(model, RandomForestClassifier):
                importances = model.feature_importances_
            else:
                importances = model.feature_importances_
            feature_imp = pd.DataFrame(
                {"feature": self.feature_columns, "importance": importances}
            )
            feature_imp = feature_imp.sort_values("importance", ascending=False).head(
                10
            )
            for _, row in feature_imp.iterrows():
                logger.info(f"{row['feature']}: {row['importance']:.3f}")

    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(self.models_dir, f"nfl_{name}_model.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Saved {name} model to {model_path}")


if __name__ == "__main__":
    trainer = NFLModelTrainer()
    trainer.train_models()
    trainer.save_models()
