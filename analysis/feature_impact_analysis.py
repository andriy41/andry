"""Analyze the impact of Vedic astrology features on model performance."""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from data_processing.data_processor import NFLDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImpactAnalyzer:
    def __init__(self):
        self.data_processor = NFLDataProcessor()
        self.scaler = StandardScaler()

    def prepare_datasets(self, df):
        """Prepare two datasets: one with and one without Vedic features."""
        # Identify Vedic features
        vedic_columns = [
            col
            for col in df.columns
            if any(
                term in col
                for term in [
                    "aggression_score",
                    "expansion_score",
                    "discipline_score",
                    "leadership_score",
                    "strategy_score",
                ]
            )
        ]

        # Dataset without Vedic features
        df_without_vedic = df.drop(columns=vedic_columns)

        return df_without_vedic, df

    def train_and_evaluate(self, X, y, is_vedic=False):
        """Train XGBoost model and evaluate performance."""
        tscv = TimeSeriesSplit(n_splits=5)
        metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        feature_importance_list = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train model
            model = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            )

            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_val_scaled)

            # Calculate metrics
            metrics["accuracy"].append(accuracy_score(y_val, y_pred))
            metrics["precision"].append(
                precision_score(y_val, y_pred, average="weighted")
            )
            metrics["recall"].append(recall_score(y_val, y_pred, average="weighted"))
            metrics["f1"].append(f1_score(y_val, y_pred, average="weighted"))

            # Feature importance
            importance = pd.DataFrame(
                {"feature": X_train.columns, "importance": model.feature_importances_}
            )
            feature_importance_list.append(importance)

        # Average feature importance across folds
        feature_importance = (
            pd.concat(feature_importance_list).groupby("feature").mean()
        )
        feature_importance = feature_importance.sort_values(
            "importance", ascending=False
        )

        return {
            "metrics": {k: np.mean(v) for k, v in metrics.items()},
            "feature_importance": feature_importance,
        }

    def plot_results(self, results_without_vedic, results_with_vedic):
        """Plot comparison of results."""
        # Metrics comparison
        metrics_df = pd.DataFrame(
            {
                "Without Vedic": results_without_vedic["metrics"],
                "With Vedic": results_with_vedic["metrics"],
            }
        ).T

        plt.figure(figsize=(12, 6))
        metrics_df.plot(kind="bar")
        plt.title("Model Performance Comparison")
        plt.xlabel("Model Type")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("analysis/results/metrics_comparison.png")
        plt.close()

        # Feature importance for model with Vedic features
        plt.figure(figsize=(12, 8))
        top_features = results_with_vedic["feature_importance"].head(20)
        sns.barplot(x=top_features["importance"], y=top_features.index)
        plt.title("Top 20 Most Important Features (With Vedic)")
        plt.xlabel("Feature Importance")
        plt.tight_layout()
        plt.savefig("analysis/results/feature_importance.png")
        plt.close()

        # Save numerical results to CSV
        metrics_df.to_csv("analysis/results/performance_metrics.csv")
        results_with_vedic["feature_importance"].to_csv(
            "analysis/results/feature_importance.csv"
        )

    def analyze_impact(self, data_path):
        """Main method to analyze feature impact."""
        logger.info("Loading and processing data...")
        df = pd.read_csv(data_path)

        # Drop any non-numeric columns except team names and dates
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        keep_cols = numeric_cols.tolist() + [
            "home_team",
            "away_team",
            "winner",
            "game_date",
        ]
        df = df[keep_cols]

        logger.info("Preparing datasets...")
        # Prepare datasets
        df_without_vedic, df_with_vedic = self.prepare_datasets(df)

        # Target variable (1 for home team win, 0 for loss)
        y = (df["winner"] == df["home_team"]).astype(int)

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Number of features without Vedic: {df_without_vedic.shape[1]}")
        logger.info(f"Number of features with Vedic: {df_with_vedic.shape[1]}")

        # Remove non-numeric columns for training
        df_without_vedic = df_without_vedic.select_dtypes(include=[np.number])
        df_with_vedic = df_with_vedic.select_dtypes(include=[np.number])

        logger.info("Training and evaluating model without Vedic features...")
        results_without_vedic = self.train_and_evaluate(
            df_without_vedic, y, is_vedic=False
        )

        logger.info("Training and evaluating model with Vedic features...")
        results_with_vedic = self.train_and_evaluate(df_with_vedic, y, is_vedic=True)

        logger.info("Plotting results...")
        self.plot_results(results_without_vedic, results_with_vedic)

        # Print numerical results
        print("\nResults without Vedic features:")
        for metric, value in results_without_vedic["metrics"].items():
            print(f"{metric}: {value:.4f}")

        print("\nResults with Vedic features:")
        for metric, value in results_with_vedic["metrics"].items():
            print(f"{metric}: {value:.4f}")

        # Check if any Vedic features are in top 20 important features
        top_features = results_with_vedic["feature_importance"].head(20)
        vedic_in_top = [
            feat
            for feat in top_features.index
            if any(
                term in feat
                for term in [
                    "aggression_score",
                    "expansion_score",
                    "discipline_score",
                    "leadership_score",
                    "strategy_score",
                ]
            )
        ]

        if vedic_in_top:
            print("\nVedic features in top 20 most important features:")
            for feat in vedic_in_top:
                importance = results_with_vedic["feature_importance"].loc[
                    feat, "importance"
                ]
                print(f"{feat}: {importance:.4f}")
        else:
            print("\nNo Vedic features found in top 20 most important features")
