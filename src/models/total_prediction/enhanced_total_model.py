"""Enhanced Total Model incorporating both statistical and astrological features."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import logging


class EnhancedTotalModel:
    """Enhanced model that combines statistical and astrological features."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            # Statistical features
            "home_off_efficiency",
            "away_off_efficiency",
            "yards_gained",
            "first_down",
            "score_diff",
            "away_team_away_turnovers_rolling_3",
            "home_team_sos",
            "away_team_away_penalty_yards_rolling_3",
            "home_turnovers",
            "under_odds",
            # Astrological features
            "vedic_mars_strength",
            "vedic_jupiter_strength",
            "vedic_saturn_strength",
            "vedic_home_team_yoga",
            "vedic_away_team_yoga",
            "vedic_home_nakshatra_score",
            "vedic_away_nakshatra_score",
            "planetary_alignment",
            "moon_phase",
            "vedic_home_score",
            "vedic_away_score",
        ]

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features including both statistical and astrological data."""
        X = X.copy()

        # Handle missing values for numeric columns
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].mean())

        # Normalize astrological features to 0-1 range
        astro_features = [
            col
            for col in X.columns
            if "vedic_" in col or col in ["planetary_alignment", "moon_phase"]
        ]
        for col in astro_features:
            if col in X.columns:
                X[col] = X[col].clip(0, 1)

        return X

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model on both statistical and astrological features."""
        X = self._preprocess_features(X)

        # Prepare LightGBM parameters
        params = {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        # Create LightGBM dataset
        train_data = lgb.Dataset(X[self.feature_columns], label=y)

        # Train model
        self.model = lgb.train(params, train_data, num_boost_round=100)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using both statistical and astrological features."""
        X = self._preprocess_features(X)
        return self.model.predict(X[self.feature_columns])

    def predict_with_confidence(self, X: pd.DataFrame) -> tuple:
        """Make predictions and return confidence scores based on feature strength."""
        X = self._preprocess_features(X)

        predictions = self.model.predict(X[self.feature_columns])

        # Calculate confidence based on both statistical and astrological factors
        confidences = []
        for _, row in X.iterrows():
            # Statistical confidence (40% weight)
            stat_confidence = min(
                1.0,
                (
                    abs(row["home_off_efficiency"] - row["away_off_efficiency"]) / 2
                    + abs(row["score_diff"]) / 10
                ),
            )

            # Astrological confidence (60% weight)
            astro_confidence = (
                row["planetary_alignment"] * 0.3
                + (1 - abs(0.5 - row["moon_phase"])) * 0.2
                + (row["vedic_home_score"] + row["vedic_away_score"]) / 2 * 0.5
            )

            # Combined confidence
            confidence = 0.4 * stat_confidence + 0.6 * astro_confidence
            confidences.append(min(1.0, confidence))

        return predictions, np.array(confidences)
