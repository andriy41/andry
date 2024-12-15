"""NFL Machine Learning Model for game predictions using advanced ML techniques."""

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLMLModel:
    """Advanced ML model for NFL game predictions."""

    def __init__(self):
        """Initialize the NFL ML model with multiple algorithms."""
        self.models = {
            "rf": RandomForestClassifier(n_estimators=200, max_depth=20),
            "gb": GradientBoostingClassifier(n_estimators=200),
            "nn": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000),
            "xgb": xgb.XGBClassifier(n_estimators=200),
            "lgb": lgb.LGBMClassifier(n_estimators=200),
            "lstm": None,  # Will be initialized during training
        }
        self.scaler = StandardScaler()
        self.feature_columns = [
            "qb_rating",
            "rush_yards_per_game",
            "pass_yards_per_game",
            "points_per_game",
            "points_allowed_per_game",
            "turnover_diff",
            "third_down_pct",
            "red_zone_pct",
            "sacks_per_game",
            "time_of_possession",
            "penalties_per_game",
            "defensive_dvoa",
            "offensive_dvoa",
            "special_teams_dvoa",
        ]

    def prepare_data(self, data):
        """Prepare and preprocess data for training."""
        try:
            # Convert categorical variables
            data = pd.get_dummies(data, columns=["home_team", "away_team", "weather"])

            # Create time-based features
            data["day_of_week"] = pd.to_datetime(data["game_date"]).dt.dayofweek
            data["month"] = pd.to_datetime(data["game_date"]).dt.month

            # Calculate rolling averages
            for team in data["team"].unique():
                team_data = data[data["team"] == team]
                for stat in self.feature_columns:
                    data.loc[data["team"] == team, f"{stat}_rolling_avg"] = (
                        team_data[stat].rolling(window=5, min_periods=1).mean()
                    )

            return data

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None

    def create_lstm_model(self, input_shape):
        """Create and compile LSTM model."""
        model = Sequential(
            [
                LSTM(128, input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models in the ensemble."""
        try:
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)

            # Train each model
            for name, model in self.models.items():
                if name == "lstm":
                    # Reshape data for LSTM
                    X_train_reshaped = X_train_scaled.reshape(
                        (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
                    )
                    self.models[name] = self.create_lstm_model(
                        (1, X_train_scaled.shape[1])
                    )
                    self.models[name].fit(
                        X_train_reshaped,
                        y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0,
                    )
                else:
                    model.fit(X_train_scaled, y_train)

            logger.info("All models trained successfully")

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")

    def predict(self, X):
        """Make predictions using all models and combine results."""
        try:
            X_scaled = self.scaler.transform(X)
            predictions = {}

            # Get predictions from each model
            for name, model in self.models.items():
                if name == "lstm":
                    X_reshaped = X_scaled.reshape(
                        (X_scaled.shape[0], 1, X_scaled.shape[1])
                    )
                    predictions[name] = model.predict(X_reshaped)
                else:
                    predictions[name] = model.predict_proba(X_scaled)[:, 1]

            # Weighted average of predictions
            weights = {
                "rf": 0.2,
                "gb": 0.2,
                "nn": 0.15,
                "xgb": 0.2,
                "lgb": 0.15,
                "lstm": 0.1,
            }

            final_prediction = sum(
                pred * weights[name] for name, pred in predictions.items()
            )

            return {
                "win_probability": final_prediction,
                "individual_predictions": predictions,
            }

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        try:
            predictions = self.predict(X_test)["win_probability"]
            y_pred = (predictions > 0.5).astype(int)

            return {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }

        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            return None

    def save_models(self, path):
        """Save all models to disk."""
        try:
            for name, model in self.models.items():
                if name == "lstm":
                    model.save(f"{path}/lstm_model")
                else:
                    joblib.dump(model, f"{path}/{name}_model.joblib")
            joblib.dump(self.scaler, f"{path}/scaler.joblib")
            logger.info("Models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def load_models(self, path):
        """Load all models from disk."""
        try:
            for name in self.models.keys():
                if name == "lstm":
                    self.models[name] = tf.keras.models.load_model(f"{path}/lstm_model")
                else:
                    self.models[name] = joblib.load(f"{path}/{name}_model.joblib")
            self.scaler = joblib.load(f"{path}/scaler.joblib")
            logger.info("Models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def analyze_feature_importance(self):
        """Analyze and return feature importance from tree-based models."""
        try:
            importance_dict = {}

            # Get feature importance from Random Forest
            rf_importance = self.models["rf"].feature_importances_
            importance_dict["random_forest"] = dict(
                zip(self.feature_columns, rf_importance)
            )

            # Get feature importance from XGBoost
            xgb_importance = self.models["xgb"].feature_importances_
            importance_dict["xgboost"] = dict(zip(self.feature_columns, xgb_importance))

            return importance_dict

        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            return None
