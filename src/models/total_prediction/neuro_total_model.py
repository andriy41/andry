import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import logging
from ..base_total_model import NFLTotalPredictionModel
from sklearn.preprocessing import StandardScaler


class NeuroTotalModel(NFLTotalPredictionModel):
    """Neural network model for NFL totals prediction"""

    def __init__(self):
        super().__init__()
        # Enhanced feature list
        self.basic_features = [
            # Offensive stats
            "home_points_per_game",
            "away_points_per_game",
            "home_yards_per_game",
            "away_yards_per_game",
            "home_pass_yards",
            "away_pass_yards",
            "home_rush_yards",
            "away_rush_yards",
            "home_turnovers",
            "away_turnovers",
            # Defensive stats
            "home_points_allowed",
            "away_points_allowed",
            "home_yards_allowed",
            "away_yards_allowed",
            "home_pass_defense",
            "away_pass_defense",
            "home_rush_defense",
            "away_rush_defense",
            "home_takeaways",
            "away_takeaways",
            # Situational stats
            "home_third_down",
            "away_third_down",
            "home_red_zone",
            "away_red_zone",
            "home_sacks",
            "away_sacks",
            # Recent form
            "home_last3_points",
            "away_last3_points",
            "home_last3_allowed",
            "away_last3_allowed",
            # Weather and conditions
            "temperature",
            "wind_speed",
            "is_dome",
            "is_grass",
        ]

        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.total_scaler = StandardScaler()
        self.spread_scaler = StandardScaler()

        # Initialize models with enhanced architecture
        self.total_model = self._build_model(model_type="total")
        self.spread_model = self._build_model(model_type="spread")
        self.win_model = self._build_model(model_type="win")

    def _build_model(self, model_type: str, output_activation="linear"):
        """Build neural network model"""
        if model_type == "total":
            model = Sequential(
                [
                    Input(shape=(len(self.basic_features),)),
                    Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
                    BatchNormalization(),
                    Dropout(0.2),
                    Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                    BatchNormalization(),
                    Dense(1, activation=output_activation),
                ]
            )
        elif model_type == "spread":
            model = Sequential(
                [
                    Input(shape=(len(self.basic_features),)),
                    Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
                    BatchNormalization(),
                    Dropout(0.2),
                    Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                    BatchNormalization(),
                    Dense(1, activation=output_activation),
                ]
            )
        elif model_type == "win":
            model = Sequential(
                [
                    Input(shape=(len(self.basic_features),)),
                    Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
                    BatchNormalization(),
                    Dropout(0.2),
                    Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
                    BatchNormalization(),
                    Dense(1, activation="sigmoid"),
                ]
            )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mse" if output_activation == "linear" else "binary_crossentropy",
            metrics=["mae"] if output_activation == "linear" else ["accuracy"],
        )

        return model

    def train(self, training_data) -> None:
        """Train neural network models"""
        try:
            # Get DataFrame and features from training data
            df = training_data.get("df")
            if df is None or not isinstance(df, pd.DataFrame):
                raise ValueError(
                    "Training data must contain a DataFrame in the 'df' key"
                )

            features = training_data.get("features", {})

            # Extract and scale features with validation
            feature_array = []
            for feature in self.basic_features:
                feature_values = features.get(feature, np.zeros(len(df)))
                feature_array.append(feature_values)

            X = np.column_stack(feature_array)
            X = self.feature_scaler.fit_transform(X)

            # Scale targets with validation
            total_points = features.get("total_points", np.zeros(len(df)))
            spread = features.get("spread", np.zeros(len(df)))
            home_win = df["home_team_won"].values

            if (
                np.any(np.isnan(total_points))
                or np.any(np.isnan(spread))
                or np.any(np.isnan(home_win))
            ):
                raise ValueError("Target values contain NaN")

            total_points = self.total_scaler.fit_transform(total_points.reshape(-1, 1))
            spread = self.spread_scaler.fit_transform(spread.reshape(-1, 1))

            # Create callbacks
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
            )

            # Train total points model with validation
            total_history = self.total_model.fit(
                X,
                total_points,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1,
            )

            # Train spread model with validation
            spread_history = self.spread_model.fit(
                X,
                spread,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1,
            )

            # Train win probability model with validation
            win_history = self.win_model.fit(
                X,
                home_win,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1,
            )

            # Log training results
            logging.info(
                f"Total points model - Final val_loss: {total_history.history['val_loss'][-1]:.4f}"
            )
            logging.info(
                f"Spread model - Final val_loss: {spread_history.history['val_loss'][-1]:.4f}"
            )
            logging.info(
                f"Win model - Final val_loss: {win_history.history['val_loss'][-1]:.4f}"
            )

            self.is_trained = True
            logging.info(f"Successfully trained {self.__class__.__name__}")

        except Exception as e:
            logging.error(f"Error training {self.__class__.__name__}: {str(e)}")
            raise

    def evaluate(self, test_data):
        """Evaluate the model on test data"""
        try:
            # Get DataFrame and features from test data
            df = test_data.get("df")
            if df is None or not isinstance(df, pd.DataFrame):
                raise ValueError("Test data must contain a DataFrame in the 'df' key")

            features = test_data.get("features", {})

            # Extract features
            feature_array = []
            for feature in self.basic_features:
                feature_values = features.get(feature, np.zeros(len(df)))
                feature_array.append(feature_values)

            X = np.column_stack(feature_array)
            X = self.feature_scaler.transform(X)

            # Get actual values
            total_points = features.get("total_points", np.zeros(len(df)))
            spread = features.get("spread", np.zeros(len(df)))
            home_win = df["home_team_won"].values

            # Make predictions
            total_pred = self.total_model.predict(X)
            spread_pred = self.spread_model.predict(X)
            win_pred = self.win_model.predict(X)

            # Calculate metrics
            total_mae = np.mean(np.abs(total_points - total_pred))
            spread_mae = np.mean(np.abs(spread - spread_pred))
            accuracy = np.mean((win_pred > 0.5) == home_win)

            return {
                "total_mae": float(total_mae),
                "spread_mae": float(spread_mae),
                "accuracy": float(accuracy),
                "avg_confidence": float(np.mean(np.abs(win_pred - 0.5)) + 0.5),
            }

        except Exception as e:
            logging.error(f"Error evaluating {self.__class__.__name__}: {str(e)}")
            raise

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict game outcomes using neural networks"""
        try:
            # Validate input data
            for feature in self.basic_features:
                if feature not in game_data:
                    logging.warning(
                        f"Missing feature: {feature}. Using default value 0."
                    )
                    game_data[feature] = 0

            # Extract and scale features
            features = np.array(
                [[game_data.get(feature, 0) for feature in self.basic_features]]
            )
            features_scaled = self.feature_scaler.transform(features)

            # Make predictions with error handling
            try:
                total_points_scaled = self.total_model.predict(
                    features_scaled, verbose=0
                )
                spread_scaled = self.spread_model.predict(features_scaled, verbose=0)
                win_prob = self.win_model.predict(features_scaled, verbose=0)[0][0]

                total_points_pred = float(
                    self.total_scaler.inverse_transform(total_points_scaled)[0][0]
                )
                spread_pred = float(
                    self.spread_scaler.inverse_transform(spread_scaled)[0][0]
                )
            except Exception as e:
                logging.error(f"Error in model prediction: {str(e)}")
                raise

            # Validate predictions
            if not (0 <= win_prob <= 1):
                logging.warning(
                    f"Invalid win probability: {win_prob}. Clamping to [0,1]"
                )
                win_prob = np.clip(win_prob, 0, 1)

            # Calculate confidence based on model uncertainty
            total_conf = 1.0 / (
                1.0
                + np.abs(
                    total_points_pred - game_data.get("vegas_total", total_points_pred)
                )
            )
            spread_conf = 1.0 / (1.0 + np.abs(spread_pred))
            win_conf = (
                np.abs(win_prob - 0.5) * 2
            )  # Higher confidence when further from 0.5

            confidence = np.mean([total_conf, spread_conf, win_conf])

            # Generate detailed explanation
            explanation = (
                f"Neural network analysis:\n"
                f"- Total Points: Predicts {total_points_pred:.1f} points "
                f"({'OVER' if total_points_pred > game_data.get('vegas_total', total_points_pred) else 'UNDER'} "
                f"Vegas total of {game_data.get('vegas_total', 'N/A')})\n"
                f"- Spread: Predicts {abs(spread_pred):.1f} point "
                f"{'advantage' if spread_pred > 0 else 'deficit'} for home team\n"
                f"- Win Probability: {win_prob * 100:.1f}% chance of home team victory\n"
                f"- Confidence: {confidence * 100:.1f}% based on model certainty"
            )

            return {
                "total_points": total_points_pred,
                "total_prediction": "OVER"
                if total_points_pred > game_data.get("vegas_total", total_points_pred)
                else "UNDER",
                "spread": spread_pred,
                "spread_prediction": "HOME" if spread_pred > 0 else "AWAY",
                "home_win_probability": win_prob * 100,
                "confidence": confidence * 100,
                "explanation": explanation,
            }

        except Exception as e:
            logging.error(f"Error in prediction pipeline: {str(e)}")
            raise
