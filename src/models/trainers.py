"""
Collection of different model trainers for NFL prediction
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform


class BaseTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}
        self.best_params = {}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance with multiple metrics"""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            log_loss,
        )

        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "auc": roc_auc_score(y, y_pred_proba[:, 1]),
            "log_loss": log_loss(y, y_pred_proba),
        }

        self.metrics.update(metrics)
        return metrics

    def save(self, path: str):
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "metrics": self.metrics,
                "best_params": self.best_params,
            },
            path,
        )

    def load(self, path: str):
        saved = joblib.load(path)
        self.model = saved["model"]
        self.scaler = saved["scaler"]
        self.metrics = saved.get("metrics", {})
        self.best_params = saved.get("best_params", {})
        self.is_trained = True


class LSTMTrainer(BaseTrainer):
    def __init__(self, sequence_length: int = 5):
        super().__init__()
        self.sequence_length = sequence_length
        self.model = None

    def create_model(self, input_shape):
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(32, return_sequences=True),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(16),
                BatchNormalization(),
                Dropout(0.3),
                Dense(8, activation="relu"),
                BatchNormalization(),
                Dense(1, activation="sigmoid"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005, clipnorm=1.0  # Gradient clipping
        )

        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    def prepare_sequences(self, X):
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i : i + self.sequence_length])
        return np.array(sequences)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
    ):
        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        X_seq = self.prepare_sequences(X_scaled)
        y_seq = y[self.sequence_length - 1 :]

        # Create validation split
        n_val = int(len(X_seq) * validation_split)
        X_train, X_val = X_seq[:-n_val], X_seq[-n_val:]
        y_train, y_val = y_seq[:-n_val], y_seq[-n_val:]

        # Create and train model
        self.model = self.create_model((self.sequence_length, X.shape[1]))

        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                min_delta=0.001,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        self.is_trained = True
        return history.history


class XGBoostTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model = None
        self.cv_results = None
        self.feature_importance = None

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train with hyperparameter tuning"""
        from sklearn.model_selection import RandomizedSearchCV

        # Define hyperparameter search space
        param_dist = {
            "n_estimators": [500, 1000, 1500],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0.1, 1.0, 5.0],
            "gamma": [0, 0.1, 0.5],
        }

        # Initialize base model
        base_model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            eval_metric=["logloss", "auc"],
            early_stopping_rounds=50,
            use_label_encoder=False,
            n_jobs=-1,
        )

        # Random search with cross-validation
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform search
        random_search.fit(X_scaled, y)

        # Save best model and parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.cv_results = random_search.cv_results_

        # Calculate feature importance
        self.feature_importance = {
            "gain": self.model.get_booster().get_score(importance_type="gain"),
            "weight": self.model.get_booster().get_score(importance_type="weight"),
        }

        self.is_trained = True
        return self


class RandomForestTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            random_state=42,
        )

    def train(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return {
            "feature_importance": dict(
                zip(
                    [f"feature_{i}" for i in range(X.shape[1])],
                    self.model.feature_importances_,
                )
            ),
            "training_accuracy": self.model.score(X_scaled, y),
        }


class SVMTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)

    def train(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return {"training_accuracy": self.model.score(X_scaled, y)}


class MLPTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train with hyperparameter tuning"""
        from sklearn.model_selection import RandomizedSearchCV

        # Define hyperparameter search space
        param_dist = {
            "hidden_layer_sizes": [
                (64, 32),
                (128, 64),
                (256, 128),
                (128, 64, 32),
                (256, 128, 64),
            ],
            "activation": ["relu", "tanh"],
            "alpha": loguniform(1e-5, 1e-2),
            "learning_rate_init": loguniform(1e-4, 1e-2),
            "batch_size": [32, 64, 128],
            "max_iter": [500, 1000],
        }

        # Initialize base model
        base_model = MLPClassifier(
            solver="adam",
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            random_state=42,
        )

        # Random search with cross-validation
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Perform search
        random_search.fit(X_scaled, y)

        # Save best model and parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.cv_results = random_search.cv_results_

        self.is_trained = True
        return self


class AdaBoostTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.model = AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=42
        )

    def train(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return {
            "feature_importance": dict(
                zip(
                    [f"feature_{i}" for i in range(X.shape[1])],
                    self.model.feature_importances_,
                )
            ),
            "training_accuracy": self.model.score(X_scaled, y),
        }
