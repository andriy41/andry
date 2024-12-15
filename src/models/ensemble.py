"""
Enhanced NFL Ensemble Model with calibration and advanced weight optimization
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from scipy.optimize import minimize
import joblib
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class NFLEnsemble:
    def __init__(self, models: Dict[str, Any] = None, weights: Dict[str, float] = None):
        self.models = models or {}
        self.weights = weights or {}
        self.calibrated_models = {}
        self.is_calibrated = False
        self.optimization_history = []

    def calibrate_models(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calibrate individual models using Platt scaling"""
        logger.info("Calibrating individual models")

        for name, model in self.models.items():
            try:
                # Skip models that are already probability calibrated
                if hasattr(model, "predict_proba") and not hasattr(
                    model, "calibrated_classifiers_"
                ):
                    calibrator = CalibratedClassifierCV(
                        base_estimator=model, cv="prefit", method="sigmoid"
                    )
                    calibrator.fit(X, y)
                    self.calibrated_models[name] = calibrator
                    logger.info(f"Calibrated {name} model")
                else:
                    self.calibrated_models[name] = model

            except Exception as e:
                logger.error(f"Error calibrating {name} model: {e}")
                self.calibrated_models[name] = model

        self.is_calibrated = True

    def _weighted_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get weighted predictions from all models"""
        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, (name, model) in enumerate(self.calibrated_models.items()):
            try:
                pred = model.predict_proba(X)[:, 1]
                predictions[:, i] = pred * self.weights.get(name, 1.0)
            except Exception as e:
                logger.error(f"Error getting predictions from {name} model: {e}")
                predictions[:, i] = 0.5  # Neutral prediction

        return predictions.mean(axis=1)

    def _objective_function(
        self, weights: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float = 0.5
    ) -> float:
        """Custom objective function for weight optimization"""
        # Normalize weights
        weights = weights / np.sum(weights)

        # Update model weights
        for i, name in enumerate(self.models.keys()):
            self.weights[name] = weights[i]

        # Get predictions
        y_pred = self._weighted_predictions(X)

        # Calculate metrics
        log_loss_score = log_loss(y, y_pred)
        auc_score = roc_auc_score(y, y_pred)
        brier_score = brier_score_loss(y, y_pred)

        # Combined objective (minimize log loss and brier score, maximize AUC)
        objective = alpha * (log_loss_score + brier_score) - (1 - alpha) * auc_score

        # Store optimization history
        self.optimization_history.append(
            {
                "weights": weights.copy(),
                "log_loss": log_loss_score,
                "auc": auc_score,
                "brier_score": brier_score,
                "objective": objective,
            }
        )

        return objective

    def optimize_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Optimize model weights using advanced optimization"""
        logger.info("Optimizing ensemble weights")

        if not self.is_calibrated:
            logger.info("Calibrating models before weight optimization")
            self.calibrate_models(X, y)

        # Initial weights
        initial_weights = np.array(
            [self.weights.get(name, 1.0) for name in self.models.keys()]
        )
        initial_weights = initial_weights / np.sum(initial_weights)

        # Constraints
        bounds = [(0.0, 1.0) for _ in range(len(self.models))]
        constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        # Optimize weights
        try:
            result = minimize(
                fun=self._objective_function,
                x0=initial_weights,
                args=(X, y),
                method="SLSQP",
                bounds=bounds,
                constraints=constraint,
                options={"maxiter": 1000},
            )

            # Update weights
            optimized_weights = result.x / np.sum(result.x)
            for i, name in enumerate(self.models.keys()):
                self.weights[name] = optimized_weights[i]

            logger.info("Optimized weights:")
            for name, weight in self.weights.items():
                logger.info(f"{name}: {weight:.3f}")

        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")

        return self.weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from ensemble"""
        if not self.is_calibrated:
            logger.warning("Models not calibrated. Using uncalibrated predictions.")

        predictions = self._weighted_predictions(X)
        return np.vstack((1 - predictions, predictions)).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions from ensemble"""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            log_loss,
            brier_score_loss,
        )

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "auc": roc_auc_score(y, y_pred_proba),
            "log_loss": log_loss(y, y_pred_proba),
            "brier_score": brier_score_loss(y, y_pred_proba),
        }

        return metrics

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of weight optimization"""
        return self.optimization_history

    def save(self, path: str) -> None:
        """Save ensemble model"""
        model_data = {
            "models": self.models,
            "calibrated_models": self.calibrated_models,
            "weights": self.weights,
            "is_calibrated": self.is_calibrated,
            "optimization_history": self.optimization_history,
        }
        joblib.dump(model_data, path)

    def load(self, path: str) -> None:
        """Load ensemble model"""
        model_data = joblib.load(path)
        self.models = model_data["models"]
        self.calibrated_models = model_data["calibrated_models"]
        self.weights = model_data["weights"]
        self.is_calibrated = model_data["is_calibrated"]
        self.optimization_history = model_data["optimization_history"]
