"""Utility functions for evaluating NFL prediction models."""

import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prediction_type: str,
    confidence_scores: np.ndarray = None,
) -> Dict[str, float]:
    """
    Evaluate predictions based on the type of prediction.

    Args:
        y_true: True values
        y_pred: Predicted values
        prediction_type: One of ['total', 'spread', 'win']
        confidence_scores: Optional array of confidence scores

    Returns:
        Dictionary containing relevant metrics
    """
    metrics = {}

    if prediction_type in ["total", "spread"]:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics["r2"] = 1 - np.sum((y_true - y_pred) ** 2) / np.sum(
            (y_true - np.mean(y_true)) ** 2
        )

        if confidence_scores is not None:
            # Evaluate high confidence predictions (>=85%)
            high_conf_mask = confidence_scores >= 0.85
            if np.any(high_conf_mask):
                metrics["high_conf_mae"] = mean_absolute_error(
                    y_true[high_conf_mask], y_pred[high_conf_mask]
                )
                metrics["high_conf_count"] = np.sum(high_conf_mask)
                metrics["high_conf_accuracy"] = np.mean(
                    np.abs(y_true[high_conf_mask] - y_pred[high_conf_mask])
                    <= 3  # Within 3 points
                )

    elif prediction_type == "win":
        metrics["accuracy"] = accuracy_score(y_true, y_pred > 0.5)

        if confidence_scores is not None:
            # Evaluate high confidence predictions (>=85%)
            high_conf_mask = confidence_scores >= 0.85
            if np.any(high_conf_mask):
                metrics["high_conf_accuracy"] = accuracy_score(
                    y_true[high_conf_mask], y_pred[high_conf_mask] > 0.5
                )
                metrics["high_conf_count"] = np.sum(high_conf_mask)

        try:
            # Only calculate AUC-ROC if we have both classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) > 1:
                metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
            else:
                metrics["auc_roc"] = 0.5  # Default for single-class case
        except Exception as e:
            logger.warning(f"Could not calculate AUC-ROC: {str(e)}")
            metrics["auc_roc"] = 0.5

    return metrics


def cross_validate_model(
    model: Any, X: pd.DataFrame, y: Dict[str, np.ndarray], n_splits: int = 5
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation on a model.

    Args:
        model: Model instance with fit and predict methods
        X: Feature DataFrame
        y: Dictionary containing arrays for 'total', 'spread', and 'win'
        n_splits: Number of folds for cross-validation

    Returns:
        Dictionary containing lists of metrics for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {
        "total_mae": [],
        "total_rmse": [],
        "total_r2": [],
        "total_high_conf_mae": [],
        "total_high_conf_accuracy": [],
        "total_high_conf_count": [],
        "spread_mae": [],
        "spread_rmse": [],
        "spread_r2": [],
        "spread_high_conf_mae": [],
        "spread_high_conf_accuracy": [],
        "spread_high_conf_count": [],
        "win_accuracy": [],
        "win_auc": [],
        "win_high_conf_accuracy": [],
        "win_high_conf_count": [],
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = {k: v[train_idx] for k, v in y.items()}
        y_val = {k: v[val_idx] for k, v in y.items()}

        # Create training DataFrame with features and targets
        train_df = X_train.copy()
        train_df["total_points"] = y_train["total"]
        train_df["spread"] = y_train["spread"]
        train_df["home_win"] = y_train["win"]

        # Train model
        model.train(train_df)

        # Make predictions
        val_predictions = []
        confidence_scores = []
        for i in range(len(X_val)):
            pred = model.predict(X_val.iloc[[i]])
            val_predictions.append(pred)
            confidence_scores.append(pred["confidence"])

        # Extract predictions and confidence scores
        confidence_scores = np.array(confidence_scores)
        y_pred = {
            "total": np.array([p["total_points"] for p in val_predictions]),
            "spread": np.array([p["spread"] for p in val_predictions]),
            "win": np.array([p["win_probability"] for p in val_predictions]),
        }

        # Calculate metrics for each prediction type
        for pred_type in ["total", "spread", "win"]:
            fold_metrics = evaluate_predictions(
                y_val[pred_type], y_pred[pred_type], pred_type, confidence_scores
            )

            # Store basic metrics
            if pred_type in ["total", "spread"]:
                metrics[f"{pred_type}_mae"].append(fold_metrics["mae"])
                metrics[f"{pred_type}_rmse"].append(fold_metrics["rmse"])
                metrics[f"{pred_type}_r2"].append(fold_metrics["r2"])
            else:  # win predictions
                metrics["win_accuracy"].append(fold_metrics["accuracy"])
                metrics["win_auc"].append(fold_metrics["auc_roc"])

            # Store high confidence metrics if available
            if "high_conf_mae" in fold_metrics:
                metrics[f"{pred_type}_high_conf_mae"].append(
                    fold_metrics["high_conf_mae"]
                )
            if "high_conf_accuracy" in fold_metrics:
                metrics[f"{pred_type}_high_conf_accuracy"].append(
                    fold_metrics["high_conf_accuracy"]
                )
            if "high_conf_count" in fold_metrics:
                metrics[f"{pred_type}_high_conf_count"].append(
                    fold_metrics["high_conf_count"]
                )

        logger.info(f"Completed fold {fold + 1}/{n_splits}")

        # Log high confidence predictions for this fold
        high_conf_mask = confidence_scores >= 0.85
        if np.any(high_conf_mask):
            logger.info(f"\nHigh confidence predictions (≥85%) for fold {fold + 1}:")
            high_conf_indices = np.where(high_conf_mask)[0]
            for idx in high_conf_indices:
                logger.info(
                    f"Prediction {idx}: "
                    f"Total (pred/true): {y_pred['total'][idx]:.1f}/{y_val['total'][idx]} | "
                    f"Spread (pred/true): {y_pred['spread'][idx]:.1f}/{y_val['spread'][idx]} | "
                    f"Win prob: {y_pred['win'][idx]:.1%} (true: {y_val['win'][idx]}) | "
                    f"Confidence: {confidence_scores[idx]:.1%}"
                )

    return metrics


def validate_input_data(
    data: pd.DataFrame, required_features: List[str]
) -> Tuple[bool, str]:
    """
    Validate input data for prediction models.

    Args:
        data: Input DataFrame
        required_features: List of required feature names

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for required features
    missing_features = [f for f in required_features if f not in data.columns]
    if missing_features:
        return False, f"Missing required features: {missing_features}"

    # Check for null values
    null_columns = (
        data[required_features].columns[data[required_features].isnull().any()].tolist()
    )
    if null_columns:
        return False, f"Null values found in columns: {null_columns}"

    # Validate numeric columns
    numeric_features = (
        data[required_features].select_dtypes(include=[np.number]).columns
    )
    for col in numeric_features:
        if not np.isfinite(data[col]).all():
            return False, f"Invalid numeric values found in column: {col}"

    return True, ""
