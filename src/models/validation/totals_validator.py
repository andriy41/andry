from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from datetime import datetime


class TotalsValidator:
    """Validates NFL totals predictions using various cross-validation strategies"""

    def __init__(self):
        self.validation_metrics = {
            "mae": [],  # Mean Absolute Error
            "rmse": [],  # Root Mean Squared Error
            "over_accuracy": [],  # Accuracy of Over predictions
            "under_accuracy": [],  # Accuracy of Under predictions
            "push_rate": [],  # Rate of pushes (exact matches)
            "profit": [],  # Theoretical profit using Kelly criterion
            "roi": [],  # Return on Investment
        }

    def time_series_cv(
        self, model, data: List[Dict[str, Any]], n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform time-series cross validation
        This is more appropriate for sports betting as we want to predict future games
        """
        # Sort data by date
        data = sorted(
            data, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d %H:%M")
        )

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for train_idx, test_idx in tscv.split(df):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]

            # Train model
            model.train({"games": train_data.to_dict("records")})

            # Get predictions
            predictions = []
            actuals = []
            lines = []

            for game in test_data.to_dict("records"):
                pred = model.predict(game)
                predictions.append(pred["predicted_total"])
                actuals.append(game["home_score"] + game["away_score"])
                lines.append(game["total_line"])

            # Calculate metrics
            self._calculate_metrics(predictions, actuals, lines)

        return self._get_average_metrics()

    def rolling_window_cv(
        self, model, data: List[Dict[str, Any]], window_size: int = 32
    ) -> Dict[str, float]:
        """
        Perform rolling window validation
        This simulates real-world betting where we use recent games to predict upcoming ones
        """
        # Sort data by date
        data = sorted(
            data, key=lambda x: datetime.strptime(x["date"], "%Y-%m-%d %H:%M")
        )

        predictions = []
        actuals = []
        lines = []

        # For each prediction point, use the previous window_size games as training
        for i in range(window_size, len(data)):
            train_data = data[i - window_size : i]
            test_game = data[i]

            # Train model
            model.train({"games": train_data})

            # Get prediction
            pred = model.predict(test_game)
            predictions.append(pred["predicted_total"])
            actuals.append(test_game["home_score"] + test_game["away_score"])
            lines.append(test_game["total_line"])

            # Calculate metrics for this window
            self._calculate_metrics(predictions[-1:], actuals[-1:], lines[-1:])

        return self._get_average_metrics()

    def k_fold_cv(
        self, model, data: List[Dict[str, Any]], n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform k-fold cross validation
        Less appropriate for time series data but useful for general model validation
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        df = pd.DataFrame(data)

        for train_idx, test_idx in kf.split(df):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]

            # Train model
            model.train({"games": train_data.to_dict("records")})

            # Get predictions
            predictions = []
            actuals = []
            lines = []

            for game in test_data.to_dict("records"):
                pred = model.predict(game)
                predictions.append(pred["predicted_total"])
                actuals.append(game["home_score"] + game["away_score"])
                lines.append(game["total_line"])

            # Calculate metrics
            self._calculate_metrics(predictions, actuals, lines)

        return self._get_average_metrics()

    def _calculate_metrics(
        self, predictions: List[float], actuals: List[float], lines: List[float]
    ) -> None:
        """Calculate all validation metrics"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        lines = np.array(lines)

        # Basic error metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))

        # Over/Under accuracy
        over_predictions = predictions > lines
        actual_overs = actuals > lines
        over_accuracy = np.mean(over_predictions == actual_overs)
        under_accuracy = np.mean(~over_predictions == ~actual_overs)

        # Push rate
        pushes = np.sum(np.abs(actuals - lines) < 0.5) / len(actuals)

        # Profit calculation using Kelly Criterion
        edge = self._calculate_edge(predictions, lines)
        profit, roi = self._calculate_profit(edge, actuals, lines)

        # Store metrics
        self.validation_metrics["mae"].append(mae)
        self.validation_metrics["rmse"].append(rmse)
        self.validation_metrics["over_accuracy"].append(over_accuracy)
        self.validation_metrics["under_accuracy"].append(under_accuracy)
        self.validation_metrics["push_rate"].append(pushes)
        self.validation_metrics["profit"].append(profit)
        self.validation_metrics["roi"].append(roi)

    def _calculate_edge(self, predictions: np.ndarray, lines: np.ndarray) -> np.ndarray:
        """Calculate predicted edge for each game"""
        return np.abs(predictions - lines)

    def _calculate_profit(
        self,
        edge: np.ndarray,
        actuals: np.ndarray,
        lines: np.ndarray,
        unit_size: float = 100,
    ) -> Tuple[float, float]:
        """Calculate theoretical profit using Kelly criterion"""
        total_bet = 0
        total_profit = 0

        for e, actual, line in zip(edge, actuals, lines):
            # Kelly bet size
            kelly_fraction = (e - 0.5) / 10  # Conservative Kelly
            bet_size = unit_size * max(0, min(kelly_fraction, 0.25))  # Cap at 25%

            total_bet += bet_size
            if actual > line:
                total_profit += bet_size * 0.909  # -110 odds
            else:
                total_profit -= bet_size

        roi = (total_profit / total_bet) if total_bet > 0 else 0
        return total_profit, roi

    def _get_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics across all validation runs"""
        return {
            metric: np.mean(values)
            for metric, values in self.validation_metrics.items()
        }

    def print_validation_report(self, metrics: Dict[str, float]) -> None:
        """Print a formatted validation report"""
        print("\nTotals Prediction Validation Report")
        print("=" * 40)
        print(f"Mean Absolute Error: {metrics['mae']:.2f} points")
        print(f"Root Mean Squared Error: {metrics['rmse']:.2f} points")
        print(f"Over Prediction Accuracy: {metrics['over_accuracy']*100:.1f}%")
        print(f"Under Prediction Accuracy: {metrics['under_accuracy']*100:.1f}%")
        print(f"Push Rate: {metrics['push_rate']*100:.1f}%")
        print(f"Theoretical Profit: ${metrics['profit']:.2f}")
        print(f"ROI: {metrics['roi']*100:.1f}%")
        print("=" * 40)


def validate_totals_model(
    model, data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Validate a totals prediction model using multiple validation strategies
    Returns metrics for each validation method
    """
    validator = TotalsValidator()

    results = {
        "time_series": validator.time_series_cv(model, data),
        "rolling_window": validator.rolling_window_cv(model, data),
        "k_fold": validator.k_fold_cv(model, data),
    }

    # Print reports
    print("\nValidation Results by Method:")
    for method, metrics in results.items():
        print(f"\n{method.replace('_', ' ').title()} Validation:")
        validator.print_validation_report(metrics)

    return results
