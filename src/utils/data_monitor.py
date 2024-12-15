"""Data quality monitoring and reporting utilities."""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Stores data quality metrics and issues."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_records: int = 0
    processed_records: int = 0
    skipped_records: int = 0
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(f"⚠️ {message}")

    def add_error(self, message: str):
        """Add an error message and increment error count."""
        self.errors.append(message)
        self.error_count += 1
        logger.error(f"❌ {message}")

    def add_metric(self, name: str, value: float):
        """Add a quality metric."""
        self.metrics[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "skipped_records": self.skipped_records,
            "error_count": self.error_count,
            "warnings": self.warnings,
            "errors": self.errors,
            "metrics": self.metrics,
        }

    def save(self, filepath: str):
        """Save report to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        """Print a formatted summary of the data quality report."""
        logger.info("\n=== Data Quality Report ===")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(
            f"Records: {self.processed_records}/{self.total_records} "
            f"({self.skipped_records} skipped)"
        )

        if self.metrics:
            logger.info("\nQuality Metrics:")
            for name, value in self.metrics.items():
                logger.info(f"  • {name}: {value:.2f}")

        if self.warnings:
            logger.info("\nWarnings:")
            for i, warning in enumerate(self.warnings[:5], 1):
                logger.info(f"  {i}. {warning}")
            if len(self.warnings) > 5:
                logger.info(f"  ... and {len(self.warnings) - 5} more warnings")

        if self.errors:
            logger.info("\nErrors:")
            for i, error in enumerate(self.errors[:5], 1):
                logger.info(f"  {i}. {error}")
            if len(self.errors) > 5:
                logger.info(f"  ... and {len(self.errors) - 5} more errors")

        logger.info("\nSummary:")
        logger.info(
            f"  • Success Rate: {(self.processed_records/self.total_records*100):.1f}%"
        )
        logger.info(f"  • Error Rate: {(self.error_count/self.total_records*100):.1f}%")
        logger.info("=" * 25)


class DataMonitor:
    """Monitors data quality during processing."""

    def __init__(self):
        self.report = DataQualityReport()
        self._current_phase = None

    def start_phase(self, phase_name: str):
        """Start a new monitoring phase."""
        self._current_phase = phase_name
        logger.info(f"\n▶️ Starting phase: {phase_name}")

    def end_phase(self):
        """End current phase and log summary."""
        if self._current_phase:
            logger.info(f"✅ Completed phase: {self._current_phase}")
        self._current_phase = None

    def check_nulls(self, df: pd.DataFrame, critical_columns: List[str]) -> bool:
        """Check for null values in critical columns."""
        phase = (
            f"{self._current_phase}: Null Check"
            if self._current_phase
            else "Null Check"
        )

        has_nulls = False
        for col in critical_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                has_nulls = True
                self.report.add_warning(
                    f"{phase} - Found {null_count} null values in column '{col}'"
                )
        return has_nulls

    def check_ranges(self, df: pd.DataFrame, ranges: Dict[str, tuple]) -> bool:
        """Check if values are within specified ranges."""
        phase = (
            f"{self._current_phase}: Range Check"
            if self._current_phase
            else "Range Check"
        )

        has_outliers = False
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue

            outliers = df[~df[col].between(min_val, max_val)]
            if not outliers.empty:
                has_outliers = True
                self.report.add_warning(
                    f"{phase} - Found {len(outliers)} values outside range "
                    f"[{min_val}, {max_val}] in column '{col}'"
                )
        return has_outliers

    def check_distributions(self, df: pd.DataFrame, columns: List[str]):
        """Check data distributions for anomalies."""
        phase = (
            f"{self._current_phase}: Distribution Check"
            if self._current_phase
            else "Distribution Check"
        )

        for col in columns:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Calculate distribution metrics
            mean = df[col].mean()
            std = df[col].std()
            skew = df[col].skew()

            # Check for extreme skewness
            if abs(skew) > 3:
                self.report.add_warning(
                    f"{phase} - Column '{col}' shows high skewness: {skew:.2f}"
                )

            # Check for outliers (3 standard deviations)
            outliers = df[abs(df[col] - mean) > 3 * std]
            if not outliers.empty:
                self.report.add_warning(
                    f"{phase} - Found {len(outliers)} outliers in column '{col}'"
                )

    def check_correlations(self, df: pd.DataFrame, threshold: float = 0.95):
        """Check for highly correlated features."""
        phase = (
            f"{self._current_phase}: Correlation Check"
            if self._current_phase
            else "Correlation Check"
        )

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return

        corr_matrix = df[numeric_cols].corr()
        high_corr = np.where(np.abs(corr_matrix) > threshold)
        high_corr = [
            (corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y])
            for x, y in zip(*high_corr)
            if x != y and x < y
        ]

        if high_corr:
            self.report.add_warning(
                f"{phase} - Found {len(high_corr)} highly correlated feature pairs:"
            )
            for col1, col2, corr in high_corr[:3]:
                self.report.add_warning(f"  • {col1} ↔ {col2}: {corr:.2f}")

    def check_class_balance(self, y: np.ndarray):
        """Check target variable class balance."""
        phase = (
            f"{self._current_phase}: Class Balance"
            if self._current_phase
            else "Class Balance"
        )

        if len(y) == 0:
            self.report.add_error(f"{phase} - Empty target variable")
            return

        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))

        total = sum(counts)
        for cls, count in class_dist.items():
            ratio = count / total
            self.report.add_metric(f"class_{cls}_ratio", ratio)

            if ratio < 0.1:  # Severe imbalance
                self.report.add_warning(
                    f"{phase} - Severe class imbalance for class {cls}: {ratio:.1%}"
                )

    def track_progress(
        self,
        current: int,
        total: int,
        success: Optional[bool] = None,
        error_msg: Optional[str] = None,
    ):
        """Track processing progress."""
        self.report.total_records = total

        if success is True:
            self.report.processed_records += 1
        elif success is False:
            self.report.skipped_records += 1
            if error_msg:
                self.report.add_error(error_msg)

        # Log progress every 10%
        if current % max(1, total // 10) == 0:
            progress = current / total * 100
            logger.info(f"Progress: {progress:.1f}% ({current}/{total})")

    def get_report(self) -> DataQualityReport:
        """Get the current data quality report."""
        return self.report
