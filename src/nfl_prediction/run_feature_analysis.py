"""Run the feature impact analysis."""

from analysis.feature_impact_analysis import FeatureImpactAnalyzer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Create analysis directory if it doesn't exist
    os.makedirs("analysis/results", exist_ok=True)

    logger.info("Starting feature impact analysis...")
    analyzer = FeatureImpactAnalyzer()
    analyzer.analyze_impact("data/processed_vedic/nfl_games_with_vedic.csv")
    logger.info(
        "Analysis complete. Check analysis/results directory for plots and metrics."
    )


if __name__ == "__main__":
    main()
