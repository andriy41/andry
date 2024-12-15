"""Process NFL data with Vedic features."""

from data_processing.data_processor import NFLDataProcessor
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    processor = NFLDataProcessor()

    # Create directories if they don't exist
    os.makedirs("data/processed_vedic", exist_ok=True)

    # Process data with new Vedic features
    processor.process_data(
        input_path="data/nfl_games_2019_2024.csv",
        output_path="data/processed_vedic/nfl_games_with_vedic.csv",
    )

    logger.info(
        "Data processing complete. Output saved to data/processed_vedic/nfl_games_with_vedic.csv"
    )


if __name__ == "__main__":
    main()
