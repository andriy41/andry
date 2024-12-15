"""Automated daily updates for NFL data collection."""

import pandas as pd
import nfl_data_py as nfl
from .nfl_data_collector import NFLDataCollector
from datetime import datetime
import logging
import os
from pathlib import Path
import json
import time
from typing import Dict, List, Optional

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            log_dir / f"nfl_updates_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class NFLDataUpdater:
    """Handles automated updates of NFL data."""

    def __init__(self):
        """Initialize the updater."""
        self.collector = NFLDataCollector()
        self.data_dir = Path("data/2024_season")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.last_update_file = self.data_dir / "last_update.json"
        self.current_season = 2024

    def _load_last_update(self) -> Dict:
        """Load the timestamp of last successful update for each data type."""
        if self.last_update_file.exists():
            with open(self.last_update_file, "r") as f:
                return json.load(f)
        return {}

    def _save_last_update(self, updates: Dict):
        """Save the timestamp of successful updates."""
        with open(self.last_update_file, "w") as f:
            json.dump(updates, f, indent=2)

    def _should_update(self, data_type: str, hours_threshold: int = 24) -> bool:
        """Check if a data type should be updated based on last update time."""
        last_updates = self._load_last_update()
        if data_type not in last_updates:
            return True

        last_update = datetime.fromisoformat(last_updates[data_type])
        hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
        return hours_since_update >= hours_threshold

    def update_schedule(self) -> bool:
        """Update the season schedule."""
        try:
            if not self._should_update("schedule", hours_threshold=12):
                logger.info("Schedule is up to date")
                return True

            logger.info("Updating schedule...")
            schedule = nfl.import_schedules([self.current_season])

            # Load existing schedule to check for changes
            old_schedule_path = self.data_dir / "schedule_2024.csv"
            if old_schedule_path.exists():
                old_schedule = pd.read_csv(old_schedule_path)
                if not schedule.equals(old_schedule):
                    # Backup old schedule before updating
                    backup_path = (
                        self.data_dir
                        / f"schedule_2024_backup_{datetime.now().strftime('%Y%m%d')}.csv"
                    )
                    old_schedule.to_csv(backup_path, index=False)

            schedule.to_csv(self.data_dir / "schedule_2024.csv", index=False)

            updates = self._load_last_update()
            updates["schedule"] = datetime.now().isoformat()
            self._save_last_update(updates)

            logger.info(f"Schedule updated successfully with {len(schedule)} games")
            return True

        except Exception as e:
            logger.error(f"Error updating schedule: {str(e)}")
            return False

    def update_team_stats(self) -> bool:
        """Update team statistics."""
        try:
            if not self._should_update("team_stats", hours_threshold=6):
                logger.info("Team stats are up to date")
                return True

            logger.info("Updating team stats...")
            team_stats = {}
            for team in self.collector.team_name_map.keys():
                try:
                    stats = self.collector._fetch_pfr_team_stats(
                        team, self.current_season
                    )
                    team_stats[team] = stats
                    logger.info(f"Updated stats for {team}")
                    time.sleep(2)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error updating stats for {team}: {str(e)}")

            # Save team stats
            stats_df = pd.DataFrame.from_dict(team_stats, orient="index")

            # Backup existing stats
            old_stats_path = self.data_dir / "team_stats_2024.csv"
            if old_stats_path.exists():
                backup_path = (
                    self.data_dir
                    / f"team_stats_2024_backup_{datetime.now().strftime('%Y%m%d')}.csv"
                )
                pd.read_csv(old_stats_path).to_csv(backup_path, index=False)

            stats_df.to_csv(self.data_dir / "team_stats_2024.csv")

            updates = self._load_last_update()
            updates["team_stats"] = datetime.now().isoformat()
            self._save_last_update(updates)

            logger.info("Team stats updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating team stats: {str(e)}")
            return False

    def update_players(self) -> bool:
        """Update player information."""
        try:
            if not self._should_update("players", hours_threshold=24):
                logger.info("Player data is up to date")
                return True

            logger.info("Updating player data...")
            players = nfl.import_players()
            players_2024 = players[players["status"] == "ACT"]

            # Backup existing player data
            old_players_path = self.data_dir / "players_2024.csv"
            if old_players_path.exists():
                backup_path = (
                    self.data_dir
                    / f"players_2024_backup_{datetime.now().strftime('%Y%m%d')}.csv"
                )
                pd.read_csv(old_players_path).to_csv(backup_path, index=False)

            players_2024.to_csv(self.data_dir / "players_2024.csv", index=False)

            updates = self._load_last_update()
            updates["players"] = datetime.now().isoformat()
            self._save_last_update(updates)

            logger.info(
                f"Player data updated successfully with {len(players_2024)} active players"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating player data: {str(e)}")
            return False

    def update_injuries(self) -> bool:
        """Update injury reports."""
        try:
            if not self._should_update("injuries", hours_threshold=3):
                logger.info("Injury data is up to date")
                return True

            logger.info("Updating injury reports...")
            injuries = self.collector.fetch_injury_reports()

            # Backup existing injury data
            old_injuries_path = self.data_dir / "injuries_current.csv"
            if old_injuries_path.exists():
                backup_path = (
                    self.data_dir
                    / f"injuries_backup_{datetime.now().strftime('%Y%m%d')}.csv"
                )
                pd.read_csv(old_injuries_path).to_csv(backup_path, index=False)

            injuries.to_csv(self.data_dir / "injuries_current.csv", index=False)

            updates = self._load_last_update()
            updates["injuries"] = datetime.now().isoformat()
            self._save_last_update(updates)

            logger.info(
                f"Injury reports updated successfully with {len(injuries)} records"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating injury reports: {str(e)}")
            return False

    def update_weekly_data(self) -> bool:
        """Update weekly performance data."""
        try:
            if not self._should_update("weekly_data", hours_threshold=6):
                logger.info("Weekly data is up to date")
                return True

            logger.info("Updating weekly data...")
            weekly_data = nfl.import_weekly_data([self.current_season])

            # Backup existing weekly data
            old_weekly_path = self.data_dir / "weekly_data_2024.csv"
            if old_weekly_path.exists():
                backup_path = (
                    self.data_dir
                    / f"weekly_data_2024_backup_{datetime.now().strftime('%Y%m%d')}.csv"
                )
                pd.read_csv(old_weekly_path).to_csv(backup_path, index=False)

            weekly_data.to_csv(self.data_dir / "weekly_data_2024.csv", index=False)

            updates = self._load_last_update()
            updates["weekly_data"] = datetime.now().isoformat()
            self._save_last_update(updates)

            logger.info("Weekly data updated successfully")
            return True

        except Exception as e:
            logger.error(f"Error updating weekly data: {str(e)}")
            return False

    def update_all(self) -> Dict[str, bool]:
        """Update all data types."""
        results = {
            "schedule": self.update_schedule(),
            "team_stats": self.update_team_stats(),
            "players": self.update_players(),
            "injuries": self.update_injuries(),
            "weekly_data": self.update_weekly_data(),
        }

        # Log overall results
        success_count = sum(1 for result in results.values() if result)
        logger.info(f"Update completed: {success_count}/{len(results)} successful")

        return results


def main():
    """Main function to run updates."""
    logger.info("Starting automated NFL data update")

    try:
        updater = NFLDataUpdater()
        results = updater.update_all()

        # Generate update report
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "success_rate": f"{(sum(1 for r in results.values() if r) / len(results)) * 100:.1f}%",
        }

        # Save report
        report_dir = Path("reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = (
            report_dir
            / f"update_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Update complete. Report saved to {report_file}")

    except Exception as e:
        logger.error(f"Critical error in update process: {str(e)}")
        raise


if __name__ == "__main__":
    main()
