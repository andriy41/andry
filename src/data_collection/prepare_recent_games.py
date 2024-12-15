"""
Prepare recent NFL games data for prediction
Processes and formats recent game data for the prediction system
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta


class NFLGamePreparation:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.recent_games_file = os.path.join(self.data_dir, "recent_nfl_games.json")
        self.boxscores_dir = os.path.join(self.data_dir, "boxscores")
        self.team_stats_dir = os.path.join(self.data_dir, "team_stats")

    def load_recent_boxscores(self):
        """Load recent game boxscores"""
        boxscores = {}
        for file in os.listdir(self.boxscores_dir):
            if file.startswith("boxscore_") and file.endswith(".json"):
                with open(os.path.join(self.boxscores_dir, file), "r") as f:
                    game_id = file.replace("boxscore_", "").replace(".json", "")
                    boxscores[game_id] = json.load(f)
        return boxscores

    def load_team_stats(self):
        """Load current team statistics"""
        stats = {}
        for file in os.listdir(self.team_stats_dir):
            if file.endswith(".csv"):
                stats[file.replace(".csv", "")] = pd.read_csv(
                    os.path.join(self.team_stats_dir, file)
                )
        return stats

    def prepare_game_data(self, boxscores, team_stats):
        """Prepare game data with relevant features"""
        games_data = []
        for game_id, boxscore in boxscores.items():
            game_data = {
                "game_id": game_id,
                "date": boxscore.get("date"),
                "home_team": boxscore.get("home_team"),
                "away_team": boxscore.get("away_team"),
                # Add more features as needed
            }
            games_data.append(game_data)

        return pd.DataFrame(games_data)

    def save_prepared_data(self, prepared_data):
        """Save prepared game data"""
        prepared_data.to_json(self.recent_games_file)
        print(f"Saved prepared data to {self.recent_games_file}")


def main():
    preparer = NFLGamePreparation()
    boxscores = preparer.load_recent_boxscores()
    team_stats = preparer.load_team_stats()
    prepared_data = preparer.prepare_game_data(boxscores, team_stats)
    preparer.save_prepared_data(prepared_data)


if __name__ == "__main__":
    main()
