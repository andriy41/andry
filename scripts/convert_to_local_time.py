"""
Convert game times to local time for NFL games
"""
import pandas as pd
from datetime import datetime
import pytz
import json
import os


def load_stadium_data():
    """Load stadium timezone data"""
    stadium_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "stadium_data.json"
    )
    if os.path.exists(stadium_file):
        with open(stadium_file, "r") as f:
            return json.load(f)
    return {}


def get_stadium_timezone(stadium, stadium_data):
    """Get timezone for a stadium"""
    if stadium in stadium_data:
        return pytz.timezone(stadium_data[stadium]["timezone"])
    return pytz.timezone("US/Eastern")  # Default NFL timezone


def convert_game_times(games_df, stadium_data):
    """Convert game times to local stadium time"""
    for idx, game in games_df.iterrows():
        stadium = game["stadium"]
        game_time = pd.to_datetime(game["game_time"])

        # Convert to stadium local time
        stadium_tz = get_stadium_timezone(stadium, stadium_data)
        local_time = game_time.astimezone(stadium_tz)

        games_df.at[idx, "local_time"] = local_time.strftime("%Y-%m-%d %H:%M:%S")

    return games_df


def main():
    # Load stadium data
    stadium_data = load_stadium_data()

    # Load games data
    games_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "upcoming_games.csv"
    )
    if os.path.exists(games_file):
        games_df = pd.read_csv(games_file)
        games_df = convert_game_times(games_df, stadium_data)
        games_df.to_csv(games_file, index=False)
        print("Updated game times to local time")


if __name__ == "__main__":
    main()
