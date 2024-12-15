"""Team data and aliases for NFL teams."""

import json
import os
from typing import Dict, Any


def load_team_data() -> Dict[str, Any]:
    """Load team data from JSON file."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(current_dir, "team_data.json")

    with open(json_path, "r") as f:
        return json.load(f)


TEAM_DATA = load_team_data()

# Team name aliases for standardization
TEAM_ALIASES = {
    "ARI": ["Arizona", "Cardinals", "Arizona Cardinals"],
    "ATL": ["Atlanta", "Falcons", "Atlanta Falcons"],
    "BAL": ["Baltimore", "Ravens", "Baltimore Ravens"],
    "BUF": ["Buffalo", "Bills", "Buffalo Bills"],
    "CAR": ["Carolina", "Panthers", "Carolina Panthers"],
    "CHI": ["Chicago", "Bears", "Chicago Bears"],
    "CIN": ["Cincinnati", "Bengals", "Cincinnati Bengals"],
    "CLE": ["Cleveland", "Browns", "Cleveland Browns"],
    "DAL": ["Dallas", "Cowboys", "Dallas Cowboys"],
    "DEN": ["Denver", "Broncos", "Denver Broncos"],
    "DET": ["Detroit", "Lions", "Detroit Lions"],
    "GB": ["Green Bay", "Packers", "Green Bay Packers"],
    "HOU": ["Houston", "Texans", "Houston Texans"],
    "IND": ["Indianapolis", "Colts", "Indianapolis Colts"],
    "JAX": ["Jacksonville", "Jaguars", "Jacksonville Jaguars"],
    "KC": ["Kansas City", "Chiefs", "Kansas City Chiefs"],
    "LAC": ["Los Angeles Chargers", "LA Chargers", "Chargers"],
    "LAR": ["Los Angeles Rams", "LA Rams", "Rams"],
    "LV": ["Las Vegas", "Raiders", "Las Vegas Raiders"],
    "MIA": ["Miami", "Dolphins", "Miami Dolphins"],
    "MIN": ["Minnesota", "Vikings", "Minnesota Vikings"],
    "NE": ["New England", "Patriots", "New England Patriots"],
    "NO": ["New Orleans", "Saints", "New Orleans Saints"],
    "NYG": ["New York Giants", "NY Giants", "Giants"],
    "NYJ": ["New York Jets", "NY Jets", "Jets"],
    "PHI": ["Philadelphia", "Eagles", "Philadelphia Eagles"],
    "PIT": ["Pittsburgh", "Steelers", "Pittsburgh Steelers"],
    "SEA": ["Seattle", "Seahawks", "Seattle Seahawks"],
    "SF": ["San Francisco", "49ers", "San Francisco 49ers"],
    "TB": ["Tampa Bay", "Buccaneers", "Tampa Bay Buccaneers"],
    "TEN": ["Tennessee", "Titans", "Tennessee Titans"],
    "WAS": ["Washington", "Commanders", "Washington Commanders"],
}
