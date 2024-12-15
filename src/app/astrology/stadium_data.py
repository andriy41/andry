"""NFL Stadium Coordinates Data"""

import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Stadium coordinates for all NFL teams
NFL_STADIUMS = {
    "ARI": {"name": "State Farm Stadium", "latitude": 33.5276, "longitude": -112.2626},
    "ATL": {
        "name": "Mercedes-Benz Stadium",
        "latitude": 33.7555,
        "longitude": -84.4008,
    },
    "BAL": {"name": "M&T Bank Stadium", "latitude": 39.2780, "longitude": -76.6227},
    "BUF": {"name": "Highmark Stadium", "latitude": 42.7738, "longitude": -78.7870},
    "CAR": {
        "name": "Bank of America Stadium",
        "latitude": 35.2258,
        "longitude": -80.8528,
    },
    "CHI": {"name": "Soldier Field", "latitude": 41.8623, "longitude": -87.6167},
    "CIN": {"name": "Paycor Stadium", "latitude": 39.0955, "longitude": -84.5161},
    "CLE": {"name": "FirstEnergy Stadium", "latitude": 41.5061, "longitude": -81.6995},
    "DAL": {"name": "AT&T Stadium", "latitude": 32.7473, "longitude": -97.0945},
    "DEN": {
        "name": "Empower Field at Mile High",
        "latitude": 39.7439,
        "longitude": -105.0201,
    },
    "DET": {"name": "Ford Field", "latitude": 42.3400, "longitude": -83.0456},
    "GB": {"name": "Lambeau Field", "latitude": 44.5013, "longitude": -88.0622},
    "HOU": {"name": "NRG Stadium", "latitude": 29.6847, "longitude": -95.4107},
    "IND": {"name": "Lucas Oil Stadium", "latitude": 39.7601, "longitude": -86.1639},
    "JAX": {"name": "TIAA Bank Field", "latitude": 30.3239, "longitude": -81.6373},
    "KC": {
        "name": "GEHA Field at Arrowhead Stadium",
        "latitude": 39.0489,
        "longitude": -94.4839,
    },
    "LAC": {"name": "SoFi Stadium", "latitude": 33.9534, "longitude": -118.3387},
    "LAR": {"name": "SoFi Stadium", "latitude": 33.9534, "longitude": -118.3387},
    "LV": {"name": "Allegiant Stadium", "latitude": 36.0909, "longitude": -115.1833},
    "MIA": {"name": "Hard Rock Stadium", "latitude": 25.9580, "longitude": -80.2389},
    "MIN": {"name": "U.S. Bank Stadium", "latitude": 44.9735, "longitude": -93.2575},
    "NE": {"name": "Gillette Stadium", "latitude": 42.0909, "longitude": -71.2643},
    "NO": {"name": "Caesars Superdome", "latitude": 29.9511, "longitude": -90.0814},
    "NYG": {"name": "MetLife Stadium", "latitude": 40.8135, "longitude": -74.0745},
    "NYJ": {"name": "MetLife Stadium", "latitude": 40.8135, "longitude": -74.0745},
    "PHI": {
        "name": "Lincoln Financial Field",
        "latitude": 39.9008,
        "longitude": -75.1675,
    },
    "PIT": {"name": "Acrisure Stadium", "latitude": 40.4468, "longitude": -80.0158},
    "SEA": {"name": "Lumen Field", "latitude": 47.5952, "longitude": -122.3316},
    "SF": {"name": "Levi's Stadium", "latitude": 37.4033, "longitude": -121.9694},
    "TB": {"name": "Raymond James Stadium", "latitude": 27.9759, "longitude": -82.5033},
    "TEN": {"name": "Nissan Stadium", "latitude": 36.1665, "longitude": -86.7713},
    "WAS": {"name": "FedExField", "latitude": 38.9077, "longitude": -76.8645},
    "WSH": {
        "name": "FedExField",
        "latitude": 38.9077,
        "longitude": -76.8645,
    },  # Alternative code
    "LA": {
        "name": "SoFi Stadium",
        "latitude": 33.9534,
        "longitude": -118.3387,
    },  # Old LA team code
}

# Team name variations mapping
TEAM_NAME_MAPPING = {
    "Arizona": "ARI",
    "Cardinals": "ARI",
    "ARI": "ARI",
    "Atlanta": "ATL",
    "Falcons": "ATL",
    "ATL": "ATL",
    "Baltimore": "BAL",
    "Ravens": "BAL",
    "BAL": "BAL",
    "Buffalo": "BUF",
    "Bills": "BUF",
    "BUF": "BUF",
    "Carolina": "CAR",
    "Panthers": "CAR",
    "CAR": "CAR",
    "Chicago": "CHI",
    "Bears": "CHI",
    "CHI": "CHI",
    "Cincinnati": "CIN",
    "Bengals": "CIN",
    "CIN": "CIN",
    "Cleveland": "CLE",
    "Browns": "CLE",
    "CLE": "CLE",
    "Dallas": "DAL",
    "Cowboys": "DAL",
    "DAL": "DAL",
    "Denver": "DEN",
    "Broncos": "DEN",
    "DEN": "DEN",
    "Detroit": "DET",
    "Lions": "DET",
    "DET": "DET",
    "Green Bay": "GB",
    "Packers": "GB",
    "GB": "GB",
    "Houston": "HOU",
    "Texans": "HOU",
    "HOU": "HOU",
    "Indianapolis": "IND",
    "Colts": "IND",
    "IND": "IND",
    "Jacksonville": "JAX",
    "Jaguars": "JAX",
    "JAX": "JAX",
    "JAC": "JAX",
    "Kansas City": "KC",
    "Chiefs": "KC",
    "KC": "KC",
    "Los Angeles Chargers": "LAC",
    "Chargers": "LAC",
    "LAC": "LAC",
    "Los Angeles Rams": "LAR",
    "Rams": "LAR",
    "LAR": "LAR",
    "LA": "LAR",
    "Las Vegas": "LV",
    "Raiders": "LV",
    "LV": "LV",
    "Miami": "MIA",
    "Dolphins": "MIA",
    "MIA": "MIA",
    "Minnesota": "MIN",
    "Vikings": "MIN",
    "MIN": "MIN",
    "New England": "NE",
    "Patriots": "NE",
    "NE": "NE",
    "New Orleans": "NO",
    "Saints": "NO",
    "NO": "NO",
    "New York Giants": "NYG",
    "Giants": "NYG",
    "NYG": "NYG",
    "New York Jets": "NYJ",
    "Jets": "NYJ",
    "NYJ": "NYJ",
    "Philadelphia": "PHI",
    "Eagles": "PHI",
    "PHI": "PHI",
    "Pittsburgh": "PIT",
    "Steelers": "PIT",
    "PIT": "PIT",
    "Seattle": "SEA",
    "Seahawks": "SEA",
    "SEA": "SEA",
    "San Francisco": "SF",
    "49ers": "SF",
    "SF": "SF",
    "Tampa Bay": "TB",
    "Buccaneers": "TB",
    "TB": "TB",
    "Tennessee": "TEN",
    "Titans": "TEN",
    "TEN": "TEN",
    "Washington": "WAS",
    "Commanders": "WAS",
    "Football Team": "WAS",
    "Redskins": "WAS",
    "WAS": "WAS",
    "WSH": "WAS",  # Handle both WAS and WSH codes
}


def get_stadium_info(team_name):
    """Get stadium information for a team"""
    # Convert team name to standard code
    team_code = TEAM_NAME_MAPPING.get(team_name)
    if not team_code:
        return None

    # Get stadium info
    return NFL_STADIUMS.get(team_code)


def get_team_code(team_name):
    """Get standardized team code"""
    return TEAM_NAME_MAPPING.get(team_name)
