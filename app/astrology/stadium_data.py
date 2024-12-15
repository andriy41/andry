"""
NFL Stadium data and location utilities
"""
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from typing import Dict, Optional, Tuple


def scrape_nfl_stadiums() -> Dict:
    # NFL Teams and their stadiums (2023-2024 season)
    nfl_stadiums = {
        "ARI": {
            "stadium": "State Farm Stadium",
            "location": "Glendale, Arizona",
            "coordinates": {"lat": 33.5276, "lng": -112.2626},
        },
        "ATL": {
            "stadium": "Mercedes-Benz Stadium",
            "location": "Atlanta, Georgia",
            "coordinates": {"lat": 33.7555, "lng": -84.4011},
        },
        "BAL": {
            "stadium": "M&T Bank Stadium",
            "location": "Baltimore, Maryland",
            "coordinates": {"lat": 39.2780, "lng": -76.6227},
        },
        "BUF": {
            "stadium": "Highmark Stadium",
            "location": "Orchard Park, New York",
            "coordinates": {"lat": 42.7738, "lng": -78.7870},
        },
        "CAR": {
            "stadium": "Bank of America Stadium",
            "location": "Charlotte, North Carolina",
            "coordinates": {"lat": 35.2258, "lng": -80.8528},
        },
        "CHI": {
            "stadium": "Soldier Field",
            "location": "Chicago, Illinois",
            "coordinates": {"lat": 41.8623, "lng": -87.6167},
        },
        "CIN": {
            "stadium": "Paycor Stadium",
            "location": "Cincinnati, Ohio",
            "coordinates": {"lat": 39.0955, "lng": -84.5161},
        },
        "CLE": {
            "stadium": "Cleveland Browns Stadium",
            "location": "Cleveland, Ohio",
            "coordinates": {"lat": 41.5061, "lng": -81.6995},
        },
        "DAL": {
            "stadium": "AT&T Stadium",
            "location": "Arlington, Texas",
            "coordinates": {"lat": 32.7473, "lng": -97.0945},
        },
        "DEN": {
            "stadium": "Empower Field at Mile High",
            "location": "Denver, Colorado",
            "coordinates": {"lat": 39.7439, "lng": -105.0201},
        },
        "DET": {
            "stadium": "Ford Field",
            "location": "Detroit, Michigan",
            "coordinates": {"lat": 42.3400, "lng": -83.0456},
        },
        "GB": {
            "stadium": "Lambeau Field",
            "location": "Green Bay, Wisconsin",
            "coordinates": {"lat": 44.5013, "lng": -88.0622},
        },
        "HOU": {
            "stadium": "NRG Stadium",
            "location": "Houston, Texas",
            "coordinates": {"lat": 29.6847, "lng": -95.4107},
        },
        "IND": {
            "stadium": "Lucas Oil Stadium",
            "location": "Indianapolis, Indiana",
            "coordinates": {"lat": 39.7601, "lng": -86.1639},
        },
        "JAX": {
            "stadium": "TIAA Bank Field",
            "location": "Jacksonville, Florida",
            "coordinates": {"lat": 30.3239, "lng": -81.6373},
        },
        "KC": {
            "stadium": "GEHA Field at Arrowhead Stadium",
            "location": "Kansas City, Missouri",
            "coordinates": {"lat": 39.0489, "lng": -94.4839},
        },
        "LV": {
            "stadium": "Allegiant Stadium",
            "location": "Las Vegas, Nevada",
            "coordinates": {"lat": 36.0909, "lng": -115.1833},
        },
        "LAC": {
            "stadium": "SoFi Stadium",
            "location": "Inglewood, California",
            "coordinates": {"lat": 33.9534, "lng": -118.3387},
        },
        "LAR": {
            "stadium": "SoFi Stadium",
            "location": "Inglewood, California",
            "coordinates": {"lat": 33.9534, "lng": -118.3387},
        },
        "MIA": {
            "stadium": "Hard Rock Stadium",
            "location": "Miami Gardens, Florida",
            "coordinates": {"lat": 25.9580, "lng": -80.2389},
        },
        "MIN": {
            "stadium": "U.S. Bank Stadium",
            "location": "Minneapolis, Minnesota",
            "coordinates": {"lat": 44.9735, "lng": -93.2575},
        },
        "NE": {
            "stadium": "Gillette Stadium",
            "location": "Foxborough, Massachusetts",
            "coordinates": {"lat": 42.0909, "lng": -71.2643},
        },
        "NO": {
            "stadium": "Caesars Superdome",
            "location": "New Orleans, Louisiana",
            "coordinates": {"lat": 29.9511, "lng": -90.0814},
        },
        "NYG": {
            "stadium": "MetLife Stadium",
            "location": "East Rutherford, New Jersey",
            "coordinates": {"lat": 40.8135, "lng": -74.0745},
        },
        "NYJ": {
            "stadium": "MetLife Stadium",
            "location": "East Rutherford, New Jersey",
            "coordinates": {"lat": 40.8135, "lng": -74.0745},
        },
        "PHI": {
            "stadium": "Lincoln Financial Field",
            "location": "Philadelphia, Pennsylvania",
            "coordinates": {"lat": 39.9008, "lng": -75.1675},
        },
        "PIT": {
            "stadium": "Acrisure Stadium",
            "location": "Pittsburgh, Pennsylvania",
            "coordinates": {"lat": 40.4468, "lng": -80.0158},
        },
        "SF": {
            "stadium": "Levi's Stadium",
            "location": "Santa Clara, California",
            "coordinates": {"lat": 37.4032, "lng": -121.9697},
        },
        "SEA": {
            "stadium": "Lumen Field",
            "location": "Seattle, Washington",
            "coordinates": {"lat": 47.5952, "lng": -122.3316},
        },
        "TB": {
            "stadium": "Raymond James Stadium",
            "location": "Tampa, Florida",
            "coordinates": {"lat": 27.9759, "lng": -82.5033},
        },
        "TEN": {
            "stadium": "Nissan Stadium",
            "location": "Nashville, Tennessee",
            "coordinates": {"lat": 36.1665, "lng": -86.7713},
        },
        "WSH": {
            "stadium": "FedExField",
            "location": "Landover, Maryland",
            "coordinates": {"lat": 38.9076, "lng": -76.8645},
        },
        # Legacy teams and alternate names
        "OAK": {
            "stadium": "Allegiant Stadium",
            "location": "Las Vegas, Nevada",
            "coordinates": {"lat": 36.0909, "lng": -115.1833},
        },  # Now LV Raiders
        "STL": {
            "stadium": "SoFi Stadium",
            "location": "Inglewood, California",
            "coordinates": {"lat": 33.9534, "lng": -118.3387},
        },  # Now LA Rams
        "SD": {
            "stadium": "SoFi Stadium",
            "location": "Inglewood, California",
            "coordinates": {"lat": 33.9534, "lng": -118.3387},
        },  # Now LA Chargers
    }
    return nfl_stadiums


def get_stadium_coordinates(team: str) -> Optional[Tuple[float, float]]:
    """Get the coordinates for a team's stadium."""
    stadiums = scrape_nfl_stadiums()
    team = team.upper()

    if team in stadiums:
        coords = stadiums[team]["coordinates"]
        return coords["lat"], coords["lng"]
    else:
        return None


def load_stadium_data() -> pd.DataFrame:
    """Load stadium data into a DataFrame."""
    stadiums = scrape_nfl_stadiums()
    data = []

    for team, info in stadiums.items():
        data.append(
            {
                "team": team,
                "stadium": info["stadium"],
                "location": info["location"],
                "latitude": info["coordinates"]["lat"],
                "longitude": info["coordinates"]["lng"],
            }
        )

    return pd.DataFrame(data)
