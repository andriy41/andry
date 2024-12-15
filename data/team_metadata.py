"""Team metadata and constants for NFL predictions."""

# Passing yards multiplier based on climate
PASSING_MULTIPLIER = {"dome": 1.2, "warm": 1.1, "moderate": 1.0, "cold": 0.9}

# Dome stadium multiplier
DOME_MULTIPLIER = 1.15

# Base penalty rates by division
PENALTY_BASE_RATE = {
    "AFC_EAST": 1.0,
    "AFC_NORTH": 1.1,
    "AFC_SOUTH": 0.9,
    "AFC_WEST": 1.0,
    "NFC_EAST": 1.1,
    "NFC_NORTH": 1.0,
    "NFC_SOUTH": 0.9,
    "NFC_WEST": 1.0,
}

# Team metadata
TEAM_INFO = {
    "ARI": {
        "name": "Arizona Cardinals",
        "division": "NFC_WEST",
        "conference": "NFC",
        "climate": "hot",
        "dome": True,
    },
    "ATL": {
        "name": "Atlanta Falcons",
        "division": "NFC_SOUTH",
        "conference": "NFC",
        "climate": "warm",
        "dome": True,
    },
    "BAL": {
        "name": "Baltimore Ravens",
        "division": "AFC_NORTH",
        "conference": "AFC",
        "climate": "moderate",
        "dome": False,
    },
    "BUF": {
        "name": "Buffalo Bills",
        "division": "AFC_EAST",
        "conference": "AFC",
        "climate": "cold",
        "dome": False,
    },
    "CAR": {
        "name": "Carolina Panthers",
        "division": "NFC_SOUTH",
        "conference": "NFC",
        "climate": "warm",
        "dome": False,
    },
    "CHI": {
        "name": "Chicago Bears",
        "division": "NFC_NORTH",
        "conference": "NFC",
        "climate": "cold",
        "dome": False,
    },
    "CIN": {
        "name": "Cincinnati Bengals",
        "division": "AFC_NORTH",
        "conference": "AFC",
        "climate": "moderate",
        "dome": False,
    },
    "CLE": {
        "name": "Cleveland Browns",
        "division": "AFC_NORTH",
        "conference": "AFC",
        "climate": "cold",
        "dome": False,
    },
    "DAL": {
        "name": "Dallas Cowboys",
        "division": "NFC_EAST",
        "conference": "NFC",
        "climate": "warm",
        "dome": True,
    },
    "DEN": {
        "name": "Denver Broncos",
        "division": "AFC_WEST",
        "conference": "AFC",
        "climate": "cold",
        "dome": False,
        "altitude": True,
    },
    "DET": {
        "name": "Detroit Lions",
        "division": "NFC_NORTH",
        "conference": "NFC",
        "climate": "cold",
        "dome": True,
    },
    "GB": {
        "name": "Green Bay Packers",
        "division": "NFC_NORTH",
        "conference": "NFC",
        "climate": "cold",
        "dome": False,
    },
    "HOU": {
        "name": "Houston Texans",
        "division": "AFC_SOUTH",
        "conference": "AFC",
        "climate": "hot",
        "dome": True,
    },
    "IND": {
        "name": "Indianapolis Colts",
        "division": "AFC_SOUTH",
        "conference": "AFC",
        "climate": "moderate",
        "dome": True,
    },
    "JAX": {
        "name": "Jacksonville Jaguars",
        "division": "AFC_SOUTH",
        "conference": "AFC",
        "climate": "warm",
        "dome": False,
    },
    "KC": {
        "name": "Kansas City Chiefs",
        "division": "AFC_WEST",
        "conference": "AFC",
        "climate": "moderate",
        "dome": False,
    },
    "LAC": {
        "name": "Los Angeles Chargers",
        "division": "AFC_WEST",
        "conference": "AFC",
        "climate": "warm",
        "dome": True,
    },
    "LAR": {
        "name": "Los Angeles Rams",
        "division": "NFC_WEST",
        "conference": "NFC",
        "climate": "warm",
        "dome": True,
    },
    "LV": {
        "name": "Las Vegas Raiders",
        "division": "AFC_WEST",
        "conference": "AFC",
        "climate": "hot",
        "dome": True,
    },
    "MIA": {
        "name": "Miami Dolphins",
        "division": "AFC_EAST",
        "conference": "AFC",
        "climate": "hot",
        "dome": False,
    },
    "MIN": {
        "name": "Minnesota Vikings",
        "division": "NFC_NORTH",
        "conference": "NFC",
        "climate": "cold",
        "dome": True,
    },
    "NE": {
        "name": "New England Patriots",
        "division": "AFC_EAST",
        "conference": "AFC",
        "climate": "cold",
        "dome": False,
    },
    "NO": {
        "name": "New Orleans Saints",
        "division": "NFC_SOUTH",
        "conference": "NFC",
        "climate": "warm",
        "dome": True,
    },
    "NYG": {
        "name": "New York Giants",
        "division": "NFC_EAST",
        "conference": "NFC",
        "climate": "moderate",
        "dome": False,
    },
    "NYJ": {
        "name": "New York Jets",
        "division": "AFC_EAST",
        "conference": "AFC",
        "climate": "moderate",
        "dome": False,
    },
    "PHI": {
        "name": "Philadelphia Eagles",
        "division": "NFC_EAST",
        "conference": "NFC",
        "climate": "moderate",
        "dome": False,
    },
    "PIT": {
        "name": "Pittsburgh Steelers",
        "division": "AFC_NORTH",
        "conference": "AFC",
        "climate": "cold",
        "dome": False,
    },
    "SEA": {
        "name": "Seattle Seahawks",
        "division": "NFC_WEST",
        "conference": "NFC",
        "climate": "moderate",
        "dome": False,
    },
    "SF": {
        "name": "San Francisco 49ers",
        "division": "NFC_WEST",
        "conference": "NFC",
        "climate": "moderate",
        "dome": False,
    },
    "TB": {
        "name": "Tampa Bay Buccaneers",
        "division": "NFC_SOUTH",
        "conference": "NFC",
        "climate": "hot",
        "dome": False,
    },
    "TEN": {
        "name": "Tennessee Titans",
        "division": "AFC_SOUTH",
        "conference": "AFC",
        "climate": "warm",
        "dome": False,
    },
    "WSH": {
        "name": "Washington Commanders",
        "division": "NFC_EAST",
        "conference": "NFC",
        "climate": "moderate",
        "dome": False,
    },
    "WAS": {
        "name": "Washington Commanders",
        "division": "NFC_EAST",
        "conference": "NFC",
        "climate": "moderate",
        "dome": False,
    },
}


def get_team_info(team_code):
    """Get metadata for a team by its code."""
    return TEAM_INFO.get(team_code, {})
