"""
Module for calculating Vimshottari Dasha periods and their influence on match outcomes.
"""

import math
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Dasha periods in years
MAHA_DASHA_PERIODS = {
    "Ketu": 7,
    "Venus": 20,
    "Sun": 6,
    "Moon": 10,
    "Mars": 7,
    "Rahu": 18,
    "Jupiter": 16,
    "Saturn": 19,
    "Mercury": 17,
}

# Planet relationships for dasha effects
PLANET_RELATIONSHIPS = {
    "Sun": {
        "friends": ["Moon", "Mars", "Jupiter"],
        "enemies": ["Saturn", "Venus"],
        "neutral": ["Mercury"],
    },
    "Moon": {
        "friends": ["Sun", "Mercury"],
        "enemies": ["Rahu", "Ketu"],
        "neutral": ["Mars", "Jupiter", "Venus", "Saturn"],
    },
    "Mars": {
        "friends": ["Sun", "Jupiter", "Moon"],
        "enemies": ["Mercury"],
        "neutral": ["Venus", "Saturn"],
    },
    "Mercury": {
        "friends": ["Sun", "Venus"],
        "enemies": ["Moon"],
        "neutral": ["Mars", "Jupiter", "Saturn"],
    },
    "Jupiter": {
        "friends": ["Sun", "Moon", "Mars"],
        "enemies": ["Mercury", "Venus"],
        "neutral": ["Saturn"],
    },
    "Venus": {
        "friends": ["Mercury", "Saturn"],
        "enemies": ["Sun"],
        "neutral": ["Mars", "Jupiter", "Moon"],
    },
    "Saturn": {
        "friends": ["Mercury", "Venus"],
        "enemies": ["Sun", "Moon", "Mars"],
        "neutral": ["Jupiter"],
    },
    "Rahu": {
        "friends": ["Venus", "Saturn"],
        "enemies": ["Sun", "Moon"],
        "neutral": ["Mars", "Jupiter", "Mercury"],
    },
    "Ketu": {
        "friends": ["Mars", "Saturn"],
        "enemies": ["Moon", "Venus"],
        "neutral": ["Sun", "Jupiter", "Mercury"],
    },
}


def calculate_vimshottari_dasha(moon_longitude: float, jd: float) -> Dict[str, Any]:
    """
    Calculate Vimshottari Dasha details for a given time.

    Args:
        moon_longitude: Moon's longitude in degrees
        jd: Julian day number for the time of calculation

    Returns:
        Dictionary containing dasha details and predictions
    """
    moon_data = calculate_moon_nakshatra(moon_longitude)
    dasha_balance = calculate_dasha_balance(moon_data)
    current_dasha = calculate_current_dasha(moon_data, dasha_balance, jd)
    antardasha = calculate_antardasha(current_dasha["lord"], current_dasha["balance"])

    return {
        "current_dasha": current_dasha,
        "antardasha": antardasha,
        "moon_nakshatra": moon_data["nakshatra"],
        "moon_pada": moon_data["pada"],
    }


def calculate_moon_nakshatra(moon_longitude: float) -> Dict[str, Any]:
    """Calculate Moon's nakshatra and pada."""
    nakshatra_span = 360 / 27  # Each nakshatra is 13°20'
    pada_span = nakshatra_span / 4  # Each pada is 3°20'

    nakshatra_num = int(moon_longitude / nakshatra_span)
    pada_num = int((moon_longitude % nakshatra_span) / pada_span) + 1

    nakshatras = [
        "Ashwini",
        "Bharani",
        "Krittika",
        "Rohini",
        "Mrigashira",
        "Ardra",
        "Punarvasu",
        "Pushya",
        "Ashlesha",
        "Magha",
        "Purva Phalguni",
        "Uttara Phalguni",
        "Hasta",
        "Chitra",
        "Swati",
        "Vishakha",
        "Anuradha",
        "Jyeshtha",
        "Mula",
        "Purva Ashadha",
        "Uttara Ashadha",
        "Shravana",
        "Dhanishta",
        "Shatabhisha",
        "Purva Bhadrapada",
        "Uttara Bhadrapada",
        "Revati",
    ]

    nakshatra_lords = {
        "Ashwini": "Ketu",
        "Bharani": "Venus",
        "Krittika": "Sun",
        "Rohini": "Moon",
        "Mrigashira": "Mars",
        "Ardra": "Rahu",
        "Punarvasu": "Jupiter",
        "Pushya": "Saturn",
        "Ashlesha": "Mercury",
        "Magha": "Ketu",
        "Purva Phalguni": "Venus",
        "Uttara Phalguni": "Sun",
        "Hasta": "Moon",
        "Chitra": "Mars",
        "Swati": "Rahu",
        "Vishakha": "Jupiter",
        "Anuradha": "Saturn",
        "Jyeshtha": "Mercury",
        "Mula": "Ketu",
        "Purva Ashadha": "Venus",
        "Uttara Ashadha": "Sun",
        "Shravana": "Moon",
        "Dhanishta": "Mars",
        "Shatabhisha": "Rahu",
        "Purva Bhadrapada": "Jupiter",
        "Uttara Bhadrapada": "Saturn",
        "Revati": "Mercury",
    }

    nakshatra = nakshatras[nakshatra_num]
    lord = nakshatra_lords[nakshatra]

    return {
        "nakshatra": nakshatra,
        "pada": pada_num,
        "lord": lord,
        "longitude": moon_longitude,
    }


def calculate_dasha_balance(moon_data: Dict[str, Any]) -> float:
    """Calculate balance of current dasha at birth."""
    nakshatra_span = 360 / 27
    nakshatra_progress = moon_data["longitude"] % nakshatra_span
    consumed_fraction = nakshatra_progress / nakshatra_span

    return 1 - consumed_fraction


def calculate_current_dasha(
    moon_data: Dict[str, Any], dasha_balance: float, jd: float
) -> Dict[str, Any]:
    """
    Calculate current dasha period for a game.

    Args:
        moon_data: Dictionary containing moon's nakshatra details
        dasha_balance: Balance of current dasha
        jd: Julian day for game time

    Returns:
        Dictionary containing current dasha details
    """
    # For game predictions, we'll use the moon's position to determine the active dasha
    lord = moon_data["lord"]

    # Calculate a normalized balance based on moon's position in nakshatra
    balance = dasha_balance / MAHA_DASHA_PERIODS[lord]

    return {
        "lord": lord,
        "balance": balance,
        "total_years": MAHA_DASHA_PERIODS[lord],
        "remaining_years": balance * MAHA_DASHA_PERIODS[lord],
    }


def calculate_antardasha(maha_lord: str, balance: float) -> Dict[str, str]:
    """Calculate current antardasha (sub-period)."""
    order = list(MAHA_DASHA_PERIODS.keys())
    start_idx = order.index(maha_lord)

    total_period = MAHA_DASHA_PERIODS[maha_lord]
    elapsed = (1 - balance) * total_period

    for i in range(len(order)):
        lord = order[(start_idx + i) % len(order)]
        sub_period = (MAHA_DASHA_PERIODS[lord] / total_period) * MAHA_DASHA_PERIODS[
            maha_lord
        ]

        if elapsed < sub_period:
            return {"lord": lord, "balance": (sub_period - elapsed) / sub_period}
        elapsed -= sub_period

    return {"lord": maha_lord, "balance": 0}


def analyze_dasha_effects(
    dasha_data: Dict[str, Any], planet_positions: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Analyze the effects of current dasha period."""

    def calculate_strength(planet: str) -> float:
        if planet not in planet_positions:
            return 0

        pos = planet_positions[planet]
        strength = 0

        # Directional strength
        if "longitude" in pos:
            zodiac_house = int(pos["longitude"] / 30) + 1
            if zodiac_house in [1, 4, 7, 10]:  # Angular houses
                strength += 1
            elif zodiac_house in [2, 5, 8, 11]:  # Succedent houses
                strength += 0.5

        # Speed strength
        if "speed" in pos:
            if pos["speed"] > 1:  # Fast
                strength += 0.5
            elif pos["speed"] < -0.5:  # Retrograde
                strength -= 0.5

        return strength

    maha_lord = dasha_data["maha_dasha"]
    antar_lord = dasha_data["antardasha"]["lord"]

    maha_strength = calculate_strength(maha_lord)
    antar_strength = calculate_strength(antar_lord)

    # Calculate relationship effects
    relationship_effect = 0
    if antar_lord in PLANET_RELATIONSHIPS[maha_lord]["friends"]:
        relationship_effect = 1
    elif antar_lord in PLANET_RELATIONSHIPS[maha_lord]["enemies"]:
        relationship_effect = -1

    total_effect = (
        maha_strength * 0.6 + antar_strength * 0.3 + relationship_effect * 0.1
    )

    return {
        "maha_dasha_effect": maha_strength,
        "antardasha_effect": antar_strength,
        "relationship_effect": relationship_effect,
        "total_effect": total_effect,
    }


def get_dasha_prediction(
    dasha_effects: Dict[str, float], threshold: float = 0.5
) -> str:
    """Generate prediction based on dasha effects."""
    total_effect = dasha_effects["total_effect"]

    if abs(total_effect) < threshold:
        return "The current dasha period shows neutral influences."
    elif total_effect > threshold:
        return f"The current dasha period is favorable, with a strength of {total_effect:.2f}"
    else:
        return f"The current dasha period is challenging, with a strength of {total_effect:.2f}"
