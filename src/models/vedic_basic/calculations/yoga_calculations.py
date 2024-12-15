"""
Module for calculating Vedic yogas (planetary combinations) for NFL predictions.
"""

from typing import Dict, List, Any, Tuple
import swisseph as swe
import math
import logging

logger = logging.getLogger(__name__)


def calculate_victory_yogas(jd: float, lat: float, lon: float) -> List[Dict[str, Any]]:
    """
    Calculate yogas that indicate victory or success.

    Args:
        jd: Julian day number
        lat: Latitude
        lon: Longitude

    Returns:
        List of yoga dictionaries with name and strength
    """
    try:
        # Set sidereal mode
        swe.set_sid_mode(swe.SIDM_LAHIRI)

        # Get planet positions
        positions = {}
        planets = {
            "Sun": swe.SUN,
            "Moon": swe.MOON,
            "Mars": swe.MARS,
            "Mercury": swe.MERCURY,
            "Jupiter": swe.JUPITER,
            "Venus": swe.VENUS,
            "Saturn": swe.SATURN,
        }

        for planet, planet_id in planets.items():
            pos = swe.calc_ut(jd, planet_id, swe.FLG_SIDEREAL)[0]
            positions[planet] = pos[0]

        active_yogas = []

        # Check for Gaja-Kesari Yoga (Moon-Jupiter angle)
        moon_jupiter_angle = abs(positions["Moon"] - positions["Jupiter"])
        if moon_jupiter_angle > 180:
            moon_jupiter_angle = 360 - moon_jupiter_angle
        if moon_jupiter_angle in [60, 90, 120]:
            active_yogas.append(
                {
                    "name": "Gaja-Kesari",
                    "strength": calculate_yoga_strength(
                        moon_jupiter_angle, [60, 90, 120]
                    ),
                }
            )

        # Check for Budha-Aditya Yoga (Sun-Mercury conjunction)
        sun_mercury_angle = abs(positions["Sun"] - positions["Mercury"])
        if sun_mercury_angle <= 15:
            active_yogas.append(
                {
                    "name": "Budha-Aditya",
                    "strength": calculate_yoga_strength(
                        sun_mercury_angle, [0], max_orb=15
                    ),
                }
            )

        # Check for Dhana Yoga (benefics in angles)
        benefics = ["Jupiter", "Venus", "Mercury"]
        angles = [0, 90, 180, 270]  # House 1, 4, 7, 10
        for planet in benefics:
            planet_house = int(positions[planet] / 30)
            house_cusp = planet_house * 30
            if house_cusp in angles:
                active_yogas.append({"name": f"Dhana-{planet}", "strength": 0.8})

        # Check for Raja Yoga (lords of trine and angle houses together)
        if abs(positions["Jupiter"] - positions["Moon"]) <= 15:
            active_yogas.append(
                {
                    "name": "Raja",
                    "strength": calculate_yoga_strength(
                        abs(positions["Jupiter"] - positions["Moon"]), [0], max_orb=15
                    ),
                }
            )

        return active_yogas

    except Exception as e:
        logger.error(f"Error calculating victory yogas: {e}")
        return []


def calculate_team_yoga(
    jd: float, lat: float, lon: float, team_planets: Dict[str, str]
) -> float:
    """
    Calculate yoga strength specific to a team based on their ruling planets.

    Args:
        jd: Julian day number
        lat: Latitude
        lon: Longitude
        team_planets: Dictionary mapping team attributes to ruling planets

    Returns:
        float: Team yoga strength between 0 and 1
    """
    try:
        # Set sidereal mode
        swe.set_sid_mode(swe.SIDM_LAHIRI)

        # Get positions of team's ruling planets
        positions = {}
        planet_map = {
            "sun": swe.SUN,
            "moon": swe.MOON,
            "mars": swe.MARS,
            "mercury": swe.MERCURY,
            "jupiter": swe.JUPITER,
            "venus": swe.VENUS,
            "saturn": swe.SATURN,
        }

        for attribute, planet in team_planets.items():
            if planet.lower() in planet_map:
                pos = swe.calc_ut(jd, planet_map[planet.lower()], swe.FLG_SIDEREAL)[0]
                positions[attribute] = {"planet": planet, "longitude": pos[0]}

        strength = 0.0
        count = 0

        # Check for favorable angles between team planets
        for attr1, pos1 in positions.items():
            for attr2, pos2 in positions.items():
                if attr1 < attr2:  # Avoid duplicate checks
                    angle = abs(pos1["longitude"] - pos2["longitude"])
                    if angle > 180:
                        angle = 360 - angle

                    # Favorable angles
                    if angle <= 15:  # Conjunction
                        strength += 1.0
                    elif abs(angle - 60) <= 10:  # Sextile
                        strength += 0.6
                    elif abs(angle - 120) <= 10:  # Trine
                        strength += 0.8
                    count += 1

        # Check planets in favorable houses
        for pos in positions.values():
            house = int(pos["longitude"] / 30) + 1
            if house in [1, 4, 7, 10]:  # Angular houses
                strength += 0.8
            elif house in [5, 9]:  # Trines
                strength += 0.6
            count += 1

        if count > 0:
            return min(strength / count, 1.0)
        else:
            return 0.5

    except Exception as e:
        logger.error(f"Error calculating team yoga: {e}")
        return 0.5


def calculate_yoga_strength(
    angle: float, favorable_angles: List[float], max_orb: float = 10
) -> float:
    """
    Calculate the strength of a yoga based on planetary angles.

    Args:
        angle: The angle between planets
        favorable_angles: List of ideal angles for the yoga
        max_orb: Maximum orb (deviation from ideal angle) allowed

    Returns:
        float: Yoga strength between 0 and 1
    """
    try:
        min_diff = min(abs(angle - fav) for fav in favorable_angles)
        if min_diff <= max_orb:
            return 1.0 - (min_diff / max_orb) * 0.5
        return 0.0
    except Exception as e:
        logger.error(f"Error calculating yoga strength: {e}")
        return 0.0
