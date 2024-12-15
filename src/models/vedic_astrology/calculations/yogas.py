"""
Module for calculating Vedic yogas (planetary combinations).
"""

from typing import Dict, List, Any
import swisseph as swe


def calculate_yogas(jd: float) -> List[str]:
    """
    Calculate Vedic yogas present at a given time.

    Args:
        jd: Julian day number

    Returns:
        List of yoga names that are active
    """
    # Set sidereal mode
    swe.set_sid_mode(swe.SIDM_LAHIRI)

    # Get planet positions
    sun_pos = swe.calc_ut(jd, swe.SUN, swe.FLG_SIDEREAL)[0]
    moon_pos = swe.calc_ut(jd, swe.MOON, swe.FLG_SIDEREAL)[0]
    mars_pos = swe.calc_ut(jd, swe.MARS, swe.FLG_SIDEREAL)[0]
    mercury_pos = swe.calc_ut(jd, swe.MERCURY, swe.FLG_SIDEREAL)[0]
    jupiter_pos = swe.calc_ut(jd, swe.JUPITER, swe.FLG_SIDEREAL)[0]
    venus_pos = swe.calc_ut(jd, swe.VENUS, swe.FLG_SIDEREAL)[0]
    saturn_pos = swe.calc_ut(jd, swe.SATURN, swe.FLG_SIDEREAL)[0]

    # Store positions
    positions = {
        "Sun": sun_pos[0],
        "Moon": moon_pos[0],
        "Mars": mars_pos[0],
        "Mercury": mercury_pos[0],
        "Jupiter": jupiter_pos[0],
        "Venus": venus_pos[0],
        "Saturn": saturn_pos[0],
    }

    active_yogas = []

    # Check for Gaja-Kesari Yoga (Moon-Jupiter angle)
    moon_jupiter_angle = abs(positions["Moon"] - positions["Jupiter"])
    if moon_jupiter_angle > 180:
        moon_jupiter_angle = 360 - moon_jupiter_angle
    if moon_jupiter_angle in [60, 90, 120]:
        active_yogas.append("Gaja-Kesari")

    # Check for Budha-Aditya Yoga (Sun-Mercury conjunction)
    sun_mercury_angle = abs(positions["Sun"] - positions["Mercury"])
    if sun_mercury_angle <= 15:
        active_yogas.append("Budha-Aditya")

    # Check for Dhana Yoga (benefics in angles)
    benefics = ["Jupiter", "Venus", "Mercury"]
    angles = [0, 90, 180, 270]  # House 1, 4, 7, 10
    for planet in benefics:
        planet_house = int(positions[planet] / 30)
        if planet_house * 30 in angles:
            active_yogas.append(f"Dhana-{planet}")

    # Check for Raja Yoga (lords of trine and angle houses together)
    if abs(positions["Jupiter"] - positions["Moon"]) <= 15:
        active_yogas.append("Raja")

    return active_yogas
