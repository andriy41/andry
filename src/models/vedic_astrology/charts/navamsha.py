"""
Module for calculating and analyzing Navamsha (D9) positions and their influence on match outcomes.
"""

import logging
from typing import Dict, List, Tuple, Any
import math

logger = logging.getLogger(__name__)

# Constants for planetary effects in D9
D9_PLANET_EFFECTS = {
    "Moon": {"effect": -1.0, "reason": "Lazy influence, can be unstable"},
    "Mars": {"effect": -2.0, "reason": "Gives frustration, anger, and self-undoing"},
    "Rahu": {"effect": 1.0, "reason": "Gives ambition to win"},
    "Jupiter": {
        "effect": 2.0,
        "reason": "Gives luck and well-rested, positive attitude",
    },
    "Saturn": {"effect": -1.5, "reason": "Restricts; team feels old, tired, weak"},
    "Mercury": {"effect": 1.5, "reason": "Gives skill and speed"},
    "Ketu": {"effect": -1.5, "reason": "Gives confusion leading to defeat"},
    "Venus": {"effect": -0.75, "reason": "Gives laziness and laissez-faire attitude"},
    "Sun": {"effect": -1.0, "reason": "Gives cautious, conservative attitude"},
    "Uranus": {"effect": 0, "reason": "Variable based on motion"},
    "Neptune": {"effect": 0, "reason": "Variable based on motion"},
    "Pluto": {"effect": -1.25, "reason": "Generally negative, conferring heaviness"},
    "Upaketu": {"effect": -1.5, "reason": "Acts like Ketu to spoil luck"},
    "Gulika": {"effect": -1.0, "reason": "Negative influence on cusps"},
}


def calculate_navamsha_degree(longitude: float) -> float:
    """
    Calculate the exact degree in the navamsha chart.

    Args:
        longitude: Longitude in degrees

    Returns:
        float: Navamsha degree
    """
    # Each navamsha is 3°20' (200 minutes of arc)
    navamsha_span = 200  # in minutes

    # Convert longitude to minutes within its navamsha
    total_minutes = longitude * 60  # Convert degrees to minutes
    navamsha_position = total_minutes % navamsha_span

    # Convert back to degrees using the 6.67 constant
    navamsha_degree = navamsha_position / 6.67

    return navamsha_degree


def calculate_navamsha(longitude: float) -> Dict[str, Any]:
    """
    Calculate the navamsha position for a given longitude.

    Args:
        longitude: Longitude in degrees

    Returns:
        Dictionary containing:
        - sign: Zodiac sign in navamsha
        - degree: Exact degree in navamsha
        - house: House position in navamsha
    """
    # Calculate basic navamsha position
    nav_degree = calculate_navamsha_degree(longitude)

    # Calculate sign (each sign is 30 degrees)
    sign_num = int(longitude / 30)
    nav_sign_num = (sign_num * 9 + int(nav_degree / 3.333)) % 12

    # Get house position (1-12)
    house = nav_sign_num + 1

    # Map sign number to zodiac sign
    zodiac_signs = [
        "Aries",
        "Taurus",
        "Gemini",
        "Cancer",
        "Leo",
        "Virgo",
        "Libra",
        "Scorpio",
        "Sagittarius",
        "Capricorn",
        "Aquarius",
        "Pisces",
    ]
    sign = zodiac_signs[nav_sign_num]

    return {"sign": sign, "degree": nav_degree, "house": house}


def calculate_d9_positions(
    positions: Dict[str, Dict[str, float]], ascendant: float
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate positions in the D9 chart.

    Args:
        positions: Dictionary of planetary positions
        ascendant: Ascendant degree

    Returns:
        Dictionary containing D9 positions and aspects
    """
    d9_positions = {}
    d9_lagna = calculate_navamsha_degree(ascendant)
    d9_seventh = (d9_lagna + 180) % 360

    # Calculate D9 positions for all planets
    for planet, data in positions.items():
        longitude = data["longitude"]
        speed = data.get("speed", 0)

        d9_degree = calculate_navamsha_degree(longitude)

        # Check if planet is within orb of lagna or 7th
        orb = 2.5 if planet not in ["Uranus", "Neptune", "Pluto"] else 2.0

        # Check aspects to lagna and 7th
        lagna_diff = abs(d9_degree - d9_lagna)
        seventh_diff = abs(d9_degree - d9_seventh)

        # Adjust for circle wrap-around
        if lagna_diff > 180:
            lagna_diff = 360 - lagna_diff
        if seventh_diff > 180:
            seventh_diff = 360 - seventh_diff

        # Store position and aspect information
        d9_positions[planet] = {
            "degree": d9_degree,
            "aspects": {
                "lagna": lagna_diff <= orb,
                "seventh": seventh_diff <= orb,
                "lagna_diff": lagna_diff,
                "seventh_diff": seventh_diff,
            },
            "speed": speed,
        }

    return d9_positions


def analyze_d9_influences(
    d9_positions: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, float], List[str]]:
    """
    Analyze planetary influences in the D9 chart.

    Args:
        d9_positions: Dictionary of D9 positions and aspects

    Returns:
        Tuple containing:
        - Dictionary of influence scores for favorite and underdog
        - List of important findings
    """
    favorite_score = 0
    underdog_score = 0
    findings = []

    for planet, data in d9_positions.items():
        base_effect = D9_PLANET_EFFECTS[planet]["effect"]
        aspects = data["aspects"]
        speed = data.get("speed", 0)

        # Adjust effect for special cases
        if planet == "Uranus":
            base_effect = 1.5 if speed > 0 else -1.5
        elif planet == "Neptune":
            base_effect = -1.0 if speed > 0 else 1.0

        # Calculate influence on favorite (lagna)
        if aspects["lagna"]:
            influence = base_effect
            favorite_score += influence
            findings.append(
                f"{planet} within {aspects['lagna_diff']:.1f}° of D9 lagna: {influence:+.1f}"
            )

        # Calculate influence on underdog (7th)
        if aspects["seventh"]:
            influence = base_effect
            underdog_score += influence
            findings.append(
                f"{planet} within {aspects['seventh_diff']:.1f}° of D9 7th: {influence:+.1f}"
            )

    return {"favorite": favorite_score, "underdog": underdog_score}, findings


def get_d9_interpretation(scores: Dict[str, float], findings: List[str]) -> str:
    """
    Generate an interpretation of the D9 analysis.

    Args:
        scores: Dictionary of scores for favorite and underdog
        findings: List of significant findings

    Returns:
        String containing interpretation
    """
    interpretation = []

    # Add overall assessment
    if abs(scores["favorite"] - scores["underdog"]) < 0.5:
        interpretation.append("The D9 chart shows balanced influences on both teams.")
    elif scores["favorite"] > scores["underdog"]:
        diff = scores["favorite"] - scores["underdog"]
        interpretation.append(f"The D9 chart favors the favorite by {diff:.1f} points.")
    else:
        diff = scores["underdog"] - scores["favorite"]
        interpretation.append(f"The D9 chart favors the underdog by {diff:.1f} points.")

    # Add key findings
    if findings:
        interpretation.append("\nKey influences:")
        interpretation.extend([f"- {finding}" for finding in findings])

    return "\n".join(interpretation)
