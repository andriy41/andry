"""Varga (divisional chart) calculations for NFL predictions."""

import math
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Divisional chart divisions
VARGA_DIVISIONS = {
    "D1": 1,  # Rashi (Main chart)
    "D2": 2,  # Hora (Wealth)
    "D3": 3,  # Drekkana (Siblings)
    "D4": 4,  # Chaturthamsha (Fortune)
    "D7": 7,  # Saptamsha (Children)
    "D9": 9,  # Navamsha (General strength)
    "D10": 10,  # Dashamsha (Career/Success)
    "D12": 12,  # Dwadashamsha (Parents)
    "D16": 16,  # Shodashamsha (Vehicles)
    "D20": 20,  # Vimshamsha (Spiritual strength)
    "D24": 24,  # Chaturvimshamsha (Learning)
    "D27": 27,  # Saptavimshamsha (Strength)
    "D30": 30,  # Trimshamsha (Evil/Obstacles)
    "D40": 40,  # Khavedamsha (Auspicious results)
    "D45": 45,  # Akshavedamsha (General indications)
    "D60": 60,  # Shashtiamsha (All areas)
}


def calculate_varga_position(longitude: float, division: int) -> float:
    """Calculate position in a divisional chart."""
    quotient = math.floor(longitude * division / 30)
    remainder = longitude * division % 30
    return quotient * 30 + remainder


def get_sign_lord(longitude: float) -> str:
    """Get the lord of the sign for a given longitude."""
    sign = math.floor(longitude / 30)
    sign_lords = [
        "Mars",
        "Venus",
        "Mercury",
        "Moon",
        "Sun",
        "Mercury",
        "Venus",
        "Mars",
        "Jupiter",
        "Saturn",
        "Saturn",
        "Jupiter",
    ]
    return sign_lords[sign]


def calculate_varga_strength(
    planet_positions: Dict[str, float], divisions: List[str] = None
) -> Dict[str, float]:
    """Calculate strength in divisional charts."""
    if divisions is None:
        divisions = ["D1", "D9", "D10"]  # Most relevant for victory

    strengths = {}

    for planet, longitude in planet_positions.items():
        planet_strength = 0
        total_weight = 0

        for division in divisions:
            # Get division number
            div_num = VARGA_DIVISIONS[division]

            # Calculate position in this division
            varga_pos = calculate_varga_position(longitude, div_num)

            # Get lord of the sign in this division
            varga_lord = get_sign_lord(varga_pos)

            # Calculate weight based on division importance
            if division == "D1":
                weight = 1.0
            elif division == "D9":
                weight = 0.8  # Navamsha is very important
            elif division == "D10":
                weight = 0.7  # Dashamsha shows success
            else:
                weight = 0.5

            # Calculate strength in this division
            div_strength = 0

            # Exaltation points
            if division == "D1":
                exaltation_points = {
                    "Sun": 10,  # Aries
                    "Moon": 33,  # Taurus
                    "Mars": 298,  # Capricorn
                    "Mercury": 165,  # Virgo
                    "Jupiter": 95,  # Cancer
                    "Venus": 357,  # Pisces
                    "Saturn": 200,  # Libra
                }
                if planet in exaltation_points:
                    distance = abs(varga_pos - exaltation_points[planet])
                    div_strength += max(0, 1 - distance / 180)

            # Own sign strength
            if varga_lord == planet:
                div_strength += 1.0

            # Friendly sign
            friendly_signs = {
                "Sun": ["Moon", "Mars", "Jupiter"],
                "Moon": ["Sun", "Mercury"],
                "Mars": ["Sun", "Moon", "Jupiter"],
                "Mercury": ["Sun", "Venus"],
                "Jupiter": ["Sun", "Moon", "Mars"],
                "Venus": ["Mercury", "Saturn"],
                "Saturn": ["Mercury", "Venus"],
            }
            if varga_lord in friendly_signs.get(planet, []):
                div_strength += 0.5

            planet_strength += div_strength * weight
            total_weight += weight

        # Normalize strength
        strengths[planet] = planet_strength / total_weight if total_weight > 0 else 0

    return strengths


def calculate_varga_aspects(
    planet_positions: Dict[str, float], divisions: List[str] = None
) -> Dict[str, float]:
    """Calculate aspect strength in divisional charts."""
    if divisions is None:
        divisions = ["D1", "D9", "D10"]

    aspect_strengths = {}

    for planet, longitude in planet_positions.items():
        total_aspect_strength = 0
        total_weight = 0

        for division in divisions:
            div_num = VARGA_DIVISIONS[division]
            varga_pos = calculate_varga_position(longitude, div_num)

            # Calculate aspects in this division
            aspects = []
            if planet in ["Mars", "Jupiter", "Saturn"]:
                # Special aspects for outer planets
                if planet == "Mars":
                    aspects = [4, 7, 8]  # 4th, 7th, 8th
                elif planet == "Jupiter":
                    aspects = [5, 7, 9]  # 5th, 7th, 9th
                elif planet == "Saturn":
                    aspects = [3, 7, 10]  # 3rd, 7th, 10th
            else:
                aspects = [7]  # 7th aspect for all planets

            # Check each aspect
            aspect_strength = 0
            for aspect in aspects:
                aspect_pos = (varga_pos + aspect * 30) % 360
                for other_pos in planet_positions.values():
                    other_varga_pos = calculate_varga_position(other_pos, div_num)
                    if abs(aspect_pos - other_varga_pos) < 10:  # Within orb
                        aspect_strength += 1

            # Weight by division importance
            weight = 1.0 if division == "D1" else 0.7 if division == "D9" else 0.5
            total_aspect_strength += aspect_strength * weight
            total_weight += weight

        # Normalize aspect strength
        aspect_strengths[planet] = (
            total_aspect_strength / total_weight if total_weight > 0 else 0
        )

    return aspect_strengths


def calculate_team_varga_strength(
    planet_positions: Dict[str, float], team_planets: List[str]
) -> float:
    """Calculate overall team strength based on varga positions."""
    try:
        # Calculate varga strengths for relevant planets
        varga_strengths = calculate_varga_strength(planet_positions)

        # Calculate aspect strengths
        aspect_strengths = calculate_varga_aspects(planet_positions)

        # Calculate team strength
        team_strength = 0
        total_planets = len(team_planets)

        for planet in team_planets:
            if planet in varga_strengths:
                # Combine position and aspect strength
                planet_strength = (
                    varga_strengths[planet] * 0.7
                    + aspect_strengths.get(planet, 0) * 0.3
                )
                team_strength += planet_strength

        # Normalize team strength
        return team_strength / total_planets if total_planets > 0 else 0

    except Exception as e:
        logger.error(f"Error calculating team varga strength: {e}")
        return 0.0
