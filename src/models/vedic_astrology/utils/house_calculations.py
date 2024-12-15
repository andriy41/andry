"""House calculations for Vedic astrology."""

from typing import Dict, Any, List
import math


def calculate_house_strength(
    house_number: int,
    planet_positions: Dict[str, Dict[str, float]],
    aspects: Dict[str, List[Dict[str, Any]]],
) -> float:
    """
    Calculate the strength of a house based on planetary positions and aspects.

    Args:
        house_number: The house number (1-12)
        planet_positions: Dictionary of planetary positions
        aspects: Dictionary of planetary aspects

    Returns:
        float: Strength score for the house
    """
    strength = 0.0

    # Base house strengths
    base_strengths = {
        1: 1.0,  # Ascendant/Self
        2: 0.8,  # Wealth
        3: 0.7,  # Courage
        4: 0.9,  # Home
        5: 0.8,  # Creativity
        6: 0.6,  # Obstacles
        7: 1.0,  # Relationships
        8: 0.5,  # Transformation
        9: 0.9,  # Fortune
        10: 1.0,  # Career
        11: 0.8,  # Gains
        12: 0.6,  # Loss
    }

    # Add base house strength
    strength += base_strengths.get(house_number, 0.7)

    # Calculate house cusp
    house_cusp = (house_number - 1) * 30

    # Add strength for planets in the house
    for planet, pos in planet_positions.items():
        planet_long = pos["longitude"]
        if is_planet_in_house(planet_long, house_number):
            strength += get_planet_weight(planet)

    # Add strength for aspects to the house
    for planet, planet_aspects in aspects.items():
        for aspect in planet_aspects:
            if is_aspect_to_house(aspect, house_cusp):
                strength += get_aspect_weight(aspect["type"]) * get_planet_weight(
                    planet
                )

    return normalize_strength(strength)


def calculate_lord_strength(
    house_number: int,
    planet_positions: Dict[str, Dict[str, float]],
    aspects: Dict[str, List[Dict[str, Any]]],
) -> float:
    """
    Calculate the strength of a house lord based on its position and aspects.

    Args:
        house_number: The house number (1-12)
        planet_positions: Dictionary of planetary positions
        aspects: Dictionary of planetary aspects

    Returns:
        float: Strength score for the house lord
    """
    # House lord assignments (simplified)
    house_lords = {
        1: "mars",  # Aries
        2: "venus",  # Taurus
        3: "mercury",  # Gemini
        4: "moon",  # Cancer
        5: "sun",  # Leo
        6: "mercury",  # Virgo
        7: "venus",  # Libra
        8: "mars",  # Scorpio
        9: "jupiter",  # Sagittarius
        10: "saturn",  # Capricorn
        11: "saturn",  # Aquarius
        12: "jupiter",  # Pisces
    }

    lord = house_lords.get(house_number)
    if not lord or lord not in planet_positions:
        return 0.5  # Default strength if lord not found

    strength = 0.0
    lord_pos = planet_positions[lord]

    # Base strength from planet
    strength += get_planet_weight(lord)

    # Add strength based on house placement
    if lord_pos["longitude"] is not None:
        current_house = calculate_house_number(lord_pos["longitude"])
        strength += get_house_placement_strength(lord, current_house)

    # Add strength from aspects
    if lord in aspects:
        for aspect in aspects[lord]:
            strength += get_aspect_weight(aspect["type"])

    return normalize_strength(strength)


def is_planet_in_house(longitude: float, house_number: int) -> bool:
    """Check if a planet is in a specific house."""
    house_start = ((house_number - 1) * 30) % 360
    house_end = (house_number * 30) % 360

    if house_start < house_end:
        return house_start <= longitude < house_end
    else:  # House spans 0°
        return longitude >= house_start or longitude < house_end


def is_aspect_to_house(aspect: Dict[str, Any], house_cusp: float) -> bool:
    """Check if an aspect affects a house cusp."""
    orb = aspect.get("orb", 0)
    return orb <= get_aspect_orb(aspect["type"])


def get_planet_weight(planet: str) -> float:
    """Get the weight/importance of a planet."""
    weights = {
        "sun": 1.0,
        "moon": 1.0,
        "mars": 0.8,
        "mercury": 0.7,
        "jupiter": 0.9,
        "venus": 0.7,
        "saturn": 0.8,
    }
    return weights.get(planet.lower(), 0.5)


def get_aspect_weight(aspect_type: str) -> float:
    """Get the weight/importance of an aspect type."""
    weights = {
        "conjunction": 1.0,
        "opposition": 0.8,
        "trine": 0.7,
        "square": 0.6,
        "sextile": 0.5,
    }
    return weights.get(aspect_type, 0.3)


def get_aspect_orb(aspect_type: str) -> float:
    """Get the maximum orb for an aspect type."""
    orbs = {
        "conjunction": 10.0,
        "opposition": 10.0,
        "trine": 8.0,
        "square": 8.0,
        "sextile": 6.0,
    }
    return orbs.get(aspect_type, 5.0)


def normalize_strength(strength: float) -> float:
    """Normalize strength value to be between 0 and 1."""
    return max(0.0, min(1.0, strength / 3.0))


def calculate_house_number(longitude: float) -> int:
    """Calculate house number (1-12) from longitude."""
    house = math.floor(longitude / 30) + 1
    return ((house - 1) % 12) + 1


def get_house_placement_strength(lord: str, current_house: int) -> float:
    """Get strength modifier based on house placement."""
    # Houses that are good for planets (simplified)
    good_houses = {
        "sun": [1, 5, 9, 10],
        "moon": [2, 4, 7, 10],
        "mars": [1, 4, 7, 10],
        "mercury": [1, 4, 7, 10],
        "jupiter": [1, 5, 9, 11],
        "venus": [1, 2, 4, 5],
        "saturn": [3, 6, 11],
    }

    if current_house in good_houses.get(lord, []):
        return 0.3
    elif current_house in [6, 8, 12]:  # Dusthana houses
        return -0.2
    else:
        return 0.1
