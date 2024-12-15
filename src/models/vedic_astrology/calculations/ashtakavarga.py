"""
Module for calculating Ashtakavarga (eight-fold strength) of planets.
"""

import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Benefic points for each planet from different houses
BENEFIC_POINTS = {
    "Sun": {
        "Sun": [1, 2, 4, 7, 8, 9, 10, 11],
        "Moon": [3, 6, 10, 11],
        "Mars": [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [3, 5, 6, 9, 10, 11, 12],
        "Jupiter": [1, 2, 3, 4, 7, 8, 9, 10, 11],
        "Venus": [3, 4, 5, 8, 9, 10, 11],
        "Saturn": [1, 2, 4, 7, 8, 9, 10, 11],
    },
    "Moon": {
        "Sun": [3, 6, 7, 8, 10, 11],
        "Moon": [1, 3, 6, 7, 10, 11],
        "Mars": [2, 3, 5, 6, 9, 10, 11],
        "Mercury": [1, 3, 4, 5, 7, 8, 10, 11],
        "Jupiter": [1, 3, 4, 5, 7, 9, 10, 11],
        "Venus": [3, 4, 5, 7, 9, 10, 11],
        "Saturn": [3, 5, 6, 9, 10, 11],
    },
    "Mars": {
        "Sun": [3, 5, 6, 10, 11],
        "Moon": [3, 6, 11],
        "Mars": [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [3, 5, 6, 11],
        "Jupiter": [6, 10, 11, 12],
        "Venus": [6, 8, 11, 12],
        "Saturn": [1, 4, 7, 8, 9, 10, 11],
    },
    "Mercury": {
        "Sun": [3, 5, 6, 10, 11],
        "Moon": [2, 4, 6, 8, 10, 11],
        "Mars": [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [1, 3, 5, 6, 9, 10, 11],
        "Jupiter": [3, 5, 6, 9, 11, 12],
        "Venus": [1, 2, 3, 4, 5, 8, 9, 11],
        "Saturn": [3, 5, 6, 9, 10, 11],
    },
    "Jupiter": {
        "Sun": [1, 2, 3, 4, 7, 8, 9, 10, 11],
        "Moon": [2, 5, 7, 9, 11],
        "Mars": [1, 2, 4, 7, 8, 9, 10, 11],
        "Mercury": [1, 2, 4, 5, 6, 9, 10, 11],
        "Jupiter": [1, 2, 3, 4, 7, 8, 9, 10, 11],
        "Venus": [2, 5, 6, 9, 10, 11],
        "Saturn": [3, 5, 6, 11, 12],
    },
    "Venus": {
        "Sun": [3, 4, 5, 8, 9, 10, 11],
        "Moon": [3, 4, 5, 7, 9, 10, 11],
        "Mars": [3, 4, 5, 8, 9, 10, 11],
        "Mercury": [1, 2, 3, 4, 5, 8, 9, 11],
        "Jupiter": [2, 5, 6, 9, 10, 11],
        "Venus": [1, 2, 3, 4, 5, 8, 9, 10, 11],
        "Saturn": [3, 4, 5, 8, 9, 10, 11],
    },
    "Saturn": {
        "Sun": [1, 2, 4, 7, 8, 9, 10, 11],
        "Moon": [3, 5, 6, 9, 10, 11],
        "Mars": [1, 4, 7, 8, 9, 10, 11],
        "Mercury": [3, 5, 6, 9, 10, 11],
        "Jupiter": [3, 5, 6, 11, 12],
        "Venus": [3, 4, 5, 8, 9, 10, 11],
        "Saturn": [3, 5, 6, 10, 11, 12],
    },
}


def normalize_bindus(bindus: Dict[int, int]) -> Dict[int, int]:
    """
    Normalize bindu values to be between 0 and 8.

    Args:
        bindus: Dictionary of house numbers to bindu values

    Returns:
        Dictionary with normalized bindu values
    """
    max_bindu = max(bindus.values())
    if max_bindu > 8:
        normalized = {}
        for house, value in bindus.items():
            normalized[house] = round((value / max_bindu) * 8)
        return normalized
    return bindus


def calculate_ashtakavarga(
    planet: str, positions: Dict[str, Dict[str, float]], houses: Dict[str, int]
) -> Dict[str, Any]:
    """
    Calculate Ashtakavarga points for a planet.

    Args:
        planet: Name of the planet
        positions: Dictionary of planetary positions
        houses: Dictionary of house positions

    Returns:
        Dictionary containing bindus and analysis
    """
    if not isinstance(positions, dict) or not isinstance(houses, dict):
        raise ValueError("Positions and houses must be dictionaries")

    if planet not in BENEFIC_POINTS:
        raise ValueError(f"Invalid planet: {planet}")

    bindus = {i + 1: 0 for i in range(12)}  # Initialize all houses with 0

    # Calculate points from each planet's position
    for contributor, points in BENEFIC_POINTS.get(planet, {}).items():
        if contributor in positions:
            house = houses.get(contributor)
            if house:
                for benefic_house in points:
                    target_house = ((house + benefic_house - 1) % 12) + 1
                    bindus[target_house] += 1

    # Normalize bindu values
    bindus = normalize_bindus(bindus)

    # Calculate total and average strength
    total_bindus = sum(bindus.values())
    avg_bindus = total_bindus / 12

    # Identify strong and weak houses
    strong_houses = [h for h, b in bindus.items() if b > avg_bindus]
    weak_houses = [h for h, b in bindus.items() if b < avg_bindus]

    return {
        "bindus": bindus,
        "total": total_bindus,
        "average": avg_bindus,
        "strong_houses": strong_houses,
        "weak_houses": weak_houses,
    }


def calculate_sarvashtakavarga(
    positions: Dict[str, Dict[str, float]], houses: Dict[str, int]
) -> Dict[str, Any]:
    """
    Calculate Sarvashtakavarga (combined strength of all planets).

    Args:
        positions: Dictionary of planetary positions
        houses: Dictionary of house positions

    Returns:
        Dictionary containing combined analysis
    """
    if not isinstance(positions, dict) or not isinstance(houses, dict):
        raise ValueError("Positions and houses must be dictionaries")

    combined_bindus = {i + 1: 0 for i in range(12)}
    planet_scores = {}

    # Calculate Ashtakavarga for each planet
    for planet in ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]:
        result = calculate_ashtakavarga(planet, positions, houses)
        planet_scores[planet] = result

        # Add to combined bindus
        for house, bindus in result["bindus"].items():
            combined_bindus[house] += bindus

    # Normalize combined bindus
    combined_bindus = normalize_bindus(combined_bindus)

    # Calculate total and average
    total_bindus = sum(combined_bindus.values())
    avg_bindus = total_bindus / 12

    # Identify strong and weak houses
    strong_houses = [h for h, b in combined_bindus.items() if b > avg_bindus]
    weak_houses = [h for h, b in combined_bindus.items() if b < avg_bindus]

    return {
        "combined_bindus": combined_bindus,
        "total": total_bindus,
        "average": avg_bindus,
        "strong_houses": strong_houses,
        "weak_houses": weak_houses,
        "planet_scores": planet_scores,
    }


def analyze_ashtakavarga_strength(
    sarva_data: Dict[str, Any], team_houses: List[int]
) -> Dict[str, Any]:
    """
    Analyze Ashtakavarga strength for team houses.

    Args:
        sarva_data: Sarvashtakavarga calculation results
        team_houses: List of houses associated with the team

    Returns:
        Dictionary containing strength analysis
    """
    if not isinstance(sarva_data, dict) or not isinstance(team_houses, list):
        raise ValueError(
            "Sarva data must be a dictionary and team houses must be a list"
        )

    bindus = sarva_data["combined_bindus"]
    total_strength = 0
    house_strengths = {}

    for house in team_houses:
        bindu_count = bindus.get(house, 0)
        relative_strength = (bindu_count - sarva_data["average"]) / sarva_data[
            "average"
        ]
        house_strengths[house] = relative_strength
        total_strength += relative_strength

    avg_strength = total_strength / len(team_houses)

    return {
        "total_strength": total_strength,
        "average_strength": avg_strength,
        "house_strengths": house_strengths,
        "is_favorable": avg_strength > 0,
    }


def get_ashtakavarga_prediction(
    analysis: Dict[str, Any], threshold: float = 0.1
) -> str:
    """
    Generate prediction based on Ashtakavarga analysis.

    Args:
        analysis: Ashtakavarga strength analysis
        threshold: Threshold for significance

    Returns:
        String containing prediction
    """
    if not isinstance(analysis, dict):
        raise ValueError("Analysis must be a dictionary")

    avg_strength = analysis["average_strength"]

    if abs(avg_strength) < threshold:
        return "Neutral influence from Ashtakavarga"
    elif avg_strength > threshold:
        return f"Favorable Ashtakavarga influence (strength: {avg_strength:.2f})"
    else:
        return f"Challenging Ashtakavarga influence (strength: {avg_strength:.2f})"


def calculate_kaksha_bala(
    planet: str, longitude: float, ashtakavarga_data: Dict[str, Any]
) -> float:
    """
    Calculate Kaksha Bala (positional strength) using Ashtakavarga.

    Args:
        planet: Name of the planet
        longitude: Longitude of the planet
        ashtakavarga_data: Ashtakavarga calculation results

    Returns:
        Float indicating Kaksha Bala strength
    """
    if not isinstance(ashtakavarga_data, dict):
        raise ValueError("Ashtakavarga data must be a dictionary")

    house = int(longitude / 30) + 1
    bindus = ashtakavarga_data["bindus"]

    # Calculate position within house (0 to 1)
    house_position = (longitude % 30) / 30

    # Get bindu values for current and next house
    current_bindus = bindus.get(house, 0)
    next_house = (house % 12) + 1
    next_bindus = bindus.get(next_house, 0)

    # Interpolate strength based on position
    strength = current_bindus * (1 - house_position) + next_bindus * house_position

    # Normalize to 0-1 range
    max_possible = max(bindus.values())
    normalized_strength = strength / max_possible if max_possible > 0 else 0

    return normalized_strength
