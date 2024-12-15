"""
Module for calculating various divisional charts (D-1 through D-12).
"""

import logging
from typing import Dict, List, Tuple, Any
import math

logger = logging.getLogger(__name__)

# Constants for divisional chart calculations
CHART_DIVISIONS = {
    "D1": 1,  # Rashi (main chart)
    "D2": 2,  # Hora
    "D3": 3,  # Drekkana
    "D4": 4,  # Chaturthamsha
    "D7": 7,  # Saptamsha
    "D9": 9,  # Navamsha
    "D10": 10,  # Dashamsha
    "D12": 12,  # Dwadashamsha
}


def calculate_division(longitude: float, division: int) -> Dict[str, Any]:
    """
    Calculate position in a divisional chart.

    Args:
        longitude: Longitude in degrees
        division: Number of divisions (2 for D2, 3 for D3, etc.)

    Returns:
        Dictionary containing divisional position details
    """
    sign_num = int(longitude / 30)
    pos_in_sign = longitude % 30

    division_size = 30 / division
    division_num = int(pos_in_sign / division_size)

    # Calculate final position
    final_sign = (sign_num * division + division_num) % 12
    final_degree = (pos_in_sign % division_size) * (30 / division_size)

    return {
        "sign": final_sign + 1,
        "degree": final_degree,
        "total_degree": final_sign * 30 + final_degree,
    }


def calculate_hora(longitude: float) -> Dict[str, Any]:
    """
    Calculate Hora (D-2) position.

    Args:
        longitude: Longitude in degrees

    Returns:
        Dictionary containing hora position details
    """
    sign_num = int(longitude / 30)
    degree_in_sign = longitude % 30

    # First half of sign goes to Sun, second to Moon
    is_sun_hora = degree_in_sign < 15

    # Even signs reverse the order
    if sign_num % 2 == 1:
        is_sun_hora = not is_sun_hora

    ruler = "Sun" if is_sun_hora else "Moon"
    strength = 1.0 if is_sun_hora else 0.5

    return {"ruler": ruler, "strength": strength, "degree": degree_in_sign * 2}


def calculate_drekkana(longitude: float) -> Dict[str, Any]:
    """
    Calculate Drekkana (D-3) position.

    Args:
        longitude: Longitude in degrees

    Returns:
        Dictionary containing drekkana position details
    """
    sign_num = int(longitude / 30)
    degree_in_sign = longitude % 30

    # Each drekkana is 10 degrees
    drekkana_num = int(degree_in_sign / 10)

    # Calculate final sign based on triplicity
    final_sign = (sign_num - (sign_num % 3) + drekkana_num) % 12

    return {
        "sign": final_sign + 1,
        "degree": (degree_in_sign % 10) * 3,
        "decanate": drekkana_num + 1,
    }


def calculate_chaturthamsha(longitude: float) -> Dict[str, Any]:
    """
    Calculate Chaturthamsha (D-4) position.

    Args:
        longitude: Longitude in degrees

    Returns:
        Dictionary containing chaturthamsha position details
    """
    return calculate_division(longitude, 4)


def calculate_saptamsha(longitude: float) -> Dict[str, Any]:
    """
    Calculate Saptamsha (D-7) position.

    Args:
        longitude: Longitude in degrees

    Returns:
        Dictionary containing saptamsha position details
    """
    return calculate_division(longitude, 7)


def calculate_dashamsha(longitude: float) -> Dict[str, Any]:
    """
    Calculate Dashamsha (D-10) position.

    Args:
        longitude: Longitude in degrees

    Returns:
        Dictionary containing dashamsha position details
    """
    return calculate_division(longitude, 10)


def calculate_dwadashamsha(longitude: float) -> Dict[str, Any]:
    """
    Calculate Dwadashamsha (D-12) position.

    Args:
        longitude: Longitude in degrees

    Returns:
        Dictionary containing dwadashamsha position details
    """
    return calculate_division(longitude, 12)


def calculate_all_divisions(
    positions: Dict[str, float]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Calculate all divisional charts for given positions.

    Args:
        positions: Dictionary of planetary positions

    Returns:
        Dictionary containing all divisional chart positions
    """
    divisional_positions = {}

    for planet, longitude in positions.items():
        divisional_positions[planet] = {
            "D1": {"sign": int(longitude / 30) + 1, "degree": longitude % 30},
            "D2": calculate_hora(longitude),
            "D3": calculate_drekkana(longitude),
            "D4": calculate_chaturthamsha(longitude),
            "D7": calculate_saptamsha(longitude),
            "D9": calculate_division(longitude, 9),
            "D10": calculate_dashamsha(longitude),
            "D12": calculate_dwadashamsha(longitude),
        }

    return divisional_positions


def analyze_divisional_strength(
    planet: str,
    div_positions: Dict[str, Dict[str, Any]],
    chart_weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Analyze planetary strength across divisional charts.

    Args:
        planet: Name of the planet
        div_positions: Divisional positions for the planet
        chart_weights: Optional weights for different charts

    Returns:
        Dictionary containing strength analysis
    """
    if chart_weights is None:
        chart_weights = {
            "D1": 1.0,
            "D2": 0.5,
            "D3": 0.75,
            "D4": 0.5,
            "D7": 0.75,
            "D9": 1.0,
            "D10": 0.75,
            "D12": 0.5,
        }

    total_strength = 0
    max_possible = sum(chart_weights.values())

    for chart, weight in chart_weights.items():
        if chart in div_positions:
            pos = div_positions[chart]

            # Calculate basic positional strength
            if isinstance(pos, dict) and "sign" in pos:
                sign_num = pos["sign"]

                # Exaltation and debilitation points
                if planet == "Sun" and sign_num == 1:  # Aries
                    total_strength += weight
                elif planet == "Moon" and sign_num == 2:  # Taurus
                    total_strength += weight
                elif planet == "Mars" and sign_num == 10:  # Capricorn
                    total_strength += weight
                elif planet == "Mercury" and sign_num == 6:  # Virgo
                    total_strength += weight
                elif planet == "Jupiter" and sign_num == 4:  # Cancer
                    total_strength += weight
                elif planet == "Venus" and sign_num == 12:  # Pisces
                    total_strength += weight
                elif planet == "Saturn" and sign_num == 7:  # Libra
                    total_strength += weight

    normalized_strength = total_strength / max_possible if max_possible > 0 else 0

    return {
        "total_strength": total_strength,
        "normalized_strength": normalized_strength,
        "chart_weights": chart_weights,
    }


def get_divisional_interpretation(
    div_analysis: Dict[str, float], threshold: float = 0.5
) -> str:
    """
    Generate interpretation of divisional chart analysis.

    Args:
        div_analysis: Divisional strength analysis
        threshold: Threshold for significance

    Returns:
        String containing interpretation
    """
    strength = div_analysis["normalized_strength"]

    if strength > threshold:
        return f"Strong divisional chart positions (strength: {strength:.2f})"
    elif strength < -threshold:
        return f"Weak divisional chart positions (strength: {strength:.2f})"
    else:
        return f"Moderate divisional chart positions (strength: {strength:.2f})"
