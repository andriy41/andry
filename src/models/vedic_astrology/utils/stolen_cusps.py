from typing import Dict, Any
import logging


def determine_stolen_cusps(
    cusps: Dict[int, float], ascmc: Dict[str, float]
) -> Dict[int, Dict[str, Any]]:
    """
    Determines which house cusps are 'stolen' when comparing Placidus to Whole Sign houses.

    Args:
        cusps: Dictionary of house cusps with house numbers as keys and longitudes as values
        ascmc: Dictionary containing ascendant and midheaven positions

    Returns:
        Dictionary of stolen cusps with details about the type of theft and affected houses
    """
    asc_longitude = ascmc.get("ascendant", 0.0)
    asc_sign = int(asc_longitude / 30) + 1
    stolen_cusps = {}

    # Define house categories
    power_houses = {1, 4, 7, 10}  # Angular houses
    neutral_houses = {3, 5, 9, 11}  # Succedent and cadent houses

    for house in range(1, 13):
        cusp_sign = int(cusps[house] / 30) + 1
        whole_sign_house = ((cusp_sign - asc_sign) % 12) + 1

        if house != whole_sign_house:
            if house in power_houses and whole_sign_house in neutral_houses:
                stolen_type = "power-to-neutral"
                impact = -1  # Negative impact on prediction
            elif house in neutral_houses and whole_sign_house in power_houses:
                stolen_type = "neutral-to-power"
                impact = 1  # Positive impact on prediction
            elif house in power_houses and whole_sign_house in power_houses:
                stolen_type = "power-to-power"
                impact = 0  # Neutral impact
            else:
                stolen_type = "neutral-to-neutral"
                impact = 0  # Neutral impact

            stolen_cusps[house] = {
                "placidus_house": house,
                "whole_sign_house": whole_sign_house,
                "type": stolen_type,
                "impact_factor": impact,
                "cusp_longitude": cusps[house],
                "sign_longitude": (cusp_sign - 1) * 30,
            }

            logging.debug(
                f"Stolen cusp detected - House {house}: {stolen_cusps[house]}"
            )

    return stolen_cusps


def calculate_stolen_cusp_impact(
    stolen_cusps: Dict[int, Dict[str, Any]], planet_houses: Dict[str, int]
) -> float:
    """
    Calculates the overall impact of stolen cusps on prediction accuracy.

    Args:
        stolen_cusps: Dictionary of stolen cusps from determine_stolen_cusps()
        planet_houses: Dictionary of planet positions by house

    Returns:
        Float representing the overall impact (-1 to 1) of stolen cusps
    """
    total_impact = 0.0
    impact_weights = {
        "power-to-neutral": -0.3,
        "neutral-to-power": 0.3,
        "power-to-power": 0.1,
        "neutral-to-neutral": 0.0,
    }

    for house_data in stolen_cusps.values():
        base_impact = impact_weights[house_data["type"]]

        # Check if any planets are in the affected houses
        planets_involved = sum(
            1
            for house in planet_houses.values()
            if house in [house_data["placidus_house"], house_data["whole_sign_house"]]
        )

        # Adjust impact based on planetary presence
        if planets_involved > 0:
            base_impact *= 1 + (0.2 * planets_involved)

        total_impact += base_impact

    # Normalize the impact to be between -1 and 1
    normalized_impact = max(min(total_impact, 1.0), -1.0)

    logging.debug(f"Total stolen cusp impact: {normalized_impact}")
    return normalized_impact
