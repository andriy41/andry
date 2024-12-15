"""Sublord calculations for NFL predictions."""

import logging
from typing import Dict, List, Union, Tuple
import pandas as pd
import swisseph as swe

logger = logging.getLogger(__name__)

# Define the sublord table for all zodiac signs
SUBLORD_TABLE = {
    "Aries": [
        {
            "from_degree": 0.0,
            "to_degree": 0.7778,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Ketu",
        },
        {
            "from_degree": 0.7778,
            "to_degree": 2.5,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Venus",
        },
        {
            "from_degree": 2.5,
            "to_degree": 3.75,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Sun",
        },
        {
            "from_degree": 3.75,
            "to_degree": 5.8333,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Moon",
        },
        {
            "from_degree": 5.8333,
            "to_degree": 7.5,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Mars",
        },
        {
            "from_degree": 7.5,
            "to_degree": 9.1667,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Rahu",
        },
        {
            "from_degree": 9.1667,
            "to_degree": 11.6667,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Jupiter",
        },
        {
            "from_degree": 11.6667,
            "to_degree": 13.3333,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Saturn",
        },
        {
            "from_degree": 13.3333,
            "to_degree": 15.0,
            "sign_lord": "Mars",
            "star_lord": "Ketu",
            "sub_lord": "Mercury",
        },
    ],
    # Add other signs as needed
}


class SublordCalculator:
    """Calculator for sublord positions and strengths."""

    def __init__(self):
        """Initialize the calculator."""
        self.sublord_table = SUBLORD_TABLE
        self.zodiac_signs = list(SUBLORD_TABLE.keys())

    def get_sublord(self, sign: str, degree: float) -> Dict[str, str]:
        """
        Determine the sublord for a given degree in a specific sign.

        Args:
            sign: Zodiac sign
            degree: Degree within the sign (0-30)

        Returns:
            Dictionary containing sign lord, star lord, and sub lord
        """
        try:
            if sign not in self.sublord_table:
                raise ValueError(f"Invalid sign: {sign}")

            for entry in self.sublord_table[sign]:
                if entry["from_degree"] <= degree < entry["to_degree"]:
                    return {
                        "sign_lord": entry["sign_lord"],
                        "star_lord": entry["star_lord"],
                        "sub_lord": entry["sub_lord"],
                    }

            raise ValueError(f"Degree {degree} not found in sign {sign}")

        except Exception as e:
            logger.error(f"Error getting sublord for {sign} at {degree}°: {str(e)}")
            return {"error": str(e)}

    def calculate_sublord_strength(
        self,
        sublord_info: Dict[str, str],
        planet_positions: Dict[str, float],
        varga_strengths: Dict[str, float],
    ) -> float:
        """
        Calculate the strength of a sublord based on planetary positions and Varga strengths.

        Args:
            sublord_info: Dictionary containing sublord information
            planet_positions: Dictionary of planet positions
            varga_strengths: Dictionary of Varga strengths for planets

        Returns:
            Strength value between 0 and 1
        """
        try:
            strength = 0.0
            weight_total = 0.0

            # Weights for different lords
            weights = {"sign_lord": 0.4, "star_lord": 0.35, "sub_lord": 0.25}

            for lord_type, planet in sublord_info.items():
                if lord_type in weights and planet in planet_positions:
                    # Base strength from position
                    base_strength = 0.5  # Default strength

                    # Add Varga strength if available
                    if planet in varga_strengths:
                        base_strength = (base_strength + varga_strengths[planet]) / 2

                    strength += base_strength * weights[lord_type]
                    weight_total += weights[lord_type]

            return strength / weight_total if weight_total > 0 else 0.5

        except Exception as e:
            logger.error(f"Error calculating sublord strength: {str(e)}")
            return 0.5

    def calculate_house_sublord_strength(
        self,
        house_number: int,
        sublord_info: Dict[str, Union[str, float]],
        planet_positions: Dict[str, float],
        varga_strengths: Dict[str, float],
    ) -> float:
        """
        Calculate the strength of a house based on its sublord positions.

        Args:
            house_number: House number (1-12)
            sublord_info: Dictionary containing sublord information
            planet_positions: Dictionary of planet positions
            varga_strengths: Dictionary of Varga strengths for planets

        Returns:
            Strength value between 0 and 1
        """
        try:
            # Extract lords information
            lords_info = {
                "sign_lord": sublord_info.get("sign_lord"),
                "star_lord": sublord_info.get("star_lord"),
                "sub_lord": sublord_info.get("sub_lord"),
            }

            # Calculate basic sublord strength
            strength = self.calculate_sublord_strength(
                lords_info, planet_positions, varga_strengths
            )

            # Adjust strength based on house number
            house_multipliers = {
                1: 1.2,  # Angular house
                4: 1.1,  # Angular house
                7: 1.2,  # Angular house
                10: 1.2,  # Angular house
                2: 0.9,  # Succedent house
                5: 0.9,
                8: 0.8,
                11: 1.0,
                3: 0.8,  # Cadent house
                6: 0.7,
                9: 1.0,
                12: 0.7,
            }

            # Apply house multiplier
            final_strength = strength * house_multipliers.get(house_number, 1.0)

            return min(
                max(final_strength, 0.1), 1.0
            )  # Ensure result is between 0.1 and 1.0

        except Exception as e:
            logger.error(f"Error calculating house sublord strength: {str(e)}")
            return 0.5
