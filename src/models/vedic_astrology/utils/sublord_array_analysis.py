"""Module for analyzing sublord arrays in sports prediction."""

import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class SublordArrayAnalyzer:
    """Analyzer for sublord array patterns in sports prediction."""

    def __init__(self):
        """Initialize the analyzer with default house configurations."""
        self.favorite_houses = {1, 3, 6, 10, 11}
        self.underdog_houses = {4, 5, 7, 9, 12}
        self.neutral_houses = {2, 8}
        self.key_houses = {1, 7, 10}

    def analyze_sublord_array(
        self, house_sublords: Dict[int, Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze sublord array patterns for match prediction.

        A complete sublord array occurs when all sublords (except those in neutral houses)
        indicate victory for one side. The 2nd and 8th houses are considered neutral.

        Args:
            house_sublords: Dict mapping house numbers to their sublord information

        Returns:
            Dict containing:
                'complete_array': team benefiting from complete array ('favorite'/'underdog'/None)
                'points': points awarded based on array type
                'key_houses_array': team benefiting from key houses array (1,7,10)
                'array_details': Description of the array pattern found
                'strength_factor': Additional strength factor based on array quality
        """
        try:
            # Count sublords in victory houses
            favorite_count = 0
            underdog_count = 0
            total_strength = 0.0

            for house, sublord_info in house_sublords.items():
                if house not in self.neutral_houses:
                    sub_lord = sublord_info.get("sub_lord")
                    if not sub_lord:
                        continue

                    # Calculate house influence
                    if self._is_benefic(sub_lord):
                        if house in self.favorite_houses:
                            favorite_count += 1
                            total_strength += self._calculate_house_strength(
                                house, sublord_info
                            )
                    elif self._is_malefic(sub_lord):
                        if house in self.underdog_houses:
                            underdog_count += 1
                            total_strength += self._calculate_house_strength(
                                house, sublord_info
                            )

            # Check for complete array
            total_houses = len(house_sublords) - len(self.neutral_houses)
            complete_array = None
            array_points = 0
            strength_factor = 1.0

            if favorite_count == total_houses:
                complete_array = "favorite"
                array_points = 8  # Second-tier boost
                strength_factor = 1.5
            elif underdog_count == total_houses:
                complete_array = "underdog"
                array_points = 8  # Second-tier boost
                strength_factor = 1.5

            # Check key houses array (1,7,10)
            key_array = self._analyze_key_houses(house_sublords)

            if key_array and not complete_array:
                array_points = 3  # Third-tier boost
                strength_factor = 1.2

            # Generate description
            details = self._generate_array_description(
                complete_array, key_array, total_strength
            )

            return {
                "complete_array": complete_array,
                "key_houses_array": key_array,
                "points": array_points,
                "array_details": ". ".join(details),
                "strength_factor": strength_factor,
            }

        except Exception as e:
            logger.error(f"Error analyzing sublord array: {str(e)}")
            return {
                "complete_array": None,
                "key_houses_array": None,
                "points": 0,
                "array_details": "Error analyzing sublord array",
                "strength_factor": 1.0,
            }

    def _is_benefic(self, planet: str) -> bool:
        """Check if a planet is benefic."""
        benefics = {"Jupiter", "Venus", "Mercury", "Moon"}
        return planet in benefics

    def _is_malefic(self, planet: str) -> bool:
        """Check if a planet is malefic."""
        malefics = {"Mars", "Saturn", "Sun", "Rahu", "Ketu"}
        return planet in malefics

    def _calculate_house_strength(
        self, house: int, sublord_info: Dict[str, str]
    ) -> float:
        """Calculate the strength of a house based on sublord information."""
        try:
            base_strength = 1.0

            # Adjust strength based on house significance
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
            strength = base_strength * house_multipliers.get(house, 1.0)

            # Consider sublord nature
            sub_lord = sublord_info.get("sub_lord")
            if self._is_benefic(sub_lord):
                strength *= 1.2
            elif self._is_malefic(sub_lord):
                strength *= 0.8

            return strength

        except Exception as e:
            logger.error(f"Error calculating house strength: {str(e)}")
            return 1.0

    def _analyze_key_houses(
        self, house_sublords: Dict[int, Dict[str, str]]
    ) -> Optional[str]:
        """Analyze key houses (1,7,10) for array patterns."""
        try:
            key_favorite_count = 0
            key_underdog_count = 0

            for house in self.key_houses:
                if house not in house_sublords:
                    continue

                sublord_info = house_sublords[house]
                sub_lord = sublord_info.get("sub_lord")

                if not sub_lord:
                    continue

                if self._is_benefic(sub_lord) and house in self.favorite_houses:
                    key_favorite_count += 1
                elif self._is_malefic(sub_lord) and house in self.underdog_houses:
                    key_underdog_count += 1

            if key_favorite_count == len(self.key_houses):
                return "favorite"
            elif key_underdog_count == len(self.key_houses):
                return "underdog"

            return None

        except Exception as e:
            logger.error(f"Error analyzing key houses: {str(e)}")
            return None

    def _generate_array_description(
        self,
        complete_array: Optional[str],
        key_array: Optional[str],
        total_strength: float,
    ) -> List[str]:
        """Generate detailed description of the array pattern."""
        details = []

        if complete_array:
            details.append(f"Complete sublord array favoring {complete_array}")
            details.append(f"Array strength: {total_strength:.2f}")

        if key_array:
            details.append(f"Key houses (1,7,10) array favoring {key_array}")

        if not details:
            details.append("No significant sublord array pattern found")

        return details
