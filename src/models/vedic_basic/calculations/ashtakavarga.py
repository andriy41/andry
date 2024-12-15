"""
Ashtakavarga calculations for NFL predictions
Adapted for sports prediction context
"""
from typing import Dict, Any, List
import math


class AshtakavargaCalculator:
    """Calculate Ashtakavarga points and interpret for NFL context"""

    def __init__(self):
        # Define benefic points for each planet (simplified for NFL context)
        self.benefic_points = {
            "sun": [1, 2, 4, 7, 8, 9, 10, 11],  # Leadership and vitality houses
            "moon": [1, 3, 6, 7, 10, 11],  # Public support and momentum
            "mars": [1, 2, 4, 7, 8, 9, 10, 11],  # Competition and aggression
            "mercury": [1, 3, 5, 6, 9, 10, 11, 12],  # Strategy and communication
            "jupiter": [1, 2, 3, 4, 7, 8, 9, 10, 11],  # Success and expansion
            "venus": [1, 2, 3, 4, 5, 8, 9, 11, 12],  # Team harmony
            "saturn": [1, 3, 4, 6, 8, 9, 10, 11],  # Discipline and defense
        }

        # NFL-specific house meanings
        self.house_significations = {
            1: ["team_strength", "overall_performance"],
            2: ["resources", "player_assets"],
            3: ["short_plays", "quick_drives"],
            4: ["home_field", "fan_base"],
            5: ["scoring_ability", "playmaking"],
            6: ["obstacles", "injuries"],
            7: ["competition", "opposition"],
            8: ["transformations", "comebacks"],
            9: ["fortune", "long_plays"],
            10: ["success", "victory"],
            11: ["gains", "points_scored"],
            12: ["hidden_factors", "secret_plays"],
        }

    def calculate_ashtakavarga(
        self, planet_positions: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculate Ashtakavarga points for all planets"""

        # Calculate individual planet points
        sarva_points = {}  # Combined points
        individual_points = {}  # Points for each planet

        for planet in self.benefic_points.keys():
            if planet in planet_positions:
                points = self._calculate_planet_points(
                    planet, planet_positions[planet]["longitude"], planet_positions
                )
                individual_points[planet] = points

                # Add to Sarva (total) points
                for house, value in points.items():
                    sarva_points[house] = sarva_points.get(house, 0) + value

        # Calculate strengths and interpretations
        return {
            "individual_points": individual_points,
            "sarva_points": sarva_points,
            "house_strengths": self._calculate_house_strengths(sarva_points),
            "game_factors": self._interpret_points_for_game(
                individual_points, sarva_points
            ),
        }

    def _calculate_planet_points(
        self,
        planet: str,
        planet_long: float,
        all_positions: Dict[str, Dict[str, float]],
    ) -> Dict[int, int]:
        """Calculate Ashtakavarga points for a single planet"""
        points = {house: 0 for house in range(1, 13)}

        # Get house occupied by planet
        planet_house = self._get_house_from_longitude(planet_long)

        # Add points for beneficial placements
        benefic_houses = self.benefic_points[planet]
        for house in range(1, 13):
            # Check if current house is benefic
            if house in benefic_houses:
                points[house] += 1

            # Add points for aspects
            for aspect_house in self._get_aspect_houses(house):
                if aspect_house in benefic_houses:
                    points[house] += 0.5  # Half point for aspects

        return points

    def _get_house_from_longitude(self, longitude: float) -> int:
        """Convert longitude to house number (1-12)"""
        house = int(longitude / 30) + 1
        return house if house <= 12 else house - 12

    def _get_aspect_houses(self, house: int) -> List[int]:
        """Get houses aspected by given house"""
        aspects = []

        # Trines (120°)
        trine1 = (house + 4) if (house + 4) <= 12 else (house + 4 - 12)
        trine2 = (house + 8) if (house + 8) <= 12 else (house + 8 - 12)
        aspects.extend([trine1, trine2])

        # Square (90°)
        square = (house + 3) if (house + 3) <= 12 else (house + 3 - 12)
        aspects.append(square)

        # Opposition (180°)
        opposition = (house + 6) if (house + 6) <= 12 else (house + 6 - 12)
        aspects.append(opposition)

        return aspects

    def _calculate_house_strengths(
        self, sarva_points: Dict[int, float]
    ) -> Dict[int, float]:
        """Calculate relative strength of each house"""
        max_points = max(sarva_points.values())
        return {house: points / max_points for house, points in sarva_points.items()}

    def _interpret_points_for_game(
        self,
        individual_points: Dict[str, Dict[int, int]],
        sarva_points: Dict[int, float],
    ) -> Dict[str, Any]:
        """Interpret Ashtakavarga points for game prediction"""

        # Calculate overall strength
        total_points = sum(sarva_points.values())
        max_possible = len(self.benefic_points) * 12  # Maximum possible points
        overall_strength = total_points / max_possible

        # Find strongest houses
        sorted_houses = sorted(sarva_points.items(), key=lambda x: x[1], reverse=True)
        strongest_houses = sorted_houses[:3]

        # Interpret key factors
        key_factors = []
        for house, points in strongest_houses:
            significations = self.house_significations.get(house, [])
            strength = points / max(sarva_points.values())
            key_factors.append(
                {
                    "house": house,
                    "points": points,
                    "strength": strength,
                    "significations": significations,
                }
            )

        # Check special combinations
        special_combinations = self._check_special_combinations(sarva_points)

        return {
            "overall_strength": overall_strength,
            "key_factors": key_factors,
            "special_combinations": special_combinations,
            "strongest_houses": [house for house, _ in strongest_houses],
        }

    def _check_special_combinations(
        self, sarva_points: Dict[int, float]
    ) -> List[Dict[str, Any]]:
        """Check for special Ashtakavarga combinations relevant to NFL games"""
        combinations = []

        # Strong 1st house (team strength)
        if sarva_points.get(1, 0) >= 28:
            combinations.append(
                {
                    "name": "Strong Team Presence",
                    "description": "High potential for dominant performance",
                    "strength": sarva_points[1] / 36,  # Normalize to 0-1
                }
            )

        # Strong 10th house (success)
        if sarva_points.get(10, 0) >= 28:
            combinations.append(
                {
                    "name": "Victory Potential",
                    "description": "High likelihood of success",
                    "strength": sarva_points[10] / 36,
                }
            )

        # Strong 11th house (gains/points)
        if sarva_points.get(11, 0) >= 28:
            combinations.append(
                {
                    "name": "Scoring Potential",
                    "description": "High potential for scoring points",
                    "strength": sarva_points[11] / 36,
                }
            )

        # Strong trine houses (1, 5, 9)
        trine_points = (
            sarva_points.get(1, 0) + sarva_points.get(5, 0) + sarva_points.get(9, 0)
        )
        if trine_points >= 75:
            combinations.append(
                {
                    "name": "Triple Strength",
                    "description": "Excellence in performance, scoring, and long plays",
                    "strength": trine_points / 108,  # Normalize to 0-1
                }
            )

        return combinations


def calculate_ashtakavarga(
    planet_positions: Dict[str, float], house_positions: Dict[int, float]
) -> float:
    """Calculate Ashtakavarga points for a given set of planetary positions."""
    calculator = AshtakavargaCalculator()
    return calculator.calculate_total_points(planet_positions, house_positions)
