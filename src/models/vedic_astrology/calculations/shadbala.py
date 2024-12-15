"""
Shadbala Calculator for Vedic Astrology NFL Predictions

This module calculates the six sources of planetary strength (Shadbala) in Vedic astrology:
1. Sthanabala (Positional Strength)
2. Dikkabala (Directional Strength)
3. Kalabala (Temporal Strength)
4. Chestabala (Motional Strength)
5. Naisargikabala (Natural Strength)
6. Drigbala (Aspectual Strength)
"""

import math
import swisseph as swe
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


def calculate_shadbala(jd: float, lat: float, lon: float) -> Dict[str, float]:
    """Calculate Shadbala (six-fold strength) for all planets."""
    calculator = ShadbalaCalculator()
    strengths = calculator.calculate_all_strengths(jd, lat, lon)

    # Normalize strengths to [0,1] range
    max_strength = max(strengths.values())
    if max_strength > 0:
        return {
            planet: strength / max_strength for planet, strength in strengths.items()
        }
    return strengths


class ShadbalaCalculator:
    def __init__(self):
        # Define the natural strengths of planets
        self.natural_strengths = {
            "Sun": 60,
            "Moon": 51.43,
            "Mars": 42.86,
            "Mercury": 34.29,
            "Jupiter": 25.71,
            "Venus": 17.14,
            "Saturn": 8.57,
        }

        # Define directional strengths
        self.directional_strengths = {
            "Sun": {"E": 100, "S": 75, "W": 50, "N": 25},
            "Moon": {"N": 100, "E": 75, "S": 50, "W": 25},
            "Mars": {"S": 100, "W": 75, "N": 50, "E": 25},
            "Mercury": {"N": 100, "E": 75, "S": 50, "W": 25},
            "Jupiter": {"N": 100, "E": 75, "S": 50, "W": 25},
            "Venus": {"E": 100, "S": 75, "W": 50, "N": 25},
            "Saturn": {"W": 100, "S": 75, "E": 50, "N": 25},
        }

    def calculate_sthanabala(
        self, planet_pos: float, house_cusps: List[float]
    ) -> float:
        """
        Calculate positional strength (Sthanabala) of a planet

        Args:
            planet_pos: Longitude of the planet
            house_cusps: List of house cusp positions

        Returns:
            float: Sthanabala strength value
        """
        # Find which house the planet is in
        house_num = self._get_house_number(planet_pos, house_cusps)

        # Define strength values for different houses
        house_strengths = {
            1: 100,  # Lagna (Ascendant)
            4: 75,  # 4th house
            7: 50,  # 7th house
            10: 25,  # 10th house
        }

        return house_strengths.get(house_num, 0)

    def calculate_dikkabala(self, planet: str, planet_pos: float) -> float:
        """
        Calculate directional strength (Dikkabala) of a planet

        Args:
            planet: Name of the planet
            planet_pos: Longitude of the planet

        Returns:
            float: Dikkabala strength value
        """
        direction = self._get_direction(planet_pos)
        return self.directional_strengths[planet][direction]

    def calculate_kalabala(
        self, planet: str, jd: float, lat: float, lon: float
    ) -> float:
        """
        Calculate temporal strength (Kalabala) of a planet

        Args:
            planet: Name of the planet
            jd: Julian day
            lat: Latitude of location
            lon: Longitude of location

        Returns:
            float: Kalabala strength value
        """
        # Calculate day/night status
        is_day = self._is_daytime(jd, lat, lon)

        # Define day/night strength values
        day_planets = ["Sun", "Jupiter", "Mars"]
        night_planets = ["Moon", "Venus", "Saturn"]

        if planet in day_planets:
            return 100 if is_day else 50
        elif planet in night_planets:
            return 100 if not is_day else 50
        else:  # Mercury
            return 75  # Neutral strength

    def calculate_chestabala(self, planet: str, speed: float) -> float:
        """
        Calculate motional strength (Chestabala) of a planet

        Args:
            planet: Name of the planet
            speed: Daily motion of the planet

        Returns:
            float: Chestabala strength value
        """
        # Define average daily motions
        avg_speeds = {
            "Sun": 0.9833,
            "Moon": 13.1763,
            "Mars": 0.5242,
            "Mercury": 1.383,
            "Jupiter": 0.0831,
            "Venus": 1.2009,
            "Saturn": 0.0334,
        }

        # Calculate ratio of current speed to average speed
        ratio = abs(speed) / avg_speeds[planet]

        # Return strength based on ratio
        if ratio >= 1:
            return 100
        else:
            return ratio * 100

    def calculate_naisargikabala(self, planet: str) -> float:
        """
        Calculate natural strength (Naisargikabala) of a planet

        Args:
            planet: Name of the planet

        Returns:
            float: Naisargikabala strength value
        """
        return self.natural_strengths[planet]

    def calculate_drigbala(
        self, planet_pos: float, all_planet_positions: Dict[str, float]
    ) -> float:
        """
        Calculate aspectual strength (Drigbala) of a planet

        Args:
            planet_pos: Longitude of the planet
            all_planet_positions: Dictionary of all planet positions

        Returns:
            float: Drigbala strength value
        """
        strength = 0

        # Define beneficial and malefic aspects
        beneficial_aspects = [60, 120]  # sextile and trine
        malefic_aspects = [90, 180]  # square and opposition

        for other_pos in all_planet_positions.values():
            aspect = self._calculate_aspect(planet_pos, other_pos)

            if aspect in beneficial_aspects:
                strength += 10
            elif aspect in malefic_aspects:
                strength -= 5

        return max(0, min(100, strength + 50))  # Normalize to 0-100 range

    def calculate_total_shadbala(self, planet: str, planet_data: Dict) -> float:
        """
        Calculate total Shadbala strength for a planet

        Args:
            planet: Name of the planet
            planet_data: Dictionary containing all required planetary data

        Returns:
            float: Total Shadbala strength value
        """
        weights = {
            "sthanabala": 0.2,
            "dikkabala": 0.15,
            "kalabala": 0.2,
            "chestabala": 0.15,
            "naisargikabala": 0.15,
            "drigbala": 0.15,
        }

        total = (
            weights["sthanabala"]
            * self.calculate_sthanabala(
                planet_data["position"], planet_data["house_cusps"]
            )
            + weights["dikkabala"]
            * self.calculate_dikkabala(planet, planet_data["position"])
            + weights["kalabala"]
            * self.calculate_kalabala(
                planet, planet_data["jd"], planet_data["lat"], planet_data["lon"]
            )
            + weights["chestabala"]
            * self.calculate_chestabala(planet, planet_data["speed"])
            + weights["naisargikabala"] * self.calculate_naisargikabala(planet)
            + weights["drigbala"]
            * self.calculate_drigbala(
                planet_data["position"], planet_data["all_positions"]
            )
        )

        return total

    def calculate_all_strengths(
        self, jd: float, lat: float, lon: float
    ) -> Dict[str, float]:
        """Calculate all Shadbala strengths for all planets."""
        planets = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
        planet_map = {
            "Sun": swe.SUN,
            "Moon": swe.MOON,
            "Mars": swe.MARS,
            "Mercury": swe.MERCURY,
            "Jupiter": swe.JUPITER,
            "Venus": swe.VENUS,
            "Saturn": swe.SATURN,
        }

        planet_data = {}

        for planet in planets:
            planet_info = swe.calc_ut(jd, planet_map[planet])[0]
            planet_data[planet] = {
                "position": planet_info[0],
                "speed": planet_info[3],
                "jd": jd,
                "lat": lat,
                "lon": lon,
                "house_cusps": self._calculate_house_cusps(jd, lat, lon),
                "all_positions": {
                    p: swe.calc_ut(jd, planet_map[p])[0][0] for p in planets
                },
            }

        return {
            planet: self.calculate_total_shadbala(planet, data)
            for planet, data in planet_data.items()
        }

    def _get_house_number(self, planet_pos: float, house_cusps: List[float]) -> int:
        """Helper method to determine house number of a planet"""
        for i in range(len(house_cusps)):
            next_cusp = house_cusps[(i + 1) % 12]
            if (house_cusps[i] <= planet_pos < next_cusp) or (
                house_cusps[i] > next_cusp
                and (planet_pos >= house_cusps[i] or planet_pos < next_cusp)
            ):
                return i + 1
        return 1

    def _get_direction(self, longitude: float) -> str:
        """Helper method to determine cardinal direction based on longitude"""
        longitude = longitude % 360
        if 315 <= longitude < 45:
            return "N"
        elif 45 <= longitude < 135:
            return "E"
        elif 135 <= longitude < 225:
            return "S"
        else:
            return "W"

    def _is_daytime(self, jd: float, lat: float, lon: float) -> bool:
        """Helper method to determine if it's daytime"""
        sun_info = swe.calc_ut(jd, swe.SUN)[0]
        return sun_info[1] > 0  # If altitude is positive, it's daytime

    def _calculate_aspect(self, pos1: float, pos2: float) -> float:
        """Helper method to calculate aspect angle between two positions"""
        diff = abs(pos1 - pos2)
        return min(diff % 360, 360 - (diff % 360))

    def _calculate_house_cusps(self, jd: float, lat: float, lon: float) -> List[float]:
        """Helper method to calculate house cusps"""
        ascendant = swe.houses(jd, lat, lon)[0][0]
        house_cusps = [ascendant + i * 30 for i in range(12)]
        return house_cusps
