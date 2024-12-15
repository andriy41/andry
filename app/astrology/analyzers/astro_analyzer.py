"""
Enhanced astrological analysis for NFL games
Integrates vedic astrology principles with NFL-specific factors
"""
import swisseph as swe
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json


class NFLAstroAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("nfl_astro")
        self._initialize_ephemeris()

        # NFL-specific planetary weights and aspects
        self.nfl_planets = {
            swe.MARS: {
                "weight": 1.5,  # Strong influence on competition and physical performance
                "aspects": {0: 1.0, 120: 0.8, 60: 0.6},  # Trine and sextile aspects
            },
            swe.JUPITER: {
                "weight": 1.2,  # Success and expansion
                "aspects": {0: 1.0, 120: 0.7, 60: 0.5},
            },
            swe.SATURN: {
                "weight": 1.0,  # Discipline and defense
                "aspects": {0: 1.0, 120: 0.6, 60: 0.4},
            },
            swe.SUN: {
                "weight": 1.1,  # Leadership and vitality
                "aspects": {0: 1.0, 120: 0.7, 60: 0.5},
            },
            swe.MOON: {
                "weight": 0.8,  # Team morale and public support
                "aspects": {0: 1.0, 120: 0.6, 60: 0.4},
            },
        }

    def _initialize_ephemeris(self):
        """Initialize Swiss Ephemeris with proper path"""
        ephe_path = str(Path(__file__).parent.parent / "ephe")
        swe.set_ephe_path(ephe_path)

    def analyze_game_time(
        self, game_time: datetime, location: Tuple[float, float]
    ) -> Dict:
        """
        Analyze astrological factors for a specific game time and location

        Args:
            game_time: Kickoff time
            location: (latitude, longitude) of stadium

        Returns:
            Dictionary of astrological factors and their strengths
        """
        lat, lon = location

        # Set time for analysis
        utc_time = game_time.timestamp()

        # Calculate key positions
        ascendant = swe.houses(utc_time, lat, lon)[0]  # Get Ascendant degree

        planet_positions = {}
        strengths = {}

        # Get positions for relevant planets
        for planet in self.nfl_planets.keys():
            pos = swe.calc_ut(utc_time, planet)[0]  # Get longitude
            planet_positions[planet] = pos

            # Calculate strength based on position and aspects
            strength = self._calculate_planet_strength(
                planet, pos, ascendant, planet_positions
            )
            strengths[planet] = strength

        return {
            "planet_positions": planet_positions,
            "strengths": strengths,
            "total_strength": sum(strengths.values()),
            "ascendant": ascendant,
        }

    def _calculate_planet_strength(
        self,
        planet: int,
        position: float,
        ascendant: float,
        all_positions: Dict[int, float],
    ) -> float:
        """Calculate strength of a planet based on position and aspects"""
        base_strength = self.nfl_planets[planet]["weight"]

        # House position influence
        house_position = (position - ascendant) % 360
        house_number = int(house_position / 30) + 1

        # Modify strength based on house placement
        house_multipliers = {
            1: 1.2,  # Angular house - very strong
            4: 1.1,  # Angular house
            7: 1.2,  # Angular house
            10: 1.1,  # Angular house
            2: 0.9,  # Succedent house
            5: 0.9,
            8: 0.8,
            11: 0.9,
            3: 0.8,  # Cadent house
            6: 0.7,
            9: 0.8,
            12: 0.7,
        }

        base_strength *= house_multipliers[house_number]

        # Check aspects with other planets
        aspect_strength = 0
        for other_planet, other_pos in all_positions.items():
            if other_planet != planet:
                aspect_angle = abs((position - other_pos) % 360)

                # Check for major aspects
                for angle, multiplier in self.nfl_planets[planet]["aspects"].items():
                    if abs(aspect_angle - angle) <= 8:  # 8 degree orb
                        aspect_strength += multiplier

        return base_strength * (1 + aspect_strength * 0.2)  # Aspect modification

    def get_favorable_times(
        self, date: datetime, location: Tuple[float, float], window_hours: int = 8
    ) -> List[datetime]:
        """
        Find favorable game times within a given window

        Args:
            date: Base date to check
            location: Stadium coordinates
            window_hours: Hours to check before and after base time

        Returns:
            List of favorable times sorted by strength
        """
        favorable_times = []
        start_time = date - timedelta(hours=window_hours)

        # Check every 30 minutes
        for minutes in range(window_hours * 2 * 60 // 30):
            check_time = start_time + timedelta(minutes=minutes * 30)
            analysis = self.analyze_game_time(check_time, location)

            favorable_times.append(
                {"time": check_time, "strength": analysis["total_strength"]}
            )

        # Sort by strength and return top times
        favorable_times.sort(key=lambda x: x["strength"], reverse=True)
        return [t["time"] for t in favorable_times[:5]]

    def analyze_team_compatibility(
        self,
        home_location: Tuple[float, float],
        away_location: Tuple[float, float],
        game_time: datetime,
    ) -> float:
        """
        Analyze astrological compatibility between two team locations

        Args:
            home_location: Home team stadium coordinates
            away_location: Away team stadium coordinates
            game_time: Kickoff time

        Returns:
            Compatibility score (-1 to 1)
        """
        home_analysis = self.analyze_game_time(game_time, home_location)
        away_analysis = self.analyze_game_time(game_time, away_location)

        # Compare planetary positions
        compatibility = 0
        for planet in self.nfl_planets.keys():
            home_pos = home_analysis["planet_positions"][planet]
            away_pos = away_analysis["planet_positions"][planet]

            # Calculate angular difference
            diff = abs((home_pos - away_pos) % 360)

            # Favorable aspects increase compatibility
            if diff <= 8 or abs(diff - 120) <= 8 or abs(diff - 60) <= 8:
                compatibility += 0.2
            # Challenging aspects decrease compatibility
            elif abs(diff - 90) <= 8 or abs(diff - 180) <= 8:
                compatibility -= 0.15

        return max(-1, min(1, compatibility))  # Normalize to -1 to 1
