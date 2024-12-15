"""Unit tests for planetary calculations."""

import unittest
from datetime import datetime
import swisseph as swe
from ..calculations.planetary_position import (
    calculate_planet_positions,
    setup_ephemeris,
)
from ..calculations.shadbala import calculate_shadbala
from ..calculations.ashtakavarga import calculate_ashtakavarga


class TestPlanetaryCalculations(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_time = datetime(2020, 1, 1, 12, 0, 0)
        self.test_lat = 40.0
        self.test_long = -80.0
        # Initialize ephemeris
        setup_ephemeris()

    def test_planet_position_calculation(self):
        """Test that planetary positions are calculated correctly."""
        # Calculate Julian Day
        jd = swe.julday(
            self.test_time.year,
            self.test_time.month,
            self.test_time.day,
            self.test_time.hour + self.test_time.minute / 60.0,
        )

        # Test planet positions calculation
        positions, houses, points = calculate_planet_positions(
            jd, self.test_lat, self.test_long
        )

        # Verify positions dictionary
        self.assertIsInstance(positions, dict)
        self.assertIn("Sun", positions)
        self.assertIn("Moon", positions)
        self.assertIn("Mars", positions)

        # Verify position ranges
        for planet, pos in positions.items():
            if isinstance(pos, (int, float)):
                self.assertGreaterEqual(pos, 0)
                self.assertLess(pos, 360)
            elif isinstance(pos, dict) and "longitude" in pos:
                self.assertGreaterEqual(pos["longitude"], 0)
                self.assertLess(pos["longitude"], 360)

        # Verify houses
        self.assertIsInstance(houses, dict)
        self.assertIn("Ascendant", houses)

    def test_shadbala_calculation(self):
        """Test Shadbala strength calculations."""
        jd = swe.julday(
            self.test_time.year,
            self.test_time.month,
            self.test_time.day,
            self.test_time.hour + self.test_time.minute / 60.0,
        )

        try:
            result = calculate_shadbala(jd, self.test_lat, self.test_long)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, dict)

            # Check each planet has a strength value between 0 and 1
            for planet in result.values():
                self.assertGreaterEqual(planet, 0)
                self.assertLessEqual(planet, 1)
        except Exception as e:
            self.fail(f"Shadbala calculation failed: {str(e)}")

    def test_ashtakavarga_calculation(self):
        """Test Ashtakavarga calculations."""
        jd = swe.julday(
            self.test_time.year,
            self.test_time.month,
            self.test_time.day,
            self.test_time.hour + self.test_time.minute / 60.0,
        )

        try:
            # Get planet positions first
            positions, houses, _ = calculate_planet_positions(
                jd, self.test_lat, self.test_long
            )

            # Convert positions to the format expected by ashtakavarga
            planet_positions = {}
            for planet, pos in positions.items():
                if isinstance(pos, dict) and "longitude" in pos:
                    planet_positions[planet] = pos["longitude"]
                elif isinstance(pos, (int, float)):
                    planet_positions[planet] = pos

            # Calculate Ashtakavarga for each planet
            for planet in [
                "Sun",
                "Moon",
                "Mars",
                "Mercury",
                "Jupiter",
                "Venus",
                "Saturn",
            ]:
                if planet in planet_positions:
                    result = calculate_ashtakavarga(planet, planet_positions, houses)
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result, dict)

                    # Check bindus are in valid range (0-8)
                    for house, bindu in result["bindus"].items():
                        if isinstance(bindu, (int, float)):  # Skip nested dictionaries
                            self.assertGreaterEqual(bindu, 0)
                            self.assertLessEqual(bindu, 8)
        except Exception as e:
            self.fail(f"Ashtakavarga calculation failed: {str(e)}")

    def test_error_handling(self):
        """Test error handling in planetary calculations."""
        # Test with invalid Julian Day
        with self.assertRaises(ValueError):
            calculate_planet_positions(-1000000, self.test_lat, self.test_long)

        # Test with invalid latitude/longitude
        with self.assertRaises(ValueError):
            calculate_planet_positions(2459000.5, -100, 200)  # Invalid lat/long

        # Test ashtakavarga with invalid inputs
        with self.assertRaises(ValueError):
            calculate_ashtakavarga("Sun", None, {})  # Invalid positions

        with self.assertRaises(ValueError):
            calculate_ashtakavarga("InvalidPlanet", {}, {})  # Invalid planet name


if __name__ == "__main__":
    unittest.main()
