"""
Test cases for NFL Vedic astrological calculations
"""
import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
)

import unittest
from datetime import datetime, timezone
from src.models.vedic_basic.calculations.nfl_vedic_calculator import NFLVedicCalculator
from src.models.vedic_basic.calculations.dashas import DashaCalculator
from src.models.vedic_basic.calculations.ashtakavarga import AshtakavargaCalculator


class TestNFLVedicCalculations(unittest.TestCase):
    """Test NFL Vedic astrological calculations"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = NFLVedicCalculator()
        self.test_time = datetime(
            2024, 2, 11, 15, 30, tzinfo=timezone.utc
        )  # Super Bowl LVIII
        self.test_location = {
            "latitude": 36.0907,  # Allegiant Stadium, Las Vegas
            "longitude": -115.1833,
        }

    def test_basic_calculations(self):
        """Test basic astrological calculations"""
        factors = self.calculator.calculate_game_factors(
            self.test_time, self.test_location
        )

        # Check that all required factors are present
        self.assertIn("planet_positions", factors)
        self.assertIn("house_cusps", factors)
        self.assertIn("planet_strengths", factors)
        self.assertIn("house_strengths", factors)
        self.assertIn("overall_strength", factors)

        # Check strength is normalized
        self.assertGreaterEqual(factors["overall_strength"], 0.0)
        self.assertLessEqual(factors["overall_strength"], 1.0)

        # Check planet positions
        for planet in ["sun", "moon", "mars", "jupiter"]:
            self.assertIn(planet, factors["planet_positions"])
            pos = factors["planet_positions"][planet]
            self.assertGreaterEqual(pos["longitude"], 0.0)
            self.assertLess(pos["longitude"], 360.0)

    def test_dasha_calculations(self):
        """Test Dasha period calculations"""
        dasha_calc = DashaCalculator()

        # Test with known moon position
        moon_longitude = 45.0  # 15° Taurus
        dasha_factors = dasha_calc.calculate_game_dashas(self.test_time, moon_longitude)

        # Check structure
        self.assertIn("major_dasha", dasha_factors)
        self.assertIn("sub_dasha", dasha_factors)
        self.assertIn("combined_strength", dasha_factors)

        # Check strength normalization
        self.assertGreaterEqual(dasha_factors["combined_strength"], 0.0)
        self.assertLessEqual(dasha_factors["combined_strength"], 1.0)

        # Check period planet names
        valid_planets = {
            "sun",
            "moon",
            "mars",
            "rahu",
            "jupiter",
            "saturn",
            "mercury",
            "ketu",
            "venus",
        }
        self.assertIn(dasha_factors["major_dasha"]["planet"], valid_planets)
        self.assertIn(dasha_factors["sub_dasha"]["planet"], valid_planets)

    def test_ashtakavarga_calculations(self):
        """Test Ashtakavarga calculations"""
        ashtak_calc = AshtakavargaCalculator()

        # Create test planetary positions
        test_positions = {
            "sun": {"longitude": 300.0},  # Capricorn
            "moon": {"longitude": 45.0},  # Taurus
            "mars": {"longitude": 120.0},  # Leo
            "mercury": {"longitude": 280.0},  # Capricorn
            "jupiter": {"longitude": 160.0},  # Virgo
            "venus": {"longitude": 330.0},  # Aquarius
            "saturn": {"longitude": 200.0},  # Libra
        }

        results = ashtak_calc.calculate_ashtakavarga(test_positions)

        # Check structure
        self.assertIn("individual_points", results)
        self.assertIn("sarva_points", results)
        self.assertIn("house_strengths", results)
        self.assertIn("game_factors", results)

        # Check points
        for house in range(1, 13):
            self.assertIn(house, results["sarva_points"])
            self.assertGreaterEqual(results["sarva_points"][house], 0)

        # Check house strengths normalization
        for strength in results["house_strengths"].values():
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)

    def test_comprehensive_strength(self):
        """Test comprehensive strength calculation"""
        factors = self.calculator.calculate_game_factors(
            self.test_time, self.test_location
        )

        # Check that all components contribute to final strength
        self.assertIn("overall_strength", factors)
        self.assertIn("dasha_factors", factors)
        self.assertIn("ashtakavarga_factors", factors)

        # Check strength normalization
        self.assertGreaterEqual(factors["overall_strength"], 0.0)
        self.assertLessEqual(factors["overall_strength"], 1.0)

        # Check component strengths
        self.assertGreaterEqual(factors["dasha_factors"]["combined_strength"], 0.0)
        self.assertLessEqual(factors["dasha_factors"]["combined_strength"], 1.0)

        self.assertGreaterEqual(
            factors["ashtakavarga_factors"]["game_factors"]["overall_strength"], 0.0
        )
        self.assertLessEqual(
            factors["ashtakavarga_factors"]["game_factors"]["overall_strength"], 1.0
        )

    def test_special_combinations(self):
        """Test special astrological combinations"""
        factors = self.calculator.calculate_game_factors(
            self.test_time, self.test_location
        )

        self.assertIn("special_combinations", factors)

        # Check each combination
        for combo in factors["special_combinations"]:
            self.assertIn("name", combo)
            self.assertIn("effect", combo)
            self.assertIn("strength", combo)

            # Check strength normalization
            self.assertGreaterEqual(combo["strength"], 0.0)
            self.assertLessEqual(combo["strength"], 1.0)


if __name__ == "__main__":
    unittest.main()
