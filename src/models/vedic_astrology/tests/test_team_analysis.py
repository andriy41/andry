"""Unit tests for team analysis module."""

import unittest
from datetime import datetime
import numpy as np
from ..predictions.team_analysis import NFLTeamAnalysis


class TestNFLTeamAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.test_team_data = {
            "name": "TEST",
            "latitude": 40.0,
            "longitude": -80.0,
            "date": datetime(2020, 1, 1),
        }
        self.analyzer = NFLTeamAnalysis(self.test_team_data)

    def test_calculate_team_strength_handles_invalid_data(self):
        """Test that team strength calculation handles invalid data gracefully."""
        # Test with None game time
        result = self.analyzer.calculate_team_strength(None)
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result.get("overall_strength", 0), 0)
        self.assertLessEqual(result.get("overall_strength", 1), 1)

        # Test with invalid coordinates
        bad_team_data = {
            "name": "BAD",
            "latitude": None,
            "longitude": None,
            "date": datetime.now(),
        }
        bad_analyzer = NFLTeamAnalysis(bad_team_data)
        result = bad_analyzer.calculate_team_strength()
        self.assertIsInstance(result, dict)

    def test_calculate_team_strength_returns_valid_range(self):
        """Test that strength calculations return values in valid ranges."""
        result = self.analyzer.calculate_team_strength()

        # Check overall strength
        self.assertGreaterEqual(result.get("overall_strength", 0), 0)
        self.assertLessEqual(result.get("overall_strength", 1), 1)

        # Check individual components
        for key in ["varga_strengths", "shadbala", "ashtakavarga"]:
            if key in result:
                values = result[key].values()
                self.assertTrue(all(0 <= v <= 1 for v in values))

    def test_calculate_team_strength_handles_edge_cases(self):
        """Test handling of edge cases in team strength calculation."""
        # Test with extreme latitudes
        extreme_team_data = {
            "name": "EXTREME",
            "latitude": 89.9,  # Near North Pole
            "longitude": 0.0,
            "date": datetime.now(),
        }
        extreme_analyzer = NFLTeamAnalysis(extreme_team_data)
        result = extreme_analyzer.calculate_team_strength()
        self.assertIsInstance(result, dict)

        # Test with date at year boundary
        year_end_data = {
            "name": "YEAREND",
            "latitude": 40.0,
            "longitude": -80.0,
            "date": datetime(2020, 12, 31, 23, 59, 59),
        }
        year_end_analyzer = NFLTeamAnalysis(year_end_data)
        result = year_end_analyzer.calculate_team_strength()
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
