"""
NFL betting analysis incorporating astrological factors
"""
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from .astro_analyzer import NFLAstroAnalyzer
import logging


class NFLBettingAnalyzer:
    def __init__(self):
        self.astro_analyzer = NFLAstroAnalyzer()

        # Weights for different factors in betting analysis
        self.weights = {
            "astro_strength": 0.3,
            "team_form": 0.25,
            "head_to_head": 0.2,
            "travel_impact": 0.15,
            "weather": 0.1,
        }

    def analyze_game(
        self,
        game_time: datetime,
        home_team: Dict,
        away_team: Dict,
        weather: Optional[str] = None,
    ) -> Dict:
        """
        Analyze a game for betting purposes

        Args:
            game_time: Kickoff time
            home_team: Home team data including location and recent form
            away_team: Away team data including location and recent form
            weather: Weather conditions (optional)

        Returns:
            Dictionary with betting analysis and recommendations
        """
        # Get astrological analysis
        home_astro = self.astro_analyzer.analyze_game_time(
            game_time, home_team["location"]
        )
        away_astro = self.astro_analyzer.analyze_game_time(
            game_time, away_team["location"]
        )

        # Calculate team compatibility
        compatibility = self.astro_analyzer.analyze_team_compatibility(
            home_team["location"], away_team["location"], game_time
        )

        # Calculate weighted scores
        home_score = self._calculate_team_score(
            home_astro["total_strength"],
            home_team.get("form", 0.5),
            home_team.get("h2h_wins", 0.5),
            0,  # No travel impact for home team
            weather,
        )

        away_score = self._calculate_team_score(
            away_astro["total_strength"],
            away_team.get("form", 0.5),
            away_team.get("h2h_wins", 0.5),
            self._calculate_travel_impact(home_team["location"], away_team["location"]),
            weather,
        )

        # Calculate win probabilities
        total_score = home_score + away_score
        if total_score == 0:
            home_prob = 0.5
            away_prob = 0.5
        else:
            home_prob = home_score / total_score
            away_prob = away_score / total_score

        return {
            "home_probability": home_prob,
            "away_probability": away_prob,
            "compatibility": compatibility,
            "home_strength": home_score,
            "away_strength": away_score,
            "confidence": self._calculate_confidence(
                home_prob, away_prob, compatibility
            ),
        }

    def _calculate_team_score(
        self,
        astro_strength: float,
        form: float,
        h2h: float,
        travel_impact: float,
        weather: Optional[str],
    ) -> float:
        """Calculate weighted score for a team"""
        weather_factor = self._analyze_weather_impact(weather)

        score = (
            self.weights["astro_strength"] * astro_strength
            + self.weights["team_form"] * form
            + self.weights["head_to_head"] * h2h
            + self.weights["travel_impact"] * (1 - travel_impact)
            + self.weights["weather"] * weather_factor
        )

        return max(0, score)  # Ensure non-negative

    def _calculate_travel_impact(
        self, home_loc: Tuple[float, float], away_loc: Tuple[float, float]
    ) -> float:
        """Calculate impact of travel on away team"""
        # Simple distance-based calculation
        lat1, lon1 = home_loc
        lat2, lon2 = away_loc

        # Rough distance calculation (could be enhanced)
        distance = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

        # Normalize to 0-1 range (max NFL distance is roughly 45 degrees)
        return min(1.0, distance / 45.0)

    def _analyze_weather_impact(self, weather: Optional[str]) -> float:
        """Analyze weather impact on game"""
        if not weather:
            return 0.5  # Neutral impact

        weather = weather.lower()
        if "snow" in weather or "rain" in weather:
            return 0.3  # Significant impact
        elif "wind" in weather:
            return 0.4  # Moderate impact
        else:
            return 0.5  # Neutral conditions

    def _calculate_confidence(
        self, home_prob: float, away_prob: float, compatibility: float
    ) -> float:
        """Calculate confidence in prediction"""
        # Larger difference in probabilities indicates higher confidence
        prob_diff = abs(home_prob - away_prob)

        # Absolute compatibility value affects confidence
        compatibility_factor = abs(compatibility)

        # Combine factors (weighted average)
        confidence = 0.7 * prob_diff + 0.3 * compatibility_factor

        return min(1.0, confidence)  # Cap at 1.0

    def get_betting_recommendation(self, analysis: Dict) -> Dict:
        """Generate betting recommendation based on analysis"""
        confidence = analysis["confidence"]
        home_prob = analysis["home_probability"]
        away_prob = analysis["away_probability"]

        # Only recommend bets with sufficient confidence
        if confidence < 0.6:
            return {
                "recommendation": "NO_BET",
                "reason": "Insufficient confidence in prediction",
                "confidence": confidence,
            }

        # Determine which team to bet on
        if home_prob > 0.6:
            return {
                "recommendation": "BET_HOME",
                "reason": "Strong home team advantage",
                "confidence": confidence,
                "probability": home_prob,
            }
        elif away_prob > 0.6:
            return {
                "recommendation": "BET_AWAY",
                "reason": "Strong away team advantage",
                "confidence": confidence,
                "probability": away_prob,
            }

        return {
            "recommendation": "NO_BET",
            "reason": "No clear advantage",
            "confidence": confidence,
        }


class EnhancedNFLBettingAnalyzer(NFLBettingAnalyzer):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("nfl_betting")

        # Configure betting thresholds
        self.sharp_money_threshold = 0.65  # 65% sharp money indicates strong movement
        self.line_movement_threshold = 2.0  # 2 point movement is significant
        self.value_threshold = 0.05  # 5% edge needed for value bet

        # Astrological factor weights for betting
        self.astro_weights = {
            "mars": 0.20,  # Aggression and scoring
            "jupiter": 0.20,  # Luck and expansion
            "saturn": 0.15,  # Defense and restrictions
            "sun": 0.15,  # Overall power
            "mercury": 0.15,  # Strategy and adaptability
            "moon": 0.10,  # Public sentiment
            "venus": 0.05,  # Home field advantage
        }

    def analyze_betting_line(self, game_data: Dict, astro_factors: Dict) -> Dict:
        """Analyze betting line incorporating astrological factors"""

        # Get base betting metrics
        base_metrics = self._calculate_base_metrics(game_data)

        # Calculate astrological influence
        astro_influence = self._calculate_astro_influence(astro_factors)

        # Adjust metrics with astrological factors
        adjusted_metrics = self._adjust_metrics_with_astro(
            base_metrics, astro_influence
        )

        # Generate betting recommendation
        recommendation = self._generate_betting_recommendation(adjusted_metrics)

        return {
            "base_metrics": base_metrics,
            "astro_influence": astro_influence,
            "adjusted_metrics": adjusted_metrics,
            "recommendation": recommendation,
        }

    def _calculate_base_metrics(self, game_data: Dict) -> Dict[str, float]:
        """Calculate base betting metrics"""
        metrics = {
            "line_value": 0.0,
            "sharp_money": 0.0,
            "public_money": 0.0,
            "line_movement": 0.0,
        }

        # Line value calculation
        if "predicted_spread" in game_data and "current_spread" in game_data:
            metrics["line_value"] = (
                game_data["predicted_spread"] - game_data["current_spread"]
            )

        # Sharp money calculation
        if "sharp_percentage" in game_data:
            metrics["sharp_money"] = game_data["sharp_percentage"]

        # Public money calculation
        if "public_percentage" in game_data:
            metrics["public_money"] = game_data["public_percentage"]

        # Line movement calculation
        if "opening_spread" in game_data and "current_spread" in game_data:
            metrics["line_movement"] = (
                game_data["current_spread"] - game_data["opening_spread"]
            )

        return metrics

    def _calculate_astro_influence(self, astro_factors: Dict) -> float:
        """Calculate astrological influence on betting factors"""
        influence = 0.0
        for planet, weight in self.astro_weights.items():
            if planet in astro_factors:
                influence += astro_factors[planet] * weight
        return influence

    def _adjust_metrics_with_astro(
        self, metrics: Dict[str, float], astro_influence: float
    ) -> Dict[str, float]:
        """Adjust betting metrics based on astrological influence"""
        adjusted = {}
        for metric, value in metrics.items():
            # Adjust each metric based on astrological influence
            adjusted[metric] = value * (1 + astro_influence)

            # Normalize values between -1 and 1
            if metric in ["line_value", "line_movement"]:
                adjusted[metric] = max(min(adjusted[metric], 1), -1)
            else:
                adjusted[metric] = max(min(adjusted[metric], 1), 0)

        return adjusted

    def _generate_betting_recommendation(self, metrics: Dict[str, float]) -> Dict:
        """Generate betting recommendation based on adjusted metrics"""
        # Calculate overall confidence
        confidence = (
            abs(metrics["line_value"]) * 0.4
            + metrics["sharp_money"] * 0.3
            + abs(metrics["line_movement"]) * 0.2
            + (1 - abs(metrics["public_money"] - 0.5)) * 0.1
        )

        # Determine bet direction
        if (
            metrics["line_value"] > self.value_threshold
            and metrics["sharp_money"] > self.sharp_money_threshold
        ):
            recommendation = "Place bet"
            reason = "Strong value with sharp money support"
        elif metrics["line_value"] < -self.value_threshold and metrics[
            "sharp_money"
        ] < (1 - self.sharp_money_threshold):
            recommendation = "Fade"
            reason = "Negative value with sharp money fading"
        else:
            recommendation = "Pass"
            reason = "Insufficient edge or unclear signals"

        return {
            "recommendation": recommendation,
            "reason": reason,
            "confidence": confidence,
            "value": metrics["line_value"],
        }

    def analyze_historical_trends(
        self, historical_data: List[Dict], astro_data: List[Dict]
    ) -> Dict:
        """Analyze historical betting trends with astrological correlation"""
        trends = {
            "overall_roi": 0.0,
            "astro_correlated_roi": 0.0,
            "strong_signals": [],
            "weak_signals": [],
        }

        if not historical_data or not astro_data:
            return trends

        # Calculate ROI for all bets
        total_bets = len(historical_data)
        winning_bets = sum(1 for bet in historical_data if bet.get("result") == "win")
        trends["overall_roi"] = (winning_bets / total_bets) - 1

        # Calculate ROI for bets with strong astrological signals
        strong_astro_bets = [
            bet
            for bet, astro in zip(historical_data, astro_data)
            if self._calculate_astro_influence(astro) > 0.6
        ]

        if strong_astro_bets:
            winning_astro_bets = sum(
                1 for bet in strong_astro_bets if bet.get("result") == "win"
            )
            trends["astro_correlated_roi"] = (
                winning_astro_bets / len(strong_astro_bets)
            ) - 1

        # Identify strong and weak signals
        if trends["astro_correlated_roi"] > trends["overall_roi"] + 0.1:
            trends["strong_signals"].append(
                "Astrological factors show significant predictive value"
            )
        elif trends["astro_correlated_roi"] < trends["overall_roi"] - 0.1:
            trends["weak_signals"].append("Astrological factors may need recalibration")

        return trends
