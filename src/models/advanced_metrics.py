"""
Advanced NFL metrics and analysis system
Implements sophisticated player and team metrics, advanced statistics, and performance analytics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json


class NFLAdvancedMetrics:
    def __init__(self):
        self.logger = logging.getLogger("nfl_metrics")

        # QB Advanced Metrics
        self.qb_metrics = {
            "expected_points_added": {"weight": 0.20, "description": "EPA per play"},
            "completion_percentage_over_expected": {
                "weight": 0.15,
                "description": "CPOE",
            },
            "adjusted_net_yards_per_attempt": {"weight": 0.15, "description": "ANY/A"},
            "pressure_performance": {
                "weight": 0.15,
                "description": "Performance under pressure",
            },
            "deep_ball_efficiency": {"weight": 0.10, "description": "20+ yard throws"},
            "red_zone_efficiency": {"weight": 0.10, "description": "Red zone TD%"},
            "third_down_conversion": {
                "weight": 0.10,
                "description": "3rd down success rate",
            },
            "clutch_performance": {
                "weight": 0.05,
                "description": "Performance in crucial situations",
            },
        }

        # RB Advanced Metrics
        self.rb_metrics = {
            "yards_after_contact_per_attempt": {"weight": 0.20, "description": "YAC/A"},
            "broken_tackles_per_attempt": {
                "weight": 0.15,
                "description": "Broken tackles rate",
            },
            "success_rate": {
                "weight": 0.15,
                "description": "Successful run percentage",
            },
            "expected_points_added_rush": {
                "weight": 0.15,
                "description": "EPA per rush",
            },
            "receiving_value": {"weight": 0.10, "description": "Receiving game impact"},
            "pass_blocking_efficiency": {
                "weight": 0.10,
                "description": "Pass protection rating",
            },
            "situational_efficiency": {
                "weight": 0.10,
                "description": "Performance in key situations",
            },
            "explosive_play_rate": {
                "weight": 0.05,
                "description": "20+ yard play percentage",
            },
        }

        # WR/TE Advanced Metrics
        self.receiver_metrics = {
            "yards_per_route_run": {"weight": 0.20, "description": "YPRR"},
            "separation_rate": {"weight": 0.15, "description": "Average separation"},
            "contested_catch_rate": {
                "weight": 0.15,
                "description": "Contested catch success",
            },
            "yards_after_catch_over_expected": {
                "weight": 0.15,
                "description": "YAC over expected",
            },
            "drop_rate": {"weight": 0.10, "description": "Drops per target"},
            "route_success_rate": {
                "weight": 0.10,
                "description": "Route running efficiency",
            },
            "red_zone_target_conversion": {
                "weight": 0.10,
                "description": "Red zone success rate",
            },
            "deep_target_efficiency": {
                "weight": 0.05,
                "description": "20+ yard reception rate",
            },
        }

        # Defense Advanced Metrics
        self.defense_metrics = {
            "pressure_rate": {"weight": 0.20, "description": "QB pressure generation"},
            "coverage_success_rate": {
                "weight": 0.15,
                "description": "Coverage effectiveness",
            },
            "run_stop_rate": {"weight": 0.15, "description": "Run defense efficiency"},
            "missed_tackle_rate": {"weight": 0.15, "description": "Tackle efficiency"},
            "pass_rush_win_rate": {"weight": 0.10, "description": "Pass rush success"},
            "yards_allowed_per_coverage_snap": {
                "weight": 0.10,
                "description": "Coverage efficiency",
            },
            "playmaking_rate": {"weight": 0.10, "description": "Big play creation"},
            "third_down_stop_rate": {"weight": 0.05, "description": "3rd down defense"},
        }

    def calculate_player_metrics(self, player_data: Dict, position: str) -> Dict:
        """Calculate advanced metrics for a player based on position"""
        try:
            if position.upper() == "QB":
                return self._calculate_qb_metrics(player_data)
            elif position.upper() == "RB":
                return self._calculate_rb_metrics(player_data)
            elif position.upper() in ["WR", "TE"]:
                return self._calculate_receiver_metrics(player_data)
            elif position.upper() in ["DE", "DT", "LB", "CB", "S"]:
                return self._calculate_defense_metrics(player_data)
            else:
                raise ValueError(f"Unsupported position: {position}")

        except Exception as e:
            self.logger.error(f"Error calculating player metrics: {e}")
            raise

    def _calculate_qb_metrics(self, player_data: Dict) -> Dict:
        """Calculate advanced QB metrics"""
        try:
            metrics = {}

            # EPA Calculation
            metrics["epa"] = self._calculate_epa(player_data.get("plays", []))

            # CPOE Calculation
            metrics["cpoe"] = self._calculate_cpoe(
                player_data.get("completions", 0),
                player_data.get("attempts", 0),
                player_data.get("situation_data", {}),
            )

            # ANY/A Calculation
            metrics["anya"] = self._calculate_anya(player_data.get("passing_stats", {}))

            # Pressure Performance
            metrics["pressure_rating"] = self._calculate_pressure_performance(
                player_data.get("pressure_stats", {})
            )

            # Calculate composite score
            metrics["composite_score"] = self._calculate_composite_score(
                metrics, self.qb_metrics
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating QB metrics: {e}")
            raise

    def _calculate_rb_metrics(self, player_data: Dict) -> Dict:
        """Calculate advanced RB metrics"""
        try:
            metrics = {}

            # YAC/A Calculation
            metrics["yac_per_attempt"] = self._calculate_yac_per_attempt(
                player_data.get("rushing_stats", {})
            )

            # Broken Tackles
            metrics["broken_tackles"] = self._calculate_broken_tackles(
                player_data.get("rushing_stats", {})
            )

            # Success Rate
            metrics["success_rate"] = self._calculate_success_rate(
                player_data.get("plays", [])
            )

            # Calculate composite score
            metrics["composite_score"] = self._calculate_composite_score(
                metrics, self.rb_metrics
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating RB metrics: {e}")
            raise

    def _calculate_receiver_metrics(self, player_data: Dict) -> Dict:
        """Calculate advanced receiver metrics"""
        try:
            metrics = {}

            # YPRR Calculation
            metrics["yards_per_route"] = self._calculate_yprr(
                player_data.get("receiving_stats", {})
            )

            # Separation Rate
            metrics["separation"] = self._calculate_separation(
                player_data.get("route_stats", {})
            )

            # Contested Catch Rate
            metrics["contested_catch"] = self._calculate_contested_catch(
                player_data.get("receiving_stats", {})
            )

            # Calculate composite score
            metrics["composite_score"] = self._calculate_composite_score(
                metrics, self.receiver_metrics
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating receiver metrics: {e}")
            raise

    def _calculate_defense_metrics(self, player_data: Dict) -> Dict:
        """Calculate advanced defensive metrics"""
        try:
            metrics = {}

            # Pressure Rate
            metrics["pressure_rate"] = self._calculate_pressure_rate(
                player_data.get("defense_stats", {})
            )

            # Coverage Success
            metrics["coverage_success"] = self._calculate_coverage_success(
                player_data.get("coverage_stats", {})
            )

            # Run Stop Rate
            metrics["run_stop_rate"] = self._calculate_run_stop_rate(
                player_data.get("run_defense_stats", {})
            )

            # Calculate composite score
            metrics["composite_score"] = self._calculate_composite_score(
                metrics, self.defense_metrics
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating defense metrics: {e}")
            raise

    def _calculate_composite_score(self, metrics: Dict, weights: Dict) -> float:
        """Calculate weighted composite score from metrics"""
        score = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            if metric in weights:
                weight = weights[metric]["weight"]
                score += value * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.0

    def save_metrics(self, metrics: Dict, player_id: str):
        """Save calculated metrics to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"player_metrics_{player_id}_{timestamp}.json"
            filepath = Path(__file__).parent / "metrics" / filename

            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(metrics, f, indent=4)

            self.logger.info(f"Saved metrics to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    metrics_calculator = NFLAdvancedMetrics()

    # Example player data
    qb_data = {
        "plays": [],
        "completions": 300,
        "attempts": 450,
        "passing_stats": {"yards": 4000, "touchdowns": 35, "interceptions": 10},
    }

    # Calculate metrics
    metrics = metrics_calculator.calculate_player_metrics(qb_data, position="QB")
    print(json.dumps(metrics, indent=2))
