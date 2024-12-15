from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from ..base_model import NFLPredictionModel
from .espn_data_fetcher import ESPNDataFetcher
import logging

logger = logging.getLogger(__name__)


class AdvancedModel(NFLPredictionModel):
    """Advanced system combining traditional and advanced NFL metrics"""

    def __init__(self):
        super().__init__()
        # Optimize GradientBoostingClassifier parameters
        self.model = GradientBoostingClassifier(
            n_estimators=200,  # Increased from 100
            learning_rate=0.05,  # Decreased from 0.1 for better generalization
            max_depth=5,  # Increased from 3 to capture more complex patterns
            subsample=0.8,  # Added subsampling for better generalization
            min_samples_split=10,  # Minimum samples required to split
            min_samples_leaf=5,  # Minimum samples required at leaf node
            max_features="sqrt",  # Use sqrt of features for each split
        )
        self.scaler = StandardScaler()  # Add feature scaling
        self.espn_fetcher = ESPNDataFetcher()

        # Feature weights for different categories
        self.feature_weights = {
            "basic_offense": 0.8,
            "advanced_offense": 1.2,
            "advanced_defense": 1.2,
            "situational": 1.1,
            "drive_efficiency": 1.15,
            "team_success": 1.0,
            "momentum": 0.9,
            "game_context": 1.05,
            "player_availability": 1.1,
            "environmental": 0.7,
            "market": 0.85,
            "advanced_analytics": 1.25,
        }

        # Expanded feature set using new ESPN endpoints
        self.feature_columns = [
            # Basic offensive metrics
            "home_points_per_game",
            "away_points_per_game",
            "home_points_allowed",
            "away_points_allowed",
            "home_passing",
            "away_passing",
            "home_qb_rating",
            "away_qb_rating",
            "home_rushing",
            "away_rushing",
            # Advanced offensive metrics
            "home_yards_per_play",
            "away_yards_per_play",
            "home_dvoa_offense",
            "away_dvoa_offense",
            "home_dvoa_defense",
            "away_dvoa_defense",
            "home_dvoa_special_teams",
            "away_dvoa_special_teams",
            "home_success_rate",
            "away_success_rate",
            "home_explosive_play_rate",
            "away_explosive_play_rate",
            "home_air_yards_per_pass",
            "away_air_yards_per_pass",
            "home_yac_per_completion",
            "away_yac_per_completion",
            # Advanced defensive metrics
            "home_defense",
            "away_defense",
            "home_sacks",
            "away_sacks",
            "home_pressure_rate",
            "away_pressure_rate",
            "home_stuff_rate",
            "away_stuff_rate",
            "home_turnover_diff",
            "away_turnover_diff",
            "home_expected_points_defense",
            "away_expected_points_defense",
            # Situational metrics
            "home_third_down",
            "away_third_down",
            "home_red_zone",
            "away_red_zone",
            "home_fourth_down",
            "away_fourth_down",
            "home_goal_line_success",
            "away_goal_line_success",
            "home_two_minute_success",
            "away_two_minute_success",
            # Drive efficiency metrics
            "home_points_per_drive",
            "away_points_per_drive",
            "home_yards_per_drive",
            "away_yards_per_drive",
            "home_plays_per_drive",
            "away_plays_per_drive",
            "home_drive_success_rate",
            "away_drive_success_rate",
            # Team success metrics
            "home_win_pct",
            "away_win_pct",
            "home_division_record",
            "away_division_record",
            "home_conference_record",
            "away_conference_record",
            "home_power_index",
            "away_power_index",
            "home_playoff_probability",
            "away_playoff_probability",
            # Momentum and form
            "home_form",
            "away_form",
            "home_points_trend",
            "away_points_trend",
            "home_yards_trend",
            "away_yards_trend",
            "home_win_streak",
            "away_win_streak",
            "home_recent_performance",
            "away_recent_performance",
            # Game context
            "is_division_game",
            "is_conference_game",
            "playoff_implications",
            "days_rest",
            "head_to_head_wins",
            "head_to_head_points_diff",
            "home_schedule_strength",
            "away_schedule_strength",
            "home_rest_advantage",
            "away_rest_advantage",
            # Player availability
            "home_qb_injury_impact",
            "away_qb_injury_impact",
            "home_offense_injury_impact",
            "away_offense_injury_impact",
            "home_defense_injury_impact",
            "away_defense_injury_impact",
            "home_key_player_injuries",
            "away_key_player_injuries",
            # Environmental factors
            "temperature",
            "wind_speed",
            "precipitation",
            "home_dome_advantage",
            "altitude_impact",
            "weather_advantage",
            "field_condition",
            # Market factors
            "spread",
            "over_under",
            "home_public_betting_percentage",
            "away_public_betting_percentage",
            "line_movement",
            "sharp_money_indicators",
            # Advanced analytics
            "home_expected_points_total",
            "away_expected_points_total",
            "home_win_probability",
            "away_win_probability",
            "home_pressure_impact",
            "away_pressure_impact",
            "home_matchup_advantage",
            "away_matchup_advantage",
            "game_importance_factor",
            "upset_potential",
            # Interaction terms
            "home_offense_defense_synergy",
            "away_offense_defense_synergy",
            "home_efficiency_momentum",
            "away_efficiency_momentum",
            "home_pressure_protection",
            "away_pressure_protection",
            "home_context_performance",
            "away_context_performance",
        ]

    def train(self, training_data: Dict[str, Any]) -> Dict[str, float]:
        """Train the model using historical game data"""
        games = training_data["games"]
        labels = training_data["labels"]

        # Extract features for all games
        features_list = []
        for game in games:
            features = self._extract_features(game)
            features_list.append(features)

        # Combine all features
        X = pd.DataFrame(features_list)

        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        # Reorder columns to match feature_columns
        X = X[self.feature_columns]

        # Fill NaN values with 0
        X = X.fillna(0)

        # Scale features
        X = self.scaler.fit_transform(X)

        # Apply feature weights
        X = self._apply_feature_weights(X)

        # Train model
        self.model.fit(X, labels)

        # Calculate and log feature importance
        feature_importance = self._log_feature_importance()

        # Calculate training metrics
        train_score = self.model.score(X, labels)

        return {
            "training_accuracy": train_score,
            "feature_importance": feature_importance,
        }

    def _create_interaction_terms(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction terms between important features"""
        interactions = {}

        # Offense-Defense interactions
        interactions["home_offense_defense_synergy"] = (
            features["home_dvoa_offense"] * features["home_dvoa_defense"]
        )
        interactions["away_offense_defense_synergy"] = (
            features["away_dvoa_offense"] * features["away_dvoa_defense"]
        )

        # Efficiency-Momentum interactions
        interactions["home_efficiency_momentum"] = (
            features["home_success_rate"] * features["home_recent_performance"]
        )
        interactions["away_efficiency_momentum"] = (
            features["away_success_rate"] * features["away_recent_performance"]
        )

        # Pressure-Protection interactions
        interactions["home_pressure_protection"] = (
            features["home_pressure_rate"] * features["home_sacks"]
        )
        interactions["away_pressure_protection"] = (
            features["away_pressure_rate"] * features["away_sacks"]
        )

        # Context-Performance interactions
        interactions["home_context_performance"] = features["home_power_index"] * (
            1 + features["is_division_game"] + features["playoff_implications"]
        )
        interactions["away_context_performance"] = features["away_power_index"] * (
            1 + features["is_division_game"] + features["playoff_implications"]
        )

        return interactions

    def _apply_feature_weights(self, X: np.ndarray) -> np.ndarray:
        """Apply category-specific weights to features"""
        weighted_X = X.copy()

        for i, feature in enumerate(self.feature_columns):
            category = self._get_feature_category(feature)
            weight = self.feature_weights.get(category, 1.0)
            weighted_X[:, i] *= weight

        return weighted_X

    def _get_feature_category(self, feature: str) -> str:
        """Determine category for a given feature"""
        if any(
            x in feature for x in ["points_per_game", "passing", "rushing", "qb_rating"]
        ):
            return "basic_offense"
        elif any(x in feature for x in ["dvoa", "success_rate", "explosive_play_rate"]):
            return "advanced_offense"
        elif any(x in feature for x in ["defense", "pressure_rate", "stuff_rate"]):
            return "advanced_defense"
        elif any(x in feature for x in ["third_down", "red_zone", "goal_line"]):
            return "situational"
        elif any(x in feature for x in ["per_drive", "drive_success"]):
            return "drive_efficiency"
        elif any(
            x in feature for x in ["win_pct", "power_index", "playoff_probability"]
        ):
            return "team_success"
        elif any(x in feature for x in ["form", "trend", "streak", "recent"]):
            return "momentum"
        elif any(
            x in feature for x in ["division", "conference", "playoff_implications"]
        ):
            return "game_context"
        elif any(x in feature for x in ["injury", "availability"]):
            return "player_availability"
        elif any(x in feature for x in ["temperature", "wind", "precipitation"]):
            return "environmental"
        elif any(x in feature for x in ["spread", "betting", "line_movement"]):
            return "market"
        else:
            return "advanced_analytics"

    def _log_feature_importance(self) -> Dict[str, float]:
        """Calculate and log feature importance"""
        try:
            # Get raw feature importances
            importances = self.model.feature_importances_

            if len(importances) != len(self.feature_columns):
                logger.warning(
                    f"Feature mismatch: {len(importances)} importances vs {len(self.feature_columns)} columns"
                )
                return {}

            # Apply feature weights to importances
            weighted_importances = []
            for i, importance in enumerate(importances):
                feature = self.feature_columns[i]
                category = self._get_feature_category(feature)
                weight = self.feature_weights.get(category, 1.0)
                weighted_importances.append(importance * weight)

            # Normalize to sum to 1
            total = sum(weighted_importances)
            if total > 0:  # Avoid division by zero
                weighted_importances = [imp / total for imp in weighted_importances]

            # Create feature importance dictionary
            feature_importance = dict(zip(self.feature_columns, weighted_importances))

            # Sort features by importance
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )

            # Log top 20 most important features
            logger.info("\nTop 20 most important features:")
            logged_features = 0
            for feature, importance in sorted_features:
                if importance > 0.001:  # Only log features with >0.1% importance
                    logger.info(f"{feature}: {importance:.4f}")
                    logged_features += 1
                    if logged_features >= 20:
                        break

            # Calculate and log importance by category
            category_importance = {}
            for feature, importance in feature_importance.items():
                if importance > 0.001:  # Only include features with >0.1% importance
                    category = self._get_feature_category(feature)
                    if category not in category_importance:
                        category_importance[category] = 0
                    category_importance[category] += importance

            # Log category importances
            if category_importance:
                logger.info("\nFeature importance by category:")
                for category, importance in sorted(
                    category_importance.items(), key=lambda x: x[1], reverse=True
                ):
                    logger.info(f"{category}: {importance:.4f}")

            return feature_importance

        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}

    def _extract_features(self, game_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from game data"""
        features = {}

        try:
            # Get ESPN API data
            game_id = game_data.get("game_id")
            home_team_id = game_data.get("home_team_id")
            away_team_id = game_data.get("away_team_id")
            season = game_data.get("season")

            if game_id and home_team_id and away_team_id and season:
                espn_game_data = self.espn_fetcher.get_game_data(game_id)
                home_season_stats = self.espn_fetcher.get_team_season_stats(
                    home_team_id, season
                )
                away_season_stats = self.espn_fetcher.get_team_season_stats(
                    away_team_id, season
                )

                # Basic offensive efficiency
                features["home_points_per_game"] = float(
                    home_season_stats["statistics"].get("points_per_game", 0)
                )
                features["away_points_per_game"] = float(
                    away_season_stats["statistics"].get("points_per_game", 0)
                )
                features["home_points_allowed"] = float(
                    home_season_stats["statistics"].get("points_allowed", 0)
                )
                features["away_points_allowed"] = float(
                    away_season_stats["statistics"].get("points_allowed", 0)
                )

                # Advanced offensive metrics
                features["home_yards_per_play"] = float(
                    home_season_stats["statistics"].get("yards_per_play", 5.5)
                )
                features["away_yards_per_play"] = float(
                    away_season_stats["statistics"].get("yards_per_play", 5.5)
                )
                features["home_dvoa_offense"] = float(
                    home_season_stats["statistics"].get("dvoa_offense", 0.0)
                )
                features["away_dvoa_offense"] = float(
                    away_season_stats["statistics"].get("dvoa_offense", 0.0)
                )
                features["home_dvoa_defense"] = float(
                    home_season_stats["statistics"].get("dvoa_defense", 0.0)
                )
                features["away_dvoa_defense"] = float(
                    away_season_stats["statistics"].get("dvoa_defense", 0.0)
                )
                features["home_dvoa_special_teams"] = float(
                    home_season_stats["statistics"].get("dvoa_special_teams", 0.0)
                )
                features["away_dvoa_special_teams"] = float(
                    away_season_stats["statistics"].get("dvoa_special_teams", 0.0)
                )
                features["home_success_rate"] = float(
                    home_season_stats["statistics"].get("success_rate", 0.0)
                )
                features["away_success_rate"] = float(
                    away_season_stats["statistics"].get("success_rate", 0.0)
                )
                features["home_explosive_play_rate"] = float(
                    home_season_stats["statistics"].get("explosive_play_rate", 0.0)
                )
                features["away_explosive_play_rate"] = float(
                    away_season_stats["statistics"].get("explosive_play_rate", 0.0)
                )
                features["home_air_yards_per_pass"] = float(
                    home_season_stats["statistics"].get("air_yards_per_pass", 0.0)
                )
                features["away_air_yards_per_pass"] = float(
                    away_season_stats["statistics"].get("air_yards_per_pass", 0.0)
                )
                features["home_yac_per_completion"] = float(
                    home_season_stats["statistics"].get("yac_per_completion", 0.0)
                )
                features["away_yac_per_completion"] = float(
                    away_season_stats["statistics"].get("yac_per_completion", 0.0)
                )

                # Passing game with advanced metrics
                features["home_passing"] = float(
                    home_season_stats["statistics"].get("passing_yards", 0)
                )
                features["away_passing"] = float(
                    away_season_stats["statistics"].get("passing_yards", 0)
                )
                features["home_qb_rating"] = float(
                    home_season_stats["statistics"].get("qb_rating", 85.0)
                )
                features["away_qb_rating"] = float(
                    away_season_stats["statistics"].get("qb_rating", 85.0)
                )

                # Running game
                features["home_rushing"] = float(
                    home_season_stats["statistics"].get("rushing_yards", 0)
                )
                features["away_rushing"] = float(
                    away_season_stats["statistics"].get("rushing_yards", 0)
                )

                # Advanced defensive metrics
                features["home_defense"] = float(
                    home_season_stats["statistics"].get("defensive_efficiency", 0)
                )
                features["away_defense"] = float(
                    away_season_stats["statistics"].get("defensive_efficiency", 0)
                )
                features["home_sacks"] = float(
                    home_season_stats["statistics"].get("sacks", 0)
                )
                features["away_sacks"] = float(
                    away_season_stats["statistics"].get("sacks", 0)
                )
                features["home_pressure_rate"] = float(
                    home_season_stats["statistics"].get("pressure_rate", 0.0)
                )
                features["away_pressure_rate"] = float(
                    away_season_stats["statistics"].get("pressure_rate", 0.0)
                )
                features["home_stuff_rate"] = float(
                    home_season_stats["statistics"].get("stuff_rate", 0.0)
                )
                features["away_stuff_rate"] = float(
                    away_season_stats["statistics"].get("stuff_rate", 0.0)
                )
                features["home_turnover_diff"] = float(
                    home_season_stats["statistics"].get("turnover_diff", 0)
                )
                features["away_turnover_diff"] = float(
                    away_season_stats["statistics"].get("turnover_diff", 0)
                )
                features["home_expected_points_defense"] = float(
                    home_season_stats["statistics"].get("expected_points_defense", 0.0)
                )
                features["away_expected_points_defense"] = float(
                    away_season_stats["statistics"].get("expected_points_defense", 0.0)
                )

                # Situational metrics
                features["home_third_down"] = float(
                    home_season_stats["statistics"].get("third_down_pct", 0)
                )
                features["away_third_down"] = float(
                    away_season_stats["statistics"].get("third_down_pct", 0)
                )
                features["home_red_zone"] = float(
                    home_season_stats["statistics"].get("red_zone_pct", 0)
                )
                features["away_red_zone"] = float(
                    away_season_stats["statistics"].get("red_zone_pct", 0)
                )
                features["home_fourth_down"] = float(
                    home_season_stats["statistics"].get("fourth_down_pct", 0)
                )
                features["away_fourth_down"] = float(
                    away_season_stats["statistics"].get("fourth_down_pct", 0)
                )
                features["home_goal_line_success"] = float(
                    home_season_stats["statistics"].get("goal_line_success", 0.0)
                )
                features["away_goal_line_success"] = float(
                    away_season_stats["statistics"].get("goal_line_success", 0.0)
                )
                features["home_two_minute_success"] = float(
                    home_season_stats["statistics"].get("two_minute_success", 0.0)
                )
                features["away_two_minute_success"] = float(
                    away_season_stats["statistics"].get("two_minute_success", 0.0)
                )

                # Drive efficiency metrics
                features["home_points_per_drive"] = float(
                    home_season_stats["statistics"].get("points_per_drive", 0.0)
                )
                features["away_points_per_drive"] = float(
                    away_season_stats["statistics"].get("points_per_drive", 0.0)
                )
                features["home_yards_per_drive"] = float(
                    home_season_stats["statistics"].get("yards_per_drive", 0.0)
                )
                features["away_yards_per_drive"] = float(
                    away_season_stats["statistics"].get("yards_per_drive", 0.0)
                )
                features["home_plays_per_drive"] = float(
                    home_season_stats["statistics"].get("plays_per_drive", 0.0)
                )
                features["away_plays_per_drive"] = float(
                    away_season_stats["statistics"].get("plays_per_drive", 0.0)
                )
                features["home_drive_success_rate"] = float(
                    home_season_stats["statistics"].get("drive_success_rate", 0.0)
                )
                features["away_drive_success_rate"] = float(
                    away_season_stats["statistics"].get("drive_success_rate", 0.0)
                )

                # Team success metrics
                features["home_win_pct"] = float(home_season_stats.get("form", 0))
                features["away_win_pct"] = float(away_season_stats.get("form", 0))
                features["home_division_record"] = self._calculate_division_record(
                    home_season_stats
                )
                features["away_division_record"] = self._calculate_division_record(
                    away_season_stats
                )
                features["home_conference_record"] = self._calculate_conference_record(
                    home_season_stats
                )
                features["away_conference_record"] = self._calculate_conference_record(
                    away_season_stats
                )
                features["home_power_index"] = float(
                    home_season_stats["statistics"].get("power_index", 0.0)
                )
                features["away_power_index"] = float(
                    away_season_stats["statistics"].get("power_index", 0.0)
                )
                features["home_playoff_probability"] = float(
                    home_season_stats["statistics"].get("playoff_probability", 0.0)
                )
                features["away_playoff_probability"] = float(
                    away_season_stats["statistics"].get("playoff_probability", 0.0)
                )

                # Momentum and form
                features["home_form"] = float(home_season_stats.get("form", 0))
                features["away_form"] = float(away_season_stats.get("form", 0))
                features["home_points_trend"] = float(
                    home_season_stats["trends"].get("points_trend", 0)
                )
                features["away_points_trend"] = float(
                    away_season_stats["trends"].get("points_trend", 0)
                )
                features["home_yards_trend"] = float(
                    home_season_stats["trends"].get("yards_trend", 0)
                )
                features["away_yards_trend"] = float(
                    away_season_stats["trends"].get("yards_trend", 0)
                )
                features["home_win_streak"] = float(
                    home_season_stats["statistics"].get("win_streak", 0)
                )
                features["away_win_streak"] = float(
                    away_season_stats["statistics"].get("win_streak", 0)
                )
                features["home_recent_performance"] = float(
                    home_season_stats["statistics"].get("recent_performance", 0.0)
                )
                features["away_recent_performance"] = float(
                    away_season_stats["statistics"].get("recent_performance", 0.0)
                )

                # Game context
                features["is_division_game"] = float(
                    self._is_division_game(home_team_id, away_team_id)
                )
                features["is_conference_game"] = float(
                    self._is_conference_game(home_team_id, away_team_id)
                )
                features["playoff_implications"] = float(
                    self._has_playoff_implications(home_season_stats, away_season_stats)
                )
                features["days_rest"] = self._calculate_days_rest(game_data)
                h2h_stats = self._get_head_to_head_stats(
                    home_team_id, away_team_id, season
                )
                features["head_to_head_wins"] = float(h2h_stats.get("wins", 0))
                features["head_to_head_points_diff"] = float(
                    h2h_stats.get("points_diff", 0)
                )
                features["home_schedule_strength"] = float(
                    home_season_stats["statistics"].get("schedule_strength", 0.0)
                )
                features["away_schedule_strength"] = float(
                    away_season_stats["statistics"].get("schedule_strength", 0.0)
                )
                features["home_rest_advantage"] = float(
                    home_season_stats["statistics"].get("rest_advantage", 0.0)
                )
                features["away_rest_advantage"] = float(
                    away_season_stats["statistics"].get("rest_advantage", 0.0)
                )

                # Player availability
                features["home_qb_injury_impact"] = float(
                    home_season_stats["roster_health"].get("qb", 0)
                )
                features["away_qb_injury_impact"] = float(
                    away_season_stats["roster_health"].get("qb", 0)
                )
                features["home_offense_injury_impact"] = float(
                    home_season_stats["roster_health"].get("offense", 0)
                )
                features["away_offense_injury_impact"] = float(
                    away_season_stats["roster_health"].get("offense", 0)
                )
                features["home_defense_injury_impact"] = float(
                    home_season_stats["roster_health"].get("defense", 0)
                )
                features["away_defense_injury_impact"] = float(
                    away_season_stats["roster_health"].get("defense", 0)
                )
                features["home_key_player_injuries"] = float(
                    home_season_stats["roster_health"].get("key_players", 0)
                )
                features["away_key_player_injuries"] = float(
                    away_season_stats["roster_health"].get("key_players", 0)
                )

                # Environmental factors
                weather = espn_game_data.get("weather", {})
                stadium = espn_game_data.get("stadium", {})
                features["temperature"] = float(weather.get("temperature", 70))
                features["wind_speed"] = float(weather.get("wind_speed", 0))
                features["precipitation"] = float(weather.get("precipitation", 0))
                features["home_dome_advantage"] = float(stadium.get("indoor", False))
                features["altitude_impact"] = (
                    float(stadium.get("altitude", 0)) / 5280.0
                )  # normalize to miles
                features["weather_advantage"] = float(
                    weather.get("weather_advantage", 0.0)
                )
                features["field_condition"] = float(stadium.get("field_condition", 0.0))

                # Market factors
                features["spread"] = float(game_data.get("spread", 0))
                features["over_under"] = float(game_data.get("over_under", 0))
                features["home_public_betting_percentage"] = float(
                    game_data.get("home_public_betting_percentage", 0.0)
                )
                features["away_public_betting_percentage"] = float(
                    game_data.get("away_public_betting_percentage", 0.0)
                )
                features["line_movement"] = float(game_data.get("line_movement", 0.0))
                features["sharp_money_indicators"] = float(
                    game_data.get("sharp_money_indicators", 0.0)
                )

                # Advanced analytics
                features["home_expected_points_total"] = float(
                    home_season_stats["statistics"].get("expected_points_total", 0.0)
                )
                features["away_expected_points_total"] = float(
                    away_season_stats["statistics"].get("expected_points_total", 0.0)
                )
                features["home_win_probability"] = float(
                    home_season_stats["statistics"].get("win_probability", 0.0)
                )
                features["away_win_probability"] = float(
                    away_season_stats["statistics"].get("win_probability", 0.0)
                )
                features["home_pressure_impact"] = float(
                    home_season_stats["statistics"].get("pressure_impact", 0.0)
                )
                features["away_pressure_impact"] = float(
                    away_season_stats["statistics"].get("pressure_impact", 0.0)
                )
                features["home_matchup_advantage"] = float(
                    home_season_stats["statistics"].get("matchup_advantage", 0.0)
                )
                features["away_matchup_advantage"] = float(
                    away_season_stats["statistics"].get("matchup_advantage", 0.0)
                )
                features["game_importance_factor"] = float(
                    game_data.get("game_importance_factor", 0.0)
                )
                features["upset_potential"] = float(
                    game_data.get("upset_potential", 0.0)
                )

            # Fill any missing features with defaults
            for feature in self.feature_columns:
                if feature not in features:
                    features[feature] = 0.0

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            # Return default features if there's an error
            return {feature: 0.0 for feature in self.feature_columns}

        return features

    def _calculate_division_record(self, season_stats: Dict[str, Any]) -> float:
        """Calculate division record"""
        division_wins = season_stats["statistics"].get("division_wins", 0)
        division_losses = season_stats["statistics"].get("division_losses", 0)
        return (
            division_wins / (division_wins + division_losses)
            if (division_wins + division_losses) > 0
            else 0.0
        )

    def _calculate_conference_record(self, season_stats: Dict[str, Any]) -> float:
        """Calculate conference record"""
        conference_wins = season_stats["statistics"].get("conference_wins", 0)
        conference_losses = season_stats["statistics"].get("conference_losses", 0)
        return (
            conference_wins / (conference_wins + conference_losses)
            if (conference_wins + conference_losses) > 0
            else 0.0
        )

    def _is_division_game(self, home_team_id: int, away_team_id: int) -> bool:
        """Check if game is a division game"""
        # Assume division IDs are stored in team data
        home_division_id = self.espn_fetcher.get_team_division(home_team_id)
        away_division_id = self.espn_fetcher.get_team_division(away_team_id)
        return home_division_id == away_division_id

    def _is_conference_game(self, home_team_id: int, away_team_id: int) -> bool:
        """Check if game is a conference game"""
        # Assume conference IDs are stored in team data
        home_conference_id = self.espn_fetcher.get_team_conference(home_team_id)
        away_conference_id = self.espn_fetcher.get_team_conference(away_team_id)
        return home_conference_id == away_conference_id

    def _has_playoff_implications(
        self, home_season_stats: Dict[str, Any], away_season_stats: Dict[str, Any]
    ) -> bool:
        """Check if game has playoff implications"""
        # Assume playoff implications are stored in season stats
        home_playoff_implications = home_season_stats.get("playoff_implications", False)
        away_playoff_implications = away_season_stats.get("playoff_implications", False)
        return home_playoff_implications or away_playoff_implications

    def _calculate_days_rest(self, game_data: Dict[str, Any]) -> int:
        """Calculate days rest for each team"""
        # Assume game schedule is stored in game data
        game_schedule = game_data.get("schedule", [])
        last_game_date = None
        for game in game_schedule:
            if game["date"] < game_data["date"]:
                last_game_date = game["date"]
                break
        if last_game_date:
            return (game_data["date"] - last_game_date).days
        else:
            return 7  # Default to 7 days rest if no previous game found

    def _get_head_to_head_stats(
        self, home_team_id: int, away_team_id: int, season: int
    ) -> Dict[str, Any]:
        """Get head-to-head stats for two teams"""
        # Assume head-to-head stats are stored in team data
        home_team_h2h_stats = self.espn_fetcher.get_team_head_to_head(
            home_team_id, away_team_id, season
        )
        away_team_h2h_stats = self.espn_fetcher.get_team_head_to_head(
            away_team_id, home_team_id, season
        )
        return {
            "wins": home_team_h2h_stats.get("wins", 0),
            "points_diff": home_team_h2h_stats.get("points_diff", 0),
        }

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single game"""
        try:
            # Extract features
            features = self._extract_features(game_data)

            # Add interaction terms
            features.update(self._create_interaction_terms(features))

            # Convert to DataFrame and fill NaN values
            X = pd.DataFrame([features])
            X = X.fillna(0)

            # Scale features
            X = self.scaler.transform(X)

            # Apply feature weights
            X = self._apply_feature_weights(X)

            # Make prediction
            prob = self.model.predict_proba(X)[0]
            home_win_prob = prob[1]

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                game_data, {"home_win_probability": home_win_prob}
            )

            return {
                "home_win_probability": float(home_win_prob),
                "confidence_score": abs(home_win_prob - 0.5)
                * 2,  # Convert probability margin to confidence
                "model_specific_factors": {
                    "home_win_probability": float(home_win_prob),
                    "feature_contributions": feature_importance,
                },
            }

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {
                "home_win_probability": 0.5,  # Default to 50% if error
                "confidence_score": 0.0,
                "model_specific_factors": {
                    "home_win_probability": 0.5,
                    "feature_contributions": {},
                },
            }

    def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        try:
            games = test_data["games"]
            labels = test_data["labels"]

            # Extract features for all games
            features_list = []
            for game in games:
                features = self._extract_features(game)

                # Add interaction terms
                features.update(self._create_interaction_terms(features))

                features_list.append(features)

            # Combine all features
            X = pd.DataFrame(features_list)

            # Fill NaN values with 0
            X = X.fillna(0)

            # Scale features
            X = self.scaler.transform(X)

            # Apply feature weights
            X = self._apply_feature_weights(X)

            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            # Calculate metrics
            accuracy = np.mean(predictions == labels)

            return {
                "accuracy": float(accuracy),
                "feature_importance": dict(
                    zip(self.feature_columns, self.model.feature_importances_)
                ),
            }

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {"accuracy": 0.0, "feature_importance": {}}

    def _calculate_feature_importance(
        self, game_data: Dict[str, Any], prediction: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate feature importance for the prediction"""
        features = self._extract_features(game_data)
        importance = {}

        # Group features by category
        categories = {
            "Basic Offense": ["points_per_game", "passing", "rushing", "qb_rating"],
            "Advanced Offense": [
                "yards_per_play",
                "dvoa_offense",
                "success_rate",
                "explosive_play_rate",
                "air_yards_per_pass",
                "yac_per_completion",
            ],
            "Basic Defense": ["points_allowed", "defense", "sacks"],
            "Advanced Defense": [
                "pressure_rate",
                "stuff_rate",
                "turnover_diff",
                "expected_points_defense",
            ],
            "Situational": [
                "third_down",
                "red_zone",
                "fourth_down",
                "goal_line_success",
                "two_minute_success",
            ],
            "Drive Efficiency": [
                "points_per_drive",
                "yards_per_drive",
                "plays_per_drive",
                "drive_success_rate",
            ],
            "Team Success": [
                "win_pct",
                "division_record",
                "conference_record",
                "power_index",
                "playoff_probability",
            ],
            "Momentum": [
                "form",
                "points_trend",
                "yards_trend",
                "win_streak",
                "recent_performance",
            ],
            "Game Context": [
                "division_game",
                "conference_game",
                "playoff_implications",
                "schedule_strength",
                "rest_advantage",
            ],
            "Player Health": [
                "qb_injury_impact",
                "offense_injury_impact",
                "defense_injury_impact",
                "key_player_injuries",
            ],
            "Environment": [
                "temperature",
                "wind_speed",
                "precipitation",
                "dome_advantage",
                "altitude_impact",
                "weather_advantage",
                "field_condition",
            ],
            "Market": [
                "spread",
                "over_under",
                "public_betting_percentage",
                "line_movement",
                "sharp_money_indicators",
            ],
            "Analytics": [
                "expected_points_total",
                "win_probability",
                "pressure_impact",
                "matchup_advantage",
                "game_importance_factor",
                "upset_potential",
            ],
        }

        # Calculate importance for each category
        for category, feature_patterns in categories.items():
            category_importance = 0.0
            count = 0

            for pattern in feature_patterns:
                for key, value in features.items():
                    if pattern in key:
                        # Calculate feature contribution
                        if "home_" in key:
                            contribution = value - features.get(
                                key.replace("home_", "away_"), 0
                            )
                        else:
                            continue

                        category_importance += abs(contribution)
                        count += 1

            if count > 0:
                importance[category] = category_importance / count

        # Normalize importance scores
        total = sum(importance.values())
        if total > 0:
            for key in importance:
                importance[key] /= total

        return importance
