from src.models import (
    TotalPredictionModel,
    NeuroTotalModel,
    AstroTotalModel,
    VedicTotalModel,
    MLTotalModel,
    StatsTotalModel,
)
from src.data_collection.totals_collector import NFLTotalsCollector
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
from src.utils.astro_utils import (
    calculate_astrological_advantage,
    calculate_yogas,
    get_planet_positions,
    calculate_planetary_hour_strength,
    calculate_mars_jupiter_relationship,
    calculate_shadow_planet_influence,
)
import pandas as pd
import logging


class TotalPredictor:
    """Comprehensive NFL total points predictor using multiple models"""

    def __init__(self):
        """Initialize all prediction models"""
        try:
            # Initialize all models
            self.models = {
                "advanced": TotalPredictionModel(),  # Advanced metrics + traditional stats
                "vedic": VedicTotalModel(),  # Pure astrological calculations
                "ml": MLTotalModel(),  # Combined ML system
                "stats": StatsTotalModel(),  # Sports-only system
                "neuro": NeuroTotalModel(),  # Neural network model
            }

            # Configure confidence thresholds
            self.confidence_thresholds = {
                "high": 0.85,  # At least one model must exceed this
                "medium": 0.80,  # All models must exceed this for high confidence
                "low": 0.70,  # Minimum acceptable confidence
            }

            # Generate and load training data
            self._initialize_with_training_data()

            logging.info("Successfully initialized all prediction models")

        except Exception as e:
            logging.error(f"Error initializing prediction models: {str(e)}")
            raise

    def _initialize_with_training_data(self):
        """Initialize models with synthetic training data"""
        try:
            # Generate synthetic training data
            np.random.seed(42)
            n_samples = 1000

            # Generate features
            features_data = []
            for _ in range(n_samples):
                features_data.append(
                    {
                        # Basic stats
                        "home_points_per_game": np.random.normal(24, 4),
                        "away_points_per_game": np.random.normal(24, 4),
                        "home_points_allowed": np.random.normal(24, 4),
                        "away_points_allowed": np.random.normal(24, 4),
                        "home_yards_per_game": np.random.normal(350, 50),
                        "away_yards_per_game": np.random.normal(350, 50),
                        "home_yards_allowed": np.random.normal(350, 50),
                        "away_yards_allowed": np.random.normal(350, 50),
                        # Advanced stats
                        "home_pass_yards_per_game": np.random.normal(240, 40),
                        "away_pass_yards_per_game": np.random.normal(240, 40),
                        "home_rush_yards_per_game": np.random.normal(110, 30),
                        "away_rush_yards_per_game": np.random.normal(110, 30),
                        "home_third_down_conv": np.random.beta(4, 6),
                        "away_third_down_conv": np.random.beta(4, 6),
                        "home_fourth_down_conv": np.random.beta(3, 7),
                        "away_fourth_down_conv": np.random.beta(3, 7),
                        "home_time_of_possession": np.random.normal(30, 2),
                        "away_time_of_possession": np.random.normal(30, 2),
                        "home_turnover_margin": np.random.normal(0, 1),
                        "away_turnover_margin": np.random.normal(0, 1),
                        # Situational stats
                        "home_red_zone_scoring": np.random.beta(5, 3),
                        "away_red_zone_scoring": np.random.beta(5, 3),
                        "home_first_half_points": np.random.normal(12, 2),
                        "away_first_half_points": np.random.normal(12, 2),
                        "home_second_half_points": np.random.normal(12, 2),
                        "away_second_half_points": np.random.normal(12, 2),
                        "home_qb_rating": np.random.normal(90, 10),
                        "away_qb_rating": np.random.normal(90, 10),
                        # Game context
                        "is_division_game": np.random.binomial(1, 0.3),
                        "is_primetime": np.random.binomial(1, 0.2),
                        "days_rest": np.random.choice([6, 7, 8, 9, 10, 11, 12, 13, 14]),
                        "playoff_implications": np.random.binomial(1, 0.4),
                        # Environmental
                        "temperature": np.random.normal(65, 15),
                        "wind_speed": np.random.exponential(8),
                        "is_dome": np.random.binomial(1, 0.25),
                        "precipitation": np.random.beta(2, 8),
                        "altitude": np.random.choice(
                            [0, 5280], p=[0.97, 0.03]
                        ),  # Denver's altitude
                        # Target variable
                        "total_points": None,  # Will be calculated below
                    }
                )

            # Calculate total points based on features
            for game in features_data:
                base_total = game["home_points_per_game"] + game["away_points_per_game"]

                # Adjust for various factors
                adjustments = [
                    -3 if game["temperature"] < 32 else 0,  # Cold weather
                    -2 if game["wind_speed"] > 15 else 0,  # High winds
                    2 if game["is_dome"] else 0,  # Indoor game
                    -1 if game["precipitation"] > 0.5 else 0,  # Rain/snow
                    1 if game["is_primetime"] else 0,  # Primetime game
                    -2 if abs(game["days_rest"] - 7) > 3 else 0,  # Unusual rest
                ]

                game["total_points"] = (
                    base_total + sum(adjustments) + np.random.normal(0, 3)
                )

            # Convert to DataFrame
            training_data = pd.DataFrame(features_data)

            # Train each model
            for name, model in self.models.items():
                try:
                    if hasattr(model, "train"):
                        model.train(training_data)
                        logging.info(f"Successfully trained {name} model")
                except Exception as e:
                    logging.error(f"Failed to train {name} model: {str(e)}")

        except Exception as e:
            logging.error(f"Error generating training data: {str(e)}")
            raise

    def initialize_models(self, training_data):
        """Initialize and train all prediction models"""
        try:
            # Initialize each model
            for name, model in self.models.items():
                try:
                    if hasattr(model, "train"):
                        model.train(training_data)
                        logging.info(f"Successfully trained {name} model")
                except Exception as e:
                    logging.error(f"Failed to train {name} model: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Error initializing models: {str(e)}")
            raise

    def get_consensus_prediction(self, predictions):
        """Calculate consensus prediction from all models"""
        valid_predictions = []
        total_confidence = 0

        for pred in predictions:
            if 20 <= pred["predicted_total"] <= 80:
                valid_predictions.append(
                    {"total": pred["predicted_total"], "confidence": pred["confidence"]}
                )
                total_confidence += pred["confidence"]

        if not valid_predictions:
            return None

        weighted_total = sum(p["total"] * p["confidence"] for p in valid_predictions)
        return weighted_total / total_confidence if total_confidence > 0 else None

    def get_consensus_rating(
        self,
        model_predictions: List[float],
        confidences: List[float],
        vegas_line: float,
    ) -> Tuple[str, float, str]:
        """
        Calculate consensus rating based on model predictions and Vegas line
        Returns: (rating, consensus_total, explanation)
        """
        consensus_total = self.get_consensus_prediction(
            [
                {"predicted_total": pred, "confidence": conf}
                for pred, conf in zip(model_predictions, confidences)
            ]
        )

        # Calculate model agreement
        max_diff = max([abs(p - consensus_total) for p in model_predictions])
        vegas_diff = abs(consensus_total - vegas_line)

        # Define rating thresholds
        if max_diff < 3 and vegas_diff > 4:
            rating = "ELITE"
            explanation = "High model agreement with significant Vegas difference"
        elif max_diff < 4 and vegas_diff > 3:
            rating = "GREAT"
            explanation = "Strong model agreement with notable Vegas difference"
        elif max_diff < 5 and vegas_diff > 2:
            rating = "GOOD"
            explanation = "Good model agreement with moderate Vegas difference"
        elif max_diff < 6:
            rating = "DECENT"
            explanation = "Moderate model agreement"
        else:
            rating = "PASS"
            explanation = "Low model agreement"

        return rating, consensus_total, explanation

    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions using all available models

        Args:
            game_data: Dictionary containing game information

        Returns:
            Dictionary containing predictions and consensus
        """
        try:
            predictions = {}
            confidences = {}

            # Get predictions from all models
            for name, model in self.models.items():
                try:
                    pred = model.predict(game_data)
                    predictions[name] = pred["predicted_total"]
                    confidences[name] = pred.get(
                        "confidence", self.confidence_thresholds["low"]
                    )
                except Exception as e:
                    logging.warning(f"{name.title()} model prediction failed: {str(e)}")
                    # Use model-specific fallback values
                    predictions[name] = self._get_fallback_prediction(name)
                    confidences[name] = self.confidence_thresholds["low"]

            # Calculate consensus
            weights = np.array([confidences[model] for model in predictions.keys()])
            weights = weights / weights.sum()  # Normalize weights

            values = np.array([predictions[model] for model in predictions.keys()])
            consensus_total = np.sum(weights * values)

            # Determine prediction confidence level
            all_high = all(
                conf >= self.confidence_thresholds["medium"]
                for conf in confidences.values()
            )
            any_very_high = any(
                conf >= self.confidence_thresholds["high"]
                for conf in confidences.values()
            )

            # Calculate model agreement
            std_dev = np.std(list(predictions.values()))
            max_diff = max(predictions.values()) - min(predictions.values())

            # Determine rating based on confidence and agreement
            if all_high and any_very_high and std_dev < 3:
                rating = "HIGH"
            elif all_high and std_dev < 5:
                rating = "MEDIUM"
            else:
                rating = "PASS"

            # Generate detailed explanation
            explanation = self._generate_explanation(
                predictions,
                confidences,
                std_dev,
                max_diff,
                game_data.get("vegas_total", 0),
            )

            return {
                "model_predictions": predictions,
                "model_confidences": confidences,
                "consensus": {
                    "total": consensus_total,
                    "rating": rating,
                    "explanation": explanation,
                    "high_confidence": all_high and any_very_high and std_dev < 3,
                },
            }

        except Exception as e:
            logging.error(f"Error in TotalPredictor.predict: {str(e)}")
            return self._get_error_prediction(str(e))

    def _get_fallback_prediction(self, model_name: str) -> float:
        """Get fallback prediction value for a specific model"""
        fallbacks = {
            "advanced": 44.0,  # Conservative baseline
            "vedic": 47.0,  # Historical average
            "ml": 45.0,  # ML model baseline
            "stats": 43.0,  # Sports analytics baseline
            "neuro": 46.0,  # Neural net baseline
        }
        return fallbacks.get(model_name, 44.0)

    def _get_error_prediction(self, error_msg: str) -> Dict[str, Any]:
        """Generate error prediction response"""
        return {
            "model_predictions": {
                model: self._get_fallback_prediction(model)
                for model in self.models.keys()
            },
            "model_confidences": {
                model: self.confidence_thresholds["low"] for model in self.models.keys()
            },
            "consensus": {
                "total": 44.0,
                "rating": "ERROR",
                "explanation": f"Prediction failed: {error_msg}",
                "high_confidence": False,
            },
        }

    def _generate_explanation(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float],
        std_dev: float,
        max_diff: float,
        vegas_total: float,
    ) -> str:
        """Generate detailed prediction explanation"""
        if std_dev > 7:
            return "Low model agreement - high variance in predictions"

        if max_diff > 10:
            return "Significant disagreement between models"

        all_high = all(
            conf >= self.confidence_thresholds["medium"]
            for conf in confidences.values()
        )
        any_very_high = any(
            conf >= self.confidence_thresholds["high"] for conf in confidences.values()
        )

        if all_high and any_very_high:
            if std_dev < 3:
                return "Strong consensus with high confidence across all models"
            else:
                return "High confidence but moderate variance between models"

        if abs(np.mean(list(predictions.values())) - vegas_total) < 2:
            return "Models align closely with Vegas line"

        direction = (
            "OVER" if np.mean(list(predictions.values())) > vegas_total else "UNDER"
        )
        return f"Models showing {direction} tendency with moderate confidence"

    def predict_game(self, game_data):
        """
        Get predictions from all models for a single game

        Args:
            game_data: Dictionary containing game information

        Returns:
            Dictionary containing aggregated predictions and explanations
        """
        predictions = []
        explanations = []

        for model in self.models:
            try:
                pred = model.predict(game_data)
                predictions.append(pred)
                explanations.append(pred["details"]["explanation"])
            except Exception as e:
                logging.error(
                    f"Error getting prediction from model {model.__class__.__name__}: {str(e)}"
                )

        if not predictions:
            return self._get_default_prediction()

        # Aggregate predictions
        aggregated = {
            "total": self._aggregate_total_predictions(
                [p["total"] for p in predictions]
            ),
            "spread": self._aggregate_spread_predictions(
                [p["spread"] for p in predictions]
            ),
            "moneyline": self._aggregate_moneyline_predictions(
                [p["moneyline"] for p in predictions]
            ),
            "details": {
                "model_explanations": explanations,
                "consensus_explanation": self._generate_consensus_explanation(
                    predictions
                ),
            },
        }

        return aggregated

    def _aggregate_total_predictions(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate total predictions from multiple models"""
        total_points = self.weighted_average(
            [p["points"] for p in predictions], [p["confidence"] for p in predictions]
        )

        # Get the most common prediction with highest confidence
        over_under_votes = {"OVER": 0, "UNDER": 0}
        for pred in predictions:
            over_under_votes[pred["prediction"]] += pred["confidence"]

        final_prediction = max(over_under_votes.items(), key=lambda x: x[1])[0]

        return {
            "points": float(total_points),
            "prediction": final_prediction,
            "line": predictions[0]["line"],  # Use same line for consistency
            "confidence": min(
                0.95, sum(p["confidence"] for p in predictions) / len(predictions)
            ),
        }

    def _aggregate_spread_predictions(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate spread predictions from multiple models"""
        spread_points = self.weighted_average(
            [p["points"] for p in predictions], [p["confidence"] for p in predictions]
        )

        # Get the most common prediction with highest confidence
        spread_votes = {"HOME": 0, "AWAY": 0}
        for pred in predictions:
            spread_votes[pred["prediction"]] += pred["confidence"]

        final_prediction = max(spread_votes.items(), key=lambda x: x[1])[0]

        return {
            "points": float(spread_points),
            "prediction": final_prediction,
            "line": predictions[0]["line"],
            "confidence": min(
                0.95, sum(p["confidence"] for p in predictions) / len(predictions)
            ),
        }

    def _aggregate_moneyline_predictions(
        self, predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate moneyline predictions from multiple models"""
        # Average win probabilities
        avg_home_win_prob = sum(p["home_win_prob"] for p in predictions) / len(
            predictions
        )

        return {
            "prediction": "HOME" if avg_home_win_prob > 0.5 else "AWAY",
            "home_win_prob": float(avg_home_win_prob),
            "confidence": min(
                0.95, sum(p["confidence"] for p in predictions) / len(predictions)
            ),
        }

    def _generate_consensus_explanation(self, predictions: List[Dict[str, Any]]) -> str:
        """Generate consensus explanation from all model predictions"""
        total_points = self.weighted_average(
            [p["total"]["points"] for p in predictions],
            [p["total"]["confidence"] for p in predictions],
        )

        spread_points = self.weighted_average(
            [p["spread"]["points"] for p in predictions],
            [p["spread"]["confidence"] for p in predictions],
        )

        home_win_prob = sum(p["moneyline"]["home_win_prob"] for p in predictions) / len(
            predictions
        )

        return (
            f"Consensus Prediction:\n"
            f"Total Points: {total_points:.1f} "
            f"({'OVER' if total_points > predictions[0]['total']['line'] else 'UNDER'} "
            f"{predictions[0]['total']['line']})\n"
            f"Spread: {abs(spread_points):.1f} points "
            f"({'HOME' if spread_points > predictions[0]['spread']['line'] else 'AWAY'} favored)\n"
            f"Win Probability: {home_win_prob:.1%} for {'HOME' if home_win_prob > 0.5 else 'AWAY'} team"
        )

    def _get_default_prediction(self) -> Dict[str, Any]:
        """Return default prediction when all models fail"""
        return {
            "total": {
                "points": 47.5,
                "prediction": "UNDER",
                "line": 47.5,
                "confidence": 0.5,
            },
            "spread": {
                "points": 0.0,
                "prediction": "HOME",
                "line": 0.0,
                "confidence": 0.5,
            },
            "moneyline": {
                "prediction": "HOME",
                "home_win_prob": 0.5,
                "confidence": 0.5,
            },
            "details": {
                "model_explanations": ["No valid predictions available"],
                "consensus_explanation": "Unable to generate predictions",
            },
        }

    def weighted_average(self, values: List[float], weights: List[float]) -> float:
        """Calculate weighted average of values"""
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)


class TotalPredictionModel:
    """Standard statistical model for predicting NFL totals"""

    def predict(self, game_data):
        """Predict total points for a game"""
        # Get team stats
        home_stats = game_data["home_stats"]
        away_stats = game_data["away_stats"]

        # Calculate base total using points scored and allowed
        home_expected = (
            home_stats["points_per_game"] + away_stats["points_allowed"]
        ) / 2
        away_expected = (
            away_stats["points_per_game"] + home_stats["points_allowed"]
        ) / 2
        base_total = home_expected + away_expected

        # Adjust for efficiency metrics
        efficiency_factor = (
            home_stats["defensive_efficiency"] + away_stats["defensive_efficiency"]
        ) / 2
        base_total *= efficiency_factor

        # Adjust for weather if outdoor game
        if not game_data["is_dome"]:
            if game_data["temperature"] < 40:
                base_total *= 0.95  # Cold weather reduces scoring
            if game_data["wind_speed"] > 15:
                base_total *= 0.93  # High winds reduce scoring
            if game_data["precipitation_chance"] > 0.5:
                base_total *= 0.97  # Rain/snow reduces scoring

        # Adjust for primetime games
        if game_data["is_primetime"]:
            base_total *= 0.98  # Primetime games tend to be lower scoring

        # Calculate confidence based on various factors
        confidence = 0.7  # Base confidence
        if abs(home_stats["avg_total_points"] - away_stats["avg_total_points"]) < 3:
            confidence += 0.1  # Teams have similar scoring patterns
        if game_data.get("matchup_historical_avg"):
            confidence += 0.1  # Historical data available

        return {
            "predicted_total": base_total,
            "confidence": min(confidence, 1.0),  # Cap confidence at 1.0
        }


class AstroTotalModel:
    """Enhanced astrological model for NFL totals"""

    def predict(self, game_data):
        """Predict total points using comprehensive astrological factors"""
        try:
            # Parse game date
            game_date = datetime.strptime(game_data["date"], "%Y-%m-%d %H:%M")

            # Get planetary positions
            positions = get_planet_positions(game_date)

            # Calculate various astrological factors
            astro_advantage = calculate_astrological_advantage(game_date, game_data)
            yoga_strength = calculate_yogas(positions)
            hour_strength = calculate_planetary_hour_strength(game_date)
            mars_jupiter = calculate_mars_jupiter_relationship(positions)
            shadow_influence = calculate_shadow_planet_influence(game_date)

            # Base prediction using team averages
            base_total = (
                game_data["home_stats"]["avg_total_points"]
                + game_data["away_stats"]["avg_total_points"]
            ) / 2

            # Adjust based on astrological factors
            adjustments = {
                "astro_advantage": (astro_advantage - 0.5) * 10,  # -5 to +5 points
                "yoga_strength": (yoga_strength - 0.5) * 8,  # -4 to +4 points
                "hour_strength": (hour_strength - 0.5) * 6,  # -3 to +3 points
                "mars_jupiter": (mars_jupiter - 0.5) * 4,  # -2 to +2 points
                "shadow_influence": (shadow_influence - 0.5) * 4,  # -2 to +2 points
            }

            # Apply adjustments
            predicted_total = base_total
            for adj in adjustments.values():
                predicted_total += adj

            # Calculate confidence based on astrological strength
            confidence_factors = {
                "astro_advantage": astro_advantage * 0.3,
                "yoga_strength": yoga_strength * 0.2,
                "hour_strength": hour_strength * 0.2,
                "mars_jupiter": mars_jupiter * 0.15,
                "shadow_influence": shadow_influence * 0.15,
            }

            confidence = sum(confidence_factors.values())

            # Additional confidence adjustments
            if game_data.get("is_primetime", False):
                confidence *= 1.1  # Boost confidence for primetime games
            if game_data.get("playoff_implications", 0) > 0.7:
                confidence *= 1.05  # Boost for important games

            # Ensure prediction is within realistic bounds
            predicted_total = max(30, min(75, predicted_total))
            confidence = max(0.3, min(1.0, confidence))

            return {
                "predicted_total": predicted_total,
                "confidence": confidence,
                "factors": {
                    "astrological_advantage": astro_advantage,
                    "yoga_strength": yoga_strength,
                    "hour_strength": hour_strength,
                    "mars_jupiter_harmony": mars_jupiter,
                    "shadow_influence": shadow_influence,
                },
            }

        except Exception as e:
            print(f"Astrological prediction error: {str(e)}")
            # Return a neutral prediction with low confidence
            return {"predicted_total": 47.0, "confidence": 0.3}  # League average


def main():
    """Run predictions for sample games"""
    predictor = TotalPredictor()

    # Super Bowl LVIII
    prediction = predictor.predict_game(
        {
            "home_stats": {"avg_total_points": 25},
            "away_stats": {"avg_total_points": 20},
            "date": "2024-02-11 18:30",
            "is_primetime": True,
        }
    )

    print("\nNFL TOTAL PREDICTION ANALYSIS")
    print("=" * 50)

    if prediction:
        print(f"Game: Chiefs @ 49ers")
        print(f"Date: 2024-02-11 18:30")
        print(f"Venue: Levi's Stadium")
        print("\nPREDICTIONS")
        print("-" * 50)
        print(f"Total Points: {prediction['total']['points']:.1f}")
        print(
            f"Over/Under: {prediction['total']['prediction']} {prediction['total']['line']}"
        )
        print(f"Confidence: {prediction['total']['confidence']:.1%}")
        print(
            f"\nSpread: {abs(prediction['spread']['points']):.1f} ({prediction['spread']['prediction']})"
        )
        print(
            f"Moneyline: {prediction['moneyline']['prediction']} ({prediction['moneyline']['home_win_prob']:.1%})"
        )
        print(f"\nExplanation:")
        print(prediction["details"]["consensus_explanation"])


if __name__ == "__main__":
    main()
