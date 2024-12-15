"""
NFL Betting Analysis System
Implements advanced betting analysis, odds movement tracking,
and prediction models for betting outcomes
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib


class NFLBettingAnalysis:
    def __init__(self):
        self.logger = logging.getLogger("nfl_betting")
        self.model = None
        self.scaler = None

    def analyze_spread_performance(
        self, team_data: Dict, historical_data: Dict
    ) -> Dict:
        """Analyze team performance against the spread"""
        try:
            analysis = {
                "ats_record": self._calculate_ats_record(historical_data),
                "home_ats": self._calculate_home_ats(historical_data),
                "away_ats": self._calculate_away_ats(historical_data),
                "favorite_ats": self._calculate_favorite_ats(historical_data),
                "underdog_ats": self._calculate_underdog_ats(historical_data),
                "trends": self._identify_betting_trends(historical_data),
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing spread performance: {e}")
            raise

    def analyze_over_under(self, team_data: Dict, historical_data: Dict) -> Dict:
        """Analyze over/under performance"""
        try:
            analysis = {
                "over_under_record": self._calculate_ou_record(historical_data),
                "home_ou": self._calculate_home_ou(historical_data),
                "away_ou": self._calculate_away_ou(historical_data),
                "weather_impact": self._analyze_weather_impact(historical_data),
                "trends": self._identify_ou_trends(historical_data),
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing over/under: {e}")
            raise

    def analyze_line_movement(self, game_data: Dict, betting_data: Dict) -> Dict:
        """Analyze betting line movement and its implications"""
        try:
            analysis = {
                "opening_line": betting_data.get("opening_line"),
                "current_line": betting_data.get("current_line"),
                "movement": self._calculate_line_movement(betting_data),
                "sharp_money": self._analyze_sharp_money(betting_data),
                "public_betting": self._analyze_public_betting(betting_data),
                "reverse_line_movement": self._check_reverse_movement(betting_data),
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing line movement: {e}")
            raise

    def predict_betting_outcome(
        self, game_data: Dict, model_type: str = "spread"
    ) -> Dict:
        """Predict betting outcomes using ML models"""
        try:
            if not self.model:
                self._load_model(model_type)

            # Prepare features
            features = self._prepare_features(game_data)

            # Make prediction
            prediction = self.model.predict_proba(features)[0]

            return {
                "prediction": prediction,
                "confidence": self._calculate_confidence(prediction),
                "factors": self._identify_key_factors(features, prediction),
            }

        except Exception as e:
            self.logger.error(f"Error predicting betting outcome: {e}")
            raise

    def train_betting_model(self, historical_data: Dict, model_type: str = "spread"):
        """Train ML model for betting predictions"""
        try:
            # Prepare training data
            X, y = self._prepare_training_data(historical_data, model_type)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }

            # Save model
            self._save_model(model_type)

            return metrics

        except Exception as e:
            self.logger.error(f"Error training betting model: {e}")
            raise

    def _calculate_ats_record(self, historical_data: Dict) -> Dict:
        """Calculate against the spread record"""
        try:
            games = historical_data.get("games", [])
            wins = sum(1 for game in games if game.get("ats_result") == "W")
            losses = sum(1 for game in games if game.get("ats_result") == "L")
            pushes = sum(1 for game in games if game.get("ats_result") == "P")

            return {
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_percentage": wins / (wins + losses) if wins + losses > 0 else 0,
            }

        except Exception as e:
            self.logger.error(f"Error calculating ATS record: {e}")
            raise

    def _analyze_sharp_money(self, betting_data: Dict) -> Dict:
        """Analyze sharp money movement"""
        try:
            line_moves = betting_data.get("line_moves", [])
            bet_percentages = betting_data.get("bet_percentages", {})
            money_percentages = betting_data.get("money_percentages", {})

            # Calculate reverse line movement indicator
            reverse_movement = (
                bet_percentages.get("home", 0) > 60 and line_moves[-1] < line_moves[0]
            ) or (
                bet_percentages.get("away", 0) > 60 and line_moves[-1] > line_moves[0]
            )

            return {
                "sharp_side": self._determine_sharp_side(
                    bet_percentages, money_percentages
                ),
                "reverse_movement": reverse_movement,
                "steam_moves": self._identify_steam_moves(line_moves),
            }

        except Exception as e:
            self.logger.error(f"Error analyzing sharp money: {e}")
            raise

    def _determine_sharp_side(
        self, bet_percentages: Dict, money_percentages: Dict
    ) -> str:
        """Determine which side is receiving sharp money"""
        try:
            # Look for ticket/money percentage disparities
            home_tickets = bet_percentages.get("home", 0)
            home_money = money_percentages.get("home", 0)

            if home_tickets < 40 and home_money > 60:
                return "home"
            elif home_tickets > 60 and home_money < 40:
                return "away"
            else:
                return "unclear"

        except Exception as e:
            self.logger.error(f"Error determining sharp side: {e}")
            raise

    def _identify_steam_moves(self, line_moves: List[float]) -> List[Dict]:
        """Identify steam moves in line movement"""
        try:
            steam_moves = []
            threshold = 0.5  # Points moved within 5 minutes

            for i in range(1, len(line_moves)):
                movement = abs(line_moves[i] - line_moves[i - 1])
                if movement >= threshold:
                    steam_moves.append(
                        {
                            "time": i,
                            "movement": movement,
                            "direction": "up"
                            if line_moves[i] > line_moves[i - 1]
                            else "down",
                        }
                    )

            return steam_moves

        except Exception as e:
            self.logger.error(f"Error identifying steam moves: {e}")
            raise

    def _save_model(self, model_type: str):
        """Save trained model to file"""
        try:
            model_dir = Path(__file__).parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / f"betting_model_{model_type}.joblib"
            joblib.dump(self.model, model_path)

            self.logger.info(f"Saved model to {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def _load_model(self, model_type: str):
        """Load trained model from file"""
        try:
            model_path = (
                Path(__file__).parent / "models" / f"betting_model_{model_type}.joblib"
            )

            if model_path.exists():
                self.model = joblib.load(model_path)
                self.logger.info(f"Loaded model from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    analyzer = NFLBettingAnalysis()

    # Example betting data
    betting_data = {
        "line_moves": [3.0, 3.5, 3.0, 2.5, 3.0],
        "bet_percentages": {"home": 40, "away": 60},
        "money_percentages": {"home": 65, "away": 35},
    }

    # Analyze sharp money
    analysis = analyzer.analyze_line_movement(game_data={}, betting_data=betting_data)
    print(analysis)
