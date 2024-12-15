"""Temporal pattern analysis and historical similarity matching for NFL predictions."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class GameContext:
    """Context information for a game."""

    date: datetime
    home_team: str
    away_team: str
    home_record: Tuple[int, int]
    away_record: Tuple[int, int]
    home_last_5: List[int]  # 1 for win, 0 for loss
    away_last_5: List[int]
    home_points_last_5: List[float]
    away_points_last_5: List[float]
    week_number: int
    is_division_game: bool
    is_conference_game: bool
    home_rest_days: int
    away_rest_days: int


class TemporalPatternAnalyzer:
    """Analyze temporal patterns and find historical similarities."""

    def __init__(self, lookback_years: int = 5):
        """Initialize the analyzer."""
        self.lookback_years = lookback_years
        self.scaler = StandardScaler()
        self.historical_games: List[GameContext] = []
        self.team_form: Dict[str, Dict[str, float]] = {}
        self.seasonal_patterns: Dict[
            str, Dict[int, float]
        ] = {}  # team -> week -> performance
        self.monthly_patterns: Dict[
            str, Dict[int, float]
        ] = {}  # team -> month -> performance

    def _calculate_time_weight(
        self, game_date: datetime, reference_date: datetime
    ) -> float:
        """Calculate exponential decay weight based on time difference."""
        days_diff = (reference_date - game_date).days
        half_life = 365  # One year half-life
        return np.exp(-np.log(2) * days_diff / half_life)

    def _get_team_vector(
        self, context: GameContext, team: str, is_home: bool
    ) -> np.ndarray:
        """Create a feature vector for a team's current state."""
        record = context.home_record if is_home else context.away_record
        last_5 = context.home_last_5 if is_home else context.away_last_5
        points_last_5 = (
            context.home_points_last_5 if is_home else context.away_points_last_5
        )
        rest_days = context.home_rest_days if is_home else context.away_rest_days

        win_pct = (
            record[0] / (record[0] + record[1]) if record[0] + record[1] > 0 else 0.5
        )
        recent_form = np.mean(last_5) if last_5 else 0.5
        points_trend = np.mean(points_last_5) if points_last_5 else 0

        # Get seasonal and monthly patterns
        week_pattern = self.seasonal_patterns.get(team, {}).get(
            context.week_number, 0.5
        )
        month = context.date.month
        month_pattern = self.monthly_patterns.get(team, {}).get(month, 0.5)

        return np.array(
            [
                win_pct,
                recent_form,
                points_trend,
                week_pattern,
                month_pattern,
                rest_days / 14,  # Normalize rest days
                1 if is_home else 0,
                1 if context.is_division_game else 0,
                1 if context.is_conference_game else 0,
            ]
        )

    def _calculate_similarity(
        self, context1: GameContext, context2: GameContext
    ) -> float:
        """Calculate similarity score between two game contexts."""
        # Create vectors for both games
        home1 = self._get_team_vector(context1, context1.home_team, True)
        away1 = self._get_team_vector(context1, context1.away_team, False)
        home2 = self._get_team_vector(context2, context2.home_team, True)
        away2 = self._get_team_vector(context2, context2.away_team, False)

        # Combine team vectors
        game1 = np.concatenate([home1, away1])
        game2 = np.concatenate([home2, away2])

        # Calculate cosine similarity
        return 1 - cosine(game1, game2)

    def update_patterns(self, games_df: pd.DataFrame):
        """Update temporal patterns with new game data."""
        games_df = games_df.sort_values("date")

        # Clear existing patterns
        self.team_form.clear()
        self.seasonal_patterns.clear()
        self.monthly_patterns.clear()
        self.historical_games.clear()

        # Process each game
        for _, game in games_df.iterrows():
            game_date = pd.to_datetime(game["date"])

            # Create game context
            context = self._create_game_context(game, games_df)
            self.historical_games.append(context)

            # Update team form
            self._update_team_form(game)

            # Update seasonal patterns
            self._update_seasonal_patterns(game)

            # Update monthly patterns
            self._update_monthly_patterns(game)

    def _create_game_context(
        self, game: pd.Series, games_df: pd.DataFrame
    ) -> GameContext:
        """Create a GameContext object for a game."""
        game_date = pd.to_datetime(game["date"])

        # Get team records before this game
        prior_games = games_df[pd.to_datetime(games_df["date"]) < game_date]

        home_record = self._get_team_record(prior_games, game["home_team"])
        away_record = self._get_team_record(prior_games, game["away_team"])

        # Get last 5 games results
        home_last_5 = self._get_last_n_results(prior_games, game["home_team"], 5)
        away_last_5 = self._get_last_n_results(prior_games, game["away_team"], 5)

        # Get last 5 games points
        home_points_last_5 = self._get_last_n_points(prior_games, game["home_team"], 5)
        away_points_last_5 = self._get_last_n_points(prior_games, game["away_team"], 5)

        # Calculate rest days
        home_rest = self._calculate_rest_days(prior_games, game["home_team"], game_date)
        away_rest = self._calculate_rest_days(prior_games, game["away_team"], game_date)

        return GameContext(
            date=game_date,
            home_team=game["home_team"],
            away_team=game["away_team"],
            home_record=home_record,
            away_record=away_record,
            home_last_5=home_last_5,
            away_last_5=away_last_5,
            home_points_last_5=home_points_last_5,
            away_points_last_5=away_points_last_5,
            week_number=game["week"],
            is_division_game=game["is_division_game"],
            is_conference_game=game["is_conference_game"],
            home_rest_days=home_rest,
            away_rest_days=away_rest,
        )

    def _update_team_form(self, game: pd.Series):
        """Update team form with exponential decay weights."""
        game_date = pd.to_datetime(game["date"])

        for team in [game["home_team"], game["away_team"]]:
            if team not in self.team_form:
                self.team_form[team] = {"recent_performance": 0.5, "games_played": 0}

            # Calculate performance score (win = 1, loss = 0)
            is_home = team == game["home_team"]
            won = (
                (game["home_score"] > game["away_score"])
                if is_home
                else (game["away_score"] > game["home_score"])
            )
            performance = 1.0 if won else 0.0

            # Update with exponential decay
            weight = self._calculate_time_weight(game_date, datetime.now())
            current = self.team_form[team]["recent_performance"]
            games = self.team_form[team]["games_played"]

            self.team_form[team]["recent_performance"] = (
                current * games + performance * weight
            ) / (games + weight)
            self.team_form[team]["games_played"] += 1

    def _update_seasonal_patterns(self, game: pd.Series):
        """Update week-by-week patterns for teams."""
        for team in [game["home_team"], game["away_team"]]:
            if team not in self.seasonal_patterns:
                self.seasonal_patterns[team] = {}

            week = game["week"]
            is_home = team == game["home_team"]
            won = (
                (game["home_score"] > game["away_score"])
                if is_home
                else (game["away_score"] > game["home_score"])
            )

            # Update week pattern
            if week not in self.seasonal_patterns[team]:
                self.seasonal_patterns[team][week] = won
            else:
                self.seasonal_patterns[team][week] = (
                    0.7 * self.seasonal_patterns[team][week] + 0.3 * won
                )

    def _update_monthly_patterns(self, game: pd.Series):
        """Update month-by-month patterns for teams."""
        game_date = pd.to_datetime(game["date"])

        for team in [game["home_team"], game["away_team"]]:
            if team not in self.monthly_patterns:
                self.monthly_patterns[team] = {}

            month = game_date.month
            is_home = team == game["home_team"]
            won = (
                (game["home_score"] > game["away_score"])
                if is_home
                else (game["away_score"] > game["home_score"])
            )

            # Update month pattern
            if month not in self.monthly_patterns[team]:
                self.monthly_patterns[team][month] = won
            else:
                self.monthly_patterns[team][month] = (
                    0.7 * self.monthly_patterns[team][month] + 0.3 * won
                )

    def get_similar_games(
        self, context: GameContext, n_matches: int = 10
    ) -> List[Tuple[GameContext, float]]:
        """Find most similar historical games."""
        similarities = []

        for hist_game in self.historical_games:
            # Skip future games
            if hist_game.date >= context.date:
                continue

            # Skip games more than lookback_years old
            if (context.date - hist_game.date).days > self.lookback_years * 365:
                continue

            sim_score = self._calculate_similarity(context, hist_game)
            similarities.append((hist_game, sim_score))

        # Sort by similarity and return top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_matches]

    def get_temporal_features(self, context: GameContext) -> Dict[str, float]:
        """Get temporal features for a game context."""
        # Get team form
        home_form = self.team_form.get(context.home_team, {"recent_performance": 0.5})[
            "recent_performance"
        ]
        away_form = self.team_form.get(context.away_team, {"recent_performance": 0.5})[
            "recent_performance"
        ]

        # Get seasonal patterns
        home_week_pattern = self.seasonal_patterns.get(context.home_team, {}).get(
            context.week_number, 0.5
        )
        away_week_pattern = self.seasonal_patterns.get(context.away_team, {}).get(
            context.week_number, 0.5
        )

        # Get monthly patterns
        home_month_pattern = self.monthly_patterns.get(context.home_team, {}).get(
            context.date.month, 0.5
        )
        away_month_pattern = self.monthly_patterns.get(context.away_team, {}).get(
            context.date.month, 0.5
        )

        # Find similar games
        similar_games = self.get_similar_games(context, n_matches=10)

        # Calculate historical similarity features
        if similar_games:
            similar_home_wins = sum(
                1
                for game, _ in similar_games
                if game.home_team == context.home_team
                and len(game.home_last_5) > 0
                and game.home_last_5[-1] == 1
            )
            historical_win_rate = similar_home_wins / len(similar_games)
            avg_similarity = np.mean([sim for _, sim in similar_games])
        else:
            historical_win_rate = 0.5
            avg_similarity = 0.0

        return {
            "home_recent_form": home_form,
            "away_recent_form": away_form,
            "home_week_pattern": home_week_pattern,
            "away_week_pattern": away_week_pattern,
            "home_month_pattern": home_month_pattern,
            "away_month_pattern": away_month_pattern,
            "historical_win_rate": historical_win_rate,
            "historical_similarity": avg_similarity,
        }

    @staticmethod
    def _get_team_record(games_df: pd.DataFrame, team: str) -> Tuple[int, int]:
        """Get team's win-loss record."""
        wins = len(
            games_df[
                (
                    (games_df["home_team"] == team)
                    & (games_df["home_score"] > games_df["away_score"])
                )
                | (
                    (games_df["away_team"] == team)
                    & (games_df["away_score"] > games_df["home_score"])
                )
            ]
        )
        losses = len(
            games_df[
                (
                    (games_df["home_team"] == team)
                    & (games_df["home_score"] < games_df["away_score"])
                )
                | (
                    (games_df["away_team"] == team)
                    & (games_df["away_score"] < games_df["home_score"])
                )
            ]
        )
        return (wins, losses)

    @staticmethod
    def _get_last_n_results(games_df: pd.DataFrame, team: str, n: int) -> List[int]:
        """Get team's last n game results."""
        team_games = games_df[
            (games_df["home_team"] == team) | (games_df["away_team"] == team)
        ].copy()
        team_games["won"] = (
            (team_games["home_team"] == team)
            & (team_games["home_score"] > team_games["away_score"])
        ) | (
            (team_games["away_team"] == team)
            & (team_games["away_score"] > team_games["home_score"])
        )
        return team_games["won"].tail(n).tolist()

    @staticmethod
    def _get_last_n_points(games_df: pd.DataFrame, team: str, n: int) -> List[float]:
        """Get team's last n game points."""
        team_games = games_df[
            (games_df["home_team"] == team) | (games_df["away_team"] == team)
        ].copy()
        team_games["points"] = np.where(
            team_games["home_team"] == team,
            team_games["home_score"],
            team_games["away_score"],
        )
        return team_games["points"].tail(n).tolist()

    @staticmethod
    def _calculate_rest_days(
        games_df: pd.DataFrame, team: str, game_date: datetime
    ) -> int:
        """Calculate days since team's last game."""
        team_games = games_df[
            (games_df["home_team"] == team) | (games_df["away_team"] == team)
        ]
        if team_games.empty:
            return 7  # Default to a week if no previous games

        last_game = pd.to_datetime(team_games["date"].max())
        return (game_date - last_game).days
