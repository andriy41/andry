"""
NFL Game Visualization System
Creates interactive visualizations for game analysis, player performance,
and betting insights using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path


class NFLGameVisualizer:
    def __init__(self):
        self.logger = logging.getLogger("nfl_visualizer")
        self.colors = {
            "primary": "#013369",  # NFL Blue
            "secondary": "#D50A0A",  # NFL Red
            "success": "#00B140",  # Success Green
            "warning": "#FFB612",  # Warning Yellow
            "background": "#FFFFFF",  # White
            "text": "#000000",  # Black
        }

    def create_game_dashboard(
        self, game_data: Dict, home_team: str, away_team: str
    ) -> go.Figure:
        """Create comprehensive game analysis dashboard"""
        try:
            # Create subplot layout
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    "Score Timeline",
                    "Team Statistics Comparison",
                    "Win Probability",
                    "Key Player Performance",
                    "Drive Success Rate",
                    "Field Position Heat Map",
                ),
            )

            # Add score timeline
            self._add_score_timeline(fig, game_data, 1, 1)

            # Add team stats comparison
            self._add_team_stats_comparison(fig, game_data, 1, 2)

            # Add win probability chart
            self._add_win_probability(fig, game_data, 2, 1)

            # Add player performance
            self._add_player_performance(fig, game_data, 2, 2)

            # Add drive success
            self._add_drive_success(fig, game_data, 3, 1)

            # Add field position
            self._add_field_position(fig, game_data, 3, 2)

            # Update layout
            fig.update_layout(
                height=1200,
                width=1600,
                showlegend=True,
                title_text=f"{away_team} @ {home_team}",
                paper_bgcolor=self.colors["background"],
                plot_bgcolor=self.colors["background"],
                font=dict(color=self.colors["text"]),
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error creating game dashboard: {e}")
            raise

    def create_betting_dashboard(
        self, betting_data: Dict, historical_data: Dict
    ) -> go.Figure:
        """Create betting analysis dashboard"""
        try:
            # Create subplot layout
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Line Movement",
                    "Public Betting Distribution",
                    "Historical Performance vs Spread",
                    "Over/Under Analysis",
                ),
            )

            # Add line movement
            self._add_line_movement(fig, betting_data, 1, 1)

            # Add betting distribution
            self._add_betting_distribution(fig, betting_data, 1, 2)

            # Add historical spread performance
            self._add_spread_performance(fig, historical_data, 2, 1)

            # Add over/under analysis
            self._add_over_under_analysis(fig, historical_data, 2, 2)

            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                showlegend=True,
                title_text="Betting Analysis Dashboard",
                paper_bgcolor=self.colors["background"],
                plot_bgcolor=self.colors["background"],
                font=dict(color=self.colors["text"]),
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error creating betting dashboard: {e}")
            raise

    def create_player_dashboard(self, player_data: Dict, position: str) -> go.Figure:
        """Create player performance dashboard"""
        try:
            # Create subplot layout based on position
            if position.upper() == "QB":
                fig = self._create_qb_dashboard(player_data)
            elif position.upper() == "RB":
                fig = self._create_rb_dashboard(player_data)
            elif position.upper() in ["WR", "TE"]:
                fig = self._create_receiver_dashboard(player_data)
            elif position.upper() in ["DE", "DT", "LB", "CB", "S"]:
                fig = self._create_defense_dashboard(player_data)
            else:
                raise ValueError(f"Unsupported position: {position}")

            # Update layout
            fig.update_layout(
                height=1000,
                width=1400,
                showlegend=True,
                title_text=f"{player_data['name']} Performance Dashboard",
                paper_bgcolor=self.colors["background"],
                plot_bgcolor=self.colors["background"],
                font=dict(color=self.colors["text"]),
            )

            return fig

        except Exception as e:
            self.logger.error(f"Error creating player dashboard: {e}")
            raise

    def _create_qb_dashboard(self, player_data: Dict) -> go.Figure:
        """Create QB-specific dashboard"""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Passing Chart",
                "EPA per Play",
                "Pressure Performance",
                "Deep Ball Accuracy",
                "Red Zone Efficiency",
                "Decision Making",
            ),
        )

        # Add QB-specific visualizations
        self._add_passing_chart(fig, player_data, 1, 1)
        self._add_epa_chart(fig, player_data, 1, 2)
        self._add_pressure_chart(fig, player_data, 2, 1)
        self._add_deep_ball_chart(fig, player_data, 2, 2)
        self._add_red_zone_chart(fig, player_data, 3, 1)
        self._add_decision_chart(fig, player_data, 3, 2)

        return fig

    def save_visualization(
        self, fig: go.Figure, filename: str, output_dir: str = "visualizations"
    ):
        """Save visualization to HTML and PNG formats"""
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save HTML version
            html_path = output_path / f"{filename}.html"
            fig.write_html(str(html_path))

            # Save PNG version
            png_path = output_path / f"{filename}.png"
            fig.write_image(str(png_path))

            self.logger.info(f"Saved visualization to {html_path} and {png_path}")

        except Exception as e:
            self.logger.error(f"Error saving visualization: {e}")
            raise

    def _add_score_timeline(self, fig: go.Figure, game_data: Dict, row: int, col: int):
        """Add score timeline visualization"""
        try:
            # Extract scoring data
            times = game_data.get("scoring_times", [])
            home_scores = game_data.get("home_scores", [])
            away_scores = game_data.get("away_scores", [])

            # Create traces
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=home_scores,
                    name="Home Team",
                    line=dict(color=self.colors["primary"]),
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=away_scores,
                    name="Away Team",
                    line=dict(color=self.colors["secondary"]),
                ),
                row=row,
                col=col,
            )

        except Exception as e:
            self.logger.error(f"Error adding score timeline: {e}")
            raise

    def _add_win_probability(self, fig: go.Figure, game_data: Dict, row: int, col: int):
        """Add win probability visualization"""
        try:
            # Extract win probability data
            times = game_data.get("game_times", [])
            probabilities = game_data.get("win_probabilities", [])

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=probabilities,
                    name="Win Probability",
                    fill="tozeroy",
                    line=dict(color=self.colors["primary"]),
                ),
                row=row,
                col=col,
            )

        except Exception as e:
            self.logger.error(f"Error adding win probability: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    visualizer = NFLGameVisualizer()

    # Example game data
    game_data = {
        "scoring_times": [0, 15, 30, 45, 60],
        "home_scores": [0, 7, 14, 21, 28],
        "away_scores": [0, 3, 10, 17, 24],
        "game_times": [0, 15, 30, 45, 60],
        "win_probabilities": [0.5, 0.6, 0.7, 0.8, 0.9],
    }

    # Create dashboard
    fig = visualizer.create_game_dashboard(
        game_data, home_team="Home Team", away_team="Away Team"
    )

    # Save visualization
    visualizer.save_visualization(fig, "game_analysis", "visualizations")
