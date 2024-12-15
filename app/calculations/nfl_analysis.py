"""NFL Analysis Module for generating visualizations and performance metrics."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go


class NFLAnalyzer:
    """Handles NFL game analysis and visualization."""

    def __init__(self, predictions_dir="analysis/predictions"):
        """Initialize analyzer with predictions directory."""
        self.predictions_dir = predictions_dir
        self.colors = {
            "primary": "#013369",  # NFL Blue
            "secondary": "#D50A0A",  # NFL Red
        }

    def generate_confusion_matrix(
        self, y_true, y_pred, save_path="analysis/confusion_matrix.png"
    ):
        """Generate and save confusion matrix visualization."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Loss", "Win"],
            yticklabels=["Loss", "Win"],
        )

        plt.title("NFL Prediction Confusion Matrix")
        plt.ylabel("True Outcome")
        plt.xlabel("Predicted Outcome")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Calculate metrics
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        return {"accuracy": accuracy, "confusion_matrix": cm.tolist()}

    def plot_feature_importance(
        self, feature_importance_df, save_path="analysis/feature_importance.png"
    ):
        """Generate feature importance visualization."""
        plt.figure(figsize=(12, 8))

        sns.barplot(
            data=feature_importance_df.sort_values("importance", ascending=True).tail(
                15
            ),
            x="importance",
            y="feature",
            palette="Blues_r",
        )

        plt.title("Top 15 Most Important Features in NFL Predictions")
        plt.xlabel("Feature Importance Score")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Save detailed CSV
        feature_importance_df.to_csv(
            "analysis/enhanced_feature_importance.csv", index=False
        )

    def analyze_form_difference(
        self, games_df, save_path="analysis/form_diff_analysis.png"
    ):
        """Analyze and visualize the impact of form difference on outcomes."""
        plt.figure(figsize=(12, 8))

        form_diff_win_rate = games_df.groupby("form_difference")["home_win"].mean()

        plt.plot(
            form_diff_win_rate.index,
            form_diff_win_rate.values,
            marker="o",
            color=self.colors["primary"],
        )

        plt.title("Impact of Form Difference on Win Probability")
        plt.xlabel("Form Difference (Home - Away)")
        plt.ylabel("Home Team Win Rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def analyze_win_rate_patterns(
        self, games_df, save_path="analysis/win_rate_diff_analysis.png"
    ):
        """Analyze win rate patterns based on various factors."""
        fig = plt.figure(figsize=(15, 10))

        # Create subplots for different analyses
        gs = fig.add_gridspec(2, 2)

        # Home vs Away Win Rates
        ax1 = fig.add_subplot(gs[0, 0])
        home_away_stats = games_df.groupby("season")[["home_win", "away_win"]].mean()
        home_away_stats.plot(kind="bar", ax=ax1)
        ax1.set_title("Home vs Away Win Rates by Season")
        ax1.set_ylabel("Win Rate")

        # Division Game Impact
        ax2 = fig.add_subplot(gs[0, 1])
        div_stats = games_df.groupby("is_division_game")["home_win"].mean()
        div_stats.plot(kind="bar", ax=ax2)
        ax2.set_title("Win Rates in Division vs Non-Division Games")
        ax2.set_ylabel("Win Rate")

        # Weather Impact
        ax3 = fig.add_subplot(gs[1, 0])
        weather_stats = games_df.groupby("weather_condition")["home_win"].mean()
        weather_stats.plot(kind="bar", ax=ax3)
        ax3.set_title("Win Rates by Weather Condition")
        ax3.set_ylabel("Win Rate")
        ax3.tick_params(axis="x", rotation=45)

        # Time Slot Impact
        ax4 = fig.add_subplot(gs[1, 1])
        time_stats = games_df.groupby("time_slot")["home_win"].mean()
        time_stats.plot(kind="bar", ax=ax4)
        ax4.set_title("Win Rates by Game Time Slot")
        ax4.set_ylabel("Win Rate")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def generate_interactive_dashboard(
        self, games_df, save_path="analysis/dashboard.html"
    ):
        """Generate interactive dashboard with Plotly."""
        # Win Probability by Vegas Line
        fig1 = px.scatter(
            games_df,
            x="vegas_line",
            y="win_probability",
            color="actual_winner",
            title="Win Probability vs Vegas Line",
        )

        # Team Performance Overview
        team_stats = (
            games_df.groupby("home_team")
            .agg({"home_win": "mean", "total_points": "mean", "yards_gained": "mean"})
            .reset_index()
        )

        fig2 = px.scatter(
            team_stats,
            x="total_points",
            y="yards_gained",
            color="home_win",
            hover_data=["home_team"],
            title="Team Performance Overview",
        )

        # Combine figures
        dashboard = go.Figure()
        dashboard.add_traces(fig1.data + fig2.data)

        # Update layout
        dashboard.update_layout(
            title="NFL Analysis Dashboard", height=800, showlegend=True
        )

        # Save dashboard
        dashboard.write_html(save_path)

    def analyze_prediction_accuracy(self, predictions_df):
        """Analyze prediction accuracy across different conditions."""
        accuracy_metrics = {
            "overall": np.mean(predictions_df["predicted"] == predictions_df["actual"]),
            "home_games": np.mean(
                predictions_df[predictions_df["is_home"]]["predicted"]
                == predictions_df[predictions_df["is_home"]]["actual"]
            ),
            "away_games": np.mean(
                predictions_df[~predictions_df["is_home"]]["predicted"]
                == predictions_df[~predictions_df["is_home"]]["actual"]
            ),
            "favorite": np.mean(
                predictions_df[predictions_df["is_favorite"]]["predicted"]
                == predictions_df[predictions_df["is_favorite"]]["actual"]
            ),
            "underdog": np.mean(
                predictions_df[~predictions_df["is_favorite"]]["predicted"]
                == predictions_df[~predictions_df["is_favorite"]]["actual"]
            ),
        }

        return pd.DataFrame.from_dict(
            accuracy_metrics, orient="index", columns=["accuracy"]
        )

    def generate_team_report(self, team_name, games_df):
        """Generate comprehensive report for a specific team."""
        team_games = games_df[
            (games_df["home_team"] == team_name) | (games_df["away_team"] == team_name)
        ]

        report = {
            "overall_win_rate": np.mean(team_games["winner"] == team_name),
            "home_win_rate": np.mean(
                team_games[team_games["home_team"] == team_name]["winner"] == team_name
            ),
            "away_win_rate": np.mean(
                team_games[team_games["away_team"] == team_name]["winner"] == team_name
            ),
            "avg_points_scored": team_games[team_games["winner"] == team_name][
                "points"
            ].mean(),
            "avg_points_allowed": team_games[team_games["winner"] != team_name][
                "points"
            ].mean(),
            "prediction_accuracy": np.mean(
                team_games["predicted_winner"] == team_games["winner"]
            ),
        }

        return pd.Series(report)

    def save_analysis_summary(self, summary_dict, save_path="analysis/summary.json"):
        """Save analysis summary to JSON file."""
        import json

        with open(save_path, "w") as f:
            json.dump(summary_dict, f, indent=4)

    def generate_full_analysis(self, games_df, predictions_df):
        """Generate comprehensive analysis package."""
        # Generate all visualizations
        self.generate_confusion_matrix(
            predictions_df["actual"], predictions_df["predicted"]
        )
        self.plot_feature_importance(self.calculate_feature_importance(games_df))
        self.analyze_form_difference(games_df)
        self.analyze_win_rate_patterns(games_df)
        self.generate_interactive_dashboard(games_df)

        # Calculate metrics
        accuracy_metrics = self.analyze_prediction_accuracy(predictions_df)

        # Generate team reports
        team_reports = {}
        for team in games_df["home_team"].unique():
            team_reports[team] = self.generate_team_report(team, games_df)

        # Save summary
        summary = {
            "accuracy_metrics": accuracy_metrics.to_dict(),
            "team_reports": {
                team: report.to_dict() for team, report in team_reports.items()
            },
        }

        self.save_analysis_summary(summary)

    def calculate_feature_importance(self, games_df):
        """Calculate feature importance using multiple methods."""
        from sklearn.ensemble import RandomForestClassifier

        # Prepare features
        feature_cols = [
            col
            for col in games_df.columns
            if col not in ["winner", "game_id", "date", "home_team", "away_team"]
        ]

        X = games_df[feature_cols]
        y = games_df["winner"]

        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": rf.feature_importances_}
        )

        return importance_df.sort_values("importance", ascending=False)
