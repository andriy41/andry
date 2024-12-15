#!/usr/bin/env python3
"""
Command-line interface for NFL prediction system
"""
import click
import logging
from datetime import datetime
import pandas as pd
from train_ensemble import train_model
from predict_games import NFLPredictor
from data_collection.data_fetcher import NFLDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """NFL Prediction System CLI"""
    pass


@cli.command()
@click.option("--start-year", default=2018, help="First year of training data")
@click.option(
    "--save-path", default="saved_models/ensemble", help="Where to save the model"
)
def train(start_year, save_path):
    """Train the ensemble model"""
    click.echo(f"Training model with data from {start_year} to present...")
    metrics = train_model(start_year=start_year, save_path=save_path)

    if metrics:
        click.echo("Training completed successfully!")
        click.echo("\nTraining Metrics:")
        for model_name, model_metrics in metrics.items():
            click.echo(f"\n{model_name}:")
            if isinstance(model_metrics, dict):
                for metric_name, value in model_metrics.items():
                    click.echo(f"  {metric_name}: {value}")
            else:
                click.echo(f"  Score: {model_metrics}")
    else:
        click.echo("Training failed. Check logs for details.")


@cli.command()
@click.option(
    "--model-path", default="saved_models/ensemble", help="Path to trained model"
)
def predict_upcoming(model_path):
    """Predict upcoming NFL games"""
    try:
        predictor = NFLPredictor(model_path=model_path)
        predictions = predictor.predict_upcoming_games()

        if predictions.empty:
            click.echo("No upcoming games found.")
            return

        click.echo("\nUpcoming Game Predictions:")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        click.echo(
            predictions[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "predicted_winner",
                    "win_probability",
                ]
            ]
        )

    except Exception as e:
        click.echo(f"Error making predictions: {e}")


@cli.command()
@click.argument("game_id")
@click.option(
    "--model-path", default="saved_models/ensemble", help="Path to trained model"
)
def predict_game(game_id, model_path):
    """Predict a specific game by ID"""
    try:
        predictor = NFLPredictor(model_path=model_path)
        prediction = predictor.predict_specific_game(game_id)

        click.echo("\nGame Prediction:")
        click.echo(f"Predicted Winner: {prediction['predicted_winner']}")
        click.echo(f"Win Probability: {prediction['win_probability']:.2%}")

        click.echo("\nModel Predictions:")
        for model, prob in prediction["model_predictions"].items():
            click.echo(f"{model}: {prob:.2%}")

    except Exception as e:
        click.echo(f"Error predicting game: {e}")


@cli.command()
@click.option("--days", default=7, help="Number of days to fetch")
def list_games(days):
    """List upcoming NFL games"""
    try:
        fetcher = NFLDataFetcher()
        today = datetime.now()
        end_date = today + pd.Timedelta(days=days)

        games = fetcher.fetch_games_by_dates(
            today.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
        )

        if not games:
            click.echo("No games found in the specified period.")
            return

        click.echo(f"\nUpcoming games for the next {days} days:")
        for game in games:
            game_time = pd.to_datetime(game["date"]).strftime("%Y-%m-%d %H:%M")
            home_team = game["competitions"][0]["competitors"][0]["team"][
                "abbreviation"
            ]
            away_team = game["competitions"][0]["competitors"][1]["team"][
                "abbreviation"
            ]
            game_id = game["id"]

            click.echo(f"\nGame ID: {game_id}")
            click.echo(f"Time: {game_time}")
            click.echo(f"Matchup: {away_team} @ {home_team}")

    except Exception as e:
        click.echo(f"Error listing games: {e}")


@cli.command()
@click.option("--year", default=datetime.now().year, help="Season year")
def collect_data(year):
    """Collect NFL data for a specific season"""
    try:
        fetcher = NFLDataFetcher()
        click.echo(f"Collecting data for {year} season...")

        df = fetcher.fetch_training_data([year])

        if df.empty:
            click.echo("No data found for the specified season.")
            return

        # Save to CSV
        filename = f"nfl_data_{year}.csv"
        df.to_csv(filename, index=False)
        click.echo(f"\nCollected {len(df)} games")
        click.echo(f"Data saved to {filename}")

    except Exception as e:
        click.echo(f"Error collecting data: {e}")


if __name__ == "__main__":
    cli()
