"""Train NFL prediction model with Vedic features."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_record_to_winpct(record):
    """Convert string record (e.g., '3-2') to win percentage."""
    try:
        if pd.isna(record) or record == "":
            return 0.0
        wins, losses = map(int, str(record).split("-"))
        total = wins + losses
        return wins / total if total > 0 else 0.0
    except:
        return 0.0


def load_data():
    """Load processed data with Vedic features."""
    df = pd.read_csv("data/processed_vedic/nfl_games_with_vedic.csv")

    # Convert record columns to win percentages
    record_columns = [col for col in df.columns if "_record" in col]
    for col in record_columns:
        df[f"{col}_pct"] = df[col].apply(convert_record_to_winpct)
        df = df.drop(columns=[col])

    return df


def prepare_features(df):
    """Prepare feature sets for model comparison."""
    # Basic features (win percentages, rankings, etc.)
    basic_features = [
        col
        for col in df.columns
        if any(x in col for x in ["winpct", "rank", "_pct", "streak"])
    ]

    # Statistical features
    stat_features = [
        col
        for col in df.columns
        if any(
            x in col
            for x in ["score", "yards", "attempts", "completions", "differential"]
        )
    ]

    # Vedic features
    vedic_features = [
        col
        for col in df.columns
        if any(
            x in col
            for x in [
                "aggression_score",
                "expansion_score",
                "discipline_score",
                "leadership_score",
                "strategy_score",
            ]
        )
    ]

    # Combine all features
    all_features = list(set(basic_features + stat_features + vedic_features))

    # Remove target variable if present
    if "home_win" in all_features:
        all_features.remove("home_win")

    # Remove any non-numeric columns
    numeric_cols = df[all_features].select_dtypes(include=[np.number]).columns
    non_numeric = set(all_features) - set(numeric_cols)
    if non_numeric:
        logger.warning(f"Removing non-numeric columns: {non_numeric}")
        all_features = list(numeric_cols)

    return {
        "basic": [f for f in basic_features if f in numeric_cols],
        "statistical": [f for f in stat_features if f in numeric_cols],
        "vedic": [f for f in vedic_features if f in numeric_cols],
        "all": all_features,
    }


def train_and_evaluate(X, y, feature_set_name):
    """Train and evaluate model using cross-validation."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    logger.info(f"\n{feature_set_name} Features CV Scores:")
    logger.info(
        f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
    )

    # Train final model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    logger.info(f"\nTest Set Performance:")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    logger.info(f"\nTop 10 Most Important Features:")
    logger.info(feature_importance.head(10))

    return model, feature_importance


def main():
    """Main training pipeline."""
    logger.info("Loading data...")
    df = load_data()

    # Prepare different feature sets
    feature_sets = prepare_features(df)

    # Target variable
    y = df["home_win"]

    # Train and evaluate models with different feature sets
    models = {}
    for set_name, features in feature_sets.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training model with {set_name} features...")
        logger.info(f"Number of features: {len(features)}")

        X = df[features]
        model, importance = train_and_evaluate(X, y, set_name)
        models[set_name] = (model, importance)

        # Save model
        joblib.dump(model, f"models/nfl_predictor_{set_name}.joblib")

    logger.info("\nTraining complete! Models saved to models/ directory")


if __name__ == "__main__":
    main()
