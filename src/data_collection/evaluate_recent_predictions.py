"""
Evaluate recent NFL game predictions
Compares predictions against actual results
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta


class NFLPredictionEvaluator:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.predictions_file = os.path.join(
            self.data_dir, "predictions", "recent_predictions.json"
        )
        self.results_file = os.path.join(self.data_dir, "recent_results.json")
        self.evaluation_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "evaluation"
        )
        os.makedirs(self.evaluation_dir, exist_ok=True)

    def load_predictions(self):
        """Load recent predictions"""
        if os.path.exists(self.predictions_file):
            with open(self.predictions_file, "r") as f:
                return json.load(f)
        return {}

    def load_actual_results(self):
        """Load actual game results"""
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                return json.load(f)
        return {}

    def evaluate_predictions(self, predictions, results):
        """Compare predictions with actual results"""
        evaluation = {
            "total_games": 0,
            "correct_winner": 0,
            "correct_spread": 0,
            "correct_total": 0,
            "average_point_diff": 0,
            "system_performance": {
                "advanced": {"correct": 0, "total": 0},
                "vedic": {"correct": 0, "total": 0},
                "ml": {"correct": 0, "total": 0},
                "sports": {"correct": 0, "total": 0},
            },
        }

        for game_id, pred in predictions.items():
            if game_id in results:
                evaluation["total_games"] += 1
                actual = results[game_id]

                # Evaluate winner prediction
                if pred["predicted_winner"] == actual["winner"]:
                    evaluation["correct_winner"] += 1

                # Evaluate spread and total predictions
                # Add specific NFL evaluation logic

        return evaluation

    def save_evaluation(self, evaluation):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.evaluation_dir, f"evaluation_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(evaluation, f, indent=2)
        print(f"Saved evaluation to {filepath}")

    def generate_report(self, evaluation):
        """Generate readable evaluation report"""
        report = f"""
NFL Prediction System Evaluation Report
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Overall Performance:
-------------------
Total Games Evaluated: {evaluation['total_games']}
Correct Winner Predictions: {evaluation['correct_winner']} ({evaluation['correct_winner']/evaluation['total_games']*100:.1f}%)
Correct Spread Predictions: {evaluation['correct_spread']} ({evaluation['correct_spread']/evaluation['total_games']*100:.1f}%)
Correct Total Predictions: {evaluation['correct_total']} ({evaluation['correct_total']/evaluation['total_games']*100:.1f}%)

System-wise Performance:
----------------------
Advanced System: {evaluation['system_performance']['advanced']['correct']}/{evaluation['system_performance']['advanced']['total']} ({evaluation['system_performance']['advanced']['correct']/evaluation['system_performance']['advanced']['total']*100:.1f}%)
Vedic System: {evaluation['system_performance']['vedic']['correct']}/{evaluation['system_performance']['vedic']['total']} ({evaluation['system_performance']['vedic']['correct']/evaluation['system_performance']['vedic']['total']*100:.1f}%)
ML System: {evaluation['system_performance']['ml']['correct']}/{evaluation['system_performance']['ml']['total']} ({evaluation['system_performance']['ml']['correct']/evaluation['system_performance']['ml']['total']*100:.1f}%)
Sports Analysis: {evaluation['system_performance']['sports']['correct']}/{evaluation['system_performance']['sports']['total']} ({evaluation['system_performance']['sports']['correct']/evaluation['system_performance']['sports']['total']*100:.1f}%)
"""
        return report


def main():
    evaluator = NFLPredictionEvaluator()
    predictions = evaluator.load_predictions()
    results = evaluator.load_actual_results()
    evaluation = evaluator.evaluate_predictions(predictions, results)
    evaluator.save_evaluation(evaluation)

    # Generate and save report
    report = evaluator.generate_report(evaluation)
    report_file = os.path.join(evaluator.evaluation_dir, "latest_evaluation_report.txt")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Generated evaluation report: {report_file}")


if __name__ == "__main__":
    main()
