import argparse
import joblib
import pandas as pd
import os

from data_loader import load_data
from train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, help="Study hours for prediction")
    args = parser.parse_args()

    model_path = "models/linear_regression.pkl"

    # Train model if not exists
    if not os.path.exists(model_path):
        X, y = load_data("data/student_scores.csv")
        train_model(X, y)

    model = joblib.load(model_path)

    if args.hours is not None:
        prediction = model.predict(pd.DataFrame([[args.hours]], columns=["hours"]))
        print(f"Predicted score for {args.hours} hours: {prediction[0]:.2f}")
    else:
        print("Model is ready. Use --hours to get prediction.")

if __name__ == "__main__":
    main()