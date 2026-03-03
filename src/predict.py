import joblib
import pandas as pd

def predict(hours, model_path="models/linear_regression.pkl"):
    model = joblib.load(model_path)
    prediction = model.predict(pd.DataFrame([[hours]], columns=["hours"]))
    return prediction[0]