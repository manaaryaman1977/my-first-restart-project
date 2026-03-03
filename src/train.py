from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_regression.pkl")

    return model