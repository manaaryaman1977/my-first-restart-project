from pydantic import BaseModel

class PredictionRequest(BaseModel):
    hours: float

from fastapi import FastAPI
from src.predict import predict
from src.data_loader import load_data
from src.train import train_model
import os

app = FastAPI()

MODEL_PATH = "models/linear_regression.pkl"

@app.on_event("startup")
def load_or_train():
    if not os.path.exists(MODEL_PATH):
        X, y = load_data("data/student_scores.csv")
        train_model(X, y)

@app.get("/")
def root():
    return {"message": "Student Score Prediction API is running"}

@app.post("/predict")
def get_prediction(request: PredictionRequest):
    result = predict(request.hours)
    return {
        "hours": request.hours,
        "predicted_score": result
    }