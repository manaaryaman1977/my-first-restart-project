from src.logger import setup_logger
logger = setup_logger()

from pydantic import BaseModel

class PredictionRequest(BaseModel):
    hours: float

from fastapi import FastAPI
from src.predict import predict
from src.data_loader import load_data
from src.train import train_model
import os

app = FastAPI(title="Student Score API", version="1.0.0")

MODEL_PATH = "models/linear_regression.pkl"

@app.on_event("startup")
def load_or_train():
    if not os.path.exists(MODEL_PATH):
        X, y = load_data("data/student_scores.csv")
        train_model(X, y)

@app.get("/")
def root():
    return {"message": "Student Score Prediction API is running"}

@app.post("/api/v1/predict")
def get_prediction(request: PredictionRequest):
    result = predict(request.hours)
    logger.info(f"Prediction requested for {request.hours} hours")
    return {
        "hours": request.hours,
        "predicted_score": result
    }
 
from fastapi import HTTPException

@app.post(
    "/api/v1/predict",
    operation_id="predict_score_v1"
)
def get_prediction(request: PredictionRequest):
    try:
        result = predict(request.hours)
        logger.info(f"Prediction made for {request.hours}")
        return {
            "hours": request.hours,
            "predicted_score": result
        }
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")    