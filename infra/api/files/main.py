import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from modelling import run_inference
from app_config import (APP_TITLE, APP_DESCRIPTION, APP_VERSION, MLFLOW_TRACKING_URI, REGISTERED_MODEL_URI)


app = FastAPI(title=APP_TITLE,
              description=APP_DESCRIPTION,
              version=APP_VERSION)


class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: int
    slope: int
    ca: int
    thal: int


class PredictionOut(BaseModel):
    heart_attack_prediction: float


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
pipeline = mlflow.pyfunc.load_model(model_uri=REGISTERED_MODEL_URI)


@app.get("/")
def home():
    return {"health_check": "OK",
            "model_version": pipeline._model_meta.run_id}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData):
    heart_attack_prediction = run_inference(payload.dict(), pipeline)
    return {"heart_attack_prediction": heart_attack_prediction}


# {
#   "age": 63,
#   "sex": 1,
#   "cp": 3,
#   "trestbps": 145,
#   "chol": 233,
#   "fbs": 1,
#   "restecg": 0,
#   "thalach": 150,
#   "exang": 0,
#   "oldpeak": 2.3,
#   "slope": 0,
#   "ca": 0,
#   "thal": 1
# }
