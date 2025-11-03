from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Flight Price Prediction API (Demo)",
    description="API REST for flight price prediction",
    version="1.0.0"
)

# Load model
# Currently dummy model for testing (TODO use project model)
model = joblib.load("model/dummy_model.pkl")

class FlightData(BaseModel):
    feature1: float
    feature2: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_price(data: FlightData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"predicted_price": float(prediction)}