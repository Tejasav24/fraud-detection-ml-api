from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return {"prediction": int(prediction[0])}


