from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Credit Card Fraud Detection API")

# Load trained model
model = joblib.load("fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    result = "Fraud Transaction" if prediction == 1 else "Normal Transaction"

    return {
        "prediction": result,
        "fraud_probability": float(probability)
    }
