from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

# 🚀 Add this for CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API Running"}

@app.post("/predict")
async def predict(data: dict):
    sample = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(sample)
    return {"fraud": bool(prediction[0])}
