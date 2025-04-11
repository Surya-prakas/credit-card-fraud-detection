from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load your trained Logistic Regression model
model = joblib.load("fraud_model.pkl")

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API Running"}

@app.post("/predict")
async def predict(data: dict):
    # Convert the incoming JSON data to numpy array
    sample = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(sample)
    # Return result
    return {"fraud": bool(prediction[0])}
