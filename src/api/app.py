# src/api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Load model and vectorizer
MODEL_PATH = "models/logreg_model.joblib"
VECTORIZER_PATH = "models/vectorizer.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Request body schema
class ReviewRequest(BaseModel):
    review: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API!"}

# Prediction endpoint
@app.post("/predict")
def predict_sentiment(data: ReviewRequest):
    review = data.review
    vectorized = vectorizer.transform([review])
    prediction = model.predict(vectorized)[0]
    label = "positive" if prediction == 1 else "negative"
    return {"sentiment": label}
