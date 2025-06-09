from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load("models/best_model.pkl")

app = FastAPI(title="Sentiment Analysis API")

# Define request schema
class RequestText(BaseModel):
    text: str

# Define response schema (optional)
class SentimentResponse(BaseModel):
    sentiment: str

# Endpoint for prediction
@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(data: RequestText):
    prediction = model.predict([data.text])[0]
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}
