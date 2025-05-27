from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from joblib import load
import re

app = FastAPI()

# Load the pre-trained model (assumed to be a Pipeline with vectorizer + classifier)
clf: Pipeline = load('spam_classifier.joblib')

# Request model
class EmailInput(BaseModel):
    text: str

# Utility to split text into sentences
def split_into_sentences(text: str):
    return re.split(r'(?<=[.!?])\s+', text.strip())

@app.post("/predict")
async def predict_scam(data: EmailInput):
    sentences = split_into_sentences(data.text)

    # Remove any empty strings
    sentences = [s for s in sentences if s.strip()]

    if not sentences:
        return {"scam_detected": 0}

    predictions = clf.predict(sentences)

    # If any sentence is predicted as scam (1), we return 1
    scam_detected = 1 if 1 in predictions else 0

    return {"scam_detected": scam_detected}
