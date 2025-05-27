from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import load
import re
from typing import List

app = FastAPI()

# Load the pre-trained spam classifier model
clf = load('spam_classifier.joblib')

# Input schema
class EmailInput(BaseModel):
    text: str

# Utility: Split text into sentences
def split_into_sentences(text: str) -> List[str]:
    # Basic sentence splitting (can be improved)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

@app.post("/predict", tags=["Spam Detection"])
async def predict_spam(input_text: EmailInput):
    sentences = split_into_sentences(input_text.text)

    if not sentences:
        raise HTTPException(status_code=400, detail="No valid sentences found.")

    predictions = clf.predict(sentences)

    # Filter and return only spammy sentences
    spam_sentences = [sentence for sentence, pred in zip(sentences, predictions) if pred == 1]

    return {"spam_sentences": spam_sentences}
