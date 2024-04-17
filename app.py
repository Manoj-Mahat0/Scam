from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from joblib import load
from typing import List

app = FastAPI()

# Load the pre-trained model
clf = load('spam_classifier.joblib')

# Define request body model
class EmailRequest(BaseModel):
    emails: List[str]

# Endpoint to predict whether an email is spam or not
@app.post('/predict')
async def predict_spam(email_request: EmailRequest):
    emails = email_request.emails

    # Make predictions
    predictions = clf.predict(emails)

    # Return the predictions
    return {'predictions': predictions.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
