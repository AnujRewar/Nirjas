#classifies comment
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

class CommentClassifier:
    def __init__(self):
        self.vectorizer = joblib.load(
            os.path.join(BASE_DIR, "vectorizer.pkl")
        )
        self.model = joblib.load(
            os.path.join(BASE_DIR, "nirjas_classifier.pkl")
        )
    
    def predict(self, text: str):
        X = self.vectorizer.transform([text])
        label = self.model.predict(X)[0]
        confidence = max(self.model.predict_proba(X)[0])
        return label, confidence
