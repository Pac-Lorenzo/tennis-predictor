"""Model loading and prediction helpers.

The API keeps the trained scikit-learn model and the saved feature order in
memory so each request only has to build a feature vector and call
`predict_proba`.
"""

import joblib
import json
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model" / "tennis_model_v2.pkl"
FEATURES_PATH = Path(__file__).parent.parent / "model" / "feature_cols.json"

# Load once at startup so requests do not repeatedly hit disk.
model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    feature_cols = json.load(f)

def predict(features: dict) -> dict:
    """Run the saved model against a fully assembled matchup feature dict."""

    # The model was trained with a fixed column order, so we rebuild the
    # incoming dict into that exact ordering before prediction.
    X = np.array([[features[col] for col in feature_cols]])
    
    prob = model.predict_proba(X)[0]
    p1_prob = round(float(prob[1]), 3)
    p2_prob = round(float(prob[0]), 3)
    
    # Confidence is a simple presentation label based on how far the predicted
    # probabilities are from a coin flip.
    gap = abs(p1_prob - p2_prob)
    if gap > 0.4:
        confidence = "high"
    elif gap > 0.2:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "p1_win_probability": p1_prob,
        "p2_win_probability": p2_prob,
        "predicted_winner": "player1" if p1_prob > 0.5 else "player2",
        "confidence": confidence
    }
