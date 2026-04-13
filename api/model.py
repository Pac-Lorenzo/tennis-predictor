import joblib
import json
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "model" / "tennis_model_v2.pkl"
FEATURES_PATH = Path(__file__).parent.parent / "model" / "feature_cols.json"

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    feature_cols = json.load(f)

SAME_FEATURES = {
    "surface",
    "level",
    "indoor",
    "round",
    "best_of",
    "p1_h2h_n",
    "p1_h2h_surface_n",
}

def _score_one_order(features: dict) -> float:
    """Base model score: probability that current p1 wins."""
    X = pd.DataFrame([[features[col] for col in feature_cols]], columns=feature_cols)
    prob = model.predict_proba(X)[0]
    return float(prob[1])

def _swap_features(features: dict) -> dict:
    """Create the mirrored feature row with players swapped."""
    swapped = {}

    for key, value in features.items():
        if key in SAME_FEATURES:
            swapped[key] = value
        elif key.startswith("p1_"):
            swapped["p2_" + key[3:]] = value
        elif key.startswith("p2_"):
            swapped["p1_" + key[3:]] = value
        elif key.endswith("_diff") or key in {
            "rank_diff", "rank_points_diff", "age_diff", "recent_wr_diff",
            "surface_wr_diff", "h2h_diff", "h2h_surface_diff", "rest_diff",
            "matches_7d_diff", "matches_30d_diff", "elo_diff", "surface_elo_diff"
        }:
            swapped[key] = -value
        else:
            swapped[key] = value

    return swapped

def predict(features: dict) -> dict:
    """
    Symmetric prediction:
    - score (A,B)
    - score (B,A)
    - combine to guarantee complementarity
    """
    p_ab = _score_one_order(features)
    swapped = _swap_features(features)
    p_ba = _score_one_order(swapped)

    # p_ba is probability that B wins when B is slot p1
    p1_prob = round((p_ab + (1 - p_ba)) / 2, 3)
    p2_prob = round(1 - p1_prob, 3)

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
