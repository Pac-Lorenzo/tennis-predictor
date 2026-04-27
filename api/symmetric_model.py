"""Prediction wrapper that makes matchup probabilities exactly symmetric."""

from __future__ import annotations

import numpy as np
import pandas as pd


SAME_FEATURES = {
    "surface",
    "level",
    "indoor",
    "round",
    "best_of",
    "p1_h2h_n",
    "p1_h2h_surface_n",
}

SIGNED_FEATURES = {
    "rank_diff",
    "rank_points_diff",
    "age_diff",
    "recent_wr_diff",
    "surface_wr_diff",
    "h2h_diff",
    "h2h_surface_diff",
    "rest_diff",
    "matches_7d_diff",
    "matches_30d_diff",
    "elo_diff",
    "surface_elo_diff",
}


class SymmetricMatchupModel:
    """Wrap a binary classifier so player-order swaps are complementary.

    The base model still scores each order, but this wrapper combines
    P(A beats B) and 1 - P(B beats A) into a single symmetric probability.
    """

    is_symmetric = True

    def __init__(self, base_model, feature_cols: list[str]):
        self.base_model = base_model
        self.feature_cols = list(feature_cols)

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.feature_cols].copy()
        return pd.DataFrame(X, columns=self.feature_cols)

    def _swap_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        swapped = X.copy()
        for col in self.feature_cols:
            if col in SAME_FEATURES:
                swapped[col] = X[col]
            elif col.startswith("p1_"):
                paired = "p2_" + col[3:]
                swapped[col] = X[paired] if paired in X else X[col]
            elif col.startswith("p2_"):
                paired = "p1_" + col[3:]
                swapped[col] = X[paired] if paired in X else X[col]
            elif col in SIGNED_FEATURES or col.endswith("_diff"):
                swapped[col] = -X[col]
            else:
                swapped[col] = X[col]
        return swapped.loc[:, self.feature_cols]

    def predict_proba(self, X) -> np.ndarray:
        X_frame = self._as_frame(X)
        X_swapped = self._swap_frame(X_frame)

        p_ab = self.base_model.predict_proba(X_frame)[:, 1]
        p_ba = self.base_model.predict_proba(X_swapped)[:, 1]
        p_final = (p_ab + (1 - p_ba)) / 2

        return np.column_stack([1 - p_final, p_final])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "base_model": self.base_model,
            "feature_cols": self.feature_cols,
        }
