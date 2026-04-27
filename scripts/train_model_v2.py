"""Deterministic training entrypoint for the v2 tennis match model.

This script mirrors the notebook's feature engineering, but fixes two model
process issues:
1. model selection uses validation data only
2. the final model can be refit on train+val before one final test evaluation

By default this script does not overwrite the production artifact. Use
`--save-artifacts` to write a candidate model/report under `model/`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import ParameterGrid


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.symmetric_model import SymmetricMatchupModel

BASELINE_PARAMS = {
    "learning_rate": 0.08,
    "max_depth": 5,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_iter": 300,
}

NOTEBOOK_TUNED_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_leaf_nodes": 40,
    "min_samples_leaf": 10,
    "l2_regularization": 0.01,
    "max_iter": 500,
}

SURFACE_MAP = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 0}
LEVEL_MAP = {"250": 0, "500": 1, "A": 2, "M": 3, "G": 4, "D": 5, "O": 5, "F": 6}
INDOOR_MAP = {"O": 0, "I": 1}
ROUND_MAP = {"RR": 0, "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7, "BR": 6}

FEATURE_COLS = [
    "p1_rank",
    "p2_rank",
    "rank_diff",
    "p1_rank_points",
    "p2_rank_points",
    "rank_points_diff",
    "p1_age",
    "p2_age",
    "age_diff",
    "surface",
    "level",
    "indoor",
    "round",
    "best_of",
    "p1_recent_wr",
    "p2_recent_wr",
    "recent_wr_diff",
    "p1_recent_n",
    "p2_recent_n",
    "p1_surface_wr",
    "p2_surface_wr",
    "surface_wr_diff",
    "p1_surface_n",
    "p2_surface_n",
    "p1_h2h_wr",
    "p2_h2h_wr",
    "h2h_diff",
    "p1_h2h_n",
    "p1_h2h_surface_wr",
    "p2_h2h_surface_wr",
    "h2h_surface_diff",
    "p1_h2h_surface_n",
    "p1_days_rest",
    "p2_days_rest",
    "rest_diff",
    "p1_matches_7d",
    "p2_matches_7d",
    "matches_7d_diff",
    "p1_matches_30d",
    "p2_matches_30d",
    "matches_30d_diff",
    "p1_elo",
    "p2_elo",
    "elo_diff",
    "p1_surface_elo",
    "p2_surface_elo",
    "surface_elo_diff",
]


def smoothed_wr(wins: int, matches: int, prior_wins: float = 2.5, prior_matches: int = 5) -> float:
    return (wins + prior_wins) / (matches + prior_matches)


def expected_score(r1: float, r2: float) -> float:
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def update_elo(r1: float, r2: float, score1: int, k: int = 32) -> tuple[float, float]:
    e1 = expected_score(r1, r2)
    new_r1 = r1 + k * (score1 - e1)
    new_r2 = r2 + k * ((1 - score1) - (1 - e1))
    return new_r1, new_r2


def days_since(last_date: pd.Timestamp | None, current_date: pd.Timestamp, default_days: int = 30) -> int:
    if pd.isna(last_date):
        return default_days
    return min((current_date - last_date).days, 60)


def prune_old_dates(date_deque: deque[pd.Timestamp], current_date: pd.Timestamp, window_days: int) -> None:
    while date_deque and (current_date - date_deque[0]).days > window_days:
        date_deque.popleft()


def load_raw_matches() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("20*.csv"))
    files = [path for path in files if path.stem.isdigit() and int(path.stem) >= 2015]
    if not files:
        raise FileNotFoundError(f"No yearly CSV files found under {DATA_DIR}")

    dfs = []
    for path in files:
        year_df = pd.read_csv(path)
        year_df["source_file"] = path.name
        dfs.append(year_df)

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["tourney_date"] != "tourney_date"]
    return df


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date"]).copy()
    df = df.dropna(subset=["winner_rank", "loser_rank", "winner_age", "loser_age"]).copy()

    df["winner_rank_points"] = pd.to_numeric(df["winner_rank_points"], errors="coerce").fillna(0)
    df["loser_rank_points"] = pd.to_numeric(df["loser_rank_points"], errors="coerce").fillna(0)
    df["indoor"] = df["indoor"].fillna("O")

    df = df[~df["score"].str.contains("W/O|RET", na=False)].copy()
    df = df.dropna(subset=["winner_id", "loser_id"]).copy()
    df = df[df["winner_id"].astype(str) != "nan"].copy()
    df = df[df["loser_id"].astype(str) != "nan"].copy()

    df["surface_enc"] = df["surface"].map(SURFACE_MAP)
    df["level_enc"] = df["tourney_level"].astype(str).map(LEVEL_MAP)
    df["indoor_enc"] = df["indoor"].map(INDOOR_MAP)
    df["round_enc"] = df["round"].map(ROUND_MAP)
    df = df.dropna(subset=["surface_enc", "level_enc", "indoor_enc", "round_enc", "best_of"]).copy()

    df["year"] = df["tourney_date"].dt.year
    if "match_num" in df.columns:
        df["match_id"] = df["tourney_id"].astype(str) + "_" + df["match_num"].astype(str)
    else:
        df["match_id"] = df["tourney_id"].astype(str) + "_" + df.index.astype(str)

    df = df.sort_values(["tourney_date", "match_id"]).reset_index(drop=True)

    numeric_cols = [
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points",
        "winner_age",
        "loser_age",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["winner_rank", "loser_rank", "winner_age", "loser_age"]).copy()
    return df


def build_feature_history(df: pd.DataFrame) -> pd.DataFrame:
    recent_results = defaultdict(lambda: deque(maxlen=10))
    recent_surface_results = defaultdict(lambda: deque(maxlen=10))
    wins_by_pair = defaultdict(int)
    wins_by_pair_surface = defaultdict(int)
    last_match_date = {}
    recent_dates_7 = defaultdict(deque)
    recent_dates_30 = defaultdict(deque)
    elo = defaultdict(lambda: 1500.0)
    elo_surface = defaultdict(lambda: 1500.0)

    feature_rows = []

    for row in df.itertuples(index=False):
        match_date = row.tourney_date
        surface = int(row.surface_enc)
        winner_id = row.winner_id
        loser_id = row.loser_id

        winner_hist = recent_results[winner_id]
        loser_hist = recent_results[loser_id]
        winner_recent_wr = smoothed_wr(sum(winner_hist), len(winner_hist))
        loser_recent_wr = smoothed_wr(sum(loser_hist), len(loser_hist))

        winner_surface_hist = recent_surface_results[(winner_id, surface)]
        loser_surface_hist = recent_surface_results[(loser_id, surface)]
        winner_surface_wr = smoothed_wr(sum(winner_surface_hist), len(winner_surface_hist))
        loser_surface_wr = smoothed_wr(sum(loser_surface_hist), len(loser_surface_hist))

        winner_h2h_wins = wins_by_pair[(winner_id, loser_id)]
        loser_h2h_wins = wins_by_pair[(loser_id, winner_id)]
        total_h2h = winner_h2h_wins + loser_h2h_wins
        winner_h2h_wr = (winner_h2h_wins + 1) / (total_h2h + 2)

        winner_h2h_surface_wins = wins_by_pair_surface[(winner_id, loser_id, surface)]
        loser_h2h_surface_wins = wins_by_pair_surface[(loser_id, winner_id, surface)]
        total_h2h_surface = winner_h2h_surface_wins + loser_h2h_surface_wins
        winner_h2h_surface_wr = (winner_h2h_surface_wins + 1) / (total_h2h_surface + 2)

        prune_old_dates(recent_dates_7[winner_id], match_date, 7)
        prune_old_dates(recent_dates_7[loser_id], match_date, 7)
        prune_old_dates(recent_dates_30[winner_id], match_date, 30)
        prune_old_dates(recent_dates_30[loser_id], match_date, 30)

        winner_elo = elo[winner_id]
        loser_elo = elo[loser_id]
        winner_surface_elo = elo_surface[(winner_id, surface)]
        loser_surface_elo = elo_surface[(loser_id, surface)]

        feature_rows.append(
            {
                "match_id": row.match_id,
                "match_date": match_date,
                "year": row.year,
                "winner_rank": row.winner_rank,
                "loser_rank": row.loser_rank,
                "winner_rank_points": row.winner_rank_points,
                "loser_rank_points": row.loser_rank_points,
                "winner_age": row.winner_age,
                "loser_age": row.loser_age,
                "surface_enc": row.surface_enc,
                "level_enc": row.level_enc,
                "indoor_enc": row.indoor_enc,
                "round_enc": row.round_enc,
                "best_of": row.best_of,
                "w_recent_wr": winner_recent_wr,
                "l_recent_wr": loser_recent_wr,
                "w_recent_n": len(winner_hist),
                "l_recent_n": len(loser_hist),
                "w_surface_wr": winner_surface_wr,
                "l_surface_wr": loser_surface_wr,
                "w_surface_n": len(winner_surface_hist),
                "l_surface_n": len(loser_surface_hist),
                "w_h2h_wr": winner_h2h_wr,
                "total_h2h": total_h2h,
                "w_h2h_surface_wr": winner_h2h_surface_wr,
                "total_h2h_surface": total_h2h_surface,
                "w_days_rest": days_since(last_match_date.get(winner_id), match_date),
                "l_days_rest": days_since(last_match_date.get(loser_id), match_date),
                "w_matches_7d": len(recent_dates_7[winner_id]),
                "l_matches_7d": len(recent_dates_7[loser_id]),
                "w_matches_30d": len(recent_dates_30[winner_id]),
                "l_matches_30d": len(recent_dates_30[loser_id]),
                "w_elo": winner_elo,
                "l_elo": loser_elo,
                "w_surface_elo": winner_surface_elo,
                "l_surface_elo": loser_surface_elo,
            }
        )

        recent_results[winner_id].append(1)
        recent_results[loser_id].append(0)
        recent_surface_results[(winner_id, surface)].append(1)
        recent_surface_results[(loser_id, surface)].append(0)
        wins_by_pair[(winner_id, loser_id)] += 1
        wins_by_pair_surface[(winner_id, loser_id, surface)] += 1
        recent_dates_7[winner_id].append(match_date)
        recent_dates_7[loser_id].append(match_date)
        recent_dates_30[winner_id].append(match_date)
        recent_dates_30[loser_id].append(match_date)
        last_match_date[winner_id] = match_date
        last_match_date[loser_id] = match_date
        elo[winner_id], elo[loser_id] = update_elo(winner_elo, loser_elo, 1)
        elo_surface[(winner_id, surface)], elo_surface[(loser_id, surface)] = update_elo(
            winner_surface_elo, loser_surface_elo, 1
        )

    return pd.DataFrame(feature_rows)


def build_model_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in feature_df.itertuples(index=False):
        base = {
            "match_id": row.match_id,
            "match_date": row.match_date,
            "year": row.year,
            "surface": row.surface_enc,
            "level": row.level_enc,
            "indoor": row.indoor_enc,
            "round": row.round_enc,
            "best_of": row.best_of,
        }

        rows.append(
            {
                **base,
                "p1_rank": row.winner_rank,
                "p2_rank": row.loser_rank,
                "rank_diff": row.loser_rank - row.winner_rank,
                "p1_rank_points": row.winner_rank_points,
                "p2_rank_points": row.loser_rank_points,
                "rank_points_diff": row.winner_rank_points - row.loser_rank_points,
                "p1_age": row.winner_age,
                "p2_age": row.loser_age,
                "age_diff": row.winner_age - row.loser_age,
                "p1_recent_wr": row.w_recent_wr,
                "p2_recent_wr": row.l_recent_wr,
                "recent_wr_diff": row.w_recent_wr - row.l_recent_wr,
                "p1_recent_n": row.w_recent_n,
                "p2_recent_n": row.l_recent_n,
                "p1_surface_wr": row.w_surface_wr,
                "p2_surface_wr": row.l_surface_wr,
                "surface_wr_diff": row.w_surface_wr - row.l_surface_wr,
                "p1_surface_n": row.w_surface_n,
                "p2_surface_n": row.l_surface_n,
                "p1_h2h_wr": row.w_h2h_wr,
                "p2_h2h_wr": 1 - row.w_h2h_wr,
                "h2h_diff": row.w_h2h_wr - (1 - row.w_h2h_wr),
                "p1_h2h_n": row.total_h2h,
                "p1_h2h_surface_wr": row.w_h2h_surface_wr,
                "p2_h2h_surface_wr": 1 - row.w_h2h_surface_wr,
                "h2h_surface_diff": row.w_h2h_surface_wr - (1 - row.w_h2h_surface_wr),
                "p1_h2h_surface_n": row.total_h2h_surface,
                "p1_days_rest": row.w_days_rest,
                "p2_days_rest": row.l_days_rest,
                "rest_diff": row.w_days_rest - row.l_days_rest,
                "p1_matches_7d": row.w_matches_7d,
                "p2_matches_7d": row.l_matches_7d,
                "matches_7d_diff": row.w_matches_7d - row.l_matches_7d,
                "p1_matches_30d": row.w_matches_30d,
                "p2_matches_30d": row.l_matches_30d,
                "matches_30d_diff": row.w_matches_30d - row.l_matches_30d,
                "p1_elo": row.w_elo,
                "p2_elo": row.l_elo,
                "elo_diff": row.w_elo - row.l_elo,
                "p1_surface_elo": row.w_surface_elo,
                "p2_surface_elo": row.l_surface_elo,
                "surface_elo_diff": row.w_surface_elo - row.l_surface_elo,
                "target": 1,
            }
        )

        rows.append(
            {
                **base,
                "p1_rank": row.loser_rank,
                "p2_rank": row.winner_rank,
                "rank_diff": row.winner_rank - row.loser_rank,
                "p1_rank_points": row.loser_rank_points,
                "p2_rank_points": row.winner_rank_points,
                "rank_points_diff": row.loser_rank_points - row.winner_rank_points,
                "p1_age": row.loser_age,
                "p2_age": row.winner_age,
                "age_diff": row.loser_age - row.winner_age,
                "p1_recent_wr": row.l_recent_wr,
                "p2_recent_wr": row.w_recent_wr,
                "recent_wr_diff": row.l_recent_wr - row.w_recent_wr,
                "p1_recent_n": row.l_recent_n,
                "p2_recent_n": row.w_recent_n,
                "p1_surface_wr": row.l_surface_wr,
                "p2_surface_wr": row.w_surface_wr,
                "surface_wr_diff": row.l_surface_wr - row.w_surface_wr,
                "p1_surface_n": row.l_surface_n,
                "p2_surface_n": row.w_surface_n,
                "p1_h2h_wr": 1 - row.w_h2h_wr,
                "p2_h2h_wr": row.w_h2h_wr,
                "h2h_diff": (1 - row.w_h2h_wr) - row.w_h2h_wr,
                "p1_h2h_n": row.total_h2h,
                "p1_h2h_surface_wr": 1 - row.w_h2h_surface_wr,
                "p2_h2h_surface_wr": row.w_h2h_surface_wr,
                "h2h_surface_diff": (1 - row.w_h2h_surface_wr) - row.w_h2h_surface_wr,
                "p1_h2h_surface_n": row.total_h2h_surface,
                "p1_days_rest": row.l_days_rest,
                "p2_days_rest": row.w_days_rest,
                "rest_diff": row.l_days_rest - row.w_days_rest,
                "p1_matches_7d": row.l_matches_7d,
                "p2_matches_7d": row.w_matches_7d,
                "matches_7d_diff": row.l_matches_7d - row.w_matches_7d,
                "p1_matches_30d": row.l_matches_30d,
                "p2_matches_30d": row.w_matches_30d,
                "matches_30d_diff": row.l_matches_30d - row.w_matches_30d,
                "p1_elo": row.l_elo,
                "p2_elo": row.w_elo,
                "elo_diff": row.l_elo - row.w_elo,
                "p1_surface_elo": row.l_surface_elo,
                "p2_surface_elo": row.w_surface_elo,
                "surface_elo_diff": row.l_surface_elo - row.w_surface_elo,
                "target": 0,
            }
        )

    return pd.DataFrame(rows)


def split_model_data(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = model_df[model_df["match_date"] < "2023-01-01"].copy()
    val_df = model_df[(model_df["match_date"] >= "2023-01-01") & (model_df["match_date"] < "2024-01-01")].copy()
    test_df = model_df[model_df["match_date"] >= "2024-01-01"].copy()
    return train_df, val_df, test_df


def evaluate_binary_model(model, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "set": label,
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
        "log_loss": float(log_loss(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
    }


def shortlist_candidates() -> list[dict]:
    candidates = [
        BASELINE_PARAMS,
        NOTEBOOK_TUNED_PARAMS,
        {**NOTEBOOK_TUNED_PARAMS, "max_iter": 300},
        {**NOTEBOOK_TUNED_PARAMS, "min_samples_leaf": 20},
        {**NOTEBOOK_TUNED_PARAMS, "max_depth": 5},
        {**NOTEBOOK_TUNED_PARAMS, "max_leaf_nodes": 31},
        {**NOTEBOOK_TUNED_PARAMS, "learning_rate": 0.08},
        {**NOTEBOOK_TUNED_PARAMS, "l2_regularization": 0.0},
    ]
    unique = []
    seen = set()
    for params in candidates:
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            unique.append(params)
    return unique


def full_grid_candidates() -> list[dict]:
    return list(
        ParameterGrid(
            {
                "learning_rate": [0.03, 0.05, 0.08, 0.10, 0.12],
                "max_depth": [3, 4, 5, 6],
                "max_leaf_nodes": [15, 20, 31, 40],
                "min_samples_leaf": [10, 20, 30, 50],
                "l2_regularization": [0.0, 0.01, 0.1, 0.5],
                "max_iter": [300, 400, 500],
            }
        )
    )


def select_candidates(mode: str) -> list[dict]:
    if mode == "notebook":
        return [BASELINE_PARAMS, NOTEBOOK_TUNED_PARAMS]
    if mode == "shortlist":
        return shortlist_candidates()
    if mode == "full":
        return full_grid_candidates()
    raise ValueError(f"Unsupported grid mode: {mode}")


def rank_candidates(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -row["val_auc"],
            row["val_log_loss"],
            row["val_brier"],
            -row["val_accuracy"],
        ),
    )


def score_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    candidates: list[dict],
) -> list[dict]:
    results = []
    total = len(candidates)
    for idx, params in enumerate(candidates, start=1):
        model = HistGradientBoostingClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        metrics = evaluate_binary_model(model, X_val, y_val, "val")
        results.append(
            {
                "candidate_index": idx,
                "params": params,
                "val_accuracy": metrics["accuracy"],
                "val_auc": metrics["auc"],
                "val_log_loss": metrics["log_loss"],
                "val_brier": metrics["brier"],
            }
        )
        print(
            f"[{idx}/{total}] val_auc={metrics['auc']:.5f} "
            f"val_log_loss={metrics['log_loss']:.5f} params={params}"
        )
    return rank_candidates(results)


def fit_base_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    return model


def fit_calibrated_model(
    base_model: HistGradientBoostingClassifier,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    method: str,
):
    if method == "none":
        return base_model
    calibrator = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    calibrator.fit(X_calibration, y_calibration)
    return calibrator


def evaluate_calibration_modes(
    selected_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    modes: list[str],
) -> tuple[object, dict, list[dict]]:
    base_model = fit_base_model(X_train, y_train, selected_params)
    comparison = []

    for mode in modes:
        model = fit_calibrated_model(copy.deepcopy(base_model), X_val, y_val, mode)
        val_metrics = evaluate_binary_model(model, X_val, y_val, f"val_{mode}")
        test_metrics = evaluate_binary_model(model, X_test, y_test, f"test_{mode}")
        row = {
            "mode": mode,
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_log_loss": val_metrics["log_loss"],
            "val_brier": val_metrics["brier"],
            "test_accuracy": test_metrics["accuracy"],
            "test_auc": test_metrics["auc"],
            "test_log_loss": test_metrics["log_loss"],
            "test_brier": test_metrics["brier"],
        }
        comparison.append(row)
        print(
            f"calibration={mode} val_log_loss={val_metrics['log_loss']:.5f} "
            f"val_brier={val_metrics['brier']:.5f} test_log_loss={test_metrics['log_loss']:.5f}"
        )

    comparison = sorted(
        comparison,
        key=lambda row: (
            row["val_log_loss"],
            row["val_brier"],
            -row["val_auc"],
            -row["val_accuracy"],
        ),
    )
    best = comparison[0]
    final_model = fit_calibrated_model(fit_base_model(X_train, y_train, selected_params), X_val, y_val, best["mode"])
    return final_model, best, comparison


def symmetry_diagnostics(
    model,
    test_df: pd.DataFrame,
    sample_size: int,
) -> dict:
    same_features = {
        "surface",
        "level",
        "indoor",
        "round",
        "best_of",
        "p1_h2h_n",
        "p1_h2h_surface_n",
    }

    def swap_features(features: dict) -> dict:
        swapped = {}
        for key, value in features.items():
            if key in same_features:
                swapped[key] = value
            elif key.startswith("p1_"):
                swapped["p2_" + key[3:]] = value
            elif key.startswith("p2_"):
                swapped["p1_" + key[3:]] = value
            elif key.endswith("_diff"):
                swapped[key] = -value
            else:
                swapped[key] = value
        return swapped

    sample_df = test_df[test_df["target"] == 1].sample(
        min(sample_size, len(test_df[test_df["target"] == 1])),
        random_state=42,
    )
    inconsistencies = []
    for _, row in sample_df.iterrows():
        features = row[FEATURE_COLS].to_dict()
        x_ab = pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
        swapped = swap_features(features)
        x_ba = pd.DataFrame([[swapped[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
        p_ab = model.predict_proba(x_ab)[0, 1]
        p_ba = model.predict_proba(x_ba)[0, 1]
        inconsistencies.append(abs(p_ab - (1 - p_ba)))

    return {
        "sample_size": len(inconsistencies),
        "raw_mean_inconsistency": float(np.mean(inconsistencies)) if inconsistencies else 0.0,
        "raw_max_inconsistency": float(np.max(inconsistencies)) if inconsistencies else 0.0,
    }


def calibration_summary(model, X_test: pd.DataFrame, y_test: pd.Series) -> list[dict]:
    proba_test = model.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=10)
    return [
        {"predicted": float(pred), "actual": float(actual)}
        for actual, pred in zip(prob_true, prob_pred)
    ]


def save_artifacts(model, report: dict) -> None:
    suffix = "_symmetric" if report.get("symmetric_output") else ""
    model_path = MODEL_DIR / f"tennis_model_v2_retrained{suffix}.pkl"
    features_path = MODEL_DIR / f"feature_cols_retrained{suffix}.json"
    report_path = MODEL_DIR / f"tennis_model_v2_retrained{suffix}_report.json"

    joblib.dump(model, model_path)
    with open(features_path, "w", encoding="utf-8") as handle:
        json.dump(FEATURE_COLS, handle)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved candidate model to {model_path}")
    print(f"Saved candidate features to {features_path}")
    print(f"Saved training report to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        choices=["notebook", "shortlist", "full"],
        default="shortlist",
        help="Candidate set to evaluate on validation data.",
    )
    parser.add_argument(
        "--refit-train-val",
        action="store_true",
        help="After validation-based selection, refit an uncalibrated model on train+val.",
    )
    parser.add_argument(
        "--calibration",
        choices=["off", "compare", "sigmoid", "isotonic"],
        default="off",
        help="Calibrate the validation-selected model on the validation split.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save a candidate model, feature list, and JSON report under model/.",
    )
    parser.add_argument(
        "--symmetric-output",
        action="store_true",
        help="Wrap the final model so predict_proba is exactly symmetric for swapped players.",
    )
    parser.add_argument(
        "--symmetry-sample-size",
        type=int,
        default=500,
        help="Number of positive-labeled test rows to sample for raw symmetry diagnostics.",
    )
    args = parser.parse_args()

    raw_df = load_raw_matches()
    clean_df = clean_matches(raw_df)
    feature_df = build_feature_history(clean_df)
    model_df = build_model_rows(feature_df)
    train_df, val_df, test_df = split_model_data(model_df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["target"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"]

    print(f"Clean rows: {len(clean_df)}")
    print(
        "Split sizes:",
        {
            "train": X_train.shape,
            "val": X_val.shape,
            "test": X_test.shape,
        },
    )

    candidates = select_candidates(args.grid)
    leaderboard = score_candidates(X_train, y_train, X_val, y_val, candidates)
    best = leaderboard[0]
    best_params = best["params"]
    print("Best validation candidate:", best_params)

    if args.calibration == "off":
        final_model = fit_base_model(X_train, y_train, best_params)
        selected_calibration = {"mode": "none"}
        calibration_comparison = []
    else:
        modes = ["none", "sigmoid", "isotonic"] if args.calibration == "compare" else [args.calibration]
        final_model, selected_calibration, calibration_comparison = evaluate_calibration_modes(
            best_params,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            modes,
        )

    if args.refit_train_val and args.calibration != "off":
        print("Skipping train+val refit because calibration uses validation as the calibration split.")

    if args.refit_train_val and args.calibration == "off":
        X_refit = pd.concat([X_train, X_val], axis=0)
        y_refit = pd.concat([y_train, y_val], axis=0)
        final_model = fit_base_model(X_refit, y_refit, best_params)
        refit_label = "train_plus_val"
        refit_X = X_refit
        refit_y = y_refit
    else:
        refit_label = "train" if args.calibration == "off" else "train_calibrated_on_val"
        refit_X = X_train
        refit_y = y_train

    if args.symmetric_output:
        final_model = SymmetricMatchupModel(final_model, FEATURE_COLS)

    refit_metrics = evaluate_binary_model(final_model, refit_X, refit_y, refit_label)
    test_metrics = evaluate_binary_model(final_model, X_test, y_test, "test")
    symmetry = symmetry_diagnostics(final_model, test_df, args.symmetry_sample_size)
    calibration = calibration_summary(final_model, X_test, y_test)

    report = {
        "grid_mode": args.grid,
        "refit_label": refit_label,
        "symmetric_output": args.symmetric_output,
        "feature_count": len(FEATURE_COLS),
        "split_summary": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_date_min": str(train_df["match_date"].min().date()),
            "train_date_max": str(train_df["match_date"].max().date()),
            "val_date_min": str(val_df["match_date"].min().date()),
            "val_date_max": str(val_df["match_date"].max().date()),
            "test_date_min": str(test_df["match_date"].min().date()),
            "test_date_max": str(test_df["match_date"].max().date()),
        },
        "selected_params": best_params,
        "selected_calibration": selected_calibration,
        "validation_leaderboard": leaderboard,
        "calibration_comparison": calibration_comparison,
        "refit_metrics": refit_metrics,
        "test_metrics": test_metrics,
        "symmetry": symmetry,
        "calibration": calibration,
    }

    print("Refit metrics:", refit_metrics)
    print("Final test metrics:", test_metrics)
    print("Raw symmetry diagnostics:", symmetry)

    if args.save_artifacts:
        save_artifacts(final_model, report)


if __name__ == "__main__":
    main()
