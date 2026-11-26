"""Candidate models. Baselines are first-class citizens: a model only counts
if it beats them under the same leak-free backtest."""
from __future__ import annotations

from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import config
from .features import BASE_FEATURES

# Each entry: fn(Xtr, ytr_log, Xte, global_mean_log) -> predicted orders (>= 0)
Predictor = Callable[[pd.DataFrame, np.ndarray, pd.DataFrame, float], np.ndarray]


def _orders(log_pred: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(log_pred), 0.0)


# ---------------------------------------------------------------- baselines
def global_mean(Xtr, ytr, Xte, gm):
    return np.full(len(Xte), float(_orders(np.array([gm]))[0]))


def meal_average(Xtr, ytr, Xte, gm):
    return _orders(Xte["meal_enc"].to_numpy())


def pair_average(Xtr, ytr, Xte, gm):
    """Smoothed historical average of the center-meal pair (what a planner would do in Excel)."""
    return _orders(Xte["pair_enc"].to_numpy())


def last_value(Xtr, ytr, Xte, gm):
    """Persistence: last observed demand of the pair, falling back to the pair average."""
    ll = Xte["last_ly"].fillna(Xte["pair_enc"]).to_numpy()
    return _orders(ll)


# ---------------------------------------------------------------- ML models
def fit_lgbm(Xtr: pd.DataFrame, ytr_log: np.ndarray, params: dict | None = None) -> lgb.LGBMRegressor:
    m = lgb.LGBMRegressor(**{**config.LGBM_PARAMS, **(params or {})})
    m.fit(Xtr, ytr_log)
    return m


def lightgbm(Xtr, ytr, Xte, gm):
    return _orders(fit_lgbm(Xtr, ytr).predict(Xte))


def lightgbm_no_history(Xtr, ytr, Xte, gm):
    """Ablation: same learner but without any history-derived feature."""
    cols = [c for c in BASE_FEATURES if c in Xtr.columns]
    return _orders(fit_lgbm(Xtr[cols], ytr).predict(Xte[cols]))


def random_forest(Xtr, ytr, Xte, gm):
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=3,
                               random_state=42, n_jobs=-1)
    rf.fit(Xtr.fillna(-1), ytr)
    return _orders(rf.predict(Xte.fillna(-1)))


MODELS: dict[str, Predictor] = {
    "global_mean": global_mean,
    "meal_average": meal_average,
    "pair_average": pair_average,
    "last_value": last_value,
    "random_forest": random_forest,
    "lightgbm_no_history": lightgbm_no_history,
    "lightgbm": lightgbm,
}
BASELINES = ["global_mean", "meal_average", "pair_average", "last_value"]
CHAMPION = "lightgbm"

LABELS = {
    "global_mean": "Global mean",
    "meal_average": "Meal average",
    "pair_average": "Center×meal average",
    "last_value": "Last value",
    "random_forest": "Random Forest",
    "lightgbm_no_history": "LightGBM (no history feats)",
    "lightgbm": "LightGBM (NARX-style)",
}
