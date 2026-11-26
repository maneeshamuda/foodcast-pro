"""Leak-free feature engineering.

Core rule: a row's features may only use observations from STRICTLY EARLIER
weeks. Every history-derived feature is computed with `pd.merge_asof(...,
allow_exact_matches=False)`, so the same function is correct for
  * training rows   (history = the training set itself, earlier weeks only),
  * backtest rows   (history = everything up to the forecast cutoff),
  * live requests   (history = everything the model was trained on).

The data is a panel of (center, meal) series observed at irregular weeks, so
"lag-1" is usually undefined. Instead of a fixed lag the model gets the last
observed demand for the pair and how many weeks ago it was seen (the
autoregressive part of a NARX-style model), plus smoothed target encodings of
the meal, the center and the pair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

KEYS_MEAL = ["meal_id"]
KEYS_CENTER = ["center_id"]
KEYS_PAIR = ["center_id", "meal_id"]

BASE_FEATURES = [
    "week", "woy_sin", "woy_cos", "checkout_price", "base_price", "log_price",
    "discount", "discount_pct", "emailer_for_promotion", "homepage_featured",
    "promo_score",
]
HISTORY_FEATURES = [
    "meal_enc", "center_enc", "pair_enc", "pair_n", "last_ly", "last_age",
    "price_vs_meal", "meal_n",
]


def _cum_stats(hist: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Cumulative (sum ly, sum log-price, count) per key up to and including each week."""
    g = (hist.groupby([*keys, "week"])
         .agg(s=("ly", "sum"), sp=("lp", "sum"), n=("ly", "size"))
         .reset_index())
    g = g.sort_values([*keys, "week"])
    g[["s", "sp", "n"]] = g.groupby(keys)[["s", "sp", "n"]].cumsum()
    return g.sort_values("week")


def _asof(rows: pd.DataFrame, table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    left = rows[[*keys, "week"]].copy()
    left["_pos"] = np.arange(len(left))
    left = left.sort_values("week")
    merged = pd.merge_asof(left, table, on="week", by=keys, allow_exact_matches=False)
    return merged.sort_values("_pos").reset_index(drop=True)


def _asof_global_mean(rows: pd.DataFrame, hist: pd.DataFrame) -> np.ndarray:
    """Mean log-demand over all strictly-earlier weeks (the shrinkage prior).

    Computed as-of per row so even this scalar never sees the future."""
    g = hist.groupby("week").agg(s=("ly", "sum"), n=("ly", "size")).reset_index().sort_values("week")
    g[["s", "n"]] = g[["s", "n"]].cumsum()
    left = rows[["week"]].copy()
    left["_pos"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("week"), g, on="week", allow_exact_matches=False)
    m = m.sort_values("_pos")
    return (m["s"] / m["n"]).fillna(config.DEFAULT_LOG_PRIOR).to_numpy()


def build_features(rows: pd.DataFrame, hist: pd.DataFrame, static: dict | None = None) -> pd.DataFrame:
    """Return the model matrix for `rows`, using only `hist` rows from earlier weeks."""
    rows = rows.reset_index(drop=True)
    hist = hist.copy()
    hist["ly"] = np.log1p(hist[config.TARGET])
    hist["lp"] = np.log(hist["checkout_price"])
    gm = _asof_global_mean(rows, hist)

    X = pd.DataFrame(index=rows.index)
    X["week"] = rows["week"].astype(float)
    ang = 2 * np.pi * rows["week"] / config.SEASON_PERIOD
    X["woy_sin"], X["woy_cos"] = np.sin(ang), np.cos(ang)
    X["checkout_price"] = rows["checkout_price"]
    X["base_price"] = rows["base_price"]
    X["log_price"] = np.log(rows["checkout_price"])
    X["discount"] = rows["base_price"] - rows["checkout_price"]
    X["discount_pct"] = X["discount"] / rows["base_price"]
    X["emailer_for_promotion"] = rows["emailer_for_promotion"]
    X["homepage_featured"] = rows["homepage_featured"]
    X["promo_score"] = rows["emailer_for_promotion"] + rows["homepage_featured"]

    k_g, k_p = config.PRIOR_STRENGTH_GROUP, config.PRIOR_STRENGTH_PAIR

    meal = _asof(rows, _cum_stats(hist, KEYS_MEAL), KEYS_MEAL)
    cen = _asof(rows, _cum_stats(hist, KEYS_CENTER), KEYS_CENTER)
    pair = _asof(rows, _cum_stats(hist, KEYS_PAIR), KEYS_PAIR)
    for t in (meal, cen, pair):
        t[["s", "sp", "n"]] = t[["s", "sp", "n"]].fillna(0.0)

    X["meal_n"] = meal["n"]
    X["meal_enc"] = (meal["s"] + k_g * gm) / (meal["n"] + k_g)
    X["center_enc"] = (cen["s"] + k_g * gm) / (cen["n"] + k_g)
    prior = X["meal_enc"] + X["center_enc"] - gm  # additive meal + center prior for the pair
    X["pair_enc"] = (pair["s"] + k_p * prior) / (pair["n"] + k_p)
    X["pair_n"] = pair["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        meal_mean_lp = np.where(meal["n"] > 0, meal["sp"] / meal["n"], np.nan)
    X["price_vs_meal"] = X["log_price"] - meal_mean_lp

    last = hist[[*KEYS_PAIR, "week", "ly"]].rename(columns={"ly": "last_ly"})
    last["last_week"] = last["week"]
    last = last.sort_values("week")
    left = rows[[*KEYS_PAIR, "week"]].copy()
    left["_pos"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("week"), last, on="week", by=KEYS_PAIR,
                      allow_exact_matches=False).sort_values("_pos").reset_index(drop=True)
    X["last_ly"] = m["last_ly"]
    X["last_age"] = rows["week"] - m["last_week"]

    if static:
        for key, table in static.items():
            joined = rows[[key]].join(table, on=key).drop(columns=[key])
            for col in joined.columns:
                X[col] = joined[col].to_numpy()
    return X


def feature_columns(X: pd.DataFrame) -> list[str]:
    return list(X.columns)
