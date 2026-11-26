"""Forecast-accuracy metrics. WMAPE and RMSLE are primary; MAPE is deliberately
not used because small order counts make it explode (the original project
reported a MAPE above 200%)."""
from __future__ import annotations

import numpy as np


def wmape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))


def rmsle(y, p) -> float:
    y, p = np.asarray(y, float), np.maximum(np.asarray(p, float), 0)
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def bias(y, p) -> float:
    """Signed error as % of actual volume. Negative = under-forecasting."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((p.sum() - y.sum()) / max(y.sum(), 1e-9))


def all_metrics(y, p) -> dict:
    return {"WMAPE": wmape(y, p), "RMSLE": rmsle(y, p), "MAE": mae(y, p),
            "RMSE": rmse(y, p), "Bias": bias(y, p)}
