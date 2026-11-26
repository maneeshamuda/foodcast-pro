"""Inference service: one `Forecaster` shared by the API and the dashboard."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .config import Economics
from .features import build_features
from .inventory import ResidualQuantiles, cost_breakdown, order_quantity

REQUEST_COLUMNS = ["week", "center_id", "meal_id", "checkout_price", "base_price",
                   "emailer_for_promotion", "homepage_featured"]


class Forecaster:
    def __init__(self, bundle: dict):
        self.b = bundle
        self.model = bundle["model"]
        self.history: pd.DataFrame = bundle["history"]
        self.static = bundle.get("static") or None
        self.calibrator = ResidualQuantiles(bundle["residuals"])
        self.meta = bundle["meta"]
        self.known_centers = set(self.history.center_id.unique())
        self.known_meals = set(self.history.meal_id.unique())

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Forecaster":
        path = Path(path or config.ARTIFACT_PATH)
        if not path.exists():
            raise FileNotFoundError(f"No model artifact at {path}. Run `python -m fdf.train` first.")
        return cls(joblib.load(path))

    # ------------------------------------------------------------------
    def predict(self, rows: pd.DataFrame, service_level: float | None = None,
                econ: Economics = config.DEFAULT_ECONOMICS) -> pd.DataFrame:
        """Point forecast, 80% interval, cold-start flag and recommended stock."""
        missing = [c for c in REQUEST_COLUMNS if c not in rows.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        rows = rows[REQUEST_COLUMNS].reset_index(drop=True)
        X = build_features(rows, self.history, self.static)[self.b["features"]]
        point = np.maximum(np.expm1(self.model.predict(X)), 0.0)
        lo, hi = self.calibrator.interval(point, 0.8)
        sl = econ.critical_ratio if service_level is None else service_level
        out = rows.copy()
        out["forecast"] = point
        out["lower_80"], out["upper_80"] = lo, hi
        out["service_level"] = sl
        out["recommended_stock"] = order_quantity(point, self.calibrator, sl)
        out["history_points"] = X["pair_n"].to_numpy().astype(int)
        out["cold_start"] = (out.history_points == 0)
        out["unknown_center"] = ~out.center_id.isin(self.known_centers)
        out["unknown_meal"] = ~out.meal_id.isin(self.known_meals)
        return out

    def expected_cost(self, forecast_row: pd.Series, stock: float, econ: Economics) -> dict:
        """Expected waste / stock-out cost for a candidate stock level, from the residual distribution."""
        qs = np.linspace(0.01, 0.99, 99)
        sims = self.calibrator.quantile(np.full(len(qs), forecast_row["forecast"]), qs)
        return cost_breakdown(sims, np.full(len(qs), stock), np.full(len(qs), forecast_row["checkout_price"]), econ)
