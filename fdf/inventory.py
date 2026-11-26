"""From forecast to decision: prediction intervals and newsvendor order quantities.

Perishable stock is a newsvendor problem. With a per-unit profit lost on a
stock-out (Cu) and a per-unit cost lost on waste (Co), the order quantity that
minimises expected cost is the demand quantile at the critical ratio
Cu / (Cu + Co).

Quantiles come from the empirical distribution of out-of-sample log-residuals
collected in the walk-forward backtest (a split-conformal style calibration),
so they need no distributional assumption and work for any service level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .config import Economics


class ResidualQuantiles:
    """Empirical quantiles of log1p(actual) - log1p(forecast)."""

    def __init__(self, residuals: np.ndarray):
        r = np.asarray(residuals, float)
        r = r[np.isfinite(r)]
        if len(r) < 20:
            raise ValueError("Need at least 20 out-of-sample residuals to calibrate intervals.")
        self.residuals = np.sort(r)

    def quantile(self, point_orders, q: float) -> np.ndarray:
        shift = np.quantile(self.residuals, q)
        return np.maximum(np.expm1(np.log1p(np.asarray(point_orders, float)) + shift), 0.0)

    def interval(self, point_orders, level: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        a = (1 - level) / 2
        return self.quantile(point_orders, a), self.quantile(point_orders, 1 - a)


def order_quantity(point_orders, calibrator: ResidualQuantiles, service_level: float) -> np.ndarray:
    """Units to stock so that P(demand <= stock) ~= service_level."""
    return np.ceil(calibrator.quantile(point_orders, service_level))


def cost_breakdown(demand, stock, price, econ: Economics) -> dict:
    d, q, p = (np.asarray(x, float) for x in (demand, stock, price))
    over = np.maximum(q - d, 0.0)
    under = np.maximum(d - q, 0.0)
    waste_cost = float((over * p * econ.waste_pct).sum())
    lost_margin = float((under * p * econ.margin_pct).sum())
    return {
        "waste_cost": waste_cost,
        "lost_margin": lost_margin,
        "total_cost": waste_cost + lost_margin,
        "fill_rate": float(np.minimum(q, d).sum() / max(d.sum(), 1e-9)),
        "waste_rate": float(over.sum() / max(q.sum(), 1e-9)),
    }


def walk_forward_policy_eval(oof: pd.DataFrame, models: list[str], econ: Economics = config.DEFAULT_ECONOMICS,
                             service_level: float | None = None) -> pd.DataFrame:
    """Simulate ordering policies on out-of-sample weeks.

    For fold i the order quantity uses residual quantiles estimated ONLY on
    folds < i, so fold 0 (nothing to calibrate on) is excluded. Every model,
    baselines included, gets the same calibrated safety-stock treatment, which
    isolates the value of better point forecasts.
    """
    service_level = econ.critical_ratio if service_level is None else service_level
    folds = sorted(oof.fold.unique())
    rows = []
    for m in models:
        parts = []
        for f in folds[1:]:
            past = oof[oof.fold < f]
            cal = ResidualQuantiles(np.log1p(past[config.TARGET]) - np.log1p(past[m]))
            cur = oof[oof.fold == f]
            stock = order_quantity(cur[m], cal, service_level)
            parts.append(pd.DataFrame({"d": cur[config.TARGET], "q": stock, "p": cur["checkout_price"]}))
        allp = pd.concat(parts)
        rows.append({"model": m, **cost_breakdown(allp.d, allp.q, allp.p, econ)})
    return pd.DataFrame(rows).set_index("model")


def interval_coverage(oof: pd.DataFrame, model: str, level: float = 0.8) -> dict:
    """Walk-forward empirical coverage of the central `level` prediction interval."""
    covered, widths, n = 0, [], 0
    for f in sorted(oof.fold.unique())[1:]:
        past, cur = oof[oof.fold < f], oof[oof.fold == f]
        cal = ResidualQuantiles(np.log1p(past[config.TARGET]) - np.log1p(past[model]))
        lo, hi = cal.interval(cur[model], level)
        y = cur[config.TARGET].to_numpy()
        covered += int(((y >= lo) & (y <= hi)).sum())
        n += len(y)
        widths.append(np.mean(hi - lo))
    return {"nominal": level, "empirical": covered / max(n, 1), "n": n}
