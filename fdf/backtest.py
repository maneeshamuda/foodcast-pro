"""Rolling-origin (walk-forward) backtest.

For every cutoff week c: train on weeks <= c, forecast weeks (c, c + HORIZON].
Features of test rows are built from training history only, so the setup
mirrors production exactly: "it is Sunday of week c, plan the next 10 weeks".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import build_features
from .metrics import all_metrics
from .models import MODELS


def make_cutoffs(last_week: int, horizon: int = config.HORIZON, n_folds: int = config.N_FOLDS,
                 step: int = config.FOLD_STEP) -> list[int]:
    newest = last_week - horizon
    return [newest - step * i for i in range(n_folds - 1, -1, -1)]


def run_backtest(panel: pd.DataFrame, cutoffs: list[int] | None = None, horizon: int = config.HORIZON,
                 models: list[str] | None = None, static: dict | None = None,
                 verbose: bool = False) -> pd.DataFrame:
    """Returns one row per (fold, test row) with a column per model's forecast."""
    cutoffs = cutoffs or make_cutoffs(int(panel.week.max()), horizon)
    models = models or list(MODELS)
    out = []
    for fold, cut in enumerate(cutoffs):
        tr = panel[panel.week <= cut]
        te = panel[(panel.week > cut) & (panel.week <= cut + horizon)]
        if len(te) == 0 or len(tr) < 50:
            continue
        ytr = np.log1p(tr[config.TARGET].to_numpy())
        gm = float(ytr.mean())
        Xtr = build_features(tr, tr, static)
        Xte = build_features(te, tr, static)
        res = te[["week", "center_id", "meal_id", "checkout_price", config.TARGET]].reset_index(drop=True).copy()
        res.insert(0, "fold", fold)
        res.insert(1, "cutoff", cut)
        for name in models:
            res[name] = MODELS[name](Xtr, ytr, Xte, gm)
        # keep log-space champion residual inputs handy
        out.append(res)
        if verbose:
            print(f"fold {fold} cutoff={cut} train={len(tr)} test={len(te)}")
    if not out:
        raise ValueError("Backtest produced no folds; not enough data for the chosen cutoffs.")
    return pd.concat(out, ignore_index=True)


def summarize(oof: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """Pooled metrics plus fold-to-fold spread of WMAPE."""
    models = models or [c for c in MODELS if c in oof.columns]
    rows = []
    for m in models:
        pooled = all_metrics(oof[config.TARGET], oof[m])
        per_fold = [all_metrics(g[config.TARGET], g[m])["WMAPE"] for _, g in oof.groupby("fold")]
        rows.append({"model": m, **pooled, "WMAPE_fold_std": float(np.std(per_fold)),
                     "folds_won_vs_pair_average": None})
    df = pd.DataFrame(rows).set_index("model")
    if "pair_average" in oof.columns:
        base_by_fold = {f: all_metrics(g[config.TARGET], g["pair_average"])["WMAPE"] for f, g in oof.groupby("fold")}
        for m in models:
            wins = sum(all_metrics(g[config.TARGET], g[m])["WMAPE"] < base_by_fold[f] for f, g in oof.groupby("fold"))
            df.loc[m, "folds_won_vs_pair_average"] = f"{wins}/{oof.fold.nunique()}"
    return df
