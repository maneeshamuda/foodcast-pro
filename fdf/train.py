"""Training entry point:  python -m fdf.train

1. validates + cleans the data, 2. runs the walk-forward backtest for every
model, 3. evaluates intervals and ordering policies, 4. fits the champion on all
data, 5. writes the model bundle, metrics.json and report figures.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import __version__, config, data  # noqa: E402
from .backtest import make_cutoffs, run_backtest, summarize  # noqa: E402
from .features import build_features  # noqa: E402
from .inventory import interval_coverage, walk_forward_policy_eval  # noqa: E402
from .models import BASELINES, CHAMPION, LABELS, MODELS, fit_lgbm  # noqa: E402

PALETTE = {"base": "#9CA3AF", "ml": "#2563EB", "champ": "#F97316", "ink": "#111827"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#D1D5DB", "axes.labelcolor": PALETTE["ink"], "text.color": PALETTE["ink"],
        "xtick.color": "#4B5563", "ytick.color": "#4B5563", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.grid": True, "grid.color": "#F3F4F6", "axes.axisbelow": True,
    })


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name)
    plt.close(fig)


def make_figures(oof, summary, policy, model_champion, feature_names):
    _style()
    order = summary.sort_values("WMAPE", ascending=False).index.tolist()
    colors = [PALETTE["champ"] if m == CHAMPION else PALETTE["base"] if m in BASELINES else PALETTE["ml"] for m in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([LABELS[m] for m in order], summary.loc[order, "WMAPE"], color=colors)
    for i, m in enumerate(order):
        ax.text(summary.loc[m, "WMAPE"] + 0.01, i, f"{summary.loc[m, 'WMAPE']:.3f}", va="center", fontsize=9)
    ax.set_xlabel("WMAPE, lower is better (pooled over 6 walk-forward folds)")
    ax.set_title("Model comparison: gray = baselines, orange = champion")
    ax.grid(axis="y", visible=False)
    _save(fig, "model_comparison.png")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    y, p = oof[config.TARGET], oof[CHAMPION]
    cap = float(np.quantile(y, 0.99))
    ax.scatter(p.clip(upper=cap), y.clip(upper=cap), s=10, alpha=0.45, color=PALETTE["ml"], edgecolor="none")
    ax.plot([0, cap], [0, cap], color=PALETTE["champ"], lw=1.5, label="perfect forecast")
    ax.set_xlabel("Forecast orders")
    ax.set_ylabel("Actual orders")
    ax.set_title("Out-of-sample: forecast vs actual")
    ax.legend(frameon=False)
    _save(fig, "actual_vs_forecast.png")

    imp = pd.Series(model_champion.booster_.feature_importance("gain"), index=feature_names)
    imp = (imp / imp.sum()).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(imp.index, imp.values, color=PALETTE["ml"])
    ax.grid(axis="y", visible=False)
    ax.set_title("Feature importance (share of total gain)")
    _save(fig, "feature_importance.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [PALETTE["champ"] if m == CHAMPION else PALETTE["base"] for m in policy.index]
    ax.bar([LABELS[m] for m in policy.index], policy.total_cost / 1e6, color=cols)
    ax.set_ylabel("Simulated cost (millions)")
    ax.set_title("Ordering-policy simulation: waste + lost margin")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    _save(fig, "inventory_policy_cost.png")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="Path to the demand CSV (default: auto-detect).")
    ap.add_argument("--quick", action="store_true", help="Skip the slow Random Forest (for CI).")
    args = ap.parse_args(argv)

    path = args.data or data.resolve_data_path()
    print(f"Loading {path}")
    panel = data.load_panel(path)
    static = data.load_static_tables()
    info = data.describe_panel(panel)
    print(info)

    models = [m for m in MODELS if not (args.quick and m == "random_forest")]
    cutoffs = make_cutoffs(int(panel.week.max()))
    print(f"Walk-forward folds (cutoffs): {cutoffs}, horizon {config.HORIZON} weeks")
    oof = run_backtest(panel, cutoffs, models=models, static=static or None, verbose=True)
    summary = summarize(oof, models)
    print(summary.round(3).to_string())

    policy_models = ["pair_average", "last_value", CHAMPION]
    policy = walk_forward_policy_eval(oof, policy_models)
    print(policy.round(3).to_string())
    coverage = interval_coverage(oof, CHAMPION, 0.8)
    print("80% interval coverage:", coverage)

    # ---- final fit on all data
    ytr = np.log1p(panel[config.TARGET].to_numpy())
    Xall = build_features(panel, panel, static or None)
    champion = fit_lgbm(Xall, ytr)

    # residuals for interval calibration = every out-of-sample backtest residual
    resid = np.log1p(oof[config.TARGET]) - np.log1p(oof[CHAMPION])
    history_cols = config.RAW_COLUMNS
    bundle = {
        "model": champion,
        "features": list(Xall.columns),
        "history": panel[history_cols].copy(),
        "static": static,
                "residuals": resid.to_numpy(),
        "meta": {
            "version": __version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_file": str(path).replace("\\", "/").split("/")[-1],
            "panel": info,
            "champion": CHAMPION,
            "horizon_weeks": config.HORIZON,
            "cutoffs": cutoffs,
            "metrics": summary.reset_index().to_dict(orient="records"),
            "policy": policy.reset_index().to_dict(orient="records"),
            "coverage": coverage,
            "economics": {"margin_pct": config.DEFAULT_ECONOMICS.margin_pct,
                          "waste_pct": config.DEFAULT_ECONOMICS.waste_pct},
        },
    }
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, config.ARTIFACT_PATH, compress=3)
    config.REPORT_DIR.mkdir(exist_ok=True)
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(bundle["meta"], indent=2, default=float))
    oof.to_csv(config.REPORT_DIR / "backtest_predictions.csv", index=False)
    make_figures(oof, summary, policy, champion, list(Xall.columns))
    print(f"Saved {config.ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
