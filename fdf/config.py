"""Central configuration. Everything tunable lives here, nowhere else."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# Drop the full Kaggle/Genpact `train.csv` into data/raw/ and it is picked up
# automatically (see data.resolve_data_path). Otherwise the bundled sample is used.
FULL_DATA_FILE = "train.csv"
SAMPLE_DATA_FILE = "food_demand_sample.csv"
CENTER_INFO_FILE = "fulfilment_center_info.csv"
MEAL_INFO_FILE = "meal_info.csv"

ARTIFACT_PATH = Path(os.environ.get("FDF_ARTIFACT", MODEL_DIR / "foodcast_bundle.joblib"))

TARGET = "num_orders"
RAW_COLUMNS = [
    "id", "week", "center_id", "meal_id", "checkout_price", "base_price",
    "emailer_for_promotion", "homepage_featured", "num_orders",
]

# Rolling-origin backtest: forecast `HORIZON` weeks after every cutoff.
HORIZON = 10
N_FOLDS = 6
FOLD_STEP = 8  # weeks between consecutive cutoffs

# Smoothing strength (pseudo-observations) for the as-of target encodings.
PRIOR_STRENGTH_GROUP = 5.0
PRIOR_STRENGTH_PAIR = 2.0

DEFAULT_LOG_PRIOR = 5.5  # log1p(~245 orders): used only when there is no earlier week at all
SEASON_PERIOD = 52  # weeks per year

# Chosen from a small sweep on the same backtest (gains over defaults were ~2-3%,
# i.e. within fold noise), so treat the reported score as mildly optimistic.
LGBM_PARAMS = dict(
    n_estimators=800, learning_rate=0.015, num_leaves=7, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
    random_state=42, verbose=-1,
)


@dataclass(frozen=True)
class Economics:
    """Unit economics used by the inventory recommendation.

    These are ASSUMPTIONS (the dataset has no cost data); they are exposed as
    sliders in the dashboard and as request fields in the API.
    """
    margin_pct: float = 0.40   # profit lost per unit of unmet demand, % of price
    waste_pct: float = 0.25    # cost lost per unsold unit, % of price

    @property
    def critical_ratio(self) -> float:
        return self.margin_pct / (self.margin_pct + self.waste_pct)


DEFAULT_ECONOMICS = Economics()
