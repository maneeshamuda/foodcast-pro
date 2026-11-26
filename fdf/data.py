"""Loading, validation and cleaning of the weekly demand panel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


class DataValidationError(ValueError):
    """Raised when the input data violates the expected schema."""


def resolve_data_path(data_dir: Path | None = None) -> Path:
    """Prefer the full dataset (`train.csv`) if present, else the bundled sample."""
    data_dir = Path(data_dir or config.DATA_DIR)
    full = data_dir / config.FULL_DATA_FILE
    return full if full.exists() else data_dir / config.SAMPLE_DATA_FILE


def validate_schema(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    required = columns or config.RAW_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop impossible rows and exact duplicates.

    A missing target is dropped, never imputed: treating it as 0 would teach
    the model that demand collapses to zero.
    """
    validate_schema(df)
    df = df.drop_duplicates().copy()
    for col in config.RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["week", "center_id", "meal_id", "checkout_price",
                           "base_price", config.TARGET])
    df = df[(df.week > 0) & (df.checkout_price > 0) & (df.base_price > 0)
            & (df[config.TARGET] >= 0)]
    for col in ["emailer_for_promotion", "homepage_featured"]:
        df[col] = df[col].fillna(0).clip(0, 1)
    int_cols = ["id", "week", "center_id", "meal_id", "emailer_for_promotion", "homepage_featured"]
    df[int_cols] = df[int_cols].astype(int)
    return df.sort_values(["week", "center_id", "meal_id"]).reset_index(drop=True)


def load_static_tables(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Optional metadata files from the full Genpact dataset.

    Returns integer-coded tables keyed by `center_id` / `meal_id`. Empty dict if
    the files are not present (the case for the bundled sample).
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    tables: dict[str, pd.DataFrame] = {}
    for fname, key in [(config.CENTER_INFO_FILE, "center_id"), (config.MEAL_INFO_FILE, "meal_id")]:
        path = data_dir / fname
        if not path.exists():
            continue
        t = pd.read_csv(path).drop_duplicates(key).set_index(key)
        for col in t.columns:
            if t[col].dtype == object:
                t[col] = t[col].astype("category").cat.codes
            t = t.rename(columns={col: f"{key.split('_')[0]}_{col}"})
        tables[key] = t.astype(float)
    return tables


def load_panel(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else resolve_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return clean(pd.read_csv(path))


def describe_panel(df: pd.DataFrame) -> dict:
    """Facts about panel density. Used for honest reporting in the UI/README."""
    pairs = df.groupby(["center_id", "meal_id"]).size()
    return {
        "rows": int(len(df)),
        "weeks": int(df.week.nunique()),
        "first_week": int(df.week.min()),
        "last_week": int(df.week.max()),
        "centers": int(df.center_id.nunique()),
        "meals": int(df.meal_id.nunique()),
        "center_meal_pairs": int(len(pairs)),
        "median_rows_per_pair": float(np.median(pairs)),
        "pair_density": float(len(df) / (len(pairs) * df.week.nunique())),
    }
