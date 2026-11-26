import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def dense_panel() -> pd.DataFrame:
    """Small synthetic DENSE panel (every pair observed every week) for logic tests."""
    rng = np.random.default_rng(0)
    rows, i = [], 0
    for week in range(1, 41):
        for c in (1, 2, 3):
            for m in (10, 20):
                base = 100 + 40 * m / 10 + 15 * c
                promo = int(rng.random() < 0.15)
                price = 200 * (1 - 0.1 * promo)
                demand = base * (1 + 0.5 * promo) * (1 + 0.2 * np.sin(2 * np.pi * week / 52)) * rng.lognormal(0, 0.1)
                rows.append(dict(id=i, week=week, center_id=c, meal_id=m, checkout_price=price,
                                 base_price=200.0, emailer_for_promotion=promo, homepage_featured=0,
                                 num_orders=int(demand)))
                i += 1
    return pd.DataFrame(rows)
