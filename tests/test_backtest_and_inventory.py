import numpy as np
import pytest

from fdf.backtest import make_cutoffs, run_backtest, summarize
from fdf.config import Economics
from fdf.inventory import ResidualQuantiles, cost_breakdown, order_quantity
from fdf.metrics import bias, rmsle, wmape


def test_metrics_basic():
    y = np.array([100, 200, 300])
    assert wmape(y, y) == 0 and rmsle(y, y) == 0
    assert wmape(y, y * 0 + 100) == pytest.approx(300 / 600)
    assert bias(y, y * 0.5) == pytest.approx(-0.5)


def test_cutoffs_end_before_horizon():
    c = make_cutoffs(145, horizon=10, n_folds=6, step=8)
    assert c[-1] == 135 and len(c) == 6 and c == sorted(c)


def test_backtest_uses_only_past_and_beats_global_mean(dense_panel):
    oof = run_backtest(dense_panel, cutoffs=[20, 25, 30], horizon=5, models=["global_mean", "pair_average", "lightgbm"])
    assert (oof.week > oof.cutoff).all() and (oof.week <= oof.cutoff + 5).all()
    s = summarize(oof, ["global_mean", "pair_average", "lightgbm"])
    assert s.loc["pair_average", "WMAPE"] < s.loc["global_mean", "WMAPE"]


def test_critical_ratio():
    assert Economics(0.4, 0.25).critical_ratio == pytest.approx(0.4 / 0.65)


def test_order_quantity_monotonic_in_service_level():
    rng = np.random.default_rng(1)
    cal = ResidualQuantiles(rng.normal(0, 0.4, 500))
    q = [order_quantity([200.0], cal, s)[0] for s in (0.5, 0.7, 0.9, 0.99)]
    assert q == sorted(q) and q[-1] > q[0]


def test_cost_breakdown_accounting():
    econ = Economics(0.5, 0.2)
    r = cost_breakdown(demand=[100, 100], stock=[150, 60], price=[10, 10], econ=econ)
    assert r["waste_cost"] == pytest.approx(50 * 10 * 0.2)
    assert r["lost_margin"] == pytest.approx(40 * 10 * 0.5)
    assert r["fill_rate"] == pytest.approx(160 / 200)


def test_calibrator_needs_enough_data():
    with pytest.raises(ValueError):
        ResidualQuantiles(np.zeros(5))
