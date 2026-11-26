import numpy as np

from fdf.features import build_features


def test_no_leakage_from_future_rows(dense_panel):
    """Changing the target of rows at/after week t must not change features of week-t rows."""
    t = 25
    rows = dense_panel[dense_panel.week == t]
    hist = dense_panel
    X1 = build_features(rows, hist)
    tampered = hist.copy()
    tampered.loc[tampered.week >= t, "num_orders"] *= 50
    X2 = build_features(rows, tampered)
    np.testing.assert_allclose(X1.fillna(-1).to_numpy(), X2.fillna(-1).to_numpy())


def test_last_value_is_previous_week_in_dense_panel(dense_panel):
    rows = dense_panel[dense_panel.week == 10]
    X = build_features(rows, dense_panel)
    prev = dense_panel[dense_panel.week == 9].set_index(["center_id", "meal_id"]).num_orders
    expected = np.log1p(prev.loc[list(zip(rows.center_id, rows.meal_id, strict=True))].to_numpy())
    np.testing.assert_allclose(X.last_ly.to_numpy(), expected)
    assert (X.last_age == 1).all()


def test_cold_start_pair_has_nan_history(dense_panel):
    rows = dense_panel[dense_panel.week == 5].iloc[:1].copy()
    rows["center_id"] = 999
    X = build_features(rows, dense_panel)
    assert X.pair_n.iloc[0] == 0 and np.isnan(X.last_ly.iloc[0])
    assert np.isfinite(X.pair_enc.iloc[0])  # falls back to the smoothed prior


def test_first_week_has_no_history(dense_panel):
    X = build_features(dense_panel[dense_panel.week == 1], dense_panel)
    assert (X.pair_n == 0).all()
