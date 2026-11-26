import pytest
from fdf.uploads import historical_panel, batch_rows
from fdf.service import REQUEST_COLUMNS

def test_history(dense_panel):
    assert len(historical_panel(dense_panel)) == len(dense_panel)

def test_duplicate(dense_panel):
    import pandas as pd
    with pytest.raises(ValueError, match='Duplicate'):
        historical_panel(pd.concat([dense_panel, dense_panel.iloc[:1]]))

def test_batch(dense_panel):
    rows = dense_panel[REQUEST_COLUMNS].head(2).copy()
    rows['week'] = 41
    assert len(batch_rows(rows, 40)) == 2
    with pytest.raises(ValueError, match='after'):
        batch_rows(rows, 41)
