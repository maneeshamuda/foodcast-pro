"""Validated session-local data and batch planning inputs."""
import numpy as np
import pandas as pd
from . import config
from .data import clean, validate_schema
from .service import REQUEST_COLUMNS
from api.schemas import ForecastRequest


def read_table(file):
    if file.name.lower().endswith('.xlsx'):
        return pd.read_excel(file, engine='openpyxl')
    if file.name.lower().endswith('.csv'):
        return pd.read_csv(file)
    raise ValueError('Use CSV or XLSX.')


def historical_panel(frame):
    validate_schema(frame)
    frame = frame[config.RAW_COLUMNS].apply(pd.to_numeric, errors='raise')
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('History must contain finite numeric rows.')
    for col in ['id','week','center_id','meal_id','emailer_for_promotion','homepage_featured','num_orders']:
        if (frame[col] < 0).any() or (frame[col] % 1 != 0).any():
            raise ValueError(f'{col} must contain nonnegative integers.')
    if frame.duplicated(['week','center_id','meal_id']).any():
        raise ValueError('Duplicate week/center/meal rows; aggregate them first.')
    if not frame[['emailer_for_promotion','homepage_featured']].isin([0,1]).all().all():
        raise ValueError('Promotion flags must be 0 or 1.')
    result = clean(frame)
    if len(result) != len(frame):
        raise ValueError('Invalid week, price or demand values.')
    return result


def batch_rows(frame, last_week):
    validate_schema(frame, REQUEST_COLUMNS)
    if not 1 <= len(frame) <= 5000:
        raise ValueError('Upload 1–5000 forecast rows.')
    rows = pd.DataFrame([ForecastRequest(**r).model_dump() for r in frame[REQUEST_COLUMNS].to_dict('records')])
    if (rows.week <= last_week).any():
        raise ValueError(f'Forecast weeks must be after week {last_week}.')
    return rows
