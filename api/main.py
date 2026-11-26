"""FoodCast REST API.   uvicorn api.main:app --reload   ->  http://localhost:8000/docs"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from fdf import __version__, config
from fdf.config import Economics as EconomicsCfg
from fdf.service import Forecaster

from .schemas import BatchRequest, ForecastRequest, ForecastResult, RecommendRequest, RecommendResult

app = FastAPI(
    title="FoodCast API",
    version=__version__,
    description="Weekly meal-demand forecasts with calibrated 80% intervals and newsvendor stock recommendations.",
)
_state: dict = {}


def get_forecaster() -> Forecaster:
    if "f" not in _state:
        try:
            _state["f"] = Forecaster.load()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    return _state["f"]


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _frame(items: list[ForecastRequest]) -> pd.DataFrame:
    return pd.DataFrame([i.model_dump() for i in items])


_RESULT_COLS = list(ForecastResult.model_fields)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": "f" in _state or config.ARTIFACT_PATH.exists()}


@app.get("/meta", tags=["meta"])
def meta():
    """Model version, training window and backtest metrics."""
    return get_forecaster().meta


@app.post("/forecast", response_model=ForecastResult, tags=["forecast"])
def forecast(req: ForecastRequest):
    out = get_forecaster().predict(_frame([req]))
    return out.iloc[0][_RESULT_COLS].to_dict()


@app.post("/forecast/batch", response_model=list[ForecastResult], tags=["forecast"])
def forecast_batch(req: BatchRequest):
    out = get_forecaster().predict(_frame(req.items))
    return out[_RESULT_COLS].to_dict(orient="records")


@app.post("/recommend-order", response_model=RecommendResult, tags=["inventory"])
def recommend_order(req: RecommendRequest):
    """Forecast + the stock level that minimises expected waste and lost margin."""
    f = get_forecaster()
    econ = EconomicsCfg(req.economics.margin_pct, req.economics.waste_pct)
    base = ForecastRequest(**req.model_dump(include=set(ForecastRequest.model_fields)))
    out = f.predict(_frame([base]), service_level=req.service_level, econ=econ).iloc[0]
    cost = f.expected_cost(out, out["recommended_stock"], econ)
    res = out[_RESULT_COLS].to_dict()
    res.update(service_level=float(out["service_level"]), recommended_stock=int(out["recommended_stock"]),
               expected_waste_cost=cost["waste_cost"] / 99, expected_lost_margin=cost["lost_margin"] / 99)
    return res

# Business workspace: uploaded data stays request-local.
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fdf.uploads import read_table, historical_panel
from fdf.data import resolve_data_path
import json
import io
import math

app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173','http://127.0.0.1:5173'], allow_methods=['GET','POST'], allow_headers=['*'])

@app.get('/workspace/sample')
def sample():
    return FileResponse(resolve_data_path(), filename='foodcast_sample.csv')

@app.get('/workspace/catalog')
def catalog():
    d = pd.read_csv(resolve_data_path())
    return {'centers': sorted(d.center_id.unique().tolist()), 'meals': sorted(d.meal_id.unique().tolist()), 'last_week': int(d.week.max()), 'rows':len(d), 'orders':int(d.num_orders.sum())}

@app.get('/workspace/metrics')
def metrics():
    return json.loads((config.REPORT_DIR/'metrics.json').read_text())

@app.post('/workspace/upload')
async def upload(file: UploadFile = File(...)):
    raw = await file.read(5*1024*1024+1)
    if len(raw)>5*1024*1024: raise HTTPException(413,'Maximum file size is 5 MB')
    buf=io.BytesIO(raw); buf.name=file.filename or ''
    try:
        d=historical_panel(read_table(buf))
    except Exception as e:
        raise HTTPException(422, str(e)) from e
    plans=[]
    last=int(d.week.max())
    for (center,meal),g in d.groupby(['center_id','meal_id']):
        recent=g.sort_values('week').tail(4)
        if len(recent)<2 or int(recent.week.max())!=last: continue
        pred=float(recent.num_orders.mean())
        plans.append({'center':int(center),'meal':int(meal),'forecast':round(pred,1),'stock':math.ceil(pred*1.15),'observations':len(recent)})
    return {'rows':len(d),'orders':int(d.num_orders.sum()),'last_week':last,'plans':plans,'weekly':[{'week':int(w),'orders':int(n)} for w,n in d.groupby('week').num_orders.sum().items()]}

@app.get('/workspace/overview')
def overview():
    d = pd.read_csv(resolve_data_path())
    return {'weekly': [{'week': int(w), 'orders': int(n)} for w,n in d.groupby('week').num_orders.sum().items()],
            'top': [{'meal': str(m), 'orders': int(n)} for m,n in d.groupby('meal_id').num_orders.sum().nlargest(5).items()]}

@app.get('/workspace/validation')
def validation():
    d = pd.read_csv(config.REPORT_DIR/'backtest_predictions.csv')
    d = d[d.fold == d.fold.max()]
    return [{'week': int(w), 'actual': float(g.num_orders.sum()), 'predicted': float(g.lightgbm.sum())} for w,g in d.groupby('week')]

@app.get('/workspace/template.xlsx')
def template():
    from fastapi.responses import Response
    buf = io.BytesIO()
    pd.read_csv(resolve_data_path()).head(20).to_excel(buf,index=False,engine='openpyxl')
    return Response(buf.getvalue(), media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers={'Content-Disposition':'attachment; filename=foodcast-template.xlsx'})

# Serve the production frontend from the same origin when built.
_dist=config.ROOT/'frontend'/'dist'
if _dist.exists():
    app.mount('/assets',StaticFiles(directory=_dist/'assets'),name='frontend-assets')
    @app.get('/')
    def web_home(): return FileResponse(_dist/'index.html')
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon(): return FileResponse(_dist/"favicon.svg", media_type="image/svg+xml")
