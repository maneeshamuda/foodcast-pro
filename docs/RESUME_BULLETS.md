# Resume / LinkedIn wording (all numbers are reproducible via `python -m fdf.train`)

**FoodCast: Demand Forecasting & Inventory Planning** · Python, LightGBM, FastAPI, Streamlit, Docker, GitHub Actions
- Built an end-to-end demand-forecasting system (leak-free as-of features, walk-forward backtest over 6 folds) that cut WMAPE 6.6% vs a center×meal-average baseline and won in all 6 folds.
- Produced calibrated 80% prediction intervals (83% empirical coverage) and a newsvendor stock recommendation; simulation shows ~9.5% lower waste + lost-margin cost vs the baseline ordering policy under stated assumptions.
- Shipped a FastAPI service (`/forecast`, `/recommend-order`) and a Streamlit planner dashboard, containerised with Docker, with a 21-test pytest suite (incl. a future-leakage test) and CI.

Interview talking point: "I found the original lag features were invalid on a sparse panel (0.4% consecutive weeks), so I replaced them with last-observation + age features and validated with walk-forward CV."
