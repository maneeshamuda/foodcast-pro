# Model card: FoodCast demand forecaster

**Task:** forecast weekly `num_orders` per (fulfilment center, meal), 1–10 weeks ahead.
**Champion:** LightGBM regressor on `log1p(orders)` with exogenous inputs (price, discount, promotions, seasonality) and autoregressive inputs (last observed demand of the pair and its age, smoothed target encodings). This is a NARX-style design adapted to an irregularly observed panel.

## Data
Bundled: 1,999 rows, weeks 1–145, 77 centers, 51 meals, 1,534 observed pairs (median 1 row per pair). Fill rate of the center×meal×week grid is 0.9%. Source: Genpact / Analytics Vidhya hackathon data (sample).

## Evaluation protocol
Rolling-origin backtest: 6 cutoffs (95, 103, …, 135), forecast the following 10 weeks, train only on weeks ≤ cutoff, features built from the training window only. Pooled over ~870 out-of-sample rows.

| Metric | Champion | Center×meal average |
|---|---|---|
| WMAPE | 0.465 | 0.498 |
| RMSLE | 0.702 | 0.768 |
| MAE (orders) | 112.5 | 120.5 |
| Bias | −15% | −22% |

## Intended use
Planning aid for weekly prep quantities, with human review. Use the interval and the `cold_start` flag.

## Known limitations
1. **Sample size.** Small, sparse data; fold-to-fold WMAPE std is ~0.03, comparable to the margin over the Random Forest. The ranking champion > baseline is consistent (6/6 folds); champion > Random Forest is not statistically established.
2. **Mild selection optimism.** LightGBM parameters came from a small sweep (7 variants) on the same folds (~2–3% gain).
3. **Under-forecast bias.** Log-space training predicts near the geometric mean. The newsvendor step uses calibrated quantiles, which correct this for ordering.
4. **No causal claims.** Price/promo response curves are associations. Promotions are usually targeted at already-strong meals.
5. **Simulated economics.** Cost savings use assumed margin/waste rates (40% / 25% of price) and are a simulation, not measured impact.
6. **Cold start.** Pairs/centers/meals unseen in training fall back to portfolio averages; the API flags them.
7. **No holiday/weather/city features** in the bundled sample. The full dataset's center/meal metadata is supported when present.

## Changes vs. the original notebook-style project
* Original lag features assumed consecutive weeks, but only 0.4% of rows have a previous-week row, so the "NARX lags" mostly referenced unrelated rows. Replaced with as-of last-observation + age.
* Original metrics (R² 0.23, MAPE 204%) came from a different, random-ish split with 400 test rows; not directly comparable to the walk-forward results above.
* Added baselines, walk-forward CV, intervals, decision layer, API, tests, CI.
