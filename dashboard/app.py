"""FoodCast dashboard.   streamlit run dashboard/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from fdf import config  # noqa: E402
from fdf.config import Economics  # noqa: E402
from fdf.models import LABELS  # noqa: E402
from fdf.uploads import read_table, historical_panel, batch_rows
from fdf.service import Forecaster  # noqa: E402

BLUE, ORANGE, GRAY = "#2563EB", "#F97316", "#9CA3AF"

st.set_page_config(page_title="FoodCast | Demand planning", page_icon="🍱", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  div[data-testid="stMetric"] {background:#F9FAFB; border:1px solid #E5E7EB; padding:14px 16px; border-radius:12px;}
  div[data-testid="stMetricValue"] {font-size:1.7rem;}
  h1, h2, h3 {letter-spacing:-0.01em;}
  .small {color:#6B7280; font-size:0.85rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load() -> Forecaster:
    return Forecaster.load()


try:
    F = load()
except FileNotFoundError:
    st.error("Model artifact not found. Run `python -m fdf.train` first.")
    st.stop()

H = F.history
META = F.meta
LAST_WEEK = int(H.week.max())


def forecast(rows: dict | list, econ: Economics, service_level: float | None = None) -> pd.DataFrame:
    rows = [rows] if isinstance(rows, dict) else rows
    return F.predict(pd.DataFrame(rows), service_level=service_level, econ=econ)


def layout(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.1, x=0), font=dict(size=13))
    fig.update_xaxes(gridcolor="#F3F4F6")
    fig.update_yaxes(gridcolor="#F3F4F6")
    return fig


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("🍱 FoodCast Pro 2.0")
st.sidebar.caption("Weekly meal-demand forecasts → stock decisions")
page = st.sidebar.radio("Navigate", ["Planner", "Price & promo what-if", "Model & validation", "Data workspace", "Batch planning", "About"],
                        label_visibility="collapsed")

with st.sidebar.expander("Unit economics (assumptions)", expanded=False):
    margin = st.slider("Profit lost per unmet unit (% of price)", 5, 100, int(config.DEFAULT_ECONOMICS.margin_pct * 100), 5) / 100
    waste = st.slider("Cost of an unsold unit (% of price)", 5, 100, int(config.DEFAULT_ECONOMICS.waste_pct * 100), 5) / 100
    st.caption("The dataset has no cost data, so these are inputs, not facts. "
               "They set the cost-optimal service level.")
ECON = Economics(margin, waste)


def scenario_inputs(prefix: str):
    centers = sorted(H.center_id.unique())
    c1, c2, c3 = st.columns(3)
    center = c1.selectbox("Fulfilment center", centers, key=f"{prefix}c", index=centers.index(55) if 55 in centers else 0)
    meals = sorted(H.meal_id.unique())
    meal = c2.selectbox("Meal", meals, key=f"{prefix}m", index=meals.index(1885) if 1885 in meals else 0)
    week = c3.number_input("Start week", min_value=1, value=LAST_WEEK + 1, step=1, key=f"{prefix}w")
    mh = H[H.meal_id == meal]
    d_base, d_price = float(mh.base_price.median()), float(mh.checkout_price.median())
    c4, c5, c6, c7 = st.columns(4)
    base = c4.number_input("Base price", min_value=1.0, value=round(d_base, 2), key=f"{prefix}b{meal}")
    price = c5.number_input("Checkout price", min_value=1.0, value=round(min(d_price, base), 2), key=f"{prefix}p{meal}")
    email = c6.toggle("Promo e-mail", key=f"{prefix}e")
    home = c7.toggle("Homepage featured", key=f"{prefix}h")
    return dict(center_id=int(center), meal_id=int(meal), week=int(week), base_price=float(base),
                checkout_price=float(price), emailer_for_promotion=int(email), homepage_featured=int(home))


def cold_banner(row):
    if row.unknown_center or row.unknown_meal:
        st.warning("This center or meal was never seen in training: the forecast falls back to the portfolio average and is much less reliable.")
    elif row.cold_start:
        st.info("This center×meal pair has no history. The forecast leans on the meal's and center's average demand, so the interval is wide.")


# --------------------------------------------------------------------------- planner
if page == "Planner":
    st.title("Demand planner")
    st.caption("Pick a center and meal, set the planned price and promotions, get a 10-week forecast and how much to stock.")
    s = scenario_inputs("pl")
    weeks = [{**s, "week": s["week"] + i} for i in range(config.HORIZON)]
    out = forecast(weeks, ECON)
    first = out.iloc[0]
    cold_banner(first)

    default_sl = ECON.critical_ratio
    sl = st.slider("Service level (probability of not running out)", 0.55, 0.99, float(round(default_sl, 2)), 0.01,
                   help=f"Cost-optimal for your economics: {default_sl:.0%}")
    out = forecast(weeks, ECON, sl)
    first = out.iloc[0]
    exp = F.expected_cost(first, first.recommended_stock, ECON)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Forecast, week {int(first.week)}", f"{first.forecast:,.0f}")
    m2.metric("80% range", f"{first.lower_80:,.0f} – {first.upper_80:,.0f}")
    m3.metric("Stock to prepare", f"{int(first.recommended_stock):,}", f"{first.recommended_stock - first.forecast:+,.0f} safety stock")
    m4.metric("Expected cost @ this stock", f"{(exp['total_cost'] / 99):,.0f}",
              help="Average waste + lost margin over the forecast distribution, in price units.")

    # context: observed demand for this meal at all centers + this pair
    mh = H[H.meal_id == s["meal_id"]]
    ph = mh[mh.center_id == s["center_id"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mh.week, y=mh.num_orders, mode="markers", name="Same meal, other centers",
                             marker=dict(color="#D1D5DB", size=6)))
    fig.add_trace(go.Scatter(x=ph.week, y=ph.num_orders, mode="markers", name="This center",
                             marker=dict(color=BLUE, size=10)))
    fig.add_trace(go.Scatter(x=out.week, y=out.upper_80, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=out.week, y=out.lower_80, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(249,115,22,0.18)", name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=out.week, y=out.forecast, mode="lines+markers", name="Forecast",
                             line=dict(color=ORANGE, width=3)))
    fig.add_trace(go.Scatter(x=out.week, y=out.recommended_stock, mode="lines", name="Recommended stock",
                             line=dict(color="#111827", dash="dash", width=2)))
    fig.update_layout(yaxis_title="Orders per week", xaxis_title="Week")
    st.plotly_chart(layout(fig, 430), width="stretch")

    with st.expander("10-week plan table"):
        t = out[["week", "forecast", "lower_80", "upper_80", "recommended_stock"]].round(0).astype(int)
        st.dataframe(t, width="stretch", hide_index=True)
        st.download_button("Download CSV", t.to_csv(index=False), "foodcast_plan.csv", "text/csv")
    st.caption("Held price and promotions are assumed constant over the 10 weeks. "
               "Intervals are calibrated on out-of-sample backtest errors.")

# --------------------------------------------------------------------------- what-if
elif page == "Price & promo what-if":
    st.title("Price & promotion what-if")
    st.caption("How does the forecast move with discount depth and promotions for one center and meal?")
    s = scenario_inputs("wi")
    rows, labels = [], []
    for e, h, lab in [(0, 0, "No promo"), (1, 0, "E-mail"), (0, 1, "Homepage"), (1, 1, "E-mail + homepage")]:
        for d in np.arange(0, 0.41, 0.05):
            rows.append({**s, "checkout_price": s["base_price"] * (1 - d), "emailer_for_promotion": e, "homepage_featured": h})
            labels.append((lab, d))
    res = forecast(rows, ECON)
    res["scenario"], res["discount"] = [name for name, _ in labels], [d for _, d in labels]
    res["revenue"] = res.forecast * res.checkout_price

    a, b = st.columns(2)
    for col, ycol, title in [(a, "forecast", "Forecast orders"), (b, "revenue", "Forecast revenue")]:
        fig = go.Figure()
        for lab, g in res.groupby("scenario", sort=False):
            fig.add_trace(go.Scatter(x=g.discount * 100, y=g[ycol], mode="lines+markers", name=lab))
        fig.update_layout(title=title, xaxis_title="Discount off base price (%)")
        col.plotly_chart(layout(fig, 360), width="stretch")
    st.warning("These curves show what the model has **associated** with price and promotions in historical data. "
               "They are not causal estimates: promotions are usually targeted at meals that were already doing well. "
               "Validate with an A/B test before committing budget.")

# --------------------------------------------------------------------------- model page
elif page == "Model & validation":
    st.title("Model & validation")
    p = META["panel"]
    st.markdown(f"Trained on **{p['rows']:,} rows**, weeks {p['first_week']}–{p['last_week']}, "
                f"{p['centers']} centers × {p['meals']} meals ({p['center_meal_pairs']:,} observed pairs). "
                f"Backtest: **{len(META['cutoffs'])} walk-forward folds**, each forecasting the next {META['horizon_weeks']} weeks "
                f"using only data available at the cutoff.")
    m = pd.DataFrame(META["metrics"]).set_index("model")
    champ, base = m.loc["lightgbm"], m.loc["pair_average"]
    pol = pd.DataFrame(META["policy"]).set_index("model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WMAPE (champion)", f"{champ.WMAPE:.3f}", f"{(champ.WMAPE / base.WMAPE - 1) * 100:.1f}% vs center×meal average", delta_color="inverse")
    c2.metric("RMSLE", f"{champ.RMSLE:.3f}", f"{(champ.RMSLE / base.RMSLE - 1) * 100:.1f}%", delta_color="inverse")
    c3.metric("Simulated cost", f"{(pol.loc['lightgbm', 'total_cost'] / pol.loc['pair_average', 'total_cost'] - 1) * 100:.1f}%",
              "vs baseline ordering policy", delta_color="off")
    c4.metric("80% interval coverage", f"{META['coverage']['empirical']:.0%}", f"n={META['coverage']['n']}", delta_color="off")

    show = m[["WMAPE", "RMSLE", "MAE", "Bias", "folds_won_vs_pair_average"]].copy()
    show.index = [LABELS.get(i, i) for i in show.index]
    show.columns = ["WMAPE ↓", "RMSLE ↓", "MAE ↓", "Bias", "Folds beating center×meal avg"]
    st.dataframe(show.style.format({"WMAPE ↓": "{:.3f}", "RMSLE ↓": "{:.3f}", "MAE ↓": "{:.1f}", "Bias": "{:+.1%}"}), width="stretch")

    g1, g2 = st.columns(2)
    for col, name in [(g1, "model_comparison.png"), (g2, "actual_vs_forecast.png"), (g1, "feature_importance.png"), (g2, "inventory_policy_cost.png")]:
        fp = config.FIG_DIR / name
        if fp.exists():
            col.image(str(fp), width="stretch")

    st.subheader("Limitations, stated plainly")
    st.markdown(f"""
- **The bundled data is a sparse sample**: only {p['pair_density']:.1%} of possible center×meal×week cells are observed, so classic lag features (last week's demand) are rarely available. The model compensates with *last observed demand and its age* plus smoothed meal/center/pair encodings. On the full 456k-row Genpact dataset the same pipeline gets dense lags automatically.
- **Hyper-parameters were chosen on the same backtest** (small sweep, ~2–3% gain), so the reported error is mildly optimistic.
- **Inventory savings are a simulation** under assumed unit economics, not measured business impact.
- **Under-forecasting bias** ({champ.Bias:+.0%}) comes from training in log space; the safety-stock step corrects for it when ordering.
""")

elif page == "Data workspace":
    st.title("Data workspace")
    st.caption("Explore uploaded history. Uploads do not change the trained model.")
    st.download_button("Download history template", H.head(20).to_csv(index=False), "history_template.csv")
    uploaded = st.file_uploader("Historical orders", type=["csv", "xlsx"])
    try:
        panel = historical_panel(read_table(uploaded)) if uploaded else H
        a,b,c,d = st.columns(4)
        a.metric("Orders", f"{panel.num_orders.sum():,.0f}")
        b.metric("Centers", panel.center_id.nunique())
        c.metric("Meals", panel.meal_id.nunique())
        d.metric("Weeks", panel.week.nunique())
        st.line_chart(panel.groupby("week").num_orders.sum())
        st.bar_chart(panel.groupby("meal_id").num_orders.sum().nlargest(10))
        st.dataframe(panel, width="stretch")
        st.download_button("Export validated history", panel.to_csv(index=False), "validated_history.csv")
        st.info("To retrain on this history, save the validated CSV and run python -m fdf.train --data validated_history.csv, then restart both services. Sufficient historical weeks are required.")
    except (ValueError, TypeError, OverflowError) as exc:
        st.error(str(exc))

elif page == "Batch planning":
    st.title("Batch demand and inventory planning")
    template = H.sort_values("week").drop_duplicates(["center_id", "meal_id"], keep="last").head(20).copy()
    template["week"] = LAST_WEEK + 1
    template = template.drop(columns=["id", "num_orders"])
    st.download_button("Download forecast template", template.to_csv(index=False), "forecast_template.csv")
    uploaded = st.file_uploader("Future demand scenarios", type=["csv", "xlsx"])
    if uploaded:
        try:
            rows = batch_rows(read_table(uploaded), LAST_WEEK)
            result = F.predict(rows, econ=ECON)
            a,b,c = st.columns(3)
            a.metric("Forecast units", f"{result.forecast.sum():,.0f}")
            b.metric("Recommended stock", f"{result.recommended_stock.sum():,.0f}")
            c.metric("Cold-start rows", int(result.cold_start.sum()))
            st.dataframe(result, width="stretch")
            st.download_button("Export inventory plan", result.to_csv(index=False), "inventory_plan.csv")
            st.caption("Forecasts use the existing trained model; costs and uncertainty are estimates.")
        except (ValueError, TypeError, OverflowError) as exc:
            st.error(str(exc))

# --------------------------------------------------------------------------- about
else:
    st.title("About this project")
    st.markdown("""
**FoodCast** turns weekly order history from a meal-delivery network into **decisions**: how many units of each meal each
fulfilment center should prepare, with calibrated uncertainty and an explicit cost trade-off between waste and stock-outs.

It started as a notebook-style NARX demo and was rebuilt as an end-to-end system:

| Layer | What is in it |
|---|---|
| Data | Schema validation, cleaning, optional center/meal metadata merge |
| Features | Leak-free *as-of* features (`merge_asof`, strictly earlier weeks), smoothed target encodings, last-observation + age |
| Modeling | 4 baselines vs Random Forest vs LightGBM, **walk-forward backtest** over 6 folds |
| Uncertainty | Conformal-style 80% intervals from out-of-sample residuals, coverage verified |
| Decisions | Newsvendor order quantity, simulated against baseline ordering policies |
| Serving | FastAPI (`/forecast`, `/forecast/batch`, `/recommend-order`), Docker, this Streamlit app |
| Quality | pytest suite incl. a no-future-leakage test, GitHub Actions CI |
""")
    st.subheader("Author")
    st.markdown("""
**Maneesh Amuda** · B.Tech Electronics & Communication Engineering (2026) · Hyderabad, India
Open to **Data Analyst · Python Developer · Associate Software Engineer · Junior ML Engineer** roles.

[GitHub](https://github.com/maneeshamuda) · [LinkedIn](https://linkedin.com/in/amudamaneesh) · maneeshamudha44@gmail.com
""")
    st.caption(f"Model v{META['version']} · trained {META['trained_at']} on `{META['data_file']}`")
