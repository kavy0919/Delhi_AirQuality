"""
app.py
======
Delhi-NCR AQI Forecasting & Intelligence Dashboard (Streamlit).

DATA NOTE -- please read before a demo/judging session:
--------------------------------------------------------
The underlying model is trained on a REAL daily Delhi-wide AQI dataset
(2021-2024): PM2.5, PM10, NO2, SO2, CO, Ozone + AQI. It does NOT contain
true per-station sensor data or weather (wind/temp/RH) columns.

To keep the "multi-station" and "what-if weather" UX from the original
brief without pretending we have data we don't, this dashboard:

  1. Shows a single real Delhi-wide forecast (the actual model output).
  2. Derives 4 station "views" (Anand Vihar, RK Puram, Punjabi Bagh, ITO)
     by applying FIXED, clearly-labelled illustrative offsets to
     the real Delhi-wide series (documented inline as ILLUSTRATIVE, not
     measured). This is disclosed directly in the sidebar UI.
  3. Replaces "wind speed / temperature" what-if sliders (not available
     in this dataset) with "regional emission" what-if sliders on the
     pollutants we DO have real data for: NO2, CO, SO2, Ozone.

Everything else (GRAP staging, advisories, forecasting, charts) runs on
the real trained model and real historical values.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import pydeck as pdk

from train_model import (
    load_raw_data, build_features, get_feature_columns,
    compute_aqi_from_pm25, train_all_horizons,
    MODEL_PATH, FEATURE_COLUMNS_PATH, METRICS_PATH, FEATURED_DATA_PATH,
    BACKTEST_PATH, IMPORTANCE_PATH,
    HORIZONS, PRIMARY_HORIZON,
)
from advisory_rules import classify_grap, get_all_stages, stage_to_dict

# --------------------------------------------------------------------------
# PAGE CONFIG & CUSTOM CSS
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Delhi-NCR AQI Intelligence Dashboard",
    page_icon="<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='14' font-size='14'>A</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1d29 0%, #1f2333 100%);
        border: 1px solid #2d3348;
        border-radius: 14px;
        padding: 18px 20px;
        text-align: left;
        min-height: 110px;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #9aa3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #f2f3f7;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: #7f8aa3;
        margin-top: 4px;
    }
    .badge {
        display: inline-block;
        padding: 10px 18px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.0rem;
        color: white;
        text-align: center;
    }
    .disclosure-box {
        background: #23262f;
        border-left: 4px solid #E67E22;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #c9cee0;
        margin-top: 8px;
    }
    h1, h2, h3 { color: #f2f3f7; }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f2f3f7;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# STATIONS — widened offsets so 4 stations land in DIFFERENT GRAP bands
# --------------------------------------------------------------------------
STATIONS = {
    "Anand Vihar":   {"lat": 28.6469, "lon": 77.3152, "offset": 1.55},
    "RK Puram":      {"lat": 28.5641, "lon": 77.1750, "offset": 0.70},
    "Punjabi Bagh":  {"lat": 28.6692, "lon": 77.1310, "offset": 1.15},
    "ITO":           {"lat": 28.6280, "lon": 77.2410, "offset": 0.92},
}

SEVERITY_COLORS = {
    0: [46, 204, 113],    # green
    1: [241, 196, 15],    # yellow
    2: [230, 126, 34],    # orange
    3: [231, 76, 60],     # red
    4: [123, 36, 28],     # dark maroon
}
SEVERITY_HEX = {
    0: "#2ECC71", 1: "#F1C40F", 2: "#E67E22", 3: "#E74C3C", 4: "#7B241C",
}


# --------------------------------------------------------------------------
# CACHED DATA / MODEL LOADING
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading (or training) forecasting models...")
def get_artifacts():
    """Loads trained models + features; trains automatically on first run."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_COLUMNS_PATH)
             and os.path.exists(FEATURED_DATA_PATH)):
        train_all_horizons()

    models = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLUMNS_PATH)
    metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else {}
    featured_df = pd.read_csv(FEATURED_DATA_PATH, parse_dates=["date"])

    backtest_df = None
    if os.path.exists(BACKTEST_PATH):
        backtest_df = pd.read_csv(BACKTEST_PATH, parse_dates=["date"])

    importances = {}
    if os.path.exists(IMPORTANCE_PATH):
        importances = joblib.load(IMPORTANCE_PATH)

    return models, feature_cols, metrics, featured_df, backtest_df, importances


models, feature_cols, metrics, featured_df, backtest_df, importances = get_artifacts()
latest_row = featured_df.iloc[-1]
latest_date = latest_row["date"]


def predict_pm25(horizon: int, feature_row: pd.Series) -> float:
    """Predicts PM2.5 `horizon` days ahead using the model closest to that horizon."""
    available = sorted(models.keys())
    closest = min(available, key=lambda h: abs(h - horizon))
    X = pd.DataFrame([feature_row[feature_cols].values], columns=feature_cols)
    pred = models[closest].predict(X)[0]
    return max(0.0, float(pred))


def get_residual_std(horizon: int) -> float:
    """Get residual std for uncertainty band from the closest horizon model."""
    available = sorted(metrics.keys())
    closest = min(available, key=lambda h: abs(h - horizon))
    return metrics.get(closest, {}).get("residual_std", 40.0)


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.markdown("### Control Panel")

st.sidebar.markdown("**Station**")
station_name = st.sidebar.selectbox("Select station", list(STATIONS.keys()), label_visibility="collapsed")
st.sidebar.markdown(
    '<div class="disclosure-box">The source dataset is a single Delhi-wide '
    'daily series (no true per-station sensors). Station values below are '
    'the Delhi-wide forecast adjusted by a fixed illustrative offset for '
    'demo purposes.</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("**Forecast Horizon**")
horizon_days = st.sidebar.slider("Days ahead", min_value=1, max_value=7, value=3, step=1)
st.sidebar.caption("Dataset is daily-resolution; horizon is expressed in days, not hours.")

station_offset = STATIONS[station_name]["offset"]

# --------------------------------------------------------------------------
# COMPUTE BASELINE FORECAST (no what-if)
# --------------------------------------------------------------------------
current_pm25 = float(latest_row["PM2.5"]) * station_offset
predicted_pm25_baseline = predict_pm25(horizon_days, latest_row) * station_offset
predicted_aqi_baseline = compute_aqi_from_pm25(predicted_pm25_baseline)
grap_stage_baseline = classify_grap(pm25=predicted_pm25_baseline, aqi=predicted_aqi_baseline)

# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.markdown("## Delhi-NCR AQI Forecasting & Intelligence Dashboard")
st.caption(
    f"Model: HistGradientBoostingRegressor | Trained on real daily Delhi AQI data "
    f"(latest record: {latest_date.strftime('%d %b %Y')}) | Station view: **{station_name}**"
)

# --------------------------------------------------------------------------
# METRIC ROW (always visible above tabs)
# --------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current PM2.5</div>
        <div class="metric-value">{current_pm25:.1f}</div>
        <div class="metric-sub">ug/m3 | as of {latest_date.strftime('%d %b %Y')}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    delta_base = predicted_pm25_baseline - current_pm25
    arrow = "+" if delta_base >= 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Baseline Forecast (+{horizon_days}d)</div>
        <div class="metric-value">{predicted_pm25_baseline:.1f}</div>
        <div class="metric-sub">{arrow}{delta_base:.1f} vs current (no what-if)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Forecasted AQI</div>
        <div class="metric-value">{predicted_aqi_baseline:.0f}</div>
        <div class="metric-sub">{grap_stage_baseline.aqi_category}</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Active GRAP Stage</div>
        <div class="badge" style="background-color:{grap_stage_baseline.color};">
            {grap_stage_baseline.stage_label}
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------------------------------
# TABS
# --------------------------------------------------------------------------
tab_overview, tab_forecast, tab_whatif, tab_spatial, tab_advisory, tab_model = st.tabs([
    "Overview", "Forecast", "What-If Simulation",
    "Spatial View", "Advisory Board", "Model Transparency",
])

# ==========================================================================
# TAB 1: OVERVIEW
# ==========================================================================
with tab_overview:
    st.markdown('<div class="section-header">Historical Trend (last 60 days)</div>',
                unsafe_allow_html=True)

    hist_window = featured_df.tail(60).copy()
    hist_window["station_pm25"] = hist_window["PM2.5"] * station_offset

    fig_ov = go.Figure()
    fig_ov.add_trace(go.Scatter(
        x=hist_window["date"], y=hist_window["station_pm25"],
        mode="lines", name="Historical PM2.5",
        line=dict(color="#5DADE2", width=2),
        fill="tozeroy", fillcolor="rgba(93,173,226,0.08)",
    ))
    fig_ov.add_hline(y=60, line_dash="dot", line_color="#F1C40F", opacity=0.4,
                     annotation_text="GRAP I (60)")
    fig_ov.add_hline(y=120, line_dash="dot", line_color="#E67E22", opacity=0.4,
                     annotation_text="GRAP II (120)")
    fig_ov.add_hline(y=250, line_dash="dot", line_color="#E74C3C", opacity=0.4,
                     annotation_text="GRAP III (250)")
    fig_ov.update_layout(
        template="plotly_dark", height=380,
        xaxis_title="Date", yaxis_title="PM2.5 (ug/m3)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_ov, width="stretch")

    st.markdown(
        '<div class="disclosure-box">'
        '<b>Data disclosure:</b> This dashboard is trained on a real, daily Delhi-wide AQI '
        'dataset (2021-2024). It does NOT contain hourly timestamps, per-station sensor '
        'readings, or meteorological variables (temperature / humidity / wind speed). '
        'The 4-station view uses fixed illustrative offsets applied to the single Delhi-wide '
        'series for demo purposes only. All limitations are documented in-place throughout '
        'the dashboard.'
        '</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# TAB 2: FORECAST (baseline + uncertainty bands)
# ==========================================================================
with tab_forecast:
    st.markdown('<div class="section-header">Baseline Forecast with Uncertainty Bands</div>',
                unsafe_allow_html=True)
    st.caption("Forecast uses the model's baseline prediction (no what-if adjustments). "
               "The shaded band shows +/- 1 residual standard deviation from the backtest holdout.")

    hist_window_fc = featured_df.tail(48).copy()
    hist_window_fc["station_pm25"] = hist_window_fc["PM2.5"] * station_offset

    forecast_dates_base, forecast_vals_base = [], []
    forecast_upper, forecast_lower = [], []
    for h in range(1, horizon_days + 1):
        fd = latest_date + pd.Timedelta(days=h)
        val = predict_pm25(h, latest_row) * station_offset
        rstd = get_residual_std(h) * station_offset
        forecast_dates_base.append(fd)
        forecast_vals_base.append(val)
        forecast_upper.append(max(0, val + rstd))
        forecast_lower.append(max(0, val - rstd))

    fig_fc = go.Figure()

    # Historical line
    fig_fc.add_trace(go.Scatter(
        x=hist_window_fc["date"], y=hist_window_fc["station_pm25"],
        mode="lines", name="Historical PM2.5",
        line=dict(color="#5DADE2", width=2),
    ))

    # Uncertainty band (upper)
    all_fc_dates = [latest_date] + forecast_dates_base
    all_fc_upper = [hist_window_fc["station_pm25"].iloc[-1]] + forecast_upper
    all_fc_lower = [hist_window_fc["station_pm25"].iloc[-1]] + forecast_lower

    fig_fc.add_trace(go.Scatter(
        x=all_fc_dates, y=all_fc_upper,
        mode="lines", name="Upper bound (+1 SD)",
        line=dict(width=0), showlegend=False,
    ))
    fig_fc.add_trace(go.Scatter(
        x=all_fc_dates, y=all_fc_lower,
        mode="lines", name="Uncertainty band (+/- 1 SD)",
        line=dict(width=0),
        fill="tonexty", fillcolor="rgba(231,76,60,0.15)",
    ))

    # Baseline forecast line
    all_fc_vals = [hist_window_fc["station_pm25"].iloc[-1]] + forecast_vals_base
    fig_fc.add_trace(go.Scatter(
        x=all_fc_dates, y=all_fc_vals,
        mode="lines+markers", name=f"Baseline forecast (+1..{horizon_days}d)",
        line=dict(color="#E74C3C", width=2, dash="dash"),
        marker=dict(size=7),
    ))

    # GRAP thresholds
    fig_fc.add_hline(y=60, line_dash="dot", line_color="#F1C40F", opacity=0.4,
                     annotation_text="Stage I (60)")
    fig_fc.add_hline(y=120, line_dash="dot", line_color="#E67E22", opacity=0.4,
                     annotation_text="Stage II (120)")
    fig_fc.add_hline(y=250, line_dash="dot", line_color="#E74C3C", opacity=0.4,
                     annotation_text="Stage III (250)")

    fig_fc.update_layout(
        template="plotly_dark", height=440,
        xaxis_title="Date", yaxis_title="PM2.5 (ug/m3)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_fc, width="stretch")

    # Day-by-day forecast table
    fc_table = pd.DataFrame({
        "Day": [f"+{h}d" for h in range(1, horizon_days + 1)],
        "Date": [d.strftime("%d %b %Y") for d in forecast_dates_base],
        "PM2.5 (ug/m3)": [f"{v:.1f}" for v in forecast_vals_base],
        "Lower bound": [f"{v:.1f}" for v in forecast_lower],
        "Upper bound": [f"{v:.1f}" for v in forecast_upper],
        "GRAP Stage": [classify_grap(pm25=v).stage_label for v in forecast_vals_base],
    })
    st.dataframe(fc_table, width="stretch", hide_index=True)


# ==========================================================================
# TAB 3: WHAT-IF SIMULATION
# ==========================================================================
with tab_whatif:
    st.markdown('<div class="section-header">What-If: Regional Emission Scenarios</div>',
                unsafe_allow_html=True)
    st.caption("Perturb the latest known pollutant levels before forecasting. "
               "No live weather feed is used -- wind/temperature are not in this dataset.")

    wi_cols = st.columns(4)
    with wi_cols[0]:
        no2_delta = st.slider("Delta NO2 (traffic, ug/m3)", -50, 100, 0, step=5)
    with wi_cols[1]:
        co_delta = st.slider("Delta CO (combustion, mg/m3)", -2.0, 3.0, 0.0, step=0.1)
    with wi_cols[2]:
        so2_delta = st.slider("Delta SO2 (industrial, ug/m3)", -20, 40, 0, step=2)
    with wi_cols[3]:
        ozone_delta = st.slider("Delta Ozone (photochem, ug/m3)", -30, 60, 0, step=5)

    sim_row = latest_row.copy()
    for lag in (1, 3, 7):
        for col, delta in [("NO2", no2_delta), ("CO", co_delta),
                           ("SO2", so2_delta), ("Ozone", ozone_delta)]:
            key = f"{col}_lag{lag}"
            if key in sim_row:
                sim_row[key] = max(0, sim_row[key] + delta)

    predicted_pm25_whatif = predict_pm25(horizon_days, sim_row) * station_offset
    predicted_aqi_whatif = compute_aqi_from_pm25(predicted_pm25_whatif)
    grap_stage_whatif = classify_grap(pm25=predicted_pm25_whatif, aqi=predicted_aqi_whatif)

    st.markdown("---")
    st.markdown('<div class="section-header">Baseline vs What-If Comparison</div>',
                unsafe_allow_html=True)

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Baseline Forecast (+{horizon_days}d)</div>
            <div class="metric-value">{predicted_pm25_baseline:.1f}</div>
            <div class="metric-sub">ug/m3 | No adjustments</div>
        </div>""", unsafe_allow_html=True)
    with wc2:
        diff = predicted_pm25_whatif - predicted_pm25_baseline
        arrow_wi = "+" if diff >= 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">What-If Forecast (+{horizon_days}d)</div>
            <div class="metric-value">{predicted_pm25_whatif:.1f}</div>
            <div class="metric-sub">{arrow_wi}{diff:.1f} vs baseline</div>
        </div>""", unsafe_allow_html=True)
    with wc3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">What-If GRAP Stage</div>
            <div class="badge" style="background-color:{grap_stage_whatif.color};">
                {grap_stage_whatif.stage_label}
            </div>
        </div>""", unsafe_allow_html=True)

    # What-if vs baseline chart
    st.markdown("---")
    hist_wi = featured_df.tail(30).copy()
    hist_wi["station_pm25"] = hist_wi["PM2.5"] * station_offset

    fc_dates_wi, fc_base_wi, fc_whatif_wi = [], [], []
    for h in range(1, horizon_days + 1):
        fd = latest_date + pd.Timedelta(days=h)
        fc_dates_wi.append(fd)
        fc_base_wi.append(predict_pm25(h, latest_row) * station_offset)
        fc_whatif_wi.append(predict_pm25(h, sim_row) * station_offset)

    fig_wi = go.Figure()
    fig_wi.add_trace(go.Scatter(
        x=hist_wi["date"], y=hist_wi["station_pm25"],
        mode="lines", name="Historical",
        line=dict(color="#5DADE2", width=2),
    ))
    fig_wi.add_trace(go.Scatter(
        x=[latest_date] + fc_dates_wi,
        y=[hist_wi["station_pm25"].iloc[-1]] + fc_base_wi,
        mode="lines+markers", name="Baseline forecast",
        line=dict(color="#2ECC71", width=2, dash="dash"), marker=dict(size=6),
    ))
    fig_wi.add_trace(go.Scatter(
        x=[latest_date] + fc_dates_wi,
        y=[hist_wi["station_pm25"].iloc[-1]] + fc_whatif_wi,
        mode="lines+markers", name="What-If forecast",
        line=dict(color="#E74C3C", width=2, dash="dot"), marker=dict(size=6),
    ))
    fig_wi.update_layout(
        template="plotly_dark", height=380,
        xaxis_title="Date", yaxis_title="PM2.5 (ug/m3)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_wi, width="stretch")


# ==========================================================================
# TAB 4: SPATIAL VIEW (token-free CARTO basemap)
# ==========================================================================
with tab_spatial:
    st.markdown('<div class="section-header">Spatial View -- Station Severity (illustrative)</div>',
                unsafe_allow_html=True)
    st.caption("Station offsets are illustrative (see disclosure). "
               "Offsets are widened so stations visibly land in different GRAP color bands.")

    map_rows = []
    for name, meta in STATIONS.items():
        pm25_val = predict_pm25(horizon_days, latest_row) * meta["offset"]
        aqi_val = compute_aqi_from_pm25(pm25_val)
        stage = classify_grap(pm25=pm25_val, aqi=aqi_val)
        color = SEVERITY_COLORS[stage.stage_id]
        map_rows.append({
            "station": name, "lat": meta["lat"], "lon": meta["lon"],
            "pm25": round(pm25_val, 1), "aqi": round(aqi_val),
            "stage": stage.stage_label,
            "color": color, "radius": 900 + stage.stage_id * 350,
        })
    map_df = pd.DataFrame(map_rows)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.80,
    )
    view_state = pdk.ViewState(latitude=28.62, longitude=77.22, zoom=10.2, pitch=15)
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="dark",
        tooltip={"text": "{station}\nPM2.5: {pm25} ug/m3 | AQI: {aqi}\n{stage}"},
    ))

    # Legend
    legend_cols = st.columns(5)
    legend_labels = ["Below GRAP", "Stage I", "Stage II", "Stage III", "Stage IV"]
    for col, sid, label in zip(legend_cols, range(5), legend_labels):
        r, g, b = SEVERITY_COLORS[sid]
        col.markdown(
            f'<div style="text-align:center;">'
            f'<div style="width:100%;height:10px;border-radius:6px;'
            f'background-color:rgb({r},{g},{b});"></div>'
            f'<div style="font-size:0.75rem;color:#9aa3b8;margin-top:4px;">{label}</div></div>',
            unsafe_allow_html=True,
        )

    # Station detail table
    st.markdown("---")
    st.markdown("**Per-station predicted values (illustrative offsets applied)**")
    st.dataframe(
        map_df[["station", "pm25", "aqi", "stage"]].rename(columns={
            "station": "Station", "pm25": "PM2.5 (ug/m3)",
            "aqi": "AQI", "stage": "GRAP Stage",
        }),
        width="stretch", hide_index=True,
    )


# ==========================================================================
# TAB 5: ADVISORY BOARD
# ==========================================================================
with tab_advisory:
    st.markdown(f'<div class="section-header">Advisory Board -- {grap_stage_baseline.stage_label}</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<span class="badge" style="background-color:{grap_stage_baseline.color};">'
        f'{grap_stage_baseline.aqi_category} | PM2.5 {grap_stage_baseline.pm25_range} | AQI {grap_stage_baseline.aqi_range}'
        f'</span>', unsafe_allow_html=True,
    )
    st.write("")

    with st.expander("Municipal / Policy Actions (GRAP)", expanded=True):
        for action in grap_stage_baseline.policy_actions:
            st.markdown(f"- {action}")

    persona_cols = st.columns(len(grap_stage_baseline.persona_advisories))
    for col, (persona, tips) in zip(persona_cols, grap_stage_baseline.persona_advisories.items()):
        with col:
            with st.expander(f"{persona}", expanded=True):
                for tip in tips:
                    st.markdown(f"- {tip}")

    st.markdown("---")
    st.markdown(
        '<div class="disclosure-box">'
        'These advisories are generated deterministically from CAQM GRAP thresholds '
        'applied to the model forecast. They are NOT official government orders. '
        'Always refer to the latest official CAQM/DPCC notifications for binding policy actions.'
        '</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# TAB 6: MODEL TRANSPARENCY
# ==========================================================================
with tab_model:
    st.markdown('<div class="section-header">Model Performance Metrics</div>',
                unsafe_allow_html=True)

    if metrics:
        perf_df = pd.DataFrame([
            {"Horizon (days)": h, "RMSE": f"{m['rmse']:.2f}", "MAE": f"{m['mae']:.2f}",
             "R2": f"{m['r2']:.3f}", "Residual SD": f"{m.get('residual_std', 0):.2f}",
             "Train rows": m["n_train"], "Test rows": m["n_test"]}
            for h, m in sorted(metrics.items())
        ])
        st.dataframe(perf_df, width="stretch", hide_index=True)

    # --- BACKTEST CHART ---
    st.markdown("---")
    st.markdown('<div class="section-header">Backtest: Predicted vs Actual PM2.5 (holdout set)</div>',
                unsafe_allow_html=True)
    st.caption(f"Showing the {PRIMARY_HORIZON}-day-ahead model's predictions against actual values "
               "on the chronological test holdout (last 20% of the dataset). This contextualizes "
               "the R2 score visually.")

    if backtest_df is not None and len(backtest_df) > 0:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=backtest_df["date"], y=backtest_df["actual"],
            mode="lines", name="Actual PM2.5",
            line=dict(color="#5DADE2", width=1.5),
        ))
        fig_bt.add_trace(go.Scatter(
            x=backtest_df["date"], y=backtest_df["predicted"],
            mode="lines", name="Predicted PM2.5",
            line=dict(color="#E74C3C", width=1.5, dash="dash"),
        ))
        # Residual band
        bt_resid_std = backtest_df["residual"].std()
        fig_bt.add_trace(go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["predicted"] + bt_resid_std,
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig_bt.add_trace(go.Scatter(
            x=backtest_df["date"],
            y=(backtest_df["predicted"] - bt_resid_std).clip(lower=0),
            mode="lines", name="+/- 1 SD band",
            line=dict(width=0),
            fill="tonexty", fillcolor="rgba(231,76,60,0.10)",
        ))
        fig_bt.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Date", yaxis_title="PM2.5 (ug/m3)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_bt, width="stretch")

        # Scatter: actual vs predicted
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=backtest_df["actual"], y=backtest_df["predicted"],
            mode="markers", name="Test samples",
            marker=dict(color="#5DADE2", size=4, opacity=0.6),
        ))
        max_val = max(backtest_df["actual"].max(), backtest_df["predicted"].max()) * 1.05
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="Perfect prediction",
            line=dict(color="#F1C40F", width=1, dash="dot"),
        ))
        fig_scatter.update_layout(
            template="plotly_dark", height=350,
            xaxis_title="Actual PM2.5", yaxis_title="Predicted PM2.5",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_scatter, width="stretch")
    else:
        st.info("No backtest data available. Re-run train_model.py to generate it.")

    # --- FEATURE IMPORTANCE ---
    st.markdown("---")
    st.markdown('<div class="section-header">Feature Importance (permutation-based)</div>',
                unsafe_allow_html=True)
    st.caption("Shows which features the model relies on most for the primary forecast horizon. "
               "Higher = more impact on PM2.5 prediction.")

    imp_horizon = PRIMARY_HORIZON if PRIMARY_HORIZON in importances else (
        min(importances.keys()) if importances else None
    )
    if imp_horizon is not None and imp_horizon in importances:
        imp_dict = importances[imp_horizon]
        imp_series = pd.Series(imp_dict).sort_values(ascending=True).tail(15)

        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            y=imp_series.index, x=imp_series.values,
            orientation="h",
            marker=dict(
                color=imp_series.values,
                colorscale=[[0, "#1a1d29"], [0.5, "#5DADE2"], [1.0, "#E74C3C"]],
            ),
        ))
        fig_imp.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Permutation importance (mean decrease in score)",
            yaxis_title="Feature",
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_imp, width="stretch")
    else:
        st.info("No feature importance data available. Re-run train_model.py to generate it.")

    # --- DATA NOTES ---
    st.markdown("---")
    st.markdown('<div class="section-header">Data Notes & Limitations</div>',
                unsafe_allow_html=True)
    st.markdown(
        "- Trained on a **real, daily** Delhi-wide AQI dataset (2021-2024): "
        "PM2.5, PM10, NO2, SO2, CO, Ozone, AQI.\n"
        "- **No wind speed / temperature / humidity columns** exist in this dataset, "
        "so the 'what-if' controls act on pollutant levels (NO2/CO/SO2/Ozone) instead.\n"
        "- **No true per-station data** exists; the 4 stations shown are the single "
        "Delhi-wide forecast adjusted by fixed illustrative offsets, clearly for "
        "demo/UX purposes only.\n"
        "- Validation uses a **chronological (time-series) holdout split** -- no future "
        "data leaks into training.\n"
        "- The R2 of ~0.15-0.19 is typical for multi-day-ahead AQI forecasting without "
        "meteorological inputs. The backtest chart above shows the model captures "
        "seasonal patterns and spikes directionally, but individual-day precision is limited."
    )
