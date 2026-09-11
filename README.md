# Delhi_AirQuality
# 🌫️ Delhi-NCR AQI Forecasting & Intelligence Dashboard

A self-contained, offline air quality forecasting dashboard for Delhi-NCR, built for a 24-hour hackathon. Trains a real machine learning model on historical Delhi AQI data, forecasts PM2.5 up to 7 days ahead, classifies results against India's official GRAP (Graded Response Action Plan) stages, and surfaces the results in an interactive Streamlit dashboard — with zero external APIs and zero paid services.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Important: Data Notes & Honest Limitations](#important-data-notes--honest-limitations)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Known Limitations & Roadmap](#known-limitations--roadmap)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Overview

Delhi-NCR regularly experiences hazardous air quality, especially in winter. This project forecasts PM2.5 concentrations using historical pollutant data and translates those forecasts into **actionable GRAP-stage policy actions and persona-specific health advisories** — the same framework India's Commission for Air Quality Management (CAQM) uses to trigger real restrictions (construction bans, vehicle restrictions, school closures, etc.).

Everything runs **100% locally**: no OpenAQ, no live weather APIs, no paid services. The model trains on a real daily Delhi AQI dataset and falls back to a realistic synthetic dataset automatically if the real file isn't present, so the project is always runnable out of the box.

## Features

- **ML Forecasting** — `HistGradientBoostingRegressor` models trained per forecast horizon (1–7 days ahead), validated with a chronological (leakage-free) time-series split.
- **GRAP Stage Classification** — deterministic rule engine mapping predicted PM2.5/AQI to Stage I–IV per CAQM's published thresholds.
- **Persona-Specific Health Advisories** — tailored guidance for vulnerable/asthmatic individuals, morning walkers/athletes, and the general public.
- **Interactive What-If Simulation** — adjust regional pollutant levels (NO2, CO, SO2, Ozone) and see the forecast and GRAP stage update live.
- **Spatial View** — map of illustrative station-level severity across Anand Vihar, RK Puram, Punjabi Bagh, and ITO.
- **Model Transparency Panel** — backtest chart (predicted vs. actual on holdout data), permutation-based feature importance, and full RMSE/MAE/R² metrics per horizon.
- **Tabbed Dashboard** — Overview, Forecast, What-If Simulation, Spatial View, Advisory Board, and Model Transparency.
- **Zero external dependencies at runtime** — no API keys, no internet connection required after `pip install`.

## Important: Data Notes & Honest Limitations

This project is built to be transparent about what its data can and cannot support — read this before demoing or extending it.

The real dataset (`delhiaqi.csv`) is a **daily, city-wide aggregate** series for Delhi covering **2021–2024** (1,461 rows), containing:

```
Date, Month, Year, Holidays_Count, Days, PM2.5, PM10, NO2, SO2, CO, Ozone, AQI
```

It does **not** contain:
- Hourly timestamps (data is daily-resolution only)
- True per-station sensor readings (it's one Delhi-wide series)
- Meteorological variables — no temperature, humidity, or wind speed

To keep the dashboard honest rather than pretending to have data it doesn't:

| Original brief | What's actually implemented |
|---|---|
| Hourly forecast horizon (1–12h) | **Daily** forecast horizon (1–7 days) |
| Multi-station sensor data | Single Delhi-wide model; the 4 "stations" shown are the real forecast adjusted by **fixed, disclosed illustrative offsets** for demo purposes only |
| Wind speed / temperature what-if sliders | **Pollutant-level what-if sliders** (NO2, CO, SO2, Ozone) — the drivers actually present in the data |

These limitations are disclosed directly in the app UI (sidebar and Model Transparency tab), not hidden. If you have access to true hourly, per-station, or meteorological data, swapping it in is straightforward — see [Roadmap](#known-limitations--roadmap).

## Tech Stack

- **Python 3.9+**
- **scikit-learn** — `HistGradientBoostingRegressor` (no compiled LightGBM dependency, so zero install-failure risk during a hackathon demo)
- **Streamlit** — dashboard framework
- **Plotly** — interactive charts (trend lines, backtest, feature importance)
- **pydeck** — spatial map (token-free CARTO basemap, no Mapbox API key required)
- **pandas / numpy / joblib** — data pipeline and model persistence

## Project Structure

```
.
├── train_model.py          # Data loading, feature engineering, model training
├── advisory_rules.py       # GRAP stage classifier + health advisory rule engine
├── app.py                  # Streamlit dashboard
├── delhiaqi.csv            # Real Delhi daily AQI dataset (2021-2024)
├── requirements.txt        # Python dependencies
│
├── delhi_aqi_model.joblib      # (generated) trained models per horizon
├── feature_columns.joblib      # (generated) feature column list
├── metrics.joblib              # (generated) RMSE/MAE/R² per horizon
├── featured_data.csv           # (generated) fully feature-engineered dataset
└── delhi_aqi_2022_2026.csv     # (generated only if delhiaqi.csv is missing) synthetic fallback data
```

The `.joblib` and generated `.csv` artifacts are **not** committed to the repo — they're created automatically on first run. Add them to `.gitignore` if you fork this.

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/kavy0919/Delhi_AirQuality/edit/main/README.md
cd AQI H

# (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

On first launch, `app.py` automatically trains the models if the artifacts (`delhi_aqi_model.joblib`, etc.) don't already exist — this takes a few seconds. Subsequent launches load the cached artifacts instantly.

To force a retrain (e.g. after changing feature engineering or swapping in new data):

```bash
rm delhi_aqi_model.joblib feature_columns.joblib metrics.joblib featured_data.csv
streamlit run app.py
```

You can also run the training pipeline standalone to inspect metrics in the terminal:

```bash
python train_model.py
```

## How It Works

1. **`train_model.py`** loads `delhi_aqi_2022_2026.csv`/`delhiaqi.csv` (real data preferred; synthetic fallback auto-generated if missing), builds a proper datetime index, and engineers features:
   - Temporal: day-of-week, month, day-of-year, weekend flag
   - Cyclical transforms: sin/cos of day-of-week, month, and day-of-year
   - Rolling statistics: 3-day and 7-day mean/std for PM2.5 and PM10
   - Lag features: 1/3/7-day lags for all six pollutants
   
   It then trains one `HistGradientBoostingRegressor` per forecast horizon (1, 2, 3, 5, 7 days), using a strict chronological train/test split to prevent data leakage, and saves all artifacts via `joblib`.

2. **`advisory_rules.py`** is a pure rule engine (no ML) that maps a PM2.5 or AQI value to one of five GRAP bands (Below GRAP, Stage I–IV) per CAQM's published thresholds, returning both municipal policy actions and persona-specific health advisories for each stage.

3. **`app.py`** loads the trained models, lets the user pick a station, forecast horizon, and what-if pollutant deltas, runs the appropriate horizon model, converts the PM2.5 forecast to an AQI value via the CPCB breakpoint formula, classifies the GRAP stage, and renders everything across the dashboard's tabs.

## Model Performance

Validated on a chronological (no-leakage) holdout — the most recent ~20% of the dataset:

| Horizon (days) | RMSE | MAE | R² |
|---|---|---|---|
| 1 | ~64.5 | ~27.7 | ~0.19 |
| 2 | ~65.7 | ~27.9 | ~0.16 |
| 3 | ~65.6 | ~29.1 | ~0.16 |
| 5 | ~64.4 | ~29.6 | ~0.19 |
| 7 | ~68.7 | ~34.9 | ~0.08 |

*(Exact values regenerate on each training run and are viewable live in the app's Model Transparency tab, alongside a backtest chart and permutation feature importance.)*

An R² in the 0.08–0.19 range is a realistic result for multi-day-ahead AQI forecasting **without meteorological inputs** (wind and temperature are major drivers of pollutant dispersion and aren't available in this dataset). The backtest chart in the app shows the model tracks seasonal pollution patterns directionally, while individual-day precision is limited — this is disclosed rather than hidden behind a bare metric.

## Known Limitations & Roadmap

- **No true per-station data** — station views are illustrative offsets on a single Delhi-wide series. Swap in real per-station CPCB monitor data to remove this limitation.
- **No meteorological inputs** — adding wind speed, temperature, and humidity (e.g. from a historical weather archive) would likely improve R² substantially, since these are primary drivers of pollutant dispersion.
- **Daily, not hourly, resolution** — an hourly-resolution dataset would allow a true hourly forecast horizon as originally scoped.
- **Correlational, not causal, what-if sliders** — the model is a tree-based regressor, not a physical emissions simulator; slider interactions can produce non-additive results since the model captures correlations in historical data, not causal mechanisms.
- **Outlier sensitivity** — the historical data contains at least one extreme PM2.5 spike that the model doesn't capture well; worth investigating as a genuine severe-pollution event vs. a data quality issue before further model tuning.

## Disclaimer

The GRAP stages, policy actions, and health advisories in this dashboard are generated **deterministically from published CAQM thresholds** applied to a **forecasted** value. They are **not official government orders** and should not be used as a substitute for real-time monitoring or official CAQM/DPCC/CPCB notifications. Always refer to official sources for binding air quality actions.

## License

MIT License — feel free to fork, adapt, and extend for your own hackathon or research projects. Attribution appreciated but not required.
