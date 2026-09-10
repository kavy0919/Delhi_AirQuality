"""
train_model.py
================
End-to-end data pipeline, feature engineering, and model training for the
Delhi-NCR AQI forecasting dashboard.

DATA NOTE (IMPORTANT):
-----------------------
The real dataset supplied for this project (`delhiaqi.csv`) is a DAILY,
CITY-WIDE aggregate series for Delhi covering 2021-01-01 through
2024-12-31 (1,461 rows). It contains:

    Date, Month, Year, Holidays_Count, Days,
    PM2.5, PM10, NO2, SO2, CO, Ozone, AQI

It does NOT contain hourly timestamps, per-station readings, or
meteorological variables (temperature / relative humidity / wind speed).
To keep this dashboard honest and demo-safe, the pipeline below is built
around what the real data actually supports:

    * Granularity      : daily (not hourly)
    * Forecast horizon  : N DAYS ahead (not hours ahead)
    * Geography         : one Delhi-wide series (no true per-station data)
    * Drivers available : PM2.5, PM10, NO2, SO2, CO, Ozone, Holidays_Count

The Streamlit app re-creates a 4-station *illustrative* view by applying
small, clearly-labelled fixed offsets to the single Delhi series (this is
disclosed in the UI) so the dashboard still demos well, but the underlying
model is trained ONLY on real, measured values.

If `delhiaqi.csv` (or whatever path you pass in) is missing, a synthetic
fallback with an IDENTICAL schema is generated automatically so the
project is always runnable end-to-end, even with zero setup.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
REAL_DATA_PATH = "delhiaqi.csv"
FALLBACK_DATA_PATH = "delhi_aqi_2022_2026.csv"
MODEL_PATH = "delhi_aqi_model.joblib"
FEATURE_COLUMNS_PATH = "feature_columns.joblib"
METRICS_PATH = "metrics.joblib"
FEATURED_DATA_PATH = "featured_data.csv"
BACKTEST_PATH = "backtest.csv"
IMPORTANCE_PATH = "feature_importances.joblib"

HORIZONS = [1, 2, 3, 5, 7]          # days ahead we train dedicated models for
PRIMARY_HORIZON = 3                  # used for the headline train/test report
TEST_FRACTION = 0.2                  # chronological holdout, no shuffling

POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]


# --------------------------------------------------------------------------
# 1. SYNTHETIC FALLBACK DATA (only used if no real CSV is found)
# --------------------------------------------------------------------------
def generate_dummy_data(output_path=FALLBACK_DATA_PATH, start_year=2022, n_years=4):
    """
    Generates a realistic DAILY synthetic Delhi AQI dataset matching the
    exact schema of the real `delhiaqi.csv` file, so the project can run
    even before the real file is available.

    Captures Delhi's well-known seasonal smog pattern: sharp PM2.5 / PM10
    spikes from late-October through January (stubble burning + winter
    inversion + Diwali firecrackers), cleaner air during the monsoon
    (July-September), weekday/weekend traffic effects, and realistic
    noise/co-pollutant correlation.
    """
    if os.path.exists(output_path):
        return output_path

    rng = np.random.default_rng(42)
    n_days = 365 * n_years + 1
    dates = pd.date_range(start=f"{start_year}-01-01", periods=n_days, freq="D")

    rows = []
    for d in dates:
        doy = d.dayofyear
        # Seasonal smog factor: peaks around day ~330 (late Nov) and dips in monsoon (~200)
        winter_peak = np.exp(-((doy - 15) % 365 - 0) ** 2 / (2 * 35 ** 2)) * 1.0
        winter_peak += np.exp(-((doy - 330) % 365) ** 2 / (2 * 35 ** 2)) * 1.0
        monsoon_dip = np.exp(-((doy - 200) ** 2) / (2 * 45 ** 2))
        seasonal = 1.0 + 1.8 * winter_peak - 0.55 * monsoon_dip

        weekday_factor = 0.92 if d.dayofweek >= 5 else 1.0  # slightly cleaner on weekends
        diwali_boost = 1.7 if (d.month == 11 and 1 <= d.day <= 5) else 1.0

        base_pm25 = 95 * seasonal * weekday_factor * diwali_boost
        pm25 = max(8, rng.normal(base_pm25, base_pm25 * 0.18))
        pm10 = max(15, pm25 * rng.uniform(1.4, 1.9))
        no2 = max(5, rng.normal(55 * seasonal * weekday_factor, 15))
        so2 = max(2, rng.normal(9 * seasonal, 3))
        co = max(0.2, rng.normal(1.1 * seasonal * weekday_factor, 0.35))
        ozone = max(2, rng.normal(38 * (1.3 - 0.3 * winter_peak), 12))

        # crude AQI proxy from PM2.5 sub-index (CPCB-like breakpoints)
        aqi = compute_aqi_from_pm25(pm25)

        rows.append({
            "Date": d.day,
            "Month": d.month,
            "Year": d.year,
            "Holidays_Count": 1 if diwali_boost > 1 else int(rng.random() < 0.02),
            "Days": d.dayofweek + 1,  # 1..7
            "PM2.5": round(pm25, 2),
            "PM10": round(pm10, 2),
            "NO2": round(no2, 2),
            "SO2": round(so2, 2),
            "CO": round(co, 2),
            "Ozone": round(ozone, 2),
            "AQI": int(round(aqi)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[generate_dummy_data] Synthetic fallback dataset written to '{output_path}' "
          f"({len(df)} rows).")
    return output_path


# --------------------------------------------------------------------------
# 2. CPCB-STYLE AQI SUB-INDEX FROM PM2.5 (used for forecasted-day AQI)
# --------------------------------------------------------------------------
_PM25_BREAKPOINTS = [
    # (C_lo, C_hi, I_lo, I_hi)
    (0.0, 30.0, 0, 50),
    (30.0, 60.0, 51, 100),
    (60.0, 90.0, 101, 200),
    (90.0, 120.0, 201, 300),
    (120.0, 250.0, 301, 400),
    (250.0, 350.0, 401, 450),
    (350.0, 500.0, 451, 500),
]


def compute_aqi_from_pm25(pm25: float) -> float:
    """Converts a PM2.5 concentration (µg/m3) into a CPCB-style AQI sub-index."""
    pm25 = max(0.0, float(pm25))
    for c_lo, c_hi, i_lo, i_hi in _PM25_BREAKPOINTS:
        if c_lo <= pm25 <= c_hi:
            return i_lo + (i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo)
    # above the top breakpoint: extrapolate linearly past 500
    c_lo, c_hi, i_lo, i_hi = _PM25_BREAKPOINTS[-1]
    return i_hi + (pm25 - c_hi) * ((i_hi - i_lo) / (c_hi - c_lo))


# --------------------------------------------------------------------------
# 3. LOAD RAW DATA (real dataset first, synthetic fallback second)
# --------------------------------------------------------------------------
def load_raw_data(real_path=REAL_DATA_PATH, fallback_path=FALLBACK_DATA_PATH):
    if os.path.exists(real_path):
        print(f"[load_raw_data] Using real dataset: '{real_path}'")
        df = pd.read_csv(real_path)
    else:
        print(f"[load_raw_data] '{real_path}' not found. Generating synthetic fallback data...")
        path = generate_dummy_data(fallback_path)
        df = pd.read_csv(path)

    required = {"Date", "Month", "Year", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Build a proper datetime index from Date/Month/Year and sort chronologically
    df["date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=df["Date"]), errors="coerce"
    )
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if "AQI" not in df.columns:
        df["AQI"] = df["PM2.5"].apply(compute_aqi_from_pm25)
    if "Holidays_Count" not in df.columns:
        df["Holidays_Count"] = 0
    if "Days" not in df.columns:
        df["Days"] = df["date"].dt.dayofweek + 1

    return df


# --------------------------------------------------------------------------
# 4. FEATURE ENGINEERING (shared by train_model.py and app.py)
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds temporal, cyclical, rolling, and lag features to a chronologically
    sorted daily dataframe. Returns a NEW dataframe (original is untouched).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # --- temporal features ---
    df["dayofweek"] = df["date"].dt.dayofweek       # 0=Mon
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # --- cyclical transforms ---
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # --- rolling statistics (PM2.5 / PM10): 3-day and 7-day windows ---
    for col in ["PM2.5", "PM10"]:
        df[f"{col}_roll_mean_3d"] = df[col].rolling(3, min_periods=1).mean().shift(1)
        df[f"{col}_roll_mean_7d"] = df[col].rolling(7, min_periods=1).mean().shift(1)
        df[f"{col}_roll_std_7d"] = df[col].rolling(7, min_periods=2).std().shift(1)

    # --- lag features: 1-day, 3-day, 7-day lags for all pollutants ---
    for col in POLLUTANT_COLS:
        for lag in (1, 3, 7):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # keep raw Holidays_Count as a feature
    df["Holidays_Count"] = df["Holidays_Count"].fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Returns the list of model-ready feature column names present in df."""
    exclude = {"date", "Date", "Month", "Year", "Days", "AQI"} | set(POLLUTANT_COLS)
    return [c for c in df.columns if c not in exclude and not c.startswith("target_")]


# --------------------------------------------------------------------------
# 5. TRAINING
# --------------------------------------------------------------------------
def make_model():
    """
    HistGradientBoostingRegressor is used (pure sklearn, no compiled
    LightGBM dependency needed -> zero install-failure risk on hackathon day).
    It natively handles NaNs — no scaler or imputer needed, and keeping
    the model unwrapped lets us directly access feature_importances_.
    """
    return HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        l2_regularization=0.1,
        random_state=42,
    )


def time_series_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Chronological split: earliest (1 - test_fraction) rows = train, rest = test."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_all_horizons(real_path=REAL_DATA_PATH, fallback_path=FALLBACK_DATA_PATH):
    raw_df = load_raw_data(real_path, fallback_path)
    featured_df = build_features(raw_df)
    feature_cols = get_feature_columns(featured_df)

    models = {}
    metrics = {}
    importances = {}
    backtest_frames = []

    for horizon in HORIZONS:
        work_df = featured_df.copy()
        work_df["target"] = work_df["PM2.5"].shift(-horizon)
        work_df = work_df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

        if len(work_df) < 30:
            print(f"[train_all_horizons] Skipping horizon={horizon} (not enough rows: {len(work_df)})")
            continue

        train_df, test_df = time_series_split(work_df)

        X_train, y_train = train_df[feature_cols], train_df["target"]
        X_test, y_test = test_df[feature_cols], test_df["target"]

        model = make_model()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        # Residual std for uncertainty bands
        residuals = y_test.values - preds
        residual_std = float(np.std(residuals))

        models[horizon] = model
        metrics[horizon] = {
            "rmse": rmse, "mae": mae, "r2": r2,
            "n_train": len(train_df), "n_test": len(test_df),
            "residual_std": residual_std,
        }

        # Feature importances via permutation importance on the test set
        perm_result = permutation_importance(
            model, X_test, y_test, n_repeats=5, random_state=42,
        )
        importances[horizon] = dict(
            zip(feature_cols, perm_result.importances_mean.tolist())
        )

        # Save backtest predictions for PRIMARY_HORIZON
        if horizon == PRIMARY_HORIZON:
            bt_df = test_df[["date"]].copy()
            bt_df["actual"] = y_test.values
            bt_df["predicted"] = preds
            bt_df["residual"] = residuals
            backtest_frames.append(bt_df)

        print(f"[horizon={horizon}d] RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}  "
              f"(train={len(train_df)}, test={len(test_df)}, resid_std={residual_std:.2f})")

    # Persist artifacts
    joblib.dump(models, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_COLUMNS_PATH)
    joblib.dump(metrics, METRICS_PATH)
    joblib.dump(importances, IMPORTANCE_PATH)
    featured_df.to_csv(FEATURED_DATA_PATH, index=False)

    if backtest_frames:
        pd.concat(backtest_frames, ignore_index=True).to_csv(BACKTEST_PATH, index=False)

    print(f"\nSaved: {MODEL_PATH}, {FEATURE_COLUMNS_PATH}, {METRICS_PATH}, "
          f"{IMPORTANCE_PATH}, {BACKTEST_PATH}, {FEATURED_DATA_PATH}")
    return models, feature_cols, metrics, featured_df


if __name__ == "__main__":
    print("=" * 70)
    print("Delhi-NCR AQI Forecasting — Training Pipeline")
    print("=" * 70)
    train_all_horizons()
    print("\nDone. Run `streamlit run app.py` to launch the dashboard.")

