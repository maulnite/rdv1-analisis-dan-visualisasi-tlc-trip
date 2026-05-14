from pathlib import Path
from typing import Iterable
import json
import math

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.clean_tlc import make_month_label


DATA_DIR = Path("/opt/airflow/data")
CURATED_DIR = DATA_DIR / "curated"
ML_DIR = DATA_DIR / "ml"


def ensure_ml_directory() -> None:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] ML directory is ready: {ML_DIR}")


def get_weather_condition(row: pd.Series) -> str:
    snowfall = row.get("snowfall", 0) or 0
    rain = row.get("rain", 0) or 0
    precipitation = row.get("precipitation", 0) or 0
    intensity = max(rain, precipitation)

    if snowfall > 0:
        return "snow"
    if intensity > 10:
        return "heavy_rain"
    if intensity >= 2.5:
        return "moderate_rain"
    if intensity > 0:
        return "light_rain"
    return "clear"


def mape_safe(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return 0.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def load_training_data(months: Iterable[str]) -> pd.DataFrame:
    month_label = make_month_label(months)
    input_path = CURATED_DIR / f"agg_hourly_demand_weather_{month_label}.parquet"

    if not input_path.exists() or input_path.stat().st_size == 0:
        raise FileNotFoundError(f"Input file not found or empty: {input_path}")

    print(f"[INFO] Loading training data: {input_path}")

    con = duckdb.connect()
    df = con.sql(
        f"""
        SELECT
            pickup_date,
            pickup_hour,
            pickup_day_name,
            is_weekend,
            pickup_location_id,
            pickup_borough,
            pickup_zone,

            temperature_2m,
            precipitation,
            rain,
            snowfall,
            wind_speed_10m,

            total_trips
        FROM read_parquet('{input_path}')
        WHERE
            total_trips IS NOT NULL
            AND total_trips >= 0
            AND pickup_location_id IS NOT NULL
            AND pickup_borough IS NOT NULL
            AND pickup_zone IS NOT NULL
            AND temperature_2m IS NOT NULL
        """
    ).fetchdf()
    con.close()

    if df.empty:
        raise ValueError("Training dataframe is empty.")

    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    df["pickup_dayofweek"] = df["pickup_date"].dt.dayofweek
    df["pickup_month"] = df["pickup_date"].dt.month
    df["pickup_day"] = df["pickup_date"].dt.day

    df["weather_condition"] = df.apply(get_weather_condition, axis=1)

    df["is_weekend"] = df["is_weekend"].astype(int)
    df["pickup_location_id"] = df["pickup_location_id"].astype(int)

    numeric_cols = [
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
    ]

    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    print(f"[INFO] Loaded rows: {len(df):,}")
    print(f"[INFO] Date range : {df['pickup_date'].min()} to {df['pickup_date'].max()}")

    return df


def train_demand_prediction_model(months: Iterable[str]) -> dict:
    ensure_ml_directory()

    months = sorted(list(months))
    month_label = make_month_label(months)

    results_path = ML_DIR / f"demand_prediction_results_{month_label}.parquet"
    metrics_path = ML_DIR / f"demand_model_metrics_{month_label}.json"
    feature_importance_path = ML_DIR / f"demand_feature_importance_{month_label}.parquet"
    model_path = ML_DIR / f"demand_prediction_model_{month_label}.joblib"

    for path in [results_path, metrics_path, feature_importance_path, model_path]:
        if path.exists():
            path.unlink()
            print(f"[INFO] Removed old output: {path}")

    df = load_training_data(months)

    # Time-based split:
    # train = Jan-Feb, test = Mar
    split_date = pd.Timestamp("2025-03-01")

    train_df = df[df["pickup_date"] < split_date].copy()
    test_df = df[df["pickup_date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced empty dataframe.")

    feature_cols = [
        "pickup_hour",
        "pickup_dayofweek",
        "pickup_month",
        "pickup_day",
        "is_weekend",
        "pickup_location_id",
        "pickup_borough",
        "pickup_zone",
        "weather_condition",
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
    ]

    categorical_cols = [
        "pickup_borough",
        "pickup_zone",
        "weather_condition",
    ]

    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    target_col = "total_trips"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print(f"[INFO] Train rows: {len(train_df):,}")
    print(f"[INFO] Test rows : {len(test_df):,}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_cols,
            ),
            ("numeric", "passthrough", numeric_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=60,
        max_depth=18,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("[STEP] Training demand prediction model...")
    pipeline.fit(X_train, y_train)

    print("[STEP] Predicting test data...")
    y_pred = pipeline.predict(X_test)
    y_pred = np.maximum(y_pred, 0)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    mape = float(mape_safe(y_test, y_pred))

    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_rmse = float(math.sqrt(mean_squared_error(y_test, baseline_pred)))

    result_df = test_df[
        [
            "pickup_date",
            "pickup_hour",
            "pickup_day_name",
            "is_weekend",
            "pickup_location_id",
            "pickup_borough",
            "pickup_zone",
            "weather_condition",
            "temperature_2m",
            "precipitation",
            "rain",
            "snowfall",
            "wind_speed_10m",
            "total_trips",
        ]
    ].copy()

    result_df["predicted_total_trips"] = np.round(y_pred, 2)
    result_df["prediction_error"] = result_df["total_trips"] - result_df["predicted_total_trips"]
    result_df["absolute_error"] = result_df["prediction_error"].abs()

    result_df.to_parquet(results_path, index=False)

    fitted_model = pipeline.named_steps["model"]
    transformed_feature_names = categorical_cols + numeric_cols

    importance_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": fitted_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_parquet(feature_importance_path, index=False)

    joblib.dump(pipeline, model_path)

    metrics = {
        "months": months,
        "model": "RandomForestRegressor",
        "target": target_col,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_period": {
            "min_date": str(train_df["pickup_date"].min().date()),
            "max_date": str(train_df["pickup_date"].max().date()),
        },
        "test_period": {
            "min_date": str(test_df["pickup_date"].min().date()),
            "max_date": str(test_df["pickup_date"].max().date()),
        },
        "features": feature_cols,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape_percent": mape,
            "r2": r2,
        },
        "baseline_metrics": {
            "baseline_mean_mae": baseline_mae,
            "baseline_mean_rmse": baseline_rmse,
        },
        "outputs": {
            "prediction_results": str(results_path),
            "feature_importance": str(feature_importance_path),
            "model": str(model_path),
        },
        "interpretation": {
            "mae": "Average absolute difference between actual and predicted trip count.",
            "rmse": "Penalizes large prediction errors more strongly.",
            "r2": "Explains how much variance in trip demand is captured by the model.",
            "mape_percent": "Average percentage error, excluding rows with zero actual trips.",
        },
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, default=str)

    print("[RESULT] Demand prediction model completed.")
    print(f"[RESULT] MAE : {mae:.4f}")
    print(f"[RESULT] RMSE: {rmse:.4f}")
    print(f"[RESULT] MAPE: {mape:.2f}%")
    print(f"[RESULT] R2  : {r2:.4f}")
    print(f"[SAVED] Results           : {results_path}")
    print(f"[SAVED] Metrics           : {metrics_path}")
    print(f"[SAVED] Feature importance: {feature_importance_path}")
    print(f"[SAVED] Model             : {model_path}")

    return metrics


def validate_demand_prediction_outputs(months: Iterable[str]) -> None:
    months = sorted(list(months))
    month_label = make_month_label(months)

    expected_files = [
        ML_DIR / f"demand_prediction_results_{month_label}.parquet",
        ML_DIR / f"demand_model_metrics_{month_label}.json",
        ML_DIR / f"demand_feature_importance_{month_label}.parquet",
        ML_DIR / f"demand_prediction_model_{month_label}.joblib",
    ]

    missing_files = []

    for file_path in expected_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing ML output files: {missing_files}")

    results_df = pd.read_parquet(expected_files[0])
    importance_df = pd.read_parquet(expected_files[2])

    if results_df.empty:
        raise ValueError("Demand prediction results are empty.")

    if importance_df.empty:
        raise ValueError("Feature importance output is empty.")

    required_columns = [
        "total_trips",
        "predicted_total_trips",
        "prediction_error",
        "absolute_error",
    ]

    for col in required_columns:
        if col not in results_df.columns:
            raise ValueError(f"Missing required prediction result column: {col}")

    print("[CHECK] Demand prediction outputs are valid.")
    print(f"[CHECK] Prediction result rows: {len(results_df):,}")
    print(f"[CHECK] Feature importance rows: {len(importance_df):,}")


if __name__ == "__main__":
    from src.pipeline_config import MONTHS

    train_demand_prediction_model(months=MONTHS)
    validate_demand_prediction_outputs(months=MONTHS)