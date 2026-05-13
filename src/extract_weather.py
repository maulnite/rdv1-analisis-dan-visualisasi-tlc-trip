from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
import json

import pandas as pd
import requests

from src.clean_tlc import get_period_bounds, make_month_label


DATA_DIR = Path("/opt/airflow/data")
EXTERNAL_DIR = DATA_DIR / "external"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060
TIMEZONE = "America/New_York"


def ensure_external_directory() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] External directory is ready: {EXTERNAL_DIR}")


def get_weather_date_range(months: Iterable[str]) -> tuple[str, str]:
    """
    Open-Meteo memakai end_date inclusive.
    Pipeline kita memakai period_end exclusive.

    Contoh:
    months = ["2025-01", "2025-02", "2025-03"]

    get_period_bounds:
    start = 2025-01-01
    end_exclusive = 2025-04-01

    Weather API:
    start_date = 2025-01-01
    end_date   = 2025-03-31
    """
    period_start, period_end_exclusive = get_period_bounds(months)

    end_dt = datetime.strptime(period_end_exclusive, "%Y-%m-%d") - timedelta(days=1)
    end_date = end_dt.strftime("%Y-%m-%d")

    return period_start, end_date


def download_weather_data(months: Iterable[str]) -> str:
    """
    Download hourly historical weather data untuk NYC.

    Data ini nanti di-join dengan agg_hourly_demand berdasarkan:
    pickup_date + pickup_hour
    """
    ensure_external_directory()

    months = sorted(list(months))
    month_label = make_month_label(months)

    start_date, end_date = get_weather_date_range(months)

    output_path = EXTERNAL_DIR / f"weather_hourly_{month_label}.parquet"
    report_path = EXTERNAL_DIR / f"weather_report_{month_label}.json"

    if output_path.exists() and output_path.stat().st_size > 0 and report_path.exists() and report_path.stat().st_size > 0:
        print(f"[SKIP] Weather data already exists: {output_path}")
        print(f"[SKIP] Weather report already exists: {report_path}")
        return str(output_path)

    params = {
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation",
                "rain",
                "snowfall",
                "weather_code",
                "wind_speed_10m",
            ]
        ),
    }

    print("[STEP] Requesting weather data from Open-Meteo...")
    print(f"[INFO] Start date: {start_date}")
    print(f"[INFO] End date  : {end_date}")
    print(f"[INFO] Location  : lat={NYC_LATITUDE}, lon={NYC_LONGITUDE}")

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=(30, 300))
    response.raise_for_status()

    payload = response.json()

    if "hourly" not in payload:
        raise ValueError(f"Open-Meteo response does not contain hourly data: {payload}")

    hourly = payload["hourly"]
    hourly_units = payload.get("hourly_units", {})

    df = pd.DataFrame(hourly)

    if df.empty:
        raise ValueError("Weather dataframe is empty.")

    df["weather_datetime"] = pd.to_datetime(df["time"])
    df["pickup_date"] = df["weather_datetime"].dt.date.astype(str)
    df["pickup_hour"] = df["weather_datetime"].dt.hour

    df = df[
        [
            "weather_datetime",
            "pickup_date",
            "pickup_hour",
            "temperature_2m",
            "precipitation",
            "rain",
            "snowfall",
            "weather_code",
            "wind_speed_10m",
        ]
    ]

    df.to_parquet(output_path, index=False)

    report = {
        "source": "Open-Meteo Historical Weather API",
        "months": months,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "rows": len(df),
        "output_file": str(output_path),
        "hourly_units": hourly_units,
        "columns": list(df.columns),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    print("[RESULT] Weather data downloaded.")
    print(f"[RESULT] Rows: {len(df):,}")
    print(f"[SAVED] Weather parquet: {output_path}")
    print(f"[SAVED] Weather report : {report_path}")

    return str(output_path)


def validate_weather_outputs(months: Iterable[str]) -> None:
    months = sorted(list(months))
    month_label = make_month_label(months)

    weather_file = EXTERNAL_DIR / f"weather_hourly_{month_label}.parquet"
    report_file = EXTERNAL_DIR / f"weather_report_{month_label}.json"

    expected_files = [weather_file, report_file]
    missing_files = []

    for file_path in expected_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing weather output files: {missing_files}")

    df = pd.read_parquet(weather_file)

    if df.empty:
        raise ValueError("Weather output is empty.")

    print("[CHECK] Weather output files are valid.")
    print(f"[CHECK] Weather rows: {len(df):,}")
    print(f"[CHECK] Min datetime : {df['weather_datetime'].min()}")
    print(f"[CHECK] Max datetime : {df['weather_datetime'].max()}")


if __name__ == "__main__":
    from src.pipeline_config import MONTHS

    download_weather_data(months=MONTHS)
    validate_weather_outputs(months=MONTHS)