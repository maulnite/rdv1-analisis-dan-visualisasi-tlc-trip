from pathlib import Path
from typing import Iterable
import json

import duckdb

from src.clean_tlc import make_month_label


DATA_DIR = Path("/opt/airflow/data")
EXTERNAL_DIR = DATA_DIR / "external"
CURATED_DIR = DATA_DIR / "curated"


def get_hourly_demand_file(months: Iterable[str]) -> Path:
    month_label = make_month_label(months)
    file_path = CURATED_DIR / f"agg_hourly_demand_{month_label}.parquet"

    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"Hourly demand file not found or empty: {file_path}")

    return file_path


def get_weather_file(months: Iterable[str]) -> Path:
    month_label = make_month_label(months)
    file_path = EXTERNAL_DIR / f"weather_hourly_{month_label}.parquet"

    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"Weather file not found or empty: {file_path}")

    return file_path


def build_hourly_demand_weather(months: Iterable[str]) -> str:
    """
    Join hourly taxi demand dengan hourly weather data.

    Join key:
    - pickup_date
    - pickup_hour
    """
    months = sorted(list(months))
    month_label = make_month_label(months)

    hourly_demand_file = get_hourly_demand_file(months)
    weather_file = get_weather_file(months)

    output_path = CURATED_DIR / f"agg_hourly_demand_weather_{month_label}.parquet"
    report_path = CURATED_DIR / f"weather_join_report_{month_label}.json"

    if output_path.exists():
        output_path.unlink()

    if report_path.exists():
        report_path.unlink()

    con = duckdb.connect()

    print("[STEP] Joining hourly demand with weather data...")

    con.execute(
        f"""
        COPY (
            SELECT
                h.pickup_date,
                h.pickup_hour,
                h.pickup_day_name,
                h.is_weekend,

                h.pickup_location_id,
                h.pickup_borough,
                h.pickup_zone,

                h.total_trips,
                h.total_passengers,

                h.avg_trip_distance,
                h.avg_trip_duration_minutes,
                h.avg_speed_mph,

                h.total_revenue,
                h.avg_total_amount,
                h.total_tip,
                h.avg_tip_percentage,

                w.weather_datetime,
                w.temperature_2m,
                w.precipitation,
                w.rain,
                w.snowfall,
                w.weather_code,
                w.wind_speed_10m,

                CASE
                    WHEN COALESCE(w.precipitation, 0) > 0
                    THEN TRUE
                    ELSE FALSE
                END AS is_precipitation_hour,

                CASE
                    WHEN COALESCE(w.rain, 0) > 0
                    THEN TRUE
                    ELSE FALSE
                END AS is_rain_hour

            FROM read_parquet('{hourly_demand_file}') h
            LEFT JOIN read_parquet('{weather_file}') w
                ON h.pickup_date = CAST(w.pickup_date AS DATE)
                AND h.pickup_hour = w.pickup_hour
            ORDER BY
                h.pickup_date,
                h.pickup_hour,
                h.total_trips DESC
        )
        TO '{output_path}'
        (FORMAT PARQUET);
        """
    )

    demand_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{hourly_demand_file}')"
    ).fetchone()[0]

    joined_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
    ).fetchone()[0]

    weather_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{weather_file}')"
    ).fetchone()[0]

    missing_weather_rows = con.sql(
        f"""
        SELECT
            SUM(CASE WHEN temperature_2m IS NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{output_path}')
        """
    ).fetchone()[0]

    weather_effect_summary = con.sql(
        f"""
        SELECT
            is_precipitation_hour,
            COUNT(*) AS total_zone_hours,
            SUM(total_trips) AS total_trips,
            ROUND(AVG(avg_trip_duration_minutes), 2) AS avg_trip_duration_minutes,
            ROUND(SUM(total_revenue), 2) AS total_revenue,
            ROUND(AVG(avg_tip_percentage), 2) AS avg_tip_percentage
        FROM read_parquet('{output_path}')
        GROUP BY is_precipitation_hour
        ORDER BY is_precipitation_hour
        """
    ).fetchdf().to_dict(orient="records")

    report = {
        "months": months,
        "input_hourly_demand": str(hourly_demand_file),
        "input_weather": str(weather_file),
        "output_file": str(output_path),
        "row_counts": {
            "hourly_demand_rows": demand_rows,
            "weather_rows": weather_rows,
            "joined_rows": joined_rows,
            "missing_weather_rows": missing_weather_rows,
        },
        "weather_effect_summary": weather_effect_summary,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    con.close()

    print("[RESULT] Weather join completed.")
    print(f"[RESULT] Hourly demand rows : {demand_rows:,}")
    print(f"[RESULT] Weather rows       : {weather_rows:,}")
    print(f"[RESULT] Joined rows        : {joined_rows:,}")
    print(f"[RESULT] Missing weather    : {missing_weather_rows:,}")
    print(f"[SAVED] Joined output      : {output_path}")
    print(f"[SAVED] Join report        : {report_path}")

    return str(output_path)


def validate_weather_join_outputs(months: Iterable[str]) -> None:
    months = sorted(list(months))
    month_label = make_month_label(months)

    output_file = CURATED_DIR / f"agg_hourly_demand_weather_{month_label}.parquet"
    report_file = CURATED_DIR / f"weather_join_report_{month_label}.json"

    expected_files = [output_file, report_file]
    missing_files = []

    for file_path in expected_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing weather join output files: {missing_files}")

    con = duckdb.connect()

    row_count = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{output_file}')"
    ).fetchone()[0]

    missing_weather_rows = con.sql(
        f"""
        SELECT
            SUM(CASE WHEN temperature_2m IS NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{output_file}')
        """
    ).fetchone()[0]

    con.close()

    if row_count == 0:
        raise ValueError("Weather join output is empty.")

    if missing_weather_rows not in (0, None):
        raise ValueError(f"Weather join still has missing weather rows: {missing_weather_rows}")

    print("[CHECK] Weather join output is valid.")
    print(f"[CHECK] Rows: {row_count:,}")
    print(f"[CHECK] Missing weather rows: {missing_weather_rows}")


if __name__ == "__main__":
    from src.pipeline_config import MONTHS

    build_hourly_demand_weather(months=MONTHS)
    validate_weather_join_outputs(months=MONTHS)