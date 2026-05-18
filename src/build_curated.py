# /*
#  * 245150200111028 Aditya Akbar
#  * 245150207111038 Mohammad Geisar Rampan
#  * 245150207111103 Muhammad Sulthon Aulia Wijaya
#  * 245150207111050 Orie Abyan Maulana
#  * 235150201111068 Pieter Christy Yan Yudhistira
#  */
from pathlib import Path
from typing import Iterable
import json

import duckdb

from src.clean_tlc import make_month_label


DATA_DIR = Path("/opt/airflow/data")
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
CURATED_DIR = DATA_DIR / "curated"


def ensure_curated_directory() -> None:
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Curated directory is ready: {CURATED_DIR}")


def get_processed_file(months: Iterable[str], taxi_type: str) -> Path:
    month_label = make_month_label(months)
    file_path = PROCESSED_DIR / f"clean_{taxi_type}_tripdata_{month_label}.parquet"

    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"Processed file not found or empty: {file_path}")

    return file_path


def get_zone_lookup_file() -> Path:
    file_path = EXTERNAL_DIR / "taxi_zone_lookup.csv"

    if not file_path.exists() or file_path.stat().st_size == 0:
        raise FileNotFoundError(f"Taxi zone lookup file not found or empty: {file_path}")

    return file_path


def remove_existing_outputs(output_paths: list[Path]) -> None:
    for path in output_paths:
        if path.exists():
            path.unlink()
            print(f"[INFO] Removed old output: {path}")


def build_curated_tables(months: Iterable[str], taxi_type: str) -> dict:
    """
    Membuat curated tables untuk dashboard:
    - dim_location
    - fact_trips
    - agg_hourly_demand
    - agg_zone_summary
    - agg_daily_summary
    """
    ensure_curated_directory()

    months = sorted(list(months))
    month_label = make_month_label(months)

    clean_file = get_processed_file(months=months, taxi_type=taxi_type)
    zone_file = get_zone_lookup_file()

    dim_location_path = CURATED_DIR / f"dim_location_{month_label}.parquet"
    fact_trips_path = CURATED_DIR / f"fact_trips_{month_label}.parquet"
    agg_hourly_path = CURATED_DIR / f"agg_hourly_demand_{month_label}.parquet"
    agg_zone_path = CURATED_DIR / f"agg_zone_summary_{month_label}.parquet"
    agg_daily_path = CURATED_DIR / f"agg_daily_summary_{month_label}.parquet"
    report_path = CURATED_DIR / f"curated_report_{month_label}.json"

    output_paths = [
        dim_location_path,
        fact_trips_path,
        agg_hourly_path,
        agg_zone_path,
        agg_daily_path,
        report_path,
    ]

    remove_existing_outputs(output_paths)

    con = duckdb.connect()

    print("[STEP] Building dim_location...")

    con.execute(
        f"""
        COPY (
            SELECT
                LocationID AS location_id,
                Borough AS borough,
                Zone AS zone,
                service_zone
            FROM read_csv_auto('{zone_file}')
            ORDER BY location_id
        )
        TO '{dim_location_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Building fact_trips...")

    con.execute(
        f"""
        COPY (
            SELECT
                row_number() OVER () AS trip_id,

                t.VendorID AS vendor_id,
                t.tpep_pickup_datetime AS pickup_datetime,
                t.tpep_dropoff_datetime AS dropoff_datetime,

                t.pickup_date,
                t.pickup_hour,
                t.pickup_month,
                t.pickup_dayofweek,
                t.pickup_day_name,
                t.is_weekend,

                t.passenger_count,
                t.trip_distance,
                t.trip_duration_minutes,
                t.average_speed_mph,

                t.PULocationID AS pickup_location_id,
                pu.Borough AS pickup_borough,
                pu.Zone AS pickup_zone,
                pu.service_zone AS pickup_service_zone,

                t.DOLocationID AS dropoff_location_id,
                dozone.Borough AS dropoff_borough,
                dozone.Zone AS dropoff_zone,
                dozone.service_zone AS dropoff_service_zone,

                t.payment_type,
                t.fare_amount,
                t.tip_amount,
                t.tip_percentage,
                t.total_amount,
                t.tolls_amount,
                t.extra,
                t.mta_tax,
                t.congestion_surcharge,
                t.Airport_fee

            FROM read_parquet('{clean_file}') t
            LEFT JOIN read_csv_auto('{zone_file}') pu
                ON t.PULocationID = pu.LocationID
            LEFT JOIN read_csv_auto('{zone_file}') dozone
                ON t.DOLocationID = dozone.LocationID
        )
        TO '{fact_trips_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Building agg_hourly_demand...")

    con.execute(
        f"""
        COPY (
            SELECT
                pickup_date,
                pickup_hour,
                pickup_day_name,
                is_weekend,

                pickup_location_id,
                pickup_borough,
                pickup_zone,

                COUNT(*) AS total_trips,
                SUM(passenger_count) AS total_passengers,

                ROUND(AVG(trip_distance), 2) AS avg_trip_distance,
                ROUND(AVG(trip_duration_minutes), 2) AS avg_trip_duration_minutes,
                ROUND(AVG(average_speed_mph), 2) AS avg_speed_mph,

                ROUND(SUM(total_amount), 2) AS total_revenue,
                ROUND(AVG(total_amount), 2) AS avg_total_amount,
                ROUND(SUM(tip_amount), 2) AS total_tip,
                ROUND(AVG(tip_percentage), 2) AS avg_tip_percentage

            FROM read_parquet('{fact_trips_path}')
            GROUP BY
                pickup_date,
                pickup_hour,
                pickup_day_name,
                is_weekend,
                pickup_location_id,
                pickup_borough,
                pickup_zone
            ORDER BY
                pickup_date,
                pickup_hour,
                total_trips DESC
        )
        TO '{agg_hourly_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Building agg_zone_summary...")

    con.execute(
        f"""
        COPY (
            SELECT
                pickup_location_id,
                pickup_borough,
                pickup_zone,
                pickup_service_zone,

                COUNT(*) AS total_trips,
                SUM(passenger_count) AS total_passengers,

                ROUND(SUM(total_amount), 2) AS total_revenue,
                ROUND(AVG(total_amount), 2) AS avg_total_amount,

                ROUND(AVG(trip_distance), 2) AS avg_trip_distance,
                ROUND(AVG(trip_duration_minutes), 2) AS avg_trip_duration_minutes,
                ROUND(AVG(average_speed_mph), 2) AS avg_speed_mph,

                ROUND(SUM(tip_amount), 2) AS total_tip,
                ROUND(AVG(tip_percentage), 2) AS avg_tip_percentage,

                MIN(pickup_datetime) AS first_pickup_datetime,
                MAX(pickup_datetime) AS last_pickup_datetime

            FROM read_parquet('{fact_trips_path}')
            GROUP BY
                pickup_location_id,
                pickup_borough,
                pickup_zone,
                pickup_service_zone
            ORDER BY total_trips DESC
        )
        TO '{agg_zone_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Building agg_daily_summary...")

    con.execute(
        f"""
        COPY (
            SELECT
                pickup_date,
                pickup_day_name,
                is_weekend,

                COUNT(*) AS total_trips,
                SUM(passenger_count) AS total_passengers,

                ROUND(SUM(total_amount), 2) AS total_revenue,
                ROUND(AVG(total_amount), 2) AS avg_total_amount,

                ROUND(AVG(trip_distance), 2) AS avg_trip_distance,
                ROUND(AVG(trip_duration_minutes), 2) AS avg_trip_duration_minutes,

                ROUND(SUM(tip_amount), 2) AS total_tip,
                ROUND(AVG(tip_percentage), 2) AS avg_tip_percentage

            FROM read_parquet('{fact_trips_path}')
            GROUP BY
                pickup_date,
                pickup_day_name,
                is_weekend
            ORDER BY pickup_date
        )
        TO '{agg_daily_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Creating curated report...")

    fact_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{fact_trips_path}')"
    ).fetchone()[0]

    hourly_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{agg_hourly_path}')"
    ).fetchone()[0]

    zone_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{agg_zone_path}')"
    ).fetchone()[0]

    daily_rows = con.sql(
        f"SELECT COUNT(*) FROM read_parquet('{agg_daily_path}')"
    ).fetchone()[0]

    top_zones = con.sql(
        f"""
        SELECT
            pickup_borough,
            pickup_zone,
            total_trips,
            total_revenue
        FROM read_parquet('{agg_zone_path}')
        ORDER BY total_trips DESC
        LIMIT 10
        """
    ).fetchdf().to_dict(orient="records")

    report = {
        "taxi_type": taxi_type,
        "months": months,
        "input_file": str(clean_file),
        "outputs": {
            "dim_location": str(dim_location_path),
            "fact_trips": str(fact_trips_path),
            "agg_hourly_demand": str(agg_hourly_path),
            "agg_zone_summary": str(agg_zone_path),
            "agg_daily_summary": str(agg_daily_path),
        },
        "row_counts": {
            "fact_trips": fact_rows,
            "agg_hourly_demand": hourly_rows,
            "agg_zone_summary": zone_rows,
            "agg_daily_summary": daily_rows,
        },
        "top_10_pickup_zones_by_trips": top_zones,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    con.close()

    print("[RESULT] Curated tables completed.")
    print(f"[RESULT] fact_trips rows        : {fact_rows:,}")
    print(f"[RESULT] agg_hourly_demand rows : {hourly_rows:,}")
    print(f"[RESULT] agg_zone_summary rows  : {zone_rows:,}")
    print(f"[RESULT] agg_daily_summary rows : {daily_rows:,}")
    print(f"[SAVED] Report: {report_path}")

    return report


def validate_curated_outputs(months: Iterable[str], taxi_type: str) -> None:
    months = sorted(list(months))
    month_label = make_month_label(months)

    expected_files = [
        CURATED_DIR / f"dim_location_{month_label}.parquet",
        CURATED_DIR / f"fact_trips_{month_label}.parquet",
        CURATED_DIR / f"agg_hourly_demand_{month_label}.parquet",
        CURATED_DIR / f"agg_zone_summary_{month_label}.parquet",
        CURATED_DIR / f"agg_daily_summary_{month_label}.parquet",
        CURATED_DIR / f"curated_report_{month_label}.json",
    ]

    missing_files = []

    for file_path in expected_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing curated output files: {missing_files}")

    con = duckdb.connect()

    fact_file = CURATED_DIR / f"fact_trips_{month_label}.parquet"
    zone_file = CURATED_DIR / f"agg_zone_summary_{month_label}.parquet"
    daily_file = CURATED_DIR / f"agg_daily_summary_{month_label}.parquet"

    fact_rows = con.sql(f"SELECT COUNT(*) FROM read_parquet('{fact_file}')").fetchone()[0]
    zone_rows = con.sql(f"SELECT COUNT(*) FROM read_parquet('{zone_file}')").fetchone()[0]
    daily_rows = con.sql(f"SELECT COUNT(*) FROM read_parquet('{daily_file}')").fetchone()[0]

    con.close()

    if fact_rows == 0:
        raise ValueError("fact_trips is empty.")

    if zone_rows == 0:
        raise ValueError("agg_zone_summary is empty.")

    if daily_rows == 0:
        raise ValueError("agg_daily_summary is empty.")

    print("[CHECK] Curated output files are valid.")
    print(f"[CHECK] fact_trips rows       : {fact_rows:,}")
    print(f"[CHECK] agg_zone_summary rows : {zone_rows:,}")
    print(f"[CHECK] agg_daily_summary rows: {daily_rows:,}")


if __name__ == "__main__":
    from src.pipeline_config import MONTHS, TAXI_TYPE

    build_curated_tables(months=MONTHS, taxi_type=TAXI_TYPE)
    validate_curated_outputs(months=MONTHS, taxi_type=TAXI_TYPE)
