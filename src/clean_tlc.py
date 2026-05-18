# /*
#  * 245150200111028 Aditya Akbar
#  * 245150207111038 Mohammad Geisar Rampan
#  * 245150207111103 Muhammad Sulthon Aulia Wijaya
#  * 245150207111050 Orie Abyan Maulana
#  * 235150201111068 Pieter Christy Yan Yudhistira
#  */
from datetime import datetime
from pathlib import Path
from typing import Iterable
import json

import duckdb


DATA_DIR = Path("/opt/airflow/data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_processed_directory() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Processed directory is ready: {PROCESSED_DIR}")


def get_period_bounds(months: Iterable[str]) -> tuple[str, str]:
    """
    Mengubah daftar bulan menjadi periode start dan end exclusive.

    Contoh:
    months = ["2025-01", "2025-02", "2025-03"]

    hasil:
    start = "2025-01-01"
    end   = "2025-04-01"
    """
    sorted_months = sorted(list(months))

    if not sorted_months:
        raise ValueError("MONTHS cannot be empty.")

    start_month = datetime.strptime(sorted_months[0], "%Y-%m")
    last_month = datetime.strptime(sorted_months[-1], "%Y-%m")

    if last_month.month == 12:
        end_year = last_month.year + 1
        end_month = 1
    else:
        end_year = last_month.year
        end_month = last_month.month + 1

    period_start = f"{start_month.year:04d}-{start_month.month:02d}-01"
    period_end = f"{end_year:04d}-{end_month:02d}-01"

    return period_start, period_end


def make_month_label(months: Iterable[str]) -> str:
    sorted_months = sorted(list(months))

    if len(sorted_months) == 1:
        return sorted_months[0]

    return f"{sorted_months[0]}_to_{sorted_months[-1]}"


def get_raw_trip_files(months: Iterable[str], taxi_type: str) -> list[Path]:
    raw_files = []

    for month in months:
        file_path = RAW_DIR / f"{taxi_type}_tripdata_{month}.parquet"

        if not file_path.exists() or file_path.stat().st_size == 0:
            raise FileNotFoundError(f"Raw trip file not found or empty: {file_path}")

        raw_files.append(file_path)

    print("[INFO] Raw files found:")
    for file_path in raw_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"  - {file_path} ({size_mb:.2f} MB)")

    return raw_files


def make_duckdb_parquet_input(file_paths: list[Path]) -> str:
    """
    Format path agar bisa dibaca DuckDB.

    Kalau 1 file:
    read_parquet('/path/file.parquet')

    Kalau banyak file:
    read_parquet(['/path/a.parquet', '/path/b.parquet'])
    """
    paths = [str(path) for path in file_paths]

    if len(paths) == 1:
        return f"'{paths[0]}'"

    quoted_paths = ", ".join([f"'{path}'" for path in paths])
    return f"[{quoted_paths}]"


def clean_tlc_trip_data(months: Iterable[str], taxi_type: str) -> str:
    """
    Membersihkan NYC TLC trip data dan menyimpan hasilnya ke data/processed.

    Cleaning rules:
    - pickup/dropoff datetime tidak boleh NULL
    - passenger_count harus valid
    - trip_distance harus > 0 dan tidak ekstrem
    - durasi trip harus > 0 dan <= 360 menit
    - fare/tip/total amount tidak boleh negatif
    - total_amount harus > 0
    - PULocationID dan DOLocationID harus valid untuk zona TLC
    - data duplikat dihapus dengan SELECT DISTINCT
    - membuat kolom turunan untuk analisis waktu, tip, durasi, dan kecepatan
    """
    ensure_processed_directory()

    months = sorted(list(months))
    period_start, period_end = get_period_bounds(months)
    month_label = make_month_label(months)

    raw_files = get_raw_trip_files(months=months, taxi_type=taxi_type)
    parquet_input = make_duckdb_parquet_input(raw_files)

    output_path = PROCESSED_DIR / f"clean_{taxi_type}_tripdata_{month_label}.parquet"
    report_path = PROCESSED_DIR / f"cleaning_report_{month_label}.json"

    if output_path.exists():
        output_path.unlink()

    if report_path.exists():
        report_path.unlink()

    con = duckdb.connect()

    print("[STEP] Counting raw rows...")
    raw_rows = con.sql(
        f"""
        SELECT COUNT(*)
        FROM read_parquet({parquet_input})
        """
    ).fetchone()[0]

    print(f"[INFO] Raw rows: {raw_rows:,}")
    print(f"[INFO] Period start: {period_start}")
    print(f"[INFO] Period end exclusive: {period_end}")

    print("[STEP] Cleaning data...")

    clean_sql = f"""
    COPY (
        WITH base AS (
            SELECT
                *,
                date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime)
                    AS trip_duration_minutes
            FROM read_parquet({parquet_input})
        ),

        cleaned AS (
            SELECT DISTINCT
                VendorID,
                tpep_pickup_datetime,
                tpep_dropoff_datetime,
                passenger_count,
                trip_distance,
                RatecodeID,
                store_and_fwd_flag,
                PULocationID,
                DOLocationID,
                payment_type,
                fare_amount,
                extra,
                mta_tax,
                tip_amount,
                tolls_amount,
                improvement_surcharge,
                total_amount,
                congestion_surcharge,
                Airport_fee,

                CAST(tpep_pickup_datetime AS DATE) AS pickup_date,
                EXTRACT(hour FROM tpep_pickup_datetime) AS pickup_hour,
                EXTRACT(month FROM tpep_pickup_datetime) AS pickup_month,
                EXTRACT(dow FROM tpep_pickup_datetime) AS pickup_dayofweek,
                strftime(tpep_pickup_datetime, '%A') AS pickup_day_name,

                CASE
                    WHEN EXTRACT(dow FROM tpep_pickup_datetime) IN (0, 6)
                    THEN TRUE
                    ELSE FALSE
                END AS is_weekend,

                trip_duration_minutes,

                CASE
                    WHEN fare_amount > 0
                    THEN ROUND((tip_amount / fare_amount) * 100, 2)
                    ELSE 0
                END AS tip_percentage,

                CASE
                    WHEN trip_duration_minutes > 0
                    THEN ROUND(trip_distance / (trip_duration_minutes / 60.0), 2)
                    ELSE NULL
                END AS average_speed_mph

            FROM base
            WHERE
                tpep_pickup_datetime IS NOT NULL
                AND tpep_dropoff_datetime IS NOT NULL

                AND tpep_pickup_datetime >= TIMESTAMP '{period_start}'
                AND tpep_pickup_datetime < TIMESTAMP '{period_end}'

                AND passenger_count IS NOT NULL
                AND passenger_count > 0
                AND passenger_count <= 6

                AND trip_distance IS NOT NULL
                AND trip_distance > 0
                AND trip_distance <= 100

                AND trip_duration_minutes > 0
                AND trip_duration_minutes <= 360

                AND fare_amount IS NOT NULL
                AND fare_amount >= 0

                AND tip_amount IS NOT NULL
                AND tip_amount >= 0

                AND total_amount IS NOT NULL
                AND total_amount > 0

                AND PULocationID IS NOT NULL
                AND DOLocationID IS NOT NULL
                AND PULocationID BETWEEN 1 AND 263
                AND DOLocationID BETWEEN 1 AND 263
        )

        SELECT *
        FROM cleaned
    )
    TO '{output_path}'
    (FORMAT PARQUET);
    """

    con.execute(clean_sql)

    print("[STEP] Counting cleaned rows...")
    cleaned_rows = con.sql(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_path}')
        """
    ).fetchone()[0]

    removed_rows = raw_rows - cleaned_rows
    removed_percentage = round((removed_rows / raw_rows) * 100, 2) if raw_rows else 0

    print("[STEP] Creating summary statistics...")

    summary = con.sql(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT pickup_date) AS total_pickup_dates,
            MIN(tpep_pickup_datetime) AS min_pickup_datetime,
            MAX(tpep_pickup_datetime) AS max_pickup_datetime,
            ROUND(AVG(trip_distance), 2) AS avg_trip_distance,
            ROUND(AVG(trip_duration_minutes), 2) AS avg_trip_duration_minutes,
            ROUND(AVG(fare_amount), 2) AS avg_fare_amount,
            ROUND(AVG(tip_amount), 2) AS avg_tip_amount,
            ROUND(AVG(total_amount), 2) AS avg_total_amount
        FROM read_parquet('{output_path}')
        """
    ).fetchdf().to_dict(orient="records")[0]

    report = {
        "taxi_type": taxi_type,
        "months": months,
        "period_start": period_start,
        "period_end_exclusive": period_end,
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "removed_rows": removed_rows,
        "removed_percentage": removed_percentage,
        "output_file": str(output_path),
        "summary": summary,
        "cleaning_rules": [
            "Remove missing pickup/dropoff datetime",
            "Keep pickup datetime inside selected period",
            "Remove invalid passenger_count",
            "Remove trip_distance <= 0 or > 100",
            "Remove trip_duration_minutes <= 0 or > 360",
            "Remove negative fare_amount, tip_amount, or total_amount",
            "Remove invalid pickup/dropoff location ID",
            "Remove duplicate rows using SELECT DISTINCT",
            "Create derived columns: pickup_date, pickup_hour, pickup_month, pickup_dayofweek, pickup_day_name, is_weekend, trip_duration_minutes, tip_percentage, average_speed_mph",
        ],
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    print("[RESULT] Cleaning completed.")
    print(f"[RESULT] Raw rows          : {raw_rows:,}")
    print(f"[RESULT] Cleaned rows      : {cleaned_rows:,}")
    print(f"[RESULT] Removed rows      : {removed_rows:,}")
    print(f"[RESULT] Removed percentage: {removed_percentage}%")
    print(f"[SAVED] Clean data         : {output_path}")
    print(f"[SAVED] Cleaning report    : {report_path}")

    con.close()

    return str(output_path)


def validate_cleaning_outputs(months: Iterable[str], taxi_type: str) -> None:
    months = sorted(list(months))
    month_label = make_month_label(months)

    clean_file = PROCESSED_DIR / f"clean_{taxi_type}_tripdata_{month_label}.parquet"
    report_file = PROCESSED_DIR / f"cleaning_report_{month_label}.json"

    expected_files = [clean_file, report_file]
    missing_files = []

    for file_path in expected_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing cleaning output files: {missing_files}")

    con = duckdb.connect()

    cleaned_rows = con.sql(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{clean_file}')
        """
    ).fetchone()[0]

    null_check = con.sql(
        f"""
        SELECT
            SUM(CASE WHEN tpep_pickup_datetime IS NULL THEN 1 ELSE 0 END) AS null_pickup,
            SUM(CASE WHEN tpep_dropoff_datetime IS NULL THEN 1 ELSE 0 END) AS null_dropoff,
            SUM(CASE WHEN trip_distance <= 0 THEN 1 ELSE 0 END) AS invalid_distance,
            SUM(CASE WHEN trip_duration_minutes <= 0 THEN 1 ELSE 0 END) AS invalid_duration,
            SUM(CASE WHEN total_amount <= 0 THEN 1 ELSE 0 END) AS invalid_total
        FROM read_parquet('{clean_file}')
        """
    ).fetchone()

    con.close()

    if cleaned_rows == 0:
        raise ValueError("Cleaned data is empty.")

    if any(value not in (0, None) for value in null_check):
        raise ValueError(f"Cleaned data still contains invalid values: {null_check}")

    print("[CHECK] Cleaning output files are valid.")
    print(f"[CHECK] Cleaned rows: {cleaned_rows:,}")
    print(f"[CHECK] Validation checks: {null_check}")


if __name__ == "__main__":
    from pipeline_config import MONTHS, TAXI_TYPE

    clean_tlc_trip_data(months=MONTHS, taxi_type=TAXI_TYPE)
    validate_cleaning_outputs(months=MONTHS, taxi_type=TAXI_TYPE)
