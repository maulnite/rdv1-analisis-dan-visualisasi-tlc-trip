from pathlib import Path
from typing import Iterable
import json

import duckdb

from src.clean_tlc import make_month_label


DATA_DIR = Path("/opt/airflow/data")
EXTERNAL_DIR = DATA_DIR / "external"
CURATED_DIR = DATA_DIR / "curated"


def weather_condition_case(prefix: str = "") -> str:
    """
    Membuat kategori cuaca berdasarkan precipitation/rain/snowfall.
    Threshold:
    - clear         : precipitation/rain = 0
    - light_rain    : 0 < rain/precipitation < 2.5 mm/h
    - moderate_rain : 2.5 <= rain/precipitation <= 10 mm/h
    - heavy_rain    : rain/precipitation > 10 mm/h
    - snow          : snowfall > 0
    """
    p = f"{prefix}." if prefix else ""

    return f"""
        CASE
            WHEN COALESCE({p}snowfall, 0) > 0 THEN 'snow'
            WHEN GREATEST(COALESCE({p}rain, 0), COALESCE({p}precipitation, 0)) > 10 THEN 'heavy_rain'
            WHEN GREATEST(COALESCE({p}rain, 0), COALESCE({p}precipitation, 0)) >= 2.5 THEN 'moderate_rain'
            WHEN GREATEST(COALESCE({p}rain, 0), COALESCE({p}precipitation, 0)) > 0 THEN 'light_rain'
            ELSE 'clear'
        END
    """


def get_required_file(path: Path, label: str) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {path}")
    return path


def build_analysis_marts(months: Iterable[str]) -> dict:
    """
    Membuat analysis-ready mart tables sesuai arahan PM:
    - daily_zone_metrics
    - hourly_demand_summary_weather
    - weather_impact_summary
    - zone_weather_elasticity
    - od_flow_weather
    """
    months = sorted(list(months))
    month_label = make_month_label(months)

    hourly_weather_file = get_required_file(
        CURATED_DIR / f"agg_hourly_demand_weather_{month_label}.parquet",
        "Hourly demand weather file",
    )

    fact_trips_file = get_required_file(
        CURATED_DIR / f"fact_trips_{month_label}.parquet",
        "Fact trips file",
    )

    weather_file = get_required_file(
        EXTERNAL_DIR / f"weather_hourly_{month_label}.parquet",
        "Weather hourly file",
    )

    daily_zone_path = CURATED_DIR / f"daily_zone_metrics_{month_label}.parquet"
    hourly_summary_path = CURATED_DIR / f"hourly_demand_summary_weather_{month_label}.parquet"
    weather_impact_path = CURATED_DIR / f"weather_impact_summary_{month_label}.parquet"
    zone_elasticity_path = CURATED_DIR / f"zone_weather_elasticity_{month_label}.parquet"
    od_flow_path = CURATED_DIR / f"od_flow_weather_{month_label}.parquet"
    report_path = CURATED_DIR / f"analysis_mart_report_{month_label}.json"

    output_paths = [
        daily_zone_path,
        hourly_summary_path,
        weather_impact_path,
        zone_elasticity_path,
        od_flow_path,
        report_path,
    ]

    for path in output_paths:
        if path.exists():
            path.unlink()
            print(f"[INFO] Removed old output: {path}")

    con = duckdb.connect()

    condition_sql = weather_condition_case()

    print("[STEP] Creating hourly_weather_enriched temp view...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW hourly_weather_enriched AS
        SELECT
            *,
            {condition_sql} AS weather_condition,

            CASE
                WHEN avg_trip_distance > 0
                THEN avg_trip_duration_minutes / avg_trip_distance
                ELSE NULL
            END AS avg_minutes_per_mile
        FROM read_parquet('{hourly_weather_file}');
        """
    )

    print("[STEP] Building daily_zone_metrics...")

    con.execute(
        f"""
        COPY (
            SELECT
                pickup_date,
                pickup_location_id AS zone_id,
                pickup_borough,
                pickup_zone,
                weather_condition,

                COUNT(*) AS active_hours,
                SUM(total_trips) AS trip_count,
                SUM(total_passengers) AS total_passengers,

                ROUND(SUM(total_revenue), 2) AS total_revenue,
                ROUND(SUM(total_tip), 2) AS total_tip,

                ROUND(SUM(total_revenue) / NULLIF(SUM(total_trips), 0), 2) AS avg_fare,
                ROUND(SUM(avg_total_amount * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_total_amount,
                ROUND(SUM(avg_trip_duration_minutes * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_duration,
                ROUND(SUM(avg_trip_distance * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_trip_distance,
                ROUND(SUM(avg_tip_percentage * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_tip_pct,
                ROUND(SUM(avg_minutes_per_mile * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_minutes_per_mile,

                ROUND(AVG(temperature_2m), 2) AS avg_temperature_2m,
                ROUND(AVG(precipitation), 2) AS avg_precipitation,
                ROUND(AVG(rain), 2) AS avg_rain,
                ROUND(AVG(snowfall), 2) AS avg_snowfall,
                ROUND(AVG(wind_speed_10m), 2) AS avg_wind_speed_10m

            FROM hourly_weather_enriched
            GROUP BY
                pickup_date,
                pickup_location_id,
                pickup_borough,
                pickup_zone,
                weather_condition
            ORDER BY
                pickup_date,
                pickup_borough,
                pickup_zone,
                weather_condition
        )
        TO '{daily_zone_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Building hourly_demand_summary_weather...")

    con.execute(
        f"""
        COPY (
            SELECT
                pickup_hour,
                pickup_borough,
                weather_condition,

                COUNT(*) AS total_zone_hours,
                SUM(total_trips) AS total_trips,
                SUM(total_passengers) AS total_passengers,

                ROUND(SUM(total_revenue), 2) AS total_revenue,
                ROUND(SUM(total_tip), 2) AS total_tip,

                ROUND(SUM(total_trips) / NULLIF(COUNT(*), 0), 2) AS avg_trips_per_zone_hour,
                ROUND(SUM(avg_trip_duration_minutes * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_duration,
                ROUND(SUM(avg_trip_distance * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_trip_distance,
                ROUND(SUM(avg_total_amount * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_total_amount,
                ROUND(SUM(avg_tip_percentage * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_tip_pct,
                ROUND(SUM(avg_minutes_per_mile * total_trips) / NULLIF(SUM(total_trips), 0), 2) AS avg_minutes_per_mile,

                ROUND(AVG(temperature_2m), 2) AS avg_temperature_2m,
                ROUND(AVG(precipitation), 2) AS avg_precipitation,
                ROUND(AVG(rain), 2) AS avg_rain,
                ROUND(AVG(wind_speed_10m), 2) AS avg_wind_speed_10m

            FROM hourly_weather_enriched
            GROUP BY
                pickup_hour,
                pickup_borough,
                weather_condition
            ORDER BY
                pickup_hour,
                pickup_borough,
                weather_condition
        )
        TO '{hourly_summary_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Building weather_impact_summary...")

    con.execute(
        f"""
        COPY (
            WITH by_condition AS (
                SELECT
                    weather_condition,

                    COUNT(*) AS total_zone_hours,
                    SUM(total_trips) AS total_trips,

                    ROUND(SUM(total_trips) / NULLIF(COUNT(*), 0), 4) AS avg_trips_per_zone_hour,
                    ROUND(SUM(avg_trip_duration_minutes * total_trips) / NULLIF(SUM(total_trips), 0), 4) AS avg_duration,
                    ROUND(SUM(avg_total_amount * total_trips) / NULLIF(SUM(total_trips), 0), 4) AS avg_total_amount,
                    ROUND(SUM(avg_tip_percentage * total_trips) / NULLIF(SUM(total_trips), 0), 4) AS avg_tip_pct,
                    ROUND(SUM(avg_minutes_per_mile * total_trips) / NULLIF(SUM(total_trips), 0), 4) AS avg_minutes_per_mile,

                    ROUND(AVG(precipitation), 4) AS avg_precipitation,
                    ROUND(AVG(rain), 4) AS avg_rain

                FROM hourly_weather_enriched
                GROUP BY weather_condition
            ),

            baseline AS (
                SELECT
                    avg_trips_per_zone_hour AS clear_avg_trips_per_zone_hour,
                    avg_duration AS clear_avg_duration,
                    avg_total_amount AS clear_avg_total_amount,
                    avg_tip_pct AS clear_avg_tip_pct,
                    avg_minutes_per_mile AS clear_avg_minutes_per_mile
                FROM by_condition
                WHERE weather_condition = 'clear'
            )

            SELECT
                c.weather_condition,
                c.total_zone_hours,
                c.total_trips,

                c.avg_trips_per_zone_hour,
                ROUND(
                    ((c.avg_trips_per_zone_hour - b.clear_avg_trips_per_zone_hour)
                    / NULLIF(b.clear_avg_trips_per_zone_hour, 0)) * 100,
                    2
                ) AS demand_lift_pct,

                c.avg_duration,
                ROUND(c.avg_duration - b.clear_avg_duration, 2) AS duration_delta_minutes,

                c.avg_minutes_per_mile,
                ROUND(c.avg_minutes_per_mile - b.clear_avg_minutes_per_mile, 2) AS minutes_per_mile_delta,

                c.avg_total_amount,
                ROUND(c.avg_total_amount - b.clear_avg_total_amount, 2) AS fare_delta_amount,

                c.avg_tip_pct,
                ROUND(c.avg_tip_pct - b.clear_avg_tip_pct, 2) AS tip_delta_pct,

                c.avg_precipitation,
                c.avg_rain

            FROM by_condition c
            CROSS JOIN baseline b
            ORDER BY
                CASE c.weather_condition
                    WHEN 'clear' THEN 1
                    WHEN 'light_rain' THEN 2
                    WHEN 'moderate_rain' THEN 3
                    WHEN 'heavy_rain' THEN 4
                    WHEN 'snow' THEN 5
                    ELSE 99
                END
        )
        TO '{weather_impact_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Building zone_weather_elasticity...")

    con.execute(
        f"""
        COPY (
            WITH zone_condition AS (
                SELECT
                    pickup_location_id AS zone_id,
                    pickup_borough,
                    pickup_zone,
                    weather_condition,

                    COUNT(*) AS total_zone_hours,
                    SUM(total_trips) AS total_trips,

                    ROUND(SUM(total_trips) / NULLIF(COUNT(*), 0), 4) AS avg_trips_per_zone_hour,
                    ROUND(SUM(avg_trip_duration_minutes * total_trips) / NULLIF(SUM(total_trips), 0), 4) AS avg_duration,
                    ROUND(SUM(avg_total_amount * total_trips) / NULLIF(SUM(total_trips), 0), 4) AS avg_total_amount,
                    ROUND(SUM(avg_tip_percentage * total_trips) / NULLIF(SUM(total_trips), 0), 4) AS avg_tip_pct,
                    ROUND(SUM(avg_minutes_per_mile * total_trips) / NULLIF(SUM(total_trips), 0), 4) AS avg_minutes_per_mile,
                    ROUND(AVG(precipitation), 4) AS avg_precipitation

                FROM hourly_weather_enriched
                GROUP BY
                    pickup_location_id,
                    pickup_borough,
                    pickup_zone,
                    weather_condition
            ),

            baseline AS (
                SELECT
                    zone_id,
                    avg_trips_per_zone_hour AS clear_avg_trips_per_zone_hour,
                    avg_duration AS clear_avg_duration,
                    avg_total_amount AS clear_avg_total_amount,
                    avg_tip_pct AS clear_avg_tip_pct,
                    avg_minutes_per_mile AS clear_avg_minutes_per_mile
                FROM zone_condition
                WHERE weather_condition = 'clear'
            ),

            metrics AS (
                SELECT
                    z.zone_id,
                    z.pickup_borough,
                    z.pickup_zone,
                    z.weather_condition,
                    z.total_zone_hours,
                    z.total_trips,

                    z.avg_trips_per_zone_hour,
                    ROUND(
                        ((z.avg_trips_per_zone_hour - b.clear_avg_trips_per_zone_hour)
                        / NULLIF(b.clear_avg_trips_per_zone_hour, 0)) * 100,
                        2
                    ) AS demand_lift_pct,

                    z.avg_duration,
                    ROUND(z.avg_duration - b.clear_avg_duration, 2) AS duration_delta_minutes,

                    z.avg_minutes_per_mile,
                    ROUND(z.avg_minutes_per_mile - b.clear_avg_minutes_per_mile, 2) AS minutes_per_mile_delta,

                    z.avg_total_amount,
                    ROUND(z.avg_total_amount - b.clear_avg_total_amount, 2) AS fare_delta_amount,

                    z.avg_tip_pct,
                    ROUND(z.avg_tip_pct - b.clear_avg_tip_pct, 2) AS tip_delta_pct,

                    z.avg_precipitation

                FROM zone_condition z
                LEFT JOIN baseline b
                    ON z.zone_id = b.zone_id
            )

            SELECT
                *,
                CASE
                    WHEN weather_condition = 'clear' THEN 'baseline'
                    WHEN demand_lift_pct >= 25 OR duration_delta_minutes >= 5 THEN 'high_weather_sensitive'
                    WHEN demand_lift_pct >= 10 OR duration_delta_minutes >= 2 THEN 'medium_weather_sensitive'
                    ELSE 'low_weather_sensitive'
                END AS weather_sensitivity_label
            FROM metrics
            ORDER BY
                demand_lift_pct DESC NULLS LAST,
                duration_delta_minutes DESC NULLS LAST
        )
        TO '{zone_elasticity_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Creating weather_enriched temp view for OD flow...")

    weather_condition_sql_w = weather_condition_case("w")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW weather_enriched AS
        SELECT
            CAST(w.pickup_date AS DATE) AS pickup_date,
            w.pickup_hour,
            {weather_condition_sql_w} AS weather_condition
        FROM read_parquet('{weather_file}') w;
        """
    )

    print("[STEP] Building od_flow_weather...")

    con.execute(
        f"""
        COPY (
            SELECT
                f.pickup_location_id AS origin_location_id,
                f.pickup_borough AS origin_borough,
                f.pickup_zone AS origin_zone,

                f.dropoff_location_id AS destination_location_id,
                f.dropoff_borough AS destination_borough,
                f.dropoff_zone AS destination_zone,

                w.weather_condition,

                COUNT(*) AS trip_count,
                ROUND(AVG(f.trip_duration_minutes), 2) AS avg_duration,
                ROUND(AVG(f.trip_distance), 2) AS avg_trip_distance,
                ROUND(AVG(f.average_speed_mph), 2) AS avg_speed_mph,
                ROUND(SUM(f.total_amount), 2) AS total_revenue,
                ROUND(AVG(f.total_amount), 2) AS avg_total_amount,
                ROUND(AVG(f.tip_percentage), 2) AS avg_tip_pct

            FROM read_parquet('{fact_trips_file}') f
            LEFT JOIN weather_enriched w
                ON f.pickup_date = w.pickup_date
                AND f.pickup_hour = w.pickup_hour

            GROUP BY
                f.pickup_location_id,
                f.pickup_borough,
                f.pickup_zone,
                f.dropoff_location_id,
                f.dropoff_borough,
                f.dropoff_zone,
                w.weather_condition

            HAVING COUNT(*) >= 50

            ORDER BY
                trip_count DESC
        )
        TO '{od_flow_path}'
        (FORMAT PARQUET);
        """
    )

    print("[STEP] Creating analysis mart report...")

    outputs = {
        "daily_zone_metrics": daily_zone_path,
        "hourly_demand_summary_weather": hourly_summary_path,
        "weather_impact_summary": weather_impact_path,
        "zone_weather_elasticity": zone_elasticity_path,
        "od_flow_weather": od_flow_path,
    }

    row_counts = {}

    for name, path in outputs.items():
        row_counts[name] = con.sql(
            f"SELECT COUNT(*) FROM read_parquet('{path}')"
        ).fetchone()[0]

    top_weather_impact = con.sql(
        f"""
        SELECT *
        FROM read_parquet('{weather_impact_path}')
        ORDER BY demand_lift_pct DESC NULLS LAST
        """
    ).fetchdf().to_dict(orient="records")

    top_sensitive_zones = con.sql(
        f"""
        SELECT
            zone_id,
            pickup_borough,
            pickup_zone,
            weather_condition,
            demand_lift_pct,
            duration_delta_minutes,
            weather_sensitivity_label
        FROM read_parquet('{zone_elasticity_path}')
        WHERE weather_condition <> 'clear'
        ORDER BY demand_lift_pct DESC NULLS LAST
        LIMIT 10
        """
    ).fetchdf().to_dict(orient="records")

    report = {
        "months": months,
        "inputs": {
            "hourly_demand_weather": str(hourly_weather_file),
            "fact_trips": str(fact_trips_file),
            "weather_hourly": str(weather_file),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
        "row_counts": row_counts,
        "weather_impact_summary": top_weather_impact,
        "top_10_weather_sensitive_zones": top_sensitive_zones,
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    con.close()

    print("[RESULT] Analysis marts completed.")
    for name, count in row_counts.items():
        print(f"[RESULT] {name}: {count:,} rows")
    print(f"[SAVED] Report: {report_path}")

    return report


def validate_analysis_marts(months: Iterable[str]) -> None:
    months = sorted(list(months))
    month_label = make_month_label(months)

    expected_files = [
        CURATED_DIR / f"daily_zone_metrics_{month_label}.parquet",
        CURATED_DIR / f"hourly_demand_summary_weather_{month_label}.parquet",
        CURATED_DIR / f"weather_impact_summary_{month_label}.parquet",
        CURATED_DIR / f"zone_weather_elasticity_{month_label}.parquet",
        CURATED_DIR / f"od_flow_weather_{month_label}.parquet",
        CURATED_DIR / f"analysis_mart_report_{month_label}.json",
    ]

    missing_files = []

    for file_path in expected_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing analysis mart output files: {missing_files}")

    con = duckdb.connect()

    for file_path in expected_files:
        if file_path.suffix == ".parquet":
            rows = con.sql(f"SELECT COUNT(*) FROM read_parquet('{file_path}')").fetchone()[0]
            if rows == 0:
                raise ValueError(f"Analysis mart is empty: {file_path}")
            print(f"[CHECK] {file_path.name}: {rows:,} rows")

    con.close()

    print("[CHECK] Analysis mart outputs are valid.")


if __name__ == "__main__":
    from src.pipeline_config import MONTHS

    build_analysis_marts(months=MONTHS)
    validate_analysis_marts(months=MONTHS)