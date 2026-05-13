import duckdb

path = "/opt/airflow/data/processed/clean_yellow_tripdata_2025-01_to_2025-03.parquet"

query = f"""
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN trip_distance <= 0 THEN 1 ELSE 0 END) AS invalid_distance,
    SUM(CASE WHEN trip_duration_minutes <= 0 THEN 1 ELSE 0 END) AS invalid_duration,
    SUM(CASE WHEN total_amount <= 0 THEN 1 ELSE 0 END) AS invalid_total_amount,
    MIN(tpep_pickup_datetime) AS min_pickup_datetime,
    MAX(tpep_pickup_datetime) AS max_pickup_datetime,
    COUNT(DISTINCT pickup_date) AS total_pickup_dates
FROM read_parquet('{path}');
"""

result = duckdb.sql(query).fetchdf()
print(result)