# /*
#  * 245150200111028 Aditya Akbar
#  * 245150207111038 Mohammad Geisar Rampan
#  * 245150207111103 Muhammad Sulthon Aulia Wijaya
#  * 245150207111050 Orie Abyan Maulana
#  * 235150201111068 Pieter Christy Yan Yudhistira
#  */
import duckdb

base = "/opt/airflow/data/curated"
period = "2025-01_to_2025-03"

files = {
    "fact_trips": f"{base}/fact_trips_{period}.parquet",
    "dim_location": f"{base}/dim_location_{period}.parquet",
    "agg_hourly_demand": f"{base}/agg_hourly_demand_{period}.parquet",
    "agg_zone_summary": f"{base}/agg_zone_summary_{period}.parquet",
    "agg_daily_summary": f"{base}/agg_daily_summary_{period}.parquet",
}

for name, path in files.items():
    rows = duckdb.sql(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
    print(f"{name}: {rows:,} rows")

print("\nTop 10 pickup zones:")
print(
    duckdb.sql(
        f"""
        SELECT 
            pickup_borough,
            pickup_zone,
            total_trips,
            total_revenue
        FROM read_parquet('{files["agg_zone_summary"]}')
        ORDER BY total_trips DESC
        LIMIT 10
        """
    ).fetchdf()
)

print("\nDaily summary date range:")
print(
    duckdb.sql(
        f"""
        SELECT 
            MIN(pickup_date) AS min_date,
            MAX(pickup_date) AS max_date,
            COUNT(*) AS total_days,
            SUM(total_trips) AS total_trips
        FROM read_parquet('{files["agg_daily_summary"]}')
        """
    ).fetchdf()
)
