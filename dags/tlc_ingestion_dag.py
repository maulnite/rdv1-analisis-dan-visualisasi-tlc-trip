import sys
from datetime import datetime

from airflow.sdk import dag, task


sys.path.append("/opt/airflow")


@dag(
    dag_id="tlc_ingestion_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "tlc", "ingestion"],
)
def tlc_ingestion_pipeline():
    @task
    def prepare_directories():
        from src.extract_tlc import ensure_directories

        ensure_directories()

    @task
    def download_tlc_trip_data():
        from src.extract_tlc import download_tlc_data

        return download_tlc_data(months=["2025-01"], taxi_type="yellow")

    @task
    def download_zone_lookup():
        from src.extract_tlc import download_taxi_zone_lookup

        return download_taxi_zone_lookup()

    @task
    def validate_outputs():
        from src.extract_tlc import validate_ingestion_outputs

        validate_ingestion_outputs()

    directories = prepare_directories()
    tlc_data = download_tlc_trip_data()
    zone_lookup = download_zone_lookup()
    validation = validate_outputs()

    directories >> [tlc_data, zone_lookup] >> validation


tlc_ingestion_pipeline()