import sys
from datetime import datetime

from airflow.sdk import dag, task


sys.path.append("/opt/airflow")


@dag(
    dag_id="tlc_full_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "tlc", "full-pipeline"],
)
def tlc_full_pipeline():
    @task
    def prepare_directories():
        from src.extract_tlc import ensure_directories
        from src.clean_tlc import ensure_processed_directory
        from src.build_curated import ensure_curated_directory
        from src.extract_weather import ensure_external_directory
        from src.train_demand_prediction import ensure_ml_directory

        ensure_directories()
        ensure_processed_directory()
        ensure_curated_directory()
        ensure_external_directory()
        ensure_ml_directory()

    @task
    def download_tlc_trip_data():
        from src.extract_tlc import download_tlc_data
        from src.pipeline_config import MONTHS, TAXI_TYPE

        return download_tlc_data(months=MONTHS, taxi_type=TAXI_TYPE)

    @task
    def download_zone_lookup():
        from src.extract_tlc import download_taxi_zone_lookup

        return download_taxi_zone_lookup()

    @task
    def validate_ingestion():
        from src.extract_tlc import validate_ingestion_outputs
        from src.pipeline_config import MONTHS, TAXI_TYPE

        validate_ingestion_outputs(months=MONTHS, taxi_type=TAXI_TYPE)

    @task
    def clean_trip_data():
        from src.clean_tlc import clean_tlc_trip_data
        from src.pipeline_config import MONTHS, TAXI_TYPE

        return clean_tlc_trip_data(months=MONTHS, taxi_type=TAXI_TYPE)

    @task
    def validate_cleaning():
        from src.clean_tlc import validate_cleaning_outputs
        from src.pipeline_config import MONTHS, TAXI_TYPE

        validate_cleaning_outputs(months=MONTHS, taxi_type=TAXI_TYPE)

    @task
    def build_curated_tables():
        from src.build_curated import build_curated_tables
        from src.pipeline_config import MONTHS, TAXI_TYPE

        return build_curated_tables(months=MONTHS, taxi_type=TAXI_TYPE)

    @task
    def validate_curated():
        from src.build_curated import validate_curated_outputs
        from src.pipeline_config import MONTHS, TAXI_TYPE

        validate_curated_outputs(months=MONTHS, taxi_type=TAXI_TYPE)

    @task
    def download_weather():
        from src.extract_weather import download_weather_data
        from src.pipeline_config import MONTHS

        return download_weather_data(months=MONTHS)

    @task
    def validate_weather():
        from src.extract_weather import validate_weather_outputs
        from src.pipeline_config import MONTHS

        validate_weather_outputs(months=MONTHS)

    @task
    def build_weather_join():
        from src.build_weather_curated import build_hourly_demand_weather
        from src.pipeline_config import MONTHS

        return build_hourly_demand_weather(months=MONTHS)

    @task
    def validate_weather_join():
        from src.build_weather_curated import validate_weather_join_outputs
        from src.pipeline_config import MONTHS

        validate_weather_join_outputs(months=MONTHS)

    @task
    def build_analysis_marts():
        from src.build_analysis_marts import build_analysis_marts
        from src.pipeline_config import MONTHS

        return build_analysis_marts(months=MONTHS)

    @task
    def validate_analysis_marts():
        from src.build_analysis_marts import validate_analysis_marts
        from src.pipeline_config import MONTHS

        validate_analysis_marts(months=MONTHS)

    @task
    def train_demand_prediction():
        from src.train_demand_prediction import train_demand_prediction_model
        from src.pipeline_config import MONTHS

        return train_demand_prediction_model(months=MONTHS)


    @task
    def validate_demand_prediction():
        from src.train_demand_prediction import validate_demand_prediction_outputs
        from src.pipeline_config import MONTHS

        validate_demand_prediction_outputs(months=MONTHS)

    start = prepare_directories()

    tlc = download_tlc_trip_data()
    zones = download_zone_lookup()
    ingestion_check = validate_ingestion()

    clean = clean_trip_data()
    clean_check = validate_cleaning()

    curated = build_curated_tables()
    curated_check = validate_curated()

    weather = download_weather()
    weather_check = validate_weather()

    weather_join = build_weather_join()
    weather_join_check = validate_weather_join()

    analysis_marts = build_analysis_marts()
    analysis_marts_check = validate_analysis_marts()

    demand_prediction = train_demand_prediction()
    demand_prediction_check = validate_demand_prediction()

    start >> [tlc, zones, weather]

    [tlc, zones] >> ingestion_check
    ingestion_check >> clean >> clean_check
    [clean_check, zones] >> curated >> curated_check

    weather >> weather_check
    [curated_check, weather_check] >> weather_join >> weather_join_check
    weather_join_check >> analysis_marts >> analysis_marts_check
    analysis_marts_check >> demand_prediction >> demand_prediction_check


tlc_full_pipeline()