import sys
from datetime import datetime

from airflow.sdk import dag, task


sys.path.append("/opt/airflow")


@dag(
    dag_id="tlc_weather_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "tlc", "weather", "external"],
)
def tlc_weather_pipeline():
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

    weather = download_weather()
    weather_validation = validate_weather()
    join = build_weather_join()
    join_validation = validate_weather_join()

    weather >> weather_validation >> join >> join_validation


tlc_weather_pipeline()