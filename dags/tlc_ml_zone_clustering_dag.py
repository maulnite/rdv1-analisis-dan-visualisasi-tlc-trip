import sys
from datetime import datetime

from airflow.sdk import dag, task


sys.path.append("/opt/airflow")


@dag(
    dag_id="tlc_ml_zone_clustering_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "tlc", "ml", "zone-clustering"],
)
def tlc_ml_zone_clustering_pipeline():
    @task
    def train_zone_clustering():
        from src.pipeline_config import MONTHS
        from src.train_zone_clustering import train_zone_weather_clustering

        return train_zone_weather_clustering(months=MONTHS)

    @task
    def validate_zone_clustering():
        from src.pipeline_config import MONTHS
        from src.train_zone_clustering import validate_zone_weather_clustering_outputs

        validate_zone_weather_clustering_outputs(months=MONTHS)

    train = train_zone_clustering()
    validate = validate_zone_clustering()

    train >> validate


tlc_ml_zone_clustering_pipeline()