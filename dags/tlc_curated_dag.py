import sys
from datetime import datetime

from airflow.sdk import dag, task


sys.path.append("/opt/airflow")


@dag(
    dag_id="tlc_curated_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "tlc", "curated"],
)
def tlc_curated_pipeline():
    @task
    def build_tables():
        from src.build_curated import build_curated_tables
        from src.pipeline_config import MONTHS, TAXI_TYPE

        return build_curated_tables(months=MONTHS, taxi_type=TAXI_TYPE)

    @task
    def validate_outputs():
        from src.build_curated import validate_curated_outputs
        from src.pipeline_config import MONTHS, TAXI_TYPE

        validate_curated_outputs(months=MONTHS, taxi_type=TAXI_TYPE)

    build = build_tables()
    validate = validate_outputs()

    build >> validate


tlc_curated_pipeline()