import sys
from datetime import datetime

from airflow.sdk import dag, task


sys.path.append("/opt/airflow")


@dag(
    dag_id="tlc_cleaning_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "tlc", "cleaning"],
)
def tlc_cleaning_pipeline():
    @task
    def clean_trip_data():
        from src.clean_tlc import clean_tlc_trip_data
        from src.pipeline_config import MONTHS, TAXI_TYPE

        return clean_tlc_trip_data(months=MONTHS, taxi_type=TAXI_TYPE)

    @task
    def validate_outputs():
        from src.clean_tlc import validate_cleaning_outputs
        from src.pipeline_config import MONTHS, TAXI_TYPE

        validate_cleaning_outputs(months=MONTHS, taxi_type=TAXI_TYPE)

    clean = clean_trip_data()
    validate = validate_outputs()

    clean >> validate


tlc_cleaning_pipeline()