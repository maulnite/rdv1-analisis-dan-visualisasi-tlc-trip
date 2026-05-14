import sys
from datetime import datetime

from airflow.sdk import dag, task


sys.path.append("/opt/airflow")


@dag(
    dag_id="tlc_ml_demand_prediction_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "tlc", "ml", "demand-prediction"],
)
def tlc_ml_demand_prediction_pipeline():
    @task
    def train_demand_prediction():
        from src.pipeline_config import MONTHS
        from src.train_demand_prediction import train_demand_prediction_model

        return train_demand_prediction_model(months=MONTHS)

    @task
    def validate_demand_prediction():
        from src.pipeline_config import MONTHS
        from src.train_demand_prediction import validate_demand_prediction_outputs

        validate_demand_prediction_outputs(months=MONTHS)

    train = train_demand_prediction()
    validate = validate_demand_prediction()

    train >> validate


tlc_ml_demand_prediction_pipeline()