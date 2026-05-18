# /*
#  * 245150200111028 Aditya Akbar
#  * 245150207111038 Mohammad Geisar Rampan
#  * 245150207111103 Muhammad Sulthon Aulia Wijaya
#  * 245150207111050 Orie Abyan Maulana
#  * 235150201111068 Pieter Christy Yan Yudhistira
#  */
import sys
from datetime import datetime

from airflow.sdk import dag, task


sys.path.append("/opt/airflow")


@dag(
    dag_id="tlc_analysis_marts_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "tlc", "analysis-mart", "weather"],
)
def tlc_analysis_marts_pipeline():
    @task
    def build_analysis_marts_task():
        from src.build_analysis_marts import build_analysis_marts
        from src.pipeline_config import MONTHS

        return build_analysis_marts(months=MONTHS)

    @task
    def validate_analysis_marts_task():
        from src.build_analysis_marts import validate_analysis_marts
        from src.pipeline_config import MONTHS

        validate_analysis_marts(months=MONTHS)

    build = build_analysis_marts_task()
    validate = validate_analysis_marts_task()

    build >> validate


tlc_analysis_marts_pipeline()
