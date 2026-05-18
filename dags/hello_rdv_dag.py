# /*
#  * 245150200111028 Aditya Akbar
#  * 245150207111038 Mohammad Geisar Rampan
#  * 245150207111103 Muhammad Sulthon Aulia Wijaya
#  * 245150207111050 Orie Abyan Maulana
#  * 235150201111068 Pieter Christy Yan Yudhistira
#  */
from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="hello_rdv_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["rdv", "test"],
)
def hello_rdv_pipeline():
    @task
    def hello():
        print("Airflow untuk project RDV sudah berhasil jalan.")

    hello()


hello_rdv_pipeline()
