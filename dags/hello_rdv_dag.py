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