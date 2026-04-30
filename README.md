\# RDV TLC Trip Data Pipeline



Project tugas akhir Rekayasa Data dan Visualisasi untuk membangun data pipeline menggunakan NYC TLC Trip Record Dataset.



\## Struktur Folder



\- `dags/` : DAG Apache Airflow

\- `src/` : Script Python untuk ingestion, cleaning, transformasi, dan agregasi

\- `data/raw/` : Data mentah hasil download

\- `data/external/` : Data eksternal seperti cuaca atau zone lookup

\- `data/processed/` : Data hasil cleaning

\- `data/curated/` : Data final/agregat untuk dashboard

\- `dashboard/` : Aplikasi dashboard Streamlit

\- `notebooks/` : Eksplorasi data awal

\- `reports/` : Screenshot dan bahan laporan

\- `logs/` : Log Airflow



\## Rencana Pipeline



1\. Download NYC TLC Trip Data

2\. Cleaning data

3\. Transformasi data

4\. Simpan intermediate/final data

5\. Buat agregasi untuk dashboard

6\. Visualisasi menggunakan Streamlit

