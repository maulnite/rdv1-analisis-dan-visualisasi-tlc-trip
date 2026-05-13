# RDV TLC Project — Data Engineering Pipeline

## Overview

Project ini membangun pipeline Data Engineering end-to-end menggunakan dataset NYC TLC Yellow Taxi Trip Data periode Januari–Maret 2025.

Pipeline dibuat menggunakan:

- Docker
- Apache Airflow
- DuckDB
- Python
- Parquet format

Pipeline mencakup:

- Data ingestion
- Data cleaning & validation
- Curated analytical tables
- Integrasi external weather data
- End-to-end orchestration menggunakan Airflow DAG

---

# Project Structure

```bash
rdv-tlc-project/
│
├── dags/
│   ├── tlc_ingestion_pipeline.py
│   ├── tlc_cleaning_pipeline.py
│   ├── tlc_curated_pipeline.py
│   ├── tlc_weather_dag.py
│   └── tlc_full_pipeline_dag.py
│
├── src/
│   ├── extract_tlc_data.py
│   ├── clean_trip_data.py
│   ├── build_curated_tables.py
│   ├── extract_weather.py
│   ├── build_weather_curated.py
│   ├── quick_audit_clean.py
│   └── quick_audit_curated.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── curated/
│   └── external/
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# Dataset

## Main Dataset

NYC TLC Yellow Taxi Trip Data:

- January 2025
- February 2025
- March 2025

Format:

```text
Parquet
```

Source:

[https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

---

# External Data

## Taxi Zone Lookup

Digunakan untuk mapping:

- LocationID
- Borough
- Zone

## Weather Data

Sumber:

```text
Open-Meteo Historical Weather API
```

Digunakan untuk:

- Analisis hubungan cuaca terhadap demand taxi
- Join dengan hourly trip demand

---

# Data Pipeline Flow

```text
Raw TLC Data
    ↓
Validation Ingestion
    ↓
Cleaning & Preprocessing
    ↓
Processed Clean Data
    ↓
Curated Tables
    ↓
Weather Integration
    ↓
Final Analytical Tables
```

---

# Airflow DAGs

## 1. Ingestion Pipeline

DAG:

```text
tlc_ingestion_pipeline
```

Tasks:

- prepare_directories
- download_tlc_trip_data
- download_zone_lookup
- validate_ingestion

Output:

```text
/data/raw/
/data/external/taxi_zone_lookup.csv
```

---

## 2. Cleaning Pipeline

DAG:

```text
tlc_cleaning_pipeline
```

Tasks:

- clean_trip_data
- validate_outputs

Cleaning rules:

- Remove missing pickup/dropoff datetime
- Remove invalid passenger count
- Remove invalid trip distance
- Remove invalid fare & total amount
- Keep rows inside selected period

Output:

```text
/data/processed/clean_yellow_tripdata_2025-01_to_2025-03.parquet
```

---

## 3. Curated Pipeline

DAG:

```text
tlc_curated_pipeline
```

Generated tables:

### fact_trips

Fact table utama seluruh perjalanan taxi.

### dim_location

Dimensi lokasi berdasarkan taxi zone lookup.

### agg_daily_summary

Ringkasan harian:

- total trips
- total revenue
- avg fare
- avg distance

### agg_hourly_demand

Demand taxi per jam.

### agg_zone_summary

Ringkasan aktivitas tiap zona.

---

## 4. Weather Pipeline

DAG:

```text
tlc_weather_pipeline
```

Tasks:

- download_weather
- validate_weather
- build_weather_join
- validate_weather_join

Output:

```text
/data/external/weather_hourly_2025-01_to_2025-03.parquet
/data/curated/agg_hourly_demand_weather_2025-01_to_2025-03.parquet
```

---

## 5. Full End-to-End Pipeline

DAG:

```text
tlc_full_pipeline
```

Pipeline utama yang menjalankan seluruh proses:

```text
Ingestion
→ Cleaning
→ Curated Tables
→ Weather Integration
→ Validation
```

---

# Output Files

## Processed Layer

```text
clean_yellow_tripdata_2025-01_to_2025-03.parquet
cleaning_report_2025-01_to_2025-03.json
```

## Curated Layer

```text
fact_trips_2025-01_to_2025-03.parquet
dim_location_2025-01_to_2025-03.parquet
agg_daily_summary_2025-01_to_2025-03.parquet
agg_hourly_demand_2025-01_to_2025-03.parquet
agg_zone_summary_2025-01_to_2025-03.parquet
agg_hourly_demand_weather_2025-01_to_2025-03.parquet
```

---

# Validation & Audit

Pipeline memiliki validation step untuk memastikan:

- File berhasil dibuat
- Row count valid
- Tidak ada nilai penting yang kosong
- Range tanggal sesuai
- Join weather berhasil

Audit script:

```text
quick_audit_clean.py
quick_audit_curated.py
```

---

# How to Run

## 1. Start Docker

```bash
docker compose up -d
```

---

## 2. Open Airflow

```text
http://localhost:8080
```

Default login:

```text
Username: airflow
Password: airflow
```

---

## 3. Trigger DAG

Run DAG berikut:

```text
tlc_full_pipeline
```

---

# Technologies Used

- Python
- Apache Airflow
- Docker
- DuckDB
- Pandas
- PyArrow
- Requests

---

# Final Result

Pipeline berhasil menghasilkan:

- Cleaned TLC trip dataset
- Curated analytical tables
- Weather-integrated demand analysis dataset
- Fully orchestrated Airflow workflow

Project ini dapat digunakan untuk:

- Dashboard analytics
- Demand analysis
- Weather impact analysis
- Data visualization
- Machine learning preparation
