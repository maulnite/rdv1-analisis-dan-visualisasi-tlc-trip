# NYC TLC Weather Analytics & ML Pipeline

Project data engineering end-to-end untuk menganalisis pengaruh cuaca terhadap demand taksi, durasi perjalanan, fare, tip, pola origin-destination, serta sensitivitas zona pada data NYC Yellow Taxi.

Project ini menggunakan Apache Airflow untuk orkestrasi pipeline, Docker untuk environment lokal, DuckDB untuk analytical processing, Streamlit untuk dashboard, dan scikit-learn untuk komponen Machine Learning bonus.

---

## Tujuan Project

Project ini bertujuan untuk menganalisis bagaimana kondisi cuaca memengaruhi operasional taksi di New York City, terutama pada aspek:

- perubahan demand taksi saat hujan atau salju
- dampak cuaca terhadap durasi perjalanan
- perubahan fare dan tip saat cuaca buruk
- zona pickup yang paling sensitif terhadap cuaca
- perubahan pola origin-destination saat kondisi cuaca tertentu
- prediksi demand berdasarkan waktu, lokasi, dan cuaca
- clustering zona berdasarkan sensitivitas terhadap cuaca

Periode data yang digunakan adalah Januari sampai Maret 2025.

---

## Tech Stack

```text
Docker
Apache Airflow
Python
DuckDB
Pandas
PyArrow
scikit-learn
Streamlit
Plotly
```

---

## Dataset

### Dataset Utama

NYC TLC Yellow Taxi trip records:

```text
yellow_tripdata_2025-01.parquet
yellow_tripdata_2025-02.parquet
yellow_tripdata_2025-03.parquet
```

### Dataset Pendukung

NYC Taxi Zone Lookup:

```text
taxi_zone_lookup.csv
```

### Dataset Eksternal

Data cuaca hourly dari Open-Meteo Archive API untuk New York City:

```text
weather_hourly_2025-01_to_2025-03.parquet
weather_report_2025-01_to_2025-03.json
```

Variabel cuaca yang digunakan:

```text
temperature_2m
precipitation
rain
snowfall
wind_speed_10m
```

---

## Struktur Project

```text
rdv-tlc-project/
├── dags/
│   ├── hello_rdv_dag.py
│   ├── tlc_ingestion_dag.py
│   ├── tlc_cleaning_dag.py
│   ├── tlc_curated_dag.py
│   ├── tlc_weather_dag.py
│   ├── tlc_analysis_marts_dag.py
│   ├── tlc_ml_demand_prediction_dag.py
│   ├── tlc_ml_zone_clustering_dag.py
│   └── tlc_full_pipeline_dag.py
│
├── src/
│   ├── pipeline_config.py
│   ├── extract_tlc.py
│   ├── extract_weather.py
│   ├── clean_tlc.py
│   ├── build_curated.py
│   ├── build_weather_join.py
│   ├── build_analysis_marts.py
│   ├── train_demand_prediction.py
│   └── train_zone_clustering.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── curated/
│   ├── external/
│   └── ml/
│
├── docker-compose.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Gambaran Pipeline Data

Pipeline project ini menggunakan arsitektur berlapis:

```text
raw
 ↓
processed
 ↓
curated
 ↓
weather join
 ↓
analysis mart
 ↓
machine learning
 ↓
dashboard
```

### Deskripsi Layer

| Layer | Deskripsi |
|---|---|
| Raw | Data asli TLC, zone lookup, dan data eksternal |
| Processed | Data trip yang sudah dibersihkan |
| Curated | Fact table, dimension table, dan aggregate table |
| Weather Join | Data demand hourly yang digabung dengan data cuaca hourly |
| Analysis Mart | Tabel siap analisis untuk kebutuhan business question |
| ML | Output model demand prediction dan zone clustering |
| Dashboard | Visualisasi hasil analisis menggunakan Streamlit |

---

## Final Airflow Pipeline

Pipeline utama dijalankan melalui DAG:

```text
tlc_full_pipeline
```

DAG final ini memiliki 18 task dan menjalankan seluruh proses dari ingestion sampai Machine Learning.

```text
prepare_directories
        ↓
download_tlc_trip_data
download_zone_lookup
download_weather
        ↓
validate_ingestion
validate_weather
        ↓
clean_trip_data
        ↓
validate_cleaning
        ↓
build_curated_tables
        ↓
validate_curated
        ↓
build_weather_join
        ↓
validate_weather_join
        ↓
build_analysis_marts
        ↓
validate_analysis_marts
        ↓
train_demand_prediction        train_zone_clustering
        ↓                              ↓
validate_demand_prediction     validate_zone_clustering
```

---

## Daftar Airflow DAG

Project ini memiliki beberapa DAG modular untuk proses development/testing, serta satu DAG utama untuk menjalankan seluruh pipeline.

```text
hello_rdv_pipeline
tlc_ingestion_pipeline
tlc_cleaning_pipeline
tlc_curated_pipeline
tlc_weather_pipeline
tlc_analysis_marts_pipeline
tlc_ml_demand_prediction_pipeline
tlc_ml_zone_clustering_pipeline
tlc_full_pipeline
```

DAG yang direkomendasikan untuk menjalankan project secara end-to-end:

```text
tlc_full_pipeline
```

---

## Ringkasan Data Cleaning

Proses cleaning dilakukan untuk menghapus record trip yang tidak valid atau tidak realistis.

Contoh aturan cleaning:

```text
trip_distance > 0
trip_distance <= 100
fare_amount > 0
total_amount > 0
trip_duration_minutes > 0
pickup timestamp valid
dropoff timestamp valid
pickup location ID valid
dropoff location ID valid
```

Hasil cleaning untuk Januari sampai Maret 2025:

```text
Raw rows      : 11,198,026
Clean rows    : 8,464,290
Removed rows  : 2,733,736
Removed ratio : 24.41%
```

---

## Output Curated Layer

Output curated yang dihasilkan:

```text
data/curated/dim_location_2025-01_to_2025-03.parquet
data/curated/fact_trips_2025-01_to_2025-03.parquet
data/curated/agg_daily_summary_2025-01_to_2025-03.parquet
data/curated/agg_hourly_demand_2025-01_to_2025-03.parquet
data/curated/agg_zone_summary_2025-01_to_2025-03.parquet
data/curated/curated_report_2025-01_to_2025-03.json
```

### Tabel Utama Curated Layer

| Tabel | Deskripsi |
|---|---|
| `fact_trips` | Tabel fact berisi data trip bersih |
| `dim_location` | Dimension table untuk taxi zone |
| `agg_daily_summary` | Ringkasan trip, revenue, fare, dan distance per hari |
| `agg_hourly_demand` | Demand per jam berdasarkan pickup zone |
| `agg_zone_summary` | Ringkasan trip dan revenue per zona |

---

## Integrasi Data Cuaca

Data cuaca hourly digabungkan dengan data demand hourly menggunakan:

```text
pickup_date
pickup_hour
```

Output weather join:

```text
data/curated/agg_hourly_demand_weather_2025-01_to_2025-03.parquet
data/curated/weather_join_report_2025-01_to_2025-03.json
```

Kategori kondisi cuaca dibuat berdasarkan nilai precipitation, rain, dan snowfall.

```text
clear
light_rain
moderate_rain
heavy_rain
snow
```

---

## Analysis Mart Layer

Analysis mart dibuat agar data lebih siap digunakan untuk analisis bisnis, dashboard, dan Machine Learning.

Output analysis mart:

```text
data/curated/daily_zone_metrics_2025-01_to_2025-03.parquet
data/curated/hourly_demand_summary_weather_2025-01_to_2025-03.parquet
data/curated/weather_impact_summary_2025-01_to_2025-03.parquet
data/curated/zone_weather_elasticity_2025-01_to_2025-03.parquet
data/curated/od_flow_weather_2025-01_to_2025-03.parquet
data/curated/analysis_mart_report_2025-01_to_2025-03.json
```

### Tabel Analysis Mart

| Tabel | Fungsi |
|---|---|
| `daily_zone_metrics` | Metrik harian per zona berdasarkan kondisi cuaca |
| `hourly_demand_summary_weather` | Ringkasan demand hourly berdasarkan borough dan kondisi cuaca |
| `weather_impact_summary` | Ringkasan demand lift dan duration impact berdasarkan kondisi cuaca |
| `zone_weather_elasticity` | Metrik sensitivitas cuaca pada level zona |
| `od_flow_weather` | Pola origin-destination berdasarkan kondisi cuaca |

### Metrik Utama

Analysis mart menghitung beberapa metrik utama:

```text
demand_lift_pct
duration_delta_minutes
minutes_per_mile_delta
fare_delta_amount
tip_delta_pct
weather_sensitivity_label
```

Metrik tersebut digunakan untuk menjawab pertanyaan seperti:

- Seberapa besar demand berubah saat hujan atau salju?
- Zona mana yang mengalami demand lift tertinggi?
- Zona mana yang mengalami kenaikan durasi perjalanan saat cuaca buruk?
- Apakah fare dan tip berubah saat cuaca buruk?
- Apakah pola origin-destination berubah saat kondisi cuaca tertentu?

---

## Machine Learning Bonus

Project ini memiliki dua komponen Machine Learning bonus:

```text
1. Demand Prediction
2. Zone Weather Sensitivity Clustering
```

---

## ML Bonus 1: Demand Prediction

Model Random Forest Regressor digunakan untuk memprediksi jumlah trip taksi per pickup zone dan per jam.

### Target

```text
total_trips
```

### Fitur

```text
pickup_hour
pickup_dayofweek
pickup_month
pickup_day
is_weekend
pickup_location_id
pickup_borough
pickup_zone
weather_condition
temperature_2m
precipitation
rain
snowfall
wind_speed_10m
```

### Train-Test Split

Model menggunakan time-based split:

```text
Training period : 2025-01-01 sampai 2025-02-28
Testing period  : 2025-03-01 sampai 2025-03-31
```

### Model

```text
RandomForestRegressor
```

### Performa Model

```text
MAE  : 7.80
RMSE : 18.31
MAPE : 73.66%
R²   : 0.94
```

Baseline mean predictor:

```text
Baseline MAE  : 49.65
Baseline RMSE : 73.28
```

Model berhasil menghasilkan prediksi yang jauh lebih baik dibanding baseline sederhana berbasis rata-rata.

Nilai MAPE relatif tinggi karena banyak kombinasi zona-jam memiliki jumlah trip yang kecil. Pada kasus seperti ini, error absolut kecil dapat menghasilkan error persentase yang besar. Karena itu, evaluasi utama menggunakan MAE, RMSE, dan R².

### Insight Feature Importance

Fitur paling penting pada model:

```text
pickup_location_id
pickup_zone
pickup_hour
pickup_borough
pickup_dayofweek
```

Hal ini menunjukkan bahwa faktor lokasi dan waktu merupakan prediktor utama demand taksi. Fitur cuaca tetap digunakan sebagai konteks tambahan, sedangkan dampak cuaca secara spesifik dianalisis melalui weather impact mart dan zone weather elasticity.

### Output Demand Prediction

```text
data/ml/demand_prediction_results_2025-01_to_2025-03.parquet
data/ml/demand_model_metrics_2025-01_to_2025-03.json
data/ml/demand_feature_importance_2025-01_to_2025-03.parquet
data/ml/demand_prediction_model_2025-01_to_2025-03.joblib
```

---

## ML Bonus 2: Zone Weather Sensitivity Clustering

Model KMeans digunakan untuk mengelompokkan taxi zone berdasarkan pola sensitivitas terhadap cuaca.

### Model

```text
KMeans
```

### Konfigurasi

```text
random_state = 42
n_init       = 20
n_clusters  = 4
```

### Fitur

```text
total_weather_trips
total_weather_zone_hours
weather_condition_count
avg_demand_lift_pct
max_demand_lift_pct
avg_duration_delta_minutes
max_duration_delta_minutes
avg_minutes_per_mile_delta
max_minutes_per_mile_delta
avg_fare_delta_amount
avg_tip_delta_pct
avg_precipitation
high_sensitive_condition_count
medium_sensitive_condition_count
```

### Hasil Clustering

```text
Number of zones    : 116
Number of clusters : 4
Silhouette score   : 0.261
```

Silhouette score menunjukkan pemisahan cluster yang sedang. Hasil ini masih cukup aman untuk exploratory clustering pada data urban mobility karena pola zona kota biasanya saling overlap. Clustering ini digunakan sebagai segmentasi eksploratif, bukan sebagai bukti kausal.

### Profil Cluster

```text
Duration-sensitive zones
High-volume lower demand & faster trip zones
Low-volume lower demand & faster trip zones
High-volume high demand lift zones
```

Cluster yang paling penting secara operasional:

```text
High-volume high demand lift zones
```

Cluster ini berisi zona dengan volume trip tinggi dan demand lift tinggi saat cuaca buruk. Zona seperti ini dapat menjadi kandidat prioritas untuk monitoring dan redistribusi armada.

### Output Zone Clustering

```text
data/ml/zone_weather_clusters_2025-01_to_2025-03.parquet
data/ml/zone_cluster_summary_2025-01_to_2025-03.json
data/ml/zone_clustering_model_2025-01_to_2025-03.joblib
```

---

## Batasan Supply-Demand Gap

Project ini tidak menghitung supply-demand gap asli secara langsung.

Dataset NYC TLC hanya mencatat trip yang berhasil terjadi. Dataset ini tidak memiliki informasi seperti:

```text
jumlah taksi yang tersedia
jumlah driver aktif
jumlah permintaan yang tidak terlayani
request yang dibatalkan
waktu tunggu penumpang
ketersediaan armada
```

Karena keterbatasan tersebut, project ini berfokus pada metrik yang lebih valid berdasarkan data yang tersedia:

```text
demand lift
duration impact
zone weather elasticity
demand prediction
weather sensitivity clustering
```

---

## Dashboard

Dashboard Streamlit berada di:

```text
dashboard/app.py
```

Dashboard digunakan untuk menampilkan:

```text
overall taxi demand
daily trip trend
top pickup zones
hourly demand pattern
weather impact summary
zone weather elasticity
OD flow under weather conditions
demand prediction results
zone weather clustering results
```

Cara menjalankan dashboard:

```bash
cd dashboard
streamlit run app.py
```

---

## Cara Menjalankan Project

### 1. Jalankan Docker Services

Dari root project:

```bash
docker compose up -d
```

### 2. Buka Airflow UI

Buka browser:

```text
http://localhost:8080
```

### 3. Trigger Full Pipeline

Di Airflow, trigger DAG:

```text
tlc_full_pipeline
```

Pipeline akan menghasilkan output pada folder:

```text
data/raw
data/processed
data/external
data/curated
data/ml
```

---

## Menjalankan DAG Secara Terpisah

Untuk development atau debugging, setiap tahap pipeline dapat dijalankan secara modular.

```text
tlc_ingestion_pipeline
tlc_cleaning_pipeline
tlc_curated_pipeline
tlc_weather_pipeline
tlc_analysis_marts_pipeline
tlc_ml_demand_prediction_pipeline
tlc_ml_zone_clustering_pipeline
```

Untuk eksekusi final, gunakan:

```text
tlc_full_pipeline
```

---

## Validasi Pipeline

Setiap tahap utama memiliki task validasi.

Validasi yang dilakukan meliputi:

```text
cek keberadaan input file
cek output tidak kosong
cek row count
cek required columns
cek validitas file parquet
cek validitas JSON report
cek artifact model Machine Learning
```

Validasi ini membantu memastikan pipeline dapat dijalankan ulang secara aman dan konsisten.

---

## Git dan Penyimpanan Data

File data hasil generate tidak dimasukkan ke Git karena ukurannya besar.

Folder yang di-ignore:

```text
data/raw/*
data/processed/*
data/curated/*
data/external/*
data/ml/*
logs/*
```

Hanya file `.gitkeep` yang boleh tetap berada di dalam folder data.

Jangan commit file hasil generate seperti:

```text
*.parquet
*.joblib
large JSON reports
```

---

## Output Final Utama

### Curated Outputs

```text
dim_location_2025-01_to_2025-03.parquet
fact_trips_2025-01_to_2025-03.parquet
agg_daily_summary_2025-01_to_2025-03.parquet
agg_hourly_demand_2025-01_to_2025-03.parquet
agg_zone_summary_2025-01_to_2025-03.parquet
agg_hourly_demand_weather_2025-01_to_2025-03.parquet
```

### Analysis Mart Outputs

```text
daily_zone_metrics_2025-01_to_2025-03.parquet
hourly_demand_summary_weather_2025-01_to_2025-03.parquet
weather_impact_summary_2025-01_to_2025-03.parquet
zone_weather_elasticity_2025-01_to_2025-03.parquet
od_flow_weather_2025-01_to_2025-03.parquet
```

### ML Outputs

```text
demand_prediction_results_2025-01_to_2025-03.parquet
demand_model_metrics_2025-01_to_2025-03.json
demand_feature_importance_2025-01_to_2025-03.parquet
demand_prediction_model_2025-01_to_2025-03.joblib

zone_weather_clusters_2025-01_to_2025-03.parquet
zone_cluster_summary_2025-01_to_2025-03.json
zone_clustering_model_2025-01_to_2025-03.joblib
```

---

## Status Akhir Project

```text
Data ingestion                  Done
Data cleaning                   Done
Curated data modeling           Done
Weather integration             Done
Analysis mart layer             Done
Full Airflow orchestration      Done
ML demand prediction            Done
ML zone clustering              Done
Streamlit dashboard             In progress / ready for enhancement
```

---

## Catatan

Project ini hanya menggunakan data completed taxi trips. Oleh karena itu, hasil analisis tidak boleh diinterpretasikan sebagai pengukuran langsung terhadap unmet demand atau kekurangan supply taksi.

Project ini paling tepat dipahami sebagai weather impact analytics pipeline yang mengukur perubahan demand, perubahan durasi, dan sensitivitas zona terhadap cuaca menggunakan data trip yang berhasil terjadi dan data cuaca eksternal.