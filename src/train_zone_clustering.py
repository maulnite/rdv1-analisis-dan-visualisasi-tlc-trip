from pathlib import Path
from typing import Iterable
import json

import duckdb
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.clean_tlc import make_month_label
from src.train_demand_prediction import ensure_ml_directory


DATA_DIR = Path("/opt/airflow/data")
CURATED_DIR = DATA_DIR / "curated"
ML_DIR = DATA_DIR / "ml"


def load_zone_sensitivity_data(months: Iterable[str]) -> pd.DataFrame:
    month_label = make_month_label(months)
    input_path = CURATED_DIR / f"zone_weather_elasticity_{month_label}.parquet"

    if not input_path.exists() or input_path.stat().st_size == 0:
        raise FileNotFoundError(f"Input file not found or empty: {input_path}")

    print(f"[INFO] Loading zone weather elasticity data: {input_path}")

    con = duckdb.connect()

    df = con.sql(
        f"""
        SELECT
            zone_id,
            pickup_borough,
            pickup_zone,

            SUM(total_trips) AS total_weather_trips,
            SUM(total_zone_hours) AS total_weather_zone_hours,

            COUNT(DISTINCT weather_condition) AS weather_condition_count,

            ROUND(AVG(demand_lift_pct), 4) AS avg_demand_lift_pct,
            ROUND(MAX(demand_lift_pct), 4) AS max_demand_lift_pct,

            ROUND(AVG(duration_delta_minutes), 4) AS avg_duration_delta_minutes,
            ROUND(MAX(duration_delta_minutes), 4) AS max_duration_delta_minutes,

            ROUND(AVG(minutes_per_mile_delta), 4) AS avg_minutes_per_mile_delta,
            ROUND(MAX(minutes_per_mile_delta), 4) AS max_minutes_per_mile_delta,

            ROUND(AVG(fare_delta_amount), 4) AS avg_fare_delta_amount,
            ROUND(AVG(tip_delta_pct), 4) AS avg_tip_delta_pct,

            ROUND(AVG(avg_precipitation), 4) AS avg_precipitation,

            SUM(
                CASE
                    WHEN weather_sensitivity_label = 'high_weather_sensitive'
                    THEN 1 ELSE 0
                END
            ) AS high_sensitive_condition_count,

            SUM(
                CASE
                    WHEN weather_sensitivity_label = 'medium_weather_sensitive'
                    THEN 1 ELSE 0
                END
            ) AS medium_sensitive_condition_count

        FROM read_parquet('{input_path}')
        WHERE
            weather_condition <> 'clear'
            AND pickup_zone IS NOT NULL
            AND pickup_borough IS NOT NULL
            AND demand_lift_pct IS NOT NULL

        GROUP BY
            zone_id,
            pickup_borough,
            pickup_zone

        HAVING SUM(total_trips) >= 100

        ORDER BY total_weather_trips DESC
        """
    ).fetchdf()

    con.close()

    if df.empty:
        raise ValueError("Zone sensitivity dataframe is empty.")

    print(f"[INFO] Loaded zone rows: {len(df):,}")

    return df


def assign_cluster_profile(row: pd.Series) -> str:
    """
    Memberi nama cluster berdasarkan karakteristik rata-rata cluster.
    Label ini dipakai untuk interpretasi bisnis/dashboard, bukan output mentah KMeans.
    """
    demand = float(row.get("avg_demand_lift_pct", 0) or 0)
    duration = float(row.get("avg_duration_delta_minutes", 0) or 0)
    minutes_per_mile = float(row.get("avg_minutes_per_mile_delta", 0) or 0)
    trips = float(row.get("total_weather_trips", 0) or 0)
    zone_count = int(row.get("zone_count", 0) or 0)

    avg_trips_per_zone = trips / max(zone_count, 1)

    is_high_volume = trips >= 50_000 or avg_trips_per_zone >= 3_000
    is_low_volume = trips < 10_000 or avg_trips_per_zone < 500

    if demand >= 15 and is_high_volume:
        return "High-volume high demand lift zones"

    if demand >= 15:
        return "High demand lift zones"

    if duration >= 2 or minutes_per_mile >= 2:
        if is_high_volume:
            return "High-volume duration-sensitive zones"
        return "Duration-sensitive zones"

    if demand <= -5 and duration <= -2:
        if is_high_volume:
            return "High-volume lower demand & faster trip zones"
        if is_low_volume:
            return "Low-volume lower demand & faster trip zones"
        return "Lower demand & faster trip zones"

    if demand <= -5:
        if is_high_volume:
            return "High-volume lower demand under weather zones"
        return "Lower demand under weather zones"

    return "Stable weather response zones"


def train_zone_weather_clustering(months: Iterable[str]) -> dict:
    ensure_ml_directory()

    months = sorted(list(months))
    month_label = make_month_label(months)

    output_path = ML_DIR / f"zone_weather_clusters_{month_label}.parquet"
    summary_path = ML_DIR / f"zone_cluster_summary_{month_label}.json"
    model_path = ML_DIR / f"zone_clustering_model_{month_label}.joblib"

    for path in [output_path, summary_path, model_path]:
        if path.exists():
            path.unlink()
            print(f"[INFO] Removed old output: {path}")

    df = load_zone_sensitivity_data(months)
    df = df.sort_values(["zone_id", "pickup_borough", "pickup_zone"]).reset_index(drop=True)

    feature_cols = [
        "total_weather_trips",
        "total_weather_zone_hours",
        "weather_condition_count",
        "avg_demand_lift_pct",
        "max_demand_lift_pct",
        "avg_duration_delta_minutes",
        "max_duration_delta_minutes",
        "avg_minutes_per_mile_delta",
        "max_minutes_per_mile_delta",
        "avg_fare_delta_amount",
        "avg_tip_delta_pct",
        "avg_precipitation",
        "high_sensitive_condition_count",
        "medium_sensitive_condition_count",
    ]

    raw_df = df.copy()

    for col in feature_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    raw_df[feature_cols] = raw_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    raw_df[feature_cols] = raw_df[feature_cols].fillna(0)

    # Clipped dataframe hanya untuk training KMeans.
    # Raw dataframe tetap dipakai untuk output/reporting agar metrik bisnis tidak berubah.
    model_features_df = raw_df.copy()

    for col in feature_cols:
        lower = model_features_df[col].quantile(0.01)
        upper = model_features_df[col].quantile(0.99)
        model_features_df[col] = model_features_df[col].clip(lower=lower, upper=upper)

    n_samples = len(raw_df)
    n_clusters = min(4, max(2, n_samples // 20))

    print(f"[INFO] Samples   : {n_samples:,}")
    print(f"[INFO] Clusters  : {n_clusters}")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=20,
                ),
            ),
        ]
    )

    print("[STEP] Training KMeans clustering model...")
    cluster_labels = pipeline.fit_predict(model_features_df[feature_cols])

    raw_df["cluster_id"] = cluster_labels

    scaled_features = pipeline.named_steps["scaler"].transform(model_features_df[feature_cols])

    if n_clusters > 1 and len(set(cluster_labels)) > 1:
        silhouette = float(silhouette_score(scaled_features, cluster_labels))
    else:
        silhouette = None

    cluster_summary_df = (
        raw_df.groupby("cluster_id")
        .agg(
            zone_count=("zone_id", "count"),
            total_weather_trips=("total_weather_trips", "sum"),
            avg_demand_lift_pct=("avg_demand_lift_pct", "mean"),
            max_demand_lift_pct=("max_demand_lift_pct", "mean"),
            avg_duration_delta_minutes=("avg_duration_delta_minutes", "mean"),
            avg_minutes_per_mile_delta=("avg_minutes_per_mile_delta", "mean"),
            avg_fare_delta_amount=("avg_fare_delta_amount", "mean"),
            avg_tip_delta_pct=("avg_tip_delta_pct", "mean"),
            high_sensitive_condition_count=("high_sensitive_condition_count", "mean"),
        )
        .reset_index()
    )

    cluster_summary_df["cluster_profile"] = cluster_summary_df.apply(
        assign_cluster_profile,
        axis=1,
    )

    raw_df = raw_df.merge(
        cluster_summary_df[["cluster_id", "cluster_profile"]],
        on="cluster_id",
        how="left",
    )

    output_cols = [
        "zone_id",
        "pickup_borough",
        "pickup_zone",
        "cluster_id",
        "cluster_profile",
    ] + feature_cols

    result_df = raw_df[output_cols].sort_values(
        ["cluster_id", "avg_demand_lift_pct"],
        ascending=[True, False],
    )

    result_df.to_parquet(output_path, index=False)
    joblib.dump(pipeline, model_path)

    top_zones_by_cluster = {}

    for cluster_id in sorted(result_df["cluster_id"].unique()):
        cluster_rows = result_df[result_df["cluster_id"] == cluster_id].copy()

        cluster_profile = str(cluster_rows["cluster_profile"].iloc[0]).lower()

        if "high-volume" in cluster_profile and "high demand" in cluster_profile:
            cluster_rows = cluster_rows.sort_values(
                ["total_weather_trips", "avg_demand_lift_pct"],
                ascending=[False, False],
            )

        elif "high-volume" in cluster_profile and "lower demand" in cluster_profile:
            cluster_rows = cluster_rows.sort_values(
                ["total_weather_trips", "avg_duration_delta_minutes", "avg_demand_lift_pct"],
                ascending=[False, True, True],
            )

        elif "low-volume" in cluster_profile and "lower demand" in cluster_profile:
            cluster_rows = cluster_rows.sort_values(
                ["total_weather_trips", "avg_duration_delta_minutes", "avg_demand_lift_pct"],
                ascending=[True, True, True],
            )

        elif "duration" in cluster_profile:
            cluster_rows = cluster_rows.sort_values(
                "avg_duration_delta_minutes",
                ascending=False,
            )

        elif "faster" in cluster_profile:
            cluster_rows = cluster_rows.sort_values(
                "avg_duration_delta_minutes",
                ascending=True,
            )

        elif "lower demand" in cluster_profile:
            cluster_rows = cluster_rows.sort_values(
                "avg_demand_lift_pct",
                ascending=True,
            )

        else:
            cluster_rows = cluster_rows.sort_values(
                "avg_demand_lift_pct",
                ascending=False,
            )

        top_zones_by_cluster[str(cluster_id)] = cluster_rows[
            [
                "pickup_borough",
                "pickup_zone",
                "cluster_profile",
                "avg_demand_lift_pct",
                "avg_duration_delta_minutes",
                "total_weather_trips",
            ]
        ].head(10).to_dict(orient="records")

    summary = {
        "months": months,
        "model": "KMeans",
        "random_state": 42,
        "n_init": 20,
        "n_clusters": int(n_clusters),
        "n_samples": int(n_samples),
        "features": feature_cols,
        "silhouette_score": silhouette,
        "outputs": {
            "zone_weather_clusters": str(output_path),
            "model": str(model_path),
        },
        "cluster_summary": cluster_summary_df.round(4).to_dict(orient="records"),
        "top_zones_by_cluster": top_zones_by_cluster,
        "interpretation": {
            "cluster_id": "KMeans-generated cluster label.",
            "cluster_profile": "Business interpretation based on demand lift, duration impact, and weather trip volume.",
            "silhouette_score": "Measures how separated clusters are. Higher is better, but business interpretability is also important.",
        },
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)

    print("[RESULT] Zone weather clustering completed.")
    print(f"[RESULT] Zone rows        : {len(result_df):,}")
    print(f"[RESULT] Clusters         : {n_clusters}")
    print(f"[RESULT] Silhouette score : {silhouette}")
    print(f"[SAVED] Cluster output   : {output_path}")
    print(f"[SAVED] Cluster summary  : {summary_path}")
    print(f"[SAVED] Model            : {model_path}")

    return summary


def validate_zone_weather_clustering_outputs(months: Iterable[str]) -> None:
    months = sorted(list(months))
    month_label = make_month_label(months)

    expected_files = [
        ML_DIR / f"zone_weather_clusters_{month_label}.parquet",
        ML_DIR / f"zone_cluster_summary_{month_label}.json",
        ML_DIR / f"zone_clustering_model_{month_label}.joblib",
    ]

    missing_files = []

    for file_path in expected_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            missing_files.append(str(file_path))

    if missing_files:
        raise FileNotFoundError(f"Missing clustering output files: {missing_files}")

    df = pd.read_parquet(expected_files[0])

    if df.empty:
        raise ValueError("Zone weather clustering output is empty.")

    required_columns = [
        "zone_id",
        "pickup_borough",
        "pickup_zone",
        "cluster_id",
        "cluster_profile",
        "avg_demand_lift_pct",
        "avg_duration_delta_minutes",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required clustering output column: {col}")

    cluster_count = df["cluster_id"].nunique()

    if cluster_count < 2:
        raise ValueError("Clustering output has fewer than 2 clusters.")

    print("[CHECK] Zone weather clustering outputs are valid.")
    print(f"[CHECK] Zone rows    : {len(df):,}")
    print(f"[CHECK] Cluster count: {cluster_count}")


if __name__ == "__main__":
    from src.pipeline_config import MONTHS

    train_zone_weather_clustering(months=MONTHS)
    validate_zone_weather_clustering_outputs(months=MONTHS)