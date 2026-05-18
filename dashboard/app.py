from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NYC TLC Weather Analytics",
    page_icon="🚕",
    layout="wide",
)

PERIOD = "2025-01_to_2025-03"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CURATED_DIR = DATA_DIR / "curated"
ML_DIR = DATA_DIR / "ml"
ASSETS_DIR = PROJECT_ROOT / "dashboard" / "assets"

TAXI_ZONES_GEOJSON_PATH = ASSETS_DIR / "taxi_zones.geojson"

WEATHER_ORDER = ["clear", "light_rain", "moderate_rain", "heavy_rain", "snow"]

BOROUGH_COLOR_MAP = {
    "Manhattan": "#636EFA",
    "Brooklyn": "#EF553B",
    "Queens": "#00CC96",
    "Bronx": "#AB63FA",
    "Staten Island": "#FFA15A",
    "EWR": "#19D3F3",
}

WEATHER_COLOR_MAP = {
    "clear": "#636EFA",
    "heavy_rain": "#EF553B",
    "snow": "#00CC96",
    "moderate_rain": "#AB63FA",
    "light_rain": "#FFA15A",
}


px.defaults.template = "plotly_dark"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1220;
            color: #e5e7eb;
        }

        .block-container {
            padding-top: 3.4rem !important;
            padding-bottom: 2rem;
        }

        .main-title {
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.35;
            margin-top: 0.25rem;
            margin-bottom: 0.35rem;
            padding-top: 0.25rem;
            color: #f8fafc;
            overflow: visible;
        }

        header[data-testid="stHeader"] {
            background: rgba(11, 18, 32, 0.95);
        }

        .subtitle {
            color: #94a3b8;
            font-size: 1rem;
            margin-bottom: 1rem;
        }

        .section-note {
            color: #94a3b8;
            font-size: 0.92rem;
            margin-bottom: 1rem;
        }

        .insight-box {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.28);
            padding: 1rem;
            border-radius: 0.9rem;
            color: #cbd5e1;
            margin-top: 0.7rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 1rem;
            border-radius: 1rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }

        div[data-testid="stMetric"] * {
            color: #f8fafc !important;
        }

        div[data-testid="stMetricLabel"] p {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 800 !important;
        }

        div[data-testid="stMetricDelta"] * {
            color: #86efac !important;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 0.75rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.75rem;
        }

        .stTabs [data-baseweb="tab"] {
            font-weight: 700;
        }

        section[data-testid="stSidebar"] {
            background-color: #111827;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def file_must_exist(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        st.error(f"File belum ada atau kosong: {path}")
        st.info("Jalankan DAG `tlc_full_pipeline` di Airflow terlebih dahulu, atau extract data ZIP ke folder project.")
        st.stop()
    return path


@st.cache_data(show_spinner=False)
def read_parquet(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    file_must_exist(path)

    query = f"""
    SELECT *
    FROM read_parquet('{path.as_posix()}')
    """

    return duckdb.sql(query).df()


@st.cache_data(show_spinner=False)
def read_json(path_str: str) -> dict:
    path = Path(path_str)
    file_must_exist(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def read_geojson(path_str: str) -> dict | None:
    path = Path(path_str)

    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("r", encoding="utf-8") as file:
        geojson_data = json.load(file)

    for feature in geojson_data.get("features", []):
        properties = feature.setdefault("properties", {})

        location_id = (
            properties.get("LocationID")
            or properties.get("location_id")
            or properties.get("location_i")
            or properties.get("locationid")
            or properties.get("LocationId")
            or properties.get("OBJECTID")
            or properties.get("objectid")
        )

        if location_id is not None:
            try:
                properties["LocationID"] = str(int(float(str(location_id))))
            except Exception:
                properties["LocationID"] = str(location_id)

    return geojson_data


def format_number(value: float, decimal: bool = False) -> str:
    try:
        if decimal:
            return f"{float(value):,.2f}"
        return f"{float(value):,.0f}"
    except Exception:
        return "0"


def format_currency(value: float) -> str:
    try:
        value = float(value)

        if value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"
        if value >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"
        if value >= 1_000:
            return f"${value / 1_000:.2f}K"

        return f"${value:,.2f}"
    except Exception:
        return "$0.00"


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def safe_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    selected_columns = existing_columns(df, columns)
    if not selected_columns:
        return df
    return df[selected_columns]


def weighted_average(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if df.empty or value_col not in df.columns or weight_col not in df.columns:
        return 0.0

    valid = df[[value_col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]

    if valid.empty or valid[weight_col].sum() == 0:
        return 0.0

    return float((valid[value_col] * valid[weight_col]).sum() / valid[weight_col].sum())


def weighted_median(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if df.empty or value_col not in df.columns or weight_col not in df.columns:
        return 0.0

    valid = df[[value_col, weight_col]].dropna()
    valid = valid[valid[weight_col] > 0]

    if valid.empty:
        return 0.0

    df_sorted = valid.sort_values(value_col)
    cumulative_weight = df_sorted[weight_col].cumsum()
    cutoff = df_sorted[weight_col].sum() / 2.0

    result = df_sorted.loc[cumulative_weight >= cutoff, value_col]

    if result.empty:
        return 0.0

    return float(result.iloc[0])


def apply_weather_order(df: pd.DataFrame, column: str = "weather_condition") -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        df[column] = pd.Categorical(
            df[column].astype(str),
            categories=WEATHER_ORDER,
            ordered=True,
        )
        df = df.sort_values(column)
    return df


def update_chart_layout(fig, height: int | None = None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=20, r=20, t=60, b=40),
    )

    if height:
        fig.update_layout(height=height)

    return fig


def filter_by_value(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if value == "All" or column not in df.columns:
        return df

    return df[df[column].astype(str) == str(value)]


def filter_by_list(df: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    if not values or column not in df.columns:
        return df

    return df[df[column].astype(str).isin([str(value) for value in values])]


def filter_by_day_type(df: pd.DataFrame, selected_days: str) -> pd.DataFrame:
    if selected_days not in ["Weekdays", "Weekends"]:
        return df

    if "is_weekend" not in df.columns:
        return df

    is_weekend_flag = selected_days == "Weekends"
    return df[df["is_weekend"].astype(bool) == is_weekend_flag]


def normalize_location_id_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def build_zone_summary_from_hourly(hourly_df: pd.DataFrame, fallback_zone_df: pd.DataFrame) -> pd.DataFrame:
    if hourly_df.empty:
        return fallback_zone_df.copy()

    required_columns = ["pickup_location_id", "pickup_borough", "pickup_zone", "total_trips"]
    if not all(column in hourly_df.columns for column in required_columns):
        return fallback_zone_df.copy()

    work = hourly_df.copy()
    work["total_trips"] = pd.to_numeric(work["total_trips"], errors="coerce").fillna(0)

    group_cols = ["pickup_location_id", "pickup_borough", "pickup_zone"]
    result = (
        work.groupby(group_cols, as_index=False)
        .agg(total_trips=("total_trips", "sum"))
    )

    if "total_revenue" in work.columns:
        result = result.merge(
            work.groupby(group_cols, as_index=False).agg(total_revenue=("total_revenue", "sum")),
            on=group_cols,
            how="left",
        )

    weighted_metrics = [
        "avg_total_amount",
        "avg_trip_duration_minutes",
        "avg_trip_distance",
        "avg_speed_mph",
        "avg_tip_percentage",
        "avg_tip_pct",
    ]

    for metric in weighted_metrics:
        if metric in work.columns:
            metric_work = work[group_cols + ["total_trips", metric]].dropna()
            metric_work = metric_work[metric_work["total_trips"] > 0]

            if not metric_work.empty:
                metric_work[f"{metric}_weighted"] = metric_work[metric] * metric_work["total_trips"]
                metric_agg = (
                    metric_work.groupby(group_cols, as_index=False)
                    .agg(
                        metric_weighted_sum=(f"{metric}_weighted", "sum"),
                        metric_weight_sum=("total_trips", "sum"),
                    )
                )
                metric_agg[metric] = metric_agg["metric_weighted_sum"] / metric_agg["metric_weight_sum"]
                metric_agg = metric_agg[group_cols + [metric]]

                result = result.merge(metric_agg, on=group_cols, how="left")

    return result


def get_prediction_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    actual_col = first_existing_column(
        df,
        ["actual_total_trips", "actual_trips", "total_trips", "y_actual"],
    )
    predicted_col = first_existing_column(
        df,
        ["predicted_total_trips", "predicted_trips", "prediction", "y_pred"],
    )
    abs_error_col = first_existing_column(
        df,
        ["absolute_error", "abs_error", "prediction_abs_error"],
    )
    return actual_col, predicted_col, abs_error_col


# ============================================================
# LOAD DATA
# ============================================================

daily_df = read_parquet(str(CURATED_DIR / f"agg_daily_summary_{PERIOD}.parquet"))
zone_df = read_parquet(str(CURATED_DIR / f"agg_zone_summary_{PERIOD}.parquet"))
hourly_df = read_parquet(str(CURATED_DIR / f"agg_hourly_demand_{PERIOD}.parquet"))
weather_hourly_df = read_parquet(str(CURATED_DIR / f"agg_hourly_demand_weather_{PERIOD}.parquet"))

weather_impact_df = read_parquet(str(CURATED_DIR / f"weather_impact_summary_{PERIOD}.parquet"))
zone_elasticity_df = read_parquet(str(CURATED_DIR / f"zone_weather_elasticity_{PERIOD}.parquet"))
od_flow_df = read_parquet(str(CURATED_DIR / f"od_flow_weather_{PERIOD}.parquet"))

demand_results_df = read_parquet(str(ML_DIR / f"demand_prediction_results_{PERIOD}.parquet"))
feature_importance_df = read_parquet(str(ML_DIR / f"demand_feature_importance_{PERIOD}.parquet"))
demand_metrics = read_json(str(ML_DIR / f"demand_model_metrics_{PERIOD}.json"))

zone_clusters_df = read_parquet(str(ML_DIR / f"zone_weather_clusters_{PERIOD}.parquet"))
cluster_summary = read_json(str(ML_DIR / f"zone_cluster_summary_{PERIOD}.json"))


# ============================================================
# BASIC DATA PREP
# ============================================================

for df in [daily_df, weather_hourly_df, demand_results_df]:
    if "pickup_date" in df.columns:
        df["pickup_date"] = pd.to_datetime(df["pickup_date"])

for df in [weather_hourly_df, demand_results_df]:
    if "is_weekend" in df.columns:
        df["is_weekend"] = df["is_weekend"].astype(bool)

weather_impact_df = apply_weather_order(weather_impact_df)
zone_elasticity_df = apply_weather_order(zone_elasticity_df)
od_flow_df = apply_weather_order(od_flow_df)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚕 NYC TLC Weather Analytics & ML Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Analisis pengaruh cuaca terhadap demand, durasi perjalanan, zona sensitif cuaca, OD flow, dan prediksi demand taksi NYC Yellow Taxi periode Januari–Maret 2025.</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Catatan: dashboard ini menggunakan completed taxi trips. "
    "Hasil analisis tidak merepresentasikan supply-demand gap asli karena data tidak memuat jumlah armada tersedia atau unmet demand."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Dashboard Filters")

raw_borough_options = (
    zone_elasticity_df["pickup_borough"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

borough_options = sorted(
    [borough for borough in raw_borough_options if borough not in ["EWR", "Unknown", "nan"]]
)

selected_borough = st.sidebar.selectbox(
    "Borough",
    ["All"] + borough_options,
)

weather_source = zone_elasticity_df["weather_condition"].dropna().astype(str).unique().tolist()
weather_options = [weather for weather in WEATHER_ORDER if weather in weather_source]

selected_weather = st.sidebar.multiselect(
    "Weather Condition",
    weather_options,
    default=weather_options,
)

active_weather_conditions = selected_weather if selected_weather else weather_options

top_n = st.sidebar.slider(
    "Top N",
    min_value=5,
    max_value=100,
    value=10,
    step=5,
)

min_zone_trips = st.sidebar.slider(
    "Min Trips for Zone Elasticity",
    min_value=0,
    max_value=500,
    value=50,
    step=25,
    help="Filter minimum jumlah trip untuk kombinasi pickup zone dan weather condition. Hanya memengaruhi tab Zone Elasticity.",
)

st.sidebar.caption("Min Trips hanya memengaruhi tab Zone Elasticity.")

days = st.sidebar.selectbox(
    "Days",
    ["All Days", "Weekdays", "Weekends", "Comparison"],
)

title_borough = "All Boroughs" if selected_borough == "All" else selected_borough


# ============================================================
# FILTERED DATASETS
# ============================================================

filtered_zone_elasticity = zone_elasticity_df.copy()
filtered_od_flow = od_flow_df.copy()
filtered_clusters = zone_clusters_df.copy()
filtered_demand_results = demand_results_df.copy()
filtered_weather_hourly = weather_hourly_df.copy()
filtered_zone_summary = zone_df.copy()

if selected_borough != "All":
    filtered_zone_elasticity = filter_by_value(filtered_zone_elasticity, "pickup_borough", selected_borough)
    filtered_clusters = filter_by_value(filtered_clusters, "pickup_borough", selected_borough)
    filtered_od_flow = filter_by_value(filtered_od_flow, "origin_borough", selected_borough)
    filtered_demand_results = filter_by_value(filtered_demand_results, "pickup_borough", selected_borough)
    filtered_weather_hourly = filter_by_value(filtered_weather_hourly, "pickup_borough", selected_borough)
    filtered_zone_summary = filter_by_value(filtered_zone_summary, "pickup_borough", selected_borough)

if active_weather_conditions:
    filtered_zone_elasticity = filter_by_list(filtered_zone_elasticity, "weather_condition", active_weather_conditions)
    filtered_od_flow = filter_by_list(filtered_od_flow, "weather_condition", active_weather_conditions)
    filtered_demand_results = filter_by_list(filtered_demand_results, "weather_condition", active_weather_conditions)
    filtered_weather_hourly = filter_by_list(filtered_weather_hourly, "weather_condition", active_weather_conditions)

if days in ["Weekdays", "Weekends"]:
    filtered_weather_hourly = filter_by_day_type(filtered_weather_hourly, days)
    filtered_demand_results = filter_by_day_type(filtered_demand_results, days)

filtered_zone_elasticity = filtered_zone_elasticity[
    filtered_zone_elasticity["total_trips"] >= min_zone_trips
].copy()

filtered_zone_summary_from_weather = build_zone_summary_from_hourly(
    filtered_weather_hourly,
    filtered_zone_summary,
)


# ============================================================
# TABS
# ============================================================

tab_overview, tab_weather, tab_zone, tab_od, tab_prediction, tab_cluster = st.tabs(
    [
        "📌 Executive Overview",
        "🌧️ Weather Impact",
        "📍 Zone Elasticity",
        "🔁 OD Flow",
        "🤖 Demand Prediction",
        "🧩 Zone Clustering",
    ]
)


# ============================================================
# TAB 1: EXECUTIVE OVERVIEW
# ============================================================

with tab_overview:
    st.subheader("Executive Overview")
    st.markdown(
        f'<div class="section-note">Ringkasan performa taxi demand, gross trip value, fare, distance, dan durasi perjalanan untuk filter: <b>{title_borough}</b>.</div>',
        unsafe_allow_html=True,
    )

    overview_source = filtered_weather_hourly.copy()

    total_trips = int(overview_source["total_trips"].sum()) if "total_trips" in overview_source.columns else 0
    num_days = overview_source["pickup_date"].nunique() if "pickup_date" in overview_source.columns else 0
    avg_trips_per_day = total_trips / num_days if num_days > 0 else 0

    total_revenue = float(overview_source["total_revenue"].sum()) if "total_revenue" in overview_source.columns else 0.0
    avg_fare = weighted_average(overview_source, "avg_total_amount", "total_trips")
    median_fare = weighted_median(overview_source, "avg_total_amount", "total_trips")
    avg_duration = weighted_average(overview_source, "avg_trip_duration_minutes", "total_trips")
    median_duration = weighted_median(overview_source, "avg_trip_duration_minutes", "total_trips")
    avg_distance = weighted_average(overview_source, "avg_trip_distance", "total_trips")
    median_distance = weighted_median(overview_source, "avg_trip_distance", "total_trips")

    if not overview_source.empty and "pickup_date" in overview_source.columns and "rain" in overview_source.columns:
        rainy_share = (
            (overview_source.groupby("pickup_date")["rain"].mean() > 0.1)
            .mean() * 100
        )
    else:
        rainy_share = 0.0

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Total Trips",
        format_number(total_trips),
        delta=f"Avg/day: {format_number(avg_trips_per_day, True)}",
        delta_color="off",
    )
    col2.metric("Gross Trip Value", format_currency(total_revenue))
    col3.metric(
        "Fare",
        format_currency(avg_fare),
        delta=f"Median: {format_currency(median_fare)}",
        delta_color="off",
    )
    col4.metric(
        "Duration",
        f"{avg_duration:.2f} min",
        delta=f"Median: {median_duration:.2f} min",
        delta_color="off",
    )
    col5.metric(
        "Distance",
        f"{avg_distance:.2f} mi",
        delta=f"Median: {median_distance:.2f} mi",
        delta_color="off",
    )
    col6.metric("Rainy Day Share", f"{rainy_share:.1f}%")

    st.divider()

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("### Daily Trip Trend")

        if overview_source.empty or "pickup_date" not in overview_source.columns:
            st.warning("Tidak ada data daily trend untuk filter yang dipilih.")
        else:
            daily_plot = (
                overview_source.groupby("pickup_date", as_index=False)
                .agg(
                    total_trips=("total_trips", "sum"),
                    avg_precipitation=("precipitation", "mean") if "precipitation" in overview_source.columns else ("total_trips", "size"),
                )
                .sort_values("pickup_date")
            )

            if "avg_precipitation" not in daily_plot.columns:
                daily_plot["avg_precipitation"] = 0.0

            fig_daily = make_subplots(specs=[[{"secondary_y": True}]])

            fig_daily.add_trace(
                go.Scatter(
                    x=daily_plot["pickup_date"],
                    y=daily_plot["total_trips"],
                    mode="lines+markers",
                    name="Total Trips",
                ),
                secondary_y=False,
            )

            fig_daily.add_trace(
                go.Bar(
                    x=daily_plot["pickup_date"],
                    y=daily_plot["avg_precipitation"],
                    name="Avg Precipitation",
                    opacity=0.35,
                ),
                secondary_y=True,
            )

            fig_daily.update_layout(
                title=f"Daily Total Trips vs Precipitation - {title_borough}",
                hovermode="x unified",
            )
            fig_daily.update_yaxes(title_text="Total Trips", secondary_y=False)
            fig_daily.update_yaxes(title_text="Avg Precipitation", secondary_y=True)
            fig_daily.update_xaxes(
                range=[daily_plot["pickup_date"].min(), daily_plot["pickup_date"].max()],
                tickformat="%d %b %Y",
                title="Pickup Date",
            )

            st.plotly_chart(update_chart_layout(fig_daily, height=430), width='stretch')

    with col_right:
        st.markdown("### Top Pickup Zones")

        if filtered_zone_summary_from_weather.empty:
            st.warning("Tidak ada top zone untuk filter yang dipilih.")
        else:
            top_zones = (
                filtered_zone_summary_from_weather
                .sort_values("total_trips", ascending=False)
                .head(top_n)
            )

            fig_zone = px.bar(
                top_zones.sort_values("total_trips", ascending=True),
                x="total_trips",
                y="pickup_zone",
                color="pickup_borough" if "pickup_borough" in top_zones.columns else None,
                color_discrete_map=BOROUGH_COLOR_MAP,
                orientation="h",
                title=f"Top {top_n} Pickup Zones - {title_borough}",
                hover_data=existing_columns(
                    top_zones,
                    ["pickup_borough", "total_revenue", "avg_total_amount", "avg_trip_duration_minutes"],
                ),
            )

            fig_zone.update_layout(
                xaxis_title="Total Trips",
                yaxis_title="Pickup Zone",
                legend_title="Borough",
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(update_chart_layout(fig_zone, height=430), width='stretch')

    st.markdown("### Hourly Demand Pattern")

    if filtered_weather_hourly.empty:
        st.warning("Tidak ada hourly demand untuk filter yang dipilih.")
    else:
        if days == "Comparison" and "is_weekend" in filtered_weather_hourly.columns:
            weekday_hourly_pattern = (
                filtered_weather_hourly[filtered_weather_hourly["is_weekend"].astype(bool) == False]
                .groupby("pickup_hour", as_index=False)
                .agg(total_trips=("total_trips", "sum"))
            )
            weekday_hourly_pattern["day_type"] = "Weekdays"

            weekend_hourly_pattern = (
                filtered_weather_hourly[filtered_weather_hourly["is_weekend"].astype(bool) == True]
                .groupby("pickup_hour", as_index=False)
                .agg(total_trips=("total_trips", "sum"))
            )
            weekend_hourly_pattern["day_type"] = "Weekends"

            comparison_df = pd.concat(
                [weekday_hourly_pattern, weekend_hourly_pattern],
                ignore_index=True,
            ).sort_values("pickup_hour")

            fig_hourly = px.line(
                comparison_df,
                x="pickup_hour",
                y="total_trips",
                color="day_type",
                markers=True,
                title=f"Total Trips by Pickup Hour - {title_borough}",
            )
        else:
            hourly_pattern = (
                filtered_weather_hourly
                .groupby("pickup_hour", as_index=False)
                .agg(total_trips=("total_trips", "sum"))
                .sort_values("pickup_hour")
            )

            fig_hourly = px.line(
                hourly_pattern,
                x="pickup_hour",
                y="total_trips",
                markers=True,
                title=f"Total Trips by Pickup Hour - {title_borough}",
            )

        fig_hourly.update_layout(
            xaxis=dict(
                title="Pickup Hour",
                tickmode="array",
                tickvals=list(range(24)),
                ticktext=[f"{hour:02d}.00" for hour in range(24)],
                range=[0, 23],
            ),
            yaxis_title="Total Trips",
            hovermode="x unified",
        )

        st.plotly_chart(update_chart_layout(fig_hourly, height=440), width='stretch')

    st.divider()
    st.markdown("### NYC Taxi Demand GeoMap")
    # st.markdown(
    #     '<div class="section-note">Peta choropleth menggunakan file lokal <code>dashboard/assets/taxi_zones.geojson</code>. Data metrik tetap berasal dari hasil pipeline.</div>',
    #     unsafe_allow_html=True,
    # )

    taxi_zones_geojson = read_geojson(str(TAXI_ZONES_GEOJSON_PATH))

    if taxi_zones_geojson is None:
        st.warning(
            "File GeoJSON belum tersedia atau kosong. "
            "Letakkan file di `dashboard/assets/taxi_zones.geojson` agar peta bisa dirender."
        )
    elif filtered_zone_summary_from_weather.empty:
        st.warning("Tidak ada data zona untuk ditampilkan di peta.")
    else:
        geomap_data = filtered_zone_summary_from_weather.copy()

        if "pickup_location_id" not in geomap_data.columns:
            st.warning("Kolom pickup_location_id tidak tersedia untuk mapping GeoJSON.")
        else:
            geomap_data["map_id"] = normalize_location_id_series(geomap_data["pickup_location_id"])
            geomap_data = geomap_data.dropna(subset=["map_id"])

            map_level = st.selectbox(
                "Level Agregasi Peta",
                ["Per Zona Taxi", "Per Borough"],
            )

            map_metric = st.selectbox(
                "Metrik Peta",
                ["Total Trips", "Average Fare (USD)", "Average Duration (Min)", "Average Distance (Miles)"],
            )

            metric_config = {
                "Total Trips": ("total_trips", "Total Trips", True),
                "Average Fare (USD)": ("avg_total_amount", "Average Fare (USD)", False),
                "Average Duration (Min)": ("avg_trip_duration_minutes", "Average Duration (Min)", False),
                "Average Distance (Miles)": ("avg_trip_distance", "Average Distance (Miles)", False),
            }

            metric_col, metric_label, use_log_scale = metric_config[map_metric]

            if metric_col not in geomap_data.columns:
                st.warning(f"Kolom `{metric_col}` tidak tersedia untuk peta.")
            else:
                geomap_data[metric_col] = pd.to_numeric(geomap_data[metric_col], errors="coerce").fillna(0)

                if map_level == "Per Borough" and "pickup_borough" in geomap_data.columns:
                    if metric_col == "total_trips":
                        geomap_data["display_value"] = geomap_data.groupby("pickup_borough")[metric_col].transform("sum")
                    else:
                        geomap_data["display_value"] = geomap_data.groupby("pickup_borough")[metric_col].transform("mean")

                    hover_title = "pickup_borough"
                    color_bar_title = f"{metric_label} by Borough"
                else:
                    geomap_data["display_value"] = geomap_data[metric_col]
                    hover_title = "pickup_zone" if "pickup_zone" in geomap_data.columns else "map_id"
                    color_bar_title = metric_label

                if use_log_scale:
                    geomap_data["map_color_value"] = np.log1p(geomap_data["display_value"])
                    color_bar_title = f"Log({color_bar_title})"

                    max_val = geomap_data["display_value"].max()
                    min_val = max(geomap_data["display_value"].min(), 1)  # avoid log(0)

                    # Update the ticxbox into exponential
                    log_min = np.floor(np.log10(min_val))
                    log_max = np.ceil(np.log10(max_val))
                    original_ticks = np.logspace(log_min, log_max, num=6)
                    original_ticks = np.clip(original_ticks, min_val, max_val)

                    tick_vals = np.log1p(original_ticks)
                    tick_text = [f"{v:,.0f}" for v in original_ticks]
                else:
                    geomap_data["map_color_value"] = geomap_data["display_value"]

                try:
                    hover_data = {
                        "map_id": False,
                        "map_color_value": False,
                        "pickup_zone": "pickup_zone" in geomap_data.columns,
                        "pickup_borough": "pickup_borough" in geomap_data.columns,
                        f"{metric_col}": ":," if metric_col == "total_trips" else ":,.2f",
                        # f"{metric_col}": False if metric_col == "total_trips" else ":,",
                    }

                    
                    fig_map = px.choropleth_map(
                        geomap_data,
                        geojson=taxi_zones_geojson,
                        featureidkey="properties.LocationID",
                        locations="map_id",
                        color="map_color_value",
                        color_continuous_scale="Viridis",
                        map_style="carto-darkmatter",
                        zoom=9,
                        center={"lat": 40.7128, "lon": -74.0060},
                        opacity=0.72,
                        hover_name=hover_title,
                        hover_data=hover_data,
                        labels={
                            "display_value": metric_label,
                            "map_color_value": color_bar_title,
                        },
                    )

                    fig_map.update_layout(
                        margin={"r": 0, "t": 0, "l": 0, "b": 0},
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e5e7eb"),
                        coloraxis_colorbar=dict(
                            title=dict(text=color_bar_title, font=dict(color="#e5e7eb")),
                            tickfont=dict(color="#e5e7eb"),
                        ),
                    )
                    
                    if use_log_scale:
                        fig_map.update_coloraxes(
                            colorbar=dict(tickvals=tick_vals, ticktext=tick_text)
                        )

                    st.plotly_chart(fig_map, width='stretch')
                except Exception as error:
                    st.error(f"Gagal merender peta: {error}")


# ============================================================
# TAB 2: WEATHER IMPACT
# ============================================================

with tab_weather:
    st.subheader("Weather Impact Analysis")
    st.markdown(
        '<div class="section-note">Analisis demand lift, perubahan durasi, fare, dan tip berdasarkan kondisi cuaca.</div>',
        unsafe_allow_html=True,
    )

    weather_display = weather_impact_df.copy()

    if "weather_condition" in weather_display.columns:
        weather_display["weather_condition"] = weather_display["weather_condition"].astype(str)

    if active_weather_conditions:
        weather_display = filter_by_list(weather_display, "weather_condition", active_weather_conditions)

    if weather_display.empty:
        st.warning("Tidak ada data weather impact untuk filter cuaca yang dipilih.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        best_condition = weather_display.sort_values("demand_lift_pct", ascending=False).iloc[0]
        highest_duration = weather_display.sort_values("duration_delta_minutes", ascending=False).iloc[0]
        highest_fare = weather_display.sort_values("fare_delta_amount", ascending=False).iloc[0]
        highest_volume_condition = weather_display.sort_values("total_trips", ascending=False).iloc[0]

        col1.metric(
            "Highest Demand Lift",
            str(best_condition["weather_condition"]),
            f"{best_condition['demand_lift_pct']:.2f}%",
        )
        col2.metric(
            "Highest Duration Impact",
            str(highest_duration["weather_condition"]),
            f"{highest_duration['duration_delta_minutes']:.2f} min",
        )
        col3.metric(
            "Highest Fare Amount Impact",
            str(highest_fare["weather_condition"]),
            f"{highest_fare['fare_delta_amount']:.2f} USD",
        )
        col4.metric(
            "Largest Total Trips",
            str(highest_volume_condition["weather_condition"]),
            format_number(highest_volume_condition["total_trips"]),
        )

        st.markdown(
            f"""
            <div class="insight-box">
            <b>Insight:</b> Pada filter cuaca aktif, kondisi <b>{best_condition['weather_condition']}</b>
            memiliki demand lift tertinggi sebesar <b>{best_condition['demand_lift_pct']:.2f}%</b>.
            Kondisi dengan volume trip terbesar adalah <b>{highest_volume_condition['weather_condition']}</b>,
            sehingga interpretasi perlu membedakan antara <i>relative lift</i> dan volume absolut.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        
        weather_display_without_clear = weather_display[weather_display["weather_condition"] != "clear"].copy()

        col_left, col_middle, col_right = st.columns(3)

        with col_left:
            fig_lift = px.bar(
                weather_display_without_clear,
                x="weather_condition",
                y="demand_lift_pct",
                title="Demand Lift by Weather Condition",
                hover_data=existing_columns(
                    weather_display,
                    ["total_trips", "avg_precipitation", "avg_rain"],
                ),
            )
            fig_lift.update_layout(
                xaxis_title="Weather Condition",
                yaxis_title="Demand Lift (%)",
            )
            st.plotly_chart(update_chart_layout(fig_lift, height=430), width='stretch')
            
        with col_middle:
            fig_fare_amount = px.bar(
                weather_display_without_clear,
                x="weather_condition",
                y="fare_delta_amount",
                title="Fare Amount by Weather Condition",
                hover_data=existing_columns(
                    weather_display,
                    ["total_trips", "avg_duration", "avg_trip_duration_minutes"],
                ),
            )
            fig_fare_amount.update_layout(
                xaxis_title="Weather Condition",
                yaxis_title="Fare Amount (USD)",
            )
            st.plotly_chart(update_chart_layout(fig_fare_amount, height=430), width='stretch')

        with col_right:
            fig_duration = px.bar(
                weather_display_without_clear,
                x="weather_condition",
                y="duration_delta_minutes",
                title="Duration Delta by Weather Condition",
                hover_data=existing_columns(
                    weather_display,
                    ["total_trips", "avg_duration", "avg_trip_duration_minutes"],
                ),
            )
            fig_duration.update_layout(
                xaxis_title="Weather Condition",
                yaxis_title="Duration Delta (minutes)",
            )
            st.plotly_chart(update_chart_layout(fig_duration, height=430), width='stretch')

        st.markdown("### Weather Trade-Off: Demand Lift vs Duration Impact")

        bubble_size = "total_trips" if "total_trips" in weather_display.columns else None

        fig_tradeoff = px.scatter(
            weather_display,
            x="demand_lift_pct",
            y="duration_delta_minutes",
            size=bubble_size,
            color="weather_condition",
            color_discrete_map=WEATHER_COLOR_MAP,
            hover_name="weather_condition",
            title="Weather Impact Positioning",
            hover_data=existing_columns(
                weather_display,
                ["total_trips", "fare_delta_amount", "tip_delta_pct"],
            ),
        )
        fig_tradeoff.update_layout(
            xaxis_title="Demand Lift (%)",
            yaxis_title="Duration Delta (minutes)",
        )
        st.plotly_chart(update_chart_layout(fig_tradeoff, height=460), width='stretch')

        st.markdown("### Weather Impact Table")
        st.dataframe(
            safe_dataframe(
                weather_display,
                [
                    "weather_condition",
                    "total_trips",
                    "avg_trips_per_zone_hour",
                    "demand_lift_pct",
                    "avg_duration",
                    "duration_delta_minutes",
                    "avg_total_amount",
                    "fare_delta_amount",
                    "avg_tip_pct",
                    "tip_delta_pct",
                    "avg_precipitation",
                    "avg_rain",
                    "avg_snowfall",
                ],
            ),
            width='stretch',
        )


# ============================================================
# TAB 3: ZONE ELASTICITY
# ============================================================

with tab_zone:
    st.subheader("Zone Weather Elasticity")
    st.markdown(
        '<div class="section-note">Identifikasi zona paling sensitif terhadap cuaca berdasarkan demand lift dan duration impact.</div>',
        unsafe_allow_html=True,
    )

    if filtered_zone_elasticity.empty:
        st.warning("Tidak ada data zone elasticity untuk filter yang dipilih.")
    else:
        col_left, col_middle, col_right = st.columns(3)

        with col_left:
            st.markdown("### Top Demand Lift Zones")

            top_lift = (
                filtered_zone_elasticity
                .sort_values("demand_lift_pct", ascending=False)
                .head(top_n)
            )

            fig_top_lift = px.bar(
                top_lift.sort_values("demand_lift_pct", ascending=True),
                x="demand_lift_pct",
                y="pickup_zone",
                color="weather_condition",
                color_discrete_map=WEATHER_COLOR_MAP,
                orientation="h",
                title=f"Top {top_n} Zones by Demand Lift",
                hover_name="pickup_zone",
                hover_data=existing_columns(
                    top_lift,
                    ["pickup_borough", "total_trips", "demand_lift_pct"],
                ),
            )
            fig_top_lift.update_layout(
                xaxis_title="Demand Lift (%)",
                yaxis_title="Pickup Zone",
                yaxis={"categoryorder": "total ascending"},
                
            )
            st.plotly_chart(update_chart_layout(fig_top_lift, height=470), width='stretch')
            
        with col_middle:
            st.markdown("### Top Fare Amount Impact Zones")

            top_duration = (
                filtered_zone_elasticity
                .sort_values("fare_delta_amount", ascending=False)
                .head(top_n)
            )

            fig_top_duration = px.bar(
                top_duration.sort_values("fare_delta_amount", ascending=True),
                x="fare_delta_amount",
                y="pickup_zone",
                color="weather_condition",
                color_discrete_map=WEATHER_COLOR_MAP,
                orientation="h",
                title=f"Top {top_n} Zones by Fare Amount Impact",
                hover_name="pickup_zone",
                hover_data=existing_columns(
                    top_duration,
                    ["pickup_borough", "total_trips", "duration_delta_minutes"],
                ),
            )
            fig_top_duration.update_layout(
                xaxis_title="Fare Amount Delta (minutes)",
                yaxis_title="Pickup Zone",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(update_chart_layout(fig_top_duration, height=470), width='stretch')

        with col_right:
            st.markdown("### Top Duration Impact Zones")

            top_duration = (
                filtered_zone_elasticity
                .sort_values("duration_delta_minutes", ascending=False)
                .head(top_n)
            )

            fig_top_duration = px.bar(
                top_duration.sort_values("duration_delta_minutes", ascending=True),
                x="duration_delta_minutes",
                y="pickup_zone",
                color="weather_condition",
                color_discrete_map=WEATHER_COLOR_MAP,
                orientation="h",
                title=f"Top {top_n} Zones by Duration Impact",
                hover_name="pickup_zone",
                hover_data=existing_columns(
                    top_duration,
                    ["pickup_borough", "total_trips", "duration_delta_minutes"],
                ),
            )
            fig_top_duration.update_layout(
                xaxis_title="Duration Delta (minutes)",
                yaxis_title="Pickup Zone",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(update_chart_layout(fig_top_duration, height=470), width='stretch')
        
        st.markdown("### Zone Map — Weather Impact")

        taxi_zones_geojson = read_geojson(str(TAXI_ZONES_GEOJSON_PATH))

        if taxi_zones_geojson is None:
            st.warning(
                "File GeoJSON belum tersedia. Letakkan di `dashboard/assets/taxi_zones.geojson`."
            )
        elif filtered_zone_elasticity.empty:
            st.warning("Tidak ada data zona untuk peta.")
        elif "zone_id" not in filtered_zone_elasticity.columns:
            st.warning("Kolom zone_id tidak tersedia.")
        else:
            map_col1, map_col2, map_col3 = st.columns(3)

            with map_col1:
                # Available weather conditions (exclude clear — it's the baseline, delta ≈ 0)
                available_weathers = [
                    w for w in filtered_zone_elasticity["weather_condition"]
                    .astype(str).unique()
                    if w != "clear"
                ]
                if not available_weathers:
                    st.warning("Tidak ada data non-clear weather untuk dianalisis.")
                    st.stop()

                # Order them sensibly
                ordered_weathers = [w for w in WEATHER_ORDER if w in available_weathers]

                selected_map_weather = st.selectbox(
                    "Weather Condition",
                    ordered_weathers,
                    index=len(ordered_weathers) - 1 if "heavy_rain" not in ordered_weathers
                          else ordered_weathers.index("heavy_rain"),  # default to heavy_rain
                    key="map_weather_selector",
                )

            with map_col2:
                map_metric = st.selectbox(
                    "Impact Metric",
                    [
                        "Demand Lift (%)",
                        "Duration Delta (min)",
                        "Fare Delta (USD)",
                        "Tip Delta (%)",
                        "Total Trips",
                    ],
                    key="map_metric_selector",
                )

            with map_col3:
                map_level = st.selectbox(
                    "Aggregation Level",
                    ["Per Zone", "Per Borough"],
                    key="map_level_selector",
                )

            # ============================================================
            # METRIC CONFIG
            # ============================================================
            # (column, label, use_log_scale, diverging_scale, aggregation_func)
            metric_config = {
                "Demand Lift (%)":      ("demand_lift_pct",         "Demand Lift (%)",       False, True,  "mean"),
                "Duration Delta (min)": ("duration_delta_minutes",  "Duration Delta (min)",  False, True,  "mean"),
                "Fare Delta (USD)":     ("fare_delta_amount",       "Fare Delta (USD)",      False, True,  "mean"),
                "Tip Delta (%)":        ("tip_delta_pct",           "Tip Delta (%)",         False, True,  "mean"),
                "Total Trips":          ("total_trips",             "Total Trips",           True,  False, "sum"),
            }

            metric_col, metric_label, use_log_scale, is_diverging, agg_func = metric_config[map_metric]

            # ============================================================
            # FILTER TO ONE WEATHER CONDITION → one row per zone
            # ============================================================
            geomap_data = filtered_zone_elasticity[
                filtered_zone_elasticity["weather_condition"].astype(str) == selected_map_weather
            ].copy()

            if geomap_data.empty:
                st.warning(f"Tidak ada data untuk weather {selected_map_weather} pada filter ini.")
                st.stop()

            if metric_col not in geomap_data.columns:
                st.warning(f"Kolom `{metric_col}` tidak tersedia.")
                st.stop()

            # Clean and normalize IDs
            geomap_data["map_id"] = normalize_location_id_series(geomap_data["zone_id"])
            geomap_data = geomap_data.dropna(subset=["map_id"])
            geomap_data[metric_col] = pd.to_numeric(geomap_data[metric_col], errors="coerce")
            geomap_data = geomap_data.dropna(subset=[metric_col])

            # ============================================================
            # BOROUGH AGGREGATION (if requested)
            # ============================================================
            if map_level == "Per Borough":
                if "pickup_borough" not in geomap_data.columns:
                    st.warning("Kolom pickup_borough tidak tersedia.")
                    st.stop()

                # Aggregate metric per borough, then broadcast back to each zone
                # (this keeps zone-level GeoJSON polygons but colors them by borough value)
                borough_agg = (
                    geomap_data.groupby("pickup_borough")[metric_col]
                    .agg(agg_func)
                    .reset_index()
                    .rename(columns={metric_col: "display_value"})
                )
                geomap_data = geomap_data.merge(borough_agg, on="pickup_borough", how="left")
                hover_title = "pickup_borough"
            else:
                geomap_data["display_value"] = geomap_data[metric_col]
                hover_title = "pickup_zone" if "pickup_zone" in geomap_data.columns else "map_id"

            # ============================================================
            # COLOR SCALE CONFIG
            # ============================================================
            if use_log_scale:
                geomap_data["map_color_value"] = np.log1p(geomap_data["display_value"].clip(lower=0))
                color_scale = "Viridis"
                color_midpoint = None

                # Log colorbar ticks (show real numbers)
                max_val = geomap_data["display_value"].max()
                min_val = max(geomap_data["display_value"].min(), 1)
                log_min = np.floor(np.log10(min_val))
                log_max = np.ceil(np.log10(max_val))
                original_ticks = np.logspace(log_min, log_max, num=6)
                tick_vals = np.log1p(original_ticks)
                tick_text = [f"{v:,.0f}" for v in original_ticks]
                colorbar_title = f"{metric_label} (log)"
            else:
                geomap_data["map_color_value"] = geomap_data["display_value"]
                if is_diverging:
                    color_scale = "RdBu_r"     # red = negative impact, blue = positive
                    color_midpoint = 0          # CENTER on zero — critical for delta metrics
                else:
                    color_scale = "Viridis"
                    color_midpoint = None
                tick_vals, tick_text = None, None
                colorbar_title = metric_label

            # ============================================================
            # BUILD HOVER DATA — show context columns
            # ============================================================
            hover_columns = {
                "map_id": False,
                "map_color_value": False,
                "display_value": False,
                "pickup_zone": "pickup_zone" in geomap_data.columns,
                "pickup_borough": "pickup_borough" in geomap_data.columns,
                "weather_condition": False,  # already in title
            }

            # Surface the actual metric + complementary metrics
            complementary_cols = {
                "demand_lift_pct": ":.2f",
                "duration_delta_minutes": ":.2f",
                "fare_delta_amount": ":.2f",
                "tip_delta_pct": ":.2f",
                "total_trips": ":,",
            }
            for col, fmt in complementary_cols.items():
                if col in geomap_data.columns:
                    hover_columns[col] = fmt

            # ============================================================
            # RENDER MAP
            # ============================================================
            try:
                fig_map = px.choropleth_map(
                    geomap_data,
                    geojson=taxi_zones_geojson,
                    featureidkey="properties.LocationID",
                    locations="map_id",
                    color="map_color_value",
                    color_continuous_scale=color_scale,
                    color_continuous_midpoint=color_midpoint,
                    map_style="carto-darkmatter",
                    zoom=9,
                    center={"lat": 40.7128, "lon": -74.0060},
                    opacity=0.75,
                    hover_name=hover_title,
                    hover_data=hover_columns,
                    labels={"map_color_value": colorbar_title},
                )

                fig_map.update_layout(
                    margin={"r": 0, "t": 30, "l": 0, "b": 0},
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e5e7eb"),
                    title={
                        "text": f"{metric_label} during {selected_map_weather.replace('_', ' ').title()} — {title_borough}",
                        "font": {"color": "#f8fafc"},
                    },
                    coloraxis_colorbar=dict(
                        title=dict(text=colorbar_title, font=dict(color="#e5e7eb")),
                        tickfont=dict(color="#e5e7eb"),
                    ),
                )

                if use_log_scale and tick_vals is not None:
                    fig_map.update_coloraxes(
                        colorbar=dict(tickvals=tick_vals, ticktext=tick_text)
                    )

                st.plotly_chart(fig_map, use_container_width=True)

                # Helpful caption
                if is_diverging:
                    st.caption(
                        f"Color scale centered at 0. **Blue** = higher than clear-weather baseline, "
                        f"**red** = lower. Showing **{len(geomap_data)} zones** during **{selected_map_weather}**."
                    )

            except Exception as error:
                st.error(f"Gagal merender peta: {error}")
                st.exception(error)  # show traceback for debugging
        
        st.markdown("### Demand Lift vs Duration Impact by Zone")

        size_col = "total_trips" if "total_trips" in filtered_zone_elasticity.columns else None
        color_col = (
            "weather_sensitivity_label"
            if "weather_sensitivity_label" in filtered_zone_elasticity.columns
            else "weather_condition"
        )

        fig_zone_scatter = px.scatter(
            filtered_zone_elasticity,
            x="demand_lift_pct",
            y="duration_delta_minutes",
            size=size_col,
            color=color_col,
            hover_name="pickup_zone",
            hover_data=existing_columns(
                filtered_zone_elasticity,
                [
                    "pickup_borough",
                    "weather_condition",
                    "total_trips",
                    "minutes_per_mile_delta",
                    "fare_delta_amount",
                    "tip_delta_pct",
                ],
            ),
            title="Zone Elasticity Positioning",
        )
        fig_zone_scatter.update_layout(
            xaxis_title="Demand Lift (%)",
            yaxis_title="Duration Delta (minutes)",
        )
        st.plotly_chart(update_chart_layout(fig_zone_scatter, height=520), width='stretch')

        st.markdown("### Zone Elasticity Data")
        st.dataframe(
            safe_dataframe(
                filtered_zone_elasticity,
                [
                    "zone_id",
                    "pickup_location_id",
                    "pickup_borough",
                    "pickup_zone",
                    "weather_condition",
                    "total_trips",
                    "demand_lift_pct",
                    "duration_delta_minutes",
                    "minutes_per_mile_delta",
                    "fare_delta_amount",
                    "tip_delta_pct",
                    "weather_sensitivity_label",
                ],
            ),
            width='stretch',
        )


# ============================================================
# TAB 4: OD FLOW
# ============================================================

with tab_od:
    st.subheader("Origin-Destination Flow under Weather Conditions")
    st.markdown(
        f'<div class="section-note">Analisis rute origin-destination yang dominan pada kondisi cuaca tertentu. Filter aktif: <b>{title_borough}</b>.</div>',
        unsafe_allow_html=True,
    )

    if filtered_od_flow.empty:
        st.warning(f"Tidak ada OD flow untuk filter Borough = {selected_borough}.")
    else:
        od_display = filtered_od_flow.copy()

        origin_col = first_existing_column(od_display, ["origin_zone", "pickup_zone"])
        dest_col = first_existing_column(od_display, ["destination_zone", "dropoff_zone"])
        trip_col = first_existing_column(od_display, ["trip_count", "total_trips"])

        if origin_col is None or dest_col is None or trip_col is None:
            st.warning("Kolom OD flow tidak lengkap untuk visualisasi.")
        else:
            od_display["route"] = (
                od_display[origin_col].astype(str)
                + " → "
                + od_display[dest_col].astype(str)
            )

            top_routes = (
                od_display
                .sort_values(trip_col, ascending=False)
                .head(top_n)
            )

            fig_routes = px.bar(
                top_routes.sort_values(trip_col, ascending=True),
                x=trip_col,
                y="route",
                color="destination_borough",
                color_discrete_map=BOROUGH_COLOR_MAP,
                orientation="h",
                title=f"Top {top_n} OD Routes",
                hover_data=existing_columns(
                    top_routes,
                    [
                        "origin_borough",
                        "destination_borough",
                        "avg_duration",
                        "avg_trip_distance",
                        "avg_speed_mph",
                        "total_revenue",
                        "avg_total_amount",
                    ],
                ),
            )
            fig_routes.update_layout(
                xaxis_title="Trip Count",
                yaxis_title="Route",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(update_chart_layout(fig_routes, height=540), width='stretch')

            st.markdown("### OD Flow Table")
            st.dataframe(
                safe_dataframe(
                    top_routes,
                    [
                        "origin_borough",
                        "origin_zone",
                        "destination_borough",
                        "destination_zone",
                        "weather_condition",
                        trip_col,
                        "avg_duration",
                        "avg_trip_distance",
                        "avg_speed_mph",
                        "total_revenue",
                        "avg_total_amount",
                        "avg_tip_pct",
                    ],
                ),
                width='stretch',
            )


# ============================================================
# TAB 5: DEMAND PREDICTION
# ============================================================

with tab_prediction:
    st.subheader("ML Demand Prediction")
    st.markdown(
        '<div class="section-note">Random Forest digunakan untuk memprediksi jumlah trip per zona dan jam berdasarkan fitur waktu, lokasi, dan cuaca.</div>',
        unsafe_allow_html=True,
    )

    metrics = demand_metrics.get("metrics", {})
    baseline_metrics = demand_metrics.get("baseline_metrics", {})

    mae = metrics.get("mae", 0)
    rmse = metrics.get("rmse", 0)
    r2 = metrics.get("r2", 0)
    mape = metrics.get("mape_percent", 0)

    baseline_mae = baseline_metrics.get("baseline_mean_mae", 0)
    baseline_rmse = baseline_metrics.get("baseline_mean_rmse", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f}", delta=f"Baseline {baseline_mae:.2f}", delta_color="inverse")
    col2.metric("RMSE", f"{rmse:.2f}", delta=f"Baseline {baseline_rmse:.2f}", delta_color="inverse")
    col3.metric("R²", f"{r2:.3f}")
    col4.metric("MAPE", f"{mape:.2f}%")

    st.divider()

    actual_col, predicted_col, abs_error_col = get_prediction_columns(filtered_demand_results)

    if filtered_demand_results.empty or actual_col is None or predicted_col is None:
        st.warning("Data hasil prediksi tidak tersedia untuk filter yang dipilih.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Actual vs Predicted Daily Demand")

            if "pickup_date" in filtered_demand_results.columns:
                daily_prediction = (
                    filtered_demand_results.groupby("pickup_date", as_index=False)
                    .agg(
                        actual_total_trips=(actual_col, "sum"),
                        predicted_total_trips=(predicted_col, "sum"),
                    )
                    .sort_values("pickup_date")
                )

                daily_prediction_long = daily_prediction.melt(
                    id_vars="pickup_date",
                    value_vars=["actual_total_trips", "predicted_total_trips"],
                    var_name="series",
                    value_name="trips",
                )

                fig_prediction_daily = px.line(
                    daily_prediction_long,
                    x="pickup_date",
                    y="trips",
                    color="series",
                    markers=True,
                    title="Actual vs Predicted Total Trips by Date",
                )
                fig_prediction_daily.update_layout(
                    xaxis_title="Pickup Date",
                    yaxis_title="Trips",
                    hovermode="x unified",
                )
                st.plotly_chart(update_chart_layout(fig_prediction_daily, height=430), width='stretch')
            else:
                st.warning("Kolom pickup_date tidak tersedia pada prediction results.")

        with col_right:
            st.markdown("### Feature Importance")

            if feature_importance_df.empty:
                st.warning("Feature importance tidak tersedia.")
            else:
                feature_display = feature_importance_df.sort_values("importance", ascending=False).head(15)

                fig_importance = px.bar(
                    feature_display.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Top Feature Importance",
                )
                fig_importance.update_layout(
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                )
                st.plotly_chart(update_chart_layout(fig_importance, height=430), width='stretch')

        st.markdown("### Prediction Error Analysis")

        col_left, col_right = st.columns(2)

        with col_left:
            color_col = "weather_condition" if "weather_condition" in filtered_demand_results.columns else None

            fig_error_scatter = px.scatter(
                filtered_demand_results,
                x=actual_col,
                y=predicted_col,
                color=color_col,
                color_discrete_map=WEATHER_COLOR_MAP,
                title="Actual vs Predicted Trips",
                hover_data=existing_columns(
                    filtered_demand_results,
                    ["pickup_date", "pickup_hour", "pickup_borough", "pickup_zone", "weather_condition"],
                ),
            )
            fig_error_scatter.update_layout(
                xaxis_title="Actual Trips",
                yaxis_title="Predicted Trips",
            )
            st.plotly_chart(update_chart_layout(fig_error_scatter, height=430), width='stretch')

        with col_right:
            if abs_error_col is None:
                error_df = filtered_demand_results.copy()
                error_df["absolute_error"] = (
                    error_df[actual_col] - error_df[predicted_col]
                ).abs()
                abs_error_col = "absolute_error"
            else:
                error_df = filtered_demand_results.copy()

            fig_error_hist = px.histogram(
                error_df,
                x=abs_error_col,
                nbins=60,
                title="Distribution of Absolute Prediction Error",
            )
            fig_error_hist.update_layout(
                xaxis_title="Absolute Error",
                yaxis_title="Row Count",
            )
            st.plotly_chart(update_chart_layout(fig_error_hist, height=430), width='stretch')

        with st.expander("Prediction Results Sample"):
            st.dataframe(
                safe_dataframe(
                    filtered_demand_results.head(100),
                    [
                        "pickup_date",
                        "pickup_hour",
                        "pickup_borough",
                        "pickup_zone",
                        "weather_condition",
                        actual_col,
                        predicted_col,
                        abs_error_col,
                    ],
                ),
                width='stretch',
            )


# ============================================================
# TAB 6: ZONE CLUSTERING
# ============================================================

with tab_cluster:
    st.subheader("ML Zone Weather Sensitivity Clustering")
    st.markdown(
        '<div class="section-note">KMeans digunakan untuk mengelompokkan zona berdasarkan demand lift, duration impact, fare/tip delta, precipitation, dan volume trip saat cuaca buruk.</div>',
        unsafe_allow_html=True,
    )

    n_clusters = cluster_summary.get("n_clusters", filtered_clusters["cluster_id"].nunique() if "cluster_id" in filtered_clusters.columns else 0)
    silhouette_score = cluster_summary.get("silhouette_score", 0)
    n_samples = cluster_summary.get("n_samples", len(zone_clusters_df))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Zones", format_number(n_samples))
    col2.metric("Clusters", format_number(n_clusters))
    col3.metric("Silhouette Score", f"{silhouette_score:.3f}")

    st.divider()

    if filtered_clusters.empty:
        st.warning("Tidak ada data cluster untuk filter yang dipilih.")
    else:
        cluster_col = "cluster_profile" if "cluster_profile" in filtered_clusters.columns else "cluster_id"

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Cluster Distribution")

            cluster_dist = (
                filtered_clusters.groupby(cluster_col, as_index=False)
                .agg(zone_count=("pickup_zone", "nunique") if "pickup_zone" in filtered_clusters.columns else ("cluster_id", "size"))
                .sort_values("zone_count", ascending=False)
            )

            fig_cluster_dist = px.bar(
                cluster_dist,
                x=cluster_col,
                y="zone_count",
                title="Number of Zones by Cluster Profile",
            )
            fig_cluster_dist.update_layout(
                xaxis_title="Cluster Profile",
                yaxis_title="Zone Count",
            )
            st.plotly_chart(update_chart_layout(fig_cluster_dist, height=430), width='stretch')

        with col_right:
            st.markdown("### Cluster Summary")

            cluster_summary_df = pd.DataFrame(cluster_summary.get("cluster_summary", []))

            if cluster_summary_df.empty:
                st.warning("Cluster summary JSON tidak tersedia.")
            else:
                st.dataframe(
                    safe_dataframe(
                        cluster_summary_df,
                        [
                            "cluster_id",
                            "cluster_profile",
                            "zone_count",
                            "total_weather_trips",
                            "avg_demand_lift_pct",
                            "avg_duration_delta_minutes",
                            "avg_fare_delta_amount",
                            "avg_tip_delta_pct",
                            "high_sensitive_condition_count",
                        ],
                    ),
                    width='stretch',
                )

        st.markdown("### Cluster Positioning")

        x_col = first_existing_column(filtered_clusters, ["avg_demand_lift_pct", "demand_lift_pct"])
        y_col = first_existing_column(filtered_clusters, ["avg_duration_delta_minutes", "duration_delta_minutes"])
        size_col = first_existing_column(filtered_clusters, ["total_weather_trips", "total_trips"])

        if x_col is None or y_col is None:
            st.warning("Kolom demand lift atau duration delta tidak tersedia untuk scatter cluster.")
        else:
            fig_cluster_scatter = px.scatter(
                filtered_clusters,
                x=x_col,
                y=y_col,
                size=size_col,
                color=cluster_col,
                hover_name="pickup_zone" if "pickup_zone" in filtered_clusters.columns else None,
                hover_data=existing_columns(
                    filtered_clusters,
                    [
                        "pickup_borough",
                        "total_weather_trips",
                        "avg_fare_delta_amount",
                        "avg_tip_delta_pct",
                        "avg_precipitation",
                    ],
                ),
                title="Demand Lift vs Duration Impact by Cluster",
            )
            fig_cluster_scatter.update_layout(
                xaxis_title="Average Demand Lift (%)",
                yaxis_title="Average Duration Delta (minutes)",
            )
            st.plotly_chart(update_chart_layout(fig_cluster_scatter, height=520), width='stretch')

        st.markdown("### High-Impact Zone Candidates")

        candidate_df = filtered_clusters.copy()

        if "cluster_profile" in candidate_df.columns:
            candidate_df = candidate_df[
                candidate_df["cluster_profile"].astype(str).str.contains(
                    "high|lift|demand|sensitive",
                    case=False,
                    na=False,
                )
            ]

        sort_col = first_existing_column(candidate_df, ["avg_demand_lift_pct", "total_weather_trips", "total_trips"])

        if sort_col is not None and not candidate_df.empty:
            candidate_df = candidate_df.sort_values(sort_col, ascending=False).head(top_n)

            st.dataframe(
                safe_dataframe(
                    candidate_df,
                    [
                        "pickup_borough",
                        "pickup_zone",
                        "cluster_id",
                        "cluster_profile",
                        "total_weather_trips",
                        "avg_demand_lift_pct",
                        "avg_duration_delta_minutes",
                        "avg_fare_delta_amount",
                        "avg_tip_delta_pct",
                    ],
                ),
                width='stretch',
            )
        else:
            st.info("Tidak ada candidate high-impact zone pada filter saat ini.")

        with st.expander("All Clustered Zones"):
            st.dataframe(
                safe_dataframe(
                    filtered_clusters,
                    [
                        "pickup_borough",
                        "pickup_zone",
                        "cluster_id",
                        "cluster_profile",
                        "total_weather_trips",
                        "avg_demand_lift_pct",
                        "avg_duration_delta_minutes",
                        "avg_fare_delta_amount",
                        "avg_tip_delta_pct",
                    ],
                ),
                width='stretch',
            )


st.divider()
st.caption(
    "Built with Streamlit, DuckDB, Apache Airflow, Parquet, and scikit-learn. "
    "Pipeline period: January–March 2025."
)