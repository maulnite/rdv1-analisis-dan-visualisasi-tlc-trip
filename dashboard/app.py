from pathlib import Path
import json

import duckdb
import pandas as pd
import plotly.express as px
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

WEATHER_ORDER = ["clear", "light_rain", "moderate_rain", "heavy_rain", "snow"]

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
            padding-top: 3.2rem !important;
            padding-bottom: 2rem;
        }

        .main-title {
            font-size: 2.2rem;
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
        st.info("Jalankan DAG `tlc_full_pipeline` di Airflow terlebih dahulu.")
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


def format_number(value: float, decimal:bool = False) -> str:
    try:
        if decimal:
            return f"{float(value):,.2f}"
        else:
            return f"{float(value):,.0f}"
    except Exception:
        return "0"


def format_currency(value: float) -> str:
    try:
        if value >= 1_000_000_000:
            return f"${value/1_000_000_000:.2f}B"
        elif value >= 1_000_000:
            return f"${value/1_000_000:.2f}M"
        elif value >= 1_000:
            return f"${value/1_000:.2f}K"
        else:
            return f"${value:,.2f}"
    except Exception:
        return "$0.0"


def weighted_average(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if df.empty:
        return 0.0

    if value_col not in df.columns or weight_col not in df.columns:
        return 0.0

    valid = df[[value_col, weight_col]].dropna()

    if valid.empty or valid[weight_col].sum() == 0:
        return 0.0

    return float((valid[value_col] * valid[weight_col]).sum() / valid[weight_col].sum())


def apply_weather_order(df: pd.DataFrame, column: str = "weather_condition") -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        df[column] = pd.Categorical(
            df[column],
            categories=WEATHER_ORDER,
            ordered=True,
        )
        df = df.sort_values(column)
    return df


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def safe_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[existing_columns(df, columns)]


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


def weighted_median(df, value_col, weight_col):
    df_sorted = df.sort_values(value_col)
    cumulative_weight = df_sorted[weight_col].cumsum()
    cutoff = df_sorted[weight_col].sum() / 2.0
    return df_sorted.loc[cumulative_weight >= cutoff, value_col].iloc[0]


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

if "pickup_date" in daily_df.columns:
    daily_df["pickup_date"] = pd.to_datetime(daily_df["pickup_date"])

if "pickup_date" in weather_hourly_df.columns:
    weather_hourly_df["pickup_date"] = pd.to_datetime(weather_hourly_df["pickup_date"])

if "pickup_date" in demand_results_df.columns:
    demand_results_df["pickup_date"] = pd.to_datetime(demand_results_df["pickup_date"])

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

borough_options = sorted(
    zone_elasticity_df["pickup_borough"].dropna().astype(str)
    # .replace({"EWR": "Newark Liberty International Airport"})
    .unique().tolist()
)

selected_borough = st.sidebar.selectbox(
    "Borough",
    ["All"] + borough_options,
)

weather_options = [
    weather for weather in WEATHER_ORDER
    if weather in zone_elasticity_df["weather_condition"].astype(str).unique()
]

selected_weather = st.sidebar.multiselect(
    "Weather Condition",
    weather_options,
    default=weather_options,
)

active_weather_conditions = selected_weather if selected_weather else weather_options

top_n = st.sidebar.slider(
    "Top Pickup Zones",
    min_value=5,
    max_value=100,
    value=10,
    step=5,
)

min_zone_trips = st.sidebar.slider(
    "Min Zone-Weather Trips",
    min_value=0,
    max_value=500,
    value=50,
    step=25,
)

days = st.sidebar.selectbox(
    "Days",
    ["All Days", "Weekdays", "Weekends", "Comparison"],
)

title_borough = "All Boroughs" if selected_borough == "All" else selected_borough

BOROUGH_COLOR_MAP = { # Plotly default pallete
    "Manhattan": "#636EFA",   # blue
    "Brooklyn": "#EF553B",    # red
    "Queens": "#00CC96",      # green
    "Bronx": "#AB63FA",       # purple
    "Staten Island": "#FFA15A", # orange
    "EWR": "#19D3F3" # light blue
}

# ============================================================
# FILTERED DATASETS
# ============================================================

def filter_by_value(df, column, value):
    if value == "All":
        return df
    return df[df[column] == value]


def filter_by_list(df, column, values):
    if not values:
        return df
    return df[df[column].isin(values)]

filtered_zone_elasticity = zone_elasticity_df.copy()
filtered_od_flow = od_flow_df.copy()
filtered_clusters = zone_clusters_df.copy()
filtered_demand_results = demand_results_df.copy()
filtered_weather_hourly = weather_hourly_df.copy()
filtered_zone_summary = zone_df.copy()

if selected_borough != "All":
    # Borough filtering
    filtered_zone_elasticity = filter_by_value(zone_elasticity_df, "pickup_borough", selected_borough)
    filtered_clusters = filter_by_value(zone_clusters_df, "pickup_borough", selected_borough)
    filtered_od_flow = filter_by_value(od_flow_df, "origin_borough", selected_borough)
    filtered_demand_results = filter_by_value(demand_results_df, "pickup_borough", selected_borough)
    filtered_weather_hourly = filter_by_value(weather_hourly_df, "pickup_borough", selected_borough)
    filtered_zone_summary = filter_by_value(zone_df, "pickup_borough", selected_borough)

if active_weather_conditions:
    # Weather filtering
    filtered_zone_elasticity = filter_by_list(filtered_zone_elasticity, "weather_condition", active_weather_conditions)
    filtered_od_flow = filter_by_list(filtered_od_flow, "weather_condition", active_weather_conditions)
    filtered_demand_results = filter_by_list(filtered_demand_results, "weather_condition", active_weather_conditions)

if days in ["Weekdays", "Weekends"]:
    is_weekend_flag = days == "Weekends"
    filtered_weather_hourly = filtered_weather_hourly[
        filtered_weather_hourly["is_weekend"] == is_weekend_flag
    ]
elif days == "Comparison":
    weekday_filtered_weather_hourly = filtered_weather_hourly[
        filtered_weather_hourly["is_weekend"] == False
    ]
    weekend_filtered_weather_hourly = filtered_weather_hourly[
        filtered_weather_hourly["is_weekend"] == True
    ]
    
filtered_zone_elasticity = filtered_zone_elasticity[
    filtered_zone_elasticity["total_trips"] >= min_zone_trips
]

# print(filtered_weather_hourly)

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
        f'<div class="section-note">Overview dari taxi demand, revenue, fare, dan durasi perjalanan untuk filter: <b>{title_borough}</b>.</div>',
        unsafe_allow_html=True,
    )

    overview_source = filtered_zone_summary.copy()

    total_trips = int(overview_source["total_trips"].sum()) if "total_trips" in overview_source.columns else 0
    num_days = filtered_weather_hourly["pickup_date"].nunique()
    avg_trips_per_day = total_trips / num_days if num_days > 0 else 0
    total_revenue = float(overview_source["total_revenue"].sum()) if "total_revenue" in overview_source.columns else 0.0
    avg_fare = weighted_average(overview_source, "avg_total_amount", "total_trips")
    median_fare = weighted_median(overview_source, "avg_total_amount", "total_trips")
    avg_duration = weighted_average(overview_source, "avg_trip_duration_minutes", "total_trips")
    median_duration = weighted_median(overview_source, "avg_trip_duration_minutes", "total_trips")
    avg_distance = weighted_average(overview_source, "avg_trip_distance", "total_trips")
    median_distance = weighted_median(overview_source, "avg_trip_distance", "total_trips")
    
    rainy_share = (
        (filtered_weather_hourly.groupby("pickup_date")["rain"].mean() > 0.1)
        .mean() * 100
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(
        "Total Trips", 
        format_number(total_trips),
        delta=f"Average: {format_number(avg_trips_per_day, True)}",
        delta_color="off",
    )
    col2.metric("Total Revenue (USD)", format_currency(total_revenue))
    col3.metric(
        "Fare (USD)",
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
        f"{avg_distance:.2f} miles",
        delta=f"Median: {median_distance:.2f} miles",
        delta_color="off",
    )
    col6.metric("Rainy Day Share", f"{rainy_share:.1f}%")

    st.divider()

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        st.markdown("### Daily Trip Trend")

        if filtered_weather_hourly.empty:
            st.warning("Tidak ada data daily trend untuk filter yang dipilih.")
        else:
            daily_plot = (
                filtered_weather_hourly.groupby("pickup_date", as_index=False)
                .agg(
                    total_trips=("total_trips", "sum"),
                    avg_precipitation=("precipitation", "mean"),
                )
                .sort_values("pickup_date")
            )
            
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig_daily = make_subplots(specs=[[{"secondary_y": True}]])
            fig_daily.add_trace(
                go.Scatter(x=daily_plot["pickup_date"], y=daily_plot["total_trips"],
                           mode="lines+markers", name="Total Trips"),
                secondary_y=False,
            )
            fig_daily.add_trace(
                go.Bar(x=daily_plot["pickup_date"], y=daily_plot["avg_precipitation"],
                       name="Precipitation", opacity=0.4),
                secondary_y=True,
            )
            fig_daily.update_yaxes(title_text="Total Trips", secondary_y=False)
            fig_daily.update_yaxes(title_text="Precipitation (mm)", secondary_y=True)

            fig_daily = px.line(
                daily_plot,
                x="pickup_date",
                y="total_trips",
                color_discrete_map=BOROUGH_COLOR_MAP,
                markers=True,
                title=f"Daily Total Trips - {title_borough}",
            )

            fig_daily.update_layout(
                xaxis_title="Pickup Date",
                yaxis_title="Total Trips",
                hovermode="x unified",
            )

            fig_daily.update_xaxes(
                range=[daily_plot["pickup_date"].min(), daily_plot["pickup_date"].max()],
                tickformat="%d %b %Y",
            )

            st.plotly_chart(update_chart_layout(fig_daily, height=430), width='stretch')

    with col_right:
        st.markdown("### Top Pickup Zones")

        if filtered_zone_summary.empty:
            st.warning("Tidak ada top zone untuk filter yang dipilih.")
        else:
            top_zones = (
                filtered_zone_summary.sort_values("total_trips", ascending=False)
                .head(top_n)
            )

            fig_zone = px.bar(
                top_zones,
                x="total_trips",
                y="pickup_zone",
                color="pickup_borough",
                color_discrete_map=BOROUGH_COLOR_MAP,
                orientation="h",
                title=f"Top {top_n} Pickup Zones - {title_borough}",
                hover_data=existing_columns(
                    top_zones,
                    ["total_revenue", "avg_total_amount", "avg_trip_duration_minutes"],
                ),
            )
            
            if selected_borough != "All":
                fig_zone.update_layout(showlegend=False)

            fig_zone.update_layout(
                xaxis_title="Total Trips",
                yaxis_title="Pickup Zone",
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(update_chart_layout(fig_zone, height=430), width='stretch')

    st.divider()
    
    st.markdown("### Hourly Demand Pattern")

    if filtered_weather_hourly.empty:
        st.warning("Tidak ada hourly demand untuk filter yang dipilih.")
    else:
        if days == "Comparison":
            # Weekday aggregation
            weekday_hourly_pattern = (
                weekday_filtered_weather_hourly
                .groupby("pickup_hour", as_index=False)
                .agg(total_trips=("total_trips", "sum"))
            )
            weekday_hourly_pattern["day_type"] = "Weekday"

            # Weekend aggregation
            weekend_hourly_pattern = (
                weekend_filtered_weather_hourly
                .groupby("pickup_hour", as_index=False)
                .agg(total_trips=("total_trips", "sum"))
            )
            weekend_hourly_pattern["day_type"] = "Weekend"

            comparison_df = pd.concat(
                [weekday_hourly_pattern, weekend_hourly_pattern],
                ignore_index=True
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
                ticktext=[f"{h:02d}.00" for h in range(24)],
                range=[0, 23],
            ),
            yaxis_title="Total Trips",
            hovermode="x unified",
        )

        st.plotly_chart(
            update_chart_layout(fig_hourly, height=440),
            width='stretch'
        )


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
    weather_display["weather_condition"] = weather_display["weather_condition"].astype(str)

    if active_weather_conditions:
        weather_display = weather_display[
            weather_display["weather_condition"].isin(active_weather_conditions)
        ]

    if weather_display.empty:
        st.warning("Tidak ada data weather impact untuk filter cuaca yang dipilih.")
    else:
        col1, col2, col3 = st.columns(3)

        best_condition = weather_display.sort_values("demand_lift_pct", ascending=False).iloc[0]
        worst_duration = weather_display.sort_values("duration_delta_minutes", ascending=False).iloc[0]
        highest_volume_condition = weather_display.sort_values("total_trips", ascending=False).iloc[0]

        col1.metric(
            "Highest Demand Lift",
            f"{best_condition['weather_condition']}",
            f"{best_condition['demand_lift_pct']:.2f}%",
        )

        col2.metric(
            "Highest Duration Impact",
            f"{worst_duration['weather_condition']}",
            f"{worst_duration['duration_delta_minutes']:.2f} min",
        )

        col3.metric(
            "Largest Trip Volume",
            f"{highest_volume_condition['weather_condition']}",
            format_number(highest_volume_condition["total_trips"]),
        )

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            fig_lift = px.bar(
                weather_display,
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

        with col_right:
            fig_duration = px.bar(
                weather_display,
                x="weather_condition",
                y="duration_delta_minutes",
                title="Duration Delta by Weather Condition",
                hover_data=existing_columns(
                    weather_display,
                    ["avg_duration", "avg_minutes_per_mile"],
                ),
            )

            fig_duration.update_layout(
                xaxis_title="Weather Condition",
                yaxis_title="Duration Delta (minutes)",
            )

            st.plotly_chart(update_chart_layout(fig_duration, height=430), width='stretch')

        st.markdown("### Weather Trade-Off: Demand Lift vs Duration Impact")

        fig_weather_scatter = px.scatter(
            weather_display,
            x="demand_lift_pct",
            y="duration_delta_minutes",
            size="total_trips",
            color="weather_condition",
            hover_name="weather_condition",
            hover_data=existing_columns(
                weather_display,
                [
                    "avg_total_amount",
                    "fare_delta_amount",
                    "avg_tip_pct",
                    "tip_delta_pct",
                ],
            ),
            title="Weather Impact Positioning",
        )

        fig_weather_scatter.update_layout(
            xaxis_title="Demand Lift (%)",
            yaxis_title="Duration Delta (minutes)",
        )

        st.plotly_chart(update_chart_layout(fig_weather_scatter, height=520), width='stretch')

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
        f'<div class="section-note">Identifikasi zona yang paling sensitif terhadap cuaca. Filter aktif: <b>{title_borough}</b>, minimum trip per zone-weather: <b>{min_zone_trips}</b>.</div>',
        unsafe_allow_html=True,
    )

    if filtered_zone_elasticity.empty:
        st.warning("Tidak ada data untuk filter yang dipilih. Coba turunkan Min Zone-Weather Trips.")
    else:
        non_clear_zone = filtered_zone_elasticity[
            filtered_zone_elasticity["weather_condition"].astype(str) != "clear"
        ].copy()

        if non_clear_zone.empty:
            st.warning("Tidak ada data non-clear weather untuk filter yang dipilih.")
        else:
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("### Top Demand Lift Zones")

                top_demand_lift = (
                    non_clear_zone.sort_values("demand_lift_pct", ascending=False)
                    .head(top_n)
                )
                
                # print(top_demand_lift)

                fig_top_lift = px.bar(
                    top_demand_lift,
                    x="demand_lift_pct",
                    y="pickup_zone",
                    color="weather_condition",
                    orientation="h",
                    title=f"Top {top_n} Zones by Demand Lift",
                    hover_data=existing_columns(
                        top_demand_lift,
                        [
                            "pickup_borough",
                            "total_trips",
                            "duration_delta_minutes",
                            "weather_sensitivity_label",
                        ],
                    ),
                )

                fig_top_lift.update_layout(
                    xaxis_title="Demand Lift (%)",
                    yaxis_title="Pickup Zone",
                    yaxis={"categoryorder": "total ascending"},
                )

                st.plotly_chart(update_chart_layout(fig_top_lift, height=470), width='stretch')

            with col_right:
                st.markdown("### Top Duration Impact Zones")

                top_duration = (
                    non_clear_zone.sort_values("duration_delta_minutes", ascending=False)
                    .head(top_n)
                )

                fig_top_duration = px.bar(
                    top_duration,
                    x="duration_delta_minutes",
                    y="pickup_zone",
                    color="weather_condition",
                    orientation="h",
                    title=f"Top {top_n} Zones by Duration Impact",
                    hover_data=existing_columns(
                        top_duration,
                        [
                            "pickup_borough",
                            "total_trips",
                            "demand_lift_pct",
                            "weather_sensitivity_label",
                        ],
                    ),
                )

                fig_top_duration.update_layout(
                    xaxis_title="Duration Delta (minutes)",
                    yaxis_title="Pickup Zone",
                    yaxis={"categoryorder": "total ascending"},
                )

                st.plotly_chart(update_chart_layout(fig_top_duration, height=470), width='stretch')

            st.markdown("### Demand Lift vs Duration Impact by Zone")

            fig_zone_scatter = px.scatter(
                non_clear_zone,
                x="demand_lift_pct",
                y="duration_delta_minutes",
                size="total_trips",
                color="weather_sensitivity_label",
                hover_name="pickup_zone",
                hover_data=existing_columns(
                    non_clear_zone,
                    [
                        "pickup_borough",
                        "weather_condition",
                        "avg_total_amount",
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

            st.plotly_chart(update_chart_layout(fig_zone_scatter, height=560), width='stretch')

            st.markdown("### Zone Elasticity Data")
            st.dataframe(
                safe_dataframe(
                    non_clear_zone.sort_values("demand_lift_pct", ascending=False),
                    [
                        "zone_id",
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
        st.warning("Tidak ada OD flow untuk filter yang dipilih.")
    else:
        od_display = filtered_od_flow.copy()
        od_display["route"] = od_display["origin_zone"] + " → " + od_display["destination_zone"]

        top_routes = od_display.sort_values("trip_count", ascending=False).head(top_n)

        fig_routes = px.bar(
            top_routes,
            x="trip_count",
            y="route",
            color="weather_condition",
            orientation="h",
            title=f"Top {top_n} OD Routes",
            hover_data=existing_columns(
                top_routes,
                [
                    "origin_borough",
                    "destination_borough",
                    "avg_duration",
                    "avg_trip_distance",
                    "total_revenue",
                    "avg_tip_pct",
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
                    "trip_count",
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
# TAB 5: ML DEMAND PREDICTION
# ============================================================

with tab_prediction:
    st.subheader("ML Demand Prediction")
    st.markdown(
        f'<div class="section-note">Model Random Forest digunakan untuk memprediksi jumlah trip per zona dan jam berdasarkan fitur waktu, lokasi, dan cuaca. Grafik prediction mengikuti filter: <b>{title_borough}</b>.</div>',
        unsafe_allow_html=True,
    )

    metrics = demand_metrics.get("metrics", {})
    baseline = demand_metrics.get("baseline_metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "MAE",
        f"{metrics.get('mae', 0):.2f}",
        f"Baseline {baseline.get('baseline_mean_mae', 0):.2f}",
    )
    col2.metric(
        "RMSE",
        f"{metrics.get('rmse', 0):.2f}",
        f"Baseline {baseline.get('baseline_mean_rmse', 0):.2f}",
    )
    col3.metric("R²", f"{metrics.get('r2', 0):.3f}")
    col4.metric("MAPE", f"{metrics.get('mape_percent', 0):.2f}%")

    st.caption(
        "Metric model di atas adalah hasil evaluasi global pada test set Maret 2025. "
        "Chart di bawah mengikuti filter sidebar."
    )

    st.divider()

    if filtered_demand_results.empty:
        st.warning("Tidak ada prediction result untuk filter yang dipilih.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Actual vs Predicted Daily Demand")

            pred_daily = (
                filtered_demand_results.groupby("pickup_date", as_index=False)
                .agg(
                    actual_total_trips=("total_trips", "sum"),
                    predicted_total_trips=("predicted_total_trips", "sum"),
                )
                .sort_values("pickup_date")
            )

            pred_daily_long = pred_daily.melt(
                id_vars="pickup_date",
                value_vars=["actual_total_trips", "predicted_total_trips"],
                var_name="series",
                value_name="trips",
            )

            fig_pred_daily = px.line(
                pred_daily_long,
                x="pickup_date",
                y="trips",
                color="series",
                markers=True,
                title="Actual vs Predicted Total Trips by Date",
            )

            fig_pred_daily.update_layout(
                xaxis_title="Pickup Date",
                yaxis_title="Trips",
                hovermode="x unified",
            )

            st.plotly_chart(update_chart_layout(fig_pred_daily, height=450), width='stretch')

        with col_right:
            st.markdown("### Feature Importance")

            top_features = (
                feature_importance_df.sort_values("importance", ascending=False)
                .head(15)
            )

            fig_importance = px.bar(
                top_features,
                x="importance",
                y="feature",
                orientation="h",
                title="Top Feature Importance",
            )

            fig_importance.update_layout(
                xaxis_title="Importance",
                yaxis_title="Feature",
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(update_chart_layout(fig_importance, height=450), width='stretch')

        st.markdown("### Prediction Error Analysis")

        col_left, col_right = st.columns(2)

        with col_left:
            sample_df = filtered_demand_results.sample(
                min(5000, len(filtered_demand_results)),
                random_state=42,
            )

            fig_actual_pred = px.scatter(
                sample_df,
                x="total_trips",
                y="predicted_total_trips",
                color="weather_condition",
                hover_data=existing_columns(
                    sample_df,
                    [
                        "pickup_date",
                        "pickup_hour",
                        "pickup_borough",
                        "pickup_zone",
                        "absolute_error",
                    ],
                ),
                title="Actual vs Predicted Trips",
            )

            fig_actual_pred.update_layout(
                xaxis_title="Actual Trips",
                yaxis_title="Predicted Trips",
            )

            st.plotly_chart(update_chart_layout(fig_actual_pred, height=500), width='stretch')

        with col_right:
            fig_error = px.histogram(
                filtered_demand_results,
                x="absolute_error",
                nbins=50,
                title="Distribution of Absolute Prediction Error",
            )

            fig_error.update_layout(
                xaxis_title="Absolute Error",
                yaxis_title="Row Count",
            )

            st.plotly_chart(update_chart_layout(fig_error, height=500), width='stretch')

        st.markdown("### Prediction Results Sample")

        st.dataframe(
            safe_dataframe(
                filtered_demand_results.sort_values("absolute_error", ascending=False).head(100),
                [
                    "pickup_date",
                    "pickup_hour",
                    "pickup_borough",
                    "pickup_zone",
                    "weather_condition",
                    "total_trips",
                    "predicted_total_trips",
                    "prediction_error",
                    "absolute_error",
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
        f'<div class="section-note">KMeans digunakan untuk mengelompokkan zona berdasarkan demand lift, duration impact, fare/tip delta, precipitation, dan volume trip saat cuaca buruk. Filter aktif: <b>{title_borough}</b>.</div>',
        unsafe_allow_html=True,
    )

    cluster_meta = pd.DataFrame(cluster_summary.get("cluster_summary", []))

    n_clusters = cluster_summary.get("n_clusters", zone_clusters_df["cluster_id"].nunique())
    silhouette = cluster_summary.get("silhouette_score", None)

    col1, col2, col3 = st.columns(3)

    col1.metric("Filtered Zones", format_number(len(filtered_clusters)))
    col2.metric("Global Clusters", format_number(n_clusters))
    col3.metric("Silhouette Score", f"{silhouette:.3f}" if silhouette is not None else "N/A")

    st.divider()

    if filtered_clusters.empty:
        st.warning("Tidak ada data cluster untuk filter yang dipilih.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Cluster Distribution")

            cluster_count = (
                filtered_clusters.groupby(["cluster_id", "cluster_profile"], as_index=False)
                .agg(zone_count=("zone_id", "count"))
                .sort_values("cluster_id")
            )

            fig_cluster_count = px.bar(
                cluster_count,
                x="cluster_profile",
                y="zone_count",
                color="cluster_profile",
                title="Number of Zones by Cluster Profile",
            )

            fig_cluster_count.update_layout(
                xaxis_title="Cluster Profile",
                yaxis_title="Zone Count",
                showlegend=False,
            )

            st.plotly_chart(update_chart_layout(fig_cluster_count, height=450), width='stretch')

        with col_right:
            st.markdown("### Cluster Summary")

            if not cluster_meta.empty:
                st.dataframe(
                    safe_dataframe(
                        cluster_meta,
                        [
                            "cluster_id",
                            "cluster_profile",
                            "zone_count",
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
                st.info("Cluster summary tidak tersedia.")

        st.markdown("### Cluster Positioning")

        fig_cluster_scatter = px.scatter(
            filtered_clusters,
            x="avg_demand_lift_pct",
            y="avg_duration_delta_minutes",
            size="total_weather_trips",
            color="cluster_profile",
            hover_name="pickup_zone",
            hover_data=existing_columns(
                filtered_clusters,
                [
                    "pickup_borough",
                    "avg_fare_delta_amount",
                    "avg_tip_delta_pct",
                    "avg_precipitation",
                    "total_weather_trips",
                ],
            ),
            title="Demand Lift vs Duration Impact by Cluster",
        )

        fig_cluster_scatter.update_layout(
            xaxis_title="Average Demand Lift (%)",
            yaxis_title="Average Duration Delta (minutes)",
        )

        st.plotly_chart(update_chart_layout(fig_cluster_scatter, height=560), width='stretch')

        st.markdown("### High-Volume High Demand Lift Zones")

        high_demand_cluster = filtered_clusters[
            filtered_clusters["cluster_profile"].str.contains(
                "High-volume high demand lift",
                case=False,
                na=False,
            )
        ].copy()

        if high_demand_cluster.empty:
            st.info("Tidak ada zona dengan cluster profile High-volume high demand lift pada filter ini.")
        else:
            top_cluster_zones = high_demand_cluster.sort_values(
                ["total_weather_trips", "avg_demand_lift_pct"],
                ascending=[False, False],
            ).head(top_n)

            fig_high_cluster = px.bar(
                top_cluster_zones,
                x="total_weather_trips",
                y="pickup_zone",
                color="pickup_borough",
                color_discrete_map=BOROUGH_COLOR_MAP,
                orientation="h",
                title=f"Top {top_n} High-Volume High Demand Lift Zones",
                hover_data=existing_columns(
                    top_cluster_zones,
                    [
                        "avg_demand_lift_pct",
                        "avg_duration_delta_minutes",
                        "avg_fare_delta_amount",
                        "avg_tip_delta_pct",
                    ],
                ),
            )

            fig_high_cluster.update_layout(
                xaxis_title="Total Weather Trips",
                yaxis_title="Pickup Zone",
                yaxis={"categoryorder": "total ascending"},
            )

            st.plotly_chart(update_chart_layout(fig_high_cluster, height=470), width='stretch')

        st.markdown("### Zone Cluster Data")

        st.dataframe(
            safe_dataframe(
                filtered_clusters.sort_values(
                    ["cluster_id", "total_weather_trips"],
                    ascending=[True, False],
                ),
                [
                    "zone_id",
                    "pickup_borough",
                    "pickup_zone",
                    "cluster_id",
                    "cluster_profile",
                    "total_weather_trips",
                    "avg_demand_lift_pct",
                    "avg_duration_delta_minutes",
                    "avg_minutes_per_mile_delta",
                    "avg_fare_delta_amount",
                    "avg_tip_delta_pct",
                    "avg_precipitation",
                ],
            ),
            width='stretch',
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Built with Streamlit, DuckDB, Apache Airflow, and scikit-learn. "
    "Pipeline period: January–March 2025."
)