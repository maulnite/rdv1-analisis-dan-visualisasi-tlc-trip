import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="NYC TLC Dashboard",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 NYC TLC Analytics Dashboard")
st.markdown("January - March 2025")

# =========================
# LOAD DATA
# =========================

daily_df = duckdb.sql("""
SELECT *
FROM '../data/curated/agg_daily_summary_2025-01_to_2025-03.parquet'
""").df()

zone_df = duckdb.sql("""
SELECT *
FROM '../data/curated/agg_zone_summary_2025-01_to_2025-03.parquet'
""").df()

hourly_df = duckdb.sql("""
SELECT *
FROM '../data/curated/agg_hourly_demand_2025-01_to_2025-03.parquet'
""").df()

weather_df = duckdb.sql("""
SELECT *
FROM '../data/curated/agg_hourly_demand_weather_2025-01_to_2025-03.parquet'
""").df()

print("DAILY DF")
print(daily_df.columns)

print("ZONE DF")
print(zone_df.columns)

print("WEATHER DF")
print(weather_df.columns)

# =========================
# KPI
# =========================

total_trips = int(daily_df["total_trips"].sum())
total_revenue = float(daily_df["total_revenue"].sum())
avg_fare = float(daily_df["avg_total_amount"].mean())
avg_distance = float(daily_df["avg_trip_distance"].mean())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Trips", f"{total_trips:,}")
col2.metric("Total Revenue", f"${total_revenue:,.2f}")
col3.metric("Average Fare", f"${avg_fare:.2f}")
col4.metric("Average Distance", f"{avg_distance:.2f} mi")

st.divider()

# =========================
# DAILY TREND
# =========================

st.subheader("Daily Trip Trend")

fig_daily = px.line(
    daily_df,
    x="pickup_date",
    y="total_trips"
)

st.plotly_chart(fig_daily, use_container_width=True)

# =========================
# TOP ZONES
# =========================

st.subheader("Top Pickup Zones")

top_zones = zone_df.sort_values(
    "total_trips",
    ascending=False
).head(10)

fig_zone = px.bar(
    top_zones,
    x="pickup_zone",
    y="total_trips",
    color="pickup_borough"
)

st.plotly_chart(fig_zone, use_container_width=True)

# =========================
# HOURLY DEMAND
# =========================

st.subheader("Hourly Demand Pattern")

fig_hourly = px.line(
    hourly_df,
    x="pickup_hour",
    y="total_trips"
)

st.plotly_chart(fig_hourly, use_container_width=True)

# =========================
# WEATHER ANALYSIS
# =========================

st.subheader("Weather vs Demand")

fig_weather = px.scatter(
    weather_df,
    x="temperature_2m",
    y="total_trips",
    color="precipitation",
    hover_data=["pickup_date"]
)

st.plotly_chart(fig_weather, use_container_width=True)