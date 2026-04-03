import streamlit as st
import requests
from datetime import datetime

# ====================== MATCH YOUR VISUALIZER EXACTLY ======================
st.set_page_config(
    page_title="Visualizer - Weather",
    page_icon="🌤️",
    layout="wide"
)

# Paste your EXACT custom CSS from the visualizer here (I copied it directly from the code you shared)
st.markdown("""
<style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 3px solid #FF4B4B;
        margin-bottom: 2rem;
    }
    .logo-text {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .stApp {
        background: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: 600;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #FF4B4B;
    }
    .metric-label {
        font-size: 1rem;
        color: #FAFAFA;
    }
    .stMetricValue {
        font-size: 2rem;
        color: #FF8C42;
    }
    .stMarkdown h2, .stMarkdown h3 {
        color: #FF8C42;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #FF4B4B;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ====================== MATCHING HEADER ======================
st.markdown("""
<div class="header-container">
    <div class="logo-text">
        🌤️ Visualizer Weather
    </div>
    <p style="color:#FAFAFA; font-size:1.1rem; margin:0;">
        Your personal weather dashboard • Built to match your main Visualizer
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ====================== CITY INPUT (default to your location) ======================
col1, col2 = st.columns([3, 1])
with col1:
    city = st.text_input(
        "🌍 Search city or zip code",
        placeholder="Pataskala, OH or London, UK",
        value="Pataskala",
        label_visibility="collapsed"
    )
with col2:
    if st.button("🔎 Get Weather", type="primary", use_container_width=True):
        pass  # trigger below

if st.button("🔎 Get Weather", type="primary", use_container_width=True):
    with st.spinner("🌎 Fetching live weather..."):
        # 1. Geocode (free, no key)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_resp = requests.get(geo_url)
        
        if geo_resp.status_code != 200 or not geo_resp.json().get("results"):
            st.error("😕 City not found. Try 'Pataskala', 'Columbus', or any city name.")
            st.stop()
        
        loc = geo_resp.json()["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc["name"]
        region = loc.get("admin1", "")
        country = loc.get("country", "")

        # 2. Weather API (free Open-Meteo)
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,"
            f"apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m&"
            f"daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
        )
        w_resp = requests.get(weather_url)
        data = w_resp.json()
        current = data["current"]
        daily = data["daily"]

        # Weather emoji helper
        def get_emoji(code):
            return {
                0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️", 45: "🌫️", 48: "🌫️",
                51: "🌧️", 53: "🌧️", 55: "🌧️", 61: "🌧️", 63: "🌧️", 65: "🌧️",
                71: "❄️", 73: "❄️", 75: "❄️", 80: "🌦️", 81: "🌦️", 82: "🌧️",
                95: "⛈️", 96: "⛈️", 99: "⛈️"
            }.get(code, "🌥️")

        # ====================== CURRENT CONDITIONS ======================
        st.subheader(f"📍 {city_name}, {region} {country}")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("🌡️ Temperature", f"{current['temperature_2m']}°C", f"Feels like {current['apparent_temperature']}°C")
        with c2:
            st.metric("💧 Humidity", f"{current['relative_humidity_2m']}%")
        with c3:
            st.metric("🌬️ Wind", f"{current['wind_speed_10m']} km/h")
        with c4:
            st.metric("☔ Precip", f"{current['precipitation']} mm")
        with c5:
            st.metric("🕒 Updated", datetime.fromtimestamp(current['time']).strftime("%I:%M %p"))

        # ====================== 7-DAY FORECAST ======================
        st.subheader("📅 7-Day Forecast")
        cols = st.columns(7)
        for i, col in enumerate(cols):
            date = datetime.strptime(daily["time"][i], "%Y-%m-%d").strftime("%a")
            emoji = get_emoji(daily["weather_code"][i])
            with col:
                st.markdown(f"**{date}**")
                st.markdown(f"{emoji}")
                st.caption(f"**{daily['temperature_2m_max'][i]}°** / {daily['temperature_2m_min'][i]}°")
                st.caption(f"💧 {daily['precipitation_sum'][i]} mm")

        st.success("✅ Weather synced! Change the city above anytime.")
        st.caption("Completely separate file — never touches your main Visualizer app.")

# ====================== INSTRUCTIONS ======================
st.caption("💡 Tip: Just run `streamlit run weather_app.py` — it will look and feel exactly like your main Visualizer.")
