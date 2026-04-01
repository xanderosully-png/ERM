import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="ERM Live Dashboard", layout="wide")
st.title("🌡️ ERM Live Forecast Dashboard — V5.4")
st.caption("Real-time temperature predictions & history")

data_dir = Path("ERM_Data")

# List all cities
cities = ["Columbus_OH", "Miami_FL", "New_York_NY", "Los_Angeles_CA", "London_UK", "Tokyo_JP"]

tab1, tab2, tab3 = st.tabs(["📊 Live Predictions", "📈 History Charts", "📋 Raw Data"])

with tab1:
    st.subheader("Current Predictions")
    cols = st.columns(3)
    for i, city in enumerate(cities):
        csv_file = data_dir / f"erm_v4.4_{city.lower()}_{datetime.now().strftime('%Y%m%d')}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file).sort_values('timestamp', ascending=False)
            latest = df.iloc[0]
            with cols[i % 3]:
                st.metric(
                    label=f"**{city.replace('_', ' ')}**",
                    value=f"{latest['live_temp']:.1f}°C",
                    delta=f"1h pred: {latest.get('next_predicted_1h', latest['live_temp']):.1f}°C"
                )
                st.caption(f"Improvement: {latest.get('improvement_pct', 0):.1f}%")
        else:
            with cols[i % 3]:
                st.warning(f"No data yet for {city}")

with tab2:
    st.subheader("Temperature History & Predictions")
    city_choice = st.selectbox("Select city", cities)
    csv_file = data_dir / f"erm_v4.4_{city_choice.lower()}_{datetime.now().strftime('%Y%m%d')}.csv"
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values('timestamp')
        
        st.line_chart(df.set_index('timestamp')[['live_temp', 'next_predicted_1h', 'next_predicted_3h', 'next_predicted_6h']])
        st.dataframe(df[['timestamp', 'live_temp', 'next_predicted_1h', 'next_predicted_3h', 
                         'next_predicted_6h', 'improvement_pct']].tail(12), use_container_width=True)
    else:
        st.info(f"No data yet for {city_choice} today.")

with tab3:
    st.subheader("All Raw CSV Files")
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        for f in sorted(csv_files, reverse=True):
            st.write(f"📄 {f.name}")
    else:
        st.info("No CSV files yet — run /update first.")

st.sidebar.success("✅ Backend is running")
st.sidebar.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
