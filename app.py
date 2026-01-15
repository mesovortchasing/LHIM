import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & IMPACT ENGINE ---

def calculate_complex_impacts(lat, lon, s_lat, s_lon, p):
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow = p
    
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    if r < 1: r = 1
    
    # 1. LIQUIDITY MULTIPLIER
    env_mult = (rh / 85.0) * (0.6 + outflow / 2.5)
    effective_v_max = v_max * env_mult
    
    # 2. WIND PHYSICS (Holland Model + Friction + Inflow)
    B = 1.3 + (effective_v_max / 150)
    v_sym = effective_v_max * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    inflow = np.radians(25 if r > r_max else 15)
    wind_dir_math = angle + np.pi/2 + inflow
    v_forward = f_speed * 0.5 * np.cos(wind_dir_math - np.radians(f_dir))
    
    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + (shear_mag / 60) * np.cos(angle - shear_rad)
    
    total_wind = (v_sym * shear_effect) + v_forward
    if lat > 30.25: total_wind *= 0.85 
    
    # 3. RAINFALL
    rain_base = (effective_v_max / 45) * np.exp(-r / (r_max * 2))
    rain_asym = 1 + (shear_mag / 30) * np.sin(angle - shear_rad)
    total_rain = max(0, rain_base * rain_asym)
    
    # 4. SIMULATED IR SATELLITE (Heatmap Weight)
    off_x, off_y = (shear_mag/8) * np.cos(shear_rad), (shear_mag/8) * np.sin(shear_rad)
    r_sat = np.sqrt((dx - off_x)**2 + (dy - off_y)**2)
    # Weight for HeatMap (0 to 1 range)
    brightness = np.exp(-r_sat / (r_max * 4)) * (rh / 100)
    
    return max(0, total_wind), total_rain, brightness

def calculate_bay_surge(v_max, s_lon):
    dist_west = -88.0 - s_lon
    base = (v_max**2 / 1600)
    multiplier = 1.8 if dist_west > 0 else 0.5
    return base * multiplier

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(layout="wide", page_title="Mobile Hurr-Sim V3")

st.markdown("""
    <style>
    .legend { position: fixed; bottom: 30px; left: 30px; width: 200px; background: rgba(255,255,255,0.9); 
              padding: 15px; border: 2px solid #333; z-index: 9999; border-radius: 10px; }
    .legend i { width: 15px; height: 15px; display: inline-block; margin-right: 8px; vertical-align: middle; }
    </style>
    <div class="legend">
        <b>Mobile Impact Key</b><br>
        <i style="background:red"></i> Severe (>95 kts)<br>
        <i style="background:orange"></i> Extensive (75-95)<br>
        <i style="background:yellow"></i> Moderate (50-75)<br>
        <i style="background:blue"></i> TS Force (34-50)<br>
        <hr>
        <b>IR Satellite:</b><br>
        Red/Cyan = Deep Convection
    </div>
    """, unsafe_allow_html=True)

st.title("🌀 Mobile County Hurricane Simulator (Alpha V3)")

with st.sidebar:
    st.header("1. Environmental Liquidity")
    rh = st.slider("Mid-Level Humidity (%)", 30, 100, 80)
    outflow = st.slider("Outflow Efficiency", 0.0, 1.0, 0.7)
    
    st.header("2. Core Dynamics")
    v_max = st.slider("Intensity (kts)", 40, 160, 90)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 15)
    f_dir = st.slider("Heading (Deg)", 0, 360, 335)
    r_max = st.slider("Radius of Max Winds (mi)", 10, 60, 25)
    
    st.header("3. Vertical Wind Shear")
    shear_mag = st.slider("Shear Magnitude (kts)", 0, 60, 15)
    shear_dir = st.slider("Shear From (Deg)", 0, 360, 270)
    
    st.header("4. Display & Landfall")
    show_ir = st.checkbox("Show Simulated IR Satellite", value=True)
    l_lat = st.number_input("Landfall Lat", value=30.35)
    l_lon = st.number_input("Landfall Lon", value=-88.15)

# --- 3. RUN SIMULATION ---
p = [v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow]
surge_res = calculate_bay_surge(v_max, l_lon)

# Generate Grid (Higher density for smooth IR)
grid_n = 40 if show_ir else 25
lats = np.linspace(29.6, 31.4, grid_n)
lons = np.linspace(-88.9, -87.3, grid_n)
results = []
sat_data = []

for lt in lats:
    for ln in lons:
        w, rain, ir_weight = calculate_complex_impacts(lt, ln, l_lat, l_lon, p)
        results.append([lt, ln, w, rain])
        if ir_weight > 0.1:
            sat_data.append([lt, ln, ir_weight])

df = pd.DataFrame(results, columns=['lat', 'lon', 'wind', 'rain'])

# --- 4. THE INTERACTIVE MAP ---
c1, c2 = st.columns([4, 1])

with c1:
    m = folium.Map(location=[30.5, -88.1], zoom_start=9, tiles="CartoDB DarkMatter")
    
    # SMOOTH IR SATELLITE LAYER
    if show_ir and len(sat_data) > 0:
        HeatMap(
            sat_data,
            radius=35,
            blur=25,
            min_opacity=0.15,
            gradient={0.2: 'gray', 0.4: 'white', 0.7: 'cyan', 0.9: 'red'}
        ).add_to(m)
    
    # Wind Impact Markers
    for _, row in df.iterrows():
        if row['wind'] > 34:
            color = 'red' if row['wind'] > 95 else 'orange' if row['wind'] > 75 else 'yellow' if row['wind'] > 50 else 'blue'
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=2,
                color=color, fill=True, weight=1,
                popup=f"Wind: {int(row['wind'])} kts | Rain: {row['rain']:.1f} in/hr"
            ).add_to(m)
            
    st_folium(m, width="100%", height=700)

with c2:
    st.metric("Est. Bay Surge", f"{surge_res:.1f} ft")
    st.write("---")
    st.write("**Local Peak Forecast**")
    st.write(f"Peak Wind: {int(df['wind'].max())} kts")
    st.write(f"Max Rain Rate: {df['rain'].max():.1f} in/hr")
    
    if v_max < 70 and shear_mag > 25:
        st.warning("⚠️ Disorganized 'Halfacane' detected.")
    if l_lon < -88.2:
        st.error("⚠️ DIRTY SIDE RISK.")

st.dataframe(df[['wind', 'rain']].describe().iloc[1:])
