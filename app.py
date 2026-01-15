import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- 1. PHYSICS ENGINES ---

def get_wind_and_rain(lat, lon, s_lat, s_lon, v_max, r_max, f_speed, f_dir, shear_mag, shear_dir):
    """Calculates asymmetric wind (kts) and rain (in/hr) using Holland + Shear Tilt"""
    # Geodesic math
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx) # math angle
    
    if r < 1: r = 1
    
    # 1. Base Holland Wind
    B = 1.5 + (v_max / 100) # Dynamic shape factor
    v_sym = v_max * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    # 2. Inflow & Forward Motion Asymmetry
    # Winds spiral in 20 deg. We add the storm's forward vector.
    inflow = np.radians(20)
    wind_dir_math = angle + np.pi/2 + inflow
    v_forward = f_speed * 0.5 * np.cos(wind_dir_math - np.radians(f_dir))
    
    # 3. Shear Disorganization (The "Halfacane" effect)
    # Reduces wind on upshear side, enhances downshear
    shear_factor = 1 + (shear_mag / 60) * np.cos(angle - np.radians(shear_dir))
    
    total_wind = (v_sym * shear_factor) + v_forward
    
    # 4. Rainfall Logic (R-CLIPER style + Shear Tilt)
    # Heaviest rain is usually downshear-left
    rain_base = (v_max / 40) * np.exp(-r / r_max)
    rain_asym = 1 + (shear_mag / 25) * np.sin(angle - np.radians(shear_dir))
    total_rain = max(0, rain_base * rain_asym)
    
    return max(0, total_wind), total_rain

def calculate_mobile_bay_surge(v_max, f_dir, storm_lon):
    """Simplified Wind Setup for Mobile Bay (Shallow Water Eq)"""
    # If storm is West of Mobile, wind blows UP the bay (South wind) = High Surge
    # If storm is East, wind blows DOWN the bay (North wind) = Blow out
    relative_pos = -88.1 - storm_lon # Mobile Lon minus Storm Lon
    
    # Base surge based on intensity
    base_surge = (v_max**2 / 1500) 
    
    # Funnel effect: If winds are from 150-210 degrees (South), multiplier increases
    if relative_pos > 0: # Storm is West of Bay (Dangerous side)
        surge = base_surge * 1.8
    else:
        surge = base_surge * 0.4
        
    return max(0.5, surge)

# --- 2. UI SETUP ---
st.set_page_config(layout="wide", page_title="Mobile Hurr-Sim")

# Custom CSS for the Bottom-Left Legend
st.markdown("""
    <style>
    .legend {
        position: fixed; bottom: 20px; left: 20px; width: 200px;
        background: rgba(255,255,255,0.9); padding: 10px;
        border: 2px solid #333; z-index: 9999; font-size: 12px;
    }
    .legend i { width: 15px; height: 15px; display: inline-block; margin-right: 5px; }
    </style>
    <div class="legend">
        <b>Impact Legend</b><br>
        <i style="background:red"></i> Severe (>95kts)<br>
        <i style="background:orange"></i> Extensive (75-95kts)<br>
        <i style="background:yellow"></i> Moderate (50-75kts)<br>
        <i style="background:blue"></i> TS Force (34-50kts)<br>
        <hr>
        <b>Current Params:</b><br>
        Model: Holland-Ekman Alpha
    </div>
    """, unsafe_allow_html=True)

st.title("🌀 Mobile County Hurricane Simulator")

# Sidebar - 15 Parameters (Grouped)
with st.sidebar:
    st.header("1. Core Dynamics")
    v_max = st.slider("Intensity (Knots)", 40, 165, 90)
    r_max = st.slider("Radius of Max Winds (miles)", 10, 60, 25)
    f_speed = st.slider("Forward Speed (mph)", 2, 35, 12)
    f_dir = st.slider("Movement Direction (Deg)", 0, 360, 340)
    
    st.header("2. System Organization")
    shear_mag = st.slider("Wind Shear (knots)", 0, 60, 15)
    shear_dir = st.slider("Shear Direction (From)", 0, 360, 270)
    organization = st.select_slider("System Symmetry", options=["Disorganized", "Symmetric", "Stretched"])
    
    st.header("3. Landfall Location")
    lat = st.number_input("Lat", value=30.3)
    lon = st.number_input("Lon", value=-88.2)

# --- 3. COMPUTATION ---
surge_val = calculate_mobile_bay_surge(v_max, f_dir, lon)

# Create high-res grid
lats = np.linspace(29.9, 31.2, 25)
lons = np.linspace(-88.6, -87.6, 25)
results = []

for lt in lats:
    for ln in lons:
        w, r = get_wind_and_rain(lt, ln, lat, lon, v_max, r_max, f_speed, f_dir, shear_mag, shear_dir)
        results.append([lt, ln, w, r])

df = pd.DataFrame(results, columns=['lat', 'lon', 'wind', 'rain'])

# --- 4. VISUALIZATION ---
col1, col2 = st.columns([3, 1])

with col1:
    m = folium.Map(location=[30.5, -88.1], zoom_start=9, tiles="cartodbpositron")
    
    for _, row in df.iterrows():
        if row['wind'] > 34:
            color = 'red' if row['wind'] > 95 else 'orange' if row['wind'] > 75 else 'yellow' if row['wind'] > 50 else 'blue'
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=row['wind']/8,
                color=color, fill=True, weight=1,
                popup=f"{int(row['wind'])} kts | {row['rain']:.1f} in/hr"
            ).add_to(m)
    
    st_folium(m, width="100%", height=600)

with col2:
    st.metric("Est. Surge (Mobile Bay)", f"{surge_val:.1f} ft")
    st.write("---")
    st.write("**Peak Local Impacts**")
    st.write(f"Max Wind: {int(df['wind'].max())} kts")
    st.write(f"Max Rain: {df['rain'].max():.2f} in/hr")
    
    if surge_val > 8:
        st.error("🚨 MAJOR INUNDATION: I-10 Bayway and Causeway likely impassable.")
