import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & WIND VECTOR ENGINE ---

def calculate_full_physics(lat, lon, s_lat, s_lon, p):
    """
    Calculates Wind, Rain, Wind Direction, and IR Weight for SIMUSAT.
    """
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow = p
    
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    if r < 1: r = 1
    
    # 1. LIQUIDITY & INTENSITY
    env_mult = (rh / 85.0) * (0.6 + outflow / 2.5)
    effective_v_max = v_max * env_mult
    
    # 2. WIND MAGNITUDE (Holland)
    B = 1.3 + (effective_v_max / 150)
    v_sym = effective_v_max * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    # 3. WIND DIRECTION (Inflow + Rotation)
    inflow_angle = np.radians(25 if r > r_max else 15)
    wind_angle_rad = angle + (np.pi / 2) + inflow_angle
    
    # Asymmetry
    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + (shear_mag / 60) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    total_wind = (v_sym * shear_effect) + v_forward
    if lat > 30.25: total_wind *= 0.85 
    
    # 4. SIMUSAT IR WEIGHT
    off_x, off_y = (shear_mag/8) * np.cos(shear_rad), (shear_mag/8) * np.sin(shear_rad)
    r_sat = np.sqrt((dx - off_x)**2 + (dy - off_y)**2)
    ir_weight = np.exp(-r_sat / (r_max * 4.5)) * (rh / 100)
    
    return max(0, total_wind), np.degrees(wind_angle_rad), ir_weight

def calculate_bay_surge(v_max, s_lon):
    dist_west = -88.0 - s_lon
    base = (v_max**2 / 1600)
    return base * (1.8 if dist_west > 0 else 0.5)

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Mobile Hurr-Sim V4")

st.markdown("""
    <style>
    .legend { position: fixed; bottom: 30px; left: 30px; width: 220px; background: rgba(255,255,255,0.9); 
              padding: 15px; border: 2px solid #333; z-index: 9999; border-radius: 10px; font-family: sans-serif; }
    .legend i { width: 12px; height: 12px; display: inline-block; margin-right: 8px; border-radius: 50%; }
    </style>
    <div class="legend">
        <b>Chaser Legend</b><br>
        <i style="background:red"></i> Severe (>95 kts)<br>
        <i style="background:orange"></i> Extensive (75-95)<br>
        <i style="background:yellow"></i> Moderate (50-75)<br>
        <i style="background:blue"></i> TS Force (34-50)<br>
        <hr>
        <b>Symbols:</b> Arrows show wind flow. <br>Circle size = Wind Magnitude.
    </div>
    """, unsafe_allow_html=True)

st.title("🌀 Mobile County Interactive Chaser Model (Alpha V4)")

with st.sidebar:
    st.header("1. Environmental Factors")
    rh = st.slider("Mid-Level Humidity (%)", 30, 100, 80)
    outflow = st.slider("Outflow Efficiency", 0.0, 1.0, 0.7)
    
    st.header("2. Core Parameters")
    v_max = st.slider("Intensity (kts)", 40, 160, 95)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 12)
    f_dir = st.slider("Storm Heading (Deg)", 0, 360, 330)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    
    st.header("3. Upper Level Shear")
    shear_mag = st.slider("Shear Magnitude (kts)", 0, 60, 15)
    shear_dir = st.slider("Shear Direction (From)", 0, 360, 270)
    
    st.header("4. Display Settings")
    show_ir = st.checkbox("Toggle SIMUSAT (Smooth IR)", value=True)
    show_barbs = st.checkbox("Show Wind Flow Vectors", value=True)
    l_lat = st.number_input("Landfall Lat", value=30.35)
    l_lon = st.number_input("Landfall Lon", value=-88.15)
    
    st.header("5. Misc")
    map_theme = st.select_slider(
        "Map Theme",
        options=["Dark Mode", "Light Mode"],
        value="Dark Mode",
        help="🌙 Dark Mode uses high-contrast satellite colors. ☀️ Light Mode uses standard street maps."
    )
    # Visual cues for the theme slider
    if map_theme == "Light Mode":
        st.caption("☀️ Sun Mode Active")
    else:
        st.caption("🌙 Moon Mode Active")
        
    show_circles = st.checkbox("Show Wind Circles", value=True)

# --- 3. COMPUTATION ---
p = [v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow]
surge_res = calculate_bay_surge(v_max, l_lon)

grid_n = 35 if show_ir else 20
lats = np.linspace(29.6, 31.4, grid_n)
lons = np.linspace(-88.9, -87.3, grid_n)
results = []
sat_data = []

for lt in lats:
    for ln in lons:
        w, w_dir, ir_w = calculate_full_physics(lt, ln, l_lat, l_lon, p)
        results.append([lt, ln, w, w_dir])
        if ir_w > 0.1:
            sat_data.append([lt, ln, ir_w])

df = pd.DataFrame(results, columns=['lat', 'lon', 'wind', 'dir'])

# --- 4. MAPPING ---
c1, c2 = st.columns([4, 1])

with c1:
    # Set tile based on Misc toggle
    tileset = "CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap"
    
    m = folium.Map(location=[30.5, -88.1], zoom_start=9, tiles=tileset)
    
    # 1. SIMUSAT (Smooth IR Satellite)
    if show_ir and len(sat_data) > 0:
        HeatMap(sat_data, radius=35, blur=25, min_opacity=0.1,
                gradient={0.2: 'gray', 0.4: 'white', 0.7: 'cyan', 0.9: 'red'}).add_to(m)
    
    # 2. Wind Impact Visualization
    for _, row in df.iterrows():
        if row['wind'] > 30:
            color = 'red' if row['wind'] > 95 else 'orange' if row['wind'] > 75 else 'yellow' if row['wind'] > 50 else 'blue'
            
            # Misc Toggle: Wind Circles
            if show_circles:
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=row['wind']/6,
                    color=color, fill=True, weight=1, fill_opacity=0.4,
                    popup=f"{int(row['wind'])} kts"
                ).add_to(m)
            
            # Flow Vectors
            if show_barbs:
                length = 0.02
                end_lat = row['lat'] + length * np.sin(np.radians(row['dir']))
                end_lon = row['lon'] + length * np.cos(np.radians(row['dir']))
                
                v_color = 'white' if map_theme == "Dark Mode" else 'black'
                folium.PolyLine(
                    locations=[[row['lat'], row['lon']], [end_lat, end_lon]],
                    color=v_color, weight=1, opacity=0.6
                ).add_to(m)

    st_folium(m, width="100%", height=750)

with c2:
    st.metric("Est. Bay Surge", f"{surge_res:.1f} ft")
    st.write("---")
    st.subheader("Peak Analytics")
    st.write(f"**Max Wind:** {int(df['wind'].max())} kts")
    st.write(f"**SIMUSAT Status:** Active")
    
    if l_lon < -88.2:
        st.error("DIRTY SIDE RISK")
