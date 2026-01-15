import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & WIND VECTOR ENGINE ---

def calculate_full_physics(lat, lon, s_lat, s_lon, p):
    """
    Enhanced Physics: Wind, Rain-Coupled IR, and Spiral Banding.
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
    
    # 4. SIMUSAT FLUID IR WEIGHT (Spiral Bands + Rain Rate)
    # Adding a logarithmic spiral factor for banding
    spiral = np.sin(r / 12.0 - angle * 2.5) # The "Fluid" band logic
    band_noise = max(0, spiral * 0.4)
    
    # Shift canopy downshear (Outflow tilt)
    off_x, off_y = (shear_mag/5) * np.cos(shear_rad), (shear_mag/5) * np.sin(shear_rad)
    r_sat = np.sqrt((dx - off_x)**2 + (dy - off_y)**2)
    
    # Base cloud + banding + RH factor
    ir_weight = np.exp(-r_sat / (r_max * 6.0)) * (rh / 100) * (0.7 + band_noise)
    
    # Rain Rate Coupling: Coldest tops (Red/Cyan) where wind is highest
    rain_coupling = (total_wind / 120) * np.exp(-r / (r_max * 2))
    ir_weight = min(1.0, ir_weight + rain_coupling)
    
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
        <b>SIMUSAT:</b><br>
        <span style="color:red">●</span> Deep Convection<br>
        <span style="color:cyan">●</span> Heavy Rain Rates<br>
    </div>
    """, unsafe_allow_html=True)

st.title("🌀 Mobile County Interactive Chaser Model (Alpha V4)")

with st.sidebar:
    st.header("1. Environmental Factors")
    rh = st.slider("Mid-Level Humidity (%)", 30, 100, 85)
    outflow = st.slider("Outflow Efficiency", 0.0, 1.0, 0.8)
    
    st.header("2. Core Parameters")
    v_max = st.slider("Intensity (kts)", 40, 160, 110)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 12)
    f_dir = st.slider("Storm Heading (Deg)", 0, 360, 330)
    r_max = st.slider("RMW (miles)", 10, 60, 30)
    
    st.header("3. Upper Level Shear")
    shear_mag = st.slider("Shear Magnitude (kts)", 0, 60, 12)
    shear_dir = st.slider("Shear Direction (From)", 0, 360, 260)
    
    st.header("4. Display Settings")
    show_ir = st.checkbox("Toggle SIMUSAT (Fluid IR)", value=True)
    show_barbs = st.checkbox("Show Wind Flow Vectors", value=True)
    l_lat = st.number_input("Landfall Lat", value=30.35)
    l_lon = st.number_input("Landfall Lon", value=-88.15)
    
    st.header("5. Misc")
    map_theme = st.select_slider("Map Theme", options=["Dark Mode", "Light Mode"], value="Dark Mode")
    show_circles = st.checkbox("Show Wind Circles", value=True)

# --- 3. COMPUTATION ---
p = [v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow]
surge_res = calculate_bay_surge(v_max, l_lon)

# Expanded satellite grid (2x larger than wind box to avoid "boxed in" look)
sat_lats = np.linspace(28.5, 32.5, 45)
sat_lons = np.linspace(-90.5, -86.5, 45)

# Original wind grid for the Mobile region
wind_lats = np.linspace(29.6, 31.4, 25)
wind_lons = np.linspace(-88.9, -87.3, 25)

sat_data = []
for lt in sat_lats:
    for ln in sat_lons:
        _, _, ir_w = calculate_full_physics(lt, ln, l_lat, l_lon, p)
        if ir_w > 0.1:
            sat_data.append([lt, ln, ir_w])

results = []
for lt in wind_lats:
    for ln in wind_lons:
        w, w_dir, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p)
        results.append([lt, ln, w, w_dir])

df = pd.DataFrame(results, columns=['lat', 'lon', 'wind', 'dir'])

# --- 4. MAPPING ---
c1, c2 = st.columns([4, 1])

with c1:
    tileset = "CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap"
    m = folium.Map(location=[30.5, -88.1], zoom_start=9, tiles=tileset)
    
    # 1. SIMUSAT (Fluid IR Satellite)
    if show_ir and len(sat_data) > 0:
        # Realistic IR Gradient: Grays (low) -> Whites -> Cyans -> Reds (Heavy Convection)
        HeatMap(sat_data, radius=38, blur=28, min_opacity=0.08,
                gradient={0.2: 'gray', 0.4: 'white', 0.6: 'cyan', 0.8: 'red', 0.95: 'maroon'}).add_to(m)
    
    # 2. Wind Impact Visualization
    for _, row in df.iterrows():
        if row['wind'] > 30:
            color = 'red' if row['wind'] > 95 else 'orange' if row['wind'] > 75 else 'yellow' if row['wind'] > 50 else 'blue'
            if show_circles:
                folium.CircleMarker(location=[row['lat'], row['lon']], radius=row['wind']/6,
                                    color=color, fill=True, weight=1, fill_opacity=0.4).add_to(m)
            if show_barbs:
                length = 0.02
                end_lat = row['lat'] + length * np.sin(np.radians(row['dir']))
                end_lon = row['lon'] + length * np.cos(np.radians(row['dir']))
                v_color = 'white' if map_theme == "Dark Mode" else 'black'
                folium.PolyLine(locations=[[row['lat'], row['lon']], [end_lat, end_lon]],
                                color=v_color, weight=1, opacity=0.6).add_to(m)

    st_folium(m, width="100%", height=750)

with c2:
    st.metric("Est. Bay Surge", f"{surge_res:.1f} ft")
    st.write("---")
    st.subheader("Peak Analytics")
    st.write(f"**Max Wind:** {int(df['wind'].max())} kts")
    st.write(f"**SIMUSAT Status:** Fluid Feed Active")
    if l_lon < -88.2:
        st.error("DIRTY SIDE RISK")
