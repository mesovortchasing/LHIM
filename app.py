import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS ENGINE ---

def get_wind_arrow(deg):
    """Converts degrees to a high-contrast directional arrow for the UI."""
    arrows = ['↓', '↙', '←', '↖', '↑', '↗', '→', '↘']
    idx = int((deg + 22.5) % 360 / 45)
    return arrows[idx]

def calculate_full_physics(lat, lon, s_lat, s_lon, p, level=1000):
    """Calculates wind, IR weights, and thermodynamics based on hurricane parameters."""
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p
    
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    if r < 1: r = 1
    
    # Scaling for vertical levels
    level_scale = {1000: 1.0, 925: 0.92, 850: 0.85, 500: 0.45, 200: 0.25}
    l_mult = level_scale.get(level, 1.0)
    
    env_mult = (rh / 85.0) * (0.6 + outflow / 2.5) * sst_mult
    effective_v_max = v_max * env_mult * l_mult
    
    # Modified Rankine Vortex
    B = 1.3 + (effective_v_max / 150)
    v_sym = effective_v_max * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    # Inflow/Outflow Angle
    inflow_val = 25 if level > 500 else -30 
    inflow_angle = np.radians(inflow_val if r > r_max else inflow_val/2)
    wind_angle_rad = angle + (np.pi / 2) + inflow_angle
    
    # Shear and Asymmetry
    shear_rad = np.radians(shear_dir)
    asym_weight = 1.0 + (1.0 - symmetry) 
    shear_factor = (shear_mag / 40) if level < 500 else (shear_mag / 60)
    shear_effect = 1 + (asym_weight * shear_factor) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    total_wind = (v_sym * shear_effect) + v_forward
    if lat > 30.25 and level == 1000: total_wind *= 0.85 # Land friction
    
    # SIMUSAT IR weights
    spiral = np.sin(r / 12.0 - angle * 2.5) 
    ir_weight = np.exp(-r / (r_max * 6.0)) * (rh / 100) * (0.7 + max(0, spiral * 0.4))
    
    return max(0, total_wind), np.degrees(wind_angle_rad), ir_weight

def get_synthetic_radar(lat, lon, s_lat, s_lon, p):
    """Generates complex radar reflectivity based on storm anatomy."""
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, _, symmetry, _ = p
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # 1. Primary Eyewall
    eyewall = 55 * np.exp(-((r - r_max)**2) / (r_max * 0.25)**2)
    
    # 2. Outer Eyewall (Secondary ring for v_max > 100)
    outer_r = r_max * 2.3
    outer_eyewall = 40 * np.exp(-((r - outer_r)**2) / (r_max * 0.4)**2) if v_max > 100 else 0
    
    # 3. Moat Suppression
    moat = 0.2 if (r_max * 1.3 < r < outer_r * 0.8 and v_max > 100) else 1.0
    
    # 4. Spiral Bands
    band_mod = np.sin(r / (r_max * 0.6) - angle * 2.8)
    bands = max(0, band_mod * 42 * np.exp(-r / 130))
    
    # Combine and add noise/stratiform background
    dbz = (eyewall + outer_eyewall + bands + 10) * moat * (rh / 100) * symmetry
    if r < r_max * 0.4: dbz *= 0.05 # Clear eye
    
    return min(70, dbz)

def get_local_conditions(lat, lon, s_lat, s_lon, r_max, p):
    """Generates the Popup UI matching the requested photo style."""
    w, _, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p)
    dist = np.sqrt(((lon-s_lon)*53)**2 + ((lat-s_lat)*69)**2)
    
    # Match the UI from the user photo
    temp = 75.0 - (dist / 20)
    dewp = temp - 0.5
    vis = max(0.1, 10 - (w / 12))
    
    return f"""
    <div style="font-family: sans-serif; min-width: 160px; background: #111; color: #eee; padding: 10px; border-radius: 8px;">
        <b style="font-size: 1.1em;">Location Area</b><br>
        <hr style="margin: 5px 0; border: 0.1px solid #444;">
        🌪️ <b>Condition:</b> {int(w)} kts<br>
        🌡️ <b>Temp:</b> {temp:.1f}°F<br>
        💧 <b>Dew Pt:</b> {dewp:.1f}°F<br>
        👁️ <b>Visibility:</b> {vis:.1f} mi
    </div>
    """

def get_vertical_profile(lat, lon, s_lat, s_lon, p):
    levels = [1000, 925, 850, 500, 200]
    profile = []
    for lvl in levels:
        w, wd, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p, level=lvl)
        dist = np.sqrt(((lon-s_lon)*53)**2 + ((lat-s_lat)*69)**2)
        temp = (30 if lvl > 800 else 0 if lvl > 400 else -50) + max(0, (15 - dist/5))
        dewp = temp - (dist/10) - (lvl/1000 * 5)
        profile.append({"Level (mb)": lvl, "Wind (kts)": int(w), "Barb": get_wind_arrow(wd), "Temp (°C)": round(temp,1), "Dewp (°C)": round(dewp,1)})
    return profile

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="LHIM | Liquid Observation System")

st.markdown("""
    <style>
    .stSlider { padding-bottom: 0px; }
    .env-panel { background: rgba(255, 75, 75, 0.05); padding: 15px; border-radius: 10px; border: 1px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("🛡️ LHIM Diagnostics")
    
    with st.expander("📡 Observation Settings", expanded=True):
        sst_season = st.selectbox("Season", ["Early", "Peak", "Late"])
        sst_mult = {"Early": 0.85, "Peak": 1.1, "Late": 0.9}[sst_season]
        radar_mode = st.checkbox("Show Synthetic Radar", value=True)
        radar_alpha = st.slider("Radar Opacity", 0.1, 1.0, 0.6)

    st.header("1. Core Parameters")
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 12)
    f_dir = st.slider("Heading (Deg)", 0, 360, 330)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    
    st.header("2. Environment")
    rh = st.slider("Humidity (%)", 30, 100, 85)
    outflow = st.slider("Outflow Eff.", 0.0, 1.0, 0.8)
    symmetry = st.slider("Symmetry", 0.0, 1.0, 0.85)
    shear_mag = st.slider("Shear (kts)", 0, 60, 8)
    shear_dir = st.slider("Shear From (Deg)", 0, 360, 260)
    
    st.header("3. Map Layers")
    st.markdown('<div class="env-panel">', unsafe_allow_html=True)
    show_ir = st.checkbox("SIMUSAT (Fluid IR)", value=True)
    ir_opacity = st.slider("IR Alpha", 0.0, 1.0, 0.2)
    active_levels = []
    if st.checkbox("200mb Wind", value=False): active_levels.append(200)
    if st.checkbox("850mb Wind", value=False): active_levels.append(850)
    st.markdown('</div>', unsafe_allow_html=True)
    
    l_lat = st.number_input("Landfall Lat", value=30.35)
    l_lon = st.number_input("Landfall Lon", value=-88.15)
    map_theme = st.selectbox("Theme", ["Dark Mode", "Light Mode"])

# --- 3. COMPUTATION ---
p = [v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult]

# Liquid Observation Grids
sat_data = []
radar_data = []

# High resolution grid for liquid rendering
grid_lats = np.linspace(28.5, 32.5, 65)
grid_lons = np.linspace(-90.5, -86.5, 65)

for lt in grid_lats:
    for ln in grid_lons:
        w, wd, ir_w = calculate_full_physics(lt, ln, l_lat, l_lon, p)
        if show_ir and ir_w > 0.05:
            sat_data.append([lt, ln, ir_w])
        if radar_mode:
            dbz = get_synthetic_radar(lt, ln, l_lat, l_lon, p)
            if dbz > 15:
                radar_data.append([lt, ln, dbz])

# --- 4. MAPPING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[30.5, -88.1], zoom_start=9, 
                   tiles="CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap")
    
    # Liquid IR Layer
    if show_ir and sat_data:
        HeatMap(sat_data, radius=22, blur=15, min_opacity=ir_opacity, 
                gradient={0.2: 'gray', 0.4: 'white', 0.7: 'cyan', 0.9: 'red'}).add_to(m)
    
    # Liquid Radar Layer (Reflectivity)
    if radar_mode and radar_data:
        HeatMap(radar_data, radius=12, blur=8, min_opacity=radar_alpha,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 0.95: 'red'}).add_to(m)

    # Physical Interaction Markers (Wind Barbs/Points)
    for lt in np.linspace(29.7, 31.3, 14):
        for ln in np.linspace(-88.9, -87.4, 14):
            w, _, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p)
            if w > 35:
                color = 'red' if w > 95 else 'orange' if w > 64 else 'yellow'
                folium.CircleMarker(
                    location=[lt, ln], radius=w/8, color=color, fill=True, weight=1, fill_opacity=0.4,
                    popup=folium.Popup(get_local_conditions(lt, ln, l_lat, l_lon, r_max, p), max_width=200)
                ).add_to(m)

    last_click = st_folium(m, width="100%", height=750)

with c2:
    st.subheader("📍 Observation Data")
    if last_click and last_click.get("last_clicked"):
        clat, clon = last_click["last_clicked"]["lat"], last_click["last_clicked"]["lng"]
        pdf = pd.DataFrame(get_vertical_profile(clat, clon, l_lat, l_lon, p))
        st.caption(f"Lat: {clat:.2f} Lon: {clon:.2f}")
        st.table(pdf.set_index('Level (mb)'))
        st.line_chart(pdf[['Temp (°C)', 'Dewp (°C)']])
    else:
        st.info("Click a wind marker to view condition data and vertical sounding.")
    
    st.subheader("Structure Analytics")
    eye_check = (v_max / 100) * (1 - (shear_mag / 40)) * symmetry
    st.write(f"**Eye Quality:** {'High Def' if eye_check > 0.8 else 'Ragged' if eye_check > 0.4 else 'None'}")
    st.progress(min(max(eye_check, 0.0), 1.0))
