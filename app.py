import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & RADAR ENGINE ---

def get_wind_arrow(deg):
    """Converts degrees to a high-contrast directional arrow for the UI."""
    arrows = ['↓', '↙', '←', '↖', '↑', '↗', '→', '↘']
    idx = int((deg + 22.5) % 360 / 45)
    return arrows[idx]

def calculate_full_physics(lat, lon, s_lat, s_lon, p, level=1000):
    """Wind, IR, and Vertical Profile Engine."""
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p
    
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    if r < 1: r = 1
    
    level_scale = {1000: 1.0, 925: 0.92, 850: 0.85, 500: 0.45, 200: 0.25}
    l_mult = level_scale.get(level, 1.0)
    
    env_mult = (rh / 85.0) * (0.6 + outflow / 2.5) * sst_mult
    effective_v_max = v_max * env_mult * l_mult
    
    B = 1.3 + (effective_v_max / 150)
    v_sym = effective_v_max * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    inflow_val = 25 if level > 500 else -30 
    inflow_angle = np.radians(inflow_val if r > r_max else inflow_val/2)
    wind_angle_rad = angle + (np.pi / 2) + inflow_angle
    
    shear_rad = np.radians(shear_dir)
    asym_weight = 1.0 + (1.0 - symmetry) 
    shear_factor = (shear_mag / 40) if level < 500 else (shear_mag / 60)
    shear_effect = 1 + (asym_weight * shear_factor) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    total_wind = (v_sym * shear_effect) + v_forward
    if lat > 30.25 and level == 1000: total_wind *= 0.85 
    
    # IR Logic for SIMUSAT
    spiral = np.sin(r / 12.0 - angle * 2.5) 
    ir_weight = np.exp(-r / (r_max * 6.0)) * (rh / 100) * (0.7 + max(0, spiral * 0.4))
    
    return max(0, total_wind), np.degrees(wind_angle_rad), ir_weight

def get_synthetic_radar(lat, lon, s_lat, s_lon, p, r_site=(30.67, -88.24)):
    """Produces Synthetic dBZ and Radial Velocity (kts)."""
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, _, symmetry, _ = p
    w, wd, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p)
    
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Reflectivity (dBZ) Model
    # 1. Eyewall Ring (Gaussian)
    eyewall = 48 * np.exp(-((r - r_max)**2) / (r_max * 0.3)**2)
    # 2. Spiral Bands (Log-Spiral)
    band_logic = np.sin(r / (r_max*0.7) - angle * 2.5)
    bands = max(0, band_logic * 35 * np.exp(-r/70))
    # 3. Moat Suppression
    moat = 0.2 if (r_max * 1.5 < r < r_max * 2.5) else 1.0
    
    dbz = (eyewall + bands) * moat * (rh / 100) * symmetry
    if r < r_max * 0.4: dbz *= 0.1 # Eye
    
    # Radial Velocity Model (Relative to Radar Site)
    rdx, rdy = (lon - r_site[1]) * 53, (lat - r_site[0]) * 69
    dist_radar = np.sqrt(rdx**2 + rdy**2)
    
    # Dot product of wind vector and radar-relative unit vector
    u_w, v_w = w * np.cos(np.radians(wd)), w * np.sin(np.radians(wd))
    v_radial = (u_w * rdx + v_w * rdy) / max(0.1, dist_radar)
    
    return min(65, dbz), v_radial, dist_radar

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

def get_local_conditions(lat, lon, s_lat, s_lon, r_max, p):
    w, _, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p)
    dbz, vr, rng = get_synthetic_radar(lat, lon, s_lat, s_lon, p)
    dist_storm = np.sqrt(((lon - s_lon) * 53)**2 + ((lat - s_lat) * 69)**2)
    
    # Feature Classification
    if dist_storm < r_max * 0.4: feat = "Eye (Clear)"
    elif abs(dist_storm - r_max) < r_max * 0.3: feat = "Eyewall (Convective)"
    elif r_max * 1.5 < dist_storm < r_max * 2.5: feat = "Moat (Subsidence)"
    else: feat = "Rainband"
        
    return f"""
    <div style="font-family: monospace; min-width: 200px; background: #000; color: #0f0; padding: 10px; border-radius: 5px;">
        <b style="color: white;">RADARSCOPE INSPECTOR (SYN)</b><br>
        <hr style="border: 0.5px solid #333;">
        Reflectivity: {int(dbz)} dBZ<br>
        Radial Vel: {int(vr)} kts<br>
        Range (KMOB): {int(rng)} mi<br>
        Feature: {feat}<br>
        SFC Wind: {int(w)} kts
    </div>
    """

def calculate_bay_surge(v_max, s_lon):
    dist_west = -88.0 - s_lon
    base = (v_max**2 / 1600)
    return base * (1.8 if dist_west > 0 else 0.5)

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="LHIM | Radar Diagnostics")

st.markdown("""
    <style>
    .stSlider { padding-bottom: 0px; }
    .env-panel { background: rgba(255, 75, 75, 0.05); padding: 15px; border-radius: 10px; border: 1px solid #ff4b4b; }
    .radar-note { font-size: 0.7em; color: #aaa; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("🛡️ LHIM Diagnostics")
    
    with st.expander("🌊 SST & Radar Config", expanded=True):
        sst_season = st.selectbox("Window", ["Early (June/July)", "Peak (Aug/Sept)", "Late (Oct/Nov)"])
        sst_mult = {"Early (June/July)": 0.85, "Peak (Aug/Sept)": 1.1, "Late (Oct/Nov)": 0.9}[sst_season]
        st.markdown("---")
        radar_mode = st.radio("Radar Layer", ["Off", "Reflectivity (dBZ)", "Radial Velocity (kts)"])
        rad_alpha = st.slider("Radar Opacity", 0.0, 1.0, 0.5)

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
    ir_opacity = st.slider("IR Alpha", 0.0, 1.0, 0.15)
    active_levels = []
    col_a, col_b = st.columns(2)
    if col_a.checkbox("200mb", value=False): active_levels.append(200)
    if col_b.checkbox("500mb", value=False): active_levels.append(500)
    if col_a.checkbox("850mb", value=False): active_levels.append(850)
    if col_b.checkbox("925mb", value=False): active_levels.append(925)
    lvl_opacity = st.slider("Diagnostics Alpha", 0.1, 1.0, 0.4)
    st.markdown('</div>', unsafe_allow_html=True)
    
    l_lat = st.number_input("Landfall Lat", value=30.35)
    l_lon = st.number_input("Landfall Lon", value=-88.15)
    map_theme = st.select_slider("Map Theme", options=["Dark Mode", "Light Mode"], value="Dark Mode")
    show_circles = st.checkbox("Show Impact Circles", value=True)

# --- 3. COMPUTATION ---
p = [v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult]
surge_res = calculate_bay_surge(v_max, l_lon)

# Satellite IR Data
sat_lats = np.linspace(28.0, 33.0, 60)
sat_lons = np.linspace(-91.0, -86.0, 60)
sat_data = []
for lt in sat_lats:
    for ln in sat_lons:
        _, _, ir_w = calculate_full_physics(lt, ln, l_lat, l_lon, p)
        if ir_w > 0.08: sat_data.append([lt, ln, ir_w])

# Radar Data Grid
radar_grid = []
if radar_mode != "Off":
    for lt in np.linspace(29.5, 31.5, 50):
        for ln in np.linspace(-89.0, -87.0, 50):
            dbz, vr, _ = get_synthetic_radar(lt, ln, l_lat, l_lon, p)
            val = dbz if radar_mode == "Reflectivity (dBZ)" else abs(vr)
            if val > (10 if radar_mode == "Reflectivity (dBZ)" else 20):
                radar_grid.append([lt, ln, val])

# --- 4. MAPPING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[30.5, -88.1], zoom_start=9, tiles="CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap")
    
    # 1. SIMUSAT IR
    if show_ir and len(sat_data) > 0:
        HeatMap(sat_data, radius=25, blur=18, min_opacity=ir_opacity, max_zoom=13,
                gradient={0.2: 'gray', 0.4: 'white', 0.6: 'cyan', 0.8: 'red', 0.95: 'maroon'}).add_to(m)
    
    # 2. Synthetic Radar
    if radar_mode != "Off" and len(radar_grid) > 0:
        grad = {0.1: 'blue', 0.4: 'green', 0.7: 'yellow', 0.9: 'red', 1.0: 'fuchsia'} if "Reflectivity" in radar_mode else {0.3: 'cyan', 0.6: 'white', 0.9: 'red'}
        HeatMap(radar_grid, radius=12, blur=10, min_opacity=rad_alpha, gradient=grad).add_to(m)

    # 3. Wind & Sounding Points
    for lt in np.linspace(29.6, 31.4, 15):
        for ln in np.linspace(-88.9, -87.3, 15):
            w, wd, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p, level=1000)
            if w > 30:
                color = 'red' if w > 95 else 'orange' if w > 75 else 'yellow' if w > 50 else 'blue'
                if show_circles:
                    folium.CircleMarker(location=[lt, ln], radius=w/6, color=color, fill=True, weight=1, fill_opacity=0.3,
                                        popup=folium.Popup(get_local_conditions(lt, ln, l_lat, l_lon, r_max, p), max_width=250)).add_to(m)
            
            for lvl in active_levels:
                lw, _, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p, level=lvl)
                if lw > 15: folium.CircleMarker(location=[lt, ln], radius=2, color='white', opacity=lvl_opacity).add_to(m)

    last_click = st_folium(m, width="100%", height=700)

with c2:
    st.metric("Est. Bay Surge", f"{surge_res:.1f} ft")
    st.markdown('<p class="radar-note">Synthetic Radar (Estimated) — generated from LHIM storm structure; NOT observed radar.</p>', unsafe_allow_html=True)
    st.write("---")
    
    if last_click and last_click.get("last_clicked"):
        clat, clon = last_click["last_clicked"]["lat"], last_click["last_clicked"]["lng"]
        st.subheader("📍 Vertical Sounding")
        pdf = pd.DataFrame(get_vertical_profile(clat, clon, l_lat, l_lon, p))
        st.table(pdf.set_index('Level (mb)'))
        st.line_chart(pdf[['Temp (°C)', 'Dewp (°C)']])
    else:
        st.info("Click the map to inspect radar cells or vertical atmospheric profiles.")
    
    st.subheader("Structure Analytics")
    eye_check = (v_max / 100) * (1 - (shear_mag / 40)) * symmetry
    st.write(f"**Eye Definition:** {'HD' if eye_check > 0.8 else 'Ragged' if eye_check > 0.4 else 'None'}")
    st.progress(min(max(eye_check, 0.0), 1.0))
