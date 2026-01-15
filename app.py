import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & WIND VECTOR ENGINE ---

def get_wind_arrow(deg):
    """Converts degrees to a high-contrast directional arrow for the UI."""
    arrows = ['↓', '↙', '←', '↖', '↑', '↗', '→', '↘']
    idx = int((deg + 22.5) % 360 / 45)
    return arrows[idx]

def calculate_full_physics(lat, lon, s_lat, s_lon, p, level=1000):
    """
    Enhanced Physics: Wind, Rain-Coupled IR, and vertical level adjustments.
    'level' parameter scales winds and temps for 200, 500, 850, 925, 1000mb.
    """
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p
    
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    if r < 1: r = 1
    
    # SST and Level-based Intensity Scaling
    level_scale = {1000: 1.0, 925: 0.92, 850: 0.85, 500: 0.45, 200: 0.25}
    l_mult = level_scale.get(level, 1.0)
    
    env_mult = (rh / 85.0) * (0.6 + outflow / 2.5) * sst_mult
    effective_v_max = v_max * env_mult * l_mult
    
    B = 1.3 + (effective_v_max / 150)
    v_sym = effective_v_max * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    # Inflow/Outflow vertical shear logic
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
    
    # IR/Satellite logic
    spiral = np.sin(r / 12.0 - angle * 2.5) 
    band_noise = max(0, spiral * 0.4)
    stretch_factor = (shear_mag / 4.5) * (2.0 - symmetry)
    off_x, off_y = stretch_factor * np.cos(shear_rad), stretch_factor * np.sin(shear_rad)
    r_sat = np.sqrt((dx - off_x)**2 + (dy - off_y)**2)
    
    ir_weight = np.exp(-r_sat / (r_max * 6.0)) * (rh / 100) * (0.7 + band_noise)
    rain_coupling = (total_wind / 120) * np.exp(-r / (r_max * 2))
    ir_weight = min(1.0, ir_weight + rain_coupling)

    eye_readiness = (v_max / 100) * (1 - (shear_mag / 45)) * (rh / 100) * symmetry
    if eye_readiness > 0.5 and r < (r_max * 0.45):
        eye_clearing = np.exp(-(r**2) / (r_max * 0.18)**2)
        ir_weight *= (1 - (eye_clearing * eye_readiness))

    return max(0, total_wind), np.degrees(wind_angle_rad), ir_weight

def get_vertical_profile(lat, lon, s_lat, s_lon, p):
    """Generates synthetic sounding data including visual wind barbs."""
    levels = [1000, 925, 850, 500, 200]
    profile = []
    for lvl in levels:
        w, wd, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p, level=lvl)
        dist = np.sqrt(((lon-s_lon)*53)**2 + ((lat-s_lat)*69)**2)
        core_warmth = max(0, (15 - dist/5)) if lvl < 850 else 0
        temp = (30 if lvl > 800 else 0 if lvl > 400 else -50) + core_warmth
        dewp = temp - (dist/10) - (lvl/1000 * 5)
        
        profile.append({
            "Level (mb)": lvl, 
            "Wind (kts)": int(w), 
            "Barb": get_wind_arrow(wd),
            "Temp (°C)": round(temp,1), 
            "Dewp (°C)": round(dewp,1)
        })
    return profile

def get_local_conditions(wind, lat, lon, s_lat, s_lon, r_max, p):
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    city = "Mobile" if lat > 30.6 else "Dauphin Island" if lat < 30.3 else "Theodore/Bayou La Batre"
    
    temp = 82 - (wind / 15) 
    dewp = temp - (2 * (1 - (wind/150))) 
    vis = max(0.1, 10 - (wind / 12)) 
    
    icon = "🌪️" if wind > 95 else "⛈️" if wind > 64 else "🌧️" if wind > 34 else "☁️"
    if r < (r_max * 0.3) and wind < 50: icon = "👁️" 
    
    prof = get_vertical_profile(lat, lon, s_lat, s_lon, p)
    upper_wind = prof[-1]['Wind (kts)']
    
    return f"""
    <div style="font-family: sans-serif; min-width: 180px;">
        <h4 style="margin:0;">{city}</h4>
        <hr style="margin:5px 0;">
        <b>{icon} SFC Wind:</b> {int(wind)} kts<br>
        <b>Temp/DP:</b> {temp:.1f}°F / {dewp:.1f}°F<br>
        <b>200mb Wind:</b> {upper_wind} kts<br>
        <hr style="margin:5px 0;">
        <i style="font-size: 0.85em;">Sounding available in Sidebar on click.</i>
    </div>
    """

def calculate_bay_surge(v_max, s_lon):
    dist_west = -88.0 - s_lon
    base = (v_max**2 / 1600)
    return base * (1.8 if dist_west > 0 else 0.5)

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="LHIM | Vertical Diagnostics")

st.markdown("""
    <style>
    .stSlider { padding-bottom: 0px; }
    .env-panel { background: rgba(255, 75, 75, 0.05); padding: 15px; border-radius: 10px; border: 1px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("🛡️ LHIM Diagnostics")
    
    with st.expander("🌊 SST Climatology", expanded=True):
        sst_season = st.selectbox("Window", ["Early (June/July)", "Peak (Aug/Sept)", "Late (Oct/Nov)"])
        sst_map = {"Early (June/July)": 0.85, "Peak (Aug/Sept)": 1.1, "Late (Oct/Nov)": 0.9}
        sst_mult = sst_map[sst_season]

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
    st.markdown("---")
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

sat_lats = np.linspace(28.0, 33.0, 60)
sat_lons = np.linspace(-91.0, -86.0, 60)
sat_data = []
for lt in sat_lats:
    for ln in sat_lons:
        _, _, ir_w = calculate_full_physics(lt, ln, l_lat, l_lon, p)
        if ir_w > 0.08: sat_data.append([lt, ln, ir_w])

wind_lats = np.linspace(29.6, 31.4, 20)
wind_lons = np.linspace(-88.9, -87.3, 20)

# --- 4. MAPPING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    tileset = "CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap"
    m = folium.Map(location=[30.5, -88.1], zoom_start=9, tiles=tileset)
    
    if show_ir and len(sat_data) > 0:
        HeatMap(sat_data, radius=25, blur=18, min_opacity=ir_opacity, max_zoom=13,
                gradient={0.2: 'gray', 0.4: 'white', 0.6: 'cyan', 0.8: 'red', 0.95: 'maroon'}).add_to(m)
    
    for lt in wind_lats:
        for ln in wind_lons:
            w, w_dir, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p, level=1000)
            if w > 30:
                color = 'red' if w > 95 else 'orange' if w > 75 else 'yellow' if w > 50 else 'blue'
                if show_circles:
                    popup_html = get_local_conditions(w, lt, ln, l_lat, l_lon, r_max, p)
                    folium.CircleMarker(
                        location=[lt, ln], radius=w/6, color=color, fill=True, weight=1, fill_opacity=0.3,
                        popup=folium.Popup(popup_html, max_width=250)
                    ).add_to(m)
            
            for lvl in active_levels:
                lw, _, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p, level=lvl)
                if lw > 15:
                    folium.CircleMarker(location=[lt, ln], radius=2, color='white', opacity=lvl_opacity).add_to(m)

    last_click = st_folium(m, width="100%", height=700)

with c2:
    st.metric("Est. Bay Surge", f"{surge_res:.1f} ft")
    st.write("---")
    
    if last_click and last_click.get("last_clicked"):
        clat, clon = last_click["last_clicked"]["lat"], last_click["last_clicked"]["lng"]
        st.subheader("📍 Vertical Profile")
        st.caption(f"Coord: {clat:.3f}, {clon:.3f} | Synthetic Sounding")
        
        pdf = pd.DataFrame(get_vertical_profile(clat, clon, l_lat, l_lon, p))
        st.table(pdf.set_index('Level (mb)'))
        
        st.line_chart(pdf[['Temp (°C)', 'Dewp (°C)']])
        st.caption("Diagonal arrows indicate direction of travel (Wind Barb equivalent).")
    else:
        st.info("Click map to generate sounding.")
    
    st.subheader("Structure Analytics")
    eye_check = (v_max / 100) * (1 - (shear_mag / 40)) * symmetry
    st.write(f"**Eye Definition:** {'HD' if eye_check > 0.8 else 'Ragged' if eye_check > 0.4 else 'None'}")
    st.progress(min(max(eye_check, 0.0), 1.0))
