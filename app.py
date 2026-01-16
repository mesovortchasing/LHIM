import streamlit as st
import numpy as np
import pandas as pd
import folium
import json
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & CLIMATOLOGY ENGINE ---

def get_sst_multiplier(month_name):
    """
    Climatological SST multipliers based on Atlantic Basin averages.
    Peak (Sept) provides the highest potential intensity.
    """
    climatology = {
        "June": 0.82, "July": 0.88, "August": 0.98, 
        "September": 1.10, "October": 0.94, "November": 0.80
    }
    return climatology.get(month_name, 1.0)

def get_wind_arrow(deg):
    arrows = ['↓', '↙', '←', '↖', '↑', '↗', '→', '↘']
    idx = int((deg + 22.5) % 360 / 45)
    return arrows[idx]

def calculate_full_physics(lat, lon, s_lat, s_lon, p, level=1000):
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
    
    ir_weight = np.exp(-r / (r_max * 6.0)) * (rh / 100) * (0.7 + max(0, np.sin(r/12 - angle*2.5) * 0.4))
    
    return max(0, total_wind), np.degrees(wind_angle_rad), ir_weight

def get_synthetic_radar(lat, lon, s_lat, s_lon, p):
    """Generates high-resolution Doppler-style reflectivity."""
    v_max, r_max, _, _, shear_mag, _, rh, _, symmetry, _ = p
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # 1. Primary Eyewall (Sharp Gradient)
    eyewall = 62 * np.exp(-((r - r_max)**2) / (r_max * 0.2)**2)
    
    # 2. Outer Eyewall / Bands
    outer_r = r_max * 2.3
    outer_eyewall = 45 * np.exp(-((r - outer_r)**2) / (r_max * 0.3)**2) if v_max > 95 else 0
    
    # 3. Moat Logic
    moat = 0.15 if (r_max * 1.3 < r < outer_r * 0.85 and v_max > 95) else 1.0
    
    # 4. Log-Spiral Rainbands
    band_mod = np.sin(r / (r_max * 0.6) - angle * 3.0)
    bands = max(0, band_mod * 48 * np.exp(-r / 140))
    
    # Composition
    dbz = (eyewall + outer_eyewall + bands + 8) * moat * (rh / 100) * symmetry
    
    # Eye Suppression
    if r < r_max * 0.45: dbz *= 0.02
    
    return min(75, dbz)

def get_dbz_color(dbz):
    """Standard NWS Reflectivity Palette."""
    if dbz < 15: return None
    if dbz < 20: return "#0099ff" # Light Blue
    if dbz < 30: return "#0000ff" # Blue
    if dbz < 40: return "#00ff00" # Green
    if dbz < 50: return "#ffff00" # Yellow
    if dbz < 60: return "#ff9900" # Orange
    return "#ff0000"             # Red (Heavy)

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="LHIM | Doppler Simulation")

with st.sidebar:
    st.title("🛡️ LHIM Doppler")
    
    with st.expander("📡 Climatology & Sensors", expanded=True):
        sel_month = st.selectbox("Climatological Month", ["June", "July", "August", "September", "October", "November"], index=3)
        sst_mult = get_sst_multiplier(sel_month)
        radar_mode = st.checkbox("Enable Doppler Reflectivity", value=True)
        radar_res = st.select_slider("Radar Grain (Resolution)", options=[40, 60, 80], value=60)
        radar_alpha = st.slider("Radar Opacity", 0.1, 1.0, 0.7)

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
    
    l_lat = st.number_input("Landfall Lat", value=30.35)
    l_lon = st.number_input("Landfall Lon", value=-88.15)
    map_theme = st.selectbox("Theme", ["Dark Mode", "Light Mode"])

p = [v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult]

# --- 3. MAPPING & GRIDDING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=8, 
                   tiles="CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap")

    # DOPPLER GRIDDING (Preserves structure at all zoom levels)
    if radar_mode:
        lat_range = np.linspace(l_lat - 2.5, l_lat + 2.5, radar_res)
        lon_range = np.linspace(l_lon - 3.0, l_lon + 3.0, radar_res)
        d_lat = lat_range[1] - lat_range[0]
        d_lon = lon_range[1] - lon_range[0]

        for lt in lat_range:
            for ln in lon_range:
                dbz = get_synthetic_radar(lt, ln, l_lat, l_lon, p)
                color = get_dbz_color(dbz)
                if color:
                    # Draw a pixel rectangle to mimic doppler bins
                    folium.Rectangle(
                        bounds=[[lt, ln], [lt + d_lat, ln + d_lon]],
                        color=color, fill=True, fill_color=color,
                        fill_opacity=radar_alpha, weight=0, stroke=False,
                        interactive=False
                    ).add_to(m)

    # Observation Points (Wind Markers)
    for lt in np.linspace(l_lat-1.2, l_lat+1.2, 12):
        for ln in np.linspace(l_lon-1.5, l_lon+1.5, 12):
            w, _, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p)
            if w > 35:
                # Inspector Popup logic
                dist = np.sqrt(((ln-l_lon)*53)**2 + ((lt-l_lat)*69)**2)
                temp, dewp = 75.0 - (dist/20), 74.5 - (dist/20)
                popup_html = f"""
                <div style='font-family: sans-serif; min-width: 140px; background: #111; color: #eee; padding: 8px; border-radius: 5px;'>
                <b>Area Reading</b><br><hr style='margin:4px 0;'>
                🌪️ <b>Condition:</b> {int(w)} kts<br>
                🌡️ <b>Temp:</b> {temp:.1f}°F<br>
                💧 <b>Dew Pt:</b> {dewp:.1f}°F<br>
                👁️ <b>Vis:</b> {max(0.1, 10-(w/12)):.1f} mi
                </div>"""
                folium.CircleMarker(
                    location=[lt, ln], radius=w/9, color='white', weight=1, fill=False,
                    popup=folium.Popup(popup_html, max_width=200)
                ).add_to(m)

    last_click = st_folium(m, width="100%", height=780)

with c2:
    st.subheader("📍 Cell Inspector")
    if last_click and last_click.get("last_clicked"):
        clat, clon = last_click["last_clicked"]["lat"], last_click["last_clicked"]["lng"]
        levels = [1000, 925, 850, 500, 200]
        p_data = []
        for lvl in levels:
            w, wd, _ = calculate_full_physics(clat, clon, l_lat, l_lon, p, level=lvl)
            p_data.append({"Level": lvl, "Wind": int(w), "Dir": get_wind_arrow(wd)})
        st.table(pd.DataFrame(p_data).set_index("Level"))
    else:
        st.info("Click the map to sample the atmosphere.")
        
    st.metric("SST Potency (Month)", f"{sel_month}", f"{sst_mult:.2f}x")
    st.write("---")
    st.caption("Radar engine uses bin-gridding to maintain anatomical realism (moats/bands) regardless of zoom.")
