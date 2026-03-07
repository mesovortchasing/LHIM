import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import time

# --- 1. CORE PHYSICS & RADAR ENGINE ---

def get_sst_mult(month, sst_boost=False):
    months = {"June": 0.85, "July": 0.92, "August": 1.05, "September": 1.15, "October": 1.02, "November": 0.88}
    base = months.get(month, 1.0)
    return base * 1.2 if sst_boost else base

def calculate_full_physics(lat, lon, s_lat, s_lon, p, level=1000, micro_scale=0.0, front_lat=None):
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    if r < 1: r = 1
    
    front_mod = 1.0
    if front_lat and lat > front_lat:
        front_dist = abs(lat - front_lat)
        front_mod = max(0.5, 1.0 - (front_dist * 0.2))
        rh = min(100, rh + 15)

    level_scale = {1000: 1.0, 925: 0.92, 850: 0.85, 500: 0.45, 200: 0.25}
    l_mult = level_scale.get(level, 1.0)
    eff_v = v_max * (rh / 85.0) * (0.6 + outflow / 2.5) * sst_mult * l_mult * front_mod
    
    B = 1.3 + (eff_v / 150)
    v_sym = eff_v * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + ((1.0 - symmetry) * (shear_mag / 45)) * np.cos(angle - shear_rad)
    
    inflow = 25 if level > 500 else -30
    wind_angle_rad = angle + (np.pi / 2) + np.radians(inflow if r > r_max else inflow/2)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    return max(0, (v_sym * shear_effect) + v_forward), np.degrees(wind_angle_rad), r

def get_synthetic_products(lat, lon, s_lat, s_lon, p, radar_coords=None, micro_scale=0.0, front_lat=None):
    v_max, r_max, _, _, shear_mag, shear_dir, rh, _, symmetry, _ = p
    w, wd, r = calculate_full_physics(lat, lon, s_lat, s_lon, p, micro_scale=micro_scale, front_lat=front_lat)
    angle = np.arctan2((lat - s_lat) * 69, (lon - s_lon) * 53)
    
    is_major = v_max >= 96
    shear_rad = np.radians(shear_dir)
    shear_offset_x = (shear_mag / 20) * np.cos(shear_rad)
    shear_offset_y = (shear_mag / 20) * np.sin(shear_rad)
    r_adj = np.sqrt(((lon - s_lon - shear_offset_x/53) * 53)**2 + ((lat - s_lat - shear_offset_y/69) * 69)**2)
    
    eyewall = 65 * np.exp(-((r - r_max)**2) / (r_max * 0.3)**2)
    moisture_flux = (rh / 100) * (1 + (shear_mag / 100))
    shield = 42 * moisture_flux * np.exp(-r_adj / (r_max * 5.0)) 
    bands = max(0, np.sin(r / (r_max * 0.8) - angle * 3.0) * 35 * np.exp(-r / 200))
    
    front_rain = 30 * np.exp(-abs(lat - front_lat) * 2) if (front_lat and lat > front_lat - 0.5) else 0

    dbz = max(eyewall, shield, bands, front_rain) * symmetry
    if r < r_max * (0.15 if is_major else 0.4): dbz *= 0.1 

    if radar_coords:
        rdx, rdy = (lon - radar_coords[1]) * 53, (lat - radar_coords[0]) * 69
        angle_to_radar = np.arctan2(rdy, rdx)
        radial_v = (w * 1.15) * np.cos(np.radians(wd) - angle_to_radar)
        aliased_v = np.clip(radial_v, -149, 149)
    else: aliased_v = 0

    # Hyper-Realistic Surge Physics
    surge = 0
    if 30.00 <= lat <= 30.55: # Expanded coastal shelf
        dist_mult = np.exp(-abs(lat - 30.25) * 5)
        # Surge peaks on the right side (onshore flow)
        onshore_factor = np.sin(np.radians(wd) - np.arctan2(0, 1)) # Relative to north/south coast
        surge = (w**1.9 / 1700) * (1.8 if lon > s_lon else -0.5) * dist_mult
    
    prob = 90 if w >= 96 else 60 if w >= 64 else 30 if w >= 34 else 0
    return min(78, dbz), aliased_v, surge, prob

# --- 2. SESSION STATE ---
if 'active_radar' not in st.session_state: st.session_state.active_radar = "KMOB"
if 'loop_idx' not in st.session_state: st.session_state.loop_idx = 0
RADAR_SITES = {"KMOB": (30.67, -88.24), "KLIX": (30.33, -89.82), "KEVX": (30.56, -85.92)}

# --- 3. UI & SIDEBAR ---
st.set_page_config(layout="wide", page_title="LHIM v2.7 | Advanced Surge")

with st.sidebar:
    st.title("🛡️ LHIM v2.7")
    radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge", "Wind Prob."])
    
    with st.expander("⚠️ Warning Settings", expanded=False):
        show_warnings = st.checkbox("Overlay Surge Warnings", value=True)
        surge_threshold = st.slider("Warning Trigger (ft)", 3, 12, 6)

    run_loop = st.checkbox("🔄 Enable Radar Loop")
    
    with st.expander("🌡️ Environmental Layers", expanded=True):
        sst_boost = st.toggle("Warm Sea Surface (SST+)", value=True)
        front_lat = st.slider("Cold Front Latitude", 30.0, 32.5, 31.8)
        shear_mag = st.slider("Deep Layer Shear (kts)", 0, 80, 15)
        rh = st.slider("Fluid RH (%)", 30, 100, 88)

    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    
    if run_loop:
        st.session_state.loop_idx = (st.session_state.loop_idx + 1) % 13
        current_time_offset = st.session_state.loop_idx
    else:
        current_time_offset = st.slider("Time Offset (Hrs)", 0, 12, 0)

    f_speed, f_dir = st.slider("Forward Speed", 2, 40, 12), st.slider("Heading", 0, 360, 330)
    l_lat, l_lon = st.number_input("Landfall Lat", value=30.35), st.number_input("Landfall Lon", value=-88.15)
    res_steps = st.select_slider("Quality", options=[30, 45, 60], value=45)

dist_back = (f_speed * current_time_offset) / 69.0
current_lat = l_lat - (dist_back * np.cos(np.radians(f_dir)))
current_lon = l_lon - (dist_back * np.sin(np.radians(f_dir)))
p = [v_max, r_max, f_speed, f_dir, shear_mag, 240, rh, 0.8, 0.85, get_sst_mult("September", sst_boost)]

# --- 4. MAP ---
c1, c2 = st.columns([4, 1.5])
with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=8, tiles="CartoDB DarkMatter")
    radar_coords = RADAR_SITES[st.session_state.active_radar]
    
    lats = np.linspace(l_lat-2.5, l_lat+2.5, res_steps)
    lons = np.linspace(l_lon-2.5, l_lon+2.5, int(res_steps * 1.2))
    d_lat, d_lon = lats[1]-lats[0], lons[1]-lons[0]
    
    for lt in lats:
        for ln in lons:
            dbz, vel, surge, prob = get_synthetic_products(lt, ln, current_lat, current_lon, p, radar_coords, front_lat=front_lat)
            color = None
            if radar_view == "Reflectivity (dBZ)" and dbz > 15:
                color = '#ff00ff' if dbz > 65 else '#ff0000' if dbz > 50 else '#ff9900' if dbz > 40 else '#ffff00' if dbz > 28 else '#00ff00'
            elif radar_view == "Velocity (kts)":
                if vel < -5: color = '#00ffff' if vel < -110 else '#00ccff' if vel < -75 else '#00aa00'
                elif vel > 5: color = '#ff00ff' if vel > 110 else '#ff0000' if vel > 75 else '#880000'
            elif radar_view == "Storm Surge" and abs(surge) > 0.5:
                # Dynamic Surge Palette 
                if surge > 12: color = '#4b0082' # Extreme (Purple)
                elif surge > 9: color = '#8b0000' # Critical (Dark Red)
                elif surge > 6: color = '#ff0000' # Life-Threatening (Red)
                elif surge > 3: color = '#ff8c00' # Major (Orange)
                elif surge > 0: color = '#ffd700' # Moderate (Yellow)
                else: color = '#00ced1' # Recession (Cyan)
            
            if color:
                folium.Rectangle(bounds=[[lt, ln], [lt+d_lat, ln+d_lon]], color=color, fill=True, fill_opacity=0.6, weight=0).add_to(m)
                if show_warnings and surge > surge_threshold:
                    folium.CircleMarker(location=[lt, ln], radius=2, color="white", weight=1).add_to(m)

    map_data = st_folium(m, width="100%", height=720, key=f"map_{st.session_state.loop_idx}", returned_objects=["last_clicked"])

with c2:
    st.subheader("📊 Dynamic Inspection")
    if map_data and map_data.get("last_clicked"):
        clat, clon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        dbz, vel, surge, _ = get_synthetic_products(clat, clon, current_lat, current_lon, p, radar_coords, front_lat=front_lat)
        w_kts, wd, r = calculate_full_physics(clat, clon, current_lat, current_lon, p, front_lat=front_lat)
        
        # Localized Weather Physics
        t_base = 84 if sst_boost else 80
        local_temp = t_base - (r * 0.08) - (dbz * 0.1)
        local_dewp = local_temp - (100 - rh) * 0.15
        gust = w_kts * (1.35 if w_kts > 90 else 1.22)

        st.metric("Sustained Wind", f"{int(w_kts)} kts", f"{int(gust)} kts Gust")
        st.metric("Storm Surge", f"{surge:.1f} ft", delta_color="inverse" if surge > 6 else "normal")
        st.metric("Ambient Temp", f"{local_temp:.1f}°F", f"Dew: {local_dewp:.1f}°F")
        st.write(f"**Reflectivity:** {dbz:.1f} dBZ")
        if surge > surge_threshold:
            st.error(f"⚠️ LIFE-THREATENING SURGE INUNDATION AT THIS LOCATION")
    else:
        st.info("Click the map to analyze localized surge and weather data.")

if run_loop:
    time.sleep(0.05)
    st.rerun()
