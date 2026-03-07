import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import time

# --- 1. CORE PHYSICS & RADAR ENGINE ---

def get_sst_mult(month):
    months = {"June": 0.85, "July": 0.92, "August": 1.05, "September": 1.15, "October": 1.02, "November": 0.88}
    return months.get(month, 1.0)

def calculate_full_physics(lat, lon, s_lat, s_lon, p, level=1000, micro_scale=0.0):
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    if r < 1: r = 1
    
    level_scale = {1000: 1.0, 925: 0.92, 850: 0.85, 500: 0.45, 200: 0.25}
    l_mult = level_scale.get(level, 1.0)
    eff_v = v_max * (rh / 85.0) * (0.6 + outflow / 2.5) * sst_mult * l_mult
    
    B = 1.3 + (eff_v / 150)
    v_sym = eff_v * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    # Mesovort Injection
    mv_bonus = 0
    if micro_scale > 0 and abs(r - r_max) < (r_max * 0.4):
        for i in range(4):
            mv_angle = (time.time() * 0.5) + (i * np.pi / 2)
            mv_x = r_max * np.cos(mv_angle)
            mv_y = r_max * np.sin(mv_angle)
            dist_to_mv = np.sqrt((dx - mv_x)**2 + (dy - mv_y)**2)
            mv_bonus += (micro_scale * 25) * np.exp(-(dist_to_mv**2) / (r_max * 0.1)**2)
    
    inflow = 25 if level > 500 else -30
    wind_angle_rad = angle + (np.pi / 2) + np.radians(inflow if r > r_max else inflow/2)
    
    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + ((1.0 - symmetry) * (shear_mag / 40)) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    return max(0, (v_sym * shear_effect) + v_forward + mv_bonus), np.degrees(wind_angle_rad), r

def get_synthetic_products(lat, lon, s_lat, s_lon, p, radar_coords=None, micro_scale=0.0):
    v_max, r_max, _, _, _, _, rh, _, symmetry, _ = p
    w_kts, wd, r = calculate_full_physics(lat, lon, s_lat, s_lon, p, micro_scale=micro_scale)
    w_mph = w_kts * 1.15078 # Convert to mph for your +/- 149 request
    
    angle = np.arctan2((lat - s_lat) * 69, (lon - s_lon) * 53)
    
    # --- ENHANCED REFLECTIVITY (Fuller Structure) ---
    # Core eyewall
    eyewall = 65 * np.exp(-((r - r_max)**2) / (r_max * 0.6)**2) 
    # Broad rain shield (Filling the hurricane)
    shield = 35 * np.exp(-r / (r_max * 4)) 
    # Spiral bands
    bands = max(0, np.sin(r / (r_max * 0.8) - angle * 3.0) * 25 * np.exp(-r / 200))
    
    dbz = (eyewall + shield + bands) * (rh / 100) * (0.5 + 0.5 * symmetry)
    if r < r_max * 0.25: dbz *= 0.1 # Clearer eye
    
    # --- FLUID VELOCITY (Radial) ---
    radial_v_mph = 0
    if radar_coords:
        rdx, rdy = (lon - radar_coords[1]) * 53, (lat - radar_coords[0]) * 69
        angle_to_radar = np.arctan2(rdy, rdx)
        # Component of wind moving toward/away from radar
        radial_v_mph = w_mph * np.cos(np.radians(wd) - angle_to_radar)
        # Cap at your requested 149mph
        radial_v_mph = np.clip(radial_v_mph, -149, 149)

    # Surge & Prob Logic
    surge = (w_kts**2 / 2000) * (1.4 if np.sin(angle) > 0 else 0.4) if 30.10 <= lat <= 30.45 else 0
    prob = 90 if w_kts >= 96 else 60 if w_kts >= 64 else 30 if w_kts >= 34 else 0

    return min(75, dbz), radial_v_mph, surge, prob

# --- 2. RADAR SITES ---
RADAR_SITES = {"KMOB": (30.67, -88.24), "KLIX": (30.33, -89.82), "KEVX": (30.56, -85.92)}

if 'active_radar' not in st.session_state:
    st.session_state.active_radar = "KMOB"

# --- 3. UI ---
st.set_page_config(layout="wide", page_title="LHIM | Velocity Update")

with st.sidebar:
    st.title("🛡️ LHIM v2.6")
    radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (mph)", "Storm Surge", "Wind Prob."])
    micro_scale = st.slider("Microphysics Scale", 0.0, 1.0, 0.4)
    time_offset = st.slider("Radar Loop (Hours Ago)", 0, 12, 0)
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    f_speed, f_dir = st.slider("Forward Speed", 2, 40, 12), st.slider("Heading", 0, 360, 330)
    l_lat, l_lon = st.number_input("Lat", value=30.35), st.number_input("Lon", value=-88.15)
    res_steps = st.select_slider("Quality", options=[30, 45, 60, 80], value=45)

# Temporal Adjustment
dist_back = (f_speed * time_offset) / 69.0
current_lat = l_lat - (dist_back * np.cos(np.radians(f_dir)))
current_lon = l_lon - (dist_back * np.sin(np.radians(f_dir)))
p = [v_max, r_max, f_speed, f_dir, 8, 260, 85, 0.8, 0.85, get_sst_mult("September")]

# --- 4. MAP RENDERING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=8, tiles="CartoDB DarkMatter")
    radar_coords = RADAR_SITES[st.session_state.active_radar]

    lats = np.linspace(l_lat-3.0, l_lat+3.0, res_steps)
    lons = np.linspace(l_lon-3.0, l_lon+3.0, int(res_steps * 1.2))
    d_lat, d_lon = lats[1]-lats[0], lons[1]-lons[0]

    for lt in lats:
        for ln in lons:
            dbz, vel, surge, prob = get_synthetic_products(lt, ln, current_lat, current_lon, p, radar_coords, micro_scale)
            color = None
            
            # REFLECTIVITY: Fuller, smoother gradients
            if radar_view == "Reflectivity (dBZ)" and dbz > 15:
                if dbz > 55: color = '#7a0000' # Deep Maroon
                elif dbz > 50: color = '#ff0000' # Red
                elif dbz > 40: color = '#ff9900' # Orange
                elif dbz > 30: color = '#ffff00' # Yellow
                elif dbz > 20: color = '#00ff00' # Green
                else: color = '#006600' # Dark Green

            # VELOCITY: Inbound (Green/Blue) vs Outbound (Red/Yellow)
            elif radar_view == "Velocity (mph)" and abs(vel) > 5:
                # Inbound (Negative values = moving toward radar)
                if vel < 0:
                    if vel < -100: color = '#003300' # Deep Inbound
                    elif vel < -64: color = '#00cc00' # Strong Inbound
                    else: color = '#99ff99' # Light Inbound
                # Outbound (Positive values = moving away from radar)
                else:
                    if vel > 100: color = '#660000' # Deep Outbound
                    elif vel > 64: color = '#ff0000' # Strong Outbound
                    else: color = '#ffcccc' # Light Outbound

            if color:
                folium.Rectangle(bounds=[[lt, ln], [lt+d_lat, ln+d_lon]], color=color, fill=True, fill_opacity=0.7, weight=0).add_to(m)

    st_folium(m, width="100%", height=750)

with c2:
    st.subheader("📊 Radar Diagnostics")
    st.write(f"**Velocity Range:** -149 to +149 mph")
    st.write("🟢 **Inbound:** Wind moving toward radar.")
    st.write("🔴 **Outbound:** Wind moving away from radar.")
    st.info("The reflectivity rain-shield has been expanded to simulate a mature hurricane 'fill'.")
