import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & RADAR ENGINE ---

def get_sst_mult(month):
    months = {"June": 0.85, "July": 0.92, "August": 1.05, "September": 1.15, "October": 1.02, "November": 0.88}
    return months.get(month, 1.0)

def get_wind_arrow(deg):
    arrows = ['↓', '↙', '←', '↖', '↑', '↗', '→', '↘']
    idx = int((deg + 22.5) % 360 / 45)
    return arrows[idx]

def calculate_full_physics(lat, lon, s_lat, s_lon, p, level=1000):
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
    
    inflow = 25 if level > 500 else -30
    wind_angle_rad = angle + (np.pi / 2) + np.radians(inflow if r > r_max else inflow/2)
    
    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + ((1.0 - symmetry) * (shear_mag / 40)) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    return max(0, (v_sym * shear_effect) + v_forward), np.degrees(wind_angle_rad), r

def get_synthetic_products(lat, lon, s_lat, s_lon, p, nyquist=60, active_radar_coords=None):
    v_max, r_max, _, _, _, _, rh, _, symmetry, _ = p
    w, wd, r = calculate_full_physics(lat, lon, s_lat, s_lon, p)
    angle = np.arctan2((lat - s_lat) * 69, (lon - s_lon) * 53)
    
    # Reflectivity logic
    eyewall = 60 * np.exp(-((r - r_max)**2) / (r_max * 0.25)**2)
    bands = max(0, np.sin(r / (r_max * 0.7) - angle * 2.5) * 40 * np.exp(-r / 150))
    dbz = (eyewall + bands + 18) * (rh / 100) * symmetry
    if r < r_max * 0.35: dbz *= 0.1 

    # Velocity Relative to specific Radar Site
    if active_radar_coords:
        rdx, rdy = (lon - active_radar_coords[1]) * 53, (lat - active_radar_coords[0]) * 69
        angle_to_radar = np.arctan2(rdy, rdx)
        radial_v = w * np.cos(np.radians(wd) - angle_to_radar)
        aliased_v = ((radial_v + nyquist) % (2 * nyquist)) - nyquist
    else:
        aliased_v = 0

    # Coastal-Only Surge logic
    surge = 0
    # Mask: Only allow surge if Lat is near coast (approx 30.1 - 30.4 for this sim) 
    # and Lon is within the storm's effective radius.
    is_coastal = 30.10 <= lat <= 30.45 
    if is_coastal:
        surge = (w**2 / 2000) * (1.4 if np.sin(angle) > 0 else 0.4)
    
    prob = 90 if w >= 96 else 60 if w >= 64 else 30 if w >= 34 else 0

    return min(75, dbz), aliased_v, surge, prob

# --- 2. SESSION STATE & RADAR SITES ---
RADAR_SITES = {
    "KMOB": (30.67, -88.24),
    "KLIX": (30.33, -89.82),
    "KEVX": (30.56, -85.92)
}

if 'active_radar' not in st.session_state:
    st.session_state.active_radar = "KMOB"

# --- 3. UI & SIDEBAR ---
st.set_page_config(layout="wide", page_title="LHIM | Coastal Alpha")

with st.sidebar:
    st.title("🛡️ LHIM v2.0")
    radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge", "Wind Prob."])
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 12)
    f_dir = st.slider("Heading (Deg)", 0, 360, 330)
    rh, symmetry = st.slider("Humidity", 30, 100, 85), st.slider("Symmetry", 0.0, 1.0, 0.85)
    l_lat, l_lon = st.number_input("Landfall Lat", value=30.35), st.number_input("Landfall Lon", value=-88.15)
    res_steps = st.select_slider("Performance / Quality", options=[30, 45, 60], value=45)

p = [v_max, r_max, f_speed, f_dir, 8, 260, rh, 0.8, symmetry, get_sst_mult("September")]

# --- 4. MAP & LOGIC ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=8, tiles="CartoDB DarkMatter")
    
    # Render Radar Site Markers (Clickable)
    for name, coords in RADAR_SITES.items():
        color = "red" if st.session_state.active_radar == name else "gray"
        folium.Marker(
            location=coords,
            popup=f"Radar: {name}",
            icon=folium.Icon(color=color, icon="broadcast-tower", prefix="fa")
        ).add_to(m)

    # Fast Grid Generation
    lats = np.linspace(l_lat-2.0, l_lat+2.0, res_steps)
    lons = np.linspace(l_lon-2.5, l_lon+2.5, int(res_steps * 1.2))
    d_lat, d_lon = lats[1]-lats[0], lons[1]-lons[0]
    
    radar_coords = RADAR_SITES[st.session_state.active_radar]
    
    for lt in lats:
        for ln in lons:
            dbz, vel, surge, prob = get_synthetic_products(lt, ln, l_lat, l_lon, p, 65, radar_coords)
            color = None
            
            if radar_view == "Reflectivity (dBZ)" and dbz > 18:
                color = '#ff0000' if dbz > 50 else '#ff9900' if dbz > 40 else '#ffff00' if dbz > 30 else '#00ff00'
            elif radar_view == "Velocity (kts)":
                v_norm = np.clip(vel / 65, -1, 1)
                color = '#ff0000' if v_norm > 0.6 else '#ff9999' if v_norm > 0 else '#99ff99' if v_norm > -0.6 else '#00aa00'
            elif radar_view == "Storm Surge" and surge > 1.2:
                color = '#330066' if surge > 10 else '#0033ff' if surge > 5 else '#00ffff'
            elif radar_view == "Wind Prob." and prob > 0:
                color = '#ff00ff' if prob >= 90 else '#ff6600' if prob >= 60 else '#ffff00'

            if color:
                folium.Rectangle(
                    bounds=[[lt, ln], [lt+d_lat, ln+d_lon]],
                    color=color, fill=True, fill_opacity=0.6, weight=0
                ).add_to(m)

    # Capture interaction
    map_data = st_folium(m, width="100%", height=700)

    # Logic to switch radar based on marker click
    if map_data.get("last_object_clicked_popup"):
        clicked_name = map_data["last_object_clicked_popup"].split(": ")[-1]
        if clicked_name in RADAR_SITES and clicked_name != st.session_state.active_radar:
            st.session_state.active_radar = clicked_name
            st.rerun()

with c2:
    st.subheader("📊 Tactical Analysis")
    st.info(f"Active Radar: **{st.session_state.active_radar}**")
    
    if map_data and map_data.get("last_clicked"):
        clat, clon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        dbz, vel, surge, prob = get_synthetic_products(clat, clon, l_lat, l_lon, p, 65, radar_coords)
        w, _, _ = calculate_full_physics(clat, clon, l_lat, l_lon, p)
        
        st.metric("Local Wind", f"{int(w)} kts", f"{int(w*1.15)} mph")
        st.metric("Storm Surge", f"{surge:.1f} ft")
        
        if surge > 0:
            st.warning("⚠️ Coastal Inundation Active at this point.")
        else:
            st.success("✅ Location Inland/Protected from Surge.")
            
        st.write(f"**Reflectivity:** {dbz:.1f} dBZ")
        st.write(f"**Radial Vel:** {vel:.1f} kts")
    else:
        st.write("Click map for point inspection.")

    st.markdown("---")
    st.caption("Surge masked to coastal shelf (30.1N-30.45N). Click grayscale markers to switch radar sites.")
