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

def get_wind_arrow(deg):
    arrows = ['↓', '↙', '←', '↖', '↑', '↗', '→', '↘']
    idx = int((deg + 22.5) % 360 / 45)
    return arrows[idx]

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
    
    # Mesovort Injection (Fluid Microphysics)
    mv_bonus = 0
    if micro_scale > 0 and abs(r - r_max) < (r_max * 0.4):
        # Create 4 orbiting mesovorts
        for i in range(4):
            mv_angle = (time.time() * 0.5) + (i * np.pi / 2)
            mv_x = r_max * np.cos(mv_angle)
            mv_y = r_max * np.sin(mv_angle)
            dist_to_mv = np.sqrt((dx - mv_x)**2 + (dy - mv_y)**2)
            mv_bonus += (micro_scale * 25) * np.exp(-(dist_to_mv**2) / (r_max * 0.1)**2)
    
    inflow = 25 if level > 500 else -30
    wind_angle_rad = angle + (np.pi / 2) + np.radians(inflow if r > r_max else inflow/2)
    
    # Keep your exact shear and forward speed math
    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + ((1.0 - symmetry) * (shear_mag / 40)) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    return max(0, (v_sym * shear_effect) + v_forward + mv_bonus), np.degrees(wind_angle_rad), r

def get_synthetic_products(lat, lon, s_lat, s_lon, p, nyquist=60, active_radar_coords=None, micro_scale=0.0):
    v_max, r_max, _, _, _, _, rh, _, symmetry, _ = p
    w, wd, r = calculate_full_physics(lat, lon, s_lat, s_lon, p, micro_scale=micro_scale)
    angle = np.arctan2((lat - s_lat) * 69, (lon - s_lon) * 53)
    
    # Reflectivity logic with Mesovort Enhancement + Fuller Shield
    eyewall = 60 * np.exp(-((r - r_max)**2) / (r_max * 0.25)**2)
    shield = 35 * np.exp(-r / (r_max * 4.0)) # Added this to make it "fuller"
    if micro_scale > 0:
        eyewall *= (1 + (micro_scale * 0.3 * np.sin(angle * 5))) # Granular texture
        
    bands = max(0, np.sin(r / (r_max * 0.7) - angle * 2.5) * 40 * np.exp(-r / 150))
    dbz = (max(eyewall, shield) + bands + 18) * (rh / 100) * symmetry
    if r < r_max * 0.35: dbz *= 0.1 

    if active_radar_coords:
        rdx, rdy = (lon - active_radar_coords[1]) * 53, (lat - active_radar_coords[0]) * 69
        angle_to_radar = np.arctan2(rdy, rdx)
        # Using 1.15 to convert kts to mph for your +/- 149 request
        radial_v = (w * 1.15) * np.cos(np.radians(wd) - angle_to_radar)
        aliased_v = np.clip(radial_v, -149, 149)
    else:
        aliased_v = 0

    surge = 0
    is_coastal = 30.10 <= lat <= 30.45 
    if is_coastal:
        # RIGHT SIDE (lon > s_lon) gets surge; LEFT SIDE gets recession
        surge_mult = 1.4 if lon > s_lon else -0.8
        surge = (w**2 / 2000) * surge_mult
    
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
st.set_page_config(layout="wide", page_title="LHIM | Microphysics Alpha")

with st.sidebar:
    st.title("🛡️ LHIM v2.5")
    radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge", "Wind Prob."])
    
    with st.expander("🌀 Advanced Physics", expanded=True):
        micro_scale = st.slider("Microphysics Scale", 0.0, 1.0, 0.4, help="Adds fluid mesovorts and granular precipitation physics.")
        time_offset = st.slider("Radar Loop (Hours Ago)", 0, 12, 0, help="Rewind the storm position based on current motion vectors.")
        
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 12)
    f_dir = st.slider("Heading (Deg)", 0, 360, 330)
    rh, symmetry = st.slider("Humidity", 30, 100, 85), st.slider("Symmetry", 0.0, 1.0, 0.85)
    l_lat, l_lon = st.number_input("Landfall Lat", value=30.35), st.number_input("Landfall Lon", value=-88.15)
    res_steps = st.select_slider("Performance / Quality", options=[30, 45, 60], value=45)

# Calculate temporal storm center
dist_back = (f_speed * time_offset) / 69.0
move_rad = np.radians(f_dir)
current_lat = l_lat - (dist_back * np.cos(move_rad))
current_lon = l_lon - (dist_back * np.sin(move_rad))

p = [v_max, r_max, f_speed, f_dir, 8, 260, rh, 0.8, symmetry, get_sst_mult("September")]

# --- 4. MAP & LOGIC ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=8, tiles="CartoDB DarkMatter")
    
    if time_offset > 0:
        folium.PolyLine([[current_lat, current_lon], [l_lat, l_lon]], color="white", weight=2, dash_array='5, 10', opacity=0.5).add_to(m)

    for name, coords in RADAR_SITES.items():
        color = "red" if st.session_state.active_radar == name else "gray"
        folium.Marker(location=coords, popup=f"Radar: {name}", icon=folium.Icon(color=color, icon="broadcast-tower", prefix="fa")).add_to(m)

    lats = np.linspace(l_lat-2.5, l_lat+2.5, res_steps)
    lons = np.linspace(l_lon-2.5, l_lon+2.5, int(res_steps * 1.2))
    d_lat, d_lon = lats[1]-lats[0], lons[1]-lons[0]
    
    radar_coords = RADAR_SITES[st.session_state.active_radar]
    
    for lt in lats:
        for ln in lons:
            dbz, vel, surge, prob = get_synthetic_products(lt, ln, current_lat, current_lon, p, 149, radar_coords, micro_scale)
            color = None
            
            if radar_view == "Reflectivity (dBZ)" and dbz > 18:
                color = '#ff0000' if dbz > 50 else '#ff9900' if dbz > 40 else '#ffff00' if dbz > 30 else '#00ff00'
            elif radar_view == "Velocity (kts)":
                # Realistic Inbound/Outbound RadarScope logic
                if vel < -5:
                    color = '#00ffff' if vel < -100 else '#0055ff' if vel < -60 else '#00aa00'
                elif vel > 5:
                    color = '#ff00ff' if vel > 100 else '#ff0000' if vel > 60 else '#880000'
            elif radar_view == "Storm Surge" and abs(surge) > 1.2:
                color = '#0033ff' if surge > 0 else '#ffcc00'
            elif radar_view == "Wind Prob." and prob > 0:
                color = '#ff00ff' if prob >= 90 else '#ff6600' if prob >= 60 else '#ffff00'

            if color:
                folium.Rectangle(bounds=[[lt, ln], [lt+d_lat, ln+d_lon]], color=color, fill=True, fill_opacity=0.6, weight=0).add_to(m)

    # ADDING DYNAMIC LEGEND
    legend_html = f'''
     <div style="position: fixed; bottom: 50px; left: 50px; width: 140px; height: 100px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:12px; padding:10px;">
     <b>{radar_view}</b><br>
     Max: <span style="color:red">High</span><br>
     Min: <span style="color:blue">Low</span>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    map_data = st_folium(m, width="100%", height=750)

    if map_data.get("last_object_clicked_popup"):
        clicked_name = map_data["last_object_clicked_popup"].split(": ")[-1]
        if clicked_name in RADAR_SITES and clicked_name != st.session_state.active_radar:
            st.session_state.active_radar = clicked_name
            st.rerun()

with c2:
    st.subheader("📊 Microphysics Analysis")
    st.info(f"Looping: **T-{time_offset}h** | Radar: **{st.session_state.active_radar}**")
    
    if map_data and map_data.get("last_clicked"):
        clat, clon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        dbz, vel, surge, prob = get_synthetic_products(clat, clon, current_lat, current_lon, p, 149, radar_coords, micro_scale)
        w, _, _ = calculate_full_physics(clat, clon, current_lat, current_lon, p, micro_scale=micro_scale)
        
        st.metric("Point Wind", f"{int(w)} kts", f"{(micro_scale*w*0.15):+.1f} MV Effect")
        st.metric("Radial Velocity", f"{int(vel)} mph")
        st.metric("Storm Surge", f"{surge:.1f} ft")
        
        st.write(f"**Vorticity Intensity:** {'High' if micro_scale > 0.7 else 'Moderate' if micro_scale > 0.3 else 'Low'}")
        st.write(f"**Reflectivity:** {dbz:.1f} dBZ")
    else:
        st.write("Click map for point inspection.")

    st.markdown("---")
    st.caption(f"Micro-vortices enabled at {micro_scale*100}%. Storm center adjusted to T-{time_offset}h coordinates: {current_lat:.2f}, {current_lon:.2f}.")
