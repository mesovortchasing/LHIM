import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import time
import random

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
    
    # Mesovort Injection
    mv_bonus = 0
    if micro_scale > 0 and abs(r - r_max) < (r_max * 0.4):
        for i in range(4):
            mv_angle = (time.time() * 0.5) + (i * np.pi / 2)
            mv_x, mv_y = r_max * np.cos(mv_angle), r_max * np.sin(mv_angle)
            dist_to_mv = np.sqrt((dx - mv_x)**2 + (dy - mv_y)**2)
            mv_bonus += (micro_scale * 25) * np.exp(-(dist_to_mv**2) / (r_max * 0.1)**2)
    
    inflow = 25 if level > 500 else -30
    wind_angle_rad = angle + (np.pi / 2) + np.radians(inflow if r > r_max else inflow/2)
    
    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + ((1.0 - symmetry) * (shear_mag / 40)) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    sustained = max(0, (v_sym * shear_effect) + v_forward + mv_bonus)
    # Gust Factor influenced by microphysics turbulence
    gust = sustained * (1.22 + (micro_scale * 0.35))
    
    return sustained, gust, np.degrees(wind_angle_rad), r

def get_synthetic_products(lat, lon, s_lat, s_lon, p, nyquist=65, active_radar_coords=None, micro_scale=0.0):
    v_max, r_max, _, _, _, _, rh, _, symmetry, _ = p
    w, gust, wd, r = calculate_full_physics(lat, lon, s_lat, s_lon, p, micro_scale=micro_scale)
    angle = np.arctan2((lat - s_lat) * 69, (lon - s_lon) * 53)
    
    eyewall = 60 * np.exp(-((r - r_max)**2) / (r_max * 0.25)**2)
    if micro_scale > 0: eyewall *= (1 + (micro_scale * 0.3 * np.sin(angle * 5)))
        
    bands = max(0, np.sin(r / (r_max * 0.7) - angle * 2.5) * 40 * np.exp(-r / 150))
    dbz = (eyewall + bands + 18) * (rh / 100) * symmetry
    if r < r_max * 0.35: dbz *= 0.1 

    if active_radar_coords:
        rdx, rdy = (lon - active_radar_coords[1]) * 53, (lat - active_radar_coords[0]) * 69
        angle_to_radar = np.arctan2(rdy, rdx)
        radial_v = w * np.cos(np.radians(wd) - angle_to_radar)
        aliased_v = ((radial_v + nyquist) % (2 * nyquist)) - nyquist
    else:
        aliased_v = 0

    surge = 0
    # Realistic surge: Vector alignment between wind flow and coast (Assuming coast is east-west 30.3N)
    if 30.10 <= lat <= 30.45:
        wind_towards_coast = np.sin(np.radians(wd)) # Positive means blowing North (inshore)
        surge_stress = (w**2 / 1900)
        surge = max(0, surge_stress * (0.4 + (wind_towards_coast * 1.1)))
    
    return min(75, dbz), aliased_v, surge

# --- 2. NHC MESSAGE GENERATOR ---
def get_nhc_alert(v_max):
    alerts = [
        f"URGENT: Hurricane Warning remains in effect. Sustained winds {v_max}kts. Extreme wind hazard expected near the RMW.",
        f"DANGER: Storm Surge Warning. Life-threatening inundation likely. Localized surge up to {v_max/15:.1f}ft possible.",
        f"ADVISORY: Microphysics analysis shows significant mesovortical development in the eyewall. Tornado-force gusts likely."
    ]
    return random.choice(alerts)

# --- 3. UI SETUP ---
RADAR_SITES = {"KMOB": (30.67, -88.24), "KLIX": (30.33, -89.82), "KEVX": (30.56, -85.92)}
if 'active_radar' not in st.session_state: st.session_state.active_radar = "KMOB"

st.set_page_config(layout="wide", page_title="LHIM | Tactical Pro v3")

with st.sidebar:
    st.title("🛡️ LHIM Pro v3.0")
    radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge"])
    show_alerts = st.toggle("Enable NHC Warnings", value=True)
    
    with st.expander("🌀 Simulation Control", expanded=True):
        micro_scale = st.slider("Microphysics Scale", 0.0, 1.0, 0.4)
        time_offset = st.slider("Temporal Loop (h)", 0, 12, 0)
        
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    f_speed = st.slider("Fwd Speed", 2, 40, 12)
    f_dir = st.slider("Heading", 0, 360, 330)
    l_lat, l_lon = st.number_input("Landfall Lat", 30.35), st.number_input("Landfall Lon", -88.15)

dist_back = (f_speed * time_offset) / 69.0
move_rad = np.radians(f_dir)
current_lat, current_lon = l_lat - (dist_back * np.cos(move_rad)), l_lon - (dist_back * np.sin(move_rad))
p = [v_max, r_max, f_speed, f_dir, 8, 260, 85, 0.8, 0.85, get_sst_mult("September")]

# --- 4. MAPPING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=8, tiles="CartoDB DarkMatter")
    
    # Legend Injection
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; right: 30px; width: 200px; z-index:9999; 
    background-color: rgba(20,20,20,0.9); border:1px solid #00ffff; color:white; padding:12px; 
    font-family: 'Courier New', Courier, monospace; border-radius: 8px; font-size: 11px;">
    <b style="color:#00ffff;">SYSTEM SCALES</b><hr style="margin:5px 0;">
    <div style="background: linear-gradient(to right, #004400, #00ff00, #ffff00, #ff0000); height:12px;"></div>
    <span>REF: 20 . 40 . 60 . 75 dBZ</span><br><br>
    <div style="background: linear-gradient(to right, #0000ff, #8888ff, #ffffff, #ff8888, #ff0000); height:12px;"></div>
    <span>VEL: -65 . 0 . +65 kts</span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Grid Calculation for Fluid Map
    res = 40
    lats = np.linspace(current_lat-2.5, current_lat+2.5, res)
    lons = np.linspace(current_lon-2.5, current_lon+2.5, int(res * 1.2))
    heat_data = []
    
    radar_coords = RADAR_SITES[st.session_state.active_radar]
    
    for lt in lats:
        for ln in lons:
            dbz, vel, surge = get_synthetic_products(lt, ln, current_lat, current_lon, p, 65, radar_coords, micro_scale)
            if radar_view == "Reflectivity (dBZ)" and dbz > 20:
                heat_data.append([lt, ln, dbz/75])
            elif radar_view == "Velocity (kts)":
                heat_data.append([lt, ln, abs(vel)/65])
            elif radar_view == "Storm Surge" and surge > 1.0:
                folium.Circle(location=[lt, ln], radius=1500, color='#00ffff', fill=True, opacity=0.3, weight=0).add_to(m)

    if radar_view != "Storm Surge":
        HeatMap(heat_data, radius=22, blur=18, min_opacity=0.3).add_to(m)

    for name, coords in RADAR_SITES.items():
        folium.CircleMarker(location=coords, radius=8, color="cyan" if st.session_state.active_radar == name else "white", 
                            fill=True, popup=f"Radar Site: {name}").add_to(m)

    map_data = st_folium(m, width="100%", height=750)

    if map_data.get("last_object_clicked_popup"):
        clicked_name = map_data["last_object_clicked_popup"].split(": ")[-1]
        if clicked_name in RADAR_SITES:
            st.session_state.active_radar = clicked_name
            st.rerun()

with c2:
    if show_alerts:
        if st.button("📩 CLICK FOR LATEST NHC ADVISORY"):
            st.warning(get_nhc_alert(v_max))
        else:
            st.error(f"**WATCH/WARNING ACTIVE** \nMultiple coastal hazards in effect.")

    st.subheader("📍 Site Inspector")
    if map_data and map_data.get("last_clicked"):
        clat, clon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        w, gust, wd, r = calculate_full_physics(clat, clon, current_lat, current_lon, p, micro_scale=micro_scale)
        dbz, vel, surge = get_synthetic_products(clat, clon, current_lat, current_lon, p, 65, radar_coords, micro_scale)
        
        # Temp/Dewp thermodynamics
        temp = 82 - (70 / (r + 4))
        dewp = temp - (1.5 * (1 - (85/100)))

        st.markdown(f"""
        <div style="background:#0a1a2a; padding:20px; border-radius:12px; border:1px solid #00ffff; color:white;">
        <h1 style="margin:0; text-align:center;">{get_wind_arrow(wd)}</h1>
        <div style="display:flex; justify-content:space-around; margin-top:10px;">
            <div style="text-align:center;"><small>SUSTAINED</small><br><b style="font-size:24px;">{int(w)}</b><br><small>kts</small></div>
            <div style="text-align:center;"><small>GUSTS</small><br><b style="font-size:24px; color:#ff4b4b;">{int(gust)}</b><br><small>kts</small></div>
        </div>
        <hr style="border-color:#333;">
        <b>🌡️ Temp:</b> {temp:.1f}°F &nbsp; | &nbsp; <b>💧 Dewpoint:</b> {dewp:.1f}°F<br>
        <b>🌊 Storm Surge:</b> <span style="color:#00ffff;">{surge:.1f} ft</span><br>
        <b>📡 Doppler:</b> {dbz:.1f} dBZ / {vel:.1f} kts
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Drop a probe on the map to see local thermodynamics and wind vectors.")

    st.markdown("---")
    st.caption("v3.0 Engine: Vector-based surge, temporal looping, and fluid doppler heatmapping enabled.")
