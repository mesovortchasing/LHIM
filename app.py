import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & RADAR ENGINE ---

def get_noise(lat, lon, seed=42):
    """Generates pseudo-random spatial noise to make radar look 'jagged'."""
    return (np.sin(lat * 50) * np.cos(lon * 50) * np.sin(lat * lon * 10))

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
    
    env_mult = (rh / 85.0) * (0.6 + outflow / 2.5) * sst_mult
    eff_v = v_max * env_mult * l_mult
    
    B = 1.3 + (eff_v / 150)
    v_sym = eff_v * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    inflow = 25 if level > 500 else -30
    wind_angle_rad = angle + (np.pi / 2) + np.radians(inflow if r > r_max else inflow/2)
    
    shear_rad = np.radians(shear_dir)
    shear_factor = (shear_mag / 40) if level < 500 else (shear_mag / 60)
    shear_effect = 1 + ((1.0 - symmetry) * shear_factor) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    total_wind = (v_sym * shear_effect) + v_forward
    ir_w = np.exp(-r / (r_max * 6.0)) * (rh / 100)
    
    return max(0, total_wind), np.degrees(wind_angle_rad), ir_w

def get_synthetic_products(lat, lon, s_lat, s_lon, p, nyquist=60):
    v_max, r_max, _, _, shear_mag, shear_dir, rh, _, symmetry, _ = p
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    noise = get_noise(lat, lon)
    
    # 1. Reflectivity (dBZ) - FREE FORM ENGINE
    # Displace bands based on shear direction
    shear_offset = (shear_mag / 20) * np.cos(angle - np.radians(shear_dir))
    eyewall = 62 * np.exp(-((r - r_max)**2) / (r_max * (0.2 + shear_offset*0.1))**2)
    
    # Fragmented rainbands using noise and sin waves
    band_freq = r / (r_max * 0.75) - angle * 2.5
    bands = np.sin(band_freq) * 45 * np.exp(-r / 150) * (0.5 + 0.5 * noise)
    
    dbz = (eyewall + bands + 18) * (rh / 100) * symmetry
    if r < r_max * 0.35: dbz *= (0.1 + 0.2 * noise) # Ragged eye
    
    # 2. Velocity
    w, wd, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p)
    radial_v = w * np.cos(np.radians(wd) - angle)
    aliased_v = ((radial_v + nyquist) % (2 * nyquist)) - nyquist
    
    # 3. Surge (Coastal Masking: Lat 30.1 - 30.4 focus)
    surge = 0
    if 30.15 < lat < 30.55: # Simulating the Gulf Coast shelf
        surge = (w**2 / 1850) * (1.3 if np.sin(angle) > 0 else 0.2)
    
    # 4. Wind Prob
    prob = (w / v_max) * 100 * symmetry

    return min(75, dbz), aliased_v, surge, prob

def get_local_conditions(lat, lon, s_lat, s_lon, p):
    w, _, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p)
    dist = np.sqrt(((lon-s_lon)*53)**2 + ((lat-s_lat)*69)**2)
    temp, dewp = 76.0 - (dist / 20), 75.0 - (dist / 20)
    return f"""<div style='background:#111;color:#eee;padding:8px;border-radius:5px;'>
    <b>Area Conditions</b><hr>Wind: {int(w)} kts<br>Temp: {temp:.1f}°F<br>Vis: {max(0.1, 10-(w/15)):.1f} mi</div>"""

def get_vertical_profile(lat, lon, s_lat, s_lon, p):
    levels = [1000, 925, 850, 500, 200]
    profile = []
    for lvl in levels:
        w, wd, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p, level=lvl)
        dist = np.sqrt(((lon-s_lon)*53)**2 + ((lat-s_lat)*69)**2)
        temp = (28 if lvl > 800 else -5 if lvl > 400 else -55) + max(0, 10 - dist/8)
        profile.append({"Level": lvl, "Wind": int(w), "Barb": get_wind_arrow(wd), "Temp": round(temp,1)})
    return profile

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="LHIM | Alpha")

def add_legend(m, mode):
    legends = {
        "Reflectivity (dBZ)": ("Reflectivity", ["#0000ff", "#00ff00", "#ffff00", "#ff9900", "#ff0000"], ["20", "30", "40", "50", "60+"]),
        "Velocity (kts)": ("Radial Vel", ["#00aa00", "#99ff99", "#ff9999", "#ff0000"], ["Toward", "-30", "+30", "Away"]),
        "Storm Surge": ("Surge (ft)", ["#00ffff", "#0033ff", "#330066"], ["1-4", "5-10", "12+"]),
        "Wind Prob.": ("Prob (%)", ["#ffcc00", "#ff3300", "#800000"], ["25", "50", "80+"])
    }
    title, colors, labels = legends[mode]
    html = f'''<div style="position:fixed; top:10px; right:70px; width:110px; z-index:9999; background:rgba(0,0,0,0.7); 
    color:white; padding:10px; border-radius:8px; font-family:sans-serif; font-size:11px; border:1px solid #555;">
    <b>{title}</b><br>'''
    for c, l in zip(colors, labels):
        html += f'<i style="background:{c}; width:10px; height:10px; float:left; margin-right:5px; margin-top:3px;"></i>{l}<br>'
    html += '</div>'
    m.get_root().html.add_child(folium.Element(html))

with st.sidebar:
    st.title("🛡️ LHIM Alpha")
    radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge", "Wind Prob."])
    radar_alpha = st.slider("Layer Opacity", 0.1, 1.0, 0.6)
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    shear_mag = st.slider("Shear (kts)", 0, 60, 15)
    shear_dir = st.slider("Shear From", 0, 360, 260)
    l_lat = st.number_input("Landfall Lat", value=30.35)
    l_lon = st.number_input("Landfall Lon", value=-88.15)
    map_theme = st.selectbox("Theme", ["Dark Mode", "Light Mode"])

p = [v_max, r_max, 12, 330, shear_mag, shear_dir, 85, 0.8, 0.85, 1.02]

# --- 3. MAPPING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=9, tiles="CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap")
    add_legend(m, radar_view)
    
    radar_group = folium.FeatureGroup(name="Active Layer")
    lats, lons = np.linspace(l_lat-2.5, l_lat+2.5, 55), np.linspace(l_lon-3.0, l_lon+3.0, 65)
    d_lat, d_lon = lats[1]-lats[0], lons[1]-lons[0]

    for lt in lats:
        for ln in lons:
            dbz, vel, surge, prob = get_synthetic_products(lt, ln, l_lat, l_lon, p, 65)
            color = None
            if radar_view == "Reflectivity (dBZ)" and dbz > 18:
                color = '#ff0000' if dbz > 52 else '#ff9900' if dbz > 42 else '#ffff00' if dbz > 32 else '#00ff00' if dbz > 22 else '#0000ff'
            elif radar_view == "Velocity (kts)":
                v_norm = np.clip(vel / 65, -1, 1)
                color = '#ff0000' if v_norm > 0.6 else '#ff9999' if v_norm > 0 else '#99ff99' if v_norm > -0.6 else '#00aa00'
            elif radar_view == "Storm Surge" and surge > 1.0:
                color = '#330066' if surge > 10 else '#0033ff' if surge > 5 else '#00ffff'
            elif radar_view == "Wind Prob." and prob > 25:
                color = '#800000' if prob > 75 else '#ff3300' if prob > 50 else '#ffcc00'
            
            if color:
                folium.Rectangle(bounds=[[lt, ln], [lt+d_lat, ln+d_lon]], color=color, fill=True, fill_color=color, fill_opacity=radar_alpha, weight=0).add_to(radar_group)
    
    radar_group.add_to(m)

    # Condition Markers
    for lt in np.linspace(l_lat-1.2, l_lat+1.2, 10):
        for ln in np.linspace(l_lon-1.5, l_lon+1.5, 10):
            w, _, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p)
            if w > 40:
                folium.CircleMarker(location=[lt, ln], radius=w/10, color='white', fill=True, weight=1, fill_opacity=0.2,
                                    popup=folium.Popup(get_local_conditions(lt, ln, l_lat, l_lon, p), max_width=200)).add_to(m)

    last_click = st_folium(m, width="100%", height=700, key="lhim_map")

with c2:
    st.subheader("📍 Conditions Tracker")
    if last_click and last_click.get("last_clicked"):
        clat, clon = last_click["last_clicked"]["lat"], last_click["last_clicked"]["lng"]
        st.table(pd.DataFrame(get_vertical_profile(clat, clon, l_lat, l_lon, p)).set_index('Level'))
    else:
        st.info("Click a marker to view vertical profile.")
    st.markdown("### Structural Health")
    st.progress(min(max((v_max/160) * (1-(shear_mag/60)), 0.0), 1.0))
