import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & WIND VECTOR ENGINE ---

def calculate_full_physics(lat, lon, s_lat, s_lon, p):
    """
    Enhanced Physics: Wind, Rain-Coupled IR, Dynamic Eye, and Symmetry Stretching.
    """
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry = p
    
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    if r < 1: r = 1
    
    env_mult = (rh / 85.0) * (0.6 + outflow / 2.5)
    effective_v_max = v_max * env_mult
    
    B = 1.3 + (effective_v_max / 150)
    v_sym = effective_v_max * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    inflow_angle = np.radians(25 if r > r_max else 15)
    wind_angle_rad = angle + (np.pi / 2) + inflow_angle
    
    shear_rad = np.radians(shear_dir)
    asym_weight = 1.0 + (1.0 - symmetry) 
    shear_effect = 1 + (asym_weight * shear_mag / 60) * np.cos(angle - shear_rad)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    total_wind = (v_sym * shear_effect) + v_forward
    if lat > 30.25: total_wind *= 0.85 
    
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

def get_local_conditions(wind, lat, lon, s_lat, s_lon, r_max):
    """Generates localized weather data for popups based on physics."""
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    
    # Logic for City names based on Mobile County Map
    city = "Mobile" if lat > 30.6 else "Dauphin Island" if lat < 30.3 else "Theodore/Bayou La Batre"
    
    # Physics-based weather derivation
    temp = 82 - (wind / 15) # Pressure drop/rain cools air
    dewp = temp - (2 * (1 - (wind/150))) # High humidity in core
    vis = max(0.1, 10 - (wind / 12)) # Rain/Spray reduces visibility
    
    icon = "🌪️" if wind > 95 else "⛈️" if wind > 64 else "🌧️" if wind > 34 else "☁️"
    if r < (r_max * 0.3) and wind < 50: icon = "👁️" # Eye calm
    
    return f"""
    <div style="font-family: sans-serif; min-width: 140px;">
        <h4 style="margin:0;">{city} Area</h4>
        <hr style="margin:5px 0;">
        <b>{icon} Condition:</b> {int(wind)} kts<br>
        <b>Temp:</b> {temp:.1f}°F<br>
        <b>Dew Pt:</b> {dewp:.1f}°F<br>
        <b>Visibility:</b> {vis:.1f} mi
    </div>
    """

def calculate_bay_surge(v_max, s_lon):
    dist_west = -88.0 - s_lon
    base = (v_max**2 / 1600)
    return base * (1.8 if dist_west > 0 else 0.5)

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Mobile Hurr-Sim V4")

st.markdown("""
    <style>
    .legend { position: fixed; bottom: 30px; left: 30px; width: 220px; background: rgba(255,255,255,0.9); 
              padding: 15px; border: 2px solid #333; z-index: 9999; border-radius: 10px; font-family: sans-serif; }
    .legend i { width: 12px; height: 12px; display: inline-block; margin-right: 8px; border-radius: 50%; }
    </style>
    <div class="legend">
        <b>Chaser Legend</b><br>
        <i style="background:red"></i> Severe (>95 kts)<br>
        <i style="background:orange"></i> Extensive (75-95)<br>
        <i style="background:yellow"></i> Moderate (50-75)<br>
        <i style="background:blue"></i> TS Force (34-50)<br>
        <hr>
        <b>SIMUSAT:</b><br>
        <span style="color:red">●</span> Deep Convection<br>
        <span style="color:cyan">●</span> Heavy Rain Rates<br>
    </div>
    """, unsafe_allow_html=True)

st.title("🌀 Mobile County Interactive Chaser Model (Alpha V4)")

with st.sidebar:
    st.header("1. Environmental Factors")
    rh = st.slider("Mid-Level Humidity (%)", 30, 100, 85)
    outflow = st.slider("Outflow Efficiency", 0.0, 1.0, 0.8)
    symmetry = st.slider("Storm Symmetry", 0.0, 1.0, 0.85)
    
    st.header("2. Core Parameters")
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 12)
    f_dir = st.slider("Storm Heading (Deg)", 0, 360, 330)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    
    st.header("3. Upper Level Shear")
    shear_mag = st.slider("Shear Magnitude (kts)", 0, 60, 8)
    shear_dir = st.slider("Shear Direction (From)", 0, 360, 260)
    
    st.header("4. Display Settings")
    show_ir = st.checkbox("Toggle SIMUSAT (Fluid IR)", value=True)
    ir_opacity = st.slider("SIMUSAT Opacity", 0.0, 1.0, 0.15)
    show_barbs = st.checkbox("Show Wind Flow Vectors", value=True)
    l_lat = st.number_input("Landfall Lat", value=30.35)
    l_lon = st.number_input("Landfall Lon", value=-88.15)
    
    st.header("5. Misc")
    map_theme = st.select_slider("Map Theme", options=["Dark Mode", "Light Mode"], value="Dark Mode")
    show_circles = st.checkbox("Show Wind Circles", value=True)

# --- 3. COMPUTATION ---
p = [v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry]
surge_res = calculate_bay_surge(v_max, l_lon)

sat_lats = np.linspace(28.0, 33.0, 70)
sat_lons = np.linspace(-91.0, -86.0, 70)

wind_lats = np.linspace(29.6, 31.4, 25)
wind_lons = np.linspace(-88.9, -87.3, 25)

sat_data = []
for lt in sat_lats:
    for ln in sat_lons:
        _, _, ir_w = calculate_full_physics(lt, ln, l_lat, l_lon, p)
        if ir_w > 0.08:
            sat_data.append([lt, ln, ir_w])

results = []
for lt in wind_lats:
    for ln in wind_lons:
        w, w_dir, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p)
        results.append([lt, ln, w, w_dir])

df = pd.DataFrame(results, columns=['lat', 'lon', 'wind', 'dir'])

# --- 4. MAPPING ---
c1, c2 = st.columns([4, 1])

with c1:
    tileset = "CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap"
    m = folium.Map(location=[30.5, -88.1], zoom_start=9, tiles=tileset)
    
    if show_ir and len(sat_data) > 0:
        HeatMap(sat_data, radius=25, blur=18, min_opacity=ir_opacity, max_zoom=13,
                gradient={0.2: 'gray', 0.4: 'white', 0.6: 'cyan', 0.8: 'red', 0.95: 'maroon'}).add_to(m)
    
    for _, row in df.iterrows():
        if row['wind'] > 30:
            color = 'red' if row['wind'] > 95 else 'orange' if row['wind'] > 75 else 'yellow' if row['wind'] > 50 else 'blue'
            if show_circles:
                popup_html = get_local_conditions(row['wind'], row['lat'], row['lon'], l_lat, l_lon, r_max)
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']], 
                    radius=row['wind']/6,
                    color=color, fill=True, weight=1, fill_opacity=0.4,
                    popup=folium.Popup(popup_html, max_width=200)
                ).add_to(m)
            
            if show_barbs:
                length = 0.02
                end_lat = row['lat'] + length * np.sin(np.radians(row['dir']))
                end_lon = row['lon'] + length * np.cos(np.radians(row['dir']))
                v_color = 'white' if map_theme == "Dark Mode" else 'black'
                folium.PolyLine(locations=[[row['lat'], row['lon']], [end_lat, end_lon]],
                                color=v_color, weight=1, opacity=0.6).add_to(m)

    st_folium(m, width="100%", height=750)

with c2:
    st.metric("Est. Bay Surge", f"{surge_res:.1f} ft")
    st.write("---")
    st.subheader("Structure Analytics")
    st.write(f"**Max Wind:** {int(df['wind'].max())} kts")
    health = "Organized" if symmetry > 0.7 and shear_mag < 15 else "Sheared/Elongated"
    st.write(f"**Storm State:** {health}")
    eye_check = (v_max / 100) * (1 - (shear_mag / 40)) * symmetry
    st.write(f"**Eye Definition:** {'High-Definition' if eye_check > 0.8 else 'Ragged' if eye_check > 0.4 else 'None'}")
