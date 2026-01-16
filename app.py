import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# --- 1. CORE PHYSICS & RADAR ENGINE ---

def get_sst_mult(month):
    """Climatological SST multipliers for the Atlantic Basin."""
    months = {
        "June": 0.85, "July": 0.92, "August": 1.05, 
        "September": 1.15, "October": 1.02, "November": 0.88
    }
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
    ir_w = np.exp(-r / (r_max * 6.0)) * (rh / 100) * (0.7 + max(0, np.sin(r/12 - angle*2.5)*0.4))
    
    return max(0, total_wind), np.degrees(wind_angle_rad), ir_w

def get_hourly_forecast(lat, lon, s_lat, s_lon, p):
    """Generates a 6-hour hyper-realistic forecast based on storm motion and physics."""
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p
    forecast_data = []
    
    for h in range(1, 7):
        # Calculate storm displacement (approx degrees per hour)
        dist_move = (f_speed * h) / 69.0 
        move_rad = np.radians(f_dir)
        new_s_lat = s_lat + (dist_move * np.cos(move_rad))
        new_s_lon = s_lon + (dist_move * np.sin(move_rad))
        
        # Land Friction Factor: Decay intensity if center moves inland
        friction = 0.85 if new_s_lat > 30.4 else 1.0
        p_current = [v_max * friction, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult]
        
        w_kts, wd, _ = calculate_full_physics(lat, lon, new_s_lat, new_s_lon, p_current)
        w_mph = w_kts * 1.15
        
        # Color Coding logic
        color = "#ffffff" # Default
        if w_mph >= 106: color = "#ff4b4b" # Red
        elif w_mph >= 76: color = "#ffa500" # Orange
        elif w_mph >= 45: color = "#ffff00" # Yellow
        
        icon = "🌀" if w_mph > 74 else "🌧️" if w_mph > 39 else "☁️"
        
        forecast_data.append({
            "Hour": f"+{h}h",
            "Icon": icon,
            "Wind (mph)": int(w_mph),
            "Direction": get_wind_arrow(wd),
            "Color": color
        })
    return forecast_data

def get_synthetic_products(lat, lon, s_lat, s_lon, p, nyquist=60):
    v_max, r_max, _, _, _, _, rh, _, symmetry, _ = p
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    eyewall = 60 * np.exp(-((r - r_max)**2) / (r_max * 0.22)**2)
    outer_r = r_max * 2.4
    outer_eyewall = 42 * np.exp(-((r - outer_r)**2) / (r_max * 0.4)**2) if v_max > 105 else 0
    moat = 0.45 if (r_max * 1.4 < r < outer_r * 0.85 and v_max > 105) else 1.0
    bands = max(0, np.sin(r / (r_max * 0.6) - angle * 2.8) * 44 * np.exp(-r / 135))
    dbz = (eyewall + outer_eyewall + bands + 15) * moat * (rh / 100) * symmetry
    if r < r_max * 0.4: dbz *= 0.05 
    
    w, wd, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p)
    radial_v = w * np.cos(np.radians(wd) - angle)
    aliased_v = ((radial_v + nyquist) % (2 * nyquist)) - nyquist
    
    surge = 0
    if lat > 30.20:
        surge = (w**2 / 1900) * (1.3 if np.sin(angle) > 0 else 0.3)
    
    prob = (w / v_max) * 100 * symmetry

    return min(75, dbz), aliased_v, surge, prob

def get_local_conditions(lat, lon, s_lat, s_lon, p):
    w, _, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p)
    dist = np.sqrt(((lon-s_lon)*53)**2 + ((lat-s_lat)*69)**2)
    temp, dewp = 76.0 - (dist / 18), 75.2 - (dist / 18)
    return f"""
    <div style="font-family: sans-serif; min-width: 160px; background: #111; color: #eee; padding: 10px; border-radius: 8px;">
        <b style="font-size: 1.1em;">Location Area</b><br>
        <hr style="margin: 5px 0; border: 0.1px solid #444;">
        🌪️ <b>Condition:</b> {int(w)} kts<br>
        🌡️ <b>Temp:</b> {temp:.1f}°F<br>
        💧 <b>Dew Pt:</b> {dewp:.1f}°F<br>
        👁️ <b>Visibility:</b> {max(0.1, 10-(w/12)):.1f} mi
    </div>
    """

def get_vertical_profile(lat, lon, s_lat, s_lon, p):
    levels = [1000, 925, 850, 500, 200]
    profile = []
    for lvl in levels:
        w, wd, _ = calculate_full_physics(lat, lon, s_lat, s_lon, p, level=lvl)
        dist = np.sqrt(((lon-s_lon)*53)**2 + ((lat-s_lat)*69)**2)
        temp = (30 if lvl > 800 else 0 if lvl > 400 else -50) + max(0, (15 - dist/5))
        dewp = temp - (dist/10) - (lvl/1000 * 5)
        profile.append({"Level": lvl, "Wind": int(w), "Barb": get_wind_arrow(wd), "Temp": round(temp,1), "Dewp": round(dewp,1)})
    return profile

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="LHIM | Alpha")

def add_legend(m, mode):
    legends = {
        "Reflectivity (dBZ)": ("Reflectivity", ["#0000ff", "#00ff00", "#ffff00", "#ff9900", "#ff0000"], ["20", "30", "40", "50", "60+"]),
        "Velocity (kts)": ("Radial Vel", ["#00aa00", "#99ff99", "#ff9999", "#ff0000"], ["Toward", "-30", "+30", "Away"]),
        "Storm Surge": ("Surge (ft)", ["#00ffff", "#0033ff", "#330066"], ["1-5", "6-12", "15+"]),
        "Wind Prob.": ("Prob (%)", ["#ffcc00", "#ff3300", "#800000"], ["20", "50", "80+"])
    }
    title, colors, labels = legends[mode]
    html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 120px; z-index:9999; background: rgba(0,0,0,0.8); 
    color: white; padding: 10px; border-radius: 5px; font-family: sans-serif; font-size: 12px; border: 1px solid #444;">
    <b>{title}</b><br>'''
    for c, l in zip(colors, labels):
        html += f'<i style="background: {c}; width: 12px; height: 12px; float: left; margin-right: 5px; opacity: 0.8;"></i> {l}<br>'
    html += '</div>'
    m.get_root().html.add_child(folium.Element(html))

with st.sidebar:
    st.title("🛡️ LHIM Alpha")
    with st.expander("📡 Sensors & Climatology", expanded=True):
        month = st.selectbox("Month", ["June", "July", "August", "September", "October", "November"], index=3)
        p_sst = get_sst_mult(month)
        radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge", "Wind Prob."])
        nyquist = st.slider("Nyquist Limit", 30, 100, 65)
        radar_alpha = st.slider("Layer Opacity", 0.1, 1.0, 0.65)
    st.header("1. Core Parameters")
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 12)
    f_dir = st.slider("Heading (Deg)", 0, 360, 330)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    st.header("2. Environment")
    rh, outflow, symmetry = st.slider("Humidity", 30, 100, 85), st.slider("Outflow", 0.0, 1.0, 0.8), st.slider("Symmetry", 0.0, 1.0, 0.85)
    shear_mag, shear_dir = st.slider("Shear (kts)", 0, 60, 8), st.slider("Shear From", 0, 360, 260)
    l_lat, l_lon = st.number_input("Landfall Lat", value=30.35), st.number_input("Landfall Lon", value=-88.15)
    map_theme = st.selectbox("Theme", ["Dark Mode", "Light Mode"])

p = [v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, p_sst]

# --- 3. MAPPING ---
c1, c2 = st.columns([4, 1.5])
with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=9, tiles="CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap")
    add_legend(m, radar_view)
    radar_group = folium.FeatureGroup(name="LHIM Layers")
    lats, lons = np.linspace(l_lat-2.5, l_lat+2.5, 55), np.linspace(l_lon-3.0, l_lon+3.0, 65)
    d_lat, d_lon = lats[1]-lats[0], lons[1]-lons[0]
    for lt in lats:
        for ln in lons:
            dbz, vel, surge, prob = get_synthetic_products(lt, ln, l_lat, l_lon, p, nyquist)
            color = None
            if radar_view == "Reflectivity (dBZ)" and dbz > 15:
                color = '#ff0000' if dbz > 50 else '#ff9900' if dbz > 40 else '#ffff00' if dbz > 30 else '#00ff00' if dbz > 20 else '#0000ff'
            elif radar_view == "Velocity (kts)":
                v_norm = np.clip(vel / nyquist, -1, 1)
                color = '#ff0000' if v_norm > 0.6 else '#ff9999' if v_norm > 0 else '#99ff99' if v_norm > -0.6 else '#00aa00'
            elif radar_view == "Storm Surge" and surge > 1.5:
                color = '#330066' if surge > 12 else '#0033ff' if surge > 6 else '#00ffff'
            elif radar_view == "Wind Prob." and prob > 20:
                color = '#800000' if prob > 80 else '#ff3300' if prob > 50 else '#ffcc00'
            if color:
                folium.Rectangle(bounds=[[lt, ln], [lt+d_lat, ln+d_lon]], color=color, fill=True, fill_color=color, fill_opacity=radar_alpha, weight=0).add_to(radar_group)
    radar_group.add_to(m)
    for lt in np.linspace(l_lat-1.2, l_lat+1.2, 12):
        for ln in np.linspace(l_lon-1.5, l_lon+1.5, 12):
            w, _, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p)
            if w > 35:
                folium.CircleMarker(location=[lt, ln], radius=w/8, color='white', fill=True, weight=1, fill_opacity=0.3,
                                    popup=folium.Popup(get_local_conditions(lt, ln, l_lat, l_lon, p), max_width=200)).add_to(m)
    last_click = st_folium(m, width="100%", height=750, key="lhim_alpha_map")

with c2:
    st.subheader("📍 Conditions Tracker")
    if last_click and last_click.get("last_clicked"):
        clat, clon = last_click["last_clicked"]["lat"], last_click["last_clicked"]["lng"]
        pdf = pd.DataFrame(get_vertical_profile(clat, clon, l_lat, l_lon, p))
        st.caption(f"Sampling: {clat:.2f}, {clon:.2f}")
        
        # New Hourly Forecast Option
        with st.expander("🕒 View 6-Hour Hourly Forecast", expanded=False):
            h_data = get_hourly_forecast(clat, clon, l_lat, l_lon, p)
            for row in h_data:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px; border-bottom: 1px solid #333;">
                    <span style="font-weight: bold; width: 40px;">{row['Hour']}</span>
                    <span style="font-size: 1.2em; width: 30px;">{row['Icon']}</span>
                    <span style="color: {row['Color']}; font-weight: bold; width: 80px;">{row['Wind (mph)']} mph</span>
                    <span style="font-size: 1.1em;">{row['Direction']}</span>
                </div>
                """, unsafe_allow_html=True)

        st.table(pdf.set_index('Level'))
        st.line_chart(pdf[['Temp', 'Dewp']])
    else:
        st.info("Click any marker to track local conditions.")
    st.metric("SST Influence", f"{month}", f"{p_sst:.2f}x")
    st.progress(min(max((v_max/160) * symmetry, 0.0), 1.0))
    st.caption("Intensity Efficiency Profile")
