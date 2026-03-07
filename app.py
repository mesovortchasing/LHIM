import streamlit as st
import numpy as np
import folium
from streamlit_folium import st_folium
import time
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import base64

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
    
    eff_v = v_max * (rh / 85.0) * (0.6 + outflow / 2.5) * sst_mult
    B = 1.3 + (eff_v / 150)
    v_sym = eff_v * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    mv_bonus = 0
    if micro_scale > 0 and abs(r - r_max) < (r_max * 0.4):
        for i in range(4):
            mv_angle = (time.time() * 0.3) + (i * np.pi / 2)
            mv_x, mv_y = r_max * np.cos(mv_angle), r_max * np.sin(mv_angle)
            dist_to_mv = np.sqrt((dx - mv_x)**2 + (dy - mv_y)**2)
            mv_bonus += (micro_scale * 35) * np.exp(-(dist_to_mv**2) / (r_max * 0.15)**2)
    
    wind_angle_rad = angle + (np.pi / 2) + np.radians(20 if r > r_max else 10)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))
    
    sustained = max(0, v_sym + v_forward + mv_bonus)
    gust = sustained * (1.25 + (micro_scale * 0.3))
    return sustained, gust, np.degrees(wind_angle_rad), r

def get_radar_array(lats, lons, current_lat, current_lon, p, micro_scale, mode, radar_coords):
    # vectorized grid calculation for performance
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    dx = (lon_grid - current_lon) * 53
    dy = (lat_grid - current_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    v_max, r_max = p[0], p[1]
    
    if mode == "Reflectivity (dBZ)":
        eyewall = 65 * np.exp(-((r - r_max)**2) / (r_max * 0.3)**2)
        bands = np.sin(r / (r_max * 0.8) - angle * 3) * 35 * np.exp(-r / 120)
        data = np.clip(eyewall + bands + 15, 0, 75)
        data[r < r_max * 0.3] *= 0.1 # Clear eye
    else:
        # Velocity logic
        rdx, rdy = (lon_grid - radar_coords[1]) * 53, (lat_grid - radar_coords[0]) * 69
        angle_to_radar = np.arctan2(rdy, rdx)
        w = v_max * np.sqrt((r_max / r)**1.5 * np.exp(1 - (r_max / r)**1.5))
        wd = angle + (np.pi / 2)
        radial_v = w * np.cos(wd - angle_to_radar)
        data = np.clip(radial_v, -80, 80)
        
    return data

# --- 2. UI & ALERTS ---
def get_nhc_alert():
    return random.choice([
        "HURRICANE WARNING: Catastrophic wind damage expected near landfall.",
        "STORM SURGE ADVISORY: Inundation of 8-12ft above ground level possible.",
        "RADAR UPDATE: Eyewall mesovortices detected. Tornadoes likely in right-front quadrant."
    ])

RADAR_SITES = {"KMOB": (30.67, -88.24), "KLIX": (30.33, -89.82), "KEVX": (30.56, -85.92)}
if 'active_radar' not in st.session_state: st.session_state.active_radar = "KMOB"

st.set_page_config(layout="wide", page_title="LHIM | NEXRAD Ultra")

with st.sidebar:
    st.title("🛡️ LHIM Ultra v4.0")
    radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (kts)"])
    show_alerts = st.toggle("NHC Watch/Warning Layer", value=True)
    
    v_max = st.slider("Intensity (kts)", 40, 165, 120)
    r_max = st.slider("RMW (miles)", 10, 60, 22)
    micro_scale = st.slider("Microphysics", 0.0, 1.0, 0.5)
    l_lat, l_lon = st.number_input("Lat", 30.35), st.number_input("Lon", -88.15)

p = [v_max, r_max, 12, 330, 8, 260, 85, 0.8, 0.85, get_sst_mult("September")]

# --- 3. RENDERING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(location=[l_lat, l_lon], zoom_start=8, tiles="CartoDB DarkMatter")
    
    # Generate high-res overlay
    res = 150
    grid_lats = np.linspace(l_lat-3, l_lat+3, res)
    grid_lons = np.linspace(l_lon-3, l_lon+3, res)
    
    radar_coords = RADAR_SITES[st.session_state.active_radar]
    data = get_radar_array(grid_lats, grid_lons, l_lat, l_lon, p, micro_scale, radar_view, radar_coords)
    
    # Custom colormaps
    if radar_view == "Reflectivity (dBZ)":
        cmap = mcolors.LinearSegmentedColormap.from_list("dbz", ["#00000000", "#0000ff", "#00ff00", "#ffff00", "#ff0000", "#ff00ff"])
        norm = plt.Normalize(15, 75)
    else:
        cmap = "coolwarm"
        norm = plt.Normalize(-80, 80)

    # Convert array to PNG
    img = cmap(norm(data))
    img_buffer = BytesIO()
    plt.imsave(img_buffer, img, format='png')
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_str}',
        bounds=[[l_lat-3, l_lon-3], [l_lat+3, l_lon+3]],
        opacity=0.7,
        interactive=True
    ).add_to(m)

    # Floating Legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 180px; z-index:9999; background: rgba(0,0,0,0.8); 
    padding: 10px; border: 1px solid cyan; color: white; border-radius: 10px; font-family: monospace;">
    <b>{radar_view}</b><br>
    <div style="background: linear-gradient(to right, blue, green, yellow, red, magenta); height: 10px;"></div>
    <div style="display: flex; justify-content: space-between;"><small>20</small><small>45</small><small>75+</small></div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    map_data = st_folium(m, width="100%", height=750)

with c2:
    if show_alerts:
        st.error(f"**NHC ADVISORY**\n{get_nhc_alert()}")

    if map_data and map_data.get("last_clicked"):
        clat, clon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        w, gust, wd, r = calculate_full_physics(clat, clon, l_lat, l_lon, p, micro_scale=micro_scale)
        
        # Realistic Surge Logic
        surge = (w**2 / 1800) * (1.2 if np.sin(np.radians(wd)) > 0 else 0.2) if 30.1 <= clat <= 30.5 else 0
        
        st.markdown(f"""
        <div style="background:#0a0a0a; padding:20px; border-radius:15px; border:2px solid #00ffff;">
            <h2 style="text-align:center;">{get_wind_arrow(wd)}</h2>
            <p style="text-align:center; font-size: 28px; margin:0;"><b>{int(w)} kts</b></p>
            <p style="text-align:center; color:red;">Gusts: {int(gust)} kts</p>
            <hr>
            🌡️ <b>Temp:</b> {82-(r/5):.1f}°F<br>
            💧 <b>Dewpt:</b> {78-(r/6):.1f}°F<br>
            🌊 <b>Surge:</b> <span style="color:cyan;">{surge:.1f} ft</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Click map for probe data.")

    st.markdown("---")
    st.caption("Engine v4: ImageOverlay used to prevent pixelation/dots during zoom. Colormap synced to NEXRAD standards.")
