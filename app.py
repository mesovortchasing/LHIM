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
        dist_move = (f_speed * h) / 69.0
        move_rad = np.radians(f_dir)
        new_s_lat = s_lat + (dist_move * np.cos(move_rad))
        new_s_lon = s_lon + (dist_move * np.sin(move_rad))

        friction = 0.85 if new_s_lat > 30.4 else 1.0
        p_current = [v_max * friction, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult]

        w_kts, wd, _ = calculate_full_physics(lat, lon, new_s_lat, new_s_lon, p_current)
        w_mph = w_kts * 1.15

        color = "#ffffff"
        if w_mph >= 106: color = "#ff4b4b"
        elif w_mph >= 76: color = "#ffa500"
        elif w_mph >= 45: color = "#ffff00"

        icon = "🌀" if w_mph > 74 else "🌧️" if w_mph > 39 else "☁️"

        forecast_data.append({
            "Hour": f"+{h}h",
            "Icon": icon,
            "Wind (mph)": int(w_mph),
            "Direction": get_wind_arrow(wd),
            "Color": color
        })
    return forecast_data

def get_synthetic_products(lat, lon, s_lat, s_lon, p, nyquist=60, erc_params=None):
    v_max, r_max, _, _, _, _, rh, _, symmetry, _ = p
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)

    # --- SURGICAL ERC INTEGRATION ---
    eyewall = 60 * np.exp(-((r - r_max)**2) / (r_max * 0.22)**2)

    # ERC Logic: Add secondary wind/reflectivity ring
    if erc_params and erc_params['active']:
        outer_r_pos = r_max * 2.8
        secondary_ring = (erc_params['strength'] * 45) * np.exp(-((r - outer_r_pos)**2) / (r_max * 0.6)**2)
        eyewall = max(eyewall, secondary_ring)
        moat_factor = 0.3 if (r_max * 1.3 < r < outer_r_pos * 0.8) else 1.0
    else:
        outer_r = r_max * 2.4
        outer_eyewall = 42 * np.exp(-((r - outer_r)**2) / (r_max * 0.4)**2) if v_max > 105 else 0
        eyewall = max(eyewall, outer_eyewall)
        moat_factor = 0.45 if (r_max * 1.4 < r < outer_r * 0.85 and v_max > 105) else 1.0

    bands = max(0, np.sin(r / (r_max * 0.6) - angle * 2.8) * 44 * np.exp(-r / 135))
    dbz = (eyewall + bands + 15) * moat_factor * (rh / 100) * symmetry
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

# --- NEW: RADAR SITES & CACHING ---
RADAR_SITES = {
    "KMOB (Mobile)": (30.67, -88.24),
    "KLIX (New Orleans)": (30.33, -89.82),
    "KEVX (Eglin AFB)": (30.56, -85.92)
}

@st.cache_data
def get_cached_radar_grid(l_lat, l_lon, p, nyquist, res_steps, radar_site_coords, use_radar_rel, erc_params):
    lats = np.linspace(l_lat-2.5, l_lat+2.5, res_steps)
    lons = np.linspace(l_lon-3.0, l_lon+3.0, int(res_steps * 1.2))
    grid_data = []
    for lt in lats:
        for ln in lons:
            dbz, vel, surge, prob = get_synthetic_products(lt, ln, l_lat, l_lon, p, nyquist, erc_params)
            if use_radar_rel:
                w, wd, _ = calculate_full_physics(lt, ln, l_lat, l_lon, p)
                angle_to_radar = np.arctan2((ln - radar_site_coords[1])*53, (lt - radar_site_coords[0])*69)
                vel = w * np.cos(np.radians(wd) - angle_to_radar)
                vel = ((vel + nyquist) % (2 * nyquist)) - nyquist
            grid_data.append({"lat": lt, "lon": ln, "dbz": dbz, "vel": vel, "surge": surge, "prob": prob})
    return grid_data, lats[1]-lats[0], lons[1]-lons[0]

# --- NEW: COLOR TABLES (VELOCITY BINS + RADARSCOPE-LIKE PALETTE) ---

def quantize_abs(val, step=10, vmax=140):
    """Quantize absolute magnitude into 0..vmax by 'step'."""
    v = min(vmax, max(0, abs(val)))
    return int(v // step) * step

def velocity_color_radarscope_like(v, step=10, vmax=140):
    """
    Diverging velocity palette (Radarscope-inspired):
    - Negative/toward: greens
    - Positive/away: reds
    - Near zero: neutral
    Quantized by bins: 0,10,20,...,140
    """
    b = quantize_abs(v, step=step, vmax=vmax)
    if b == 0:
        return "#1a1a1a"

    neg_bins = {
        10:"#0b2e13", 20:"#145a23", 30:"#1f7a2e", 40:"#2aa144",
        50:"#3bc15b", 60:"#61d67b", 70:"#8be8a3", 80:"#b8f5c7",
        90:"#d9ffe3", 100:"#ecfff2", 110:"#f4fff8", 120:"#fbfffd",
        130:"#ffffff", 140:"#ffffff"
    }
    pos_bins = {
        10:"#3b0a0a", 20:"#5c1111", 30:"#7a1c1c", 40:"#a12a2a",
        50:"#c13b3b", 60:"#d66161", 70:"#e88b8b", 80:"#f5b8b8",
        90:"#ffd9d9", 100:"#ffecec", 110:"#fff4f4", 120:"#fffafa",
        130:"#ffffff", 140:"#ffffff"
    }
    return pos_bins.get(b, "#ffffff") if v > 0 else neg_bins.get(b, "#ffffff")

def build_velocity_legend_bins(step=10, vmax=140):
    """Rows: -vmax..-step, 0, +step..+vmax"""
    bins = list(range(step, vmax + step, step))
    neg = [(-b, velocity_color_radarscope_like(-b, step, vmax)) for b in reversed(bins)]
    zero = [(0, velocity_color_radarscope_like(0, step, vmax))]
    pos = [(b, velocity_color_radarscope_like(b, step, vmax)) for b in bins]
    return neg + zero + pos

def add_velocity_legend(m, step=10, vmax=140):
    rows = build_velocity_legend_bins(step=step, vmax=vmax)
    html = f'''
    <div style="position: fixed; top: 10px; right: 10px; width: 170px; z-index:9999;
         background: rgba(0,0,0,0.82); color: white; padding: 10px; border-radius: 6px;
         font-family: sans-serif; font-size: 12px; border: 1px solid #444;">
      <b>Radial Velocity (kts)</b><br>
      <span style="color:#aaa;">Synthetic / Estimated</span><br><br>
    '''
    for v, c in rows:
        label = "0" if v == 0 else f"{v:+d}"
        html += f'''
        <div style="display:flex; align-items:center; margin:2px 0;">
          <div style="background:{c}; width:14px; height:12px; margin-right:6px; border:1px solid rgba(255,255,255,0.15);"></div>
          <div style="width:46px;">{label}</div>
          <div style="color:#aaa;">kts</div>
        </div>
        '''
    html += "</div>"
    m.get_root().html.add_child(folium.Element(html))

def add_reflectivity_legend(m):
    rows = [
        ("20–30", "#00ff00"),
        ("30–40", "#ffff00"),
        ("40–50", "#ff9900"),
        ("50+",   "#ff0000"),
    ]
    html = '''
    <div style="position: fixed; top: 10px; right: 10px; width: 140px; z-index:9999;
         background: rgba(0,0,0,0.82); color: white; padding: 10px; border-radius: 6px;
         font-family: sans-serif; font-size: 12px; border: 1px solid #444;">
      <b>Reflectivity (dBZ)</b><br>
      <span style="color:#aaa;">Synthetic / Estimated</span><br><br>
    '''
    for lbl, c in rows:
        html += f'''
        <div style="display:flex; align-items:center; margin:2px 0;">
          <div style="background:{c}; width:14px; height:12px; margin-right:6px; border:1px solid rgba(255,255,255,0.15);"></div>
          <div>{lbl}</div>
        </div>
        '''
    html += "</div>"
    m.get_root().html.add_child(folium.Element(html))

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="LHIM | Alpha")

# (Legacy legend kept, but we’ll override for reflectivity/velocity for better bins)
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
    with st.expander("📡 Sensors & Performance", expanded=True):
        month = st.selectbox("Month", ["June", "July", "August", "September", "October", "November"], index=3)
        p_sst = get_sst_mult(month)
        radar_view = st.radio("Display Mode", ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge", "Wind Prob."])
        radar_site = st.selectbox("Radar Site", list(RADAR_SITES.keys()))
        use_radar_rel = st.toggle("Radar-Relative Velocity", value=False)
        res_mode = st.selectbox("Resolution", ["Low (35x)", "Standard (55x)", "High (80x)"], index=1)
        res_steps = {"Low (35x)": 35, "Standard (55x)": 55, "High (80x)": 80}[res_mode]
        radar_alpha = st.slider("Opacity", 0.1, 1.0, 0.65)
        nyquist = st.slider("Nyquist", 30, 100, 65)

        st.markdown("---")
        st.caption("Velocity Color Bins")
        vel_step = st.select_slider("Bin Size (kts)", options=[5, 10, 15, 20], value=10)
        vel_vmax = st.select_slider("Max (kts)", options=[80, 100, 120, 140, 160], value=140)

    st.header("1. Core Parameters")
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    f_speed = st.slider("Forward Speed (mph)", 2, 40, 12)
    f_dir = st.slider("Heading (Deg)", 0, 360, 330)
    r_max = st.slider("RMW (miles)", 10, 60, 25)

    with st.expander("🌀 Storm Structure & ERC"):
        eyewall_org = st.slider("Organization", 0.0, 1.0, 1.0)
        erc_active = st.toggle("Active ERC")
        erc_strength = st.slider("Outer Ring Strength", 0.0, 1.0, 0.7) if erc_active else 0
        erc_params = {'active': erc_active, 'strength': erc_strength}

    st.header("2. Environment")
    rh, outflow, symmetry = st.slider("Humidity", 30, 100, 85), st.slider("Outflow", 0.0, 1.0, 0.8), st.slider("Symmetry", 0.0, 1.0, 0.85)
    shear_mag, shear_dir = st.slider("Shear (kts)", 0, 60, 8), st.slider("Shear From", 0, 360, 260)
    l_lat, l_lon = st.number_input("Landfall Lat", value=30.35), st.number_input("Landfall Lon", value=-88.15)
    map_theme = st.selectbox("Theme", ["Dark Mode", "Light Mode"])

p = [v_max * eyewall_org, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, p_sst]

# --- 3. MAPPING ---
c1, c2 = st.columns([4, 1.5])

with c1:
    m = folium.Map(
        location=[l_lat, l_lon],
        zoom_start=9,
        tiles="CartoDB DarkMatter" if map_theme == "Dark Mode" else "OpenStreetMap"
    )

    # Use improved legends for reflectivity/velocity; keep old for the other modes
    if radar_view == "Reflectivity (dBZ)":
        add_reflectivity_legend(m)
    elif radar_view == "Velocity (kts)":
        add_velocity_legend(m, step=vel_step, vmax=vel_vmax)
    else:
        add_legend(m, radar_view)

    # Layer Groups
    fg_radar = folium.FeatureGroup(name="Reflectivity").add_to(m)
    fg_vel = folium.FeatureGroup(name="Velocity").add_to(m)
    fg_surge = folium.FeatureGroup(name="Surge/Prob").add_to(m)

    data_grid, d_lat, d_lon = get_cached_radar_grid(
        l_lat, l_lon, p, nyquist, res_steps,
        RADAR_SITES[radar_site], use_radar_rel, erc_params
    )

    for cell in data_grid:
        lt, ln = cell['lat'], cell['lon']

        if radar_view == "Reflectivity (dBZ)" and cell['dbz'] > 15:
            color = '#ff0000' if cell['dbz'] > 50 else '#ff9900' if cell['dbz'] > 40 else '#ffff00' if cell['dbz'] > 30 else '#00ff00' if cell['dbz'] > 20 else '#0000ff'
            folium.Rectangle(
                bounds=[[lt, ln], [lt + d_lat, ln + d_lon]],
                color=color, fill=True, fill_opacity=radar_alpha, weight=0
            ).add_to(fg_radar)

        elif radar_view == "Velocity (kts)":
            # NEW: Discrete Radarscope-like bins, fixed increments up to vel_vmax
            color = velocity_color_radarscope_like(cell['vel'], step=vel_step, vmax=vel_vmax)
            folium.Rectangle(
                bounds=[[lt, ln], [lt + d_lat, ln + d_lon]],
                color=color, fill=True, fill_opacity=radar_alpha, weight=0
            ).add_to(fg_vel)

        elif radar_view == "Storm Surge" and cell['surge'] > 1.5:
            color = '#330066' if cell['surge'] > 12 else '#0033ff' if cell['surge'] > 6 else '#00ffff'
            folium.Rectangle(
                bounds=[[lt, ln], [lt + d_lat, ln + d_lon]],
                color=color, fill=True, fill_opacity=radar_alpha, weight=0
            ).add_to(fg_surge)

        elif radar_view == "Wind Prob." and cell.get('prob', 0) > 20:
            # Optional: show prob field (you had it as view option but not plotted in this block)
            # Keeping it minimal; you can add a FeatureGroup if you want later.
            pass

    folium.LayerControl().add_to(m)
    last_click = st_folium(m, width="100%", height=750, key="lhim_alpha_map")

with c2:
    tab_inspect, tab_ct = st.tabs(["📍 Inspector", "🎨 Color Tables"])

    with tab_inspect:
        st.subheader("📍 Point Inspector")
        if last_click and last_click.get("last_clicked"):
            clat, clon = last_click["last_clicked"]["lat"], last_click["last_clicked"]["lng"]
            idbz, ivel, isurge, iprob = get_synthetic_products(clat, clon, l_lat, l_lon, p, nyquist, erc_params)
            iw, iwd, _ = calculate_full_physics(clat, clon, l_lat, l_lon, p)
            dist = np.sqrt(((clon - l_lon) * 53)**2 + ((clat - l_lat) * 69)**2)

            st.markdown(f"""
            <div style="background:#1e1e1e; padding:15px; border-radius:10px; border-left: 5px solid #ff4b4b;">
                <b>Point:</b> {clat:.2f}, {clon:.2f}<br>
                <b>Reflectivity:</b> {idbz:.1f} dBZ<br>
                <b>Velocity:</b> {ivel:.1f} kts | <b>Surge:</b> {isurge:.1f} ft<br>
                <b>Wind:</b> {iw:.1f} kts ({iw*1.15:.1f} mph) {get_wind_arrow(iwd)}<br>
                <b>Dist from Eye:</b> {dist:.1f} miles
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🕒 6-Hour Forecast", expanded=True):
                h_data = get_hourly_forecast(clat, clon, l_lat, l_lon, p)
                for row in h_data:
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between;'>"
                        f"<span>{row['Hour']} {row['Icon']}</span> "
                        f"<b style='color:{row['Color']}'>{row['Wind (mph)']} mph</b> "
                        f"<span>{row['Direction']}</span></div>",
                        unsafe_allow_html=True
                    )

            st.table(pd.DataFrame(get_vertical_profile(clat, clon, l_lat, l_lon, p)).set_index('Level'))
        else:
            st.info("Click map to inspect data.")

        st.metric("SST Influence", f"{month}", f"{p_sst:.2f}x")
        st.progress(min(max((v_max / 160) * symmetry, 0.0), 1.0))

    with tab_ct:
        st.subheader("🎨 Color Tables")
        st.caption("These are the discrete color bins used for Synthetic Radar layers.")

        st.markdown("### Reflectivity (dBZ)")
        for lbl, c in [("20–30", "#00ff00"), ("30–40", "#ffff00"), ("40–50", "#ff9900"), ("50+", "#ff0000")]:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; margin:4px 0;'>"
                f"<div style='width:20px; height:12px; background:{c}; border:1px solid #333;'></div>"
                f"<div>{lbl}</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("### Velocity (kts) — Radarscope-like (Synthetic)")
        st.caption("Greens = toward radar (negative), Reds = away from radar (positive).")

        rows = build_velocity_legend_bins(step=vel_step, vmax=vel_vmax)
        for v, c in rows:
            label = "0" if v == 0 else f"{v:+d}"
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; margin:3px 0;'>"
                f"<div style='width:20px; height:12px; background:{c}; border:1px solid #333;'></div>"
                f"<div style='width:50px;'>{label}</div>"
                f"<div style='color:#888;'>kts</div></div>",
                unsafe_allow_html=True
            )
