import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.features import DivIcon
from streamlit_folium import st_folium
import time
from geopy.geocoders import Nominatim

# =========================================================
# LHIM MOBILE COUNTY v3.0
# Extended from the user's original v2.9 code.
# Keeps original core structure while ADDING:
# - 12 Mobile County forecast zones
# - zone/city dropdown system
# - dynamic local parameter calculations
# - dynamically calculated wind, gust, temp, dewpoint,
#   visibility, rain direction, surge, tornado risk
# - 6 hour location forecast
# - county zone summary table
# =========================================================

# -----------------------------
# 1. CORE PHYSICS & RADAR ENGINE
# -----------------------------

def get_sst_mult(month, sst_boost=False):
    months = {
        "June": 0.85,
        "July": 0.92,
        "August": 1.05,
        "September": 1.15,
        "October": 1.02,
        "November": 0.88,
    }
    base = months.get(month, 1.0)
    return base * 1.2 if sst_boost else base


def calculate_full_physics(lat, lon, s_lat, s_lon, p, level=1000, micro_scale=0.0, front_lat=None, terrain_friction=1.0):
    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p
    dx, dy = (lon - s_lon) * 53, (lat - s_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    if r < 1:
        r = 1

    front_mod = 1.0
    local_rh = rh
    if front_lat and lat > front_lat:
        front_dist = abs(lat - front_lat)
        front_mod = max(0.5, 1.0 - (front_dist * 0.2))
        local_rh = min(100, rh + 15)

    level_scale = {1000: 1.0, 925: 0.92, 850: 0.85, 500: 0.45, 200: 0.25}
    l_mult = level_scale.get(level, 1.0)
    eff_v = v_max * (local_rh / 85.0) * (0.6 + outflow / 2.5) * sst_mult * l_mult * front_mod

    B = 1.3 + (eff_v / 150)
    v_sym = eff_v * np.sqrt((r_max / r) ** B * np.exp(1 - (r_max / r) ** B))

    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + ((1.0 - symmetry) * (shear_mag / 45)) * np.cos(angle - shear_rad)

    inflow = 25 if level > 500 else -30
    wind_angle_rad = angle + (np.pi / 2) + np.radians(inflow if r > r_max else inflow / 2)
    v_forward = f_speed * 0.5 * np.cos(wind_angle_rad - np.radians(f_dir))

    surface_wind = max(0, ((v_sym * shear_effect) + v_forward) * terrain_friction)
    return surface_wind, np.degrees(wind_angle_rad), r


def get_synthetic_products(lat, lon, s_lat, s_lon, p, radar_coords=None, micro_scale=0.0, front_lat=None, terrain_friction=1.0, coastal_exposure=1.0):
    v_max, r_max, _, _, shear_mag, shear_dir, rh, _, symmetry, _ = p
    w, wd, r = calculate_full_physics(
        lat, lon, s_lat, s_lon, p,
        micro_scale=micro_scale,
        front_lat=front_lat,
        terrain_friction=terrain_friction
    )
    angle = np.arctan2((lat - s_lat) * 69, (lon - s_lon) * 53)

    is_major = v_max >= 96
    shear_rad = np.radians(shear_dir)
    shear_offset_x = (shear_mag / 20) * np.cos(shear_rad)
    shear_offset_y = (shear_mag / 20) * np.sin(shear_rad)
    r_adj = np.sqrt(((lon - s_lon - shear_offset_x / 53) * 53) ** 2 + ((lat - s_lat - shear_offset_y / 69) * 69) ** 2)

    eyewall = 65 * np.exp(-((r - r_max) ** 2) / (r_max * 0.3) ** 2)
    moisture_flux = (rh / 100) * (1 + (shear_mag / 100))
    shield = 42 * moisture_flux * np.exp(-r_adj / (r_max * 5.0))
    bands = max(0, np.sin(r / (r_max * 0.8) - angle * 3.0) * 35 * np.exp(-r / 200))
    front_rain = 30 * np.exp(-abs(lat - front_lat) * 2) if (front_lat and lat > front_lat - 0.5) else 0

    dbz = max(eyewall, shield, bands, front_rain) * symmetry
    if r < r_max * (0.15 if is_major else 0.4):
        dbz *= 0.1

    if radar_coords:
        rdx, rdy = (lon - radar_coords[1]) * 53, (lat - radar_coords[0]) * 69
        angle_to_radar = np.arctan2(rdy, rdx)
        radial_v = (w * 1.15) * np.cos(np.radians(wd) - angle_to_radar)
        aliased_v = np.clip(radial_v, -149, 149)
    else:
        aliased_v = 0

    surge = 0
    # original rough coastal envelope retained, but exposure weighting added
    if 30.00 <= lat <= 30.55:
        dist_mult = np.exp(-abs(lat - 30.25) * 5)
        surge = (w ** 1.9 / 1700) * (1.8 if lon > s_lon else -0.5) * dist_mult * coastal_exposure

    prob = 90 if w >= 96 else 60 if w >= 64 else 30 if w >= 34 else 0
    return min(78, dbz), aliased_v, surge, prob


# -----------------------------
# 2. MOBILE COUNTY ZONES & CITIES
# -----------------------------

# These are practical forecast sub-zones for Mobile County only.
# Bounds are simplified rectangles for fast Streamlit rendering.
ZONES = {
    "Citronelle": {
        "center": (31.090, -88.230),
        "bounds": ((31.000, -88.340), (31.180, -88.100)),
        "terrain_friction": 0.88,
        "coastal_exposure": 0.08,
        "urban_factor": 0.10,
        "surge_bias": 0.0,
    },
    "Mount Vernon": {
        "center": (31.090, -88.020),
        "bounds": ((31.000, -88.100), (31.170, -87.900)),
        "terrain_friction": 0.90,
        "coastal_exposure": 0.10,
        "urban_factor": 0.08,
        "surge_bias": 0.0,
    },
    "Axis-Satsuma": {
        "center": (30.980, -88.000),
        "bounds": ((30.900, -88.110), (31.040, -87.890)),
        "terrain_friction": 0.91,
        "coastal_exposure": 0.18,
        "urban_factor": 0.18,
        "surge_bias": 0.1,
    },
    "Saraland": {
        "center": (30.820, -88.070),
        "bounds": ((30.760, -88.150), (30.900, -87.980)),
        "terrain_friction": 0.90,
        "coastal_exposure": 0.22,
        "urban_factor": 0.25,
        "surge_bias": 0.15,
    },
    "Prichard-Chickasaw": {
        "center": (30.760, -88.090),
        "bounds": ((30.700, -88.170), (30.810, -88.010)),
        "terrain_friction": 0.89,
        "coastal_exposure": 0.26,
        "urban_factor": 0.35,
        "surge_bias": 0.15,
    },
    "Semmes": {
        "center": (30.780, -88.260),
        "bounds": ((30.710, -88.360), (30.850, -88.150)),
        "terrain_friction": 0.87,
        "coastal_exposure": 0.15,
        "urban_factor": 0.18,
        "surge_bias": 0.0,
    },
    "West Mobile": {
        "center": (30.690, -88.220),
        "bounds": ((30.610, -88.340), (30.760, -88.090)),
        "terrain_friction": 0.88,
        "coastal_exposure": 0.22,
        "urban_factor": 0.28,
        "surge_bias": 0.05,
    },
    "Downtown-Mobile": {
        "center": (30.690, -88.040),
        "bounds": ((30.640, -88.090), (30.740, -87.980)),
        "terrain_friction": 0.90,
        "coastal_exposure": 0.42,
        "urban_factor": 0.55,
        "surge_bias": 0.4,
    },
    "Tillmans Corner": {
        "center": (30.590, -88.170),
        "bounds": ((30.520, -88.260), (30.650, -88.070)),
        "terrain_friction": 0.89,
        "coastal_exposure": 0.32,
        "urban_factor": 0.20,
        "surge_bias": 0.1,
    },
    "Theodore-Dawes": {
        "center": (30.550, -88.180),
        "bounds": ((30.470, -88.290), (30.610, -88.060)),
        "terrain_friction": 0.91,
        "coastal_exposure": 0.45,
        "urban_factor": 0.16,
        "surge_bias": 0.25,
    },
    "Grand Bay-Irvington": {
        "center": (30.450, -88.340),
        "bounds": ((30.320, -88.470), (30.560, -88.200)),
        "terrain_friction": 0.92,
        "coastal_exposure": 0.40,
        "urban_factor": 0.08,
        "surge_bias": 0.15,
    },
    "Bayou La Batre-Coden": {
        "center": (30.390, -88.250),
        "bounds": ((30.230, -88.400), (30.500, -88.080)),
        "terrain_friction": 0.94,
        "coastal_exposure": 0.95,
        "urban_factor": 0.07,
        "surge_bias": 0.7,
    },
}

CITY_POINTS = {
    "Mobile": (30.6944, -88.0431),
    "Downtown Mobile": (30.6911, -88.0399),
    "Prichard": (30.7388, -88.0789),
    "Chickasaw": (30.7669, -88.0731),
    "Saraland": (30.8207, -88.0706),
    "Satsuma": (30.8532, -88.0561),
    "Axis": (30.9291, -87.9944),
    "Semmes": (30.7785, -88.2597),
    "Wilmer": (30.8246, -88.3581),
    "Theodore": (30.5474, -88.1811),
    "Dawes": (30.6030, -88.2140),
    "Tillmans Corner": (30.5919, -88.1711),
    "Irvington": (30.5121, -88.2281),
    "Grand Bay": (30.4769, -88.3428),
    "Bayou La Batre": (30.4032, -88.2485),
    "Coden": (30.3935, -88.1517),
    "Mount Vernon": (31.0877, -88.0130),
    "Citronelle": (31.0907, -88.2281),
}

RADAR_SITES = {
    "KMOB": (30.67, -88.24),
    "KLIX": (30.33, -89.82),
    "KEVX": (30.56, -85.92),
}


def get_zone_for_point(lat, lon):
    for zone_name, meta in ZONES.items():
        (lat1, lon1), (lat2, lon2) = meta["bounds"]
        if min(lat1, lat2) <= lat <= max(lat1, lat2) and min(lon1, lon2) <= lon <= max(lon1, lon2):
            return zone_name

    # nearest-center fallback
    nearest = min(
        ZONES.keys(),
        key=lambda z: np.hypot((lat - ZONES[z]["center"][0]) * 69, (lon - ZONES[z]["center"][1]) * 53)
    )
    return nearest


def get_zone_meta(lat, lon):
    zone_name = get_zone_for_point(lat, lon)
    return zone_name, ZONES[zone_name]


# -----------------------------
# 3. DERIVED ENVIRONMENTAL METRICS
# -----------------------------

def normalize_angle(deg):
    return deg % 360


def deg_to_compass(deg):
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = int((normalize_angle(deg) + 11.25) / 22.5) % 16
    return directions[idx]


def compute_local_environment(lat, lon, s_lat, s_lon, p, radar_coords, front_lat, pressure_drop_hpa=32, dry_air=0, urban_heat=0):
    zone_name, zone_meta = get_zone_meta(lat, lon)

    dbz, vel, surge_raw, prob = get_synthetic_products(
        lat, lon, s_lat, s_lon, p,
        radar_coords=radar_coords,
        front_lat=front_lat,
        terrain_friction=zone_meta["terrain_friction"],
        coastal_exposure=zone_meta["coastal_exposure"],
    )

    w_kts, wd, r = calculate_full_physics(
        lat, lon, s_lat, s_lon, p,
        front_lat=front_lat,
        terrain_friction=zone_meta["terrain_friction"],
    )

    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p

    coastal_boost = 1 + (zone_meta["coastal_exposure"] * 0.08)
    friction_recovery = 1 + ((1 - zone_meta["terrain_friction"]) * 0.15)
    w_kts = w_kts * coastal_boost * friction_recovery

    gust_factor = 1.18 + (0.0018 * w_kts) + (zone_meta["urban_factor"] * 0.08)
    gust = w_kts * gust_factor

    # temperature / dewpoint
    t_base = 84 if sst_mult >= 1.15 else 81
    temp_f = (
        t_base
        - (r * 0.07)
        - (dbz * 0.11)
        - (pressure_drop_hpa * 0.03)
        + (urban_heat * zone_meta["urban_factor"])
        + (zone_meta["coastal_exposure"] * 1.2)
        - (dry_air * 0.15)
    )
    dewp_f = temp_f - max(1.5, (100 - rh + dry_air) * 0.16)
    dewp_f = min(dewp_f, temp_f)

    # visibility in miles
    rain_factor = dbz / 75
    wind_reduction = min(1.2, w_kts / 120)
    vis_mi = 10 - (rain_factor * 7.2) - (wind_reduction * 1.4)
    if zone_meta["coastal_exposure"] > 0.65:
        vis_mi -= 0.4
    vis_mi = float(np.clip(vis_mi, 0.2, 10.0))

    # rain direction approximated by low-level inflow toward center with band curvature
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69
    storm_bearing_from_center = np.degrees(np.arctan2(dx, dy)) % 360
    inflow_dir = (storm_bearing_from_center + 180 - 20) % 360
    rain_dir = normalize_angle(inflow_dir + (shear_mag * 0.15) - ((1 - symmetry) * 25))

    # surge adjusted by zone bias and coastal exposure
    surge_ft = max(0.0, surge_raw + zone_meta["surge_bias"] + (zone_meta["coastal_exposure"] * 0.6))

    # tornado risk: 0-100
    right_front_bonus = max(0, np.cos(np.radians((storm_bearing_from_center - f_dir) - 45)))
    shear_term = np.clip((shear_mag - 10) / 40, 0, 1)
    moisture_term = np.clip((rh - 65) / 30, 0, 1)
    friction_term = np.clip((1 - zone_meta["terrain_friction"]) * 3.2 + zone_meta["urban_factor"] * 0.6, 0, 1)
    band_term = np.clip(dbz / 55, 0, 1)
    wind_term = np.clip(w_kts / 110, 0, 1)
    tornado_risk = 100 * (
        0.24 * shear_term +
        0.18 * moisture_term +
        0.18 * right_front_bonus +
        0.14 * friction_term +
        0.14 * band_term +
        0.12 * wind_term
    )
    tornado_risk = float(np.clip(tornado_risk, 0, 100))

    if tornado_risk >= 75:
        tornado_label = "High"
    elif tornado_risk >= 50:
        tornado_label = "Elevated"
    elif tornado_risk >= 25:
        tornado_label = "Moderate"
    else:
        tornado_label = "Low"

    return {
        "zone": zone_name,
        "wind_kts": float(w_kts),
        "gust_kts": float(gust),
        "temp_f": float(temp_f),
        "dewp_f": float(dewp_f),
        "visibility_mi": vis_mi,
        "rain_dir_deg": float(rain_dir),
        "rain_dir_text": deg_to_compass(rain_dir),
        "surge_ft": float(surge_ft),
        "tornado_risk": tornado_risk,
        "tornado_label": tornado_label,
        "dbz": float(dbz),
        "vel": float(vel),
        "wind_prob": int(prob),
        "wind_dir_deg": float(normalize_angle(wd)),
        "wind_dir_text": deg_to_compass(wd),
        "radius_mi": float(r),
    }


def condition_from_wind(w_kts, radius_mi, r_max):
    if w_kts > 115:
        w_desc = "DEVASTATING"
    elif w_kts > 95:
        w_desc = "EXTREME"
    elif w_kts > 64:
        w_desc = "HURRICANE"
    elif w_kts > 34:
        w_desc = "TROPICAL STORM"
    elif w_kts > 20:
        w_desc = "BREEZY"
    else:
        w_desc = "LIGHT WINDS"

    return f"EYEWALL: {w_desc}" if radius_mi < r_max * 1.2 else f"{w_desc} / RAIN"


def zone_summary_table(s_lat, s_lon, p, radar_coords, front_lat, pressure_drop_hpa, dry_air, urban_heat):
    rows = []
    for zone_name, meta in ZONES.items():
        lat, lon = meta["center"]
        env = compute_local_environment(
            lat, lon, s_lat, s_lon, p, radar_coords, front_lat,
            pressure_drop_hpa=pressure_drop_hpa,
            dry_air=dry_air,
            urban_heat=urban_heat,
        )
        rows.append({
            "Zone": zone_name,
            "Wind": f"{env['wind_kts']:.0f} kt",
            "Gust": f"{env['gust_kts']:.0f} kt",
            "Temp": f"{env['temp_f']:.0f}°F",
            "Dewpoint": f"{env['dewp_f']:.0f}°F",
            "Visibility": f"{env['visibility_mi']:.1f} mi",
            "Rain Dir": env["rain_dir_text"],
            "Surge": f"{env['surge_ft']:.1f} ft",
            "Tornado": f"{env['tornado_label']} ({env['tornado_risk']:.0f})",
        })
    return pd.DataFrame(rows)


# -----------------------------
# 4. SESSION STATE
# -----------------------------
if "active_radar" not in st.session_state:
    st.session_state.active_radar = "KMOB"
if "loop_idx" not in st.session_state:
    st.session_state.loop_idx = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

geolocator = Nominatim(user_agent="lhim_mobile_county_v30")


# -----------------------------
# 5. UI & SIDEBAR
# -----------------------------
st.set_page_config(layout="wide", page_title="LHIM Mobile County v3.0 | Parameter Impact Mode")

with st.sidebar:
    st.title("🛡️ LHIM Mobile County v3.0")
    radar_view = st.radio(
        "Display Mode",
        ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge", "Wind Prob."]
    )

    with st.expander("⚠️ Warning Settings", expanded=False):
        show_warnings = st.checkbox("Overlay Surge Warnings", value=True)
        surge_threshold = st.slider("Warning Trigger (ft)", 3, 12, 6)
        show_zone_boxes = st.checkbox("Show Zone Boxes", value=True)
        show_city_markers = st.checkbox("Show City Markers", value=True)

    st.subheader("📡 Radar Controls")
    st.session_state.active_radar = st.selectbox("Radar Site", list(RADAR_SITES.keys()), index=list(RADAR_SITES.keys()).index(st.session_state.active_radar))
    run_loop = st.checkbox("🔄 Enable Radar Loop", value=st.session_state.is_playing)
    st.session_state.is_playing = run_loop
    current_time_offset = st.slider("Time Offset (Hours)", -12, 12, st.session_state.loop_idx)
    st.session_state.loop_idx = current_time_offset

    with st.expander("🌡️ Environmental Layers", expanded=True):
        season_month = st.selectbox("Seasonal SST Month", ["June", "July", "August", "September", "October", "November"], index=3)
        sst_boost = st.toggle("Warm Sea Surface (SST+)", value=True)
        front_lat = st.slider("Cold Front Latitude", 30.0, 32.5, 31.8)
        shear_mag = st.slider("Deep Layer Shear (kts)", 0, 80, 15)
        shear_dir = st.slider("Shear Vector Dir", 0, 360, 240)
        rh = st.slider("Fluid RH (%)", 30, 100, 88)
        outflow = st.slider("Upper Outflow", 0.2, 1.5, 0.8, 0.05)
        symmetry = st.slider("Core Symmetry", 0.35, 1.00, 0.85, 0.01)
        pressure_drop_hpa = st.slider("Pressure Fall Signal (hPa)", 0, 60, 32)
        dry_air = st.slider("Dry Air Entrapment", 0, 40, 8)
        urban_heat = st.slider("Urban Heat Bias", 0.0, 4.0, 1.2, 0.1)

    st.subheader("🌀 Storm Structure")
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    f_speed = st.slider("Forward Speed", 2, 40, 12)
    f_dir = st.slider("Heading", 0, 360, 330)
    l_lat = st.number_input("Landfall Lat", value=30.35, format="%.4f")
    l_lon = st.number_input("Landfall Lon", value=-88.15, format="%.4f")
    res_steps = st.select_slider("Quality", options=[30, 45, 60], value=45)

    st.subheader("📍 Mobile County Selector")
    selected_zone = st.selectbox("Zone", list(ZONES.keys()), index=list(ZONES.keys()).index("Downtown-Mobile"))
    selected_city = st.selectbox("City / Place", list(CITY_POINTS.keys()), index=list(CITY_POINTS.keys()).index("Mobile"))
    use_city_selection = st.checkbox("Lock analysis panel to selected city/place", value=False)


# current storm position from original logic
# retained, but parameter list now uses sidebar values directly

dist_moved = (f_speed * current_time_offset) / 69.0
current_lat = l_lat + (dist_moved * np.cos(np.radians(f_dir)))
current_lon = l_lon + (dist_moved * np.sin(np.radians(f_dir)))
p = [
    v_max,
    r_max,
    f_speed,
    f_dir,
    shear_mag,
    shear_dir,
    rh,
    outflow,
    symmetry,
    get_sst_mult(season_month, sst_boost),
]

radar_coords = RADAR_SITES[st.session_state.active_radar]


# -----------------------------
# 6. MAP & DASHBOARD
# -----------------------------
c1, c2 = st.columns([4, 1.8])

with c1:
    m = folium.Map(location=[30.75, -88.12], zoom_start=9, tiles="CartoDB DarkMatter")

    lats = np.linspace(l_lat - 2.5, l_lat + 2.5, res_steps)
    lons = np.linspace(l_lon - 2.5, l_lon + 2.5, int(res_steps * 1.2))
    d_lat, d_lon = lats[1] - lats[0], lons[1] - lons[0]

    for lt in lats:
        for ln in lons:
            zone_name, zone_meta = get_zone_meta(lt, ln)
            dbz, vel, surge, prob = get_synthetic_products(
                lt, ln, current_lat, current_lon, p,
                radar_coords=radar_coords,
                front_lat=front_lat,
                terrain_friction=zone_meta["terrain_friction"],
                coastal_exposure=zone_meta["coastal_exposure"],
            )
            color = None

            if radar_view == "Reflectivity (dBZ)" and dbz > 15:
                color = (
                    "#ff00ff" if dbz > 65 else
                    "#ff0000" if dbz > 50 else
                    "#ff9900" if dbz > 40 else
                    "#ffff00" if dbz > 28 else
                    "#00ff00"
                )
            elif radar_view == "Velocity (kts)":
                if vel < -5:
                    color = "#00ffff" if vel < -110 else "#00ccff" if vel < -75 else "#00aa00"
                elif vel > 5:
                    color = "#ff00ff" if vel > 110 else "#ff0000" if vel > 75 else "#880000"
            elif radar_view == "Storm Surge" and abs(surge) > 0.5:
                color = (
                    "#4b0082" if surge > 12 else
                    "#8b0000" if surge > 9 else
                    "#ff0000" if surge > 6 else
                    "#ff8c00" if surge > 3 else
                    "#ffd700" if surge > 0 else
                    "#00ced1"
                )
            elif radar_view == "Wind Prob." and prob > 0:
                color = (
                    "#ff00ff" if prob >= 90 else
                    "#ff8c00" if prob >= 60 else
                    "#ffff00"
                )

            if color:
                folium.Rectangle(
                    bounds=[[lt, ln], [lt + d_lat, ln + d_lon]],
                    color=color,
                    fill=True,
                    fill_opacity=0.55,
                    weight=0,
                ).add_to(m)

    # zone overlays
    if show_zone_boxes:
        for zone_name, meta in ZONES.items():
            (lat1, lon1), (lat2, lon2) = meta["bounds"]
            folium.Rectangle(
                bounds=[[lat1, lon1], [lat2, lon2]],
                color="#6fa8dc",
                fill=False,
                weight=1.2,
                opacity=0.8,
                tooltip=zone_name,
            ).add_to(m)

            c_lat, c_lon = meta["center"]
            folium.map.Marker(
                [c_lat, c_lon],
                icon=DivIcon(
                    icon_size=(160, 16),
                    icon_anchor=(0, 0),
                    html=f"<div style='font-size:10px;color:white;text-shadow:0 0 3px black;'>{zone_name}</div>",
                ),
            ).add_to(m)

    # city markers
    if show_city_markers:
        for city_name, (ct_lat, ct_lon) in CITY_POINTS.items():
            folium.CircleMarker(
                location=[ct_lat, ct_lon],
                radius=3,
                color="#ffffff",
                fill=True,
                fill_color="#00d4ff",
                fill_opacity=0.85,
                tooltip=city_name,
            ).add_to(m)

    # storm center and selected zone/city markers
    folium.Marker(
        [current_lat, current_lon],
        tooltip="Storm Center",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    zone_lat, zone_lon = ZONES[selected_zone]["center"]
    folium.CircleMarker(
        [zone_lat, zone_lon], radius=6, color="#00ffcc", fill=True,
        fill_color="#00ffcc", tooltip=f"Selected Zone: {selected_zone}"
    ).add_to(m)

    city_lat, city_lon = CITY_POINTS[selected_city]
    folium.CircleMarker(
        [city_lat, city_lon], radius=6, color="#ffd700", fill=True,
        fill_color="#ffd700", tooltip=f"Selected City: {selected_city}"
    ).add_to(m)

    if show_warnings:
        for zone_name, meta in ZONES.items():
            zlat, zlon = meta["center"]
            env = compute_local_environment(
                zlat, zlon, current_lat, current_lon, p, radar_coords, front_lat,
                pressure_drop_hpa=pressure_drop_hpa, dry_air=dry_air, urban_heat=urban_heat,
            )
            if env["surge_ft"] >= surge_threshold:
                folium.Circle(
                    location=[zlat, zlon],
                    radius=9000,
                    color="#ff4444",
                    fill=False,
                    weight=2,
                    opacity=0.85,
                    tooltip=f"{zone_name} surge warning: {env['surge_ft']:.1f} ft",
                ).add_to(m)

    map_data = st_folium(
        m,
        width="100%",
        height=770,
        key=f"map_frame_{st.session_state.loop_idx}",
        returned_objects=["last_clicked"],
    )

with c2:
    st.markdown(
        """
        <style>
        .weather-card {
            background-color: #003366;
            color: white;
            padding: 18px;
            border-radius: 12px;
            border-left: 8px solid #ffcc00;
            margin-bottom: 10px;
        }
        .mini-note {
            font-size: 0.82rem;
            color: #b7d8ff;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    click_lat = None
    click_lon = None
    loc_name = None

    if use_city_selection:
        click_lat, click_lon = CITY_POINTS[selected_city]
        loc_name = selected_city
    elif map_data and map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]
        try:
            location = geolocator.reverse(f"{click_lat}, {click_lon}", timeout=3)
            address = location.raw.get("address", {})
            loc_name = address.get("city") or address.get("town") or address.get("village") or address.get("hamlet") or f"Grid {click_lat:.2f}, {click_lon:.2f}"
        except Exception:
            loc_name = f"Grid {click_lat:.2f}, {click_lon:.2f}"
    else:
        click_lat, click_lon = CITY_POINTS[selected_city]
        loc_name = selected_city

    env = compute_local_environment(
        click_lat, click_lon, current_lat, current_lon, p, radar_coords, front_lat,
        pressure_drop_hpa=pressure_drop_hpa,
        dry_air=dry_air,
        urban_heat=urban_heat,
    )

    st.markdown(
        f"""
        <div class='weather-card'>
            <h2>📍 {loc_name}</h2>
            <h3>ZONE: {env['zone']}</h3>
            <div class='mini-note'>AT T{current_time_offset:+} HOURS · Wind dir {env['wind_dir_text']} · Rain inflow {env['rain_dir_text']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2 = st.columns(2)
    k1.metric("TEMP", f"{env['temp_f']:.0f}°F")
    k1.metric("DEW PT", f"{env['dewp_f']:.0f}°F")
    k1.metric("VISIBILITY", f"{env['visibility_mi']:.1f} mi")
    k1.metric("SURGE", f"{env['surge_ft']:.1f} ft")

    k2.metric("WIND", f"{env['wind_kts']:.0f} kt")
    k2.metric("GUST", f"{env['gust_kts']:.0f} kt")
    k2.metric("RAIN DIR", env['rain_dir_text'])
    k2.metric("TORNADO", f"{env['tornado_label']} ({env['tornado_risk']:.0f})")

    st.divider()
    st.subheader("⏱️ 6-Hour Impact Forecast")

    forecast_rows = []
    for hour in range(1, 7):
        f_dist = (f_speed * (current_time_offset + hour)) / 69.0
        f_lat = l_lat + (f_dist * np.cos(np.radians(f_dir)))
        f_lon = l_lon + (f_dist * np.sin(np.radians(f_dir)))

        f_env = compute_local_environment(
            click_lat, click_lon, f_lat, f_lon, p, radar_coords, front_lat,
            pressure_drop_hpa=pressure_drop_hpa,
            dry_air=dry_air,
            urban_heat=urban_heat,
        )

        forecast_rows.append({
            "Time": f"T+{hour}h",
            "Condition": condition_from_wind(f_env["wind_kts"], f_env["radius_mi"], r_max),
            "Wind": f"{f_env['wind_kts']:.0f} kt",
            "Gust": f"{f_env['gust_kts']:.0f} kt",
            "Temp": f"{f_env['temp_f']:.0f}°F",
            "Visibility": f"{f_env['visibility_mi']:.1f} mi",
            "Surge": f"{f_env['surge_ft']:.1f} ft",
            "Tornado": f"{f_env['tornado_label']} ({f_env['tornado_risk']:.0f})",
        })

    st.dataframe(pd.DataFrame(forecast_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("🎯 Selected Zone Snapshot")
    zlat, zlon = ZONES[selected_zone]["center"]
    zenv = compute_local_environment(
        zlat, zlon, current_lat, current_lon, p, radar_coords, front_lat,
        pressure_drop_hpa=pressure_drop_hpa,
        dry_air=dry_air,
        urban_heat=urban_heat,
    )
    st.dataframe(pd.DataFrame([{
        "Zone": selected_zone,
        "Wind": f"{zenv['wind_kts']:.0f} kt",
        "Gust": f"{zenv['gust_kts']:.0f} kt",
        "Temp": f"{zenv['temp_f']:.0f}°F",
        "Dewpoint": f"{zenv['dewp_f']:.0f}°F",
        "Visibility": f"{zenv['visibility_mi']:.1f} mi",
        "Rain Dir": zenv['rain_dir_text'],
        "Surge": f"{zenv['surge_ft']:.1f} ft",
        "Tornado": f"{zenv['tornado_label']} ({zenv['tornado_risk']:.0f})",
    }]), hide_index=True, use_container_width=True)


# -----------------------------
# 7. COUNTY-WIDE ZONE TABLE
# -----------------------------
st.divider()
st.subheader("📊 Mobile County 12-Zone Dynamic Summary")
summary_df = zone_summary_table(
    current_lat,
    current_lon,
    p,
    radar_coords,
    front_lat,
    pressure_drop_hpa,
    dry_air,
    urban_heat,
)
st.dataframe(summary_df, hide_index=True, use_container_width=True)


# -----------------------------
# 8. AUTOMATED LOOP ENGINE
# -----------------------------
if st.session_state.is_playing:
    st.session_state.loop_idx += 1
    if st.session_state.loop_idx > 12:
        st.session_state.loop_idx = -12
    time.sleep(0.1)
    st.rerun()
