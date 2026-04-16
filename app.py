import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.features import DivIcon
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import time
from geopy.geocoders import Nominatim
st.set_page_config(layout="wide")
# =========================================================
# LHIM MOBILE COUNTY v4.0 HYPERREALISTIC
# Built as a drop-in extension of the user's v3.0 sandbox.
#
# Added without changing the core idea:
# - Realistic on-map legends for every parameter mode
# - NWS-style reflectivity palette
# - Dual wind units (kt + mph)
# - Pressure estimates / pressure tendency
# - Multi-point forecast track (0/6/12/24/36/48h)
# - Cone of uncertainty rendering
# - Satellite / street / dark base map toggle
# - Optional traffic tile toggle hook (requires tile URL/API)
# - Radar beam degradation / beam height / cone of silence effects
# - Optional eyewall replacement cycle structure
# - Inland decay / land interaction refinement
# - Cleaner professional map symbology / advisory panel
#
# NOTE:
# True live traffic overlays generally require an external provider/API.
# This app supports it if you provide a tile URL in the sidebar.
# =========================================================

# -----------------------------
# INSPECTOR DEFAULTS (CRITICAL FIX)
# -----------------------------
season_month = "September"
sst_boost = True
front_lat = 31.8
shear_mag = 15
shear_dir = 240
rh = 88
outflow = 0.8
symmetry = 0.85
pressure_drop_hpa = 32
dry_air = 8
urban_heat = 1.2
ewr_phase = 0.0

extreme_wind_threshold_mph = 115
show_warnings = True
show_extreme_wind_warning = True
show_warning_text_panel = True
surge_threshold = 6
show_zone_boxes = True
show_city_markers = True
show_forecast_track = True
show_cone = True

#Inspired Inspector Tool 
if "inspector_mode" not in st.session_state:
    st.session_state.inspector_mode = False

def toggle_inspector():
    st.session_state.inspector_mode = not st.session_state.inspector_mode

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


def normalize_angle(deg):
    return deg % 360


def deg_to_compass(deg):
    directions = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = int((normalize_angle(deg) + 11.25) / 22.5) % 16
    return directions[idx]


def kt_to_mph(kts):
    return kts * 1.15078


def calculate_mslp(v_max, pressure_drop_hpa):
    # Synthetic but realistic-facing pressure estimate.
    # Keeps original sandbox spirit while presenting an interpretable value.
    return max(860.0, 1012.0 - pressure_drop_hpa - (v_max * 0.55))


def calculate_pressure_tendency_mbhr(pressure_drop_hpa):
    return pressure_drop_hpa / 6.0


def saffir_simpson_category(v_max_kts):
    mph = kt_to_mph(v_max_kts)
    if mph < 39:
        return "Tropical Depression"
    if mph < 74:
        return "Tropical Storm"
    if mph < 96:
        return "Category 1"
    if mph < 111:
        return "Category 2"
    if mph < 130:
        return "Category 3"
    if mph < 157:
        return "Category 4"
    return "Category 5"

# -----------------------------
# NEW: PRESSURE GRADIENT + VECTOR WIND ENGINE
# -----------------------------

def compute_pressure_gradient(p_center, p_neighbors, distances):
    gradients = []
    for p, d in zip(p_neighbors, distances):
        if d == 0:
            continue
        gradients.append((p_center - p) / d)
    return sum(gradients) / len(gradients) if gradients else 0


def compute_directional_gradient(p_grid, i, j):
    rows, cols = p_grid.shape

    left  = p_grid[i][j-1] if j-1 >= 0 else p_grid[i][j]
    right = p_grid[i][j+1] if j+1 < cols else p_grid[i][j]
    up    = p_grid[i-1][j] if i-1 >= 0 else p_grid[i][j]
    down  = p_grid[i+1][j] if i+1 < rows else p_grid[i][j]

    gradient_x = (right - left) * 0.5
    gradient_y = (down - up) * 0.5

    return gradient_x, gradient_y


def compute_wind_vector(gradient_x, gradient_y, coriolis=0.22):
    # Simple geostrophic-like balance (fast + stable)
    u = -gradient_y * coriolis
    v = gradient_x * coriolis
    return u, v

def calculate_full_physics(
    lat, lon, s_lat, s_lon, p,
    level=1000,
    micro_scale=0.0,
    front_lat=None,
    terrain_friction=1.0,
    inland_decay=True,
):
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

    eff_v = (
        v_max
        * (local_rh / 85.0)
        * (0.6 + outflow / 2.5)
        * sst_mult
        * l_mult
        * front_mod
    )

    B = 1.3 + (eff_v / 150)

    v_sym = eff_v * np.sqrt(
        (r_max / r) ** B * np.exp(1 - (r_max / r) ** B)
    )

    shear_rad = np.radians(shear_dir)
    shear_effect = 1 + ((1.0 - symmetry) * (shear_mag / 45)) * np.cos(angle - shear_rad)

    inflow = 25 if level > 500 else -30

    wind_angle_rad = angle + (np.pi / 2) + np.radians(
        inflow if r > r_max else inflow / 2
    )

    v_forward = f_speed * 0.5 * np.cos(
        wind_angle_rad - np.radians(f_dir)
    )

    surface_wind = max(
        0,
        ((v_sym * shear_effect) + v_forward) * terrain_friction
    )

    # --- pressure gradient boost ---
    gradient_boost = (v_max / max(r_max, 1)) * 0.08
    surface_wind *= (1.0 + gradient_boost)

    # --- inland decay ---
    if inland_decay:
        if lat > 30.15:
            inland_miles = (lat - 30.15) * 69
            land_decay = max(0.72, np.exp(-inland_miles / 260.0))
            surface_wind *= land_decay

    return surface_wind, np.degrees(wind_angle_rad), r

def get_synthetic_products(
    lat,
    lon,
    s_lat,
    s_lon,
    p,
    radar_coords=None,
    micro_scale=0.0,
    front_lat=None,
    terrain_friction=1.0,
    coastal_exposure=1.0,
    ewr_phase=0.0,
    radar_decay=True,
    cone_of_silence=True,
):
    v_max, r_max, _, _, shear_mag, shear_dir, rh, _, symmetry, _ = p

    w, wd, r = calculate_full_physics(
        lat, lon, s_lat, s_lon, p,
        micro_scale=micro_scale,
        front_lat=front_lat,
        terrain_friction=terrain_friction,
    )

    angle = np.arctan2((lat - s_lat) * 69, (lon - s_lon) * 53)

    is_major = v_max >= 96

    shear_rad = np.radians(shear_dir)
    shear_offset_x = (shear_mag / 20) * np.cos(shear_rad)
    shear_offset_y = (shear_mag / 20) * np.sin(shear_rad)

    r_adj = np.sqrt(
        ((lon - s_lon - shear_offset_x / 53) * 53) ** 2 +
        ((lat - s_lat - shear_offset_y / 69) * 69) ** 2
    )

    eyewall = 65 * np.exp(-((r - r_max) ** 2) / (r_max * 0.3) ** 2)

    moisture_flux = (rh / 100) * (1 + (shear_mag / 100))
    shield = 42 * moisture_flux * np.exp(-r_adj / (r_max * 5.0))

    bands = max(
        0,
        np.sin(r / (r_max * 0.8) - angle * 3.0) * 35 * np.exp(-r / 200)
    )

    front_rain = (
        30 * np.exp(-abs(lat - front_lat) * 2)
        if (front_lat and lat > front_lat - 0.5)
        else 0
    )

    # Eyewall replacement cycle structure.
    if ewr_phase > 0:
        outer_ring = 0.72 * 65 * np.exp(
            -((r - (r_max * 1.8)) ** 2) / (r_max * 0.58) ** 2
        )
        eyewall = eyewall * (1 - min(1.0, ewr_phase))
        bands = max(bands, outer_ring)
    dbz = max(eyewall, shield, bands, front_rain) * symmetry

    if r < r_max * (0.15 if v_max >= 96 else 0.4):
        dbz *= 0.1

    if radar_coords:
        rdx, rdy = (lon - radar_coords[1]) * 53, (lat - radar_coords[0]) * 69
        angle_to_radar = np.arctan2(rdy, rdx)

        radial_v = (w * 1.15) * np.cos(np.radians(wd) - angle_to_radar)
        aliased_v = np.clip(radial_v, -149, 149)

        range_mi = np.sqrt(rdx**2 + rdy**2)
        range_km = range_mi * 1.60934
        beam_height_km = 0.018 * (range_km ** 1.12)

        if radar_decay:
            dbz *= np.exp(-range_km / 320.0)
            aliased_v *= np.exp(-range_km / 420.0)

        if cone_of_silence and range_km < 7:
            dbz *= 0.35
            aliased_v *= 0.25

        if beam_height_km > 7.0:
            dbz *= 0.65
            aliased_v *= 0.70
    else:
        aliased_v = 0
        beam_height_km = 0.0

    surge = 0
    if 30.00 <= lat <= 30.55:
        dist_mult = np.exp(-abs(lat - 30.25) * 5)
        surge = (w ** 1.9 / 1700) * (1.8 if lon > s_lon else -0.5) * dist_mult * coastal_exposure

    prob = 90 if w >= 96 else 60 if w >= 64 else 30 if w >= 34 else 0

    return min(78, dbz), aliased_v, surge, prob, beam_height_km

# -----------------------------
# 2. MOBILE COUNTY ZONES & CITIES
# -----------------------------
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
    "Dauphin Island": (30.2505, -88.1097),
}

PLACE_CONTEXT = {
    "Mobile": {
        "terrain": "urban core and mixed residential areas",
        "risk_notes": "Damage may be amplified by dense development, traffic signal failures, and widespread tree and powerline impacts."
    },
    "Downtown Mobile": {
        "terrain": "dense urban district with mid-rise buildings, port infrastructure, and exposed road corridors",
        "risk_notes": "Strong winds may accelerate around taller buildings and create dangerous debris conditions near commercial blocks and port facilities."
    },
    "Prichard": {
        "terrain": "urban-residential corridor with tree cover and older infrastructure",
        "risk_notes": "Older structures, snapped limbs, and scattered utility damage may produce blocked roads and long-duration outages."
    },
    "Chickasaw": {
        "terrain": "compact residential-industrial area",
        "risk_notes": "Damage risk includes roof failure, tree fall, and industrial debris in exposed corridors."
    },
    "Saraland": {
        "terrain": "suburban-commercial corridor with neighborhoods and retail exposure",
        "risk_notes": "Shopping centers, signage, and wide roadways are vulnerable to destructive gusts and flying debris."
    },
    "Satsuma": {
        "terrain": "suburban wooded residential area",
        "risk_notes": "Large trees and neighborhood access roads may become blocked quickly under prolonged extreme winds."
    },
    "Axis": {
        "terrain": "rural-residential area with open exposure",
        "risk_notes": "Open terrain may allow stronger gust penetration with falling timber and utility line damage."
    },
    "Semmes": {
        "terrain": "inland suburban-wooded terrain",
        "risk_notes": "Tree damage and prolonged utility outages may become a primary hazard in residential sections."
    },
    "Wilmer": {
        "terrain": "rural inland terrain with open stretches and scattered development",
        "risk_notes": "Exposure across open land may increase structural and tree-fall damage potential."
    },
    "Theodore": {
        "terrain": "mixed residential, industrial, and coastal-transition corridor",
        "risk_notes": "Structural damage risk is elevated near exposed industrial areas and open road corridors."
    },
    "Dawes": {
        "terrain": "suburban corridor with commercial strips and residential pockets",
        "risk_notes": "Signage, roofing, and roadside debris may become major hazards."
    },
    "Tillmans Corner": {
        "terrain": "busy commercial corridor with major road exposure",
        "risk_notes": "Gas stations, storefronts, signage, and wide parking lots increase debris and access hazards."
    },
    "Irvington": {
        "terrain": "low-lying semi-rural area with scattered communities",
        "risk_notes": "Falling trees, road blockage, and isolated structural failure may cut off local access routes."
    },
    "Grand Bay": {
        "terrain": "rural low-lying inland-coastal transition zone",
        "risk_notes": "Open terrain and tree exposure may support widespread wind damage and difficult travel conditions."
    },
    "Bayou La Batre": {
        "terrain": "working waterfront fishing community with marine exposure",
        "risk_notes": "Extreme wind impacts may be severe near docks, marine facilities, and exposed low-elevation structures."
    },
    "Coden": {
        "terrain": "coastal low-lying community with strong marine exposure",
        "risk_notes": "Extreme winds may combine with coastal flooding, structural damage, and isolation of access routes."
    },
    "Mount Vernon": {
        "terrain": "inland industrial-rural corridor",
        "risk_notes": "Industrial exposure and tree damage may create localized but serious access and debris hazards."
    },
    "Citronelle": {
        "terrain": "inland wooded small-town environment",
        "risk_notes": "Tree fall and roof damage may become widespread if strong winds remain intact inland."
    },
    "West Mobile": {
        "terrain": "large suburban-commercial corridor with shopping centers, multilane roads, and dense retail strips",
        "risk_notes": "Big-box stores, parking lots, power poles, signage, and broad road exposure increase debris and infrastructure vulnerability."
    },
    "Dauphin Island": {
        "terrain": "exposed barrier island with sparse development and limited access",
        "risk_notes": "Extreme isolation risk, dune overwash, structural exposure, and rapid loss of safe travel routes are likely in severe scenarios."
    },
}

RADAR_SITES = {
    "KMOB": (30.67, -88.24),
    "KLIX": (30.33, -89.82),
    "KEVX": (30.56, -85.92),
}


# -----------------------------
# 3. COLOR TABLES / LEGENDS
# -----------------------------
def nws_reflectivity_color(dbz):
    if dbz < 5:
        return None
    if dbz < 20:
        return "#04e9e7"
    if dbz < 25:
        return "#019ff4"
    if dbz < 30:
        return "#0300f4"
    if dbz < 35:
        return "#02fd02"
    if dbz < 40:
        return "#01c501"
    if dbz < 45:
        return "#008e00"
    if dbz < 50:
        return "#fdf802"
    if dbz < 55:
        return "#e5bc00"
    if dbz < 60:
        return "#fd9500"
    if dbz < 65:
        return "#fd0000"
    if dbz < 70:
        return "#d40000"
    return "#bc00bc"


def velocity_color_hyperrealistic(vel):
    if abs(vel) < 3:
        return None

    mag = min(149.0, abs(float(vel)))

    if vel < 0:
        if mag < 10:
            return "#0b2818"
        elif mag < 20:
            return "#0e3a22"
        elif mag < 30:
            return "#11502f"
        elif mag < 40:
            return "#14643b"
        elif mag < 50:
            return "#187947"
        elif mag < 60:
            return "#1d8f56"
        elif mag < 70:
            return "#16956b"
        elif mag < 80:
            return "#0b8f7e"
        elif mag < 90:
            return "#0b8691"
        elif mag < 100:
            return "#0f78a3"
        elif mag < 110:
            return "#1367b2"
        elif mag < 120:
            return "#1854bc"
        elif mag < 130:
            return "#1d43c5"
        elif mag < 140:
            return "#2431ca"
        else:
            return "#2c1fcf"
    else:
        if mag < 10:
            return "#2a0b0b"
        elif mag < 20:
            return "#3c1010"
        elif mag < 30:
            return "#551313"
        elif mag < 40:
            return "#6d1414"
        elif mag < 50:
            return "#861616"
        elif mag < 60:
            return "#9f1717"
        elif mag < 70:
            return "#b31b1b"
        elif mag < 80:
            return "#bf1828"
        elif mag < 90:
            return "#c3153a"
        elif mag < 100:
            return "#c1144f"
        elif mag < 110:
            return "#bc1364"
        elif mag < 120:
            return "#b5147c"
        elif mag < 130:
            return "#ab1693"
        elif mag < 140:
            return "#9c18ab"
        else:
            return "#8d1bc2"


def surge_color(surge):
    if abs(surge) < 0.5:
        return None
    if surge <= 0:
        return "#00ced1"
    if surge <= 3:
        return "#ffd700"
    if surge <= 6:
        return "#ff8c00"
    if surge <= 9:
        return "#ff0000"
    if surge <= 12:
        return "#8b0000"
    return "#4b0082"


def wind_prob_color(prob):
    if prob <= 0:
        return None
    if prob >= 90:
        return "#ff00ff"
    if prob >= 60:
        return "#ff8c00"
    return "#ffff00"


def add_map_legend(m, radar_view):
    if radar_view == "Reflectivity (dBZ)":
        legend_html = """
        <div style="
        position: fixed;
        bottom: 18px; left: 18px; z-index: 9999;
        background: rgba(0,0,0,0.88);
        color: white;
        padding: 12px;
        border-radius: 10px;
        font-size: 12px;
        min-width: 175px;
        box-shadow: 0 0 12px rgba(0,0,0,0.45);
        ">
        <b>Reflectivity (dBZ)</b><br>
        <span style='color:#04e9e7'>■</span> 5-20 Light<br>
        <span style='color:#019ff4'>■</span> 20-25 Light-Mod<br>
        <span style='color:#0300f4'>■</span> 25-30 Moderate<br>
        <span style='color:#02fd02'>■</span> 30-35 Moderate/Heavy<br>
        <span style='color:#01c501'>■</span> 35-40 Heavy<br>
        <span style='color:#008e00'>■</span> 40-45 Very Heavy<br>
        <span style='color:#fdf802'>■</span> 45-50 Intense<br>
        <span style='color:#fd9500'>■</span> 55-60 Extreme<br>
        <span style='color:#fd0000'>■</span> 60-65 Violent<br>
        <span style='color:#bc00bc'>■</span> 70+ Core / Hail-like
        </div>
        """
    elif radar_view == "Velocity (kts)":
        legend_html = """
        <div style="
        position: fixed;
        bottom: 18px; left: 18px; z-index: 9999;
        background: rgba(0,0,0,0.88);
        color: white;
        padding: 12px;
        border-radius: 10px;
        font-size: 12px;
        min-width: 185px;
        box-shadow: 0 0 12px rgba(0,0,0,0.45);
        ">
        <b>Radial Velocity (kts)</b><br>
        <span style='color:#16956b'>■</span> Inbound 20-70<br>
        <span style='color:#0f78a3'>■</span> Inbound 70-110<br>
        <span style='color:#2c1fcf'>■</span> Inbound 110-149<br>
        <span style='color:#861616'>■</span> Outbound 20-50<br>
        <span style='color:#c3153a'>■</span> Outbound 80-90<br>
        <span style='color:#8d1bc2'>■</span> Outbound 140-149<br>
        Range: -149 to +149 kts
        </div>
        """
    elif radar_view == "Storm Surge":
        legend_html = """
        <div style="
        position: fixed;
        bottom: 18px; left: 18px; z-index: 9999;
        background: rgba(0,0,0,0.88);
        color: white;
        padding: 12px;
        border-radius: 10px;
        font-size: 12px;
        min-width: 165px;
        box-shadow: 0 0 12px rgba(0,0,0,0.45);
        ">
        <b>Storm Surge (ft)</b><br>
        <span style='color:#00ced1'>■</span> Offshore / negative<br>
        <span style='color:#ffd700'>■</span> 1-3<br>
        <span style='color:#ff8c00'>■</span> 3-6<br>
        <span style='color:#ff0000'>■</span> 6-9<br>
        <span style='color:#8b0000'>■</span> 9-12<br>
        <span style='color:#4b0082'>■</span> 12+
        </div>
        """
    else:
        legend_html = """
        <div style="
        position: fixed;
        bottom: 18px; left: 18px; z-index: 9999;
        background: rgba(0,0,0,0.88);
        color: white;
        padding: 12px;
        border-radius: 10px;
        font-size: 12px;
        min-width: 170px;
        box-shadow: 0 0 12px rgba(0,0,0,0.45);
        ">
        <b>Wind Probability</b><br>
        <span style='color:#ffff00'>■</span> 30% Possible<br>
        <span style='color:#ff8c00'>■</span> 60% Likely<br>
        <span style='color:#ff00ff'>■</span> 90% Near Certain
        </div>
        """

    legend_html += """
    <hr style='border:0.5px solid #666; margin:6px 0;'>
    <span style='color:#ff4d4d'>■</span> Extreme Wind Warning Polygon
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# -----------------------------
# 4. ZONE / ENVIRONMENT HELPERS
# -----------------------------
def get_zone_for_point(lat, lon):
    for zone_name, meta in ZONES.items():
        (lat1, lon1), (lat2, lon2) = meta["bounds"]
        if min(lat1, lat2) <= lat <= max(lat1, lat2) and min(lon1, lon2) <= lon <= max(lon1, lon2):
            return zone_name

    nearest = min(
        ZONES.keys(),
        key=lambda z: np.hypot((lat - ZONES[z]["center"][0]) * 69, (lon - ZONES[z]["center"][1]) * 53)
    )
    return nearest


def get_zone_meta(lat, lon):
    zone_name = get_zone_for_point(lat, lon)
    return zone_name, ZONES[zone_name]


def compute_local_environment(
    lat, lon, s_lat, s_lon, p, radar_coords, front_lat,
    pressure_drop_hpa=32, dry_air=0, urban_heat=0, ewr_phase=0.0
):
    zone_name, zone_meta = get_zone_meta(lat, lon)

    dbz, vel, surge_raw, prob, beam_height_km = get_synthetic_products(
        lat, lon, s_lat, s_lon, p,
        radar_coords=radar_coords,
        front_lat=front_lat,
        terrain_friction=zone_meta["terrain_friction"],
        coastal_exposure=zone_meta["coastal_exposure"],
        ewr_phase=ewr_phase,
    )

    w_kts, wd, r = calculate_full_physics(
        lat, lon, s_lat, s_lon, p,
        front_lat=front_lat,
        terrain_friction=zone_meta["terrain_friction"],
    )

    v_max, r_max, f_speed, f_dir, shear_mag, shear_dir, rh, outflow, symmetry, sst_mult = p

    # -----------------------------
    # WIND ADJUSTMENTS
    # -----------------------------
    coastal_boost = 1 + (zone_meta["coastal_exposure"] * 0.08)
    friction_recovery = 1 + ((1 - zone_meta["terrain_friction"]) * 0.15)
    w_kts = w_kts * coastal_boost * friction_recovery

    # -----------------------------
    # URBAN / INLAND DECAY (NEW)
    # -----------------------------
    urban = zone_meta["urban_factor"]  # 0 → rural, 1 → dense city

    # stronger decay with distance + urban density
    inland_decay = np.clip((r / 80), 0, 1.5)
    urban_penalty = 1 - (urban * 0.35 * inland_decay)

    # apply to sustained wind
    w_kts = w_kts * urban_penalty

    # -----------------------------
    # REALISTIC GUST LOGIC (IMPROVED)
    # -----------------------------
    if r < r_max:
        gust_kts = w_kts * 1.20
    else:
        inland_factor = np.clip(1 - (r / 120), 0.7, 1.0)
        urban_factor = 1 - (zone_meta["urban_factor"] * 0.25)
        gust_kts = w_kts * inland_factor * urban_factor

    # caps
    gust_kts = min(gust_kts, 155 if r < 20 else 130)

    # -----------------------------
    # TEMPERATURE / DEWPOINT
    # -----------------------------
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

    # -----------------------------
    # VISIBILITY
    # -----------------------------
    rain_factor = dbz / 75
    wind_reduction = min(1.2, w_kts / 120)

    vis_mi = 10 - (rain_factor * 7.2) - (wind_reduction * 1.4)
    if zone_meta["coastal_exposure"] > 0.65:
        vis_mi -= 0.4

    vis_mi = float(np.clip(vis_mi, 0.2, 10.0))

    # -----------------------------
    # WIND / RAIN DIRECTION
    # -----------------------------
    dx = (lon - s_lon) * 53
    dy = (lat - s_lat) * 69

    storm_bearing_from_center = np.degrees(np.arctan2(dx, dy)) % 360
    inflow_dir = (storm_bearing_from_center + 180 - 20) % 360

    rain_dir = normalize_angle(
        inflow_dir + (shear_mag * 0.15) - ((1 - symmetry) * 25)
    )

    # -----------------------------
    # SURGE
    # -----------------------------
    surge_ft = max(
        0.0,
        surge_raw + zone_meta["surge_bias"] + (zone_meta["coastal_exposure"] * 0.6)
    )

    # -----------------------------
    # TORNADO RISK
    # -----------------------------
    right_front_bonus = max(
        0,
        np.cos(np.radians((storm_bearing_from_center - f_dir) - 45))
    )

    shear_term = np.clip((shear_mag - 10) / 40, 0, 1)
    moisture_term = np.clip((rh - 65) / 30, 0, 1)
    friction_term = np.clip(
        (1 - zone_meta["terrain_friction"]) * 3.2 + zone_meta["urban_factor"] * 0.6,
        0, 1
    )
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

    # -----------------------------
    # FINAL CONVERSIONS
    # -----------------------------
    wind_mph = kt_to_mph(w_kts)
    gust_mph = kt_to_mph(gust_kts)

    return {
        "zone": zone_name,
        "wind_kts": float(w_kts),
        "gust_kts": float(gust_kts),
        "wind_mph": float(wind_mph),
        "gust_mph": float(gust_mph),
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
        "beam_height_km": float(beam_height_km),
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


def zone_summary_table(s_lat, s_lon, p, radar_coords, front_lat, pressure_drop_hpa, dry_air, urban_heat, ewr_phase):
    rows = []
    for zone_name, meta in ZONES.items():
        lat, lon = meta["center"]
        env = compute_local_environment(
            lat, lon, s_lat, s_lon, p, radar_coords, front_lat,
            pressure_drop_hpa=pressure_drop_hpa,
            dry_air=dry_air,
            urban_heat=urban_heat,
            ewr_phase=ewr_phase,
        )
        rows.append({
            "Zone": zone_name,
            "Wind": f"{env['wind_kts']:.0f} kt / {env['wind_mph']:.0f} mph",
            "Gust": f"{env['gust_kts']:.0f} kt / {env['gust_mph']:.0f} mph",
            "Temp": f"{env['temp_f']:.0f}°F",
            "Dewpoint": f"{env['dewp_f']:.0f}°F",
            "Visibility": f"{env['visibility_mi']:.1f} mi",
            "Rain Dir": env["rain_dir_text"],
            "Surge": f"{env['surge_ft']:.1f} ft",
            "Tornado": f"{env['tornado_label']} ({env['tornado_risk']:.0f})",
        })
    return pd.DataFrame(rows)


def forecast_cone_radius_mi(hour, r_max):
    base = {
        0: 1.2,
        6: 1.6,
        12: 2.0,
        24: 2.8,
        36: 3.6,
        48: 4.5,
    }

    scale = base.get(hour, 3.0)

    return r_max * scale


def build_forecast_track(l_lat, l_lon, f_speed, f_dir, r_max):
    points = []
    for hour in [0, 6, 12, 24, 36, 48]:
        dist_moved = (f_speed * hour) / 69.0
        f_lat = l_lat + (dist_moved * np.cos(np.radians(f_dir)))
        f_lon = l_lon + (dist_moved * np.sin(np.radians(f_dir)))
        points.append({
            "hour": hour,
            "lat": f_lat,
            "lon": f_lon,
            "cone_radius_mi": forecast_cone_radius_mi(hour, r_max),
        })
    return points

def offset_latlon(lat, lon, miles, bearing_deg):
    """Move a point by distance/bearing. Good enough for local warning polygons."""
    bearing = np.radians(bearing_deg)
    dlat = (miles * np.cos(bearing)) / 69.0
    dlon = (miles * np.sin(bearing)) / (53.0 * np.cos(np.radians(max(1, abs(lat)))))
    return lat + dlat, lon + dlon

def build_hurricane_warning_polygon(center_lat, center_lon, r_max):
    radius = max(40, r_max * 3.5)

    coords = []
    for angle in np.linspace(0, 360, 36):
        lat = center_lat + (radius / 69.0) * np.cos(np.radians(angle))
        lon = center_lon + (radius / 53.0) * np.sin(np.radians(angle))
        coords.append([lat, lon])

    return coords

def build_surge_polygon(center_lat, center_lon, heading_deg, r_max):
    length = max(60, r_max * 4)
    width = max(20, r_max * 2)

    front_lat, front_lon = offset_latlon(center_lat, center_lon, length, heading_deg)
    left_lat, left_lon = offset_latlon(front_lat, front_lon, width, heading_deg - 90)
    right_lat, right_lon = offset_latlon(front_lat, front_lon, width, heading_deg + 90)

    rear_lat, rear_lon = offset_latlon(center_lat, center_lon, width, heading_deg + 180)

    return [
        [rear_lat, rear_lon],
        [left_lat, left_lon],
        [front_lat, front_lon],
        [right_lat, right_lon],
    ]

def build_extreme_wind_warning_polygon(
    center_lat,
    center_lon,
    heading_deg,
    forward_speed,
    r_max,
    v_max,
    symmetry,
    shear_mag,
    terrain_friction,
    urban_factor,
):
    """
    Build a realistic warning polygon elongated along the storm motion.
    Shapes the polygon more like a real NWS downstream wind warning.
    """
    intensity_factor = np.clip(kt_to_mph(v_max) / 120.0, 0.75, 1.6)
    asymmetry_factor = 1.0 + ((1.0 - symmetry) * 0.8) + (shear_mag / 120.0)
    roughness_factor = 1.0 + (urban_factor * 0.35) + ((1.0 - terrain_friction) * 0.4)

    # -----------------------------
    # DYNAMIC SIZE (RMW-DRIVEN)
    # -----------------------------
    base = r_max

    lead_miles = (base * 1.8 + forward_speed * 1.6) * intensity_factor * asymmetry_factor
    trail_miles = base * 0.6 * roughness_factor

    half_width_left = base * 0.7 * roughness_factor
    half_width_right = base * 1.0 * intensity_factor * asymmetry_factor

    # soft caps (less restrictive)
    lead_miles = np.clip(lead_miles, 25, 140)
    trail_miles = np.clip(trail_miles, 10, 45)
    half_width_left = np.clip(half_width_left, 10, 45)
    half_width_right = np.clip(half_width_right, 15, 65)

    front_lat, front_lon = offset_latlon(center_lat, center_lon, lead_miles, heading_deg)
    rear_lat, rear_lon = offset_latlon(center_lat, center_lon, trail_miles, heading_deg + 180)

    left_bearing = heading_deg - 90
    right_bearing = heading_deg + 90

    f_left_lat, f_left_lon = offset_latlon(front_lat, front_lon, half_width_left, left_bearing)
    f_right_lat, f_right_lon = offset_latlon(front_lat, front_lon, half_width_right, right_bearing)

    r_left_lat, r_left_lon = offset_latlon(rear_lat, rear_lon, half_width_left * 0.72, left_bearing)
    r_right_lat, r_right_lon = offset_latlon(rear_lat, rear_lon, half_width_right * 0.72, right_bearing)

    tip_lat, tip_lon = offset_latlon(front_lat, front_lon, max(4, r_max * 0.25), heading_deg + 20)

    polygon = [
        [r_left_lat, r_left_lon],
        [f_left_lat, f_left_lon],
        [tip_lat, tip_lon],
        [f_right_lat, f_right_lon],
        [r_right_lat, r_right_lon],
    ]
    return polygon


def pick_impacted_places(polygon, city_points, max_places=6):
    """
    Pick city labels that are closest to or inside the warning polygon envelope.
    This is synthetic but gives a realistic impacted-place list.
    """
    lats = [pt[0] for pt in polygon]
    lons = [pt[1] for pt in polygon]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    candidates = []
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)

    for name, (lat, lon) in city_points.items():
        pad = 0.08
        if (min_lat - pad) <= lat <= (max_lat + pad) and (min_lon - pad) <= lon <= (max_lon + pad):
            dist = np.hypot((lat - center_lat) * 69, (lon - center_lon) * 53)
            candidates.append((dist, name))

    candidates = sorted(candidates, key=lambda x: x[0])
    return [name for _, name in candidates[:max_places]]


def generate_fake_ugc():
    """Synthetic UGC-like code for display only."""
    county_part = str(np.random.randint(3, 9)).zfill(3)
    zone_part = str(np.random.randint(1, 999)).zfill(3)
    return f"ALC{county_part}-{zone_part}"


def build_localized_risk_text(selected_places, gust_mph):
    place_bits = []

    for place in selected_places[:4]:
        if place in PLACE_CONTEXT:
            ctx = PLACE_CONTEXT[place]
            place_bits.append(f"{place}: {ctx['risk_notes']}")

    if gust_mph >= 130:
        severity_line = (
            "This scenario supports destructive wind damage comparable to the most dangerous hurricane core impacts, "
            "including major roof failure, extensive tree loss, and long-duration utility failure."
        )
    elif gust_mph >= 110:
        severity_line = (
            "This scenario supports widespread destructive wind damage, including major tree loss, structural damage, "
            "and impassable roads from debris."
        )
    else:
        severity_line = (
            "This scenario supports scattered to widespread wind damage with falling trees, utility damage, and dangerous debris."
        )

    if not place_bits:
        return severity_line

    return severity_line + " " + " ".join(place_bits)

def summarize_place_terrain(selected_places):
    terrain_bits = []
    for place in selected_places[:3]:
        if place in PLACE_CONTEXT:
            terrain_bits.append(f"{place}: {PLACE_CONTEXT[place]['terrain']}")
    return " | ".join(terrain_bits) if terrain_bits else "mixed coastal and inland terrain"

    for place in selected_places[:4]:
        if place in PLACE_CONTEXT:
            ctx = PLACE_CONTEXT[place]
            place_bits.append(f"{place}: {ctx['risk_notes']}")

    if gust_mph >= 130:
        severity_line = (
            "This scenario supports destructive wind damage comparable to the most dangerous hurricane core impacts, "
            "including major roof failure, extensive tree loss, and long-duration utility failure."
        )
    elif gust_mph >= 110:
        severity_line = (
            "This scenario supports widespread destructive wind damage, including major tree loss, structural damage, "
            "and impassable roads from debris."
        )
    else:
        severity_line = (
            "This scenario supports scattered to widespread wind damage with falling trees, utility damage, and dangerous debris."
        )

    if not place_bits:
        return severity_line

    return severity_line + " " + " ".join(place_bits)

def generate_extreme_wind_warning_text(
    polygon,
    landfall_lat,
    landfall_lon,
    current_lat,
    current_lon,
    heading_deg,
    f_speed,
    v_max,
    wind_mph,
    gust_mph,
    selected_places,
):
    """
    Example warning text styled like NWS formatting, but clearly synthetic.
    """
    issue_hour = np.random.randint(1, 12)
    issue_min = np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    expire_min = (issue_min + 45) % 60
    ugc = generate_fake_ugc()

    lat1, lon1 = polygon[0]
    lat2, lon2 = polygon[1]
    lat3, lon3 = polygon[2]
    lat4, lon4 = polygon[3]
    lat5, lon5 = polygon[4]

    places_line = ", ".join(selected_places) if selected_places else "portions of coastal Mobile County"
    localized_risk_text = build_localized_risk_text(selected_places, gust_mph)
    terrain_summary = summarize_place_terrain(selected_places)
    motion_mph = int(round(f_speed * 1.15078))

    text = f"""BULLETIN - EAS ACTIVATION REQUESTED
Extreme Wind Warning
National Weather Service Mobile AL
{issue_hour:02d}{issue_min:02d} PM CDT Mon Apr 13 2026

...THIS IS A SYNTHETIC EXAMPLE WARNING FOR SANDBOX DISPLAY...
...EXTREME WIND WARNING FOR SOUTHERN MOBILE COUNTY...

* WHAT...Widespread destructive winds of {int(round(wind_mph))} to {int(round(gust_mph))} mph associated with the inner core of a major hurricane.

* WHERE...Including {places_line}.

* WHEN...Until {issue_hour:02d}{expire_min:02d} PM CDT.

* IMPACTS...Expect extremely dangerous winds capable of producing extensive structural damage, downed trees, blocked roads, and prolonged power outages. Shelter in the interior portion of a well-built structure away from windows.

* LOCAL RISK CONTEXT...{localized_risk_text}

* ADDITIONAL DETAILS...
At {issue_hour:02d}{issue_min:02d} PM CDT, the simulated landfalling eyewall was centered near {current_lat:.2f}N {abs(current_lon):.2f}W, moving {deg_to_compass(heading_deg)} at {motion_mph} mph.
Landfall reference point: {landfall_lat:.2f}N {abs(landfall_lon):.2f}W.
Maximum sustained winds were estimated near {int(round(kt_to_mph(v_max)))} mph.
Terrain focus within the warned area includes {terrain_summary}.

LAT...LON {int(lat1*100):04d} {int(abs(lon1)*100):04d} {int(lat2*100):04d} {int(abs(lon2)*100):04d} {int(lat3*100):04d} {int(abs(lon3)*100):04d}
      {int(lat4*100):04d} {int(abs(lon4)*100):04d} {int(lat5*100):04d} {int(abs(lon5)*100):04d}

$$
{ugc}
"""
    return text

# -----------------------------
# 5. SESSION STATE
# -----------------------------
if "active_radar" not in st.session_state:
    st.session_state.active_radar = "KMOB"
if "loop_idx" not in st.session_state:
    st.session_state.loop_idx = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

geolocator = Nominatim(user_agent="lhim_mobile_county_v40_hyperrealistic")

# -----------------------------
# 6. UI & SIDEBAR
# -----------------------------
st.set_page_config(layout="wide", page_title="LHIM Mobile County v4.0 | Hyperrealistic Mode")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ LHIM Mobile County v4.0")

    radar_view = st.radio(
        "Display Mode",
        ["Reflectivity (dBZ)", "Velocity (kts)", "Storm Surge", "Wind Prob."]
    )

    st.subheader("📡 Radar Controls")
    st.session_state.active_radar = st.selectbox(
        "Radar Site",
        list(RADAR_SITES.keys()),
        index=list(RADAR_SITES.keys()).index(st.session_state.active_radar)
    )

    run_loop = st.checkbox("🔄 Enable Radar Loop", value=st.session_state.is_playing)
    st.session_state.is_playing = run_loop

    current_time_offset = st.slider("Time Offset (Hours)", -12, 12, st.session_state.loop_idx)
    st.session_state.loop_idx = current_time_offset

    with st.expander("🗺️ Base Map Controls", expanded=True):
        basemap_mode = st.selectbox("Base Map", ["Dark", "Street", "Satellite"], index=0)
        enable_satellite = st.checkbox("Enable Satellite Layer Toggle", value=True)
        enable_street = st.checkbox("Enable Street Layer Toggle", value=True)
        enable_dark = st.checkbox("Enable Dark Layer Toggle", value=True)
        enable_traffic = st.checkbox("Enable Traffic Overlay", value=False)
        traffic_tile_url = st.text_input("Traffic Tile URL", value="")

    st.subheader("🌀 Storm Structure")
    v_max = st.slider("Intensity (kts)", 40, 160, 115)
    r_max = st.slider("RMW (miles)", 10, 60, 25)
    f_speed = st.slider("Forward Speed", 2, 40, 12)
    f_dir = st.slider("Heading", 0, 360, 330)
    l_lat = st.number_input("Landfall Lat", value=30.35, format="%.4f")
    l_lon = st.number_input("Landfall Lon", value=-88.15, format="%.4f")
    res_steps = st.select_slider("Quality", options=[30, 45, 60], value=45)

    with st.expander("⚠️ Warning Settings"):
        show_hurricane_warning = st.checkbox("Show Hurricane Warning", value=True)
        show_surge_warning = st.checkbox("Show Storm Surge Warning", value=True)

        hurricane_opacity = st.slider("Hurricane Warning Opacity", 0.1, 1.0, 0.4)
        surge_opacity = st.slider("Storm Surge Warning Opacity", 0.1, 1.0, 0.4)
        extreme_opacity = st.slider("Extreme Wind Warning Opacity", 0.1, 1.0, 0.5)

        warning_distance_trigger = st.slider("Trigger Distance to Land (mi)", 10, 150, 60)  
        show_warnings = st.checkbox("Overlay Surge Warnings", value=True)
        show_extreme_wind_warning = st.checkbox("Show Extreme Wind Warning Polygon", value=True)
        extreme_wind_threshold_mph = st.slider("Extreme Wind Warning Trigger (mph)", 80, 140, 115)
        show_warning_text_panel = st.checkbox("Show Example Warning Text", value=True)
        surge_threshold = st.slider("Warning Trigger (ft)", 3, 12, 6)
        show_zone_boxes = st.checkbox("Show Zone Boxes", value=True)
        show_city_markers = st.checkbox("Show City Markers", value=True)
        show_forecast_track = st.checkbox("Show Forecast Track", value=True)
        show_cone = st.checkbox("Show Cone of Uncertainty", value=True)

    with st.expander("🌡️ Environmental Layers", expanded=True):
        season_month = st.selectbox("Seasonal SST Month", ["June","July","August","September","October","November"], index=3)
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
        ewr_phase = st.slider("Eyewall Cycle Phase", 0.0, 1.0, 0.0, 0.05)

    st.subheader("📍 Mobile County Selector")
    selected_zone = st.selectbox("Zone", list(ZONES.keys()), index=list(ZONES.keys()).index("Downtown-Mobile"))
    selected_city = st.selectbox("City / Place", list(CITY_POINTS.keys()), index=list(CITY_POINTS.keys()).index("Mobile"))
    use_city_selection = st.checkbox("Lock analysis panel to selected city/place", value=False)


# -----------------------------
# STORM CALCULATIONS
# -----------------------------
dist_moved = (f_speed * current_time_offset) / 69.0
current_lat = l_lat + (dist_moved * np.cos(np.radians(f_dir)))
current_lon = l_lon + (dist_moved * np.sin(np.radians(f_dir)))

p = [
    v_max, r_max, f_speed, f_dir,
    shear_mag, shear_dir, rh, outflow,
    symmetry, get_sst_mult(season_month, sst_boost),
]

radar_coords = RADAR_SITES[st.session_state.active_radar]

# -----------------------------
# TOP BAR
# -----------------------------
col_top_left, col_top_right = st.columns([0.92, 0.08])
with col_top_right:
    if st.button("🔍" if not st.session_state.inspector_mode else "❌"):
        st.session_state.inspector_mode = not st.session_state.inspector_mode

mslp = calculate_mslp(v_max, pressure_drop_hpa)
pressure_tendency = calculate_pressure_tendency_mbhr(pressure_drop_hpa)
storm_class = saffir_simpson_category(v_max)
forecast_track = build_forecast_track(l_lat, l_lon, f_speed, f_dir, r_max)

landfall_env = compute_local_environment(
    l_lat, l_lon, current_lat, current_lon, p, radar_coords, front_lat,
    pressure_drop_hpa=pressure_drop_hpa,
    dry_air=dry_air,
    urban_heat=urban_heat,
    ewr_phase=ewr_phase,
)

landfall_zone_name, landfall_zone_meta = get_zone_meta(l_lat, l_lon)

warning_polygon = None
warning_places = []
example_warning_text = ""

if (
    landfall_env["gust_mph"] >= extreme_wind_threshold_mph
    or landfall_env["wind_mph"] >= (extreme_wind_threshold_mph - 15)
):
    warning_polygon = build_extreme_wind_warning_polygon(
        l_lat, l_lon, f_dir, f_speed,
        r_max, v_max, symmetry, shear_mag,
        landfall_zone_meta["terrain_friction"],
        landfall_zone_meta["urban_factor"],
    )

    warning_places = pick_impacted_places(warning_polygon, CITY_POINTS, max_places=6)

    example_warning_text = generate_extreme_wind_warning_text(
        warning_polygon,
        l_lat, l_lon,
        current_lat, current_lon,
        f_dir, f_speed,
        v_max,
        landfall_env["wind_mph"],
        landfall_env["gust_mph"],
        warning_places,
    )

# -----------------------------
# MAP (FIXED — ALWAYS RUNS)
# -----------------------------
m = folium.Map(
    location=[30.5, -88.5],
    zoom_start=7,
    min_zoom=6,
    max_zoom=10,
    max_bounds=True,
)

bounds = [
    [28.0, -91.5],
    [32.5, -84.5],
]

m.fit_bounds(bounds)
m.options["maxBounds"] = bounds
m.options["maxBoundsViscosity"] = 1.0

folium.TileLayer("CartoDB dark_matter").add_to(m)

# -----------------------------
# RADAR GRID
# -----------------------------
lats = np.linspace(l_lat - 2.5, l_lat + 2.5, res_steps)
lons = np.linspace(l_lon - 2.5, l_lon + 2.5, int(res_steps * 1.2))
d_lat, d_lon = lats[1] - lats[0], lons[1] - lons[0]

for lt in lats:
    for ln in lons:
        zone_name, zone_meta = get_zone_meta(lt, ln)

        dbz, vel, surge, prob, beam = get_synthetic_products(
            lt, ln, current_lat, current_lon, p,
            radar_coords=radar_coords,
            front_lat=front_lat,
            terrain_friction=zone_meta["terrain_friction"],
            coastal_exposure=zone_meta["coastal_exposure"],
            ewr_phase=ewr_phase,
        )

        if radar_view == "Reflectivity (dBZ)":
            color = nws_reflectivity_color(dbz)
        elif radar_view == "Velocity (kts)":
            color = velocity_color_hyperrealistic(vel)
        elif radar_view == "Storm Surge":
            color = surge_color(surge)
        else:
            color = wind_prob_color(prob)

        if color:
            folium.Rectangle(
                [[lt, ln], [lt + d_lat, ln + d_lon]],
                color=color,
                fill=True,
                fill_opacity=0.55,
                weight=0
            ).add_to(m)

# -----------------------------
# FORECAST TRACK
# -----------------------------
if show_forecast_track:
    track_coords = [(pt["lat"], pt["lon"]) for pt in forecast_track]
    folium.PolyLine(track_coords, color="white", weight=2.2).add_to(m)

# -----------------------------
# STORM CENTER
# -----------------------------
folium.Marker(
    [current_lat, current_lon],
    tooltip="Storm Center",
    icon=folium.Icon(color="red"),
).add_to(m)

# -----------------------------
# CONE
# -----------------------------
if show_cone and forecast_track:
    cone_coords = []

    for pt in forecast_track:
        radius_mi = forecast_cone_radius_mi(pt["hour"], r_max)
        radius_deg = radius_mi / 69.0

        for angle in np.linspace(0, 360, 24):
            lat = pt["lat"] + radius_deg * np.cos(np.radians(angle))
            lon = pt["lon"] + radius_deg * np.sin(np.radians(angle))
            cone_coords.append((lat, lon))

    folium.Polygon(
        locations=cone_coords,
        color="white",
        weight=2,
        fill=True,
        fill_opacity=0.08
    ).add_to(m)

# -----------------------------
# WARNINGS (FIXED)
# -----------------------------
dist_to_landfall = np.hypot((current_lat - l_lat) * 69, (current_lon - l_lon) * 53)
warnings_active = dist_to_landfall <= warning_distance_trigger

if show_extreme_wind_warning and warnings_active:
    poly = build_extreme_wind_warning_polygon(
        current_lat, current_lon, f_dir, f_speed,
        r_max, v_max, symmetry, shear_mag,
        landfall_zone_meta["terrain_friction"], urban_heat
    )
    if poly and len(poly) >= 3:
        folium.Polygon(
            locations=poly,
            color="red",
            fill=True,
            fill_opacity=extreme_opacity,
            weight=2
        ).add_to(m)

if show_hurricane_warning and warnings_active:
    poly = build_hurricane_warning_polygon(current_lat, current_lon, r_max)
    if poly and len(poly) >= 3:
        folium.Polygon(
            locations=poly,
            color="orange",
            fill=True,
            fill_opacity=extreme_opacity,
            weight=2
        ).add_to(m)

if show_surge_warning and warnings_active:
    poly = build_surge_polygon(current_lat, current_lon, f_dir, r_max)
    if poly and len(poly) >= 3:
        folium.Polygon(
            locations=poly,
            color="purple",
            fill=True,
            fill_opacity=extreme_opacity,
            weight=2
        ).add_to(m)

# -----------------------------
# RENDER MAP
# -----------------------------
map_data = st_folium(
    m,
    height=850,
    use_container_width=True,
    key="main_map",
    returned_objects=["last_clicked"]
)

# -----------------------------
# WARNING PANEL (UNDER MAP ✅)
# -----------------------------
st.subheader("⚠️ Warning Panel")

if show_warning_text_panel and warnings_active:
    if poly and len(poly) >= 5:
        selected_places = pick_impacted_places(poly, CITY_POINTS)

        warning_text = generate_extreme_wind_warning_text(
            poly,
            l_lat, l_lon,
            current_lat, current_lon,
            f_dir, f_speed,
            v_max,
            kt_to_mph(v_max),
            kt_to_mph(v_max) * 1.2,
            selected_places
        )

        st.text_area("Warning Text", warning_text, height=400)
    else:
        st.warning("Polygon not large enough to generate warning text.")

# -----------------------------
# CLICK UPDATE
# -----------------------------
if map_data and map_data.get("last_clicked"):
    st.session_state.last_click = (
        map_data["last_clicked"]["lat"],
        map_data["last_clicked"]["lng"]
    )
    
    # -----------------------------
    # INSPECTOR PANEL
    # -----------------------------
    if st.session_state.inspector_mode:
        dbz, vel, surge, prob, beam = get_synthetic_products(
            inspect_lat, inspect_lon, current_lat, current_lon, p,
            radar_coords=radar_coords
        )

        st.markdown(f"""
        <div style="position: fixed; top:70px; right:20px;
        background: rgba(0,0,0,0.6); padding:10px; border-radius:8px; color:white;">
        <b>Inspector</b><br>
        Lat: {inspect_lat:.3f} Lon: {inspect_lon:.3f}<br>
        Reflectivity: {dbz:.1f} dBZ<br>
        Velocity: {vel:.1f} kt
        </div>
        """, unsafe_allow_html=True)

if show_warning_text_panel and warning_polygon is not None:
    st.divider()
    st.subheader("🚨 Example Extreme Wind Warning")

    tab1, tab2, tab3 = st.tabs(["Warning Text", "Local Risk", "Polygon Meta"])

    with tab1:
        st.markdown(
            f"""
            <div style="
                background:#11131a;
                border:2px solid #ff4d4d;
                border-radius:12px;
                padding:14px;
                color:#f5f7fa;
                font-family:monospace;
                font-size:13px;
                white-space:pre-wrap;
                line-height:1.25;
                max-height:420px;
                overflow-y:auto;
                box-shadow:0 0 14px rgba(255,77,77,0.18);
            ">{example_warning_text}</div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        risk_rows = []
        for place in warning_places:
            if place in PLACE_CONTEXT:
                risk_rows.append({
                    "Place": place,
                    "Terrain": PLACE_CONTEXT[place]["terrain"],
                    "Risk": PLACE_CONTEXT[place]["risk_notes"],
                })
        if risk_rows:
            st.dataframe(pd.DataFrame(risk_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No localized risk notes available for this scenario.")

    with tab3:
        st.dataframe(pd.DataFrame([{
            "Landfall Zone": landfall_zone_name,
            "Storm Heading": deg_to_compass(f_dir),
            "Forward Speed": f"{f_speed:.0f} kt / {kt_to_mph(f_speed):.0f} mph",
            "Intensity": f"{v_max:.0f} kt / {kt_to_mph(v_max):.0f} mph",
            "RMW": f"{r_max:.0f} mi",
            "Symmetry": f"{symmetry:.2f}",
            "Shear": f"{shear_mag:.0f} kt",
            "Terrain Friction": f"{landfall_zone_meta['terrain_friction']:.2f}",
            "Urban Factor": f"{landfall_zone_meta['urban_factor']:.2f}",
        }]), hide_index=True, use_container_width=True)

# -----------------------------
# CLICK LOCATION (RESTORE)
# -----------------------------
click_lat = None
click_lon = None

if map_data and map_data.get("last_clicked"):
    click_lat = map_data["last_clicked"]["lat"]
    click_lon = map_data["last_clicked"]["lng"]
else:
    click_lat, click_lon = current_lat, current_lon
    
# -----------------------------
# ENVIRONMENT (CRITICAL FIX)
# -----------------------------
env = compute_local_environment(
    click_lat,
    click_lon,
    current_lat,
    current_lon,
    p,
    radar_coords,
    front_lat,
    pressure_drop_hpa=pressure_drop_hpa,
    dry_air=dry_air,
    urban_heat=urban_heat,
    ewr_phase=ewr_phase,
)

k1, k2 = st.columns(2)

k1.metric("TEMP", f"{env['temp_f']:.0f}°F")
k1.metric("DEW PT", f"{env['dewp_f']:.0f}°F")
k1.metric("VISIBILITY", f"{env['visibility_mi']:.1f} mi")
k1.metric("SURGE", f"{env['surge_ft']:.1f} ft")
k1.metric("BEAM HT", f"{env['beam_height_km']:.1f} km")

k2.metric("WIND", f"{env['wind_kts']:.0f} kt / {env['wind_mph']:.0f} mph")
k2.metric("GUST", f"{env['gust_kts']:.0f} kt / {env['gust_mph']:.0f} mph")
k2.metric("RAIN DIR", env['rain_dir_text'])
k2.metric("TORNADO", f"{env['tornado_label']} ({env['tornado_risk']:.0f})")
k2.metric("RADAR VEL", f"{env['vel']:.0f} kt")

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
        ewr_phase=ewr_phase,
    )

    forecast_rows.append({
        "Time": f"T+{hour}h",
        "Condition": condition_from_wind(f_env["wind_kts"], f_env["radius_mi"], r_max),
        "Wind": f"{f_env['wind_kts']:.0f} kt / {f_env['wind_mph']:.0f} mph",
        "Gust": f"{f_env['gust_kts']:.0f} kt / {f_env['gust_mph']:.0f} mph",
        "Temp": f"{f_env['temp_f']:.0f}°F",
        "Visibility": f"{f_env['visibility_mi']:.1f} mi",
        "Surge": f"{f_env['surge_ft']:.1f} ft",
        "Tornado": f"{f_env['tornado_label']} ({f_env['tornado_risk']:.0f})",
    })

st.dataframe(pd.DataFrame(forecast_rows), hide_index=True, use_container_width=True)

st.divider()
st.subheader("🧭 Multi-Point Forecast")

mp_rows = []
for pt in forecast_track:
    f_env = compute_local_environment(
        click_lat, click_lon, pt["lat"], pt["lon"], p, radar_coords, front_lat,
        pressure_drop_hpa=pressure_drop_hpa,
        dry_air=dry_air,
        urban_heat=urban_heat,
        ewr_phase=ewr_phase,
    )
    mp_rows.append({
        "Forecast": f"+{pt['hour']}h",
        "Storm Lat": f"{pt['lat']:.2f}",
        "Storm Lon": f"{pt['lon']:.2f}",
        "Cone Radius": f"{pt['cone_radius_mi']:.0f} mi",
        "Local Wind": f"{f_env['wind_kts']:.0f} kt / {f_env['wind_mph']:.0f} mph",
        "Local Surge": f"{f_env['surge_ft']:.1f} ft",
        "Condition": condition_from_wind(f_env["wind_kts"], f_env["radius_mi"], r_max),
    })

st.dataframe(pd.DataFrame(mp_rows), hide_index=True, use_container_width=True)

st.divider()
st.subheader("🎯 Selected Zone Snapshot")

zlat, zlon = ZONES[selected_zone]["center"]
zenv = compute_local_environment(
    zlat, zlon, current_lat, current_lon, p, radar_coords, front_lat,
    pressure_drop_hpa=pressure_drop_hpa,
    dry_air=dry_air,
    urban_heat=urban_heat,
    ewr_phase=ewr_phase,
)

st.dataframe(pd.DataFrame([{
    "Zone": selected_zone,
    "Wind": f"{zenv['wind_kts']:.0f} kt / {zenv['wind_mph']:.0f} mph",
    "Gust": f"{zenv['gust_kts']:.0f} kt / {zenv['gust_mph']:.0f} mph",
    "Temp": f"{zenv['temp_f']:.0f}°F",
    "Dewpoint": f"{zenv['dewp_f']:.0f}°F",
    "Visibility": f"{zenv['visibility_mi']:.1f} mi",
    "Rain Dir": zenv['rain_dir_text'],
    "Surge": f"{zenv['surge_ft']:.1f} ft",
    "Tornado": f"{zenv['tornado_label']} ({zenv['tornado_risk']:.0f})",
}]), hide_index=True, use_container_width=True)

# -----------------------------
# 8. COUNTY-WIDE ZONE TABLE
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
    ewr_phase,
)

st.dataframe(summary_df, hide_index=True, use_container_width=True)
# -----------------------------
# 9. AUTOMATED LOOP ENGINE
# -----------------------------
if st.session_state.is_playing:
    st.session_state.loop_idx += 1
    if st.session_state.loop_idx > 12:
        st.session_state.loop_idx = -12
    time.sleep(0.1)
    st.rerun()
