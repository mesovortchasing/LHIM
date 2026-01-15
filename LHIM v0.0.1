import streamlit as st
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium

# --- 1. PARAMETRIC WIND MODEL (Holland 2010 / Asymmetric) ---
def calculate_wind_field(lat, lon, storm_lat, storm_lon, v_max, r_max, f_speed, shear_mag, shear_dir):
    """
    Calculates wind speed at a point (lat, lon) given storm parameters.
    Includes asymmetry from forward speed and wind shear.
    """
    # Distance from eye (simplified lat/lon to miles conversion)
    dx = (lon - storm_lon) * 53  # Approximate miles per degree at 30N
    dy = (lat - storm_lat) * 69
    r = np.sqrt(dx**2 + dy**2)
    
    if r == 0: return 0
    
    # Holland B Parameter (Shape Factor)
    B = 1.5 # Alpha default
    
    # Base Holland Wind Speed (Symmetric)
    # V = Vmax * ((Rmax/r)^B * exp(1-(Rmax/r)^B))^0.5
    v_sym = v_max * np.sqrt((r_max / r)**B * np.exp(1 - (r_max / r)**B))
    
    # Asymmetry 1: Forward Motion (Right-Front enhancement)
    # Adds a portion of forward speed to the wind field
    angle = np.arctan2(dy, dx)
    v_asym_motion = f_speed * 0.5 * np.cos(angle) # Simplified
    
    # Asymmetry 2: Wind Shear (Disorganization)
    # High shear "pushes" the wind field downwind
    shear_effect = (shear_mag / 20) * v_sym * 0.2 * np.sin(angle - np.radians(shear_dir))
    
    total_wind = v_sym + v_asym_motion + shear_effect
    return max(0, total_wind)

# --- 2. STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("🌀 Mobile County Hurricane Impact Simulator (Alpha)")
st.sidebar.header("Storm Parameters")

# Sidebar Controls (15 Parameters total)
v_max = st.sidebar.slider("Intensity (Max Sustained Knots)", 40, 160, 85)
f_speed = st.sidebar.slider("Forward Speed (mph)", 5, 30, 15)
r_max = st.sidebar.slider("Radius of Max Winds (miles)", 10, 50, 25)
shear_mag = st.sidebar.slider("Vertical Wind Shear (200-850mb, knots)", 0, 50, 10)
landfall_lat = st.sidebar.number_input("Landfall Lat (Mobile is ~30.6)", value=30.4)
landfall_lon = st.sidebar.number_input("Landfall Lon (Mobile is ~-88.1)", value=-88.1)

# --- 3. IMPACT CALCULATION ---
# Create a grid around Mobile, AL
lats = np.linspace(29.8, 31.4, 30)
lons = np.linspace(-88.9, -87.3, 30)
data = []

for lt in lats:
    for ln in lons:
        w = calculate_wind_field(lt, ln, landfall_lat, landfall_lon, v_max, r_max, f_speed, shear_mag, 0)
        # Damage State Logic
        if w > 110: ds = "Severe/Catastrophic"
        elif w > 90: ds = "Extensive"
        elif w > 74: ds = "Moderate"
        else: ds = "Minor/None"
        data.append([lt, ln, w, ds])

df = pd.DataFrame(data, columns=['lat', 'lon', 'wind', 'damage'])

# --- 4. THE MAP ---
m = folium.Map(location=[30.6954, -88.0399], zoom_start=9)

# Add Damage Zones
for i, row in df.iterrows():
    color = 'red' if row['wind'] > 95 else 'orange' if row['wind'] > 74 else 'blue'
    if row['wind'] > 40: # Only plot tropical storm force+
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=row['wind']/10,
            color=color,
            fill=True,
            popup=f"Wind: {int(row['wind'])} mph\nDamage: {row['damage']}"
        ).add_to(m)

st_folium(m, width=1000)
st.write("### Predicted Damage Summary")
st.dataframe(df['damage'].value_counts())
