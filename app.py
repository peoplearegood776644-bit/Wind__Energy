import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ==========================================
# 0. CORE PLATFORM ARCHITECTURE & STYLING
# ==========================================
st.set_page_config(
    page_title="NexGen Solar Intelligence Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Injected Modern Glassmorphism Styling
st.markdown("""
    <style>
        /* Main Workspace Container Adjustments */
        .reportview-container .main .block-container { max-width: 96%; padding-top: 1.5rem; }
        
        /* Glassmorphism Metric Cards */
        div[data-testid="stMetric"] {
            background: rgba(30, 34, 45, 0.65);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        }
        
        /* Modernized Tabs Styling */
        button[data-baseweb="tab"] {
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
        }
        
        /* Card Containers */
        .glass-panel {
            background: rgba(21, 25, 34, 0.7);
            border-radius: 12px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.04);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# System Navigation Sidebar Control
st.sidebar.markdown("<h1 style='text-align: center; color: #00CC96; margin-bottom: 0;'>⚡ NexGen Solar</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; font-size: 0.85em; opacity: 0.6; margin-top: 0;'>IoT Intelligence & Analytics Suite</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

selected_module = st.sidebar.radio(
    "Select Workspace Module",
    [
        "📊 Module 1: Advanced Efficiency Lab",
        "🤖 Module 2: ML Training Sandbox",
        "💵 Module 3: Capital Savings Matrix",
        "🌐 Module 4: Spatial Adoption Trends"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("🤖 Powered by Scikit-Learn, Plotly, & Open-Meteo APIs.")
st.sidebar.caption("📅 Current Operational Horizon: 2026")

# ==========================================
# DATA & MACHINE LEARNING PIPELINE ENGINES
# ==========================================

@st.cache_data(ttl=900)
def fetch_geocoded_weather_data(city_name: str):
    """
    Leverages a 2-stage dynamic open-access routing pipeline.
    First geocodes the target city string to lat/long coordinates, 
    then fetches real-time meteorological metrics from Open-Meteo.
    """
    clean_city = city_name.strip()
    fallback_metrics = {"temp": 25.0, "cloud_cover": 15, "humidity": 45, "source": "Internal Fallback Baseline"}
    
    if not clean_city:
        return fallback_metrics
        
    try:
        # Step 1: Geocoding Coordinate Extraction
        geo_url = f"https://open-meteo.com{clean_city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5)
        if geo_res.status_code != 200 or "results" not in geo_res.json():
            return fallback_metrics
            
        geo_data = geo_res.json()["results"][0]
        lat, lon = geo_data["latitude"], geo_data["longitude"]
        resolved_name = f"{geo_data.get('name')}, {geo_data.get('country')}"
        
        # Step 2: Meteorological Telemetry Retrieval
        weather_url = f"https://open-meteo.com{lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,cloud_cover"
        w_res = requests.get(weather_url, timeout=5)
        if w_res.status_code == 200:
            current_data = w_res.json()["current"]
            return {
                "temp": current_data.get("temperature_2m", 25.0),
                "cloud_cover": current_data.get("cloud_cover", 15),
                "humidity": current_data.get("relative_humidity_2m", 45),
                "source": f"Live Telemetry Node ({resolved_name})"
            }
    except Exception:
        pass
    return fallback_metrics

@st.cache_resource
def generate_and_cache_historical_logs():
    """
    Generates an internal high-dimensional synthetic asset logging matrix
    representing real-world solar plant performance conditions.
    """
    np.random.seed(101)
    n_samples = 2500
    
    irradiance = np.random.uniform(150, 1250, n_samples)
    ambient_t = np.random.uniform(-5, 48, n_samples)
    age = np.random.uniform(0, 25, n_samples)
    soiling = np.random.uniform(0.65, 1.0, n_samples) # Soiling index cleanliness metric
    
    # Physics-derived operational capacity algorithm
    cell_t = ambient_t + (irradiance * 0.03)
    thermal_loss = np.where(cell_t > 25.0, 1.0 - 0.0038 * (cell_t - 25.0), 1.0)
    thermal_loss = np.clip(thermal_loss, 0.1, 1.0)
    age_loss = (1.0 - 0.005) ** age
    
    # Target value: Power Output (kW) for a standard 10kW array configuration
    base_capacity_kw = 10.0
    power_output = (irradiance / 1000.0) * base_capacity_kw * thermal_loss * age_loss * soiling
    power_output = np.clip(power_output, 0, None) + np.random.normal(0, 0.15, n_samples)
    power_output = np.clip(power_output, 0, None)
    
    df = pd.DataFrame({
        "Irradiance": irradiance,
        "Cell_Temperature": cell_t,
        "Array_Age": age,
        "Soiling_Index": soiling,
        "Power_Output_kW": power_output
    })
    return df

# Initialize shared assets background matrix
historical_logs_df = generate_and_cache_historical_logs()

# ==========================================
# MODULE 1: 📊 ADVANCED EFFICIENCY LAB
# ==========================================
if selected_module == "📊 Module 1: Advanced Efficiency Lab":
    st.title("📊 Advanced Photovoltaic Efficiency Laboratory")
    st.markdown("Perform mathematical multi-variable de-rating calculations and evaluate physical cell responses under localized meteorological telemetry loads.")
    
    # Location Search Bar Container
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    target_city = st.text_input("📡 Query Real-Time Meteorological Station Node (Enter City Name):", "London")
    telemetry = fetch_geocoded_weather_data(target_city)
    st.caption(f"**Data Pipeline Source Boundary:** {telemetry['source']}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Derived Irradiance proxy modeling based on live cloud layer attenuation vectors
    derived_irradiance_w_m2 = max(200.0, 1000.0 - (telemetry["cloud_cover"] * 7.0))
    
    col_input, col_metrics = st.columns(, gap="large")
    
    with col_input:
        st.subheader("⚙️ Technical Stressors")
        override_toggle = st.toggle("Override Live Telemetry Feed with Manual Control Sliders")
        
        if override_toggle:
            ambient_temp = st.slider("Ambient Temperature Override (°C)", -10, 50, 25)
            irradiance = st.slider("Solar Plane Irradiance Override (W/m²)", 200, 1200, 800)
        else:
            ambient_temp = st.slider("Ambient Temperature Feed (°C)", -10, 50, int(telemetry["temp"]), disabled=True)
            irradiance = st.slider("Solar Plane Irradiance Proxy (W/m²)", 200, 1200, int(derived_irradiance_w_m2), disabled=True)
            
        system_age_yrs = st.slider("Asset Active Operating Age (Years)", 0, 25, 5)
        cell_architecture = st.selectbox(
            "Photovoltaic Silicon Material Selection Architecture",
            ["Monocrystalline Silicon", "Polycrystalline Silicon"]
        )
        
    # Core Mathematical Formulations Architecture
    specs = {
        "Monocrystalline Silicon": {"base_eff": 0.22, "temp_coeff": -0.0035},
        "Polycrystalline Silicon": {"base_eff": 0.17, "temp_coeff": -0.0040}
    }
    
    selected_spec = specs[cell_architecture]
    base_eta = selected_spec["base_eff"]
    t_coefficient = selected_spec["temp_coeff"]
    panel_area_m2 = 1.7  # Standard utility footprint module parameter
    
    # 1. Panel Cell Thermal Formula Calculation
    calculated_cell_temp = ambient_temp + (irradiance * 0.03)
    
    # 2. Thermal Deficiency Factor Loss Evaluation
    if calculated_cell_temp > 25.0:
        thermal_loss_factor = 1.0 + (t_coefficient * (calculated_cell_temp - 25.0))
        thermal_loss_factor = max(0.05, thermal_loss_factor)
    else:
        thermal_loss_factor = 1.0
        
    # 3. Compounding Long-Term Wear Degradation Equation
    age_loss_factor = (1.0 - 0.005) ** system_age_yrs
    
    # Final Operational Outputs
    effective_efficiency = base_eta * thermal_loss_factor * age_loss_factor
    instant_power_output_watts = irradiance * panel_area_m2 * effective_efficiency
    ideal_stc_output_watts = irradiance * panel_area_m2 * base_eta
    absolute_thermal_age_loss_watts = max(0.0, ideal_stc_output_watts - instant_power_output_watts)
    
    with col_metrics:
        st.subheader("📈 Real-Time Photovoltaic Analytics")
        
        m1, m2 = st.columns(2)
        m1.metric("Effective Asset Efficiency", f"{effective_efficiency * 100:.2f}%", f"Nominal Base: {base_eta*100:.0f}%", delta_color="off")
        m2.metric("Calculated Enclosure Cell Temp", f"{calculated_cell_temp:.1f} °C", f"Delta: {calculated_cell_temp - ambient_temp:+.1f}°C")
        
        m3, m4 = st.columns(2)
        m3.metric("Instant Modulus Power Yield", f"{instant_power_output_watts:.1f} W", f"Ideal STC Peak: {1000*panel_area_m2*base_eta:.0f}W", delta_color="off")
        m4.metric("Atmospheric & Aging Asset Losses", f"{absolute_thermal_age_loss_watts:.1f} W")
        
    st.markdown("---")
    st.subheader("🗺️ 3D Equivalent Photovoltaic System Response Surface Model")
