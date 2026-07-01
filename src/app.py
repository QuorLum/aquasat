"""
AquaSat — Precision Irrigation Advisory Dashboard
Two-Speed Geospatial Intelligence Pipeline powered by NASA Prithvi-100M
"""
import os
import sys
import numpy as np
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import GeospatialConfig
from src.gee_pipeline import download_chunk_numpy, get_spatial_chunks
from src.gap_filling import reconstruct_cloud_gaps
from src.slow_track import run_slow_track, CROP_CLASSES, STAGE_CLASSES
from src.fast_track import run_fast_track
from src.weather_analytics import fetch_open_meteo_daily, analyze_regional_winds
from src.advisory import (
    generate_canal_command_zones,
    generate_irrigation_advisory,
    export_advisory_map,
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AquaSat — Irrigation Advisory",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, high-tech aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #21262d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #58a6ff;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #58a6ff !important;
        font-weight: 700;
    }

    /* Status banners */
    .status-banner {
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 12px;
        text-align: center;
    }
    .status-ok {
        background: linear-gradient(90deg, #0d4429, #1a5c38);
        color: #3fb950;
        border: 1px solid #238636;
    }
    .status-warn {
        background: linear-gradient(90deg, #4a2d0a, #5c3a12);
        color: #d29922;
        border: 1px solid #9e6a03;
    }
    .status-crit {
        background: linear-gradient(90deg, #4a0d0d, #5c1212);
        color: #f85149;
        border: 1px solid #da3633;
    }

    /* Map container */
    .map-container {
        border: 1px solid #30363d;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Legend chips */
    .legend-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ADVISORY_LABELS = {
    0: ("Fallow", "#3d3d3d"),
    1: ("Optimal", "#3fb950"),
    2: ("Light Irrigation", "#58a6ff"),
    3: ("Conserve Moisture", "#d29922"),
    4: ("Critical Irrigation", "#f0883e"),
    5: ("Severe Deficit", "#f85149"),
}

CROP_COLORS = {
    0: "#3d3d3d",  # Bare Soil
    1: "#2ea043",  # Rice
    2: "#dbab09",  # Wheat
    3: "#a371f7",  # Sugarcane
    4: "#f0883e",  # Maize
}


def _build_folium_map(grid: np.ndarray, coords: list, color_map: dict, title: str):
    """Render a numpy grid as a Folium choropleth overlay on a satellite basemap."""
    try:
        import folium
        from folium.raster_layers import ImageOverlay
    except ImportError:
        st.error("Folium is required for map rendering. Install via `pip install folium`.")
        return None

    min_lon, min_lat, max_lon, max_lat = coords
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )

    # Build RGBA image from grid
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for val, (_, hex_color) in color_map.items():
        mask = grid == val
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        rgba[mask] = [r, g, b, 180]

    # Convert to PNG bytes
    from io import BytesIO
    try:
        from PIL import Image
        img = Image.fromarray(rgba, "RGBA")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        import base64
        img_b64 = base64.b64encode(buf.read()).decode()
        img_url = f"data:image/png;base64,{img_b64}"
    except ImportError:
        st.warning("Pillow (PIL) is needed for overlay rendering.")
        return m

    bounds = [[min_lat, min_lon], [max_lat, max_lon]]
    ImageOverlay(image=img_url, bounds=bounds, opacity=0.65, name=title).add_to(m)
    folium.LayerControl().add_to(m)
    return m


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
for key in [
    "pipeline_ran", "crop_map", "stage_map", "stress_prob",
    "advisory_grid", "weather_data", "wind_analysis",
    "irrigation_mask", "chunk_coords", "saved_path",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🛰️ AquaSat")
    st.caption("Precision Irrigation Advisory System")
    st.divider()

    st.markdown("### Region of Interest")
    col_a, col_b = st.columns(2)
    with col_a:
        roi_min_lon = st.number_input("Min Lon", value=GeospatialConfig.ROI_COORDINATES[0], format="%.2f")
        roi_min_lat = st.number_input("Min Lat", value=GeospatialConfig.ROI_COORDINATES[1], format="%.2f")
    with col_b:
        roi_max_lon = st.number_input("Max Lon", value=GeospatialConfig.ROI_COORDINATES[2], format="%.2f")
        roi_max_lat = st.number_input("Max Lat", value=GeospatialConfig.ROI_COORDINATES[3], format="%.2f")

    st.divider()
    st.markdown("### Pipeline Controls")
    target_date = st.date_input("Forecast Date", value=None)
    use_fallback = st.checkbox("Use Conv3D Fallback Encoder", value=True, help="Skip Prithvi-100M download; use local Conv3D encoder.")

    st.divider()
    run_slow = st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary")
    run_fast_only = st.button("⚡ Run Fast Track Only", use_container_width=True, disabled=st.session_state.pipeline_ran is None)

    st.divider()
    st.markdown(
        "<div style='color:#8b949e;font-size:0.75rem;text-align:center;'>"
        "Powered by NASA/IBM Prithvi-100M<br>"
        "Google Earth Engine · Open-Meteo<br>"
        "Sentinel-2 + Sentinel-1 Fusion"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

def _run_full_pipeline(roi, date_str, fallback):
    """Execute the complete two-speed pipeline and cache results in session state."""
    chunks = get_spatial_chunks(list(roi), GeospatialConfig.CHUNK_SIZE_DEG)
    chunk = chunks[0]
    st.session_state.chunk_coords = chunk

    progress = st.progress(0, text="Downloading satellite imagery from GEE…")

    # 1. Download
    fused_raw = download_chunk_numpy(chunk)
    progress.progress(15, text="Cloud gap filling & radar-anchored interpolation…")

    # 2. Gap fill
    reconstructed = reconstruct_cloud_gaps(fused_raw)
    progress.progress(30, text="Running Slow Track crop classification…")

    # 3. Slow Track
    crop_map, stage_map, baseline_emb = run_slow_track(reconstructed, use_fallback=fallback)
    st.session_state.crop_map = crop_map
    st.session_state.stage_map = stage_map
    progress.progress(55, text="Fetching meteorological data…")

    # 4. Weather
    c_lon = (chunk[0] + chunk[2]) / 2.0
    c_lat = (chunk[1] + chunk[3]) / 2.0
    weather = fetch_open_meteo_daily(c_lat, c_lon, date_str)
    weather["date"] = date_str
    wind = analyze_regional_winds(c_lat, c_lon, date_str, radius_km=GeospatialConfig.WIND_RADIUS_KM)
    st.session_state.weather_data = weather
    st.session_state.wind_analysis = wind
    progress.progress(70, text="Running Fast Track daily forecast…")

    # 5. Fast Track
    h, w = GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE
    irr_mask = generate_canal_command_zones(h, w, buffer_pixels=30.0)
    soil_whc = np.ones((h, w), dtype=np.float32) * 0.15
    prev_dep = np.zeros((h, w), dtype=np.float32)

    stress, _, _ = run_fast_track(
        baseline_emb, weather, prev_dep, irr_mask, soil_whc, crop_map, stage_map, lat_deg=c_lat
    )
    st.session_state.stress_prob = stress
    st.session_state.irrigation_mask = irr_mask
    progress.progress(85, text="Generating irrigation advisories…")

    # 6. Advisory
    advisory = generate_irrigation_advisory(stress, crop_map, irr_mask)
    saved = export_advisory_map(advisory, "advisory_map.tif", roi_coords=chunk)
    st.session_state.advisory_grid = advisory
    st.session_state.saved_path = saved
    st.session_state.pipeline_ran = True
    progress.progress(100, text="Pipeline complete ✓")


def _run_fast_track_only(date_str):
    """Re-run only the fast track with a new date (uses cached slow-track embeddings)."""
    chunk = st.session_state.chunk_coords
    c_lon = (chunk[0] + chunk[2]) / 2.0
    c_lat = (chunk[1] + chunk[3]) / 2.0

    progress = st.progress(0, text="Fetching weather for new date…")
    weather = fetch_open_meteo_daily(c_lat, c_lon, date_str)
    weather["date"] = date_str
    wind = analyze_regional_winds(c_lat, c_lon, date_str, radius_km=GeospatialConfig.WIND_RADIUS_KM)
    st.session_state.weather_data = weather
    st.session_state.wind_analysis = wind
    progress.progress(40, text="Running Fast Track forecast…")

    h, w = GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE
    soil_whc = np.ones((h, w), dtype=np.float32) * 0.15
    prev_dep = np.zeros((h, w), dtype=np.float32)
    # Re-derive baseline embeddings via slow track cache (stored crop_map is still valid)
    # For fast-only re-runs we need baseline_emb; re-run slow track silently
    from src.slow_track import run_slow_track as _st
    reconstructed = reconstruct_cloud_gaps(download_chunk_numpy(chunk))
    _, _, baseline_emb = _st(reconstructed, use_fallback=True)

    stress, _, _ = run_fast_track(
        baseline_emb, weather, prev_dep, st.session_state.irrigation_mask,
        soil_whc, st.session_state.crop_map, st.session_state.stage_map, lat_deg=c_lat
    )
    st.session_state.stress_prob = stress
    progress.progress(80, text="Generating advisories…")

    advisory = generate_irrigation_advisory(stress, st.session_state.crop_map, st.session_state.irrigation_mask)
    saved = export_advisory_map(advisory, "advisory_map.tif", roi_coords=chunk)
    st.session_state.advisory_grid = advisory
    st.session_state.saved_path = saved
    progress.progress(100, text="Fast Track update complete ✓")


# Trigger pipeline runs
if run_slow:
    roi = (roi_min_lon, roi_min_lat, roi_max_lon, roi_max_lat)
    d = target_date.strftime("%Y-%m-%d") if target_date else "2025-08-15"
    _run_full_pipeline(roi, d, use_fallback)

if run_fast_only:
    d = target_date.strftime("%Y-%m-%d") if target_date else "2025-08-15"
    _run_fast_track_only(d)

# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------
st.markdown("# 🛰️ AquaSat — Precision Irrigation Advisory")
st.caption("Two-Speed Geospatial Intelligence · Bihar, India · Gandak Canal Command Area")

if st.session_state.pipeline_ran is None:
    st.info("Configure the Region of Interest in the sidebar and press **Run Full Pipeline** to begin.", icon="👈")
    st.stop()

# ---- Metrics Row ----
weather = st.session_state.weather_data
wind = st.session_state.wind_analysis

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🌡️ Temp Max", f"{weather['temp_max']:.1f} °C")
col2.metric("🌧️ Rainfall", f"{weather['rain']:.1f} mm")
col3.metric("💨 Wind Speed", f"{weather['wind_speed']:.1f} m/s")
col4.metric("🌀 Wind Dir", f"{weather['wind_dir']:.0f}°")
col5.metric("🔥 Evap Factor", f"{wind['evap_acceleration_factor']:.2f}×")

st.divider()

# ---- Advisory Status Banner ----
advisory = st.session_state.advisory_grid
unique, counts = np.unique(advisory, return_counts=True)
dist = dict(zip(unique.tolist(), counts.tolist()))
total_active = sum(v for k, v in dist.items() if k > 0)
critical_pct = sum(dist.get(k, 0) for k in [4, 5]) / max(total_active, 1) * 100
moderate_pct = sum(dist.get(k, 0) for k in [2, 3]) / max(total_active, 1) * 100

if critical_pct > 10:
    st.markdown('<div class="status-banner status-crit">⚠️ CRITICAL — Significant irrigation deficit detected across the command area</div>', unsafe_allow_html=True)
elif moderate_pct > 20:
    st.markdown('<div class="status-banner status-warn">⚡ MODERATE STRESS — Targeted irrigation recommended in tail-end zones</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-banner status-ok">✅ OPTIMAL — Crop moisture levels are within safe thresholds</div>', unsafe_allow_html=True)

# ---- Map Tabs ----
tab_advisory, tab_crop, tab_stress = st.tabs(["🗺️ Irrigation Advisory", "🌾 Crop Classification", "💧 Moisture Stress"])

chunk = st.session_state.chunk_coords

with tab_advisory:
    st.markdown("#### Pixel-Level Irrigation Advisory Map")
    fmap = _build_folium_map(advisory, chunk, ADVISORY_LABELS, "Irrigation Advisory")
    if fmap:
        from streamlit_folium import st_folium
        st_folium(fmap, width=None, height=520, use_container_width=True)

    # Legend
    legend_html = " ".join(
        f'<span class="legend-chip" style="background:{color};color:#fff;">{label}</span>'
        for code, (label, color) in ADVISORY_LABELS.items()
    )
    st.markdown(f"<div style='margin-top:8px'>{legend_html}</div>", unsafe_allow_html=True)

    # Download button
    if st.session_state.saved_path and os.path.exists(st.session_state.saved_path):
        with open(st.session_state.saved_path, "rb") as f:
            st.download_button("📥 Download Advisory GeoTIFF", f, file_name="advisory_map.tif", mime="image/tiff")

with tab_crop:
    st.markdown("#### Crop Type Classification (Slow Track)")
    crop_map = st.session_state.crop_map
    crop_labels = {k: (v, CROP_COLORS[k]) for k, v in CROP_CLASSES.items()}
    fmap_crop = _build_folium_map(crop_map, chunk, crop_labels, "Crop Classification")
    if fmap_crop:
        from streamlit_folium import st_folium
        st_folium(fmap_crop, width=None, height=520, use_container_width=True)

    legend_html = " ".join(
        f'<span class="legend-chip" style="background:{color};color:#fff;">{label}</span>'
        for code, (label, color) in crop_labels.items()
    )
    st.markdown(f"<div style='margin-top:8px'>{legend_html}</div>", unsafe_allow_html=True)

with tab_stress:
    st.markdown("#### Daily Moisture Stress Probability (Fast Track)")
    stress = st.session_state.stress_prob

    # Discretize into 5 bins for map coloring
    stress_discrete = np.digitize(stress, bins=[0.2, 0.4, 0.6, 0.8]) 
    stress_labels = {
        0: ("Low", "#3fb950"),
        1: ("Moderate", "#58a6ff"),
        2: ("Elevated", "#d29922"),
        3: ("High", "#f0883e"),
        4: ("Critical", "#f85149"),
    }
    fmap_stress = _build_folium_map(stress_discrete, chunk, stress_labels, "Moisture Stress")
    if fmap_stress:
        from streamlit_folium import st_folium
        st_folium(fmap_stress, width=None, height=520, use_container_width=True)

    legend_html = " ".join(
        f'<span class="legend-chip" style="background:{color};color:#fff;">{label}</span>'
        for code, (label, color) in stress_labels.items()
    )
    st.markdown(f"<div style='margin-top:8px'>{legend_html}</div>", unsafe_allow_html=True)

# ---- Advisory Distribution Table ----
st.divider()
st.markdown("#### 📊 Advisory Distribution")
dist_data = []
for code, (label, color) in ADVISORY_LABELS.items():
    count = dist.get(code, 0)
    pct = count / (advisory.shape[0] * advisory.shape[1]) * 100
    dist_data.append({"Advisory": label, "Pixels": count, "Coverage (%)": round(pct, 2)})

st.dataframe(dist_data, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:0.8rem;'>"
    "AquaSat v1.0 · Two-Speed Architecture · NASA/IBM Prithvi-100M · "
    "Sentinel-2 + Sentinel-1 Fusion · FAO-56 Penman-Monteith · Open-Meteo"
    "</div>",
    unsafe_allow_html=True,
)
