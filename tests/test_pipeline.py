import os
import sys
import numpy as np

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import GeospatialConfig
from src.gee_pipeline import download_chunk_numpy, get_spatial_chunks
from src.gap_filling import reconstruct_cloud_gaps
from src.slow_track import run_slow_track, CROP_CLASSES, STAGE_CLASSES
from src.fast_track import run_fast_track
from src.advisory import generate_canal_command_zones, generate_irrigation_advisory, export_advisory_map


def render_ascii_map(grid: np.ndarray, title: str, mapping_dict: dict = None, default_char: str = "."):
    """
    Downsamples a 2D array and renders it as an ASCII character map in the console.
    """
    h, w = grid.shape
    # Target size for console rendering (40 columns, 16 rows)
    target_cols = 40
    target_rows = 16
    
    row_step = max(1, h // target_rows)
    col_step = max(1, w // target_cols)
    
    print("\n" + "=" * 50)
    print(f" DIAGNOSTIC MAP: {title.upper()}")
    print("=" * 50)
    
    for r in range(0, h, row_step):
        row_str = ""
        for c in range(0, w, col_step):
            val = grid[r, c]
            
            if mapping_dict is not None:
                char = mapping_dict.get(int(val), default_char)
            else:
                # Continuous value mapping (e.g. stress probability from 0.0 to 1.0)
                if val < 0.2:
                    char = "."
                elif val < 0.4:
                    char = "-"
                elif val < 0.6:
                    char = "+"
                elif val < 0.8:
                    char = "*"
                else:
                    char = "#"
            row_str += char
        print(row_str)
    print("=" * 50)


def test_pipeline_e2e():
    """
    End-to-End integration test simulating the entire crop monitoring and advisory workflow.
    """
    print("[E2E Test] Starting Geospatial Architecture End-to-End Test...")
    
    # 1. Setup outputs
    GeospatialConfig.setup_directories()
    
    # Get test spatial chunk coordinates (e.g. first chunk in ROI)
    chunks = get_spatial_chunks(GeospatialConfig.ROI_COORDINATES, GeospatialConfig.CHUNK_SIZE_DEG)
    test_chunk = chunks[0]
    print(f"[E2E Test] Sampling ROI Chunk coordinates: {test_chunk}")
    
    # 2. Ingest and Fuse Virtual Constellation
    # In local testing, this yields high-fidelity simulated crop growth stacks automatically
    fused_raw = download_chunk_numpy(test_chunk)
    
    # 3. Gap Filling & Cloud Mitigation (Monsoon Reconstruction)
    print("[E2E Test] Running cloud gap filling and radar-anchored interpolation...")
    reconstructed_optical = reconstruct_cloud_gaps(fused_raw)
    
    # Verify shape: must be (6, Timesteps, H=224, W=224)
    assert reconstructed_optical.shape == (6, fused_raw.shape[1], GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE), \
        f"Incorrect reconstructed shape: {reconstructed_optical.shape}"
    print("[E2E Test] Gap filling completed. Shape verified.")
    
    # 4. Slow Track: Crop Mapping and Phenology Stage Classification (Runs every 10-15 days)
    # Using Conv3D fallback encoder for testing to run locally without internet or GPU limits
    print("[E2E Test] Running Slow Track crop classification and stage tracking (Conv3D Fallback)...")
    crop_map, stage_map, baseline_emb = run_slow_track(reconstructed_optical, use_fallback=True)
    
    # Verify classifications
    assert crop_map.shape == (GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE)
    assert stage_map.shape == (GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE)
    assert baseline_emb.shape == (768, GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE)
    print("[E2E Test] Slow Track inference validated.")
    
    # Render Slow Track Crop map
    crop_chars = {0: ".", 1: "R", 2: "W", 3: "S", 4: "M"}
    render_ascii_map(crop_map, "Crop Type Map (.=Bare, R=Rice, W=Wheat, S=Sugarcane, M=Maize)", crop_chars)
    
    # 5. Meteorological & Environmental Context Ingestion
    print("[E2E Test] Loading Meteorological daily data from Open-Meteo...")
    from src.weather_analytics import fetch_open_meteo_daily, analyze_regional_winds
    
    # Central coordinate of test chunk
    c_lon = (test_chunk[0] + test_chunk[2]) / 2.0
    c_lat = (test_chunk[1] + test_chunk[3]) / 2.0
    
    target_date = "2025-08-15"  # Mid-monsoon test date
    weather_data = fetch_open_meteo_daily(c_lat, c_lon, target_date)
    weather_data["date"] = target_date
    
    # Broad regional wind tracking within 350 KM radius
    wind_analysis = analyze_regional_winds(c_lat, c_lon, target_date, radius_km=GeospatialConfig.WIND_RADIUS_KM)
    print(f"[E2E Test] Weather Fetched (Status: {weather_data['status']}). Temp Max: {weather_data['temp_max']:.1f}C, Rain: {weather_data['rain']:.1f}mm.")
    print(f"[E2E Test] Regional Wind (350km): Speed {wind_analysis['avg_speed_m_s']:.2f} m/s, Evaporation Acceleration: {wind_analysis['evap_acceleration_factor']:.2f}x.")
    
    # 6. Fast Track: Daily Predictive Moisture Stress & Deficits
    print("[E2E Test] Initiating Fast Track daily forecasting loops...")
    
    # Initialize daily tracking state matrices (depletion levels)
    h, w = GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE
    prev_depletion = np.zeros((h, w), dtype=np.float32)
    
    # Generate spatial soil water holding capacity grid (loamy soil = 0.15 average WHC)
    soil_whc = np.ones((h, w), dtype=np.float32) * 0.15
    
    # Generate canal boundary command zones
    irrigation_mask = generate_canal_command_zones(h, w, buffer_pixels=30.0)
    render_ascii_map(irrigation_mask, "Canal Command Zones (I=Irrigated, .=Rainfed)", {1: "I", 0: "."})
    
    # Run Fast Track Prediction
    stress_prob, next_depletion, ndwi = run_fast_track(
        baseline_emb,
        weather_data,
        prev_depletion,
        irrigation_mask,
        soil_whc,
        crop_map,
        stage_map,
        lat_deg=c_lat
    )
    
    # Verify fast track outputs
    assert stress_prob.shape == (h, w)
    assert ndwi.shape == (h, w)
    print("[E2E Test] Fast Track execution validated.")
    
    # Render Moisture Stress probability map
    render_ascii_map(stress_prob, "Daily Crop Moisture Stress Probability (.=Low, #=High)")
    
    # 7. Generate Pixel-Level Irrigation Advisories & GeoTIFF Exporter
    print("[E2E Test] Generating pixel-level irrigation advisories...")
    advisory_grid = generate_irrigation_advisory(stress_prob, crop_map, irrigation_mask)
    
    # Save advisory map
    out_file = "test_advisory_map.tif"
    saved_path = export_advisory_map(advisory_grid, out_file, roi_coords=test_chunk)
    
    # Verify file creation
    assert os.path.exists(saved_path), f"Output advisory file not found: {saved_path}"
    print(f"[E2E Test] Advisory map exported successfully: {saved_path}")
    
    # Render final irrigation advisory codes
    advisory_chars = {
        0: ".",  # Fallow
        1: "O",  # Optimal (No stress)
        2: "i",  # Light Irrigation (Irrigated zone)
        3: "r",  # Conserve moisture (Rainfed zone)
        4: "I",  # Critical irrigation (Irrigated zone)
        5: "R"   # Severe deficit warning (Rainfed zone)
    }
    render_ascii_map(advisory_grid, "Irrigation Advisories (O=Opt, i=LightIrr, r=Conserve, I=CRITICAL, R=SEVERE WARNING)", advisory_chars)
    
    print("\n[E2E Test] ALL INTEGRATION TESTS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    test_pipeline_e2e()
