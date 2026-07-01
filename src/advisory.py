import os
import numpy as np
from src.config import GeospatialConfig

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def create_canal_distance_map(height: int, width: int) -> np.ndarray:
    """
    Simulates a canal network across the grid and computes the distance (in pixels)
    of each pixel to the nearest canal.
    Represented as a primary canal cutting diagonally and lateral channels horizontally.
    """
    y_indices, x_indices = np.indices((height, width))
    
    # Distance to diagonal primary canal: y = x
    dist_primary = np.abs(y_indices - x_indices) / np.sqrt(2)
    
    # Distance to horizontal lateral canals at rows H/4 and 3H/4
    dist_lateral_1 = np.abs(y_indices - height // 4)
    dist_lateral_2 = np.abs(y_indices - 3 * height // 4)
    
    # Take minimum distance to any canal
    min_dist = np.minimum(dist_primary, np.minimum(dist_lateral_1, dist_lateral_2))
    return min_dist


def generate_canal_command_zones(height: int, width: int, buffer_pixels: float = 25.0) -> np.ndarray:
    """
    Classifies the study area into:
      - Irrigated Zone (1.0): Close to canals (within buffer_pixels)
      - Tail-End Rainfed Zone (0.0): Far from canals (outside buffer_pixels)
    """
    dist_map = create_canal_distance_map(height, width)
    irrigation_mask = np.where(dist_map <= buffer_pixels, 1.0, 0.0)
    return irrigation_mask


def generate_irrigation_advisory(
    moisture_stress_prob: np.ndarray,
    crop_map: np.ndarray,
    irrigation_mask: np.ndarray
) -> np.ndarray:
    """
    Generates pixel-level irrigation advisory codes:
      - 0: Bare Soil / Fallow (No Action)
      - 1: Healthy Crop - No Stress (Optimal)
      - 2: Moderate Stress - Irrigated Zone (Irrigation Recommended)
      - 3: Moderate Stress - Tail-End Rainfed (Conserve Moisture, Rainfed Warning)
      - 4: Severe Stress - Irrigated Zone (CRITICAL: Immediate Irrigation Required)
      - 5: Severe Stress - Tail-End Rainfed (CRITICAL: Severe Deficit Warning)
    """
    height, width = moisture_stress_prob.shape
    advisory_map = np.zeros((height, width), dtype=np.uint8)
    
    # Crop mask: active when crop is not bare soil (class 0)
    active_crop = crop_map > 0
    
    # Conditions
    no_stress = moisture_stress_prob < 0.35
    mod_stress = (moisture_stress_prob >= 0.35) & (moisture_stress_prob < 0.65)
    sev_stress = moisture_stress_prob >= 0.65
    
    irrigated = irrigation_mask > 0.5
    rainfed = ~irrigated
    
    # Map codes
    advisory_map[active_crop & no_stress] = 1
    advisory_map[active_crop & mod_stress & irrigated] = 2
    advisory_map[active_crop & mod_stress & rainfed] = 3
    advisory_map[active_crop & sev_stress & irrigated] = 4
    advisory_map[active_crop & sev_stress & rainfed] = 5
    
    return advisory_map


def export_advisory_map(
    advisory_grid: np.ndarray, 
    filename: str, 
    roi_coords: list = None
) -> str:
    """
    Exports the advisory map as a GeoTIFF.
    Falls back to saving as a binary NumPy file and a CSV if rasterio is not installed.
    """
    GeospatialConfig.setup_directories()
    out_path = os.path.join(GeospatialConfig.OUTPUT_DIR, filename)
    
    h, w = advisory_grid.shape
    coords = roi_coords if roi_coords is not None else GeospatialConfig.ROI_COORDINATES
    
    if HAS_RASTERIO:
        # Calculate geospatial affine transform from ROI coordinates
        # coords: [min_lon, min_lat, max_lon, max_lat]
        min_lon, min_lat, max_lon, max_lat = coords
        res_lon = (max_lon - min_lon) / w
        res_lat = (max_lat - min_lat) / h
        
        # Affine transform: west_lon, pixel_width, 0, north_lat, 0, pixel_height (negative)
        transform = from_origin(min_lon, max_lat, res_lon, res_lat)
        
        try:
            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=h,
                width=w,
                count=1,
                dtype=advisory_grid.dtype,
                crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                transform=transform,
                nodata=0
            ) as dst:
                dst.write(advisory_grid, 1)
                
            print(f"[Advisory Exporter] Successfully exported GeoTIFF map to {out_path}")
            return out_path
        except Exception as e:
            print(f"[Advisory Exporter] GeoTIFF write failed: {e}. Falling back to NumPy format.")
            
    # Fallback to NumPy save format
    npy_path = out_path.replace(".tif", ".npy")
    np.save(npy_path, advisory_grid)
    print(f"[Advisory Exporter] Exported binary map (no georeferencing) to {npy_path}")
    return npy_path
