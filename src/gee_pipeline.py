import os
import io
import time
import zipfile
import numpy as np
import requests
from src.config import GeospatialConfig

# Global flag to track if GEE is initialized and authenticated
GEE_INITIALIZED = False
GEE_MOCKED = False

try:
    import ee
    # Attempt to initialize
    try:
        ee.Initialize(project="kaggle-project-499515")
        GEE_INITIALIZED = True
        print("[GEE API] Earth Engine initialized successfully.")
    except Exception as e:
        print(f"[GEE API] Earth Engine Initialization failed: {e}")
        print("[GEE API] Switching to Local Simulation Mode for development.")
        GEE_MOCKED = True
except ImportError:
    print("[GEE API] ee module not found. Run 'pip install earthengine-api' to install.")
    print("[GEE API] Switching to Local Simulation Mode for development.")
    GEE_MOCKED = True


def initialize_gee() -> bool:
    """
    Ensures Google Earth Engine is initialized. Returns True if successful, False if mocked.
    """
    global GEE_INITIALIZED, GEE_MOCKED
    if GEE_INITIALIZED:
        return True
    if GEE_MOCKED:
        return False
        
    try:
        import ee
        ee.Initialize(project="kaggle-project-499515")
        GEE_INITIALIZED = True
        return True
    except Exception as e:

        print(f"[GEE API] GEE Init failed, falling back to simulated data. Error: {e}")
        GEE_MOCKED = True
        return False


def get_spatial_chunks(roi_coords: list, chunk_size_deg: float) -> list:
    """
    Splits the overall ROI bounding box into smaller spatial tiles/chunks.
    roi_coords: [min_lon, min_lat, max_lon, max_lat]
    Returns a list of chunks, each formatted as [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = roi_coords
    chunks = []
    
    # Calculate grids
    lon_steps = np.arange(min_lon, max_lon, chunk_size_deg)
    lat_steps = np.arange(min_lat, max_lat, chunk_size_deg)
    
    for lon in lon_steps:
        for lat in lat_steps:
            c_min_lon = lon
            c_max_lon = min(lon + chunk_size_deg, max_lon)
            c_min_lat = lat
            c_max_lat = min(lat + chunk_size_deg, max_lat)
            chunks.append([c_min_lon, c_min_lat, c_max_lon, c_max_lat])
            
    return chunks


def fetch_sentinel2_collection(geometry, start_date: str, end_date: str):
    """
    Fetches QA-cloud-masked Sentinel-2 Surface Reflectance collection in GEE.
    """
    import ee
    
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        # SCL band is also useful for cloud shadowing and aerosol masking
        scl = image.select('SCL')
        # SCL codes: 3 = cloud shadow, 8 = cloud medium prob, 9 = cloud high prob, 10 = cirrus
        cloud_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0) \
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0)) \
            .And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
            
        # Add a custom 1-band mask where 1 is clear and 0 is cloudy
        clear_mask = cloud_mask.rename('clear_mask')
        
        # Divide by 10000 to convert to reflectance values (0..1)
        optical_bands = image.select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']).divide(10000.0)
        
        return optical_bands.addBands(clear_mask).updateMask(cloud_mask)

    s2 = ee.ImageCollection(GeospatialConfig.S2_COLLECTION) \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)) \
        .map(mask_s2_clouds)

        
    return s2


def fetch_radar_collection(geometry, start_date: str, end_date: str):
    """
    Fetches Sentinel-1 GRD SAR data in GEE (also serves as a proxy for EOS-04).
    """
    import ee
    s1 = ee.ImageCollection(GeospatialConfig.S1_COLLECTION) \
        .filterBounds(geometry) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .select(['VV', 'VH'])
        
    return s1


def bin_collection(collection, start_date_str: str, end_date_str: str, interval_days: int, band_names: list):
    """
    Bins the image collection into N-day intervals and computes the median for each bin.
    """
    import ee
    start_date = ee.Date(start_date_str)
    end_date = ee.Date(end_date_str)
    
    total_days = end_date.difference(start_date, 'days')
    num_bins = total_days.divide(interval_days).ceil()
    
    def process_bin(index):
        index = ee.Number(index)
        bin_start = start_date.advance(index.multiply(interval_days), 'days')
        bin_end = bin_start.advance(interval_days, 'days')
        
        bin_images = collection.filterDate(bin_start, bin_end)
        
        # Calculate median. If no image exists, return a default zero image with mask
        median_img = bin_images.median()
        
        # Set up a constant image in case of empty bins
        default_bands = [ee.Image.constant(0.0).rename(name) for name in band_names]
        default_img = ee.Image.cat(default_bands)
        
        # If median image has no bands (empty collection), use default
        final_img = ee.Algorithms.If(
            bin_images.size().gt(0),
            median_img.unmask(0.0),
            default_img
        )
        
        # Cast to ee.Image and set temporal metadata
        return ee.Image(final_img).set('system:time_start', bin_start.millis()).set('bin_index', index)
        
    bin_list = ee.List.sequence(0, num_bins.subtract(1)).map(process_bin)
    return ee.ImageCollection(bin_list)


def download_chunk_numpy(chunk_coords: list, simulated_shape: tuple = None) -> np.ndarray:
    """
    Downloads fused optical and radar imagery for a single spatial chunk.
    If GEE is mocked or offline, generates synthetic data in the specified shape.
    Returns: numpy array of shape (Channels, Timesteps, Height, Width)
    Channels order: 0: Blue, 1: Green, 2: Red, 3: Narrow_NIR, 4: SWIR1, 5: SWIR2, 6: clear_mask, 7: VV, 8: VH
    """
    global GEE_MOCKED
    
    # Calculate timesteps
    from datetime import datetime
    d1 = datetime.strptime(GeospatialConfig.START_DATE, "%Y-%m-%d")
    d2 = datetime.strptime(GeospatialConfig.END_DATE, "%Y-%m-%d")
    total_days = (d2 - d1).days
    timesteps = int(np.ceil(total_days / GeospatialConfig.TEMPORAL_WINDOW_DAYS))
    
    # Target grid resolution size (H=224, W=224)
    h, w = GeospatialConfig.TILE_SIZE, GeospatialConfig.TILE_SIZE
    num_channels = len(GeospatialConfig.PRITHVI_BANDS) + 1 + len(GeospatialConfig.SAR_BANDS) # 6 S2 + 1 Mask + 2 SAR = 9 channels
    
    if GEE_MOCKED or not initialize_gee():
        # Return synthetic dataset for testing
        print(f"[GEE API] Generating Mock Fused Data for chunk {chunk_coords} (shape: {num_channels}x{timesteps}x{h}x{w})")
        
        # Seed for reproducible mock data
        np.random.seed(int(sum(chunk_coords) * 100) % 123456)
        
        data = np.zeros((num_channels, timesteps, h, w), dtype=np.float32)
        
        # 1. Generate Optical Bands: Blue, Green, Red, Narrow_NIR, SWIR1, SWIR2 (values 0.0 - 1.0)
        # Add basic spatial and temporal patterns (crop growth simulation)
        x = np.linspace(-2.0, 2.0, w)
        y = np.linspace(-2.0, 2.0, h)
        xx, yy = np.meshgrid(x, y)
        base_pattern = np.sin(xx**2 + yy**2) * 0.2 + 0.3
        
        for t in range(timesteps):
            # Simulate growth curve (NDVI increases, SWIR changes)
            growth_factor = 0.3 + 0.4 * np.sin(np.pi * t / timesteps)
            
            # Blue, Green, Red
            data[0, t] = np.clip(base_pattern * 0.3 * (1.0 - growth_factor * 0.2) + np.random.normal(0, 0.01, (h, w)), 0, 1)
            data[1, t] = np.clip(base_pattern * 0.4 * (1.0 + growth_factor * 0.1) + np.random.normal(0, 0.01, (h, w)), 0, 1)
            data[2, t] = np.clip(base_pattern * 0.3 * (1.0 - growth_factor * 0.3) + np.random.normal(0, 0.01, (h, w)), 0, 1)
            
            # Narrow NIR (increases significantly during vegetative peak)
            data[3, t] = np.clip(base_pattern * 0.5 * (1.0 + growth_factor * 1.5) + np.random.normal(0, 0.02, (h, w)), 0, 1)
            
            # SWIR1, SWIR2 (sensitive to moisture)
            data[4, t] = np.clip(base_pattern * 0.4 * (1.0 - growth_factor * 0.4) + np.random.normal(0, 0.02, (h, w)), 0, 1)
            data[5, t] = np.clip(base_pattern * 0.3 * (1.0 - growth_factor * 0.5) + np.random.normal(0, 0.02, (h, w)), 0, 1)
            
            # clear_mask (1 = clear, 0 = cloudy). Simulate a heavy cloud cover in early-mid steps (Monsoon)
            is_cloudy = (t in [1, 2, 3]) and (np.random.rand() > 0.4)
            if is_cloudy:
                # Create spatial cloud blob
                cloud_blob = (np.sin(xx) + np.cos(yy) > 0.0).astype(np.float32)
                data[6, t] = cloud_blob
                # Zero out cloudy spectral bands
                for b in range(6):
                    data[b, t] *= cloud_blob
            else:
                data[6, t] = np.ones((h, w), dtype=np.float32)
                
            # 2. Generate SAR Radar Bands: VV and VH (values typically -25 to -5 dB)
            # Radar penetrates clouds and tracks soil/roughness
            data[7, t] = -12.0 + growth_factor * 4.0 + np.random.normal(0, 0.5, (h, w)) # VV
            data[8, t] = -18.0 + growth_factor * 5.0 + np.random.normal(0, 0.5, (h, w)) # VH
            
        return data

    # GEE Active Path
    import ee
    import rasterio
    from datetime import datetime, timedelta

    
    # Pre-calculate interval dates in Python
    d1_dt = datetime.strptime(GeospatialConfig.START_DATE, "%Y-%m-%d")
    d2_dt = datetime.strptime(GeospatialConfig.END_DATE, "%Y-%m-%d")
    
    intervals = []
    current_date = d1_dt
    while current_date < d2_dt:
        next_date = min(current_date + timedelta(days=GeospatialConfig.TEMPORAL_WINDOW_DAYS), d2_dt)
        intervals.append((current_date.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")))
        current_date = next_date
        
    try:
        # Define chunk geometry
        geom = ee.Geometry.Rectangle(chunk_coords)
        
        # Initialize numpy block
        # Shape: (Channels=9, Timesteps, H=224, W=224)
        chunk_data = np.zeros((num_channels, timesteps, h, w), dtype=np.float32)
        
        for t, (start, end) in enumerate(intervals):
            # Fetch optical and radar collections for this specific interval
            s2_coll = fetch_sentinel2_collection(geom, start, end)
            s1_coll = fetch_radar_collection(geom, start, end)
            
            # Check sizes to prevent empty downloads
            s2_size = int(s2_coll.size().getInfo())
            s1_size = int(s1_coll.size().getInfo())
            
            if s2_size == 0 and s1_size == 0:
                print(f"[GEE API] Timestep {t} ({start} to {end}) is empty. Using dummy zeros.")
                continue
                
            # Compute median for this slice
            if s2_size > 0:
                s2_img = s2_coll.median().unmask(0.0)
            else:
                s2_img = ee.Image.cat([ee.Image.constant(0.0).rename(b) for b in ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12', 'clear_mask']])
                
            if s1_size > 0:
                s1_img = s1_coll.median().unmask(0.0)
            else:
                s1_img = ee.Image.cat([ee.Image.constant(0.0).rename(b) for b in ['VV', 'VH']])
                
            # Stack and clip
            fused_img = s2_img.addBands(s1_img).clip(geom)
            
            # Download URL for this specific timestep image
            url = fused_img.getDownloadURL({
                'scale': GeospatialConfig.MODEL_RESOLUTION,
                'crs': 'EPSG:4326',
                'region': geom.getInfo(),
                'format': 'GEO_TIFF'
            })
            
            # Download and read GeoTIFF via rasterio
            resp = requests.get(url)
            if resp.status_code == 200:
                with rasterio.open(io.BytesIO(resp.content)) as src:
                    img_data = src.read()  # Shape: (9, H, W)
                    
                    th, tw = img_data.shape[1], img_data.shape[2]
                    if th != h or tw != w:
                        from scipy.ndimage import zoom
                        zoom_y = h / th
                        zoom_x = w / tw
                        for b in range(num_channels):
                            chunk_data[b, t] = zoom(img_data[b], (zoom_y, zoom_x), order=1)
                    else:
                        chunk_data[:, t] = img_data
                print(f"[GEE API] Timestep {t} ({start} to {end}) downloaded successfully.")
            else:
                print(f"[GEE API] Warning: Download failed for timestep {t} ({start} to {end}) with status {resp.status_code}. Response: {resp.text[:300]}")
                print("[GEE API] Using dummy zeros for this timestep.")
                
        return chunk_data
        
    except Exception as e:
        print(f"[GEE API] Error during active pipeline download: {e}")
        print("[GEE API] Falling back to synthetic chunk data generation.")
        GEE_MOCKED = True
        return download_chunk_numpy(chunk_coords)

