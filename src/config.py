import os

class GeospatialConfig:
    # Bounding box for canal command area (default study area in Bihar, India)
    # Format: [min_lon, min_lat, max_lon, max_lat]
    # Coordinates focus on the Gandak Canal Command Area
    ROI_COORDINATES = [84.5, 25.8, 85.0, 26.3]
    
    # Temporal Window Settings (Slow Track)
    TEMPORAL_WINDOW_DAYS = 15  # 15-day intervals for slow-track temporal grouping
    START_DATE = "2025-06-01"  # Monsoon start (June 1)
    END_DATE = "2025-10-31"    # Crop season end (October 31)
    
    # GEE Collection IDs
    S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
    S1_COLLECTION = "COPERNICUS/S1_GRD"
    LISS3_COLLECTION = "ISRO/LISS3"  # Standard GEE asset ID for LISS-III if available, else fallback
    
    # Band mappings for NASA/IBM Prithvi-100M
    # Prithvi-100M expects 6 bands in the following order:
    # 0: Blue, 1: Green, 2: Red, 3: Narrow NIR, 4: SWIR-1, 5: SWIR-2
    PRITHVI_BANDS = ["Blue", "Green", "Red", "Narrow_NIR", "SWIR1", "SWIR2"]
    
    # Mapping GEE band names to Prithvi indices
    S2_BAND_MAP = {
        "B2": "Blue",
        "B3": "Green",
        "B4": "Red",
        "B8A": "Narrow_NIR",
        "B11": "SWIR1",
        "B12": "SWIR2"
    }
    
    # LISS-III bands mapping (Note: LISS-III lacks Blue and SWIR2. We fuse Sentinel-2 for these).
    LISS3_BAND_MAP = {
        "B1": "Green",
        "B2": "Red",
        "B3": "Narrow_NIR",
        "B4": "SWIR1"
    }
    
    # Radar (Sentinel-1 & EOS-04) mappings
    SAR_BANDS = ["VV", "VH"]
    
    # Model Configurations
    MODEL_NAME = "ibm-nasa-geospatial/Prithvi-100M"
    MODEL_RESOLUTION = 30  # spatial resolution in meters
    TILE_SIZE = 224        # Prithvi expects 224x224 input patches
    NUM_CHANNELS = 6       # 6 spectral bands
    NUM_TIMESTEPS = 3      # Prithvi expects 3 timesteps (e.g. binned frames)
    
    # PEFT / LoRA fine-tuning hyperparameters
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LEARNING_RATE = 5e-4
    BATCH_SIZE = 4
    EPOCHS = 10
    
    # Spatial Ingestion Chunking
    CHUNK_SIZE_DEG = 0.1  # Bounding box chunk size (0.1 deg ~11km) to prevent GEE payload timeouts
    
    # Wind dynamics radius
    WIND_RADIUS_KM = 350.0  # Search radius for meteorological atmospheric dynamics
    
    # Open-Meteo API config
    OPEN_METEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Output directories
    OUTPUT_DIR = "d:/MY Projects/antigravity/Kaggle Project/output"
    MODEL_DIR = "d:/MY Projects/antigravity/Kaggle Project/models"
    
    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)

try:
    import ee
    try:
        print("[Config] Initializing Earth Engine project: kaggle-project-499515")
        ee.Initialize(project="kaggle-project-499515")
        print("[Config] Earth Engine successfully initialized.")
    except Exception as init_err:
        print(f"[Config] Direct initialization failed: {init_err}")
        print("[Config] Authenticating Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project="kaggle-project-499515")
        print("[Config] Earth Engine successfully authenticated and initialized.")
except Exception as e:
    print(f"[Config] Warning: Earth Engine initialization failed: {e}")





