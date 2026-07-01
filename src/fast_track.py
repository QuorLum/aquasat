import numpy as np
import torch
import torch.nn as nn
from src.config import GeospatialConfig

class FastTrackPredictor(nn.Module):
    """
    Lightweight convolutional layer to process 768-dim baseline embeddings 
    and map them to NDWI moisture stress regression coefficients.
    """
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        # Reduce dimensionality from 768 to 16
        self.conv1 = nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.regressor = nn.Conv2d(8, 1, kernel_size=1)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(embeddings))
        x = torch.relu(self.conv2(x))
        ndwi_multiplier = torch.sigmoid(self.regressor(x))
        return ndwi_multiplier


def run_fast_track(
    baseline_embeddings: np.ndarray,
    daily_weather: dict,
    prev_depletion: np.ndarray,
    irrigation_mask: np.ndarray,
    soil_whc: np.ndarray,
    crop_map: np.ndarray,
    stage_map: np.ndarray,
    lat_deg: float = 26.0
) -> tuple:
    """
    Runs the daily fast-track moisture stress prediction loop.
    
    Arguments:
      - baseline_embeddings: np.ndarray (768, H, W) from the latest slow-track run.
      - daily_weather: dict containing temp_max, temp_min, rain, wind_speed, wind_dir.
      - prev_depletion: np.ndarray (H, W) soil water depletion at day t-1.
      - irrigation_mask: np.ndarray (H, W) 1 = irrigated, 0 = tail-end rainfed.
      - soil_whc: np.ndarray (H, W) soil water holding capacity.
      - crop_map: np.ndarray (H, W) crop classification IDs.
      - stage_map: np.ndarray (H, W) phenology growth stage IDs.
      - lat_deg: float, central latitude of the tile.
      
    Returns:
      - moisture_stress_prob: np.ndarray (H, W) daily probability of moisture stress (0.0 to 1.0)
      - updated_depletion: np.ndarray (H, W) updated daily depletion map (mm)
      - calculated_ndwi: np.ndarray (H, W) predicted NDWI index mapping crop leaf moisture
    """
    h, w = baseline_embeddings.shape[1], baseline_embeddings.shape[2]
    
    # 1. Fetch meteorological parameters
    t_max = daily_weather["temp_max"]
    t_min = daily_weather["temp_min"]
    rain = daily_weather["rain"]
    wind_speed = daily_weather["wind_speed"]
    
    # 2. Perform FAO-56 Penman-Monteith Evapotranspiration Calculations
    from src.water_deficit import calculate_solar_radiation, calculate_fao56_eto, calculate_etc, track_soil_moisture_depletion
    
    today_str = datetime_to_date_str() if "date" not in daily_weather else daily_weather["date"]
    
    solar_rad = calculate_solar_radiation(lat_deg, today_str, t_max, t_min)
    eto = calculate_fao56_eto(t_max, t_min, wind_speed, solar_rad)
    
    # Calculate pixel-level ETc based on crop and growth stage
    etc_map = np.zeros((h, w), dtype=np.float32)
    crop_names = ["Bare Soil", "Rice", "Wheat", "Sugarcane", "Maize"]
    stage_names = ["Initial", "Development", "Mid", "Late"]
    
    for c_id, crop_name in enumerate(crop_names):
        for s_id, stage_name in enumerate(stage_names):
            mask = (crop_map == c_id) & (stage_map == s_id)
            if np.any(mask):
                etc_val = calculate_etc(eto, crop_name, stage_name)
                etc_map[mask] = etc_val
                
    # 3. Simulate physical soil water depletion balance
    # Assuming average crop rooting depth of 0.6 meters
    updated_depletion, physical_stress_prob = track_soil_moisture_depletion(
        prev_depletion,
        etc_map,
        rain,
        irrigation_mask,
        soil_whc,
        crop_roots_m=0.6
    )
    
    # 4. Neural mapping of embeddings to NDWI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = FastTrackPredictor(embed_dim=768).to(device)
    predictor.eval()
    
    emb_tensor = torch.from_numpy(baseline_embeddings).unsqueeze(0).float().to(device)
    with torch.no_grad():
        ndwi_factor = predictor(emb_tensor).squeeze(0).squeeze(0).cpu().numpy()
        
    # NDWI behaves inversely to moisture stress.
    # Healthy (well watered): high NDWI (e.g. 0.4 to 0.6)
    # Stressed (depleted): low NDWI (e.g. -0.1 to 0.2)
    # Let's map NDWI based on baseline embedding weights adjusted by physical moisture stress
    # Daily NDWI = Baseline NDWI multiplier * (1.0 - 0.7 * physical_stress_prob)
    calculated_ndwi = ndwi_factor * (1.0 - 0.7 * physical_stress_prob)
    calculated_ndwi = np.clip(calculated_ndwi, -1.0, 1.0)
    
    # Combine neural baseline and physical hydrology depletion to forecast moisture stress probability
    # Fast-track moisture stress is calibrated as 0.7 * physical_stress_prob + 0.3 * (1.0 - NDWI_rescaled)
    ndwi_scaled = (calculated_ndwi + 1.0) / 2.0  # scale from [-1, 1] to [0, 1]
    neural_stress_proxy = 1.0 - ndwi_scaled
    
    final_moisture_stress_prob = 0.7 * physical_stress_prob + 0.3 * neural_stress_proxy
    final_moisture_stress_prob = np.clip(final_moisture_stress_prob, 0.0, 1.0)
    
    # Zero out stress for bare soil areas (class 0)
    final_moisture_stress_prob = np.where(crop_map == 0, 0.0, final_moisture_stress_prob)
    
    return final_moisture_stress_prob, updated_depletion, calculated_ndwi


def datetime_to_date_str() -> str:
    """Helper to return current date in YYYY-MM-DD"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
