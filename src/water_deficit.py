import numpy as np
from datetime import datetime
from src.config import GeospatialConfig

# FAO-56 Crop Coefficient (Kc) Lookup Table
# Represents Initial, Development, Mid, and Late stages
KC_TABLE = {
    "Rice": {"Initial": 1.05, "Development": 1.15, "Mid": 1.20, "Late": 0.90},
    "Wheat": {"Initial": 0.40, "Development": 0.70, "Mid": 1.15, "Late": 0.40},
    "Sugarcane": {"Initial": 0.40, "Development": 0.85, "Mid": 1.25, "Late": 0.75},
    "Maize": {"Initial": 0.40, "Development": 0.80, "Mid": 1.20, "Late": 0.60},
    "Vegetable": {"Initial": 0.50, "Development": 0.70, "Mid": 1.05, "Late": 0.85},
    "Default": {"Initial": 0.50, "Development": 0.75, "Mid": 1.10, "Late": 0.70}
}


def calculate_solar_radiation(lat_deg: float, date_str: str, temp_max: float, temp_min: float) -> float:
    """
    Estimates daily solar radiation (Rs, MJ/m2/day) using Hargreaves temperature method.
    Formula: Rs = 0.16 * sqrt(Tmax - Tmin) * Ra
    Where Ra is extraterrestrial radiation calculated from latitude and day of year.
    """
    # 1. Day of the year
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    
    # 2. Latitude in radians
    lat_rad = np.radians(lat_deg)
    
    # 3. Solar declination (dec)
    dec = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    
    # 4. Sunset hour angle (ws)
    # Cosine sunset hour angle can be clamped for extreme latitudes
    cos_ws = -np.tan(lat_rad) * np.tan(dec)
    cos_ws = np.clip(cos_ws, -1.0, 1.0)
    ws = np.arccos(cos_ws)
    
    # 5. Inverse relative distance Earth-Sun (dr)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    
    # 6. Extraterrestrial radiation (Ra, MJ/m2/day)
    # Gsc = solar constant = 0.0820 MJ/m2/min
    gsc = 0.0820
    ra = (24.0 * 60.0 / np.pi) * gsc * dr * (
        ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
    )
    
    # 7. Hargreaves solar radiation estimation
    temp_diff = max(0.1, temp_max - temp_min)
    rs = 0.16 * np.sqrt(temp_diff) * ra
    return rs


def calculate_fao56_eto(
    temp_max: float, 
    temp_min: float, 
    wind_speed: float, 
    solar_rad: float, 
    elevation_m: float = 100.0
) -> float:
    """
    Calculates reference crop evapotranspiration (ETo, mm/day) using FAO-56 Penman-Monteith.
    Arguments:
      - temp_max: Max daily temperature at 2m (deg C)
      - temp_min: Min daily temperature at 2m (deg C)
      - wind_speed: Wind speed at 2m (m/s)
      - solar_rad: Solar radiation (MJ/m2/day)
      - elevation_m: Station elevation above sea level (meters)
    """
    # 1. Mean Temperature
    t_mean = (temp_max + temp_min) / 2.0
    
    # 2. Atmospheric Pressure (P)
    p = 101.3 * ((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26
    
    # 3. Psychrometric Constant (gamma)
    gamma = 0.000665 * p
    
    # 4. Saturation Vapor Pressure (es)
    es_tmax = 0.6108 * np.exp(17.27 * temp_max / (temp_max + 237.3))
    es_tmin = 0.6108 * np.exp(17.27 * temp_min / (temp_min + 237.3))
    es = (es_tmax + es_tmin) / 2.0
    
    # 5. Actual Vapor Pressure (ea)
    # Standard FAO-56 recommendation when RH is missing: assume Tmin approximates Dew Point.
    ea = es_tmin
    
    # 6. Slope of Vapor Pressure Curve (Delta)
    delta = 4098.0 * (0.6108 * np.exp(17.27 * t_mean / (t_mean + 237.3))) / ((t_mean + 237.3) ** 2)
    
    # 7. Net Radiation (Rn)
    # Net shortwave radiation (Rns) assuming standard canopy albedo = 0.23
    rns = (1.0 - 0.23) * solar_rad
    # Net longwave radiation (Rnl) modeled using simplified clear-sky factor
    # For a daily scale, soil heat flux (G) is approximated as 0.
    g = 0.0
    
    # Standard longwave radiation approximation
    # sigma = Boltzmann constant = 4.903e-9 MJ/K4/m2/day
    sigma = 4.903e-9
    tmax_k = temp_max + 273.16
    tmin_k = temp_min + 273.16
    rnl = sigma * ((tmax_k**4 + tmin_k**4) / 2.0) * (0.34 - 0.14 * np.sqrt(ea)) * 0.8 # assuming avg cloudiness
    rn = rns - rnl
    
    # 8. Penman-Monteith ETo Equation
    numerator = 0.408 * delta * (rn - g) + gamma * (900.0 / (t_mean + 273.0)) * wind_speed * (es - ea)
    denominator = delta + gamma * (1.0 + 0.34 * wind_speed)
    
    eto = numerator / denominator
    return float(max(0.0, eto))


def calculate_etc(eto: float, crop_type: str, growth_stage: str) -> float:
    """
    Computes crop evapotranspiration (ETc, mm/day) from reference ETo and growth stage.
    """
    crop = crop_type if crop_type in KC_TABLE else "Default"
    stage = growth_stage if growth_stage in KC_TABLE[crop] else "Mid"
    
    kc = KC_TABLE[crop][stage]
    return eto * kc


def track_soil_moisture_depletion(
    prev_depletion: np.ndarray,
    etc: np.ndarray,
    rainfall: float,
    irrigation_mask: np.ndarray,
    soil_whc: np.ndarray,
    crop_roots_m: float = 0.6
) -> tuple:
    """
    Runs a spatial daily soil water balance simulation for the crop root zone.
    Inputs:
      - prev_depletion: numpy array of soil water depletion (mm) at day t-1.
      - etc: numpy array of crop evapotranspiration (mm/day) at day t.
      - rainfall: daily rain height (mm) on day t.
      - irrigation_mask: binary array (1 = irrigated zone, 0 = tail-end rainfed).
      - soil_whc: soil water holding capacity (fraction, e.g. 0.15 for loam).
      - crop_roots_m: active root depth (m).
    
    Returns:
      - new_depletion: updated soil water depletion array (mm) at day t.
      - moisture_stress_prob: daily moisture stress probability [0..1] array.
    """
    # 1. Total Available Water (TAW, mm) in the root zone
    # TAW = 1000 * WHC_fraction * Root_depth_m
    taw = 1000.0 * soil_whc * crop_roots_m
    
    # 2. Daily Irrigation contribution estimation
    # Irrigated zones receive water when ETc exceeds rain, targeting field capacity (depletion -> 0)
    # Tail-end rainfed zones receive 0 irrigation.
    irrigation = np.zeros_like(prev_depletion)
    # Simulate canal irrigation delivery (e.g., 8mm depth if irrigated and needs water)
    irrigation = np.where((irrigation_mask > 0.5) & (prev_depletion > 10.0), 8.0, 0.0)
    
    # 3. Water Balance Update
    # Depletion increases with ETc, decreases with rain & irrigation
    # D_t = D_{t-1} + ETc - Rain - Irrigation
    new_depletion = prev_depletion + etc - rainfall - irrigation
    
    # Soil physical bounds:
    # Depletion cannot be negative (soil cannot hold more than field capacity; excess drains as runoff)
    new_depletion = np.clip(new_depletion, 0.0, taw)
    
    # 4. Compute Moisture Stress Probability
    # Readily Available Water (RAW) represents depletion threshold before stress occurs.
    # RAW = p * TAW, where p (Management Allowed Depletion, MAD) is typically 0.5
    raw = 0.5 * taw
    
    # Stress probability scales linearly from 0 (at depletion <= RAW) to 1.0 (at depletion = TAW)
    stress_index = (new_depletion - raw) / (taw - raw)
    moisture_stress_prob = np.clip(stress_index, 0.0, 1.0)
    
    return new_depletion, moisture_stress_prob
