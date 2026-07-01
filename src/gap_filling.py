import numpy as np
from src.config import GeospatialConfig

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def db_to_linear(val_db: np.ndarray) -> np.ndarray:
    """
    Converts decibels (dB) backscatter values to linear scale.
    """
    return 10.0 ** (val_db / 10.0)


def compute_radar_vegetation_index(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """
    Computes a simplified Radar Vegetation Index (RVI) for dual-pol SAR data.
    Formula: RVI = (4 * VH_linear) / (VV_linear + VH_linear)
    Values range from 0 to 1. Higher values correspond to dense vegetation/canopy scatter.
    """
    vv_linear = db_to_linear(vv_db)
    vh_linear = db_to_linear(vh_db)
    
    # Prevent division by zero
    denominator = vv_linear + vh_linear
    denominator = np.where(denominator == 0, 1e-6, denominator)
    
    rvi = (4.0 * vh_linear) / denominator
    # Standardize RVI to [0, 1] range for stability
    rvi = np.clip(rvi, 0.0, 1.0)
    return rvi


def temporal_linear_interpolate(fused_chunk: np.ndarray) -> np.ndarray:
    """
    Linearly interpolates missing (cloudy) optical data across the temporal dimension.
    Input shape: (Channels=9, Timesteps, Height, Width)
    Output shape: (Channels=9, Timesteps, Height, Width) with filled optical values.
    """
    channels, timesteps, h, w = fused_chunk.shape
    filled_chunk = fused_chunk.copy()
    
    # Loop over pixels
    for y in range(h):
        for x in range(w):
            clear_mask = fused_chunk[6, :, y, x]  # clear_mask channel
            
            # Find indices of clear observations
            clear_indices = np.where(clear_mask > 0.5)[0]
            
            # If no observations are clear, we cannot interpolate; leave as zero or constant
            if len(clear_indices) == 0:
                # Use a default constant profile for typical agricultural regions
                # Blue, Green, Red, Narrow_NIR, SWIR1, SWIR2
                default_optical = [0.05, 0.08, 0.06, 0.25, 0.15, 0.10]
                for b in range(6):
                    filled_chunk[b, :, y, x] = default_optical[b]
                continue
                
            # If only a subset of observations are clear, interpolate
            for b in range(6):
                profile = fused_chunk[b, :, y, x]
                
                # Check if there are missing steps
                if len(clear_indices) < timesteps:
                    # Perform linear interpolation
                    interp_profile = np.interp(
                        np.arange(timesteps),
                        clear_indices,
                        profile[clear_indices]
                    )
                    filled_chunk[b, :, y, x] = interp_profile
                    
    return filled_chunk


def apply_savitzky_golay(optical_data: np.ndarray, window_len: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Applies Savitzky-Golay filter along the temporal axis to smooth the reconstructed timeseries.
    Input shape: (6, Timesteps, Height, Width)
    Output shape: (6, Timesteps, Height, Width)
    """
    num_bands, timesteps, h, w = optical_data.shape
    smoothed = optical_data.copy()
    
    # Adjust window length if timesteps is too small
    if timesteps <= window_len:
        # Window length must be odd and less than timesteps
        window_len = timesteps if timesteps % 2 != 0 else timesteps - 1
        if window_len < 3:
            # Not enough steps for Savitzky-Golay; fallback to linear smoothing or return as-is
            return optical_data
            
    if HAS_SCIPY:
        for b in range(num_bands):
            for y in range(h):
                for x in range(w):
                    smoothed[b, :, y, x] = savgol_filter(
                        optical_data[b, :, y, x],
                        window_length=window_len,
                        polyorder=polyorder,
                        mode='nearest'
                    )
    else:
        # Fallback to simple moving average smoothing if scipy is not installed
        half_w = window_len // 2
        for b in range(num_bands):
            for y in range(h):
                for x in range(w):
                    profile = optical_data[b, :, y, x]
                    smoothed_profile = np.zeros_like(profile)
                    for t in range(timesteps):
                        start = max(0, t - half_w)
                        end = min(timesteps, t + half_w + 1)
                        smoothed_profile[t] = np.mean(profile[start:end])
                    smoothed[b, :, y, x] = smoothed_profile
                    
    # Clip results to valid reflectance range
    return np.clip(smoothed, 0.0, 1.0)


def reconstruct_cloud_gaps(fused_chunk: np.ndarray) -> np.ndarray:
    """
    Complete Cloud Gap Filling and Preprocessing Pipeline.
    1. Linearly interpolates cloudy gaps.
    2. Uses continuous Radar (Sentinel-1 / EOS-04) data to calculate Radar Vegetation Index (RVI).
    3. Corrects/guides the optical bands using RVI anchors during cloud-obscured timesteps.
    4. Applies Savitzky-Golay smoothing across timesteps.
    
    Input:
      fused_chunk: np.ndarray of shape (9, Timesteps, Height, Width)
    Returns:
      reconstructed: np.ndarray of shape (6, Timesteps, Height, Width) (Only the 6 optical bands)
    """
    # 1. Linearly interpolate the optical bands
    interpolated = temporal_linear_interpolate(fused_chunk)
    
    channels, timesteps, h, w = fused_chunk.shape
    optical_reconstructed = interpolated[:6].copy()
    
    # 2. Extract Radar bands and calculate RVI
    vv = fused_chunk[7]  # Shape: (Timesteps, Height, Width)
    vh = fused_chunk[8]  # Shape: (Timesteps, Height, Width)
    rvi = compute_radar_vegetation_index(vv, vh)  # Shape: (Timesteps, Height, Width)
    
    # 3. Radar-guided adjustment for cloudy timesteps
    # During heavily cloudy dates, linear interpolation might overshoot or undershoot.
    # We use RVI as an anchor: High RVI -> High NIR, Lower SWIR. Low RVI -> Low NIR.
    for t in range(timesteps):
        for y in range(h):
            for x in range(w):
                is_cloudy = fused_chunk[6, t, y, x] < 0.5
                if is_cloudy:
                    pixel_rvi = rvi[t, y, x]
                    
                    # Target bands:
                    # Index 3 is Narrow NIR (B8A)
                    # Index 4 is SWIR1 (B11)
                    # Index 5 is SWIR2 (B12)
                    
                    # Guide NIR (B8A): NIR reflectance is highly correlated with vegetative density (RVI)
                    # A typical NIR reflectance ranges from 0.10 (soil/no crops) to 0.60 (fully closed canopy)
                    expected_nir = 0.10 + 0.50 * pixel_rvi
                    # Blend the linear-interpolated NIR and the radar-estimated NIR
                    optical_reconstructed[3, t, y, x] = 0.4 * optical_reconstructed[3, t, y, x] + 0.6 * expected_nir
                    
                    # Guide SWIR1 & SWIR2: SWIR reflectance decreases with vegetation water content (higher RVI)
                    # A typical SWIR1 ranges from 0.40 (dry soil/stressed) to 0.12 (well-watered crop)
                    expected_swir1 = 0.40 - 0.28 * pixel_rvi
                    optical_reconstructed[4, t, y, x] = 0.4 * optical_reconstructed[4, t, y, x] + 0.6 * expected_swir1
                    
                    expected_swir2 = 0.30 - 0.22 * pixel_rvi
                    optical_reconstructed[5, t, y, x] = 0.4 * optical_reconstructed[5, t, y, x] + 0.6 * expected_swir2
                    
                    # Guide Red (B4): Red light absorption increases with chlorophyll (higher RVI)
                    # Typical Red ranges from 0.20 (soil) to 0.02 (healthy crop)
                    expected_red = 0.22 - 0.20 * pixel_rvi
                    optical_reconstructed[2, t, y, x] = 0.4 * optical_reconstructed[2, t, y, x] + 0.6 * expected_red
                    
    # 4. Smooth the reconstructed optical bands using Savitzky-Golay
    smoothed_optical = apply_savitzky_golay(optical_reconstructed, window_len=5, polyorder=2)
    
    return smoothed_optical
