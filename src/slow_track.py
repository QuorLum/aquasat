import torch
import numpy as np
from src.config import GeospatialConfig
from src.foundation_model import load_foundation_model, PrithviDownstreamAdapter

# Crop Mapping (ID to Label)
CROP_CLASSES = {
    0: "Bare Soil",
    1: "Rice",
    2: "Wheat",
    3: "Sugarcane",
    4: "Maize"
}

# Growth Stage Mapping (ID to Label)
STAGE_CLASSES = {
    0: "Initial",
    1: "Development",
    2: "Mid",
    3: "Late"
}


def run_slow_track(
    reconstructed_optical: np.ndarray, 
    use_fallback: bool = False
) -> tuple:
    """
    Executes the slow-track crop mapping and growth stage tracking pipeline.
    Runs every 10-15 days.
    
    Arguments:
      - reconstructed_optical: np.ndarray of shape (6, Timesteps, H, W)
      - use_fallback: bool, whether to force the Conv3D fallback model
      
    Returns:
      - crop_map: np.ndarray (H, W) of crop IDs
      - stage_map: np.ndarray (H, W) of growth stage IDs
      - baseline_embeddings: np.ndarray (768, H, W) to anchor the fast-track daily model
    """
    c, t, h, w = reconstructed_optical.shape
    
    # 1. Initialize foundation model and downstream adapter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, embed_dim, is_fallback = load_foundation_model(use_fallback=use_fallback)
    encoder = encoder.to(device)
    
    adapter = PrithviDownstreamAdapter(
        embed_dim=embed_dim, 
        num_crop_classes=len(CROP_CLASSES), 
        num_stages=len(STAGE_CLASSES)
    ).to(device)
    
    # Put in evaluation mode
    encoder.eval()
    adapter.eval()
    
    # 2. Format inputs to shape: (Batch=1, Channels=6, Timesteps=3, Height=224, Width=224)
    # Prithvi-100m expects exactly 3 timesteps. If we have more or less, resample/pad.
    target_t = GeospatialConfig.NUM_TIMESTEPS  # 3
    if t != target_t:
        # Resample along time axis
        indices = np.linspace(0, t - 1, target_t).astype(int)
        input_data = reconstructed_optical[:, indices]
    else:
        input_data = reconstructed_optical
        
    # Add batch dimension
    input_tensor = torch.from_numpy(input_data).unsqueeze(0).float().to(device)
    
    # 3. Extract Embeddings & Downstream Inference
    with torch.no_grad():
        # Get raw embeddings from foundation model (encoder)
        embeddings = encoder(input_tensor)
        
        # Reshape / upsample patch embeddings if using standard Prithvi (B, SeqLen, Dim) -> (B, Dim, H, W)
        if len(embeddings.shape) == 3:
            b_size, seq_len, d_dim = embeddings.shape
            spatial_patches = 14
            t_patches = seq_len // (spatial_patches * spatial_patches)
            embeddings = embeddings.view(b_size, t_patches, spatial_patches, spatial_patches, d_dim)
            embeddings = torch.mean(embeddings, dim=1) # Average over temporal patches
            embeddings = embeddings.permute(0, 3, 1, 2) # (B, Dim, 14, 14)
            
        # Ensure embedding shape matches spatial image dimensions (B, Dim, H, W)
        if embeddings.shape[-1] != h or embeddings.shape[-2] != w:
            embeddings = torch.nn.functional.interpolate(
                embeddings, 
                size=(h, w), 
                mode='bilinear', 
                align_corners=False
            )
            
        # Get classification maps
        crop_logits, stage_logits = adapter(embeddings)
        
        # Softmax / Argmax for hard maps
        crop_preds = torch.argmax(crop_logits, dim=1).squeeze(0).cpu().numpy()
        stage_preds = torch.argmax(stage_logits, dim=1).squeeze(0).cpu().numpy()
        
        # Convert baseline embeddings tensor to numpy array
        baseline_embeddings = embeddings.squeeze(0).cpu().numpy()
        
    print(f"[Slow Track] Completed. Classification maps generated using {'Prithvi-100M' if not is_fallback else 'Conv3D Fallback'}.")
    return crop_preds, stage_preds, baseline_embeddings
