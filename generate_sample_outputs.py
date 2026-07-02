import os
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_visual_png(grid: np.ndarray, color_map: dict, legend_labels: dict, title: str, output_path: str):
    """
    Converts a 2D grid into an upscaled PNG visualization with a styled title and legend.
    """
    h, w = grid.shape
    scale = 4  # Upscale factor for sharp pixel rendering
    grid_img = Image.new("RGB", (w * scale, h * scale))
    
    # Fill grid pixels
    for y in range(h):
        for x in range(w):
            val = int(grid[y, x])
            color = color_map.get(val, (61, 61, 61))
            # Draw upscaled block
            for sy in range(scale):
                for sx in range(scale):
                    grid_img.putpixel((x * scale + sx, y * scale + sy), color)
                    
    # Create canvas for visual container (Map + Title + Legend)
    canvas_w = w * scale + 300
    canvas_h = h * scale + 100
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(13, 17, 23)) # Sleek dark background
    
    # Paste map grid
    canvas.paste(grid_img, (20, 80))
    
    # Setup drawing context
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 22)
        text_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        
    # Draw Title
    draw.text((20, 20), title, fill=(88, 166, 255), font=title_font)
    draw.text((20, 50), "AquaSat Spatial Intelligence v1.0 — Gandak Canal Command Area", fill=(139, 148, 158), font=text_font)
    
    # Draw Legend Section
    legend_x = w * scale + 40
    legend_y = 80
    draw.text((legend_x, legend_y), "LEGEND", fill=(88, 166, 255), font=text_font)
    draw.line([(legend_x, legend_y + 20), (legend_x + 220, legend_y + 20)], fill=(48, 54, 61), width=1)
    
    for i, (val, label) in enumerate(legend_labels.items()):
        y_pos = legend_y + 35 + i * 25
        color = color_map.get(val, (61, 61, 61))
        # Draw legend color box
        draw.rectangle([legend_x, y_pos, legend_x + 15, y_pos + 15], fill=color, outline=(48, 54, 61))
        # Draw legend label
        draw.text((legend_x + 25, y_pos), label, fill=(201, 209, 217), font=text_font)
        
    # Save image
    canvas.save(output_path)
    print(f"[Exporter] Visual PNG successfully exported to {output_path}")


def main():
    print("[Exporter] Starting sample output generation...")
    os.makedirs("sample_outputs", exist_ok=True)
    
    # 1. Copy the advisory GeoTIFF
    src_tif = "output/test_advisory_map.tif"
    dst_tif = "sample_outputs/sample_advisory_map.tif"
    
    if os.path.exists(src_tif):
        shutil.copy(src_tif, dst_tif)
        print(f"[Exporter] Copied GeoTIFF to {dst_tif}")
    else:
        print("[Exporter] Warning: Source test advisory GeoTIFF not found. Run integration tests first.")
        
    # 2. Load and visualize the Advisory Map
    src_npy = "output/test_advisory_map.npy"
    if os.path.exists(src_npy):
        advisory_grid = np.load(src_npy)
        
        # Color mapping matching app.py
        # 0: Fallow, 1: Optimal, 2: Light Irrigation, 3: Conserve Moisture, 4: Critical Irrigation, 5: Severe Deficit
        advisory_colors = {
            0: (61, 61, 61),
            1: (63, 185, 80),
            2: (88, 166, 255),
            3: (210, 153, 34),
            4: (240, 136, 62),
            5: (248, 81, 73)
        }
        
        advisory_labels = {
            0: "0: Fallow (Bare Soil)",
            1: "1: Optimal (No Stress)",
            2: "2: Light Irrigation (Irrigated)",
            3: "3: Conserve Moisture (Rainfed)",
            4: "4: Critical Irrigation (CRITICAL)",
            5: "5: Severe Deficit (CRITICAL)"
        }
        
        generate_visual_png(
            advisory_grid,
            advisory_colors,
            advisory_labels,
            "🗺️ Precision Irrigation Advisory Map",
            "sample_outputs/sample_advisory_map_visual.png"
        )
    else:
        print("[Exporter] Warning: Source test advisory grid (.npy) not found. Run integration tests first.")
        
    # 3. Ingest GEE to generate and visualize Crop Map
    print("[Exporter] Initializing Earth Engine connection for crop mapping...")
    try:
        import ee
        # Initialize project
        ee.Initialize(project="kaggle-project-499515")
        
        from src.gee_pipeline import download_chunk_numpy, get_spatial_chunks
        from src.gap_filling import reconstruct_cloud_gaps
        from src.slow_track import run_slow_track
        from src.config import GeospatialConfig
        
        chunks = get_spatial_chunks(GeospatialConfig.ROI_COORDINATES, GeospatialConfig.CHUNK_SIZE_DEG)
        test_chunk = chunks[0]
        
        print(f"[Exporter] Downloading S2/S1 time-series for chunk: {test_chunk}...")
        fused_raw = download_chunk_numpy(test_chunk)
        
        print("[Exporter] Reconstructing cloud gaps via RVI and Savitzky-Golay...")
        reconstructed = reconstruct_cloud_gaps(fused_raw)
        
        print("[Exporter] Generating crop classification map via Slow Track model...")
        crop_map, _, _ = run_slow_track(reconstructed, use_fallback=True)
        
        # Save crop map array
        np.save("sample_outputs/sample_crop_map.npy", crop_map)
        print("[Exporter] Saved sample crop map array (.npy)")
        
        # Color mapping matching app.py
        # 0: Bare Soil, 1: Rice, 2: Wheat, 3: Sugarcane, 4: Maize
        crop_colors = {
            0: (61, 61, 61),     # Bare Soil
            1: (46, 160, 67),    # Rice
            2: (219, 171, 9),    # Wheat
            3: (163, 113, 247),  # Sugarcane
            4: (240, 136, 62)    # Maize
        }
        
        crop_labels = {
            0: "0: Bare Soil / Fallow",
            1: "1: Rice (Monsoon wet-crop)",
            2: "2: Wheat (Rabi winter-crop)",
            3: "3: Sugarcane (Perennial)",
            4: "4: Maize (Upland dry-crop)"
        }
        
        generate_visual_png(
            crop_map,
            crop_colors,
            crop_labels,
            "🌾 Crop Classification Map",
            "sample_outputs/sample_crop_map_visual.png"
        )
    except Exception as e:
        print(f"[Exporter] Error generating GEE crop classification: {e}")


if __name__ == "__main__":
    main()
