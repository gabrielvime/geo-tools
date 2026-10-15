import os
import shutil
from pathlib import Path
import rasterio
from rasterio.windows import Window

input_folder = "Google_Imagery"
output_folder = "Google_Imagery_512"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

tile_size = 512
stride = 512

for file_name in sorted(os.listdir(input_folder)):
    if file_name.startswith('.'):
        continue
        
    if file_name.lower().endswith(('.tif', '.tiff')):
        img_path = os.path.join(input_folder, file_name)
        
        # Abre o GeoTIFF original preservando o sistema de coordenadas (CRS)
        with rasterio.open(img_path) as src:
            w, h = src.width, src.height
            base_name = Path(file_name).stem

            img_output_folder = os.path.join(output_folder, base_name)
            os.makedirs(img_output_folder, exist_ok=True)

            tiles_count = 0
            
            for col in range(0, w - tile_size + 1, stride):
                for row in range(0, h - tile_size + 1, stride):
                    window = Window(col, row, tile_size, tile_size)
                    
                    transform = rasterio.windows.transform(window, src.transform)
                    
                    tile_data = src.read(window=window)
                    
                    profile = src.profile.copy()
                    profile.update({
                        'height': tile_size,
                        'width': tile_size,
                        'transform': transform,
                        'driver': 'GTiff'
                    })
                    
                    save_name = f"{base_name}_tile_{col}_{row}.tif"
                    save_path = os.path.join(img_output_folder, save_name)
                    
                    with rasterio.open(save_path, 'w', **profile) as dst:
                        dst.write(tile_data)
                        
                    tiles_count += 1
                    
            print(f"OK: '{file_name}' -> gerados {tiles_count} tiles GeoTIFF georreferenciados.")

print("\nProcessamento concluÃ­do com sucesso!")