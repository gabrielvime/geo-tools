import os
from io import BytesIO
from pathlib import Path
import mercantile
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import requests
from PIL import Image
import time
import random

import shape_tools, config

def getData(gdf, shapefile_name, source):
    
    output_dir = os.path.join(config.SOURCES.get(source)[1], shapefile_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Pasta criada/verificada: {output_dir}")

    sub_parts = shape_tools.divide(gdf)
    total_parts = len(sub_parts)
    print(f"O polígono foi dividido em {total_parts} partes otimizadas.")

    for idx, part in enumerate(sub_parts):
        print(f"\n--- Processando parte {idx + 1} de {total_parts} ---")
        
        requests_cont = 0
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        exp_minx, exp_miny, exp_maxx, exp_maxy = part.bounds

        w_4326, s_4326, e_4326, n_4326 = transform_bounds("EPSG:3857", "EPSG:4326", exp_minx, exp_miny, exp_maxx, exp_maxy)

        print("Buscando XYZ tiles para a sub-área...")
        tiles = list(mercantile.tiles(w_4326, s_4326, e_4326, n_4326, config.ZOOM))

        min_x = min(t.x for t in tiles)
        max_x = max(t.x for t in tiles)
        min_y = min(t.y for t in tiles)
        max_y = max(t.y for t in tiles)

        print("Criando mosaico da parte...")
        img_w = (max_x - min_x + 1) * 256
        img_h = (max_y - min_y + 1) * 256
        
        mosaic = Image.new("RGB", (img_w, img_h))

        total_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
        i = 1
        for ty in range(min_y, max_y + 1):
            for tx in range(min_x, max_x + 1):
                if source == 'google':
                    url = f'https://mt1.google.com/vt/lyrs=s&x={tx}&y={ty}&z={config.ZOOM}'
                elif source == 'bing':
                    qk = mercantile.quadkey(tx, ty, config.ZOOM)
                    url = f""
                    
                if requests_cont >= 100:
                    print(f'Limite de requisições excedido!! Pausando...')
                    for s in range(300, -1, -1):
                        time.sleep(1)
                    requests_cont = 0

                response = requests.get(url, headers=headers)
                requests_cont += 1

                if response.status_code == 200:
                    tile_img = Image.open(BytesIO(response.content))
                    px = (tx - min_x) * 256
                    py = (ty - min_y) * 256
                    mosaic.paste(tile_img, (px, py))
                i += 1

        top_left_bounds = mercantile.xy_bounds(min_x, min_y, config.ZOOM)
        bottom_right_bounds = mercantile.xy_bounds(max_x, max_y, config.ZOOM)

        transform = from_bounds(top_left_bounds.left, bottom_right_bounds.bottom, bottom_right_bounds.right, top_left_bounds.top, img_w, img_h)
        arr = np.array(mosaic)

        out_w = config.RESOLUTION
        out_h = config.RESOLUTION

        meta = {
            "driver": "GTiff",
            "height": img_h,
            "width": img_w,
            "count": 3,
            "dtype": "uint8",
            "crs": "EPSG:3857",
            "transform": transform,
        }

        with MemoryFile() as memfile:
            with memfile.open(**meta) as dataset:
                dataset.write(arr[:, :, 0], 1)
                dataset.write(arr[:, :, 1], 2)
                dataset.write(arr[:, :, 2], 3)
                
                window = window_from_bounds(exp_minx, exp_miny, exp_maxx, exp_maxy, dataset.transform)
                
                cropped_image = dataset.read(
                    window=window, 
                    out_shape=(3, out_h, out_w),
                    resampling=Resampling.bilinear,
                    boundless=False, fill_value=0
                )
                
                final_transform = rasterio.transform.from_bounds(
                    exp_minx, exp_miny, exp_maxx, exp_maxy, 
                    out_w, out_h
                )
        
        meta.update({
            'height': out_h,
            'width': out_w,
            'transform': final_transform
        })

        # Desenha apenas a porção real da propriedade que pertence a este bloco
        if config.DRAW_POLYGON:
            print(f'Desenhando polígono na parte {idx + 1}...')
            img_to_draw = np.transpose(cropped_image, (1, 2, 0))
            arr = shape_tools.draw(
                image=img_to_draw, 
                shape=gdf,
                transform=final_transform, 
                crs="EPSG:3857")
        else:
            arr = np.transpose(cropped_image, (1, 2, 0))

        output_filepath = os.path.join(output_dir, f"{shapefile_name}_part_{idx}_z{config.ZOOM}.tif")

        print(f"Exportando parte para {output_filepath}...")
        with rasterio.open(output_filepath, "w", **meta) as dst:
            dst.write(arr[:, :, 0], 1)
            dst.write(arr[:, :, 1], 2)
            dst.write(arr[:, :, 2], 3)

    print("Processamento de todas as partes concluído com sucesso!")