import geopandas as gpd
import os
import rasterio
from rasterio.features import rasterize
import numpy as np
from rasterio.warp import transform_bounds
from shapely.geometry import box

import config

'''
ShapeFile functions
'''


def gdf(shape_file):
    '''
    Get GeoPandas GeoDataFrame in WGS84
    '''

    gdf = gpd.read_file(shape_file)

    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    return gdf


def draw(image, shape, transform, crs="EPSG:4326"):
    """
    Draw polygon shape with automatic array format detection.
    """


    # 1. Garante que as geometrias estejam no CRS correto da imagem
    shape = shape.to_crs(crs)
    boundaries = shape.geometry.boundary
    
    # 2. Aplica buffer se a linha for mais larga que 1px
    if config.POLYGON_WIDTH > 1:
        pixel_size = abs(transform.a)
        boundaries = boundaries.buffer(config.POLYGON_WIDTH * pixel_size)

    # 3. Define a cor RGB (valores de 0 a 255)
    colors = {
        'red': (255, 0, 0),
        'yellow': (255, 255, 0),
        'blue': (0, 0, 255)  # Corrigido de (255, 255, 255) para o azul real
    }
    rgb_color = colors.get(config.POLYGON_COLOR.lower(), (255, 0, 0))
    
    is_channels_first = image.shape[0] < 10

    if is_channels_first:
        mask_shape = (image.shape[1], image.shape[2])
    else:    
        mask_shape = (image.shape[0], image.shape[1])

    # 5. Cria máscara booleana rasterizando as geometrias
    polygon_mask = rasterize(
        shapes=boundaries,
        out_shape=mask_shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=True
    ) > 0

    # 6. Aplica a cor em cada banda do array RGB de acordo com o formato
    for band_idx in range(3):
        if is_channels_first:
            # Lógica STAC / Rasterio
            image[band_idx, polygon_mask] = rgb_color[band_idx]
        else:
            # Lógica Web Tiles / Numpy
            #image[:, :, band_idx][polygon_mask] = rgb_color[band_idx]
            image[polygon_mask, band_idx] = rgb_color[band_idx]
        
    return image


def window(shape, src, max_size=True, size=config.RESOLUTION, min_max=False, crs="EPSG:4326"):
    '''
    Return bounding window fo given gdf shape
    if min_max, return minX, maX, minY, maxY
    '''
    
    if crs=='EPSG:3857':
        shape = shape.to_crs(crs)

    # calculates width and height
    minx, miny, maxx, maxy = shape.total_bounds
    width = maxx - minx
    height = maxy - miny

    # centers
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    pixel_size = 1 if not hasattr(src, 'res') else src.res[0]

    if config.SQUARE:

        square = max(width, height) * config.EXPAND_FACTOR if config.MAX_SIZE else size * pixel_size * config.EXPAND_FACTOR

        exp_minx = cx - (square / 2)
        exp_maxx = cx + (square / 2)
        exp_miny = cy - (square / 2)
        exp_maxy = cy + (square / 2)

    else:

        expand = max(width, height) - min(height, width) if config.MAX_SIZE else 0 
        exp_minx = cx - (((width + expand) * config.EXPAND_FACTOR) / 2) 
        exp_maxx = cx + (((width + expand) * config.EXPAND_FACTOR) / 2) 
        exp_miny = cy - (((height + expand) * config.EXPAND_FACTOR) / 2)
        exp_maxy = cy + (((height + expand) * config.EXPAND_FACTOR) / 2)
   
    if shape.crs == 'EPSG:3857':
        bounds = transform_bounds(shape.crs, 'EPSG:4326', exp_minx, exp_miny, exp_maxx, exp_maxy)
        exp_minx, exp_miny, exp_maxx, exp_maxy = bounds
    
    
    if min_max:
        return exp_minx, exp_miny, exp_maxx, exp_maxy

    window = rasterio.windows.from_bounds(
        exp_minx, exp_miny, exp_maxx, exp_maxy, src.transform
    )

    return window

def divide(shape, crs="EPSG:3857", overlap = config.RESOLUTION * 0.25):
    shape = shape.to_crs(crs)
    unified_shape = shape.unary_union
    minx, miny, maxx, maxy = shape.total_bounds
    
    passo = config.RESOLUTION - overlap

    if passo < 0:
        raise ValueError("O overlap não pode ser maior ou igual ao tamanho do quadrado!")

    x_coords = np.arange(minx, maxx, passo)
    y_coords = np.arange(miny, maxy, passo)

    grids_validos = []

    for x in x_coords:
        for y in y_coords:
            quadrado = box(x, y, x + config.RESOLUTION, y + config.RESOLUTION)

            if unified_shape.intersects(quadrado):
                grids_validos.append(quadrado)

    return grids_validos