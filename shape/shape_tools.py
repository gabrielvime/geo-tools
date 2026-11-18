import geopandas as gpd
import os
import rasterio
from rasterio.features import rasterize
import numpy as np
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.ops import unary_union
from rasterio.windows import Window

import config
from type_aliases import ImageHW, ImageCHW, ImageHWC, ImageT

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

        square = max(width, height) * config.EXPAND_FACTOR if max_size else size * pixel_size * config.EXPAND_FACTOR

        exp_minx = cx - (square / 2)
        exp_maxx = cx + (square / 2)
        exp_miny = cy - (square / 2)
        exp_maxy = cy + (square / 2)

    else:

        expand = max(width, height) - min(height, width) if max_size else 0 
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

def divide(
    image : ImageT | None, 
    shape_file : gpd.GeoDataFrame,
    transform : rasterio.Affine,
    crs : str | rasterio.crs.CRS="EPSG:3857", 
    tile_size : int = config.RESOLUTION,
    overlap : int = int(config.RESOLUTION * 0.05), 
    spatial_resolution : float = 1
) -> list[ImageT] | list[gpd.GeoDataFrame]:
    """Divides an image or a GeoDataFrame into spatial tiles intersecting a target area.

    Operates in two distinct modes:
    1. Image mode: Slices a 3D NumPy array (CHW or HWC) into sub-tiles intersecting
       the shapefile geometries, preserving the original channel layout.
    2. Vector mode: Generates a grid of GeoDataFrame bounding boxes intersecting the
       shapefile geometries when no image is provided.

    Args:
        image: Input 3D array formatted as CHW or HWC, or None for vector mode.
        shape_file: GeoDataFrame defining the target region of interest.
        transform: Affine transformation matrix mapping pixel coordinates to CRS.
        crs: Coordinate Reference System for spatial alignment. Defaults to "EPSG:3857".
        tile_size: Width and height of the output tiles in pixels. Defaults to config.RESOLUTION.
        overlap: Overlap between adjacent tiles in pixels. Defaults to 5% of config.RESOLUTION.
        spatial_resolution: Ground sampling distance (meters per pixel). Defaults to 1.0.

    Returns:
        A list of sliced NumPy arrays matching the input layout (`ImageT`) if `image`
        is provided, or a list of single-row GeoDataFrames if `image` is None.

    Raises:
        ValueError: If `overlap` >= `tile_size`, or if `image` is not a 3D array.
    """
    if overlap >= tile_size:
        raise ValueError("Overlap must be strictly less than tile_size.")

    if shape_file.crs != crs:
        shape_file = shape_file.to_crs(crs)

    target_geometry = unary_union(shape_file.geometry)

    # Mode 1: Image Slicing
    if image is not None:
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image (CHW or HWC), got shape {image.shape}.")

        # Infer layout: assume channels-first if axis 0 is strictly smaller than spatial axes
        is_chw = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
        work_image = image if is_chw else np.transpose(image, (2, 0, 1))

        _, height, width = work_image.shape
        step = tile_size - overlap
        image_tiles: list[ImageT] = []

        for row in range(0, height - tile_size + 1, step):
            for col in range(0, width - tile_size + 1, step):
                window = Window(col_off=col, row_off=row, width=tile_size, height=tile_size)
                bounds = rasterio.windows.bounds(window, transform)
                tile_bbox = box(*bounds)

                if target_geometry.intersects(tile_bbox):
                    print(f'+1 intersect')
                    tile_chw = work_image[:, row : row + tile_size, col : col + tile_size]
                    out_tile = tile_chw if is_chw else np.transpose(tile_chw, (1, 2, 0))
                    image_tiles.append(out_tile)  # type: ignore[arg-type]

        return image_tiles

    # Mode 2: Vector Grid Generation
    minx, miny, maxx, maxy = shape_file.total_bounds
    spatial_step = (tile_size - overlap) * spatial_resolution
    tile_ground_size = tile_size * spatial_resolution

    x_coords = np.arange(minx, maxx, spatial_step)
    y_coords = np.arange(miny, maxy, spatial_step)

    vector_tiles: list[gpd.GeoDataFrame] = []

    for x in x_coords:
        for y in y_coords:
            tile_bbox = box(x, y, x + tile_ground_size, y + tile_ground_size)

            if target_geometry.intersects(tile_bbox):
                tile_gdf = gpd.GeoDataFrame({"geometry": [tile_bbox]}, crs=crs)
                vector_tiles.append(tile_gdf)

    return vector_tiles